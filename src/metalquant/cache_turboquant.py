"""TurboQuant_mse KV cache backend for MetalQuant.

Implements the MSE-optimized TurboQuant algorithm from:
  "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
  Zandieh et al., arXiv 2504.19874, April 2025

Algorithm (Q_mse)
-----------------
For each K or V vector x ∈ R^d (per token, per head):

  1. Normalize:    scale = ||x||₂,  x_norm = x / scale
  2. Rotate:       y = Π · x_norm    (Π is a random orthogonal matrix)
     Each coordinate of y ≈ N(0, 1/d) and near-independent in high-d.
  3. Quantize:     idx = argmin_k |y_j - c_k|  per coordinate j
     c_k are Max-Lloyd centroids for N(0, 1/d), precomputed at init time.
  4. Store:        uint8 indices (D bytes/vector) + float32 scale (4 bytes/vector)

Dequantize:
  1. Reconstruct:  ỹ[j] = c_{idx[j]}
  2. Unrotate:     x̃ = Π⊤ · ỹ
  3. Rescale:      x̂ = x̃ * scale

Memory comparison (D=128, float16 baseline):
  - float16 baseline:          256 bytes/vector
  - INT8CacheBackend:          132 bytes/vector (~1.9x reduction)
  - TurboQuantMSECache (b≤8): 132 bytes/vector (~1.9x reduction, better quality)
  - Future: pack indices to b bits → D*b/8 + 4 bytes/vector
    b=2: 36 bytes/vector (~7.1x reduction)
    b=3: 52 bytes/vector (~4.9x reduction)
    b=4: 68 bytes/vector (~3.8x reduction)

Quality advantage over INT8
----------------------------
Same storage as INT8CacheBackend, but:
  - Random rotation makes coordinates near-independent → scalar quantizers are optimal
  - Max-Lloyd codebook minimises MSE for the actual coordinate distribution
  - INT8 uses uniform quantization with arbitrary per-vector abs-max scaling
  - TurboQuant is within ~2.7x of the Shannon distortion-rate lower bound
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import KVCache


# ---------------------------------------------------------------------------
# Precomputed Max-Lloyd centroids for N(0, 1)
# Source: standard ECSQ optimal scalar quantizer tables, widely reproduced.
# These are centroids, not boundaries.  Values are symmetric around 0.
# Scaled to N(0, 1); at runtime we further scale by 1/sqrt(head_dim).
# ---------------------------------------------------------------------------
_CENTROIDS_N01: dict[int, list[float]] = {
    1: [-0.7979, 0.7979],
    2: [-1.5104, -0.4528, 0.4528, 1.5104],
    3: [-2.1518, -1.3439, -0.7560, -0.2451, 0.2451, 0.7560, 1.3439, 2.1518],
    4: [
        -2.7326, -2.0694, -1.6180, -1.2562, -0.9423, -0.6568, -0.3880, -0.1268,
         0.1268,  0.3880,  0.6568,  0.9423,  1.2562,  1.6180,  2.0694,  2.7326,
    ],
}


def _make_rotation(d: int, seed: int = 42) -> np.ndarray:
    """Random orthogonal rotation matrix via QR decomposition of N(0,1) matrix.

    Deterministic for reproducibility (seed is fixed per cache instance via
    the caller; each layer should use a different seed to get independent
    rotations).
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d)).astype(np.float32)
    Q, _ = np.linalg.qr(G)
    return Q  # shape (d, d), orthogonal


class TurboQuantMSECache(KVCache):
    """MSE-optimized TurboQuant KV cache.

    Args:
        bits: Bits per coordinate.  Must be in {1, 2, 3, 4}.  Default: 2.
        layer_idx: Used to seed the rotation matrix so each layer gets an
            independent random rotation.  Default: 0.
    """

    def __init__(self, bits: int = 2, layer_idx: int = 0) -> None:
        super().__init__()
        if bits not in _CENTROIDS_N01:
            raise ValueError(f"bits must be one of {list(_CENTROIDS_N01.keys())}, got {bits}")
        self.tq_bits = bits
        self._layer_idx = layer_idx

        # Initialized lazily on first update_and_fetch call (need head_dim).
        self._rotation: mx.array | None = None       # (D, D) float32
        self._codebook: mx.array | None = None       # (2^bits,) float32
        self._head_dim: int | None = None

        # Accumulated quantized cache storage.
        # Shape: (B, H, S, D)  dtype: uint8
        self._k_indices: mx.array | None = None
        self._v_indices: mx.array | None = None
        # Shape: (B, H, S, 1)  dtype: float32
        self._k_scales: mx.array | None = None
        self._v_scales: mx.array | None = None

    # ------------------------------------------------------------------
    # Lazy initialisation
    # ------------------------------------------------------------------

    def _init_if_needed(self, head_dim: int) -> None:
        if self._head_dim is not None:
            return
        self._head_dim = head_dim
        sigma = 1.0 / np.sqrt(head_dim)
        raw = np.array(_CENTROIDS_N01[self.tq_bits], dtype=np.float32) * sigma
        self._codebook = mx.array(raw)  # (2^bits,)
        Q = _make_rotation(head_dim, seed=42 + self._layer_idx)
        self._rotation = mx.array(Q)   # (D, D)

    # ------------------------------------------------------------------
    # Quantization helpers
    # ------------------------------------------------------------------

    def _quantize(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Quantize a float16 K or V tensor to TurboQuant_mse format.

        Args:
            x: shape (B, H, S, D)

        Returns:
            indices: uint8, shape (B, H, S, D)  — index into codebook
            scales:  float32, shape (B, H, S, 1) — L2 norms for rescaling
        """
        B, H, S, D = x.shape

        # 1. L2 norm for rescaling.
        norms = mx.linalg.norm(x.astype(mx.float32), axis=-1, keepdims=True)  # (B, H, S, 1)
        safe_norms = mx.maximum(norms, mx.array(1e-8, dtype=mx.float32))

        # 2. Normalize to unit sphere.
        x_norm = (x.astype(mx.float32) / safe_norms)  # (B, H, S, D)

        # 3. Random rotation: reshape for batched matmul, then restore.
        # x_flat: (B*H*S, D)
        x_flat = x_norm.reshape(-1, D)
        # Π is (D, D); y_flat = x_flat @ Π.T since Π·x = x·Π⊤
        y_flat = x_flat @ self._rotation.T  # (B*H*S, D)
        y = y_flat.reshape(B, H, S, D)     # (B, H, S, D)

        # 4. Nearest codebook centroid per coordinate.
        # distances: (B, H, S, D, 2^bits)
        distances = mx.abs(y[..., None] - self._codebook)
        indices_fp = mx.argmin(distances, axis=-1)  # (B, H, S, D), dtype int32
        indices = indices_fp.astype(mx.uint8)

        return indices, norms.astype(mx.float32)

    def _dequantize(self, indices: mx.array, scales: mx.array) -> mx.array:
        """Reconstruct float16 tensor from TurboQuant_mse indices + scales.

        Args:
            indices: uint8, shape (B, H, S, D)
            scales:  float32, shape (B, H, S, 1)

        Returns:
            float16 tensor, shape (B, H, S, D)
        """
        B, H, S, D = indices.shape

        # 1. Lookup centroids.
        idx_flat = indices.reshape(-1).astype(mx.int32)  # (B*H*S*D,)
        y_flat = mx.take(self._codebook, idx_flat)       # (B*H*S*D,) float32
        y = y_flat.reshape(B, H, S, D)                   # (B, H, S, D)

        # 2. Inverse rotation: y_hat = y @ Π  (since Π⁻¹ = Π⊤, so x̃ = Π⊤·ỹ = ỹ @ Π)
        y2 = y.reshape(-1, D)
        x_hat_flat = y2 @ self._rotation        # (B*H*S, D)
        x_hat = x_hat_flat.reshape(B, H, S, D)

        # 3. Rescale and return float16.
        return (x_hat * scales).astype(mx.float16)

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Quantize incoming K/V, append, return dequantized full cache.

        Args:
            keys:   float16, shape (B, H, S_new, D)
            values: float16, shape (B, H, S_new, D)

        Returns:
            full_keys, full_values — float16, shape (B, H, S_total, D)
        """
        head_dim = keys.shape[-1]
        self._init_if_needed(head_dim)

        k_idx, k_scale = self._quantize(keys)
        v_idx, v_scale = self._quantize(values)

        if self._k_indices is None:
            self._k_indices = k_idx
            self._v_indices = v_idx
            self._k_scales = k_scale
            self._v_scales = v_scale
        else:
            self._k_indices = mx.concatenate([self._k_indices, k_idx], axis=2)
            self._v_indices = mx.concatenate([self._v_indices, v_idx], axis=2)
            self._k_scales = mx.concatenate([self._k_scales, k_scale], axis=2)
            self._v_scales = mx.concatenate([self._v_scales, v_scale], axis=2)

        # Keep offset in sync so make_mask produces correct causal masks.
        self.offset += keys.shape[2]

        full_keys = self._dequantize(self._k_indices, self._k_scales)
        full_values = self._dequantize(self._v_indices, self._v_scales)
        return full_keys, full_values

    @property
    def nbytes(self) -> int:
        """Memory used by quantized indices and scales."""
        total = 0
        for arr in (self._k_indices, self._v_indices):
            if arr is not None:
                total += arr.nbytes
        for arr in (self._k_scales, self._v_scales):
            if arr is not None:
                total += arr.nbytes
        return total
