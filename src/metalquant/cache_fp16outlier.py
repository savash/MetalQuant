"""Fp16-outlier + TurboQuant-regular KV cache backend for MetalQuant.

Key insight for Qwen2.5-7B and similar models with large KV activations
=======================================================================
For models where K/V vectors have large L2 norms (e.g. ~274 for Qwen2.5-7B-4bit),
standard TurboQuant fails because MSE = ||x||² × ε_unit — the reconstruction error
scales as the square of the L2 norm.

However, these large norms are concentrated in a small number of "outlier channels"
(identified by calibration).  The regular channels have small norms (≈9-16).

Strategy:
  1. Store outlier channels (typically 32) at full fp16 precision — zero error.
  2. Apply TurboQuant to regular channels (typically 96) — small norms, low MSE.

Expected MSE comparison (Qwen2.5-7B layer 0, head_dim=128):
  - Regular channel L2 norms ≈ 15.7
  - TQ4 on regular: MSE ≈ 15.7² × ε_unit_4bit ≈ 246 × 0.000079 ≈ 0.019
  - INT8 on full vector: MSE ≈ 0.082
  - Quality gain: ~4× better than INT8

Memory (per token-head vector, unpacked):
  - Fp16 outliers (32 ch): 64 bytes
  - TQ4 regular (96 ch): 96 + 4 = 100 bytes
  - Total: 164 bytes vs 256 bytes fp16 baseline (1.56× reduction)
  - vs INT8: 132 bytes per vector (1.94× reduction)

With bit-packing (future work):
  - TQ2 regular packed (96 ch @ 2-bit): 24 + 4 = 28 bytes
  - Total: 64 + 28 = 92 bytes (2.78× reduction) at INT8-comparable quality
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx
from mlx_lm.models.cache import KVCache

from metalquant.cache_turboquant import _CENTROIDS_N01, _make_rotation


class TurboQuantFp16OutlierCache(KVCache):
    """Store outlier channels as fp16, TurboQuant-compress regular channels.

    This backend is optimised for models with large KV activation norms
    concentrated in a few channels (e.g. Qwen2.5 layers 0 and 27).

    Args:
        outlier_channels: Channel indices (into head_dim) with high variance.
            Produced by metalquant.calibrate.calibrate_outlier_channels.
        bits: Bits for TurboQuant on regular channels.  Default: 4.
        layer_idx: Used to seed the rotation matrix.
    """

    def __init__(
        self,
        outlier_channels: list[int],
        bits: int = 4,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        if bits not in _CENTROIDS_N01:
            raise ValueError(f"bits must be one of {list(_CENTROIDS_N01.keys())}, got {bits}")
        self._outlier_ch_np = np.array(sorted(outlier_channels), dtype=np.int32)
        self._tq_bits = bits
        self._layer_idx = layer_idx

        # Lazy init (needs head_dim).
        self._regular_ch_np: np.ndarray | None = None
        self._head_dim: int | None = None
        self._codebook: mx.array | None = None    # (2^bits,)
        self._rotation: mx.array | None = None    # (n_reg, n_reg)

        # Accumulated storage.
        # Fp16 outlier channels: (B, H, S, n_out)
        self._k_out: mx.array | None = None
        self._v_out: mx.array | None = None
        # TQ regular channels: uint8 indices + float32 scale
        self._k_reg_idx: mx.array | None = None   # (B, H, S, n_reg)
        self._k_reg_sc:  mx.array | None = None   # (B, H, S, 1)
        self._v_reg_idx: mx.array | None = None
        self._v_reg_sc:  mx.array | None = None

    # ------------------------------------------------------------------
    def _init_if_needed(self, head_dim: int) -> None:
        if self._head_dim is not None:
            return
        self._head_dim = head_dim
        all_ch = np.arange(head_dim, dtype=np.int32)
        out_set = set(self._outlier_ch_np.tolist())
        self._regular_ch_np = np.array(
            [c for c in all_ch if c not in out_set], dtype=np.int32
        )
        n_out = len(self._outlier_ch_np)
        n_reg = len(self._regular_ch_np)
        sigma = 1.0 / np.sqrt(n_reg)
        raw = np.array(_CENTROIDS_N01[self._tq_bits], dtype=np.float32) * sigma
        self._codebook = mx.array(raw)
        Q = _make_rotation(n_reg, seed=42 + self._layer_idx)
        self._rotation = mx.array(Q)

        # Scatter matrices for GPU-friendly merge (no numpy round-trips).
        # scatter_out[i, j] = 1.0 iff outlier_ch[i] == j  →  shape (n_out, D)
        # scatter_reg[i, j] = 1.0 iff regular_ch[i] == j  →  shape (n_reg, D)
        # merge = x_out @ scatter_out + x_reg @ scatter_reg
        scatter_out_np = np.zeros((n_out, head_dim), dtype=np.float16)
        scatter_reg_np = np.zeros((n_reg, head_dim), dtype=np.float16)
        for i, ch in enumerate(self._outlier_ch_np):
            scatter_out_np[i, ch] = 1.0
        for i, ch in enumerate(self._regular_ch_np):
            scatter_reg_np[i, ch] = 1.0
        self._scatter_out = mx.array(scatter_out_np)  # (n_out, D)
        self._scatter_reg = mx.array(scatter_reg_np)  # (n_reg, D)

    # ------------------------------------------------------------------
    def _tq_quantize(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """TurboQuant_mse on (B, H, S, n_reg) tensor."""
        B, H, S, D = x.shape
        x_f32 = x.astype(mx.float32)
        norms = mx.linalg.norm(x_f32, axis=-1, keepdims=True)
        safe = mx.maximum(norms, mx.array(1e-8, dtype=mx.float32))
        x_unit = x_f32 / safe
        y = x_unit.reshape(-1, D) @ self._rotation.T
        y = y.reshape(B, H, S, D)
        distances = mx.abs(y[..., None] - self._codebook)
        indices = mx.argmin(distances, axis=-1).astype(mx.uint8)
        return indices, norms.astype(mx.float32)

    def _tq_dequantize(self, indices: mx.array, scales: mx.array) -> mx.array:
        """Reconstruct float16 from TQ indices + scales."""
        B, H, S, D = indices.shape
        idx_flat = indices.reshape(-1).astype(mx.int32)
        y_flat = mx.take(self._codebook, idx_flat).reshape(-1, D)
        x_hat = (y_flat @ self._rotation).reshape(B, H, S, D)
        return (x_hat * scales).astype(mx.float16)

    # ------------------------------------------------------------------
    @staticmethod
    def _append(existing: mx.array | None, new: mx.array) -> mx.array:
        if existing is None:
            return new
        return mx.concatenate([existing, new], axis=2)

    # ------------------------------------------------------------------
    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        head_dim = keys.shape[-1]
        self._init_if_needed(head_dim)

        out_idx = mx.array(self._outlier_ch_np)
        reg_idx = mx.array(self._regular_ch_np)

        # Split into outlier (fp16) and regular (TQ).
        k_out_new = keys[..., out_idx].astype(mx.float16)
        v_out_new = values[..., out_idx].astype(mx.float16)
        k_reg_new = keys[..., reg_idx]
        v_reg_new = values[..., reg_idx]

        # Quantize regular channels with TurboQuant.
        k_reg_idx, k_reg_sc = self._tq_quantize(k_reg_new)
        v_reg_idx, v_reg_sc = self._tq_quantize(v_reg_new)

        # Accumulate.
        self._k_out     = self._append(self._k_out,     k_out_new)
        self._v_out     = self._append(self._v_out,     v_out_new)
        self._k_reg_idx = self._append(self._k_reg_idx, k_reg_idx)
        self._k_reg_sc  = self._append(self._k_reg_sc,  k_reg_sc)
        self._v_reg_idx = self._append(self._v_reg_idx, v_reg_idx)
        self._v_reg_sc  = self._append(self._v_reg_sc,  v_reg_sc)

        self.offset += keys.shape[2]

        # Dequantize regular channels.
        k_reg_hat = self._tq_dequantize(self._k_reg_idx, self._k_reg_sc)
        v_reg_hat = self._tq_dequantize(self._v_reg_idx, self._v_reg_sc)

        # Merge via GPU matmul: avoids numpy CPU round-trips entirely.
        # x_out  @ scatter_out  →  (B,H,S,n_out) @ (n_out,D)  =  (B,H,S,D)  (outlier channels)
        # x_reg  @ scatter_reg  →  (B,H,S,n_reg) @ (n_reg,D)  =  (B,H,S,D)  (regular channels)
        # Sum: each output channel gets exactly one non-zero contribution.
        full_k = (self._k_out.astype(mx.float32) @ self._scatter_out.astype(mx.float32) +
                  k_reg_hat.astype(mx.float32) @ self._scatter_reg.astype(mx.float32)).astype(mx.float16)
        full_v = (self._v_out.astype(mx.float32) @ self._scatter_out.astype(mx.float32) +
                  v_reg_hat.astype(mx.float32) @ self._scatter_reg.astype(mx.float32)).astype(mx.float16)
        return full_k, full_v

    @property
    def nbytes(self) -> int:
        total = 0
        for arr in (self._k_out, self._v_out):
            if arr is not None:
                total += arr.nbytes
        for arr in (self._k_reg_idx, self._v_reg_idx, self._k_reg_sc, self._v_reg_sc):
            if arr is not None:
                total += arr.nbytes
        return total
