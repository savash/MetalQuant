"""INT8 quantized KV cache backend for MetalQuant.

Strategy
--------
Per-token, per-head scalar quantization:
  - Find abs-max of each (head, token) vector.
  - Scale to [-127, 127] and round to int8.
  - Store: int8 tensor + float32 scale per token-head.
  - On fetch: dequantize back to float16.

This is the simplest credible quantized cache.  It is NOT TurboQuant.
It serves as a measurable midpoint between baseline (float16) and the
rotation-based vector quantization in TurboQuant.

Expected outcome (to be validated):
  - ~2x reduction in K/V memory footprint vs baseline float16
  - Some decode quality drift (to be measured)
  - Possible decode speed regression due to quant/dequant overhead

The quantization operates on the CPU-side numpy-free path using pure
MLX ops so it stays on the Metal compute path.
"""
from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import KVCache


class INT8CacheBackend(KVCache):
    """KVCache subclass that stores K and V tensors quantized to int8.

    Inherits KVCache's slot management.  Overrides update_and_fetch to
    intercept tensors going in and out of the cache.

    Layout in self.keys / self.values (after first update):
        Stored as int8  — shape (B, n_heads, S, head_dim)
    Scales:
        self._k_scales / self._v_scales — float32, shape (B, n_heads, S, 1)
        One scale per (batch, head, token) triplet.
    """

    def __init__(self) -> None:
        super().__init__()
        self._k_scales: mx.array | None = None
        self._v_scales: mx.array | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _quantize(x: mx.array) -> tuple[mx.array, mx.array]:
        """Quantize float16/32 tensor x to int8.

        Args:
            x: shape (B, n_heads, S, head_dim) in float16 or float32

        Returns:
            quant: int8 array, same shape
            scale: float32 array, shape (B, n_heads, S, 1)
        """
        # Compute per-token-head abs-max for scaling.
        abs_max = mx.max(mx.abs(x), axis=-1, keepdims=True)  # (..., S, 1)
        # Avoid division by zero for all-zero vectors.
        abs_max = mx.maximum(abs_max, mx.array(1e-8, dtype=mx.float32))
        scale = abs_max.astype(mx.float32) / 127.0
        # Scale and clamp.
        scaled = x / abs_max * 127.0
        quant = mx.clip(mx.round(scaled), -127, 127).astype(mx.int8)
        return quant, scale

    @staticmethod
    def _dequantize(quant: mx.array, scale: mx.array) -> mx.array:
        """Reconstruct float16 tensor from int8 + scale.

        Args:
            quant: int8 array, shape (B, n_heads, S, head_dim)
            scale: float32 array, shape (B, n_heads, S, 1)

        Returns:
            float16 array, same shape as quant
        """
        return (quant.astype(mx.float16) * scale.astype(mx.float16))

    @staticmethod
    def _cat_scales(existing: mx.array | None, new_scale: mx.array) -> mx.array:
        """Concatenate scale tensors along the sequence dimension (axis=2)."""
        if existing is None:
            return new_scale
        return mx.concatenate([existing, new_scale], axis=2)

    # ------------------------------------------------------------------
    # Protocol implementation
    # ------------------------------------------------------------------

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        """Quantize incoming K/V, append to cache, return dequantized full K/V.

        Args:
            keys:   float16 tensor, shape (B, n_heads, S_new, head_dim)
            values: float16 tensor, shape (B, n_heads, S_new, head_dim)

        Returns:
            full_keys:   float16, shape (B, n_heads, S_total, head_dim)
            full_values: float16, shape (B, n_heads, S_total, head_dim)
        """
        # Quantize new tokens.
        k_quant, k_scale = self._quantize(keys)
        v_quant, v_scale = self._quantize(values)

        # Append quantized tokens to stored cache.
        if self.keys is None:
            self.keys = k_quant
            self.values = v_quant
        else:
            self.keys = mx.concatenate([self.keys, k_quant], axis=2)
            self.values = mx.concatenate([self.values, v_quant], axis=2)

        # Accumulate scales.
        self._k_scales = self._cat_scales(self._k_scales, k_scale)
        self._v_scales = self._cat_scales(self._v_scales, v_scale)

        # Keep offset in sync so make_mask produces correct causal masks.
        self.offset += keys.shape[2]

        # Dequantize full cache before returning to attention.
        full_keys = self._dequantize(self.keys, self._k_scales)
        full_values = self._dequantize(self.values, self._v_scales)

        return full_keys, full_values

    @property
    def nbytes(self) -> int:
        """Approximate memory used by quantized K/V tensors + scales."""
        total = 0
        for arr in (self.keys, self.values):
            if arr is not None:
                total += arr.nbytes
        for arr in (self._k_scales, self._v_scales):
            if arr is not None:
                total += arr.nbytes
        return total
