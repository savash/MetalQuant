"""Calibration for TurboQuant outlier channel identification.

Runs a forward pass on calibration prompts and measures per-channel
variance of K/V vectors at each transformer layer (in the original
pre-rotation space).  The top-N highest-variance channels are the
"outlier channels" that receive more quantization bits.

Also computes per-channel standard deviations (k_ch_stds, v_ch_stds) which
are used by TurboQuant backends for per-channel normalization.  This reduces
the effective L2 norm from ~274 (Qwen2.5-7B) to ~sqrt(head_dim) ≈ 11.3,
cutting final MSE by ~600×.

Usage:
    from metalquant.calibrate import calibrate_outlier_channels
    calib = calibrate_outlier_channels(model, tokenizer, prompts, n_outlier=32)
    # calib is a list of dicts, one per layer:
    # {"outlier_channels": [int, ...], "head_dim": int,
    #  "k_ch_stds": [float, ...], "v_ch_stds": [float, ...]}
    # Save / load via save_calibration / load_calibration.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache


def calibrate_outlier_channels(
    model,
    tokenizer,
    prompts: list[str],
    n_outlier: int = 32,
) -> list[dict]:
    """Identify high-variance KV channels and compute per-channel stds.

    Runs each prompt through the model with a standard KVCache, collects
    the stored K and V vectors per layer, computes per-channel E[x²] across
    all (batch, head, token) positions, and returns:
      - top-n_outlier K-channel indices for outlier-aware quantization
      - per-channel stds for K and V for per-channel normalization

    Per-channel normalization (dividing each channel by its std before L2
    normalization) reduces the effective L2 norm from ~274 to ~sqrt(head_dim),
    which reduces TurboQuant MSE by ~600×.

    Args:
        model:      Loaded mlx-lm model.
        tokenizer:  Matching tokenizer.
        prompts:    List of calibration prompts (3–10 is sufficient).
        n_outlier:  Number of outlier channels to flag per layer (default 32).

    Returns:
        List of dicts, one per layer:
        {
            "outlier_channels": [int, ...],  # top-n_outlier K channels by variance
            "head_dim": int,
            "k_ch_stds": [float, ...],       # per-channel sqrt(E[k²]) for K
            "v_ch_stds": [float, ...],       # per-channel sqrt(E[v²]) for V
        }
    """
    n_layers = len(model.layers)
    # Accumulate sum-of-squares per (layer, channel) for K and V.
    k_sum_sq: list[np.ndarray | None] = [None] * n_layers
    v_sum_sq: list[np.ndarray | None] = [None] * n_layers
    counts: list[int] = [0] * n_layers
    head_dims: list[int | None] = [None] * n_layers

    for prompt in prompts:
        cache = [KVCache() for _ in range(n_layers)]
        input_ids = mx.array(tokenizer.encode(prompt))[None]
        logits = model(input_ids, cache=cache)
        mx.eval(logits)

        for layer_idx, c in enumerate(cache):
            if c.keys is None:
                continue
            # K/V shape: (B, n_kv_heads, S_padded, head_dim)
            # Use offset to trim to actual sequence length.
            k = c.keys[..., : c.offset, :]    # (B, H, S, D)
            mx.eval(k)
            k_np = np.array(k.astype(mx.float32))
            D = k_np.shape[-1]
            head_dims[layer_idx] = D
            k_flat = k_np.reshape(-1, D)       # (N, D)
            layer_count = k_flat.shape[0]
            layer_k_sq = np.sum(k_flat ** 2, axis=0)  # (D,)

            # V vectors.
            if c.values is not None:
                v = c.values[..., : c.offset, :]
                mx.eval(v)
                v_np = np.array(v.astype(mx.float32))
                v_flat = v_np.reshape(-1, D)
                layer_v_sq = np.sum(v_flat ** 2, axis=0)
            else:
                layer_v_sq = layer_k_sq  # fallback: use K stats

            if k_sum_sq[layer_idx] is None:
                k_sum_sq[layer_idx] = layer_k_sq
                v_sum_sq[layer_idx] = layer_v_sq
                counts[layer_idx] = layer_count
            else:
                k_sum_sq[layer_idx] += layer_k_sq
                v_sum_sq[layer_idx] += layer_v_sq
                counts[layer_idx] += layer_count

    result = []
    for layer_idx in range(n_layers):
        if k_sum_sq[layer_idx] is None:
            D = 128
            result.append({
                "outlier_channels": list(range(n_outlier)),
                "head_dim": D,
                "k_ch_stds": [1.0] * D,
                "v_ch_stds": [1.0] * D,
            })
            continue

        k_var = k_sum_sq[layer_idx] / counts[layer_idx]  # (D,)
        v_var = v_sum_sq[layer_idx] / counts[layer_idx]

        # Top-n_outlier K-channels by variance — sorted for determinism.
        outlier_idx = np.argsort(k_var)[-n_outlier:]
        outlier_idx = sorted(outlier_idx.tolist())

        # Per-channel stds: sqrt(E[x²]).  Clamp to avoid division-by-zero.
        k_stds = np.sqrt(np.maximum(k_var, 1e-8)).tolist()
        v_stds = np.sqrt(np.maximum(v_var, 1e-8)).tolist()

        result.append({
            "outlier_channels": outlier_idx,
            "head_dim": int(head_dims[layer_idx]),
            "k_ch_stds": k_stds,
            "v_ch_stds": v_stds,
        })

    return result


def save_calibration(calib: list[dict], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(calib, indent=2) + "\n")


def load_calibration(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text())
