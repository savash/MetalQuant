# MetalQuant Architecture

## Intent

MetalQuant compresses the KV cache of local LLMs running on Apple Silicon via MLX,
enabling longer context windows on memory-constrained hardware (e.g. 16GB M4 Mac Mini).

The project implements and validates TurboQuant (arXiv 2504.19874) and documents
a critical failure mode for 4-bit weight-quantized models along with its fix.

---

## Layers

### 1. Benchmark layer
`benchmarks/`
- Reproducible prompt suites (`prompts.py`)
- Experiment runner with pluggable backends (`run_experiment.py`)
- Calibration runner for outlier channel identification (`run_calibrate.py`)
- Result comparison utility (`compare_results.py`)
- All results written as JSON for scripted diffing

### 2. Cache backend layer
`src/metalquant/`

The central abstraction is `make_cache(model, backend, calibration)` in `cache.py`.
It returns a list of KVCache-compatible objects, one per transformer layer.

| Backend | File | Use case |
|---|---|---|
| `baseline` | mlx-lm built-in | Reference — no compression |
| `int8` | `cache_quantized.py` | Safe default — 1.94× compression, full quality |
| `tq2` / `tq4` | `cache_turboquant.py` | Healthy models only (8-bit weights) — up to 7.1× |
| `tq-outlier` | `cache_turboquant_v2.py` | Experimental mixed-precision variant |
| `fp16-outlier` | `cache_fp16outlier.py` | 4-bit weight-quantized models — best quality fix |

All backends implement the mlx-lm `KVCache` protocol:
- `update_and_fetch(keys, values)` — compress new tokens, return full decompressed cache
- `self.offset` — must be incremented by `keys.shape[2]` every call (mask correctness)
- Avoid `self.bits` as an attribute name (conflicts with mlx-lm internals)

### 3. Calibration layer
`src/metalquant/calibrate.py`

Required by `fp16-outlier` and `tq-outlier` backends. Runs a forward pass on a small
set of prompts, measures per-channel variance of K/V vectors at each layer, and returns:
- Top-N highest-variance channel indices (outlier channels)
- Per-channel standard deviations for K and V

### 4. Profiling and reporting layer
- Decode timing and throughput measurement in `run_experiment.py`
- Cache size tracked via `backend.nbytes`
- JSON outputs in `results/` (gitignored — regenerate with benchmark scripts)

---

## Key Design Decisions

**Why MLX?** It's the native Apple Silicon framework with Metal GPU support.
No other stack provides this on Mac without significant overhead.

**Why per-vector L2 normalization in TurboQuant?** The algorithm requires vectors on the
unit sphere. After rotation, each coordinate is approximately N(0, 1/D), which the
Max-Lloyd codebook is calibrated for. Large norms amplify reconstruction error as `norm²`.

**Why fp16 for outlier channels?** The 32 highest-variance channels in 4-bit models
carry essentially all the large-norm energy. Quantizing them (even at 3-bit) still fails
because the partition norm is ~273 for a vector with full norm ~274. Storing them exactly
costs 64 bytes but eliminates the norm amplification entirely for the remaining channels.

**Why scatter matmul for channel merge?**
```python
full = x_out @ scatter_out + x_reg @ scatter_reg
```
This runs entirely on the Metal GPU. The alternative (numpy indexed assignment) requires
CPU ↔ GPU data transfer on every decode step — a 3× slowdown in practice.

---

## Non-Goals

- Training or fine-tuning
- Non-Apple backends
- Quantizing model weights (only KV cache activations)
- Supporting non-MLX inference stacks
