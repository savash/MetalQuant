# MetalQuant Architecture

## Intent

MetalQuant is a local-first MLX toolkit for KV-cache compression on Apple Silicon.

The project has three jobs:
- provide KV-cache backends that can drop into MLX workflows
- measure whether those backends are actually worth using
- guide users toward a safe backend for a given model

---

## Core pieces

### 1. Backends

The core abstraction is `make_cache(model, backend, calibration)` in `src/metalquant/cache.py`.
It returns one KV-cache object per transformer layer.

| Backend | File | Use case |
|---|---|---|
| `baseline` | mlx-lm built-in | Reference — no compression |
| `int8` | `cache_quantized.py` | Safe default — 1.94× compression, full quality |
| `tq2` / `tq4` | `cache_turboquant.py` | Healthy models only (8-bit weights) — up to 7.1× |
| `tq-outlier` | `cache_turboquant_v2.py` | Experimental mixed-precision variant |
| `fp16-outlier` | `cache_fp16outlier.py` | 4-bit weight-quantized models — best quality fix |

All custom backends have to behave like mlx-lm `KVCache` objects, especially around
`update_and_fetch(...)` and `self.offset`.

### 2. Calibration

`src/metalquant/calibrate.py` measures per-layer channel variance and produces the
artifacts needed by the outlier-aware backends.

### 3. Benchmarking

`benchmarks/run_experiment.py` and `benchmarks/compare_results.py` are the honesty layer.
They track throughput, cache size, and outputs in JSON so backend claims stay measurable.

### 4. CLI

`src/metalquant/cli.py` is the user-facing workflow:
- `diagnose`
- `calibrate`
- `benchmark`
- `generate`

The CLI sits on top of the existing benchmark and calibration code instead of replacing it.

### 5. Local environment

`scripts/bootstrap.sh` and `scripts/activate.sh` keep the Python environment, package cache,
and model cache local to the project folder as much as possible.

---

## Workflow

1. Diagnose a model from metadata and optional KV norms.
2. Calibrate if the model needs the outlier-aware path.
3. Benchmark or generate with the selected backend.
4. Compare memory savings against quality and speed.

---

## Non-goals

- training or fine-tuning
- quantizing model weights
- non-Apple inference stacks
- pretending all model families behave the same
