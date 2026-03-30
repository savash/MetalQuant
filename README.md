# MetalQuant

**KV cache compression for local LLMs on Apple Silicon.**

MetalQuant implements and validates the [TurboQuant algorithm](https://arxiv.org/abs/2504.19874) on MLX, making it practical for developers running models on a 16GB Mac. We also document a critical failure mode the paper doesn't mention — and the fix for it.

The short version: this repo is about getting more useful context per GB on Apple Silicon without hand-wavy quality claims. The emphasis is practical local inference, not just reproducing a paper in isolation.

## Current status

- TurboQuant works well on healthy 8-bit MLX models such as `Meta-Llama-3.1-8B-Instruct-8bit`
- Standard TurboQuant fails on many 4-bit weight-quantized models because KV norms are too large
- `fp16-outlier` is the current practical fix for those 4-bit models
- Bit packing is not implemented yet, so the headline compression ratios are algorithmic targets already validated mathematically, not the exact in-memory storage used by the current code

**Measured results on M4 Mac Mini 16GB (`Meta-Llama-3.1-8B-Instruct-8bit`):**

| Backend | KV memory/token | Compression | Decode speed | Output quality |
|---|---|---|---|---|
| Baseline (fp16) | 256 bytes | 1.0× | 61.8 tok/s | reference |
| INT8 | 132 bytes | **1.9×** | 57.6 tok/s | ✅ matches baseline |
| TQ4 (4-bit TurboQuant) | 68 bytes | **3.8×** | 52.5 tok/s | ✅ matches baseline |
| TQ2 (2-bit TurboQuant) | 36 bytes | **7.1×** | 52.9 tok/s | ✅ correct output |

> **Note**: compression numbers assume bit-packed index storage. The algorithm is fully implemented and validated; bit packing is the remaining engineering step to realise the full numbers in practice.

## What matters

- Healthy 8-bit models can use standard TurboQuant well.
- Many 4-bit models need `fp16-outlier` because KV norms are too large.
- The CLI is now the main entry point for diagnose, calibrate, benchmark, and generate.
- Bit packing is still the main missing engineering step before the headline compression ratios are fully realized in memory.

---

## Quick start

### Requirements

- Apple Silicon Mac (M1 or later)
- macOS 13+
- `uv` installed ([astral.sh/uv](https://astral.sh/uv))
- Python 3.11+ (installed automatically by `./scripts/bootstrap.sh`)
- ~10GB free disk space for model weights

### Install

```bash
git clone https://github.com/savash/MetalQuant
cd MetalQuant
./scripts/bootstrap.sh
source scripts/activate.sh
```

`bootstrap.sh` creates a local `.venv`, installs Python 3.11, and installs the package in editable mode.
`scripts/activate.sh` activates the venv and points `uv` and Hugging Face caches into the project-local `.cache/` directory.

### Choose a backend

| Backend | When to use it | Tradeoff |
|---|---|---|
| `baseline` | Reference run, debugging, quality comparisons | No compression |
| `int8` | Safe default for almost any model | Best stability, smallest quality risk, lower compression |
| `tq4` | Healthy 8-bit models where you want a strong speed/quality/compression balance | Needs healthy KV norms |
| `tq2` | Healthy 8-bit models where maximum compression matters most | More aggressive compression, still model-dependent |
| `fp16-outlier` | 4-bit models with inflated KV norms | Better quality than INT8 on problematic models, but less compression than ideal TQ |

### CLI commands

```bash
# Recommend a backend from model metadata or measured norms
metalquant diagnose --model mlx-community/Meta-Llama-3.1-8B-Instruct-8bit

# Run calibration for 4-bit models that need the outlier-aware path
metalquant calibrate \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --out results/calibration-qwen2-5-7b-instruct-4bit.json

# Run a benchmark through the CLI with an explicit backend
metalquant benchmark \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-8bit \
  --cache-backend tq4 \
  --out results/tq4.json

# Or let the CLI choose the backend from model metadata and KV norms
metalquant benchmark \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-8bit \
  --cache-backend auto \
  --kv-norm 18

# If --out is omitted, MetalQuant writes model-specific result files under results/

# Generate with the automatic backend recommendation path
metalquant generate \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-8bit \
  --prompt "Explain KV cache compression in simple terms." \
  --backend auto

# The package also supports module execution
python -m metalquant diagnose --model mlx-community/Meta-Llama-3.1-8B-Instruct-8bit
```

### Raw benchmark scripts

```bash
# Baseline
python benchmarks/run_experiment.py \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-8bit \
  --cache-backend baseline \
  --out results/baseline.json

# TQ4
python benchmarks/run_experiment.py \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-8bit \
  --cache-backend tq4 \
  --out results/tq4.json

# Compare two runs
python benchmarks/compare_results.py results/baseline.json results/tq4.json
```

### Run the lightweight test suite

```bash
python -m pytest
```

### For 4-bit models

```bash
# Preferred CLI flow
metalquant calibrate \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --out results/calibration-qwen2-5-7b-instruct-4bit.json

metalquant benchmark \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --cache-backend fp16-outlier \
  --calibration results/calibration-qwen2-5-7b-instruct-4bit.json
```

---

## Recommended model for 16GB Mac

```
mlx-community/Meta-Llama-3.1-8B-Instruct-8bit
```

- Weights: ~8GB — fits comfortably with 6GB left for KV cache
- KV norms are healthy (~18) — TurboQuant works correctly
- Same model family the paper benchmarked against
- Strong coding and instruction-following quality

---

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — high-level system layout
- [docs/ROADMAP.md](docs/ROADMAP.md) — current phases and next steps
- [docs/DEVLOG.md](docs/DEVLOG.md) — detailed research journal

---

## What's next

- **Calibration reuse** — cache diagnosis and calibration artifacts per model so users do not repeat setup work
- **Bit packing** — pack 2-bit indices into bytes to realise the full 7.1× memory reduction in practice (currently algorithm is correct, storage is not yet packed)
- **Perplexity benchmarks** — WikiText-2 and HumanEval scores for rigorous quality comparison
- **Larger models** — test on Qwen2.5-Coder-14B and other coding-focused models
- **More architectures** — validate KV norm behaviour across Mistral, Gemma, Phi

## What success looks like

MetalQuant is successful if it becomes a reliable answer to:

- which MLX models are safe for TurboQuant as-is
- which models need the outlier-aware path
- what memory reduction is actually achievable in practice
- how to reproduce those results on a local Apple Silicon machine without hidden setup steps

---

## Research journal

The full story of what we tried, what broke, and what we learned is in [`docs/DEVLOG.md`](docs/DEVLOG.md). It covers every bug hit, every failed approach, the math behind the failure mode, and the reasoning behind each fix. Written to be useful to anyone who wants to reproduce or extend this work.

---

## Background

This project implements and extends:

> **TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate**
> Zandieh et al., arXiv 2504.19874, April 2025

The algorithm uses random orthogonal rotation + Max-Lloyd scalar quantization to compress KV cache vectors near the theoretical optimum for their bit rate.

## License

MIT
