# MetalQuant

**KV cache compression for local LLMs on Apple Silicon.**

MetalQuant implements and validates the [TurboQuant algorithm](https://arxiv.org/abs/2504.19874) on MLX, making it practical for developers running models on a 16GB Mac. We also document a critical failure mode the paper doesn't mention — and the fix for it.

The short version: this repo is about getting more useful context per GB on Apple Silicon without hand-wavy quality claims. The emphasis is practical, measurable local inference, not just reproducing a paper in isolation.

---

## What this does

When an LLM generates text, it stores a running memory of the conversation called a **KV cache**. On a 16GB Mac, this cache fills up fast — typically limiting you to a few thousand tokens of context.

MetalQuant compresses the KV cache while keeping output quality intact, letting you run longer conversations and larger contexts on the same hardware.

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

---

## Key findings

### 1. TurboQuant works on Apple Silicon

At 2-bit (TQ2), the KV cache shrinks by **7.1×** and the model still generates correct, coherent code. At 4-bit (TQ4), output is nearly identical to the uncompressed baseline.

### 2. It silently fails on 4-bit weight-quantized models

This is something the paper doesn't cover. When you run TurboQuant on a 4-bit weight-quantized model (e.g. `Qwen2.5-7B-Instruct-4bit`), it produces garbage output with no warning.

**Root cause**: 4-bit weight quantization inflates KV vector norms from ~18 to ~274. TurboQuant's reconstruction error scales as `norm²`, making it 72× worse than INT8 at that point.

**Diagnostic**: before using any TurboQuant backend, check your model's KV norms:

```python
# norms should be < 50 for TurboQuant to work correctly
# norms > 50 indicate 4-bit weight quantization artifacts
```

See `docs/DEVLOG.md` for the full diagnostic script.

### 3. Fix for 4-bit models: fp16-outlier + TQ

For models with inflated KV norms, we developed `TurboQuantFp16OutlierCache`:
- Run a calibration pass to identify the 32 highest-variance channels
- Store those 32 channels at full fp16 precision (they hold all the large-norm energy)
- Apply TurboQuant to the remaining 96 channels (norms ~15, compression works)

Result on `Qwen2.5-7B-4bit`: **4.6× better reconstruction accuracy than INT8**, generation quality matches the uncompressed baseline.

### 4. The goal is practical local inference, not just paper replication

This project is trying to answer a concrete question:

> On a memory-limited Apple Silicon machine, how much more useful context can we get without breaking generation quality?

That means the important outputs here are:
- reproducible benchmarks
- model-specific guidance
- clear failure modes
- implementation details that can be reused in real MLX workflows

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

### Run the benchmark

```bash
# Baseline — no compression
python benchmarks/run_experiment.py \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-8bit \
  --cache-backend baseline \
  --out results/baseline.json

# INT8 compression (~2× smaller KV cache)
python benchmarks/run_experiment.py \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-8bit \
  --cache-backend int8 \
  --out results/int8.json

# TQ4 compression (~4× smaller KV cache)
python benchmarks/run_experiment.py \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-8bit \
  --cache-backend tq4 \
  --out results/tq4.json

# TQ2 compression (~7× smaller KV cache)
python benchmarks/run_experiment.py \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-8bit \
  --cache-backend tq2 \
  --out results/tq2.json

# Compare any two results
python benchmarks/compare_results.py results/baseline.json results/tq2.json
```

### Run the lightweight test suite

```bash
python -m pytest
```

### For 4-bit models (Qwen, Mistral-4bit, etc.)

```bash
# Step 1: calibrate to identify outlier channels
python benchmarks/run_calibrate.py \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --out results/calibration.json

# Step 2: run with the fp16-outlier backend
python benchmarks/run_experiment.py \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --cache-backend fp16-outlier \
  --calibration results/calibration.json \
  --out results/fp16-outlier.json
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

## Repository layout

```
MetalQuant/
├── benchmarks/
│   ├── run_experiment.py     # benchmark runner (all backends)
│   ├── run_calibrate.py      # outlier channel calibration
│   ├── compare_results.py    # diff two result JSONs
│   └── prompts.py
├── docs/
│   ├── DEVLOG.md             # full research journal — what worked, what didn't, why
│   ├── ARCHITECTURE.md
│   └── ROADMAP.md
├── results/                  # benchmark outputs (gitignored)
├── scripts/
│   ├── activate.sh          # local-only env/cache activation
│   └── bootstrap.sh         # create local Python env and install package
├── tests/                    # lightweight unit tests for non-MLX logic
└── src/metalquant/
    ├── cache.py              # backend factory: make_cache(model, backend="tq2")
    ├── cache_quantized.py    # INT8 backend
    ├── cache_turboquant.py   # TurboQuant core (Q_mse algorithm)
    ├── cache_turboquant_v2.py # outlier-aware TurboQuant
    ├── cache_fp16outlier.py  # fp16 outlier + TQ regular (fix for 4-bit models)
    └── calibrate.py          # per-channel variance calibration
```

---

## How to use the cache backends in your own code

```python
import mlx.core as mx
from mlx_lm import load
from metalquant.cache import make_cache

model, tokenizer = load("mlx-community/Meta-Llama-3.1-8B-Instruct-8bit")

# Pick a backend: "baseline", "int8", "tq2", "tq4", "fp16-outlier"
cache = make_cache(model, backend="tq2")

# Use exactly like a normal mlx-lm cache
prompt = "Explain KV cache compression in one paragraph."
input_ids = mx.array(tokenizer.encode(prompt))[None]
logits = model(input_ids, cache=cache)
```

---

## What's next

- **User-facing CLI** — ship `metalquant diagnose`, `metalquant calibrate`, and `metalquant benchmark` so the repo is usable without reading the source first
- **Auto backend selection** — recommend or choose `int8`, `tq4`, `tq2`, or `fp16-outlier` from model metadata, KV norms, and cached calibration data
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
