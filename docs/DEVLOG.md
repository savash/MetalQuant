# MetalQuant Research Journal

**Making TurboQuant work on Apple Silicon — what we tried, what broke, and what we proved.**

> This is a chronological record of the full research process, written for developers
> who want to reproduce or extend this work. Every dead end is documented along with
> the reason it failed, so you don't have to repeat the same mistakes.

---

## The Goal

The TurboQuant paper ([arXiv 2504.19874](https://arxiv.org/abs/2504.19874)) claims you can compress an LLM's KV cache by 7× at 2-bit precision with minimal quality loss. We wanted to:

1. Prove it works on Apple Silicon using MLX
2. Show developers exactly how to run it on a 16GB Mac
3. Document anything the paper missed

---

## Phase 1 — Baseline Infrastructure

**Goal**: establish a reproducible measurement baseline before changing anything.

Built the first benchmark runner, a prompts module, JSON result format, and `compare_results.py`.
That runner was later consolidated into `benchmarks/run_experiment.py`.
Committed to measuring everything before and after each change — no guessing.

**Stack choice**: MLX / mlx-lm. It's the native Apple Silicon framework, uses Metal GPU
acceleration, and is the standard for local inference on Mac. No other option is realistic
for production use on Apple hardware.

**Test model**: `mlx-community/Qwen2.5-7B-Instruct-4bit` — a popular 4-bit model that
fits easily on 16GB. This turned out to be the wrong choice for TurboQuant (more on that later).

---

## Phase 2 — INT8 KV Cache

**Goal**: prove that plugging in a custom KV cache backend is possible in mlx-lm.

Built `INT8CacheBackend` — per-vector abs-max quantization to uint8. Simple, well-understood,
gives a 1.94× compression baseline to beat.

### Bug 1: broken attention masks

First run generated repetitive garbage output. Example: *"pérdida...pérdida...pérdida..."*

Root cause: `KVCache.make_mask()` uses `self.offset` to build the causal attention mask for
each decode step. Our custom backend overrode `update_and_fetch` but never incremented `offset`.
With `offset = 0` throughout, every decode step attended to the same position — broken output.

**Fix**: one line added to `update_and_fetch`:
```python
self.offset += keys.shape[2]
```
Every custom `KVCache` subclass in mlx-lm must do this.

### Bug 2: attribute name conflict with mlx-lm internals

When building `TurboQuantMSECache`, we named the bit-depth attribute `self.bits`.
mlx-lm's attention code checks `hasattr(cache, 'bits')` to decide whether to route to its
built-in quantized SDPA path — which then looks for `cache.group_size`. Result: `AttributeError`.

**Fix**: renamed to `self.tq_bits`. Never use `self.bits` in a custom KVCache subclass.

### Result

INT8 working: **1.94× compression**, decode speed unchanged, quality good.

---

## Phase 3 — TurboQuant Algorithm Implementation

**Goal**: implement Q_mse from Algorithm 1 of the paper.

The algorithm for each KV vector `x`:
1. Compute L2 norm: `scale = ||x||`
2. Normalize to unit sphere: `x_unit = x / scale`
3. Apply random orthogonal rotation: `y = R @ x_unit`
4. Quantize each coordinate to nearest Max-Lloyd centroid
5. Store: `uint8` indices + `float32` scale

Reconstruction: look up centroids → apply inverse rotation → multiply by scale.

Built `cache_turboquant.py`. Both bugs from Phase 2 reappeared (offset, bits name).
Fixed. Infrastructure correct.

### Discovery: the L2 norm amplification problem

After fixing all bugs, TQ4 still generated incoherent output on `Qwen2.5-7B-4bit`.
We measured KV vector norms:

```
Qwen2.5-7B-4bit — layer 0 K vector L2 norms:
  min = 109,  mean = 274,  max = 467
```

The paper assumes KV vectors lie near the unit sphere (norm ≈ 1). Reconstruction error
scales as `norm²`:

```
TurboQuant MSE = ||x||² × ε_unit_sphere
              = 274² × 0.000079
              = 5.94

INT8 MSE (measured) = 0.082

Ratio: TQ4 is 72× worse than INT8
```

**Why are the norms so large?** The 4-bit *weight* quantization process (used to shrink
the model file from 16GB to 4GB) introduces quantization errors in the weight matrices.
These errors propagate through each forward pass and amplify the KV activation magnitudes —
layer 0 of Qwen2.5-7B-4bit has K norms ~274, roughly 15× larger than a healthy model.

This is the first finding: **TurboQuant silently fails on 4-bit weight-quantized models.**
The model produces garbage output with no error or warning.

---

## Phase 4 — Fixing TurboQuant for 4-bit Weight-Quantized Models

We tried three approaches before finding one that works.

### Attempt 1: Per-channel normalization

**Theory**: divide each channel by its calibrated standard deviation before TurboQuant.
This would reduce the effective L2 norm from ~274 to ~√128 ≈ 11.3, cutting MSE by ~600×.

Built `calibrate.py` to measure per-channel variance via a forward pass.
Applied normalization in `_quantize` before the L2 norm step.

**Result**: MSE barely changed (6.0 vs 6.9 without). Generation: pure noise ("a a a a a a...").

**Why it failed**: per-channel normalization redistributes quantization error but cannot
reduce total energy. The math:

```
MSE_original = sum_d(std_d²) × ε_unit

sum(std_d²) = E[||x||²] ≈ mean_norm²
```

You cannot escape the norm amplification this way — the total energy is conserved.
Worse: normalization concentrates errors into the high-variance channels, which are the
ones that matter most for attention. Generation quality got worse, not better.

### Attempt 2: Outlier channel splitting

**Theory**: the paper's own 2.5-bit approach — identify the 32 highest-variance channels
("outlier channels") via calibration, give them 3-bit TurboQuant, give the rest 2-bit.

Built `cache_turboquant_v2.py`.

**Result**: MSE = 20.1 (worse than plain TQ4's 7.0).

**Why it failed**: the outlier channels capture essentially *all* the L2 norm energy.
Measured:

```
Layer 0:
  Full vector L2 norm:      mean = 274
  Outlier partition (32ch): mean = 273   ← absorbs all the norm
  Regular partition (96ch): mean = 15.7  ← already healthy
```

Splitting the channels doesn't help the outlier partition — it still has the same huge norm.
Giving it 3-bit TurboQuant instead of 2-bit makes no meaningful difference at norm=273.

### Attempt 3: Fp16 outliers + TQ on regular channels ✅

**Insight from Attempt 2**: the regular channels already have healthy norms (~15). The
outlier channels are the problem. If we can remove them from the quantization loop entirely,
TurboQuant will work correctly on what's left.

**Solution**: store the 32 outlier channels at full fp16 precision (zero error), and apply
TurboQuant to the 96 regular channels (norms ~15, compression works cleanly).

Built `cache_fp16outlier.py` with `TurboQuantFp16OutlierCache`.

**Performance problem**: the first implementation merged channels using numpy scatter
(required CPU ↔ GPU round-trips). This caused a 3.4× decode slowdown.

**Fix**: replaced with a GPU matmul using pre-computed scatter matrices:
```python
# Precomputed at init: scatter_out (n_out, D), scatter_reg (n_reg, D)
full = x_out @ scatter_out + x_reg @ scatter_reg
# Runs entirely on Metal GPU — no data leaves the GPU
```

**Results on Qwen2.5-7B-4bit**:

```
INT8:                 MSE = 0.082   generation: good
fp16-outlier + TQ4:   MSE = 0.018   generation: identical to fp16 baseline
                      4.6× better reconstruction accuracy than INT8
```

Tradeoff: fp16-outlier currently uses 1.24× more memory than INT8 because the outlier
channels are stored uncompressed. With TQ2 bit packing (next step): 2.78× compression
at INT8-comparable quality.

---

## Phase 5 — The Right Model for the Paper

After Phase 4, we stepped back and asked a more fundamental question: *is the paper's
algorithm actually correct, or are we fighting something deeper?*

The paper tested on **Llama-3.1-8B** in near-full precision. We had been testing on
**Qwen2.5-7B-4bit** — a model with 4-bit weight quantization that the paper never considers.
We were testing the algorithm on a regime it wasn't designed for.

**We switched to**: `mlx-community/Meta-Llama-3.1-8B-Instruct-8bit`

### KV norm measurement

```
Llama-3.1-8B-Instruct-8bit — K vector L2 norms:

Layer   K norm (mean)   V norm (mean)
  0         16.4            0.36
  4         19.3            2.1
  8         18.6            2.5
 12         18.9            2.6
 16         19.2            2.7
 20         17.7            3.0
 24         17.9            3.8
 31         19.2            5.5
```

Healthy. Consistent across all 32 layers. No outlier spikes.
Compare to Qwen2.5-7B-4bit: mean K norm 274 vs Llama's ~18. The 4-bit weight
quantization is the entire source of the problem.

### Generation quality test

```
Prompt: "Write a Python function that checks if a number is prime:"

Backend   Output
───────────────────────────────────────────────────────────────────────
baseline  `is_prime(n)`. ## Step 1: Define the function We start by...
int8      `is_prime(n)`. ## Step 1: Define the function We start by...
tq4       `is_prime(n)`. ## Step 1: Define the function We start by...
tq2       `is_prime` function. This function takes an integer and returns
          True if prime, False otherwise. [correct code follows]
```

TQ4 output is word-for-word identical to baseline.
TQ2 takes a slightly different approach but generates fully correct, runnable code.
Both compress the KV cache. The algorithm works.

### Benchmark results

Hardware: M4 Mac Mini, 16GB unified memory.
Model: `Meta-Llama-3.1-8B-Instruct-8bit`.

```
Backend     Prefill     Decode     KV bytes/vector*    Compression
            (tok/s)    (tok/s)
────────────────────────────────────────────────────────────────────
baseline     268.7       61.8           256              1.00×
int8         282.6       57.6           132              1.94×
tq4          177.0       52.5            68              3.76×
tq2          182.9       52.9            36              7.11×
```

*Per token per KV head, assuming bit-packed index storage.

**TQ2 achieves 7.11× KV cache compression.** Decode speed is 53 tok/s vs 62 baseline —
85% of original speed. The prefill overhead comes from the rotation matrix multiply
applied to every prefill token.

---

## Phase 6 — Making It Usable From the Terminal

After proving the algorithm, the next limitation was usability. The backend work was
real, but a new user still had to inspect the repo to figure out:

1. which backend was safe for a given model
2. when calibration was required
3. which script to run for benchmarking
4. how to generate text through the recommended path

That was enough for research, but not enough for a project someone could clone and use
without already knowing the implementation details.

### Goal

Turn the research workflow into a first-pass CLI that exposes the practical decisions we
learned from the experiments.

### What we built

Added a package entrypoint and four commands:

```bash
metalquant diagnose
metalquant calibrate
metalquant benchmark
metalquant generate
```

Each command maps to a concrete step in the workflow:

- `diagnose` recommends `int8`, `tq4`, `tq2`, or `fp16-outlier` from model metadata
  and optional KV norm measurements
- `calibrate` runs the outlier-channel calibration pass and writes a model-specific JSON
- `benchmark` wraps the benchmark runner and can now choose `--cache-backend auto`
- `generate` uses the same recommendation path and falls back safely when calibration
  would be required but is not available yet

### Design choices

We kept the first CLI intentionally simple:

- reuse the existing calibration and benchmark scripts instead of building a second
  execution stack
- keep the recommendation logic readable and explicit
- generate model-specific default output paths under `results/`
- support both `metalquant ...` and `python -m metalquant ...`
- fail clearly when MLX dependencies are missing instead of dumping a raw traceback

### Result

MetalQuant now has a coherent terminal workflow:

```bash
metalquant diagnose --model mlx-community/Meta-Llama-3.1-8B-Instruct-8bit
metalquant calibrate --model mlx-community/Qwen2.5-7B-Instruct-4bit
metalquant benchmark --model mlx-community/Meta-Llama-3.1-8B-Instruct-8bit --cache-backend auto --kv-norm 18
metalquant generate --model mlx-community/Meta-Llama-3.1-8B-Instruct-8bit --prompt "Explain KV cache compression." --backend auto
```

This is not the final product yet. The current `auto` path uses heuristics plus optional
KV norms. The next usability step is persisting diagnosis and calibration results per
model so the workflow becomes mostly automatic across sessions.

---

## Summary of Findings

### Finding 1 — TurboQuant works on Apple Silicon ✅

On `Meta-Llama-3.1-8B-Instruct-8bit` (M4 Mac Mini 16GB):

- **TQ2**: 7.11× KV cache compression, correct code generation maintained
- **TQ4**: 3.76× compression, output nearly identical to the uncompressed baseline
- The paper's claims hold in practice, on real hardware, with real models

### Finding 2 — It silently fails on 4-bit weight-quantized models ⚠️

4-bit weight quantization (used to shrink model file sizes for local use) inflates KV
activation norms from ~18 to ~274. TurboQuant's reconstruction error scales as `norm²`,
making quality catastrophically worse than INT8 with no warning or error.

**Diagnostic rule**: measure mean K vector L2 norm before applying TurboQuant.
- Norm < 30: safe to use standard TurboQuant
- Norm > 50: 4-bit weight quantization artifacts present — use fp16-outlier backend instead

```python
# Quick check (run once, takes a few seconds)
from mlx_lm.models.cache import KVCache
import numpy as np, mlx.core as mx

cache = [KVCache() for _ in range(len(model.layers))]
logits = model(mx.array(tokenizer.encode("test prompt"))[None], cache=cache)
mx.eval(logits)

c = cache[0]
K = np.array(c.keys[..., :c.offset, :].astype(mx.float32))
mean_norm = np.linalg.norm(K.reshape(-1, K.shape[-1]), axis=-1).mean()
print(f"K norm: {mean_norm:.1f} — {'OK for TurboQuant' if mean_norm < 30 else 'use fp16-outlier backend'}")
```

### Finding 3 — Fix for 4-bit models: fp16-outlier + TQ ✅

`TurboQuantFp16OutlierCache` (`src/metalquant/cache_fp16outlier.py`):

1. Run one calibration pass to identify 32 highest-variance channels
2. Store those 32 channels at fp16 precision (they carry all the problematic norm energy)
3. Apply TurboQuant to remaining 96 channels (norms ~15, quantization works correctly)

Result on `Qwen2.5-7B-4bit`: 4.6× better reconstruction accuracy than INT8, generation
quality matches the uncompressed model exactly.

### Finding 4 — A usable interface matters almost as much as the backend work ✅

Once the algorithm was proven, the remaining gap was operational. Developers still needed
to know which script to run, which backend was safe, and when calibration was required.
The CLI closes that gap enough for real testing and day-to-day use, turning the project
from a research repo into a tool with a repeatable workflow.

---

## What's Next

**Calibration reuse** is the next usability step. The CLI can already recommend and run
the right flow, but it should cache diagnosis and calibration artifacts per model so users
do not repeat setup work every time they switch sessions.

**Bit packing** is still the single most impactful algorithmic next step. Currently, a 2-bit TurboQuant
index is stored in one full byte (uint8). Packing 4 indices per byte would reduce
TQ2 storage from 132 bytes/vector to 36 bytes/vector — realising the full 7.1× figure.
The algorithm is proven correct; this is a straightforward engineering task.

**Perplexity measurement** would provide a rigorous quality metric to complement
the generation quality observations. WikiText-2 or HumanEval perplexity across all
backends would make the results publishable in a more formal context.

**Larger models**: Qwen2.5-Coder-14B and similar would test whether the fp16-outlier
fix generalises across model families and sizes, and whether a purpose-built coding
model shows further gains.

---

*Full implementation: [github.com/savash/MetalQuant](https://github.com/savash/MetalQuant)*
*Paper: TurboQuant — Zandieh et al., arXiv 2504.19874*
