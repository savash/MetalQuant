# MetalQuant Roadmap

## Phase 1 — Foundation ✅
- [x] Project scaffold, README, bootstrap flow
- [x] Baseline benchmark runner
- [x] Architecture and roadmap docs
- [x] Licensing and attribution

## Phase 2 — Baseline & Infrastructure ✅
- [x] Reproducible baseline benchmarks
- [x] Result comparison utilities (`benchmarks/compare_results.py`)
- [x] Cache backend abstraction (`src/metalquant/cache.py`)
- [x] Pluggable experiment runner (`benchmarks/run_experiment.py`)

## Phase 3 — INT8 KV Cache ✅
- [x] INT8 backend (`src/metalquant/cache_quantized.py`)
- [x] Fixed critical mlx-lm integration bugs (offset tracking, `self.bits` conflict)
- [x] Benchmarked INT8 vs baseline

**Results:** 1.94× KV cache compression, decode speed unchanged, full quality.

## Phase 4 — TurboQuant Investigation & Fix ✅
- [x] Implemented TurboQuant Q_mse algorithm (`src/metalquant/cache_turboquant.py`)
- [x] Discovered silent failure on 4-bit weight-quantized models (KV norm amplification)
- [x] Proved per-channel normalization does not fix the problem (math documented)
- [x] Built calibration pass (`src/metalquant/calibrate.py`, `benchmarks/run_calibrate.py`)
- [x] Built fp16-outlier fix (`src/metalquant/cache_fp16outlier.py`)
- [x] GPU-native channel merge (scatter matmul — no CPU round-trips)

**Results on Qwen2.5-7B-4bit:** fp16-outlier+TQ4 gives 4.6× better reconstruction
accuracy than INT8, generation quality matches uncompressed baseline exactly.

## Phase 5 — Proof on Correct Model ✅
- [x] Switched to `Meta-Llama-3.1-8B-Instruct-8bit` (paper's model family)
- [x] Confirmed healthy KV norms (~18 across all layers)
- [x] TQ4 output identical to baseline, TQ2 output correct and coherent
- [x] Full benchmark suite on M4 Mac Mini 16GB
- [x] Published research journal (`docs/DEVLOG.md`)

**Results:**

| Backend | KV compression (packed) | Decode speed | Quality |
|---|---|---|---|
| INT8 | 1.94× | 57.6 tok/s | ✅ matches baseline |
| TQ4 | 3.76× | 52.5 tok/s | ✅ matches baseline |
| TQ2 | **7.11×** | 52.9 tok/s | ✅ correct output |

## Phase 6 — Bit Packing (Next)
- [ ] Pack 2-bit indices: 4 indices per byte instead of 1
- [ ] TQ2: 128 bytes + 4 = 132 today → 32 bytes + 4 = 36 bytes (realise full 7.1× compression)
- [ ] TQ4: 128 bytes + 4 = 132 today → 64 bytes + 4 = 68 bytes (realise full 3.8×)

## Phase 7 — Rigorous Quality Evaluation
- [ ] WikiText-2 perplexity for all backends
- [ ] HumanEval pass@1 for coding quality
- [ ] Formal quality vs compression curve

## Phase 8 — Broader Model Coverage
- [ ] Qwen2.5-Coder-14B (larger 4-bit model — test fp16-outlier generalisation)
- [ ] Mistral-7B-8bit
- [ ] Gemma-3-12B
- [ ] Document which model families are safe for standard TurboQuant vs needing fp16-outlier

## Phase 9 — Publication
- [ ] Blog post with benchmark methodology and findings
- [ ] "Check your model" guide (KV norm diagnostic)
- [ ] Contribution guidelines
