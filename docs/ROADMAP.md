# MetalQuant Roadmap

## Phase 1 — Foundation
- [x] Create clean project scaffold
- [x] Add developer-facing README
- [x] Add bootstrap flow
- [x] Add baseline benchmark runner
- [x] Add architecture and roadmap docs
- [x] Add licensing and attribution

## Phase 2 — Baseline discipline
- [x] Run the clean baseline inside MetalQuant
- [x] Save baseline result conventions
- [x] Add result comparison utilities (`benchmarks/compare_results.py`)

## Phase 3 — Cache backend work
- [x] Add cache backend abstraction (`src/metalquant/cache.py`)
- [x] Add first experimental compressed cache backend (INT8, `src/metalquant/cache_quantized.py`)
- [x] Add experiment runner with pluggable cache backend (`benchmarks/run_experiment.py`)
- [x] Run baseline vs INT8 benchmark comparison and save results
- [x] Compare baseline vs compressed cache honestly — see results/report-phase3.md

**Phase 3 findings:**
- INT8: 84% cache reduction, −7.4% decode speed, full quality preservation ✓
- TurboQuant (2-bit/4-bit): correct infrastructure, broken output — missing outlier channel splitting from paper

## Phase 4 — Outlier-Aware TurboQuant
- [ ] Calibration pass: measure per-channel variance across real inference
- [ ] Identify top-32 outlier channels per layer (static, post-rotation)
- [ ] Implement mixed-precision quantize: outlier@3-bit + regular@2-bit = 2.5-bit effective
- [ ] Benchmark 2.5-bit TurboQuant vs INT8 vs baseline
- [ ] Consider replacing rotation matmul with randomized Hadamard transform (O(d log d))

## Phase 5 — Optimization work
- [ ] Profile decode hot path in detail
- [ ] Improve one bottleneck at a time
- [ ] Save before/after reports for every accepted change

## Phase 6 — Publication readiness
- [ ] Add contribution guidelines
- [ ] Add benchmark methodology notes
- [ ] Prepare first public technical writeup
