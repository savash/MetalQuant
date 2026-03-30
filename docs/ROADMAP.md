# MetalQuant Roadmap

## Goal

Make MetalQuant a practical MLX tool for choosing and running the right KV-cache backend on Apple Silicon.

## Done

- Proven TurboQuant works on healthy 8-bit MLX models
- Documented the 4-bit failure mode and built the `fp16-outlier` fix
- Added a user-facing CLI with `diagnose`, `calibrate`, `benchmark`, and `generate`

## Next

- Persist model-specific diagnosis and calibration so setup is one-time
- Upgrade `auto` from heuristics to measured model-state decisions
- Add a simple "prepare this model" workflow
- Publish a clear backend-selection guide for new users

## Later

- Implement bit packing
- Add stronger quality benchmarks
- Broaden model coverage across more families
- Publish a short user guide and benchmark write-up
