# MetalQuant Architecture

## Intent

MetalQuant is a clean Apple Silicon-first project for MLX inference optimization.

The initial focus is KV-cache efficiency, long-context behavior, and decode-path performance.

## Layers

### 1. Benchmark layer
- reproducible prompt suites
- model selection
- output capture
- summary metrics

### 2. Runtime adapter layer
- MLX / mlx-lm loading
- cache backend integration points
- hardware-aware execution assumptions

### 3. Cache backend layer
- baseline cache
- future compressed cache backends
- future adaptive or asymmetric variants

### 4. Profiling and reporting layer
- decode timing
- cache size tracking
- saved JSON outputs
- reproducible before/after comparisons

## Immediate direction

- baseline first
- cache abstraction second
- compressed-cache experiments third
- optimization only after measurement exists

## Non-goals for now

- training or fine-tuning
- non-Apple backends
- premature branding over reproducible engineering
