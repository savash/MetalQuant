# MetalQuant

KV-cache compression for local LLMs on Apple Silicon.

MetalQuant validates TurboQuant on MLX, documents where it fails, and provides a CLI for diagnosing, calibrating, benchmarking, and generating with the right cache backend.

## Current state

- Healthy 8-bit models work well with standard TurboQuant.
- Many 4-bit models need `fp16-outlier` because KV norms are too large.
- The CLI is the main entry point: `diagnose`, `calibrate`, `benchmark`, `generate`.
- Bit packing is not implemented yet, so the best compression numbers are algorithmically validated but not fully realized in memory.

**Measured results on M4 Mac Mini 16GB (`Meta-Llama-3.1-8B-Instruct-8bit`):**

| Backend | KV memory/token | Compression | Decode speed | Quality |
|---|---|---|---|---|
| Baseline | 256 bytes | 1.0× | 61.8 tok/s | reference |
| INT8 | 132 bytes | 1.9× | 57.6 tok/s | matches baseline |
| TQ4 | 68 bytes | 3.8× | 52.5 tok/s | matches baseline |
| TQ2 | 36 bytes | 7.1× | 52.9 tok/s | correct output |

## Quick start

```bash
git clone https://github.com/savash/MetalQuant
cd MetalQuant
./scripts/bootstrap.sh
source scripts/activate.sh
```

Then try:

```bash
metalquant diagnose --model mlx-community/Meta-Llama-3.1-8B-Instruct-8bit

metalquant benchmark \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-8bit \
  --cache-backend auto \
  --kv-norm 18

metalquant generate \
  --model mlx-community/Meta-Llama-3.1-8B-Instruct-8bit \
  --prompt "Explain KV cache compression in simple terms." \
  --backend auto
```

For a 4-bit model:

```bash
metalquant calibrate \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit

metalquant benchmark \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --cache-backend fp16-outlier \
  --calibration results/calibration-qwen2-5-7b-instruct-4bit.json
```

## Backend guide

- `baseline`: reference only
- `int8`: safest default
- `tq4`: best starting point for healthy 8-bit models
- `tq2`: most aggressive option for healthy 8-bit models
- `fp16-outlier`: practical fix for problematic 4-bit models

## Recommended model

`mlx-community/Meta-Llama-3.1-8B-Instruct-8bit`

## Docs

- [docs/ROADMAP.md](docs/ROADMAP.md) — what’s next
- [docs/DEVLOG.md](docs/DEVLOG.md) — full research journal

## License

MIT
