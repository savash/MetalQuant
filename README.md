# MetalQuant

MetalQuant is an Apple Silicon-first LLM inference optimization project for MLX.

The project is focused on KV-cache efficiency, long-context behavior, and decode-path performance for local inference workloads on Apple hardware.

## Goals

- benchmark MLX inference reproducibly
- measure KV-cache and decode-path behavior honestly
- develop optimized cache backends for local LLM workloads
- improve long-context efficiency on Apple Silicon
- keep results developer-verifiable and easy to reproduce

## Status

MetalQuant is in early development.

Current repository contents include:

- baseline benchmark runner
- bootstrap/setup flow
- package scaffold
- architecture notes
- roadmap

## Requirements

- Apple Silicon Mac
- macOS
- Python 3.11+
- `mlx`
- `mlx-lm`
- internet access for first-time model downloads

## Repository layout

```text
metalquant/
├── benchmarks/
│   ├── prompts.py
│   └── run_baseline.py
├── docs/
│   ├── ARCHITECTURE.md
│   └── ROADMAP.md
├── results/
├── scripts/
│   └── bootstrap.sh
├── src/
│   └── metalquant/
│       ├── __init__.py
│       ├── config.py
│       └── hardware.py
├── .gitignore
├── LICENSE
├── NOTICE
├── pyproject.toml
└── README.md
```

## Installation

### Bootstrap

```bash
cd metalquant
./scripts/bootstrap.sh
source .venv/bin/activate
```

If `uv` is not installed:

```bash
curl -L https://astral.sh/uv/install.sh | sh
```

### Manual install

```bash
cd metalquant
uv python install 3.11
uv venv --python 3.11 .venv
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install -U pip setuptools wheel
python -m pip install -e .
```

## Running the baseline benchmark

```bash
cd metalquant
source .venv/bin/activate
PYTHONPATH=src python benchmarks/run_baseline.py \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --max-new-tokens 64 \
  --out results/baseline-qwen25-7b.json
```

## Running with a different model

Any model loadable by `mlx-lm` can be used.

Example:

```bash
PYTHONPATH=src python benchmarks/run_baseline.py \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --max-new-tokens 96 \
  --out results/qwen25-1_5b-baseline.json
```

## Current baseline

Current native MetalQuant baseline (`results/baseline-qwen25-7b.json`):

- model: `mlx-community/Qwen2.5-7B-Instruct-4bit`
- max new tokens: `64`
- average prefill throughput: `78.98 tok/s`
- average decode throughput: `24.66 tok/s`
- average decode latency: `41.20 ms`
- average cache size after decode: `14,680,064 bytes`

These numbers come from the current baseline benchmark runner in this repository.

## Benchmark output

Current benchmark output includes:

- benchmark configuration
- hardware metadata
- per-prompt outputs
- prefill throughput
- decode throughput
- average decode latency
- cache size after decode

Outputs are written as JSON to support scripted comparisons and repeatable reporting.

Example summary payload:

```json
{
  "summary": {
    "avg_prefill_tok_per_s": 78.98479418494838,
    "avg_decode_tok_per_s": 24.663847482740795,
    "avg_decode_latency_ms": 41.20268941630249,
    "avg_cache_bytes_after_decode": 14680064.0
  }
}
```

## Development approach

MetalQuant follows a simple rule:

1. measure the baseline
2. profile the bottleneck
3. change one thing
4. measure again

## Documentation

- `docs/ARCHITECTURE.md`
- `docs/ROADMAP.md`

## Author

**Savash Kalay**  
<savash@mac.com>

## License

MIT
