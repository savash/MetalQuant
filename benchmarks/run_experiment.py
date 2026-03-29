#!/usr/bin/env python3
"""MetalQuant experiment benchmark runner.

Accepts a --cache-backend argument to select the cache implementation,
otherwise identical in output format to run_baseline.py so results can
be fed directly into compare_results.py.

Usage:
    PYTHONPATH=src python benchmarks/run_experiment.py \\
        --model mlx-community/Qwen2.5-7B-Instruct-4bit \\
        --cache-backend int8 \\
        --max-new-tokens 64 \\
        --out results/int8-cache.json
"""
import argparse
import json
import resource
import statistics
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from prompts import PROMPTS
from metalquant.cache import make_cache
from metalquant.config import BenchmarkConfig
from metalquant.hardware import detect_hardware


def rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def cache_nbytes(cache_list) -> int:
    total = 0
    for cache in cache_list:
        try:
            total += int(cache.nbytes)
        except Exception:
            pass
    return total


def run_one(model, tokenizer, prompt: str, max_new_tokens: int, cache_backend: str) -> dict:
    cache_list = make_cache(model, backend=cache_backend)
    input_ids = mx.array(tokenizer.encode(prompt))[None]

    rss_before = rss_bytes()
    t0 = time.perf_counter()
    logits = model(input_ids, cache=cache_list)
    mx.eval(logits)
    prefill_seconds = time.perf_counter() - t0
    rss_after_prefill = rss_bytes()

    token = mx.argmax(logits[:, -1, :], axis=-1)
    generated = [int(token.item())]
    decode_latencies = []

    for _ in range(max_new_tokens - 1):
        t1 = time.perf_counter()
        logits = model(token.reshape(1, 1), cache=cache_list)
        mx.eval(logits)
        decode_latencies.append(time.perf_counter() - t1)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        token_id = int(token.item())
        generated.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break

    decode_seconds = sum(decode_latencies)
    rss_after_decode = rss_bytes()

    return {
        "prompt": prompt,
        "output": tokenizer.decode(generated),
        "prefill_tokens": int(input_ids.shape[1]),
        "decode_tokens": len(generated),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "prefill_tok_per_s": (int(input_ids.shape[1]) / prefill_seconds) if prefill_seconds > 0 else 0.0,
        "decode_tok_per_s": (len(generated) / decode_seconds) if decode_seconds > 0 else 0.0,
        "avg_decode_latency_ms": statistics.mean(decode_latencies) * 1000.0 if decode_latencies else 0.0,
        "cache_bytes_after_decode": cache_nbytes(cache_list),
        "rss_before": rss_before,
        "rss_after_prefill": rss_after_prefill,
        "rss_after_decode": rss_after_decode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MetalQuant experiment benchmark runner")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--cache-backend",
        default="baseline",
        choices=["baseline", "int8", "turboquant", "tq2", "tq3", "tq4"],
        help="Cache backend to use (default: baseline)",
    )
    parser.add_argument("--out", default="results/experiment.json")
    args = parser.parse_args()

    config = BenchmarkConfig(model=args.model, max_new_tokens=args.max_new_tokens)
    model, tokenizer = load(config.model)

    runs = [
        run_one(model, tokenizer, prompt, config.max_new_tokens, args.cache_backend)
        for prompt in PROMPTS
    ]
    summary = {
        "avg_prefill_tok_per_s": sum(r["prefill_tok_per_s"] for r in runs) / len(runs),
        "avg_decode_tok_per_s": sum(r["decode_tok_per_s"] for r in runs) / len(runs),
        "avg_decode_latency_ms": sum(r["avg_decode_latency_ms"] for r in runs) / len(runs),
        "avg_cache_bytes_after_decode": sum(r["cache_bytes_after_decode"] for r in runs) / len(runs),
    }

    payload = {
        "config": {**config.to_dict(), "cache_backend": args.cache_backend},
        "hardware": detect_hardware(),
        "runs": runs,
        "summary": summary,
    }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
