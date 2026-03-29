#!/usr/bin/env python3
"""Run calibration to identify outlier KV channels per layer.

Usage:
    PYTHONPATH=src python benchmarks/run_calibrate.py \
        --model mlx-community/Qwen2.5-7B-Instruct-4bit \
        --out results/calibration-qwen25-7b.json
"""
import argparse
from pathlib import Path

from mlx_lm import load

from prompts import PROMPTS
from metalquant.calibrate import calibrate_outlier_channels, save_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="MetalQuant calibration pass")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--n-outlier", type=int, default=32,
                        help="Number of outlier channels per layer (default 32)")
    parser.add_argument("--out", default="results/calibration.json")
    args = parser.parse_args()

    print(f"loading {args.model} ...")
    model, tokenizer = load(args.model)

    print(f"calibrating on {len(PROMPTS)} prompts ...")
    calib = calibrate_outlier_channels(
        model, tokenizer, PROMPTS, n_outlier=args.n_outlier
    )

    save_calibration(calib, args.out)
    print(f"wrote {args.out}  ({len(calib)} layers, {args.n_outlier} outlier channels each)")

    # Quick sanity print
    layer0 = calib[0]
    print(f"layer 0 head_dim={layer0['head_dim']}, "
          f"outlier_channels[0:8]={layer0['outlier_channels'][:8]}")


if __name__ == "__main__":
    main()
