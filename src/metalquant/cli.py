from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import subprocess
import sys

from metalquant.diagnose import diagnose_backend
from metalquant.generate import generate_text


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _script_env() -> dict[str, str]:
    root = _repo_root()
    src = str(root / "src")
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
    return env


def _run_repo_script(script_name: str, script_args: list[str]) -> int:
    root = _repo_root()
    script_path = root / "benchmarks" / script_name
    completed = subprocess.run(
        [sys.executable, str(script_path), *script_args],
        cwd=root,
        env=_script_env(),
        check=False,
    )
    return completed.returncode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="metalquant", description="MetalQuant command-line tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    diagnose_parser = subparsers.add_parser(
        "diagnose",
        help="Recommend a cache backend from model metadata and/or measured KV norms",
    )
    diagnose_parser.add_argument("--model", default=None, help="Model identifier, e.g. mlx-community/Meta-Llama-3.1-8B-Instruct-8bit")
    diagnose_parser.add_argument("--kv-norm", type=float, default=None, help="Measured mean KV norm")
    diagnose_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Run calibration for outlier-aware backends",
    )
    calibrate_parser.add_argument("--model", required=True, help="Model identifier to calibrate")
    calibrate_parser.add_argument("--n-outlier", type=int, default=32, help="Number of outlier channels to keep")
    calibrate_parser.add_argument("--out", default="results/calibration.json", help="Output JSON path")

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run the benchmark suite for a chosen cache backend",
    )
    benchmark_parser.add_argument("--model", required=True, help="Model identifier to benchmark")
    benchmark_parser.add_argument("--cache-backend", default="baseline", help="Cache backend to use")
    benchmark_parser.add_argument("--max-new-tokens", type=int, default=64, help="Decode token count per prompt")
    benchmark_parser.add_argument("--calibration", default=None, help="Calibration JSON path for outlier-aware backends")
    benchmark_parser.add_argument("--out", default="results/experiment.json", help="Output JSON path")

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate text with a selected backend or the auto recommendation path",
    )
    generate_parser.add_argument("--model", required=True, help="Model identifier to use for generation")
    generate_parser.add_argument("--prompt", required=True, help="Prompt text")
    generate_parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "baseline", "int8", "tq2", "tq3", "tq4", "fp16-outlier", "fp16-outlier-tq2"],
        help="Backend to use for generation",
    )
    generate_parser.add_argument("--kv-norm", type=float, default=None, help="Measured mean KV norm to improve auto selection")
    generate_parser.add_argument("--calibration", default=None, help="Calibration JSON path for outlier-aware backends")
    generate_parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum number of generated tokens")
    generate_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "diagnose":
        diagnosis = diagnose_backend(model=args.model, kv_norm=args.kv_norm)
        if args.json:
            print(json.dumps(diagnosis.to_dict(), indent=2))
            return 0

        print(f"Recommended backend: {diagnosis.recommended_backend}")
        print(f"Confidence: {diagnosis.confidence}")
        print()
        print(diagnosis.summary)
        print()
        print("Rationale:")
        for item in diagnosis.rationale:
            print(f"- {item}")
        print()
        print("Next steps:")
        for item in diagnosis.next_steps:
            print(f"- {item}")
        return 0

    if args.command == "calibrate":
        return _run_repo_script(
            "run_calibrate.py",
            [
                "--model",
                args.model,
                "--n-outlier",
                str(args.n_outlier),
                "--out",
                args.out,
            ],
        )

    if args.command == "benchmark":
        script_args = [
            "--model",
            args.model,
            "--cache-backend",
            args.cache_backend,
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--out",
            args.out,
        ]
        if args.calibration:
            script_args.extend(["--calibration", args.calibration])
        return _run_repo_script("run_experiment.py", script_args)

    if args.command == "generate":
        plan, output_text = generate_text(
            model_name=args.model,
            prompt=args.prompt,
            backend=args.backend,
            kv_norm=args.kv_norm,
            calibration_path=args.calibration,
            max_new_tokens=args.max_new_tokens,
        )
        if args.json:
            print(
                json.dumps(
                    {
                        "plan": plan.to_dict(),
                        "output": output_text,
                    },
                    indent=2,
                )
            )
            return 0

        print(f"Selected backend: {plan.selected_backend}")
        print(f"Confidence: {plan.confidence}")
        print()
        print(plan.summary)
        if plan.rationale:
            print()
            print("Rationale:")
            for item in plan.rationale:
                print(f"- {item}")
        if plan.notes:
            print()
            print("Notes:")
            for item in plan.notes:
                print(f"- {item}")
        print()
        print("Output:")
        print(output_text)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
