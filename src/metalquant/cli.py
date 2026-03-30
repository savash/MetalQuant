from __future__ import annotations

import argparse
import json

from metalquant.diagnose import diagnose_backend


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

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
