#!/usr/bin/env python3
"""Compare two MetalQuant benchmark JSON result files and print a delta table.

Usage:
    python benchmarks/compare_results.py results/baseline.json results/experiment.json
    python benchmarks/compare_results.py results/baseline.json results/experiment.json --md report.md
"""
import argparse
import json
from pathlib import Path


METRICS = [
    ("avg_prefill_tok_per_s", "Prefill tok/s", "higher_is_better"),
    ("avg_decode_tok_per_s", "Decode tok/s", "higher_is_better"),
    ("avg_decode_latency_ms", "Decode latency (ms)", "lower_is_better"),
    ("avg_cache_bytes_after_decode", "Cache size (bytes)", "lower_is_better"),
]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def fmt_bytes(n: float) -> str:
    if n >= 1_048_576:
        return f"{n / 1_048_576:.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n:.0f} B"


def fmt_value(key: str, val: float) -> str:
    if "bytes" in key:
        return fmt_bytes(val)
    if "latency" in key:
        return f"{val:.2f} ms"
    return f"{val:.2f}"


def delta_str(key: str, base: float, exp: float, direction: str) -> str:
    if base == 0:
        return "N/A"
    pct = (exp - base) / base * 100
    sign = "+" if pct >= 0 else ""
    improve = (pct > 0 and direction == "higher_is_better") or (
        pct < 0 and direction == "lower_is_better"
    )
    tag = " [BETTER]" if improve else (" [WORSE]" if pct != 0 else "")
    return f"{sign}{pct:.1f}%{tag}"


def compare(base: dict, exp: dict) -> list[dict]:
    base_s = base["summary"]
    exp_s = exp["summary"]
    rows = []
    for key, label, direction in METRICS:
        bv = base_s.get(key, 0.0)
        ev = exp_s.get(key, 0.0)
        rows.append(
            {
                "metric": label,
                "baseline": fmt_value(key, bv),
                "experiment": fmt_value(key, ev),
                "delta": delta_str(key, bv, ev, direction),
            }
        )
    return rows


def print_table(rows: list[dict], base_path: str, exp_path: str) -> None:
    col_widths = {
        "metric": max(len("Metric"), max(len(r["metric"]) for r in rows)),
        "baseline": max(len("Baseline"), max(len(r["baseline"]) for r in rows)),
        "experiment": max(len("Experiment"), max(len(r["experiment"]) for r in rows)),
        "delta": max(len("Delta"), max(len(r["delta"]) for r in rows)),
    }

    def row_str(cells: dict) -> str:
        return (
            f"  {cells['metric']:<{col_widths['metric']}}  "
            f"{cells['baseline']:>{col_widths['baseline']}}  "
            f"{cells['experiment']:>{col_widths['experiment']}}  "
            f"{cells['delta']}"
        )

    sep = "-" * (sum(col_widths.values()) + 12)
    header = row_str({"metric": "Metric", "baseline": "Baseline", "experiment": "Experiment", "delta": "Delta"})

    print()
    print(f"  baseline:   {base_path}")
    print(f"  experiment: {exp_path}")
    print()
    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print(row_str(r))
    print(sep)
    print()


def build_markdown(rows: list[dict], base_path: str, exp_path: str) -> str:
    lines = [
        "# MetalQuant Benchmark Comparison",
        "",
        f"- **Baseline:** `{base_path}`",
        f"- **Experiment:** `{exp_path}`",
        "",
        "| Metric | Baseline | Experiment | Delta |",
        "|--------|----------|------------|-------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['metric']} | {r['baseline']} | {r['experiment']} | {r['delta']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two MetalQuant benchmark results")
    parser.add_argument("baseline", type=Path, help="Baseline JSON result file")
    parser.add_argument("experiment", type=Path, help="Experiment JSON result file")
    parser.add_argument("--md", type=Path, default=None, help="Write markdown report to this path")
    args = parser.parse_args()

    base = load(args.baseline)
    exp = load(args.experiment)
    rows = compare(base, exp)

    print_table(rows, str(args.baseline), str(args.experiment))

    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(build_markdown(rows, str(args.baseline), str(args.experiment)))
        print(f"wrote {args.md}")


if __name__ == "__main__":
    main()
