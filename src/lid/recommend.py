"""Recommend the best inference hyperparameters from benchmark results.

Reads local ``experiments/all_results.csv`` (or W&B) to find the optimal
configuration and prints a ready-to-paste ``lid-infer`` command.

Usage::

    lid-recommend                          # best throughput from local CSV
    lid-recommend --optimize energy        # lowest energy per sample
    lid-recommend --optimize memory        # lowest GPU memory
    lid-recommend --min-accuracy 0.03      # only configs with acc >= 3%
    lid-recommend --from-wandb             # fetch from W&B instead of CSV
    lid-recommend --json                   # machine-readable output
    lid-recommend --top 5                  # show top-5 configs
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv

from lid.constants import DEFAULT_MODEL

OPTIMIZE_METRICS = {
    "throughput": ("throughput_sps", True),
    "energy": ("energy_per_sample_mj", False),
    "memory": ("gpu_mem_peak_mb", False),
    "latency": ("latency_ms_per_sample", False),
}

LOCAL_CSV = Path("experiments/all_results.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Recommend best inference config from benchmark results",
    )
    p.add_argument(
        "--optimize",
        choices=list(OPTIMIZE_METRICS),
        default="throughput",
        help="Metric to optimize (default: throughput)",
    )
    p.add_argument(
        "--min-accuracy",
        type=float,
        default=0.0,
        help="Minimum last-layer accuracy to consider (default: 0.0)",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to results CSV (default: experiments/all_results.csv)",
    )
    p.add_argument(
        "--from-wandb",
        action="store_true",
        help="Fetch results from W&B instead of local CSV",
    )
    p.add_argument(
        "--project",
        default="lid-bench",
        help="W&B project name (used with --from-wandb)",
    )
    p.add_argument(
        "--entity",
        default=None,
        help="W&B entity (used with --from-wandb)",
    )
    p.add_argument(
        "--top",
        type=int,
        default=1,
        help="Number of top configs to show (default: 1)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output as JSON for scripting",
    )
    return p.parse_args()


def _load_from_csv(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    {
                        "strategy": row["strategy"],
                        "dtype": row["dtype"],
                        "batch_size": int(row["batch_size"]),
                        "max_length": int(row["max_length"]),
                        "throughput_sps": float(row["throughput_sps"]),
                        "latency_ms_per_sample": float(
                            row.get("latency_ms_per_sample", 0) or row.get("latency_ms", 0)
                        ),
                        "gpu_mem_peak_mb": float(row["gpu_mem_peak_mb"]),
                        "energy_per_sample_mj": float(row["energy_per_sample_mj"]),
                        "accuracy_last_layer": float(row["accuracy_last_layer"]),
                        "wall_sec": float(row.get("wall_sec", 0) or row.get("total_wall_sec", 0)),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def _load_from_wandb(project: str, entity: str | None) -> list[dict]:
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"state": "finished"})

    rows: list[dict] = []
    for run in runs:
        s = run.summary
        c = run.config
        sps = s.get("throughput_sps", 0)
        if not sps:
            continue
        rows.append(
            {
                "strategy": c.get("strategy", "unknown"),
                "dtype": c.get("dtype", "fp16"),
                "batch_size": int(c.get("batch_size", 0)),
                "max_length": int(c.get("max_length", 0)),
                "throughput_sps": float(sps),
                "latency_ms_per_sample": float(s.get("latency_ms_per_sample", 0)),
                "gpu_mem_peak_mb": float(s.get("gpu_mem_peak_mb", 0)),
                "energy_per_sample_mj": float(s.get("energy_per_sample_mj", 0)),
                "accuracy_last_layer": float(s.get("accuracy_last_layer", 0)),
                "wall_sec": float(s.get("total_wall_sec", 0)),
            }
        )
    return rows


def _group_and_rank(
    rows: list[dict],
    optimize: str,
    min_accuracy: float,
) -> list[dict]:
    metric_key, higher_is_better = OPTIMIZE_METRICS[optimize]

    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["strategy"], r["dtype"], r["batch_size"], r["max_length"])
        groups.setdefault(key, []).append(r)

    ranked: list[dict] = []
    for (strategy, dtype, bs, ml), runs in groups.items():
        mean_acc = statistics.mean(r["accuracy_last_layer"] for r in runs)
        if mean_acc < min_accuracy:
            continue

        mean_sps = statistics.mean(r["throughput_sps"] for r in runs)
        mean_metric = statistics.mean(r[metric_key] for r in runs)
        std_metric = statistics.stdev(r[metric_key] for r in runs) if len(runs) > 1 else 0.0
        mean_mem = statistics.mean(r["gpu_mem_peak_mb"] for r in runs)
        mean_energy = statistics.mean(r["energy_per_sample_mj"] for r in runs)
        mean_latency = statistics.mean(r["latency_ms_per_sample"] for r in runs)
        mean_wall = statistics.mean(r["wall_sec"] for r in runs)

        ranked.append(
            {
                "strategy": strategy,
                "dtype": dtype,
                "batch_size": bs,
                "max_length": ml,
                "n_repeats": len(runs),
                "throughput_sps": round(mean_sps, 1),
                "latency_ms_per_sample": round(mean_latency, 2),
                "gpu_mem_peak_mb": round(mean_mem, 0),
                "energy_per_sample_mj": round(mean_energy, 0),
                "accuracy_last_layer": round(mean_acc, 4),
                "wall_sec": round(mean_wall, 1),
                f"{metric_key}_std": round(std_metric, 2),
                "_sort_key": mean_metric,
            }
        )

    ranked.sort(key=lambda r: r["_sort_key"], reverse=higher_is_better)
    for r in ranked:
        del r["_sort_key"]

    return ranked


def _build_infer_command(rec: dict) -> str:
    return (
        f"uv run lid-infer"
        f" --model {DEFAULT_MODEL}"
        f" --batch-size {rec['batch_size']}"
        f" --max-length {rec['max_length']}"
    )


def _print_recommendation(ranked: list[dict], optimize: str, top: int) -> None:
    metric_label = {
        "throughput": "throughput (sps)",
        "energy": "energy (mJ/sample)",
        "memory": "GPU memory (MB)",
        "latency": "latency (ms/sample)",
    }[optimize]

    print()
    print("=" * 80)
    print(f"LID-RECOMMEND  |  Optimizing: {metric_label}")
    print("=" * 80)

    for i, rec in enumerate(ranked[:top], 1):
        if i > 1:
            print("-" * 80)

        rank_label = f"#{i}" if top > 1 else "BEST"
        print(
            f"\n  {rank_label}: {rec['strategy']}  /  {rec['dtype']}"
            f"  /  bs={rec['batch_size']}  /  ml={rec['max_length']}"
        )
        print(f"       Throughput:  {rec['throughput_sps']:>8.1f} sps")
        print(f"       Latency:    {rec['latency_ms_per_sample']:>8.2f} ms/sample")
        print(f"       GPU Memory: {rec['gpu_mem_peak_mb']:>8,.0f} MB")
        print(f"       Energy:     {rec['energy_per_sample_mj']:>8,.0f} mJ/sample")
        print(f"       Accuracy:   {rec['accuracy_last_layer']:>8.4f}")
        print(f"       Wall time:  {rec['wall_sec']:>8.1f} s")
        print(f"       Repeats:    {rec['n_repeats']}")
        print()
        print("  Command:")
        print(f"    {_build_infer_command(rec)}")

    print()
    print("=" * 80)
    print()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.from_wandb:
        rows = _load_from_wandb(args.project, args.entity)
        source = f"W&B project: {args.project}"
    else:
        csv_path = Path(args.csv) if args.csv else LOCAL_CSV
        if not csv_path.exists():
            print(f"Results CSV not found: {csv_path}", file=sys.stderr)
            print("Run benchmarks first, or use --from-wandb.", file=sys.stderr)
            sys.exit(1)
        rows = _load_from_csv(csv_path)
        source = str(csv_path)

    if not rows:
        print(f"No results found in {source}.", file=sys.stderr)
        sys.exit(1)

    ranked = _group_and_rank(rows, args.optimize, args.min_accuracy)

    if not ranked:
        print(
            f"No configs meet the criteria (min_accuracy={args.min_accuracy}).",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.json_output:
        output = {
            "source": source,
            "optimize": args.optimize,
            "min_accuracy": args.min_accuracy,
            "recommendations": ranked[: args.top],
            "infer_command": _build_infer_command(ranked[0]),
        }
        json.dump(output, sys.stdout, indent=2)
        print()
    else:
        _print_recommendation(ranked, args.optimize, args.top)


if __name__ == "__main__":
    main()
