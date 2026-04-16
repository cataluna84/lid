"""Generate a comparison report from W&B runs in the lid-bench project.

Fetches all finished runs, identifies the baseline, computes speedup
and memory deltas, and prints a terminal summary table.  Optionally
logs a W&B summary Table for dashboard use.

Usage::

    lid-report                              # all runs in lid-bench
    lid-report --group runbook-v1           # filter by group
    lid-report --log-summary               # also log comparison table to W&B
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv


def parse_args():
    p = argparse.ArgumentParser(description="Generate comparison report from W&B lid-bench runs")
    p.add_argument("--project", default="lid-bench")
    p.add_argument("--entity", default=None, help="W&B entity (user or team)")
    p.add_argument("--group", default=None, help="Filter runs by W&B group")
    p.add_argument(
        "--baseline-tag",
        default="baseline",
        help="Tag used to identify the baseline run",
    )
    p.add_argument(
        "--log-summary",
        action="store_true",
        help="Log a comparison Table back to W&B",
    )
    p.add_argument(
        "--best-per-strategy",
        action="store_true",
        help="Show only the best config per strategy",
    )
    return p.parse_args()


def _fetch_runs(project, entity=None, group=None):
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    filters = {"state": "finished"}
    if group:
        filters["group"] = group
    return list(api.runs(path, filters=filters))


def _find_baseline(runs, baseline_tag):
    for run in runs:
        if baseline_tag in (run.tags or []):
            return run
    for run in runs:
        if run.config.get("strategy") in ("baseline_infer", "eager"):
            return run
    return runs[0] if runs else None


def _build_rows(runs, baseline):
    baseline_sps = baseline.summary.get("throughput_sps", 0) if baseline else 1.0
    baseline_mem = baseline.summary.get("gpu_mem_peak_mb", 0) if baseline else 1.0

    rows = []
    for run in runs:
        sps = run.summary.get("throughput_sps", 0)
        mem = run.summary.get("gpu_mem_peak_mb", 0)
        acc = run.summary.get("accuracy_last_layer", 0)
        energy = run.summary.get("energy_per_sample_mj", 0)
        latency = run.summary.get("latency_ms_per_sample", 0)

        rows.append(
            {
                "run_name": run.name,
                "strategy": run.config.get("strategy", "unknown"),
                "dtype": run.config.get("dtype", "fp16"),
                "batch_size": run.config.get("batch_size", 0),
                "max_length": run.config.get("max_length", 0),
                "throughput_sps": sps,
                "speedup": sps / max(baseline_sps, 1e-9),
                "gpu_mem_peak_mb": mem,
                "mem_savings_pct": (1 - mem / max(baseline_mem, 1e-9)) * 100
                if baseline_mem > 0
                else 0,
                "accuracy_last_layer": acc,
                "energy_per_sample_mj": energy,
                "latency_ms": latency,
                "wall_sec": run.summary.get("total_wall_sec", 0),
            }
        )

    return sorted(rows, key=lambda r: r["throughput_sps"], reverse=True)


def _best_per_strategy(rows):
    seen: dict[str, dict] = {}
    for r in rows:
        key = r["strategy"]
        if key not in seen or r["throughput_sps"] > seen[key]["throughput_sps"]:
            seen[key] = r
    return sorted(seen.values(), key=lambda r: r["throughput_sps"], reverse=True)


def _print_report(rows, baseline_name):
    print()
    print("=" * 110)
    print("LID-BENCH COMPARISON REPORT")
    print(f"Baseline: {baseline_name}")
    print("=" * 110)
    header = (
        f"{'strategy':<25} {'dtype':<6} {'bs':>4} {'ml':>4} "
        f"{'sps':>8} {'speedup':>8} {'mem_mb':>8} {'mem_sav':>8} "
        f"{'acc':>6} {'energy':>8}"
    )
    print(header)
    print("-" * 110)
    for r in rows:
        is_baseline = r["speedup"] == 1.0 and r["strategy"] in ("baseline_infer", "eager")
        marker = " *" if is_baseline else ""
        print(
            f"{r['strategy']:<25} {r['dtype']:<6} {r['batch_size']:>4} "
            f"{r['max_length']:>4} {r['throughput_sps']:>8.1f} "
            f"{r['speedup']:>7.1f}x {r['gpu_mem_peak_mb']:>8.0f} "
            f"{r['mem_savings_pct']:>7.1f}% {r['accuracy_last_layer']:>6.3f} "
            f"{r['energy_per_sample_mj']:>8.1f}{marker}"
        )
    print("-" * 110)
    print("  * = baseline run")
    print()


def _log_summary_to_wandb(rows, project, group):
    import wandb

    columns = [
        "strategy",
        "dtype",
        "batch_size",
        "max_length",
        "throughput_sps",
        "speedup",
        "gpu_mem_peak_mb",
        "mem_savings_pct",
        "accuracy_last_layer",
        "energy_per_sample_mj",
        "latency_ms",
        "wall_sec",
    ]
    table = wandb.Table(columns=columns)
    for r in rows:
        table.add_data(*[r.get(c, 0) for c in columns])

    run = wandb.init(
        project=project,
        group=group,
        name="comparison-report",
        job_type="report",
        tags=["report"],
    )
    run.log({"comparison_table": table})

    bar_data = [[r["strategy"], r["speedup"]] for r in rows]
    bar_table = wandb.Table(data=bar_data, columns=["strategy", "speedup_vs_baseline"])
    run.log(
        {
            "speedup_chart": wandb.plot.bar(
                bar_table,
                "strategy",
                "speedup_vs_baseline",
                title="Speedup vs Baseline",
            ),
        }
    )

    print(f"Summary table logged to W&B: {run.url}")
    run.finish()


def main():
    load_dotenv()
    args = parse_args()

    try:
        runs = _fetch_runs(args.project, args.entity, args.group)
    except Exception as e:
        print(f"Failed to fetch runs from W&B: {e}", file=sys.stderr)
        sys.exit(1)

    if not runs:
        print("No finished runs found.", file=sys.stderr)
        sys.exit(1)

    baseline = _find_baseline(runs, args.baseline_tag)
    if baseline is None:
        print("Could not identify a baseline run.", file=sys.stderr)
        sys.exit(1)

    rows = _build_rows(runs, baseline)

    if args.best_per_strategy:
        rows = _best_per_strategy(rows)

    _print_report(rows, baseline.name)

    if args.log_summary:
        try:
            _log_summary_to_wandb(rows, args.project, args.group)
        except Exception as e:
            print(f"W&B summary logging failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
