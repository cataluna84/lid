"""Post-hoc upload of experiment results to W&B.

Reads saved CSVs from an experiment directory and creates a W&B run
so that historical results can be compared alongside new runs in the
central ``lid-bench`` project.

Usage::

    lid-upload experiments/step1_baseline/ --strategy baseline_infer
    lid-upload experiments/step1_baseline/ --strategy baseline_infer --no-artifact
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv


def parse_args():
    p = argparse.ArgumentParser(
        description="Upload experiment results to W&B for central comparison"
    )
    p.add_argument("results_dir", help="Path to experiment output directory")
    p.add_argument("--strategy", default="baseline_infer")
    p.add_argument("--dtype", default="fp16")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--model", default="CohereLabs/tiny-aya-global")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--wandb-project", default="lid-bench")
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-tags", nargs="*", default=None)
    p.add_argument("--no-artifact", action="store_true", help="Skip uploading result files")
    return p.parse_args()


def main():
    load_dotenv()
    args = parse_args()
    import wandb

    acc_path = os.path.join(args.results_dir, "layer_accuracy.csv")
    prob_path = os.path.join(args.results_dir, "layer_avg_probs.csv")
    bench_path = os.path.join(args.results_dir, "benchmark_row.csv")

    if not os.path.exists(acc_path):
        raise FileNotFoundError(f"Missing {acc_path} -- run lid-infer first")

    acc_df = pd.read_csv(acc_path)

    avg_probs = np.zeros(len(acc_df))
    if os.path.exists(prob_path):
        prob_df = pd.read_csv(prob_path)
        avg_probs = prob_df["avg_prob"].values

    extra_metrics: dict[str, float] = {}
    if os.path.exists(bench_path):
        bench_df = pd.read_csv(bench_path)
        if len(bench_df) > 0:
            extra_metrics = bench_df.iloc[0].to_dict()

    tags = args.wandb_tags or [args.strategy]
    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=f"{args.strategy}_{args.dtype}_bs{args.batch_size}_ml{args.max_length}",
        config={
            "strategy": args.strategy,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "model": args.model,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        tags=tags,
    )

    run.define_metric("throughput_sps", summary="max")
    run.define_metric("accuracy_last_layer", summary="max")
    run.define_metric("energy_per_sample_mj", summary="min")
    run.define_metric("gpu_mem_peak_mb", summary="min")

    summary = {"accuracy_last_layer": float(acc_df["accuracy"].iloc[-1])}
    for key in (
        "throughput_sps",
        "latency_ms_per_sample",
        "gpu_mem_peak_mb",
        "energy_per_sample_mj",
        "total_wall_sec",
        "inference_wall_sec",
        "postproc_wall_sec",
    ):
        if key in extra_metrics:
            summary[key] = float(extra_metrics[key])
    run.summary.update(summary)

    layer_table = wandb.Table(columns=["layer_idx", "accuracy", "avg_correct_prob"])
    for i, row in acc_df.iterrows():
        layer_table.add_data(i + 1, row["accuracy"], float(avg_probs[i]))

    run.log(
        {
            "layer_accuracy_table": layer_table,
            "layer_accuracy_curve": wandb.plot.line(
                layer_table,
                "layer_idx",
                "accuracy",
                title=f"Layer Accuracy: {args.strategy}",
            ),
            "layer_prob_curve": wandb.plot.line(
                layer_table,
                "layer_idx",
                "avg_correct_prob",
                title=f"Correct-Class Prob: {args.strategy}",
            ),
        }
    )

    if not args.no_artifact and os.path.isdir(args.results_dir):
        artifact = wandb.Artifact(
            name=f"results-{args.strategy}-{run.id}",
            type="inference-results",
            description=f"Uploaded results for {args.strategy}",
        )
        for fname in os.listdir(args.results_dir):
            fpath = os.path.join(args.results_dir, fname)
            if os.path.isfile(fpath) and not fname.startswith("."):
                artifact.add_file(fpath)
        run.log_artifact(artifact)

    print(f"Uploaded to W&B: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
