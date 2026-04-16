import argparse
import csv
import gc
import os
import pickle

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from lid.bench.metrics import MetricsCollector
from lid.constants import DEFAULT_DATASET, DEFAULT_DATASET_FILE, DEFAULT_MODEL, VALID_OPTIONS
from lid.data import load_lid_dataset
from lid.model import batched_layer_text_outputs, load_model_and_tokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Run layer-wise LID inference and analysis")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dataset", default=DEFAULT_DATASET)
    p.add_argument("--dataset-file", default=DEFAULT_DATASET_FILE)
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--sample-frac", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    p.add_argument("--wandb-project", default="lid-bench")
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-tags", nargs="*", default=["baseline"])
    return p.parse_args()


def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    max_list = []
    prob_list = []

    for i in tqdm(range(len(df)), desc="Post-processing"):
        row_data = df.iloc[i]["DATA"]
        actual_iso = df.iloc[i]["ISO-693-3"]
        layer_keys = sorted(k for k in row_data if k.startswith("LAYER-"))

        max_row_dict = {}
        prob_row_list = []

        for layer in layer_keys:
            layer_content = row_data[layer]
            probs = {
                k.replace("norm_prob_", ""): v
                for k, v in layer_content.items()
                if k.startswith("norm_prob_")
            }
            best_label = max(probs, key=probs.get)
            best_val = probs[best_label]
            max_row_dict[layer] = {"label": best_label, "prob": float(best_val)}
            prob_row_list.append(float(layer_content.get(f"norm_prob_{actual_iso}", 0.0)))

        max_list.append(max_row_dict)
        prob_list.append(prob_row_list)

    df["MAX"] = pd.Series(max_list, index=df.index, dtype=object)
    df["PROB"] = pd.Series(prob_list, index=df.index, dtype=object)
    return df


def compute_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    layer_keys = sorted(k for k in df.iloc[0]["MAX"] if k.startswith("LAYER-"))
    for layer in layer_keys:
        correct = sum(
            1 for _, row in df.iterrows() if row["MAX"][layer]["label"] == row["ISO-693-3"]
        )
        records.append({"layer": layer, "accuracy": correct / len(df), "total": len(df)})
    return pd.DataFrame(records)


def _save_benchmark_row(args, acc_df, run_metrics):
    """Save a single CSV row compatible with lid-bench output."""
    row = {
        "strategy": "baseline_infer",
        "dtype": "fp16",
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "repeat_idx": 0,
        "accuracy_last_layer": acc_df["accuracy"].iloc[-1],
        **run_metrics.to_dict(),
    }
    path = os.path.join(args.output_dir, "benchmark_row.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"Benchmark row saved to {path}")


def _log_to_wandb(args, acc_df, avg_probs, run_metrics):
    """Log inference results to W&B for central comparison."""
    import wandb

    run = wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=f"baseline_infer_bs{args.batch_size}_ml{args.max_length}",
        config={
            "strategy": "baseline_infer",
            "dtype": "fp16",
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "model": args.model,
            "temperature": args.temperature,
            "n_samples": acc_df["total"].iloc[0],
            "sample_frac": args.sample_frac,
            "seed": args.seed,
            "dataset": args.dataset,
            "dataset_file": args.dataset_file,
        },
        tags=args.wandb_tags or ["baseline"],
    )

    run.define_metric("throughput_sps", summary="max")
    run.define_metric("accuracy_last_layer", summary="max")
    run.define_metric("energy_per_sample_mj", summary="min")
    run.define_metric("gpu_mem_peak_mb", summary="min")

    run.summary.update(run_metrics.to_dict())
    run.summary["accuracy_last_layer"] = float(acc_df["accuracy"].iloc[-1])

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
                title="Layer Accuracy: baseline_infer",
            ),
            "layer_prob_curve": wandb.plot.line(
                layer_table,
                "layer_idx",
                "avg_correct_prob",
                title="Correct-Class Prob: baseline_infer",
            ),
        }
    )

    artifact = wandb.Artifact(
        name=f"baseline-results-{run.id}",
        type="inference-results",
        description="Baseline inference results from lid-infer",
    )
    artifact.add_dir(args.output_dir)
    run.log_artifact(artifact)

    print(f"W&B run logged: {run.url}")
    run.finish()


def main():
    load_dotenv()
    args = parse_args()
    torch.manual_seed(args.seed)

    collector = MetricsCollector()
    collector.start_run()

    print(f"Loading model: {args.model}")
    model, tokenizer, _device = load_model_and_tokenizer(args.model)
    collector.record_model_loaded()

    print("Loading dataset...")
    df = load_lid_dataset(
        dataset_name=args.dataset,
        file_path=args.dataset_file,
        token=args.hf_token,
        sample_frac=args.sample_frac,
        random_state=args.seed,
    )
    print(f"Running inference on {len(df)} samples (batch_size={args.batch_size})")

    all_results = []
    collector.start_inference()
    for i in tqdm(range(0, len(df), args.batch_size), desc="Inference"):
        batch_prompts = df["INSTRUCT"].iloc[i : i + args.batch_size].tolist()
        batch_out = batched_layer_text_outputs(
            model,
            tokenizer,
            batch_prompts,
            valid_options=VALID_OPTIONS,
            temperature=args.temperature,
            max_length=args.max_length,
        )
        all_results.extend(batch_out)
        if i % (args.batch_size * 10) == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    collector.end_inference()

    df["DATA"] = all_results

    collector.start_postprocessing()
    print("Post-processing results...")
    df = postprocess(df)

    acc_df = compute_accuracy(df)
    print("\nPer-layer accuracy:")
    print(acc_df.to_string(index=False))

    n_batches = (len(df) + args.batch_size - 1) // args.batch_size
    run_metrics = collector.finalize(len(df), n_batches)

    os.makedirs(args.output_dir, exist_ok=True)
    pkl_path = os.path.join(args.output_dir, "results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(df, f)
    print(f"\nResults saved to {pkl_path}")

    acc_path = os.path.join(args.output_dir, "layer_accuracy.csv")
    acc_df.to_csv(acc_path, index=False)
    print(f"Layer accuracy saved to {acc_path}")

    avg_probs = np.mean(np.stack(df["PROB"].values), axis=0)
    prob_path = os.path.join(args.output_dir, "layer_avg_probs.csv")
    pd.DataFrame({"layer": range(1, len(avg_probs) + 1), "avg_prob": avg_probs}).to_csv(
        prob_path, index=False
    )
    print(f"Average probabilities saved to {prob_path}")

    _save_benchmark_row(args, acc_df, run_metrics)

    if not args.no_wandb:
        try:
            _log_to_wandb(args, acc_df, avg_probs, run_metrics)
        except Exception as e:
            print(f"W&B logging failed (use --no-wandb to skip): {e}")


if __name__ == "__main__":
    main()
