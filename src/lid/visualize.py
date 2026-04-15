import argparse
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np


FONT_SIZE = 12


def plot_global_avg(df, output_dir: str):
    avg_probs = np.mean(np.stack(df["PROB"].values), axis=0)
    layers = np.arange(1, len(avg_probs) + 1)

    plt.figure(figsize=(12, 6))
    plt.bar(layers, avg_probs, color="#7a17cc")
    plt.xlabel("Layer Number", fontsize=FONT_SIZE)
    plt.ylabel("Average Probability", fontsize=FONT_SIZE)
    plt.title("Average Correct-Language Probability per Layer", fontsize=FONT_SIZE)
    plt.xticks(layers, fontsize=FONT_SIZE, rotation=90)
    plt.yticks(fontsize=FONT_SIZE)
    plt.ylim(0, max(avg_probs) * 1.05)
    plt.tight_layout()

    path = os.path.join(output_dir, "layer_avg_probs.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_per_language(df, output_dir: str):
    lang_dir = os.path.join(output_dir, "lang_plots")
    os.makedirs(lang_dir, exist_ok=True)

    for lang in sorted(df["ISO-693-3"].unique()):
        lang_df = df[df["ISO-693-3"] == lang]
        avg_probs = np.mean(np.stack(lang_df["PROB"].values), axis=0)
        layers = np.arange(1, len(avg_probs) + 1)

        color = f"#{random.randint(0, 0xFFFFFF):06x}"
        plt.figure(figsize=(10, 6))
        plt.bar(layers, avg_probs, color=color, edgecolor="black", linewidth=0.8)
        plt.xlabel("Layer Number", fontsize=FONT_SIZE)
        plt.ylabel("Average Probability", fontsize=FONT_SIZE)
        plt.title(
            f"Average Probability per Layer: {lang} ({len(lang_df)} samples)",
            fontsize=FONT_SIZE,
        )
        plt.xticks(layers, fontsize=FONT_SIZE, rotation=90)
        plt.yticks(fontsize=FONT_SIZE)
        plt.ylim(0, max(avg_probs) * 1.1)
        plt.tight_layout()

        path = os.path.join(lang_dir, f"layer_avg_{lang}.png")
        plt.savefig(path, dpi=150)
        plt.close()

    print(f"Saved {len(df['ISO-693-3'].unique())} per-language plots to {lang_dir}")


def plot_layer_accuracy(df, output_dir: str):
    layer_keys = sorted(
        k for k in df.iloc[0]["MAX"] if k.startswith("LAYER-")
    )
    accuracies = []
    for layer in layer_keys:
        correct = sum(
            1
            for _, row in df.iterrows()
            if row["MAX"][layer]["label"] == row["ISO-693-3"]
        )
        accuracies.append(correct / len(df))

    layers = np.arange(1, len(accuracies) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(layers, accuracies, marker="o", color="#cc1717", linewidth=2)
    plt.xlabel("Layer Number", fontsize=FONT_SIZE)
    plt.ylabel("Accuracy", fontsize=FONT_SIZE)
    plt.title("LID Accuracy per Layer", fontsize=FONT_SIZE)
    plt.xticks(layers, fontsize=FONT_SIZE, rotation=90)
    plt.yticks(fontsize=FONT_SIZE)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "layer_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Visualize LID inference results")
    p.add_argument(
        "--results",
        default="outputs/results.pkl",
        help="Path to results pickle from infer.py",
    )
    p.add_argument("--output-dir", default="outputs")
    p.add_argument(
        "--plots",
        nargs="+",
        default=["global", "per_language", "accuracy"],
        choices=["global", "per_language", "accuracy"],
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading results from {args.results}")
    with open(args.results, "rb") as f:
        df = pickle.load(f)

    if "global" in args.plots:
        plot_global_avg(df, args.output_dir)
    if "per_language" in args.plots:
        plot_per_language(df, args.output_dir)
    if "accuracy" in args.plots:
        plot_layer_accuracy(df, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
