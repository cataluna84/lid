from __future__ import annotations

import argparse
import gc
import os
import time

import torch
from dotenv import load_dotenv
from tqdm import tqdm

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# The os.environ setdefault above must run before any of these imports trigger
# torch initialization, hence the noqa: E402 across the block below.
import lid.bench.strategies  # noqa: E402, F401  -- register all strategies
from lid.bench.configs import ExperimentGrid, RunConfig  # noqa: E402
from lid.bench.local_logger import LocalResultsLogger  # noqa: E402
from lid.bench.metrics import MetricsCollector  # noqa: E402
from lid.bench.strategy import StrategyRegistry, build_token_index  # noqa: E402
from lid.bench.wandb_logger import WandbBenchLogger  # noqa: E402
from lid.constants import VALID_OPTIONS  # noqa: E402
from lid.data import load_lid_dataset  # noqa: E402


def _run_single(
    config: RunConfig,
    df,
    logger: WandbBenchLogger,
) -> dict:
    """Execute a single benchmark run and return results dict."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    collector = MetricsCollector()
    collector.start_run()

    strategy = StrategyRegistry.get(config.strategy)
    model, tokenizer = strategy.load_model(config.model, config.dtype, device)
    collector.record_model_loaded()

    token_ids, token_mask = build_token_index(tokenizer, VALID_OPTIONS, device)

    prompts = df["INSTRUCT"].tolist()[: config.n_samples]
    n_samples = len(prompts)
    n_batches = (n_samples + config.batch_size - 1) // config.batch_size

    all_probs = []
    collector.start_inference()

    for batch_idx in tqdm(range(n_batches), desc=f"  {config.run_name}"):
        start_i = batch_idx * config.batch_size
        end_i = min(start_i + config.batch_size, n_samples)
        batch_prompts = prompts[start_i:end_i]

        batch_start = time.perf_counter()
        probs = strategy.extract_layer_probs(
            model,
            tokenizer,
            batch_prompts,
            VALID_OPTIONS,
            token_ids,
            token_mask,
            config.temperature,
            config.max_length,
        )
        batch_elapsed = time.perf_counter() - batch_start

        if batch_idx >= config.warmup_batches:
            all_probs.append(probs.cpu())

        if batch_idx >= config.warmup_batches:
            logger.log_batch(
                batch_idx,
                {
                    "latency_ms": batch_elapsed * 1000,
                    "throughput_sps": len(batch_prompts) / max(batch_elapsed, 1e-9),
                    "gpu_mem_mb": (
                        torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
                    ),
                },
            )

        del probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    collector.end_inference()

    # Post-processing
    collector.start_postprocessing()
    stacked = torch.cat(all_probs, dim=1) if all_probs else torch.zeros(1, 1, len(VALID_OPTIONS))

    actual_isos = df["ISO-693-3"].tolist()[: config.n_samples]
    warmup_skip = config.warmup_batches * config.batch_size
    actual_isos = actual_isos[warmup_skip:]

    iso_to_idx = {iso: i for i, iso in enumerate(VALID_OPTIONS)}
    n_valid = min(stacked.shape[1], len(actual_isos))

    layer_accs = []
    layer_probs_avg = []
    n_layers = stacked.shape[0]

    for layer_idx in range(n_layers):
        correct = 0
        prob_sum = 0.0
        for s in range(n_valid):
            pred_idx = int(stacked[layer_idx, s].argmax().item())
            pred_iso = VALID_OPTIONS[pred_idx]
            actual = actual_isos[s]
            if pred_iso == actual:
                correct += 1
            gt_idx = iso_to_idx.get(actual, 0)
            prob_sum += stacked[layer_idx, s, gt_idx].item()
        layer_accs.append(correct / max(n_valid, 1))
        layer_probs_avg.append(prob_sum / max(n_valid, 1))

    run_metrics = collector.finalize(n_samples, n_batches)

    logger.log_run_metrics(run_metrics)
    logger.log_layer_accuracy(layer_accs, layer_probs_avg, config.strategy)

    result = {
        "strategy": config.strategy,
        "dtype": config.dtype,
        "batch_size": config.batch_size,
        "max_length": config.max_length,
        "repeat_idx": config.repeat_idx,
        "accuracy_last_layer": layer_accs[-1] if layer_accs else 0.0,
        "wall_sec": run_metrics.total_wall_sec,
        "throughput_sps": run_metrics.throughput_sps,
        "latency_ms": run_metrics.latency_ms_per_sample,
        "gpu_mem_peak_mb": run_metrics.gpu_mem_peak_mb,
        "energy_per_sample_mj": run_metrics.energy_per_sample_mj,
        "mfu_pct": run_metrics.mfu_pct,
        **run_metrics.to_dict(),
    }

    # Cleanup
    del model, tokenizer, stacked, all_probs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return result


class BenchmarkRunner:
    """Grid expansion, sequential execution, and results management."""

    def __init__(
        self,
        grid: ExperimentGrid,
        wandb_enabled: bool = True,
        config_path: str = "",
    ):
        self.grid = grid
        self.wandb_enabled = wandb_enabled
        self.config_path = config_path

    def run(self) -> list[dict]:
        configs = self.grid.expand()
        print(f"Expanded grid: {len(configs)} runs")
        print(f"Available strategies: {StrategyRegistry.available()}")

        df = load_lid_dataset(
            dataset_name=self.grid.dataset,
            file_path=self.grid.dataset_file or None,
            sample_frac=1.0,
        )
        print(f"Dataset loaded: {len(df)} samples")

        logger = WandbBenchLogger(
            project=self.grid.wandb_project,
            group=self.grid.name,
            enabled=self.wandb_enabled,
        )

        local_logger = LocalResultsLogger(
            grid_name=self.grid.name,
            base_dir="experiments",
        )
        local_logger.setup()

        results: list[dict] = []
        for i, config in enumerate(configs):
            print(f"\n[{i + 1}/{len(configs)}] {config.run_name}")
            logger.init_run(config)

            if logger.run_url:
                local_logger.record_wandb_url(config.run_name, logger.run_url)
                local_logger.set_wandb_project_url(logger.project_url)

            try:
                result = _run_single(config, df, logger)
                results.append(result)
            except Exception as e:
                print(f"  FAILED: {e}")
                results.append({"strategy": config.strategy, "error": str(e)})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            finally:
                logger.finish()

        # Save local results and generate report
        local_logger.finalize(results, self.grid, self.config_path)

        # Print summary
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        header = f"{'strategy':<25} {'dtype':<6} {'bs':<4} {'sps':>8} {'mem_mb':>8} {'acc':>6}"
        print(header)
        print("-" * 60)
        for r in results:
            if "error" not in r:
                print(
                    f"{r['strategy']:<25} {r['dtype']:<6} {r['batch_size']:<4} "
                    f"{r['throughput_sps']:>8.1f} {r['gpu_mem_peak_mb']:>8.0f} "
                    f"{r['accuracy_last_layer']:>6.3f}"
                )

        return results


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run LID inference benchmark grid")
    parser.add_argument("config", help="Path to benchmark grid YAML")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    grid = ExperimentGrid.from_yaml(args.config)
    runner = BenchmarkRunner(grid, wandb_enabled=not args.no_wandb, config_path=args.config)
    runner.run()


if __name__ == "__main__":
    main()
