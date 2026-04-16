from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lid.bench.configs import RunConfig
    from lid.bench.metrics import RunMetrics


class WandbBenchLogger:
    """Wrapper for W&B logging in the benchmark framework."""

    def __init__(self, project: str, group: str, enabled: bool = True):
        self._project = project
        self._group = group
        self._enabled = enabled
        self._run = None

    def init_run(self, config: RunConfig) -> None:
        """Initialize a new W&B run for a benchmark config."""
        if not self._enabled:
            return

        import wandb

        self._run = wandb.init(
            project=self._project,
            group=self._group,
            name=config.run_name,
            config=config.to_dict(),
            tags=[config.strategy, config.dtype, f"bs{config.batch_size}"],
            reinit=True,
        )

        self._run.define_metric("batch_idx")
        self._run.define_metric("batch/*", step_metric="batch_idx")
        self._run.define_metric("throughput_sps", summary="max")
        self._run.define_metric("accuracy_last_layer", summary="max")
        self._run.define_metric("energy_per_sample_mj", summary="min")
        self._run.define_metric("gpu_mem_peak_mb", summary="min")

    def log_batch(self, batch_idx: int, metrics: dict[str, float]) -> None:
        """Log per-batch streaming metrics."""
        if not self._enabled or self._run is None:
            return
        self._run.log({"batch_idx": batch_idx, **{f"batch/{k}": v for k, v in metrics.items()}})

    def log_run_metrics(self, metrics: RunMetrics) -> None:
        """Log aggregate run metrics to summary."""
        if not self._enabled or self._run is None:
            return
        self._run.summary.update(metrics.to_dict())

    def log_layer_accuracy(
        self,
        layer_accs: list[float],
        layer_probs: list[float],
        strategy: str,
    ) -> None:
        """Log per-layer accuracy as a W&B Table with line charts."""
        if not self._enabled or self._run is None:
            return

        import wandb

        table = wandb.Table(columns=["layer_idx", "accuracy", "avg_correct_prob"])
        for i, (acc, prob) in enumerate(zip(layer_accs, layer_probs, strict=False)):
            table.add_data(i + 1, acc, prob)

        self._run.log({
            "layer_accuracy_table": table,
            "layer_accuracy_curve": wandb.plot.line(
                table, "layer_idx", "accuracy",
                title=f"Layer Accuracy: {strategy}",
            ),
            "layer_prob_curve": wandb.plot.line(
                table, "layer_idx", "avg_correct_prob",
                title=f"Correct-Class Prob: {strategy}",
            ),
        })

    def log_summary_table(self, results: list[dict[str, Any]]) -> None:
        """Log the full benchmark summary table."""
        if not self._enabled or self._run is None:
            return

        import wandb

        columns = [
            "strategy", "dtype", "batch_size", "max_length",
            "throughput_sps", "latency_ms", "gpu_mem_peak_mb",
            "energy_per_sample_mj", "accuracy_last_layer",
            "wall_sec", "mfu_pct",
        ]
        table = wandb.Table(columns=columns)
        for r in results:
            table.add_data(*[r.get(c, 0) for c in columns])
        self._run.log({"benchmark_summary": table})

    def log_profiler_trace(self, trace_path: str, description: str = "") -> None:
        """Log a torch.profiler Chrome trace as a W&B Artifact."""
        if not self._enabled or self._run is None:
            return

        import wandb

        artifact = wandb.Artifact(
            name=f"profiler-trace-{self._run.id}",
            type="profiler-trace",
            description=description,
        )
        artifact.add_file(trace_path)
        self._run.log_artifact(artifact)

    def log_config_artifact(self, config_path: str) -> None:
        """Log the grid config YAML as an artifact."""
        if not self._enabled or self._run is None:
            return

        import wandb

        artifact = wandb.Artifact(
            name=f"bench-config-{self._group}",
            type="benchmark-config",
        )
        artifact.add_file(config_path)
        self._run.log_artifact(artifact)

    def log_results_artifact(self, csv_path: str) -> None:
        """Log the results CSV as an artifact."""
        if not self._enabled or self._run is None:
            return

        import wandb

        artifact = wandb.Artifact(
            name=f"bench-results-{self._group}",
            type="benchmark-results",
        )
        artifact.add_file(csv_path)
        self._run.log_artifact(artifact)

    @property
    def run_url(self) -> str:
        if self._run is not None:
            return self._run.url
        return ""

    @property
    def project_url(self) -> str:
        if self._run is not None:
            return f"https://wandb.ai/{self._run.entity}/{self._run.project}"
        return ""

    def finish(self) -> None:
        if not self._enabled or self._run is None:
            return
        self._run.finish()
        self._run = None
