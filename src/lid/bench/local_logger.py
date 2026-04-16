from __future__ import annotations

import csv
import json
import platform
import shutil
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lid.bench.configs import ExperimentGrid


class LocalResultsLogger:
    """Manages per-step local filesystem output for benchmark runs.

    Creates a timestamped directory per grid execution:
        experiments/{grid_name}/{YYYYMMDD_HHMMSS}/
            benchmark_results.csv
            config.yaml
            platform.json
            REPORT.md

    Also appends results to a cumulative experiments/all_results.csv.
    """

    def __init__(self, grid_name: str, base_dir: str = "experiments"):
        self.grid_name = grid_name
        self.base_dir = Path(base_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.step_dir = self.base_dir / grid_name / self.timestamp
        self._wandb_urls: dict[str, str] = {}
        self._wandb_project_url: str = ""

    def setup(self) -> Path:
        self.step_dir.mkdir(parents=True, exist_ok=True)
        return self.step_dir

    def record_wandb_url(self, run_name: str, url: str):
        self._wandb_urls[run_name] = url

    def set_wandb_project_url(self, url: str):
        if url and not self._wandb_project_url:
            self._wandb_project_url = url

    def save_config(self, config_path: str):
        src = Path(config_path)
        if src.exists():
            shutil.copy2(src, self.step_dir / "config.yaml")

    def save_platform_info(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "timestamp": self.timestamp,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "python_version": platform.python_version(),
            "os": f"{platform.system()} {platform.release()}",
            "machine": platform.machine(),
            "command": " ".join(sys.argv),
        }

        try:
            import torch

            info["pytorch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
                info["cuda_version"] = torch.version.cuda or "N/A"
        except ImportError:
            pass

        try:
            import bitsandbytes

            info["bitsandbytes_version"] = bitsandbytes.__version__
        except (ImportError, AttributeError):
            pass

        try:
            import flash_attn

            info["flash_attn_version"] = flash_attn.__version__
        except (ImportError, AttributeError):
            pass

        with open(self.step_dir / "platform.json", "w") as f:
            json.dump(info, f, indent=2)

        return info

    def save_results_csv(self, results: list[dict]):
        successful = [r for r in results if "error" not in r]
        if not successful:
            return

        keys = list(successful[0].keys())

        step_csv = self.step_dir / "benchmark_results.csv"
        with open(step_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(successful)

        all_csv = self.base_dir / "all_results.csv"
        write_header = not all_csv.exists()
        with open(all_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            if write_header:
                writer.writeheader()
            writer.writerows(successful)

    def _update_latest_symlink(self):
        latest = self.base_dir / self.grid_name / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(self.timestamp)

    def generate_report(
        self,
        results: list[dict],
        grid: ExperimentGrid,
        config_path: str = "",
    ) -> Path:
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        total = len(results)

        platform_info: dict[str, Any] = {}
        platform_path = self.step_dir / "platform.json"
        if platform_path.exists():
            with open(platform_path) as f:
                platform_info = json.load(f)

        lines: list[str] = []

        # -- Header --
        lines.append(f"# Benchmark Report: {self.grid_name}\n")
        lines.append(f"> **Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if config_path:
            lines.append(f"> **Config:** `{config_path}`")
        lines.append(f"> **Command:** `{platform_info.get('command', ' '.join(sys.argv))}`")
        wb_url = self._wandb_project_url or f"https://wandb.ai/_/{grid.wandb_project}"
        lines.append(f"> **W&B Project:** [{grid.wandb_project}]({wb_url})")
        if platform_info.get("gpu_name"):
            lines.append(
                f"> **GPU:** {platform_info['gpu_name']}"
                f" ({platform_info.get('gpu_memory_gb', '?')} GB)"
            )
        if platform_info.get("pytorch_version"):
            lines.append(f"> **PyTorch:** {platform_info['pytorch_version']}")
        lines.append(f"> **Model:** {grid.fixed.get('model', 'N/A')}")
        lines.append(f"> **Dataset:** {grid.dataset} ({grid.n_samples:,} samples)")
        lines.append(f"> **Completed:** {len(successful)}/{total} runs")
        lines.append("")
        lines.append("---\n")

        # -- Grid Configuration --
        lines.append("## Grid Configuration\n")
        for k, v in grid.grid.items():
            lines.append(f"- **{k}:** {v}")
        lines.append(f"- **Repeats:** {grid.repeat}")
        lines.append(f"- **Warmup batches:** {grid.warmup_batches}")
        lines.append(f"- **Seed:** {grid.seed}")
        lines.append("")
        lines.append("---\n")

        # -- Summary table (mean of repeats) --
        if successful:
            lines.append("## Summary (mean of repeats)\n")

            groups: dict[tuple, list[dict]] = defaultdict(list)
            for r in successful:
                key = (r["strategy"], r["dtype"], r["batch_size"], r["max_length"])
                groups[key].append(r)

            lines.append(
                "| strategy | dtype | bs | ml | sps (mean) | sps (std) "
                "| mem_mb | acc | energy_mj | wall_sec |"
            )
            lines.append(
                "|----------|-------|----|----|------------|-----------|"
                "--------|------|-----------|----------|"
            )

            best_sps = 0.0
            best_key: tuple | None = None
            for key in sorted(groups.keys()):
                runs = groups[key]
                sps_vals = [r["throughput_sps"] for r in runs]
                mean_sps = statistics.mean(sps_vals)
                std_sps = statistics.stdev(sps_vals) if len(sps_vals) > 1 else 0.0
                mean_mem = statistics.mean(r["gpu_mem_peak_mb"] for r in runs)
                mean_acc = statistics.mean(r["accuracy_last_layer"] for r in runs)
                mean_energy = statistics.mean(r["energy_per_sample_mj"] for r in runs)
                mean_wall = statistics.mean(r["wall_sec"] for r in runs)

                if mean_sps > best_sps:
                    best_sps = mean_sps
                    best_key = key

                strat, dtype, bs, ml = key
                lines.append(
                    f"| {strat} | {dtype} | {bs} | {ml} "
                    f"| {mean_sps:.1f} | {std_sps:.1f} "
                    f"| {mean_mem:,.0f} | {mean_acc:.3f} "
                    f"| {mean_energy:,.0f} | {mean_wall:.1f} |"
                )

            if best_key:
                lines.append(
                    f"\n**Best config:** {best_key[0]} / {best_key[1]} "
                    f"/ bs={best_key[2]} / ml={best_key[3]} "
                    f"-> {best_sps:.1f} sps\n"
                )
            lines.append("---\n")

        # -- Individual runs --
        if successful:
            lines.append("## Individual Runs\n")
            lines.append(
                "| # | run_name | sps | mem_mb | acc "
                "| wall_sec | energy_mj | W&B |"
            )
            lines.append(
                "|---|----------|-----|--------|------"
                "|----------|-----------|-----|"
            )

            for i, r in enumerate(successful, 1):
                run_name = (
                    f"{r['strategy']}_{r['dtype']}"
                    f"_bs{r['batch_size']}_ml{r['max_length']}"
                    f"_r{r['repeat_idx']}"
                )
                wandb_url = self._wandb_urls.get(run_name, "")
                wandb_link = f"[link]({wandb_url})" if wandb_url else "--"
                lines.append(
                    f"| {i} | {run_name} "
                    f"| {r['throughput_sps']:.1f} "
                    f"| {r['gpu_mem_peak_mb']:,.0f} "
                    f"| {r['accuracy_last_layer']:.3f} "
                    f"| {r['wall_sec']:.1f} "
                    f"| {r['energy_per_sample_mj']:,.0f} "
                    f"| {wandb_link} |"
                )

            lines.append("")
            lines.append("---\n")

        # -- Errors --
        if failed:
            lines.append("## Errors\n")
            for r in failed:
                lines.append(
                    f"- **{r.get('strategy', '?')}**: {r.get('error', 'Unknown error')}"
                )
            lines.append("")
        else:
            lines.append(f"## Errors\n\nNone ({len(successful)}/{total} succeeded)\n")

        lines.append("---\n")
        lines.append(
            f"*Generated by lid-bench on"
            f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
        )

        report_path = self.step_dir / "REPORT.md"
        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        return report_path

    def finalize(
        self,
        results: list[dict],
        grid: ExperimentGrid,
        config_path: str = "",
    ) -> Path:
        """Save all local artifacts and generate the report."""
        self.setup()
        if config_path:
            self.save_config(config_path)
        self.save_platform_info()
        self.save_results_csv(results)
        report_path = self.generate_report(results, grid, config_path)
        self._update_latest_symlink()

        print(f"\nLocal results saved to: {self.step_dir}/")
        print(f"  benchmark_results.csv  (step results)")
        print(f"  config.yaml            (experiment config)")
        print(f"  platform.json          (system info)")
        print(f"  REPORT.md              (auto-generated report)")
        print(f"Cumulative results appended to: {self.base_dir / 'all_results.csv'}")

        return self.step_dir
