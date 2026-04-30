from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunMetrics:
    """Collected metrics from a single benchmark run."""

    # Tier 1: wall-clock
    total_wall_sec: float = 0.0
    inference_wall_sec: float = 0.0
    postproc_wall_sec: float = 0.0
    throughput_sps: float = 0.0
    latency_ms_per_sample: float = 0.0
    batches_per_sec: float = 0.0

    # Tier 2: GPU hardware
    gpu_mem_peak_mb: float = 0.0
    gpu_mem_reserved_mb: float = 0.0
    gpu_mem_model_mb: float = 0.0
    gpu_power_avg_w: float = 0.0
    energy_joules: float = 0.0
    energy_per_sample_mj: float = 0.0
    cuda_kernel_time_ms: float = 0.0
    host_overhead_ms: float = 0.0

    # Tier 3: model-specific
    mfu_pct: float = 0.0
    arithmetic_intensity: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "total_wall_sec": self.total_wall_sec,
            "inference_wall_sec": self.inference_wall_sec,
            "postproc_wall_sec": self.postproc_wall_sec,
            "throughput_sps": self.throughput_sps,
            "latency_ms_per_sample": self.latency_ms_per_sample,
            "batches_per_sec": self.batches_per_sec,
            "gpu_mem_peak_mb": self.gpu_mem_peak_mb,
            "gpu_mem_reserved_mb": self.gpu_mem_reserved_mb,
            "gpu_mem_model_mb": self.gpu_mem_model_mb,
            "gpu_power_avg_w": self.gpu_power_avg_w,
            "energy_joules": self.energy_joules,
            "energy_per_sample_mj": self.energy_per_sample_mj,
            "cuda_kernel_time_ms": self.cuda_kernel_time_ms,
            "host_overhead_ms": self.host_overhead_ms,
            "mfu_pct": self.mfu_pct,
            "arithmetic_intensity": self.arithmetic_intensity,
        }


class _NvidiaSmiPoller:
    """Background thread polling nvidia-smi for power and utilization."""

    def __init__(self, interval: float = 0.1):
        self._interval = interval
        self._running = False
        self._thread: threading.Thread | None = None
        self._power_samples: list[float] = []
        self._util_samples: list[float] = []

    def start(self):
        self._power_samples.clear()
        self._util_samples.clear()
        self._running = True
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[float, float]:
        """Stop polling and return (avg_power_w, avg_util_pct)."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        avg_power = sum(self._power_samples) / max(len(self._power_samples), 1)
        avg_util = sum(self._util_samples) / max(len(self._util_samples), 1)
        return avg_power, avg_util

    def _poll(self):
        while self._running:
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=power.draw,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        parts = line.split(",")
                        if len(parts) >= 2:
                            self._power_samples.append(float(parts[0].strip()))
                            self._util_samples.append(float(parts[1].strip()))
            except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
                pass
            time.sleep(self._interval)


@dataclass
class MetricsCollector:
    """Three-tier metrics collection for benchmark runs."""

    _total_start: float = 0.0
    _infer_start: float = 0.0
    _infer_end: float = 0.0
    _postproc_start: float = 0.0
    _cuda_start: Any = None
    _cuda_end: Any = None
    _model_mem_bytes: float = 0.0
    _poller: _NvidiaSmiPoller = field(default_factory=_NvidiaSmiPoller)
    _has_cuda: bool = False

    def start_run(self):
        """Call before model loading."""
        import torch

        self._has_cuda = torch.cuda.is_available()
        self._total_start = time.perf_counter()
        if self._has_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def record_model_loaded(self):
        """Call after model is loaded, before inference."""
        import torch

        if self._has_cuda:
            torch.cuda.synchronize()
            self._model_mem_bytes = torch.cuda.max_memory_allocated()

    def start_inference(self):
        """Call before the inference loop."""
        import torch

        self._infer_start = time.perf_counter()
        if self._has_cuda:
            torch.cuda.reset_peak_memory_stats()
            self._cuda_start = torch.cuda.Event(enable_timing=True)
            self._cuda_end = torch.cuda.Event(enable_timing=True)
            self._cuda_start.record()
        self._poller.start()

    def end_inference(self):
        """Call after the inference loop."""
        import torch

        self._infer_end = time.perf_counter()
        if self._has_cuda:
            self._cuda_end.record()
            torch.cuda.synchronize()

    def start_postprocessing(self):
        self._postproc_start = time.perf_counter()

    def finalize(self, n_samples: int, n_batches: int) -> RunMetrics:
        """Compute all metrics and return."""
        import torch

        total_end = time.perf_counter()
        avg_power, _ = self._poller.stop()

        infer_wall = self._infer_end - self._infer_start
        postproc_wall = total_end - self._postproc_start if self._postproc_start else 0.0
        total_wall = total_end - self._total_start

        throughput = n_samples / max(infer_wall, 1e-9)
        latency = 1000.0 * infer_wall / max(n_samples, 1)

        cuda_kernel_ms = 0.0
        peak_mem = 0.0
        reserved_mem = 0.0
        if self._has_cuda and self._cuda_start is not None:
            cuda_kernel_ms = self._cuda_start.elapsed_time(self._cuda_end)
            peak_mem = torch.cuda.max_memory_allocated() / 1e6
            reserved_mem = torch.cuda.max_memory_reserved() / 1e6

        energy = avg_power * infer_wall
        energy_per_sample = (energy / max(n_samples, 1)) * 1000.0

        return RunMetrics(
            total_wall_sec=total_wall,
            inference_wall_sec=infer_wall,
            postproc_wall_sec=postproc_wall,
            throughput_sps=throughput,
            latency_ms_per_sample=latency,
            batches_per_sec=n_batches / max(infer_wall, 1e-9),
            gpu_mem_peak_mb=peak_mem,
            gpu_mem_reserved_mb=reserved_mem,
            gpu_mem_model_mb=self._model_mem_bytes / 1e6,
            gpu_power_avg_w=avg_power,
            energy_joules=energy,
            energy_per_sample_mj=energy_per_sample,
            cuda_kernel_time_ms=cuda_kernel_ms,
            host_overhead_ms=max(infer_wall * 1000.0 - cuda_kernel_ms, 0.0),
        )

    @staticmethod
    def compute_mfu(
        n_params: int,
        n_tokens: int,
        wall_sec: float,
        peak_tflops: float,
    ) -> float:
        """Compute Model FLOPs Utilization percentage.

        MFU = (2 * n_params * n_tokens) / (peak_FLOPS * wall_sec) * 100
        """
        if wall_sec <= 0 or peak_tflops <= 0:
            return 0.0
        flops = 2.0 * n_params * n_tokens
        return (flops / (peak_tflops * 1e12 * wall_sec)) * 100.0

    @staticmethod
    def compute_arithmetic_intensity(bytes_per_param: int = 2) -> float:
        """Compute operational intensity for decode: OI = 2 / bytes_per_param."""
        return 2.0 / bytes_per_param
