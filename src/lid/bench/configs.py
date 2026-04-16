from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class RunConfig:
    """Configuration for a single benchmark run."""

    strategy: str
    dtype: str
    batch_size: int
    max_length: int
    repeat_idx: int
    model: str
    temperature: float
    n_samples: int
    warmup_batches: int
    seed: int
    dataset: str
    dataset_file: str
    profile: bool = False

    @property
    def run_name(self) -> str:
        return (
            f"{self.strategy}_{self.dtype}_bs{self.batch_size}"
            f"_ml{self.max_length}_r{self.repeat_idx}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "repeat_idx": self.repeat_idx,
            "model": self.model,
            "temperature": self.temperature,
            "n_samples": self.n_samples,
            "warmup_batches": self.warmup_batches,
            "seed": self.seed,
            "dataset": self.dataset,
            "dataset_file": self.dataset_file,
            "profile": self.profile,
        }


@dataclass
class ExperimentGrid:
    """YAML-driven experiment grid with cartesian expansion."""

    name: str
    wandb_project: str
    dataset: str
    dataset_file: str
    n_samples: int
    warmup_batches: int
    repeat: int
    seed: int
    profile: bool
    grid: dict[str, list[Any]]
    exclude: list[dict[str, Any]] = field(default_factory=list)
    fixed: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentGrid:
        with open(path) as f:
            raw = yaml.safe_load(f)

        meta = raw["meta"]
        return cls(
            name=meta["name"],
            wandb_project=meta.get("wandb_project", "lid-bench"),
            dataset=meta["dataset"],
            dataset_file=meta.get("dataset_file", ""),
            n_samples=meta.get("n_samples", 3350),
            warmup_batches=meta.get("warmup_batches", 2),
            repeat=meta.get("repeat", 3),
            seed=meta.get("seed", 1024),
            profile=meta.get("profile", False),
            grid=raw["grid"],
            exclude=raw.get("exclude", []),
            fixed=raw.get("fixed", {}),
        )

    def _is_excluded(self, combo: dict[str, Any]) -> bool:
        return any(all(combo.get(k) == v for k, v in rule.items()) for rule in self.exclude)

    def expand(self) -> list[RunConfig]:
        keys = sorted(self.grid.keys())
        values = [self.grid[k] for k in keys]
        configs: list[RunConfig] = []

        for combo_vals in itertools.product(*values):
            combo = dict(zip(keys, combo_vals, strict=False))
            if self._is_excluded(combo):
                continue
            for r in range(self.repeat):
                configs.append(
                    RunConfig(
                        strategy=combo["strategy"],
                        dtype=combo["dtype"],
                        batch_size=combo["batch_size"],
                        max_length=combo["max_length"],
                        repeat_idx=r,
                        model=self.fixed.get("model", "CohereLabs/tiny-aya-global"),
                        temperature=self.fixed.get("temperature", 0.2),
                        n_samples=self.n_samples,
                        warmup_batches=self.warmup_batches,
                        seed=self.seed + r,
                        dataset=self.dataset,
                        dataset_file=self.dataset_file,
                        profile=self.profile,
                    )
                )

        return configs
