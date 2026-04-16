# Contributing to LID

Thank you for your interest in contributing to the Layer-Wise LID research project. This guide will help you get set up and follow our conventions.

## Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Git
- (Optional) CUDA-compatible GPU for training/inference

### Setup

```bash
git clone git@github.com:cataluna84/lid.git
cd lid

# Install all dependencies (including dev tools)
make dev

# This runs:
#   uv sync --group dev
#   uv run pre-commit install
```

### Verify the setup

```bash
make lint       # Ruff linting
make format     # Auto-format code
make typecheck  # mypy static type checking
make test       # Run test suite
```

## Project Structure

```
lid/
├── src/lid/               # Core library
│   ├── __init__.py
│   ├── constants.py       # Language mappings, defaults
│   ├── data.py            # Dataset loading, prompt construction
│   ├── model.py           # Model loading, layer-wise inference
│   ├── train.py           # Fine-tuning pipeline (LoRA, quantization)
│   ├── infer.py           # Batched layer-wise inference
│   ├── visualize.py       # Plotting and analysis
│   └── bench/             # Benchmarking framework
│       ├── configs.py     # Experiment grid / run config dataclasses
│       ├── runner.py      # Benchmark orchestration
│       ├── strategy.py    # Strategy base class
│       ├── metrics.py     # Timing and accuracy metrics
│       ├── local_logger.py # Local filesystem results logger
│       └── wandb_logger.py # Weights & Biases integration
├── tests/                 # Test suite (mirrors src/)
├── configs/               # YAML experiment configs
├── docs/                  # Research drafts and documentation
│   └── project_proposal.md # Research proposal and objectives
├── notebooks/             # Exploration notebooks (outputs stripped)
│   ├── LID_Ngrams_Classifier.ipynb
│   ├── LID_Unicode_Blocks_Classifier.ipynb
│   └── LID_Embedding_Classifier.ipynb
├── experiments/           # Experiment logs and results
├── .github/workflows/     # CI/CD
├── pyproject.toml         # Project metadata and dependencies (uv)
├── Makefile               # Common commands
├── .pre-commit-config.yaml
└── CONTRIBUTING.md        # This file
```

## Development Workflow

### 1. Create a feature branch

```bash
git checkout -b feat/your-feature-name
```

Branch naming conventions:
- `feat/` -- New features or experiments
- `fix/` -- Bug fixes
- `exp/` -- Experiment branches (may not be merged)
- `docs/` -- Documentation updates

### 2. Write code

- All source code goes in `src/lid/`.
- Follow existing patterns; the codebase uses `argparse` for CLI entry points.
- Use type hints everywhere.
- Keep functions focused; prefer composition over monoliths.

### 3. Add dependencies

```bash
# Production dependency
uv add some-package

# Dev-only dependency
uv add --group dev some-dev-tool
```

Never edit `uv.lock` manually -- it is auto-generated.

### 4. Run quality checks

```bash
make lint       # Must pass
make format     # Auto-fix style
make typecheck  # Should pass (best effort for ML code)
make test       # Must pass
```

Pre-commit hooks run `ruff` and `nbstripout` automatically on every commit.

### 5. Commit and push

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add layer-wise accuracy tracking
fix: handle edge case in batched inference for short prompts
exp: qwen3.5-1b 4bit quantization sweep
docs: update draft with ORPO results
```

### 6. Open a pull request

- Target `main` branch.
- Describe what changed and why.
- Link any relevant experiment results or plots.
- CI must pass before merge.

## Experiment Conventions

### Configs

Store experiment configs as YAML in `configs/`:

```yaml
# configs/exp_tinyaya_4bit.yaml
model: CohereLabs/tiny-aya-global
dataset: 1024m/LID
sample_frac: 0.1
quantization: 4bit
lora_r: 16
lora_alpha: 32
batch_size: 16
epochs: 3
```

### Results

Each `lid-bench` run automatically creates a structured output directory via `LocalResultsLogger`:

```
experiments/
  all_results.csv                       # Cumulative results across all runs
  step3-vectorized/
    20260416_093000/
      benchmark_results.csv             # Per-run detailed results
      config.yaml                       # Frozen config snapshot
      platform.json                     # GPU, PyTorch, CUDA version info
      REPORT.md                         # Human-readable summary
    latest -> 20260416_093000/          # Symlink to most recent run
```

- Never edit `all_results.csv` by hand — it is append-only and managed by the logger.
- Use the `latest` symlink for quick access to the most recent run of a given step.

### Notebooks

- Notebooks are for **exploration only**, never for production logic.
- Outputs are stripped before commit by the `nbstripout` pre-commit hook.
- If a notebook produces a useful function, refactor it into `src/lid/`.
- Name notebooks descriptively: `LID_{Method}_{Purpose}.ipynb` (e.g., `LID_Ngrams_Classifier.ipynb`).

## Research Experiment Conventions

### Reproducibility

- **Always set random seeds.** The project default seed is `1024`.
- **Record platform info.** GPU model, PyTorch version, and CUDA version are captured automatically by `LocalResultsLogger` into `platform.json`.
- **Pin all dependencies** via `uv.lock` — never edit it manually; use `uv add` / `uv remove`.
- **Store experiment configs** as YAML files in `configs/`. Configs are frozen into each run directory so results are always traceable.

### W&B Logging

- All benchmark runs log to the W&B project `lid-bench` by default.
- Pass `--no-wandb` when you want quick local-only iteration without network overhead.
- **Never commit W&B API keys or `.env` files.** Use environment variables or a gitignored `.env` file.

### Notebook Workflow

- Notebooks live in `notebooks/` and are for rapid prototyping and visualization.
- The `nbstripout` pre-commit hook ensures cell outputs are never committed to git.
- When a notebook function proves useful, extract it into `src/lid/` with proper tests.
- Follow the naming convention `LID_{Method}_{Purpose}.ipynb`.

## Code Style

**Tooling overview:**

- **Formatter**: Ruff (line length 100)
- **Linter**: Ruff (pycodestyle, pyflakes, isort, pep8-naming, pyupgrade, bugbear, simplify, type-checking, ruff-specific)
- **Type checker**: mypy (Python 3.12, `warn_return_any = true`, `ignore_missing_imports = true` for ML libs)
- **Imports**: sorted by ruff-isort, `lid` as first-party

### PEP 8 Naming Conventions

Enforced automatically by ruff's `N` (pep8-naming) rules:

| Element | Convention | Example |
|---|---|---|
| Functions, methods, variables | `snake_case` | `compute_lid_score` |
| Classes | `PascalCase` | `LocalResultsLogger` |
| Module-level constants | `UPPER_SNAKE_CASE` | `DEFAULT_SEED = 1024` |
| Modules and packages | `snake_case` | `local_logger.py` |
| Private / internal | Leading underscore | `_parse_row` |

Never use `l`, `O`, or `I` as single-character variable names (ambiguous in many fonts).

### Type Hints

Type annotations are **required for all new code**. mypy runs in CI via `make typecheck`.

- Use Python 3.12 native generics: `list[int]`, `dict[str, float]`, `tuple[int, ...]` — not `List`, `Dict`, `Tuple` from `typing`.
- Use union syntax: `X | None` — not `Optional[X]`.
- Use `from __future__ import annotations` at the top of every module for forward-reference support and consistent behaviour.
- All function signatures must annotate **both** parameters and return type.
- Use the `TYPE_CHECKING` guard for import-only types to avoid circular imports and runtime overhead:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lid.bench.configs import RunConfig


def process(config: RunConfig, data: list[dict[str, float]]) -> float:
    """Process benchmark data and return aggregate score."""
    ...
```

### Docstrings

Use **Google-style** docstrings for all public modules, classes, and functions.

**One-liner** (simple helpers):

```python
def seed_everything(seed: int = 1024) -> None:
    """Set random seeds for reproducibility across all frameworks."""
```

**Multi-line** (complex functions):

```python
def run_benchmark(
    config_path: Path,
    strategies: list[str],
    *,
    repeats: int = 3,
) -> dict[str, float]:
    """Execute a benchmark grid from a YAML config file.

    Args:
        config_path: Path to the YAML experiment configuration.
        strategies: List of strategy names to evaluate.
        repeats: Number of repetitions per configuration.

    Returns:
        Mapping of strategy name to mean latency in seconds.

    Raises:
        FileNotFoundError: If config_path does not exist.
        ValueError: If an unknown strategy name is provided.
    """
```

### Import Order

Enforced by ruff's `I` (isort) rules. Imports are grouped in this order:

1. **Standard library** (`os`, `sys`, `pathlib`, …)
2. **Third-party** (`torch`, `wandb`, `pandas`, …)
3. **First-party** (`lid`, `lid.bench`, …)

Additional rules:

- Prefer **absolute imports**: `from lid.bench.configs import RunConfig`.
- Place type-only imports inside a `TYPE_CHECKING` block (enforced by ruff `TCH` rules).
- One blank line between each import group.

### Code Layout

- **Max line length**: 100 characters (configured in ruff; `E501` is ignored so long lines won't block CI, but aim for 100).
- **Indentation**: 4 spaces (no tabs).
- **Blank lines**: 2 blank lines before and after top-level function/class definitions; 1 blank line between methods inside a class.
- **Trailing commas**: Always use trailing commas on multi-line collections, function signatures, and argument lists — this produces cleaner diffs.
- **String quotes**: Double quotes preferred (ruff-format default).

## Security

- **Never commit API keys, tokens, or credentials** to the repository.
- Store secrets in a `.env` file (gitignored) or use environment variables.
- The `detect-private-key` pre-commit hook will catch accidental private key commits.
- Always review `git diff --cached` before every commit to confirm no sensitive data is staged.
- If you accidentally commit a secret, rotate it immediately and notify the team.

## Adding a New Model

1. Ensure the model works with `AutoModelForCausalLM` / `AutoTokenizer`.
2. Add any model-specific config to `configs/`.
3. Test that `lid-infer --model your-model` runs without error.
4. Document any quirks (e.g., special tokenizer handling) in the config or docstring.

## Questions?

Reach out on the project's Discord thread or open a GitHub issue.
