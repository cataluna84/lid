# Contributing to LID

Thank you for your interest in contributing to the Layer-Wise LID research project. This guide will help you get set up and follow our conventions.

## Getting Started

### Prerequisites

- Python 3.11+
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
│   └── visualize.py       # Plotting and analysis
├── tests/                 # Test suite (mirrors src/)
├── configs/               # YAML experiment configs
├── docs/                  # Research drafts and documentation
│   └── draft.md           # Living research document
├── notebooks/             # Exploration notebooks (outputs stripped)
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

Save experiment outputs in `experiments/` using the pattern:

```
experiments/
  YYYY-MM-DD_experiment-name/
    config.yaml
    results.pkl
    layer_accuracy.csv
    plots/
```

### Notebooks

- Notebooks are for **exploration only**, never for production logic.
- Always strip outputs before committing (handled by `nbstripout` pre-commit hook).
- If a notebook produces a useful function, refactor it into `src/lid/`.

## Code Style

- **Formatter**: Ruff (line length 100)
- **Linter**: Ruff (pycodestyle, pyflakes, isort, bugbear, simplify, naming)
- **Type checker**: mypy (best effort; `ignore_missing_imports = true` for ML libs)
- **Imports**: sorted by ruff-isort, `lid` as first-party

## Adding a New Model

1. Ensure the model works with `AutoModelForCausalLM` / `AutoTokenizer`.
2. Add any model-specific config to `configs/`.
3. Test that `lid-infer --model your-model` runs without error.
4. Document any quirks (e.g., special tokenizer handling) in the config or docstring.

## Questions?

Reach out on the project's Discord thread or open a GitHub issue.
