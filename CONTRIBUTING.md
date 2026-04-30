# Contributing to LID

Thank you for your interest in this project! LID
(*Layer-wise Multilingual Language Identification in Compact
Foundation Models*) is a public, Apache-2.0-licensed research
artifact, and we welcome:

- **Bug fixes and reproducibility improvements** -- the most useful
  contribution surface, especially around dataset access, hardware
  variability, and dependency drift.
- **Documentation PRs** -- typo fixes, clarifications, or new
  worked examples in [`docs/RUNBOOK.md`](docs/RUNBOOK.md).
- **New optimisation strategies** that fit the
  `InferenceStrategy` registry in
  [`src/lid/bench/strategy.py`](src/lid/bench/strategy.py).
- **Small features** that are easy to review without rearchitecting
  the codebase.

Larger refactors and net-new research directions are usually a
better fit for the *successor umbrella project* (link TBA in
[`README.md`](README.md#project-status----in-transition)) and for
the forward-looking research roadmap in
[`docs/paperback.md`](docs/paperback.md).

> **Before you start writing code,** open a GitHub issue (or
> discussion) describing the change. We use GitHub Issues +
> Discussions as the public coordination channel; there is no
> private chat that contributors need to be added to.

---

## Table of contents

1. [Getting started](#getting-started)
2. [Repository orientation](#repository-orientation)
3. [Development workflow](#development-workflow)
4. [Coding standards](#coding-standards)
5. [Notebooks](#notebooks)
6. [Recipes](#recipes)
   - [How to add a new optimisation strategy](#how-to-add-a-new-optimisation-strategy)
   - [How to add a new model](#how-to-add-a-new-model)
   - [How to add a new dataset](#how-to-add-a-new-dataset)
   - [How to add a new metric](#how-to-add-a-new-metric)
   - [How to add a new notebook](#how-to-add-a-new-notebook)
   - [How to add a new CLI entrypoint](#how-to-add-a-new-cli-entrypoint)
7. [Pull request checklist](#pull-request-checklist)
8. [Security and secret hygiene](#security-and-secret-hygiene)
9. [Code of Conduct](#code-of-conduct)
10. [License grant on contributions](#license-grant-on-contributions)

---

## Getting started

### Prerequisites

- **Python 3.12** (pinned in `.python-version`).
- **uv** -- the project's package and lockfile manager. Install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Git** with `git --version >= 2.30`.
- *(Optional but recommended)* **CUDA 12.8** + a CUDA-compatible GPU
  (≥ 24 GB VRAM for development; 80 GB to reproduce
  `experiments/EXPERIMENT_LOG.md` exactly). CPU-only mode works for
  unit tests and lint, but not for training or benchmarking.

### Clone and set up

```bash
git clone https://github.com/cataluna84/lid.git
cd lid

# Copy the env template and fill in your tokens (see .env.example).
cp .env.example .env
# Edit .env:
#   HF_TOKEN=hf_...                (required for the gated 1024m/LID dataset)
#   WANDB_API_KEY=...              (optional; only for W&B logging)
#   WANDB_ENTITY=<your-entity>     (optional; defaults to your wandb login)
#   WANDB_PROJECT=lid-bench        (optional; default value)

# Install all dependencies (production + dev) and pre-commit hooks.
make dev
```

`make dev` runs:

- `uv sync --group dev` -- installs the locked dependency set,
  including `ruff`, `mypy`, `pytest`, `pre-commit`, and `nbstripout`.
- `uv run pre-commit install` -- registers the
  `ruff` / `nbstripout` / `detect-private-key` hooks so they run on
  every commit.

### Verify the setup

```bash
make lint       # ruff (style + correctness)
make format     # ruff format (auto-fix)
make typecheck  # mypy
make test       # pytest

# Or all of the above + environment sanity checks:
make verify
```

`make verify` walks [`VERIFY.md`](VERIFY.md) and exits non-zero on
the first failure. The exit-code contract is documented at the
bottom of `VERIFY.md`.

---

## Repository orientation

```
lid/
├── src/lid/                  Core Python library
│   ├── constants.py          67-language mappings, default seed/paths
│   ├── data.py               Dataset loading + prompt construction
│   ├── model.py              Model / tokenizer loading helpers
│   ├── train.py              LoRA fine-tuning pipeline
│   ├── infer.py              Layer-wise inference (single-strategy)
│   ├── visualize.py          Plot per-layer accuracy / confusion
│   ├── recommend.py          Pick best config from results CSV / W&B
│   ├── report.py             Comparison report from W&B `lid-bench`
│   ├── upload.py             Backfill local results to W&B
│   └── bench/                Optimisation + benchmarking framework
│       ├── configs.py        RunConfig / ExperimentGrid dataclasses
│       ├── strategy.py       InferenceStrategy ABC + global registry
│       ├── strategies/       9 registered strategy implementations
│       ├── metrics.py        3-tier metrics collector (wall, GPU, MFU)
│       ├── wandb_logger.py   W&B integration wrapper
│       ├── local_logger.py   Local filesystem results logger
│       └── runner.py         Grid expansion + run orchestration
├── tests/                    Pytest suite (mirrors src/lid/)
├── configs/                  YAML grid configs for `lid-bench`
├── notebooks/                5 exploratory notebooks (see Notebooks below)
├── docs/
│   ├── RUNBOOK.md            Step-by-step copy-pasteable commands
│   ├── proposal_original.md  Genesis proposal (source)
│   ├── project_proposal.md   Extended proposal + publishability
│   ├── paperback.md          Forward-looking research-direction document
│   └── optimization_spec.md  Optimisation strategy spec
├── experiments/
│   ├── all_results.csv       Append-only cumulative results
│   ├── EXPERIMENT_LOG.md     Recorded H100 80 GB run log
│   └── <step>/<timestamp>/   Per-run output directory
├── pyproject.toml            uv project config (dependencies + CLI entries)
├── Makefile                  Common commands
├── .pre-commit-config.yaml   ruff + nbstripout + detect-private-key
├── LICENSE / NOTICE          Apache 2.0 + attribution
├── CITATION.cff              Software citation metadata
├── CODE_OF_CONDUCT.md        Contributor Covenant 2.1 (by reference)
├── SECURITY.md               Coordinated-disclosure policy
├── CHANGELOG.md              Keep-a-Changelog
└── .env.example              Token template
```

The repository also contains an **AI-agent external-memory system**
(`AGENTS.md`, `PLAN.md`, `VERIFY.md`, `.factory/memories.md`,
`.factory/skills/`, `notebooks/AGENTS.md`). These files document the
development process and conventions for AI coding agents (Factory
Droid in particular) that work in the repo. They are kept at the
root for transparency; a casual external contributor can ignore
them, and this `CONTRIBUTING.md` is the canonical contributor-facing
source of rules.

### Documentation lineage

When you are reading the research framing, check the lineage in
[`README.md`](README.md#project-genesis-and-document-lineage):

- **Genesis:** [`docs/proposal_original.md`](docs/proposal_original.md)
- **Extended:** [`docs/project_proposal.md`](docs/project_proposal.md)
- **Forward-looking:** [`docs/paperback.md`](docs/paperback.md)

The optimisation pipeline that this codebase implements is documented
in [`docs/optimization_spec.md`](docs/optimization_spec.md).

---

## Development workflow

### 1. Open an issue first (for non-trivial changes)

We use GitHub Issues + Discussions for coordination. Tag your issue
with the closest matching label:

- `bug` -- something is broken in the released artifact.
- `enhancement` -- a new feature, strategy, model, or dataset.
- `experiment` -- a research question that uses the repo's tooling
  (these usually belong in `docs/paperback.md` follow-ups).
- `docs` -- documentation-only.
- `question` -- ask before changing.

A trivial typo fix or a one-line bugfix can skip the issue and go
straight to a PR.

### 2. Create a branch

Branch naming follows
[Conventional Commits](https://www.conventionalcommits.org/):

| Prefix    | Use for |
|-----------|---------|
| `feat/`   | New features, strategies, or CLI entries. |
| `fix/`    | Bug fixes. |
| `docs/`   | Documentation-only PRs. |
| `refactor/` | Refactors with no behavioural change. |
| `exp/`    | Experiment branches that may not be merged. |

```bash
git checkout -b fix/oom-on-large-batch-size
```

### 3. Make changes

- All Python source belongs in `src/lid/`. Keep modules focused and
  prefer composition over monolithic functions.
- Use **absolute imports** (`from lid.bench.configs import RunConfig`),
  never relative.
- Match the surrounding style; the codebase is consistent and
  ruff-driven.

### 4. Add or update tests

Every new function in `src/lid/` should have at least one test in
`tests/` (mirroring the package layout). Tests should be fast
(no GPU dependency in the default suite) and deterministic.

### 5. Add or update dependencies

```bash
# Production dep:
uv add some-package

# Dev-only dep (lint, test, typecheck):
uv add --group dev some-dev-tool

# Optional extra (e.g. unsloth):
uv add --optional unsloth some-package
```

**Never** edit `pyproject.toml` dep lists by hand and **never**
edit `uv.lock`. Both are managed by `uv add` / `uv remove` and any
manual edit will desync.

### 6. Run the local quality gate

```bash
make lint && make format && make typecheck && make test
```

Pre-commit hooks (`ruff`, `nbstripout`, `detect-private-key`) run on
every commit. If a hook fails, the commit is blocked.

### 7. Commit and open a PR

We follow Conventional Commits:

```text
feat(bench): add gradient-checkpointing strategy
fix(infer): handle short prompts < 5 tokens
docs(readme): clarify HF token requirement
exp: qwen3.5-1b sweep on flores-200 (do not merge)
refactor(metrics): extract MFU calculation
```

When you push the branch, open a PR against `main`. The PR template
in `.github/PULL_REQUEST_TEMPLATE.md` will guide the description.

---

## Coding standards

The full standard is enforced by `ruff` (configuration in
`pyproject.toml`); these are the highlights to know before you write
code.

### Tooling

| Tool | Role | Configuration |
|------|------|---------------|
| **ruff** | Lint + format | `pyproject.toml` `[tool.ruff]`. Line length 100. |
| **mypy** | Type-check | `pyproject.toml` `[tool.mypy]`. Python 3.12, `warn_return_any = true`. |
| **pytest** | Test runner | `pyproject.toml` `[tool.pytest.ini_options]`. |
| **pre-commit** | Hook orchestrator | `.pre-commit-config.yaml`. |

### Type hints

- **Required** on every public function signature.
- Use Python 3.12 native generics (`list[int]`, `dict[str, float]`,
  `tuple[int, ...]`), not `typing.List` / `Dict` / `Tuple`.
- Use `X | None`, not `Optional[X]`.
- Add `from __future__ import annotations` at the top of every
  module.
- Place type-only imports inside `TYPE_CHECKING` to satisfy the
  ruff `TCH` rules:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lid.bench.configs import RunConfig
```

### Naming

| Element | Convention | Example |
|---|---|---|
| Functions, methods, vars | `snake_case` | `compute_lid_score` |
| Classes | `PascalCase` | `LocalResultsLogger` |
| Module-level constants | `UPPER_SNAKE_CASE` | `DEFAULT_SEED = 1024` |
| Modules / packages | `snake_case` | `local_logger.py` |
| Private | leading `_` | `_parse_row` |

Single-letter names `l`, `O`, `I` are banned (ruff's `E741`).

### Docstrings

Google-style on every public symbol:

```python
def run_benchmark(
    config_path: Path,
    strategies: list[str],
    *,
    repeats: int = 3,
) -> dict[str, float]:
    """Execute a benchmark grid from a YAML config.

    Args:
        config_path: Path to the YAML experiment configuration.
        strategies: Strategy names to evaluate.
        repeats: Repeats per configuration.

    Returns:
        Mapping of strategy name to mean throughput (samples/sec).

    Raises:
        FileNotFoundError: If `config_path` does not exist.
        ValueError: If an unknown strategy name is provided.
    """
```

### Imports

ruff's `I` (isort) rules enforce import order:

1. Standard library (`os`, `sys`, `pathlib`, ...)
2. Third-party (`torch`, `wandb`, `pandas`, ...)
3. First-party (`lid`, `lid.bench`, ...)

One blank line between groups; never edit imports manually if
`ruff format` disagrees.

### Numerical / GPU code

- Always set the project default seed `1024` (in
  `lid.constants.DEFAULT_SEED`) at the top of every entrypoint.
- Always read `HF_TOKEN` / `WANDB_API_KEY` via `python-dotenv`,
  never via `os.environ` alone (let `.env` win) and never via Colab
  `userdata`.
- Route `torch` and `torchvision` through the `pytorch-cu128` index
  (configured in `[tool.uv.sources]`). PyPI wheels will silently ABI-
  mismatch.

---

## Notebooks

The five notebooks under [`notebooks/`](notebooks/) are exploratory.
They are **not** the canonical implementation of anything; if a
notebook function proves useful, refactor it into `src/lid/`.

| Notebook | Approach | Extra group | Detailed description |
|---|---|---|---|
| `LID_Regex.ipynb` | Unicode-block heuristic | none | [README → Notebooks](README.md#notebooks) |
| `LID_Unicode_Blocks_Classifier.ipynb` | Unicode-block + classifier | none | [README → Notebooks](README.md#notebooks) |
| `LID_Ngrams_Classifier.ipynb` | Char n-grams (1--5) | none | [README → Notebooks](README.md#notebooks) |
| `LID_Embedding_Classifier.ipynb` | `multilingual-e5-large` + LoRA + unsloth | `unsloth` | [README → Notebooks](README.md#notebooks) |
| `LID_Inference_Vibecoded.ipynb` | Layer-wise extraction on `tiny-aya-global` | none | [README → Notebooks](README.md#notebooks) |

The embedding-classifier notebook needs the `unsloth` extras group:

```bash
uv sync --extra unsloth
```

All notebook-specific gotchas (the three required bug-fix shims,
nbstripout consequences, kernel selection) are documented in
[`notebooks/AGENTS.md`](notebooks/AGENTS.md). Read it before editing
any notebook cell.

**Notebook conventions:**

- Cell outputs are stripped on commit by `nbstripout`. Don't depend
  on them being there.
- Never `!pip install` inside a cell. Use `uv sync --extra <group>`
  and document the group in `pyproject.toml`.
- Name notebooks `LID_<Method>_<Purpose>.ipynb`.
- Cell 0 is a markdown cell pointing at the required `uv` extras
  group (or "no extras -- base `make dev` is enough"). Cell 1 is
  imports + any compat shims. Cell 2 is `.env` loading + auth.

---

## Recipes

These are short, copy-pasteable patterns for the most common kinds
of contribution.

### How to add a new optimisation strategy

1. Implement the `InferenceStrategy` ABC in
   `src/lid/bench/strategies/<your_strategy>.py`. The minimal API is
   `setup(...)` and `extract_layer_probs(...)`. Look at
   `src/lid/bench/strategies/vectorized.py` for a clean reference
   implementation.
2. Register the strategy in
   `src/lid/bench/strategies/__init__.py` so `StrategyRegistry`
   picks it up:
   ```python
   from .your_strategy import YourStrategy
   register("your_strategy", YourStrategy)
   ```
3. Add a YAML config in `configs/` showing one viable grid
   (e.g. `configs/step_your.yaml`).
4. Add a unit test in `tests/test_strategies.py` that constructs a
   `RunConfig` with `strategy="your_strategy"` and verifies a smoke
   forward pass on a tiny model.
5. Update the **Strategy reference** table in
   [`README.md`](README.md#strategy-reference) and add an
   exclusion rule to `configs/bench_grid.yaml` if your strategy is
   incompatible with any combination of dtype / batch-size /
   max-length.
6. Run `make verify` and confirm `make bench-quick` still passes.

### How to add a new model

1. Confirm the model loads via `AutoModelForCausalLM.from_pretrained`
   and `AutoTokenizer.from_pretrained` from the HuggingFace Hub.
2. Add it to `src/lid/constants.py` if you need to hard-code
   tokenizer-specific quirks (most of the time, no entry is needed).
3. Add a `configs/<model>_*.yaml` for the smoke run.
4. Run `lid-infer --model <hf-id> --sample-frac 0.01 --no-wandb` to
   confirm a forward pass.
5. Document any model-specific quirks in the YAML or the strategy
   docstring (e.g. `gemma-3` requires `eager` attention; *Cohere2*
   models have 37 hidden states, etc.).

### How to add a new dataset

1. Add a loader in `src/lid/data.py` -- ideally something like:
   ```python
   def load_<your_dataset>(split: str = "train") -> datasets.Dataset:
       ...
   ```
2. Pick a stable column schema. The bench framework expects
   `text: str` and `iso-693-3: str` (the gold label). Convert.
3. Add a unit test that verifies the loader returns the right
   schema on a 10-row slice.
4. Document the dataset in the Reproducibility caveats of
   [`README.md`](README.md#reproducibility-caveats), especially if
   it is gated or has unusual licensing.

### How to add a new metric

1. Add the metric calculation to `src/lid/bench/metrics.py` or a
   new `src/lid/eval/<your_metric>.py` module.
2. Pick the right tier (see
   [`docs/optimization_spec.md`](docs/optimization_spec.md) §5):
   Tier 1 (always logged), Tier 2 (GPU-specific, always logged),
   Tier 3 (model-specific, flag-gated).
3. Wire it into `wandb_logger.py` so it shows up in `run.summary`.
4. Add a unit test against a known input (e.g. a synthetic
   confusion matrix with hand-computed expected output).

### How to add a new notebook

1. Run `cp notebooks/LID_Ngrams_Classifier.ipynb notebooks/LID_<New>.ipynb`
   to copy the dependency-loader and `.env` cells.
2. Edit the *first markdown cell* to point at the required `uv`
   extras group.
3. If you need a new dependency group, add it via `uv add --optional
   <group>` and update
   [`notebooks/AGENTS.md`](notebooks/AGENTS.md) Dependencies section.
4. Update the **Notebooks** table in
   [`README.md`](README.md#notebooks) and the table in this file.

### How to add a new CLI entrypoint

1. Implement the `main()` function in
   `src/lid/<your_command>.py`. Use `argparse` (matches the rest of
   the codebase) and pull defaults from `lid.constants`.
2. Register the entry in `pyproject.toml`:
   ```toml
   [project.scripts]
   lid-yourcommand = "lid.<your_command>:main"
   ```
3. Add a smoke test in `tests/test_<your_command>.py` using
   `subprocess` to confirm the CLI is wired correctly.
4. Update the **Installed CLI entrypoints** table in
   [`README.md`](README.md#installed-cli-entrypoints).
5. If the command is referenced from `docs/RUNBOOK.md`, add an
   appropriate Step.

---

## Pull request checklist

Before clicking "Ready for review":

- [ ] Branch named per Conventional Commits (`feat/`, `fix/`,
      `docs/`, `refactor/`, `exp/`).
- [ ] Commits use Conventional-Commits-style messages.
- [ ] `make lint && make format && make typecheck && make test`
      all pass.
- [ ] `make verify` exits zero (or the failure is documented and
      explained in the PR body).
- [ ] No secrets committed (`git diff --cached` reviewed manually
      even though `detect-private-key` runs in pre-commit).
- [ ] Tests added or updated for any new behaviour.
- [ ] [`README.md`](README.md), [`CHANGELOG.md`](CHANGELOG.md), and
      any affected `docs/*.md` updated when the change is
      user-visible.
- [ ] If a notebook is touched, [`notebooks/AGENTS.md`](notebooks/AGENTS.md)
      is updated when applicable.
- [ ] If a config is added, the YAML is committed under `configs/`
      and referenced from `docs/RUNBOOK.md` if appropriate.

The PR template (`.github/PULL_REQUEST_TEMPLATE.md`) reproduces this
list as checkboxes.

---

## Security and secret hygiene

- **Never commit API keys, tokens, or credentials.** The
  `detect-private-key` pre-commit hook catches many leaks but not
  all -- always review `git diff --cached` before every commit.
- **Always `.env`-load secrets** (`HF_TOKEN`, `WANDB_API_KEY`,
  `WANDB_ENTITY`, `WANDB_PROJECT`) via `python-dotenv`. The repo's
  `.gitignore` excludes `.env*`.
- **If a secret is exposed accidentally**, rotate the credential
  immediately, then follow [`SECURITY.md`](SECURITY.md) (GitHub
  private advisory or email).
- **Notebook outputs can leak secrets** (e.g. a printed
  `os.environ`). The `nbstripout` hook strips outputs on commit, but
  the safest pattern is to never print credentials, even in a cell.

For coordinated disclosure of a security vulnerability, see
[`SECURITY.md`](SECURITY.md).

---

## Code of Conduct

This project adopts the
[Contributor Covenant 2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
The full text and reporting channels live in
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

---

## License grant on contributions

LID is released under the
[Apache License 2.0](LICENSE). By submitting a pull request you
agree that your contribution is licensed under the same terms (per
Apache 2.0 §5); we do not require a separate Contributor License
Agreement.

If you import third-party code, ensure its license is compatible
with Apache 2.0 (BSD, MIT, ISC, Apache, MPL with care -- not GPL or
AGPL) and that you keep the upstream `NOTICE` text intact in our
[`NOTICE`](NOTICE) file.

---

## Questions?

Open a [GitHub Discussion](https://github.com/cataluna84/lid/discussions)
for general questions, or a
[GitHub Issue](https://github.com/cataluna84/lid/issues/new/choose)
for actionable bug reports / feature requests. The maintainer's
public email is in [`CITATION.cff`](CITATION.cff) for project-related
correspondence that does not fit into Issues / Discussions.
