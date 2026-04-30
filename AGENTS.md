---
description: Layer-wise multilingual language ID research repo (67 langs, public Apache-2.0 artifact). Python 3.12 + uv + PyTorch CUDA 12.8 + unsloth.
tags: [python, uv, pytorch-cu128, transformers, unsloth, wandb, lid, notebook]
---

# AGENTS.md

This is the index file for all AI coding agents working on the LID repo.
Read this first, every session.

> LID (layer-wise language identification) is a mechanistic-interpretability
> research project for multilingual LID in compact foundation models
> (0–4B params). The repository is **public** under Apache-2.0 and has
> been released as a finished artifact -- the active research surface
> is the maintenance + reproducibility surface, not net-new
> experiments.
>
> No paper is currently in submission. The genesis proposal lives at
> `docs/proposal_original.md`, its extension at
> `docs/project_proposal.md`, and a forward-looking research-direction
> document for anyone continuing this work lives at `docs/paperback.md`.

## Memory system (external memory, per tip 10)

The repo uses a four-file external memory system. Retrieve them in this order
at the start of every session:

1. **`AGENTS.md`** (this file) — repo-wide rules, commands, safety, gotchas.
2. **`PLAN.md`** — the **active** task's checklist and definition of done.
3. **`.factory/memories.md`** — rolling episodic log of what was done,
    learned, and broken (read top 5 entries).
4. **`VERIFY.md`** — deterministic pass/fail commands that prove the repo
    works (skim section headers; run with `make verify`).

Hierarchical scope:
- `notebooks/AGENTS.md` adds notebook-specific gotchas (unsloth, kernel,
  nbstripout, three bug-fix shims). Child files accumulate + may override.

Writing back (updating) policy is defined under
[Memory maintenance](#memory-maintenance) below.

## Dev environment

- Python 3.12 (pinned via `.python-version`)
- `uv` package manager (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- CUDA 12.8 toolkit on the host; `torch==2.11.0+cu128` wheels
- Target GPUs: H100 80 GB (primary), A100 80 GB (budget), L40S 48 GB (ok),
  24 GB cards (fit only with gradient-accum tricks)

Bootstrap from a fresh clone:

```bash
cp .env.example .env   # then fill HF_TOKEN, WANDB_API_KEY
make dev               # uv sync --group dev + pre-commit install
make test              # sanity check
```

The embedding-classifier notebook needs an extra group:
`uv sync --extra unsloth`.

## Canonical commands

File-scoped commands are preferred in the inner loop (fast feedback).
Full-repo commands are used by CI and before PR submission.

| Intent | Full repo | File-scoped (fast) |
|---|---|---|
| Lint | `make lint` | `uv run ruff check src/lid/bench/runner.py` |
| Format | `make format` | `uv run ruff format src/lid/bench/runner.py` |
| Typecheck | `make typecheck` | `uv run mypy src/lid/bench/runner.py` |
| Tests | `make test` | `uv run pytest tests/test_runner.py::test_grid_expansion -x` |
| Smoke bench | `make bench-quick` | (same — already minimal) |
| Full verify | `make verify` | (runs VERIFY.md in order, stops on first failure) |

CLI entrypoints (installed as console scripts by `pyproject.toml`):
`lid-train`, `lid-infer`, `lid-visualize`, `lid-bench`, `lid-upload`,
`lid-report`, `lid-recommend`.

## Do

- Use `X | None`, not `Optional[X]`. Use native generics (`list[int]`,
  `dict[str, float]`). Import `from __future__ import annotations` at the
  top of every module.
- Use **Google-style** docstrings and **full type annotations** on every
  public function signature.
- Keep line length ≤ 100 (ruff-configured, `E501` ignored but aim for 100).
- Follow `snake_case` (funcs/vars), `PascalCase` (classes),
  `UPPER_SNAKE_CASE` (module-level constants).
- Use Conventional Commits: `feat:`, `fix:`, `exp:`, `docs:`, `refactor:`.
- Branch prefixes: `feat/`, `fix/`, `exp/`, `docs/`.
- Prefer absolute imports: `from lid.bench.configs import RunConfig`.
- Place type-only imports inside a `TYPE_CHECKING` block (ruff `TCH` rules
  will enforce).
- Route `torch` **and** `torchvision` through the `pytorch-cu128` index in
  `[tool.uv.sources]`. PyPI wheels will silently ABI-mismatch.
- Load secrets (HF / W&B) from the repo-root `.env` via `python-dotenv`.
  Never use Colab `userdata` in this codebase.
- When editing a notebook, also read `notebooks/AGENTS.md` first.

## Don't

- **Don't** edit `uv.lock` by hand — it is auto-managed by `uv add` /
  `uv remove`.
- **Don't** edit `experiments/all_results.csv` by hand — it is
  append-only, written by `LocalResultsLogger`.
- **Don't** `!pip install ...` inside notebook cells. Use
  `uv sync --extra <group>` and document the group in `pyproject.toml`.
- **Don't** commit `.env`, `.env.*`, API keys, W&B keys, or HF tokens.
  `detect-private-key` pre-commit hook will catch many leaks, but do a
  manual `git diff --cached` review before every commit anyway.
- **Don't** introduce new Python dependencies without running `uv add`
  (never touch `pyproject.toml` dep lists by hand unless adding a new
  optional-dependencies group).
- **Don't** push without CI green. Pre-commit hooks must pass locally
  first (`nbstripout`, `ruff`, `detect-private-key`).
- **Don't** rename or delete files in `checkpoints/`, `experiments/`, or
  `wandb/` without explicit user approval.
- **Don't** single-letter-name variables `l`, `O`, `I` (ruff will flag).

## Safety and permissions

**Allowed without prompt:**
- Read any file under the repo.
- `make lint`, `make format`, `make typecheck`, `make test`, `make verify`.
- `make bench-quick` (local-only, `--no-wandb`).
- `uv run python -c "..."` for one-off diagnostics.
- Run pre-commit hooks locally.

**Ask first (confirmation required):**
- `uv add` / `uv remove` (changes `pyproject.toml` + `uv.lock`).
- `git commit` / `git push` (both local and remote history changes).
- `lid-bench` **without** `--no-wandb` (costs W&B credits + network).
- `lid-train ...` runs (expensive GPU time).
- `trainer.push_to_hub(...)` or `hf_api.upload_file(...)` (writes to HF Hub).
- `rm -rf` on anything under `checkpoints/`, `experiments/`, `wandb/`,
  `outputs/`.
- Modifying the `.github/workflows/` CI pipelines.

## Gotchas (project-specific)

These are traps we've actually hit. Keep fixes in mind before touching
related code.

- **torchvision ABI mismatch.** The PyPI torchvision wheel is built
  against a different CUDA than `torch==2.11.0+cu128`. Route through
  `[tool.uv.sources].torchvision -> pytorch-cu128`.
- **datasets 4.3.0 VideoReader.** `datasets` imports
  `torchvision.io.VideoReader` unconditionally; the OSS CUDA wheel
  omits the video backend. The embedding-classifier notebook installs
  a `VideoReader` stub shim in cell 1 — keep it.
- **unsloth flex_attention + XLM-RoBERTa dropout.** unsloth patches
  XLM-RoBERTa to use `flex_attention`, which rejects any non-zero
  attention dropout. `multilingual-e5-large` ships with
  `attention_probs_dropout_prob=0.1` → zero it out on both
  `model.config` and every `SelfAttention.dropout.p`.
- **OOM at `per_device_train_batch_size=1440` on 22 GB GPUs.** The
  original A100-80GB config needs ~80 GB. On smaller cards, drop
  `per_device_train_batch_size` to 32, set
  `gradient_accumulation_steps=45` (effective batch preserved),
  enable `gradient_checkpointing=True` with `use_reentrant=False`.
- **W&B `WANDB_LOG_MODEL=end`.** Per official docs, this only uploads
  when `load_best_model_at_end=True` is also set on `TrainingArguments`.
  If that's not the case, use an explicit `wandb.Artifact().add_dir(...)`
  upload. Set `WANDB_WATCH=all` before Trainer init to log gradient
  + parameter histograms.
- **HuggingFace auth.** Prefer `HF_TOKEN`; fall back to `HF_NEW`. Load
  via `load_dotenv()` from the repo-root `.env`.
- **nbstripout.** Every commit to a notebook strips cell outputs. Never
  commit outputs; never depend on committed outputs.

## Memory maintenance

**Retrieval (every session start):**
1. Read this file in full.
2. Read `PLAN.md` in full.
3. Read the top 5 entries of `.factory/memories.md`.
4. Skim `VERIFY.md` section headers.

**Updating:**
1. Check off completed items in `PLAN.md` as you go.
2. At meaningful checkpoints, append a dated CoALA-tagged entry to
   `.factory/memories.md` — either manually or via the `#` prefix
   (captured by `~/.factory/hooks/memory-capture.py`) or the
   `/remember` slash command.
3. Update **this** file only when a convention, command, or gotcha
   genuinely changes. Rare.
4. Update `VERIFY.md` when you add a new CLI, Makefile target, or
   test suite.

**Task transitions:** when a task is complete, summarise its outcome as
one `.factory/memories.md` entry, then overwrite `PLAN.md` with the next
task's scaffold.

**Sizing (per AGENTS.md v1.1 spec + industry experience):**
- This file: **≤ 400 lines**. If growing, move details to hierarchical
  AGENTS.md files under subdirectories.
- `.factory/memories.md`: **≤ 200 lines**. Archive oldest entries into
  `docs/memory-archive/YYYY-MM.md` when exceeded.
- `PLAN.md`: **≤ 120 lines**. One task at a time.
- `VERIFY.md`: **≤ 250 lines**. Keep exit-code tables terse.

## External references

When this file is not specific enough, consult the ground-truth docs
directly (progressive disclosure — do not inline them):

- **uv**: https://docs.astral.sh/uv/
- **PyTorch wheels**: https://pytorch.org/get-started/locally/
- **unsloth docs**: https://docs.unsloth.ai
- **HF Trainer + W&B integration**: https://docs.wandb.ai/models/integrations/huggingface_transformers
- **AGENTS.md spec**: https://agents.md
- **Factory Droid docs**: https://docs.factory.ai

## Repo-local pointers

Documentation lineage (genesis → extension → forward-looking):

- `docs/proposal_original.md` — **canonical source** proposal.
  Twelve research questions, original framing, reading list. Treat
  as the historical genesis document.
- `docs/project_proposal.md` — extended proposal with publishability
  assessment + experimental-design table.
- `docs/paperback.md` — venue-agnostic, deadline-agnostic
  forward-looking research-direction document; twelve well-scoped
  experiments (E1–E12), 9-model cohort, ~50-ref literature map,
  phased dependency graph.

Operational pointers:

- `docs/RUNBOOK.md` — step-by-step experiment runbook (phases 1–11).
- `docs/optimization_spec.md` — optimization strategy spec
  (problem analysis, architecture, per-strategy mathematics, W&B
  integration, metrics taxonomy).
- `experiments/EXPERIMENT_LOG.md` — recorded H100 80 GB run log
  (steps 1–7 with exact W&B run URLs and metric tables).
- `CONTRIBUTING.md` — full coding standards, PR workflow, recipes
  (canonical contributor-facing rules; this file is the distilled
  Do/Don't version for AI agents).
- `notebooks/AGENTS.md` — notebook-specific gotchas (the three bug-
  fix shims, nbstripout, kernel selection, dependency groups).
- `README.md` — public user-facing overview.
