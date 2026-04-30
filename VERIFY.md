---
description: Deterministic pass/fail commands that prove the lid repo works. Run `make verify` to execute all checks in order.
tags: [verify, protocol, ci-local]
---

# VERIFY.md

Deterministic, machine-readable verification contracts for the `lid`
repo. Each check specifies **the exact command**, **the expected exit
code**, and **an expected output pattern**.

> **Status (2026-04-30):** the repository is **public** under
> Apache-2.0. The checks below are stable across the public-release
> sweep; the only post-release behavioural change is that the
> `1024m/LID` HuggingFace dataset is now **gated** (request access on
> the dataset page) rather than fully private -- `HF_TOKEN`-based
> access still works for everyone with approved access.

Run the entire suite with:

```bash
make verify
```

It stops on the first failure. Individual sections can be run in
isolation with the commands listed below.

> **Scope of VERIFY.md:** cheap, local, deterministic checks. Expensive
> experiments (`lid-bench` on the full grid, `lid-train` end-to-end,
> notebook end-to-end execution on GPU) are **not** part of `make
> verify`. They live in `docs/RUNBOOK.md`.

---

## Section 1 — Environment sanity

| Check | Command | Expected exit | Expected output pattern |
|---|---|---|---|
| Python version | `python3 --version` | 0 | `Python 3.12\.` |
| uv installed | `uv --version` | 0 | `^uv \d+\.\d+` |
| CUDA available | `uv run python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"` | 0 | *(no output)* |
| torch cu128 build | `uv run python -c "import torch; assert '+cu128' in torch.__version__, torch.__version__"` | 0 | *(no output)* |
| HF token present | `uv run python -c "import os; from dotenv import load_dotenv; load_dotenv(); assert os.environ.get('HF_TOKEN') or os.environ.get('HF_NEW'), 'missing HF token'"` | 0 | *(no output)* |

## Section 2 — Code quality

| Check | Command | Expected exit | Expected output pattern |
|---|---|---|---|
| Lint (ruff check) | `make lint` | 0 | `All checks passed!` or empty |
| Format (ruff format, no-op) | `uv run ruff format --check src/ tests/` | 0 | `N files already formatted` |
| Typecheck (mypy) | `make typecheck` | 0 | `Success: no issues found in N source files` |
| Tests (pytest) | `make test` | 0 | `N passed` (no failures) |

## Section 3 — Smoke benchmark

| Check | Command | Expected exit | Expected output pattern |
|---|---|---|---|
| Quick bench runs | `make bench-quick` | 0 | Summary table containing `eager` and `vectorized` rows |
| vectorized ≥ 5x eager | grep the printed `throughput_sps` column | 0 | `vectorized` row sps ≥ 5 × `eager` row sps |
| Identical accuracy | same CSV | 0 | `acc` column equal within ± 0.01 between eager and vectorized |

The numeric thresholds are enforced inside `make bench-quick` when run
against the default `configs/bench_quick.yaml`. On H100 80 GB expect
~14 sps eager / ~70 sps vectorized; on smaller GPUs the ratio is what
matters, not the absolute numbers.

## Section 4 — Notebook smoke

| Check | Command | Expected exit | Expected output pattern |
|---|---|---|---|
| Unsloth extra synced | `uv run python -c "import unsloth, unsloth_zoo, xformers, trl"` | 0 | *(no output)* |
| FastModel imports | `uv run python -c "from unsloth import FastModel"` | 0 | *(no output)* |
| VideoReader shim works | `uv run python -c "import torchvision.io as tvio; _ = getattr(tvio, 'VideoReader', None); print('ok')"` | 0 | `ok` |
| Notebook JSON valid | `uv run python -c "import json; json.loads(open('notebooks/LID_Embedding_Classifier.ipynb').read()); print('ok')"` | 0 | `ok` |

For an exhaustive (but expensive) end-to-end notebook execution, use:

```bash
uv run jupyter nbconvert --to notebook --execute \
  notebooks/LID_Embedding_Classifier.ipynb \
  --ExecutePreprocessor.timeout=600 \
  --output /tmp/LID_Embedding_Classifier.executed.ipynb
```

This requires GPU + HF token + W&B key; not part of `make verify`.

## Section 5 — Data access

| Check | Command | Expected exit | Expected output pattern |
|---|---|---|---|
| HF dataset loads | `uv run python -c "from datasets import load_dataset; d = load_dataset('1024m/LID', data_files='Data_Hackathon/LID-1000.parquet'); print(sorted(d.keys()))"` | 0 | `['train']` |

Requires `HF_TOKEN` in `.env`. If this fails with `403`, check that
your token has been granted access to the **gated** `1024m/LID`
dataset (request access on the dataset's HuggingFace page).

## Section 6 — W&B connectivity

| Check | Command | Expected exit | Expected output pattern |
|---|---|---|---|
| W&B login | `uv run python -c "import wandb; wandb.login(anonymous='allow'); print('ok')"` | 0 | `ok` |
| W&B online mode | `uv run python -c "import os; assert os.environ.get('WANDB_MODE', 'online') != 'disabled', 'W&B is disabled'; print('ok')"` | 0 | `ok` |

Requires `WANDB_API_KEY` in `.env` for `online` runs. If absent, W&B
falls back to `offline` (which still passes this check, but logs nothing
to the web UI).

## Section 7 — Memory system files present

| Check | Command | Expected exit | Expected output pattern |
|---|---|---|---|
| AGENTS.md | `test -f AGENTS.md` | 0 | |
| PLAN.md | `test -f PLAN.md` | 0 | |
| memories.md | `test -f .factory/memories.md` | 0 | |
| VERIFY.md | `test -f VERIFY.md` | 0 | |
| notebooks/AGENTS.md | `test -f notebooks/AGENTS.md` | 0 | |
| context-bootstrap SKILL | `test -f .factory/skills/context-bootstrap/SKILL.md` | 0 | |

---

## Exit-code contract for `make verify`

- Exit `0` — every section above passed.
- Exit `1` — environment sanity failed (sections 1 or 7). Abort before
  running expensive checks.
- Exit `2` — code quality failed (section 2).
- Exit `3` — smoke benchmark failed (section 3).
- Exit `4` — notebook smoke failed (section 4).
- Exit `5` — data or W&B connectivity failed (sections 5 or 6).

Failure messages should include the command that failed and the first
100 characters of its stderr, so a Droid can post-mortem without
re-running everything.
