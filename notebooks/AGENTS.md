---
description: Notebook-specific conventions and gotchas for lid/notebooks. Extends the root AGENTS.md; unsloth + jupyter + nbstripout land mines live here.
tags: [notebooks, unsloth, jupyter, nbstripout, xlm-roberta]
---

# notebooks/AGENTS.md

Scope: every `.ipynb` under `notebooks/`. Inherits rules from the root
`AGENTS.md`; this file adds notebook-specific constraints.

> The repository is **public** under Apache-2.0. The notebook ladder
> (Regex → Unicode-Blocks → N-grams → Embedding → Layer-wise
> Vibecoded) is described for newcomers in
> [`README.md`](../README.md#notebooks); a recipe for adding a new
> notebook lives in
> [`CONTRIBUTING.md`](../CONTRIBUTING.md#how-to-add-a-new-notebook).
> This file is the AI-agent / maintainer-facing version with the
> three required bug-fix shims that **must not** be removed.

## Dependencies

- The embedding-classifier notebook requires the `unsloth` extras group.
  Run `uv sync --extra unsloth` from the repo root **before** launching
  Jupyter. Never use `!pip install ...` inside cells.
- Other notebooks (`LID_Inference_Vibecoded`, `LID_Regex`,
  `LID_Ngrams_Classifier`, `LID_Unicode_Blocks_Classifier`) use only
  the base dependencies (`make dev`).

## Kernel selection

Select the `LID (Python 3.12)` kernel (registered by `ipykernel` during
`make dev`). Do NOT create per-notebook venvs; they'd miss the uv-managed
lockfile.

## nbstripout pre-commit hook

The repo's `.pre-commit-config.yaml` runs `nbstripout` on every commit.
Consequences:

- Cell outputs are stripped from every notebook at commit time.
- **Never** depend on committed outputs being present — they won't be.
- **Never** try to commit a notebook with outputs intact (it's a no-op
  but clutters diffs).
- Long outputs visible locally while you work; they disappear after
  commit.

## Auth: no Colab `userdata`

Load HF / W&B tokens via `python-dotenv` from the repo-root `.env`:

```python
import os
from dotenv import load_dotenv
load_dotenv()
hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_NEW")
assert hf_token, "Set HF_TOKEN (or HF_NEW) in your .env file at the repo root"
```

Do NOT use `from google.colab import userdata; userdata.get(...)`. It
won't work outside Colab and breaks local execution.

## The three bug-fix shims in LID_Embedding_Classifier.ipynb

These fixes live in specific cells and **must stay**. Do not remove them
without first updating the `unsloth` / `datasets` / `torchvision` version
pins in `pyproject.toml` and re-verifying.

### Cell 1 — `torchvision.io.VideoReader` compat shim

```python
# torchvision.io.VideoReader is imported unconditionally by
# datasets==4.3.0 (torch formatter). The OSS CUDA wheel lacks it.
try:
    from torchvision.io import VideoReader  # noqa: F401
except ImportError:
    import torchvision.io as _tvio
    class VideoReader:  # type: ignore[no-redef]
        def __init__(self, *a, **k): raise NotImplementedError
    _tvio.VideoReader = VideoReader
```

### Cell 6 — flex_attention dropout fix

```python
# unsloth patches XLM-RoBERTa self-attention with flex_attention,
# which rejects non-zero attention dropout. multilingual-e5-large
# ships with attention_probs_dropout_prob=0.1.
model.config.attention_probs_dropout_prob = 0.0
for _name, _m in model.named_modules():
    if _m.__class__.__name__.endswith("SelfAttention") and hasattr(_m, "dropout"):
        _m.dropout.p = 0.0
```

### Cell 14 — memory-optimized TrainingArguments (for ≤ 24 GB GPUs)

```python
# Original A100-80GB config: per_device_train_batch_size=1440.
# On 22 GB GPUs, use 32 + gradient_accumulation_steps=45 to preserve
# effective batch (32 * 45 = 1440) and enable gradient_checkpointing.
TrainingArguments(
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=45,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    # ... rest unchanged
)
```

## W&B instrumentation (embedding-classifier notebook)

Follow the docs-aligned pattern (see
`.factory/memories.md` entry for 2026-04-17 08:30Z):

1. Call `wandb.init(project, entity, name, group, tags, config,
   save_code, reinit)` **before** instantiating `Trainer`.
2. Set `WANDB_WATCH=all` and `WANDB_LOG_MODEL=false` env vars **before**
   Trainer init.
3. Use `report_to="wandb"` + `run_name=wandb_run.name` on
   `TrainingArguments`.
4. Call `wandb.define_metric(..., summary="max"|"min")` for each eval
   metric so the Runs-table surfaces best values.
5. Log per-language tables + `wandb.plot.confusion_matrix` inside
   `compute_metrics`.
6. Upload the final model as `wandb.Artifact("model-<run_name>",
   type="model").add_dir(...)` — do NOT rely on `WANDB_LOG_MODEL=end`
   without `load_best_model_at_end=True`.
7. Call `wandb.finish()` at the last cell — required in notebooks.

## Conventions for adding a new notebook

- Name it `LID_<Method>_<Purpose>.ipynb`.
- Cell 0: a markdown cell pointing at the `uv` extras group needed
  (or "no extras — base `make dev` is enough").
- Cell 1: imports + any required compat shims.
- Cell 2: `.env` loading + auth.
- Cells 3…N: the notebook's actual work.
- After a useful function lands, refactor it into `src/lid/` with
  proper type annotations and tests (per root `CONTRIBUTING.md`).

## Don't

- **Don't** commit notebook outputs. (nbstripout will strip them, but
  large pre-stripped diffs slow down review.)
- **Don't** `pip install` inside cells. (See dependencies section above.)
- **Don't** remove the three bug-fix shims without a plan for the
  replacement dependencies.
- **Don't** hard-code GPU VRAM assumptions (e.g. `bs=1440`) — use
  grad-accum so the notebook runs on both H100 80 GB and 24 GB cards.
