---
description: Project memory for the lid repo. Rolling episodic log of decisions, bugs, fixes, and learnings.
tags: [memory, project-memory, episodic]
---

# Project memories — `lid`

Rolling journal, newest first. Entries are **CoALA-typed**:

- `[semantic]` — a learned fact, true until the world changes
- `[episodic]` — something that happened (a bug, a run, a deploy)
- `[procedural]` — a new workflow, tool, or command

Capture rules:

- Every entry MUST be date-stamped `YYYY-MM-DD HH:MMZ` (UTC).
- Three ways to add entries:
  1. Type `#<text>` in a Droid session — the `memory-capture.py` hook
     auto-appends a `[episodic]` entry with today's timestamp.
  2. Run `/remember <text>` (slash command) — prompts for CoALA tag.
  3. Edit this file directly.
- Archive policy: when this file exceeds ~200 lines, move the oldest
  entries into `docs/memory-archive/YYYY-MM.md` and leave a one-line
  pointer here.

---

### 2026-04-17 09:30Z [procedural] Four-file external memory system added
Authored `AGENTS.md`, `PLAN.md`, this file, `VERIFY.md`,
`notebooks/AGENTS.md`, and the Factory automation layer
(`.factory/skills/context-bootstrap/SKILL.md`,
`~/.factory/hooks/memory-capture.py`, `~/.factory/commands/remember.md`).
Design based on `docs/memory-system.PNG` (Tip 10), AGENTS.md v1.1 spec,
OpenAI PLANS.md convention, CoALA externalization theory.

### 2026-04-17 08:30Z [procedural] W&B instrumentation v2 aligned with official HF+W&B docs
Replaced v1 with docs-aligned pattern in embedding-classifier notebook:
`wandb.init(project, entity, name, group, tags, config, save_code, reinit)`
BEFORE Trainer; `report_to="wandb"` + `run_name=wandb_run.name` on
TrainingArguments; `wandb.define_metric(..., summary="max"|"min")` for
eval metrics; `wandb.plot.confusion_matrix(y_true, preds, class_names)`
inside `compute_metrics`; explicit `wandb.Artifact("model-<run_name>",
type="model").add_dir(...)` upload instead of relying on
`WANDB_LOG_MODEL=end`; `wandb.finish()` at the end.

### 2026-04-17 08:00Z [semantic] `WANDB_LOG_MODEL=end` silently skips upload without `load_best_model_at_end=True`
Per W&B integration docs, `end` only triggers when the Trainer keeps a
best-model-at-end. Without that flag set on `TrainingArguments`, nothing
uploads. For our single-epoch runs, use explicit `wandb.Artifact` +
`add_dir(...)` + `log_artifact(...)`. Set `WANDB_WATCH=all` before
Trainer init to log gradient + parameter histograms.

### 2026-04-16 22:40Z [episodic] Hit CUDA OOM at `per_device_train_batch_size=1440` on 22 GB GPU
Tried to allocate 720 MiB on top of 21.12 GiB already allocated. Peak
activation memory for bs=1440, sl=256, 24 layers with full-finetuning
projects to ~72 GB. Original A100-80GB config needs ~80 GB peak.
**Fix in cell 14:** drop `per_device_train_batch_size` from 1440 → 32,
raise `gradient_accumulation_steps` from 1 → 45 (preserves effective
batch of 1440), drop `per_device_eval_batch_size` from 512 → 32, enable
`gradient_checkpointing=True` with
`gradient_checkpointing_kwargs={"use_reentrant": False}`. Projected peak
~8–10 GB train, ~7 GB eval.

### 2026-04-16 21:10Z [semantic] unsloth-patched XLM-RoBERTa `flex_attention` rejects `attention_probs_dropout_prob != 0`
unsloth replaces XLM-RoBERTa self-attention with PyTorch's
`flex_attention` kernel, which does not support attention-probs dropout.
`intfloat/multilingual-e5-large` ships with `0.1`.
**Fix in cell 6:** after building the classifier head, set
`model.config.attention_probs_dropout_prob = 0.0` and loop over every
`nn.Module` whose class name ends in `SelfAttention`, zeroing
`module.dropout.p` attribute.

### 2026-04-16 20:00Z [semantic] `datasets==4.3.0` unconditionally imports `torchvision.io.VideoReader`
`datasets.formatting.torch_formatter` uses `isinstance(value, VideoReader)`
in its tensor-conversion path. The OSS CUDA torchvision wheel omits the
video backend, so `from torchvision.io import VideoReader` raises
`ImportError`. This blocks `trainer.train()` at the first `DataLoader`
iteration.
**Fix in cell 1:** compat shim — if `torchvision.io.VideoReader` does
not import, create a dummy stub class and inject it into
`torchvision.io`. The `isinstance(...)` check then falls through
harmlessly.

### 2026-04-16 19:00Z [procedural] `torchvision` must route through `pytorch-cu128` index to match `torch==2.11.0+cu128` ABI
PyPI's default `torchvision==0.26.0` wheel is built against a different
CUDA. On `from unsloth import FastModel`, it fails with
`ImportError: cannot import name 'VideoReader' from 'torchvision.io'`
(symbol present in cu128 build, absent in cpu build).
**Fix in `pyproject.toml`:** add `torchvision` to `[tool.uv.sources]`
with the same index routing as `torch`. Re-run `uv sync --extra unsloth`.
Installs `torchvision==0.26.0+cu128`.

### 2026-04-16 18:00Z [procedural] Notebook moved from `!pip install` to `uv sync --extra unsloth`
Added a new `unsloth` optional-dependencies group in `pyproject.toml`
containing: `unsloth, unsloth_zoo, sentencepiece, protobuf, hf_transfer,
xformers, trl, torchvision`. Notebook cell 0 now instructs the user to
run `uv sync --extra unsloth` from repo root BEFORE starting Jupyter.
Notebook cell 2 switched from Colab `userdata.get('HF_NEW')` to
`load_dotenv()` + `os.environ.get("HF_TOKEN") or os.environ.get("HF_NEW")`.

---

## Monthly review checklist (run every first Monday of the month)

- [ ] Remove entries that are no longer relevant (outdated dependencies,
  retired strategies, superseded bug fixes).
- [ ] Promote long-lived `[semantic]` facts to `AGENTS.md` gotchas if
  they now apply broadly.
- [ ] Check line count — if > 200, archive oldest entries into
  `docs/memory-archive/YYYY-MM.md`.
- [ ] Verify every entry is date-stamped; delete any undated entries
  (staleness cannot be tracked without them).

### 2026-04-17 09:32Z [procedural] memory-capture hook is installed
