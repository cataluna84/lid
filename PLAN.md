---
description: Active task plan (living document). Replace on task switch; archive outcome into .factory/memories.md.
tags: [plan, living-document]
---

# PLAN.md

This is a **living document**. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` MUST be kept
up to date as work proceeds. Structure is modelled on OpenAI's
[PLANS.md convention](https://github.com/openai/openai-agents-js/blob/main/PLANS.md).

## Current task: Four-file external memory system (+ W&B-instrumented embedding classifier)

- **Last updated:** 2026-04-17
- **Related branch/PR:** `feat/memory-system`
- **Session lineage:** builds on commits `46ccf6b memory system` and
  `bdf2074 feat: add embedding classifier notebook`.

## Purpose / Big Picture

After this change, any Droid (or human) dropping into this repo will:

1. Find `AGENTS.md`, `PLAN.md`, `.factory/memories.md`, `VERIFY.md`, and
   `notebooks/AGENTS.md` at predictable locations and learn the
   conventions without re-deriving them.
2. Have a `context-bootstrap` skill that auto-reads those files at
   session start.
3. Be able to auto-append memories by typing `#something` (hook) or
   `/remember something` (slash command).
4. Be able to prove the repo works end-to-end with a single `make verify`.

## Progress

Checkboxes with timestamps. Every stopping point must appear here, even
partial completions.

- [x] (2026-04-16 18:00Z) Move notebook from `!pip install` to `uv sync --extra unsloth`.
- [x] (2026-04-16 19:00Z) Route `torchvision` through `pytorch-cu128` index; verify `from unsloth import FastModel` works.
- [x] (2026-04-16 20:00Z) Add `torchvision.io.VideoReader` compat shim to notebook cell 1.
- [x] (2026-04-16 21:10Z) Zero out `attention_probs_dropout_prob` + all `SelfAttention.dropout.p` in notebook cell 6.
- [x] (2026-04-16 22:40Z) Drop `per_device_train_batch_size` to 32, set `gradient_accumulation_steps=45`, enable `gradient_checkpointing`.
- [x] (2026-04-17 08:00Z) W&B instrumentation v1: `wandb.init`, `report_to="wandb"`, per-language tables.
- [x] (2026-04-17 08:30Z) W&B instrumentation v2 (docs-aligned): `define_metric`, `wandb.plot.confusion_matrix`, explicit artifact upload, `wandb.finish()`.
- [x] (2026-04-17 09:10Z) Research Exa + Ref for memory-system best practices.
- [x] (2026-04-17 09:25Z) Author AGENTS.md, PLAN.md, .factory/memories.md, VERIFY.md, notebooks/AGENTS.md.
- [ ] Author `.factory/skills/context-bootstrap/SKILL.md`.
- [ ] Author `~/.factory/hooks/memory-capture.py` + `~/.factory/commands/remember.md`.
- [ ] Add `make verify` target that walks VERIFY.md.
- [ ] Run `make lint && make typecheck && make test` locally.

## Surprises & Discoveries

Document unexpected behaviours with concise evidence.

- **Observation:** `WANDB_LOG_MODEL=end` silently skips the upload if
  `load_best_model_at_end=True` is not also set on `TrainingArguments`.
  **Evidence:** W&B docs note; our TrainingArguments does not set it.
  **Resolution:** use an explicit `wandb.Artifact().add_dir(...)` call
  in cell 17.
- **Observation:** unsloth's `flex_attention` path rejects any non-zero
  attention-probs dropout, but `intfloat/multilingual-e5-large` ships
  with `attention_probs_dropout_prob=0.1`.
  **Evidence:** `ValueError: flex_attention does not support dropout`
  on `trainer.train()`.
  **Resolution:** zero out `model.config.attention_probs_dropout_prob`
  and every `SelfAttention.dropout.p` after loading.

## Decision Log

- **Decision:** store PROGRESS-style journal at `.factory/memories.md`
  instead of `PROGRESS.md` at root.
  **Rationale:** Factory Droid auto-indexes `.factory/memories.md` as
  **project memory**; keeps the file discoverable by both Droid-native
  flows and cross-tool workflows (via AGENTS.md pointer).
  **Date / Author:** 2026-04-17 / user approval.
- **Decision:** add `notebooks/AGENTS.md` now (not later).
  **Rationale:** the three bug-fix shims (VideoReader, flex_attention,
  OOM) are notebook-specific and already documented — cheap to capture
  now, expensive to rediscover later.
  **Date / Author:** 2026-04-17 / user approval.
- **Decision:** implement full Factory automation layer (bootstrap
  skill + `#` hook + `/remember` command).
  **Rationale:** fixes the "Fire-and-Forget" anti-pattern (55% of teams
  lack verification hooks) and reduces retrieval friction to near-zero.
  **Date / Author:** 2026-04-17 / user approval.

## Validation and Acceptance

After this task is complete, the following must all be true:

- [ ] `ls AGENTS.md PLAN.md VERIFY.md notebooks/AGENTS.md .factory/memories.md .factory/skills/context-bootstrap/SKILL.md` all exist.
- [ ] `make verify` exits 0 on a clean checkout (assuming GPU + tokens
  configured).
- [ ] `make lint`, `make typecheck`, `make test` all exit 0.
- [ ] A fresh Droid session retrieves AGENTS.md + PLAN.md +
  `.factory/memories.md` top-5 + VERIFY.md headers via the
  `context-bootstrap` skill without the user having to ask.
- [ ] Typing `#foo bar baz` appends a correctly-formatted dated entry
  to `.factory/memories.md` via the hook.

## Outcomes & Retrospective

*(Fill in at task completion.)*

## Out of scope

- Creating `.factory/rules/*.md` files — `CONTRIBUTING.md` is the
  canonical source of rules; AGENTS.md distils them. Avoid divergence.
- Configuring hooks in `settings.json` automatically — that requires
  the `/hooks` UI. The hook script is installed; the user pastes the
  config block per instructions in `~/.factory/hooks/memory-capture.py`.
- Writing new `.factory/droids/` subagents — not needed for this task.
- Touching any existing source file under `src/lid/` or existing
  notebook contents.
