---
name: context-bootstrap
description: Load the four-file external memory system at the start of every session in the lid repo. Use PROACTIVELY on the first turn of any conversation about the lid repository to retrieve project conventions, active plan, recent history, and verification protocols.
---

# context-bootstrap

## When to invoke

**Always** at the start of any new Droid session on the `lid` repo,
before responding to the user's first substantive request. Invoke
automatically when any of the following are true:

- The working directory is `/home/ubuntu/Workspace/lid` or a descendant.
- The user's first message mentions LID, language identification,
  embedding classifier, `lid-bench`, or any other LID-specific concept.
- You have not already loaded these files in the current session.

## What to do

Read the following files in this exact order. Do not skip any step.

### 1. Read `AGENTS.md` — full file

This is the repo's index file. It contains conventions, canonical
commands, Do/Don't rules, safety/permissions rules, and a list of
project-specific gotchas. Every decision you make in this session
should be consistent with the rules in `AGENTS.md`.

### 2. Read `PLAN.md` — full file

This is the active task's living document (purpose, progress,
surprises, decisions, acceptance criteria, out-of-scope items).
Cross-reference everything you do against the current task.

### 3. Read `.factory/memories.md` — top 5 entries only

The file is reverse-chronological. The top 5 entries capture the most
recent facts, bugs, and decisions. Older entries are archived under
`docs/memory-archive/` when the file exceeds 200 lines.

You do NOT need to read the whole file unless the user asks about a
specific older entry.

### 4. Skim `VERIFY.md` — section headers only

This file lists deterministic pass/fail commands that prove the repo
works. Knowing its section headers lets you answer "how do I verify
X?" without re-reading. The entire file can be executed as
`make verify`.

### 5. Read `notebooks/AGENTS.md` — only if the user's task touches `.ipynb` files

Notebook-specific gotchas (unsloth, nbstripout, kernel selection,
three bug-fix shims) live here. Inherits the root AGENTS.md.

## What to tell the user

After retrieving the above, acknowledge the load with ONE short line
(no more) before addressing the task, e.g.:

> "Loaded AGENTS.md, PLAN.md, top 5 memories, VERIFY.md headers."

Then proceed to the user's request. Do NOT summarise the contents of
the four files to the user — they can read them — unless the user
explicitly asks.

## Why this skill exists

Research (agents.md v1.1 spec, aihackers.net, claudelab.net) shows
that the two most common anti-patterns for agentic coding are:

1. **"The Blank Slate"** — skipping project memory, affecting 85% of
   new users and requiring 2–3x more correction rounds.
2. **"Kitchen Sink Session"** — mixing unrelated tasks without
   consulting prior context, leading to 40% slower responses at 85%
   context fill.

This skill mitigates both by making retrieval the first action of
every session.

## What NOT to do

- Don't load the four files every single turn — once per session is
  enough. Re-load only if the user says "I've updated PLAN.md" or
  "check memories again".
- Don't dump the contents of the files back to the user.
- Don't silently modify any of the files based on this skill alone —
  memory updates must go through the documented capture methods
  (the `#` prefix hook, the `/remember` slash command, or an
  explicit user request).
