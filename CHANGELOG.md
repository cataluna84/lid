# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `docs/proposal_original.md` -- the **canonical source proposal**
  recovered from the original Google Doc that started the project,
  with a lineage header explaining the
  `proposal_original.md` → `project_proposal.md` → `paperback.md`
  documentation chain.

### Changed

- `README.md`: added a **Project genesis and document lineage**
  section after the status banner, expanded the **Detailed
  walkthrough** with a two-execution-paths table and an 11-step
  orientation list, and replaced the terse Notebooks table with five
  per-notebook subsections (`LID_Regex`, `LID_Unicode_Blocks_
  Classifier`, `LID_Ngrams_Classifier`, `LID_Embedding_Classifier`,
  `LID_Inference_Vibecoded`) describing approach, dependencies, and
  role in the LID ladder.
- `CONTRIBUTING.md`: full external-contributor rewrite -- HTTPS
  clone, table of contents, repo orientation, six-step development
  workflow, full coding standards (ruff / mypy / typing / naming /
  docstrings / imports), the five-notebook table, six recipes (add
  a strategy / model / dataset / metric / notebook / CLI), PR
  checklist, security & secret hygiene, Code of Conduct reference,
  Apache-2.0 §5 license-grant note.
- `AGENTS.md`: post-release status header citing the
  `proposal_original.md` / `project_proposal.md` / `paperback.md`
  chain; reorganised "Repo-local pointers" into Documentation
  lineage + Operational pointers.
- `PLAN.md`: rewritten as the **Public-release end-to-end
  documentation sweep** plan; the previous *memory-system* task
  outcome is archived in `.factory/memories.md`.
- `docs/project_proposal.md`, `docs/optimization_spec.md`,
  `docs/RUNBOOK.md`, `experiments/EXPERIMENT_LOG.md`,
  `notebooks/AGENTS.md`, `VERIFY.md`: each now opens with a
  status callout pointing at the documentation lineage and the
  public-release context.

## [0.1.0] -- 2026-04-30

First public release. The repository is now public on GitHub at
<https://github.com/cataluna84/lid> and accepts community contributions
under the Apache License 2.0.

### Added

- **Public release of the full research artifact**, including the
  training, inference, optimization, and benchmarking pipelines.
- `LICENSE` (Apache-2.0) and `NOTICE` (Apache-2.0 attribution).
- `CITATION.cff` with software-entry citation metadata. There is no
  paper in submission; a forward-looking research-direction document
  lives at `docs/paperback.md`.
- `CODE_OF_CONDUCT.md` adopting the Contributor Covenant 2.1.
- `SECURITY.md` documenting GitHub private-advisory disclosure and an
  email fallback.
- `.env.example` matching the secrets read by `infer.py`, `train.py`,
  the benchmark runner, and the notebooks (`HF_TOKEN`, `WANDB_API_KEY`,
  `WANDB_PROJECT`, `WANDB_ENTITY`).
- `.github/ISSUE_TEMPLATE/` (bug, feature, experiment-question, plus
  blank-issue redirect) and `.github/PULL_REQUEST_TEMPLATE.md`.
- `.github/dependabot.yml` for weekly Python and `actions` updates.
- `[project.urls]` block in `pyproject.toml` (Homepage, Repository,
  Issues, Changelog, Documentation) so the PyPI/Hub surface and
  GitHub repository sidebar show the right links.

### Changed

- **License: MIT  →  Apache-2.0.** The Apache 2.0 license adds an
  explicit patent grant (§3) and a patent-retaliation defence,
  matching the de-facto standard for AI/ML research code (PyTorch,
  HuggingFace `transformers`, Llama, Qwen, Mistral). Sole-authorship
  was confirmed via `git log` before relicensing. Inbound contributions
  are accepted under Apache-2.0 §5 -- no separate CLA required.
- `README.md` rewritten from scratch around a public audience: status
  banner pointing to the successor umbrella project (TBA), TL;DR,
  quickstart, repository map, **Reproducibility caveats** (private HF
  dataset, personal W&B entity, hardware notes), citation, license,
  and acknowledgments. The previous detailed walkthrough is preserved
  and pointed at `docs/RUNBOOK.md` as the canonical step-by-step.
- `CONTRIBUTING.md` rewritten for external contributors: Discord
  references replaced with GitHub Issues + Discussions, HTTPS clone
  URL, explicit links to `LICENSE` / `NOTICE` / `CODE_OF_CONDUCT.md` /
  `SECURITY.md`, a license-grant note (Apache-2.0 §5), and a "Recipes"
  section covering "How to add a strategy / model / dataset".
- `pyproject.toml`: development-status classifier `3 - Alpha`  →
  `4 - Beta`; license classifier swap; expanded `keywords`;
  `description` tightened.
- `.gitignore`: now excludes `.coverage`, `.coverage.*`, `htmlcov/`,
  `coverage.xml`, `*.profraw`, `*.gcov`.
- `docs/RUNBOOK.md`: personal `wandb.ai/cataluna84/...` references are
  now `wandb.ai/<your-entity>/...` placeholders, with a callout.
- `docs/project_proposal.md`, `docs/optimization_spec.md`,
  `experiments/EXPERIMENT_LOG.md`: each now opens with a status
  callout clarifying its role in the public artifact.
- `docs/do.md`  →  `docs/paperback.md`. The internal execution
  plan (which was scoped against an ARR May 2026 submission window)
  has been generalised into a venue-agnostic, deadline-agnostic
  **forward-looking research-direction document** so any future
  researcher can pick up the line of work without inheriting a
  defunct deadline. Linked from `README.md` under
  *Future research directions*.

### Removed

- `.coverage` (~53 KB binary). It was an ephemeral pytest-cov artifact
  that should never have been tracked; it is now git-ignored.

### Unchanged (intentionally preserved)

- `AGENTS.md`, `PLAN.md`, `VERIFY.md`, `.factory/memories.md`,
  `.factory/skills/context-bootstrap/SKILL.md`, `notebooks/AGENTS.md`.
  These are part of the project's external-memory system and are
  retained verbatim as a transparent record of the development
  process. They are documented in the README under
  *Reproducibility caveats* so external readers know what they are.

[Unreleased]: https://github.com/cataluna84/lid/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cataluna84/lid/releases/tag/v0.1.0
