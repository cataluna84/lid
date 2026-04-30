---
description: Active task plan (living document). Replace on task switch; archive outcome into .factory/memories.md.
tags: [plan, living-document]
---

# PLAN.md

This is a **living document**. The sections `Progress`, `Surprises &
Discoveries`, `Decision Log`, and `Outcomes & Retrospective` MUST be
kept up to date as work proceeds. Structure is modelled on OpenAI's
[PLANS.md convention](https://github.com/openai/openai-agents-js/blob/main/PLANS.md).

## Current task: Public-release end-to-end documentation sweep

- **Last updated:** 2026-04-30
- **Related branch/PR:** `docs/public-release-sweep`
- **Session lineage:** builds on commits `31b09cb` (notebook
  initialisation + memory system commit) and `46ccf6b` (memory
  system).

## Purpose / Big picture

The repository transitions from a private Droid-driven research
sandbox to a **public, Apache-2.0 OSS-friendly artifact**. After
this task, any newcomer should be able to:

1. Land on `README.md`, understand the genesis → extension →
   forward-looking documentation lineage, and pick the right entry
   point for their level of interest.
2. Read `CONTRIBUTING.md` and have a complete external-contributor
   guide -- coding standards, recipes, PR checklist, license grant
   -- without needing access to the AI-agent memory files.
3. Find every notebook (all five) explained in enough detail to
   choose which one to run.
4. Know that `docs/proposal_original.md` is the canonical source
   proposal, that `docs/project_proposal.md` extends it, and that
   `docs/paperback.md` is the venue/deadline-agnostic forward-
   looking research-direction document for anyone continuing the
   work.

## Progress

Checkboxes with timestamps. Every stopping point must appear here,
even partial completions.

### Phase 0 -- Pre-flight hygiene
- [x] (2026-04-25) Untrack `.coverage`; extend `.gitignore` with
  coverage artifacts.

### Phase 1 -- Relicensing
- [x] (2026-04-25) Verify sole authorship via `git log`; replace
  `LICENSE` with verbatim Apache 2.0; create `NOTICE` per Apache
  §4(d) with third-party acknowledgements; update `pyproject.toml`
  (license, classifier, keywords, project URLs).

### Phase 2 -- Community-health files
- [x] (2026-04-25) Create `.env.example`, `CODE_OF_CONDUCT.md`
  (Contributor Covenant 2.1 by reference), `SECURITY.md`,
  `CITATION.cff` (CFF 1.2.0, Apache-2.0), `CHANGELOG.md`
  (Keep-a-Changelog 0.1.0).
- [x] (2026-04-25) Create `.github/ISSUE_TEMPLATE/{config,bug_report,
  feature_request,experiment_question}.yml`,
  `.github/PULL_REQUEST_TEMPLATE.md`, `.github/dependabot.yml`.

### Phase 3 -- README rewrite
- [x] (2026-04-26) Rewrite `README.md` end-to-end: status banner,
  doc lineage, badges, TL;DR, quickstart, repo map, reproducibility
  caveats, hardware tables, strategy reference, FAQ, future
  research directions, citation, license, acknowledgments.

### Phase 4 -- Paperback rename and forward-looking reframe
- [x] (2026-04-27) Rename `docs/do.md` → `docs/paperback.md`.
- [x] (2026-04-27) Replace §14 six-week ARR Gantt with
  dependency-graph + per-experiment H100-hour effort estimates.
- [x] (2026-04-27) Reframe §15 paper outline conditionally; reframe
  §17 OQs for "future researcher".
- [x] (2026-04-27) Strip ARR May 2026 framing from `README.md`,
  `AGENTS.md`, `CITATION.cff`, `CHANGELOG.md`, `SECURITY.md`,
  `NOTICE`, `docs/project_proposal.md`.

### Phase 5 -- Documentation lineage integration
- [x] (2026-04-30) Fetch the genesis Google Doc proposal; save as
  `docs/proposal_original.md` with lineage header (proposal_original
  → project_proposal → paperback) and faithful body.
- [x] (2026-04-30) Add "Project genesis and document lineage"
  section to `README.md`; expand the **Detailed walkthrough** and
  **Notebooks** sections so every newcomer can navigate.
- [x] (2026-04-30) Rewrite `CONTRIBUTING.md` end-to-end (HTTPS
  clone, full TOC, repo orientation, 6-step workflow, coding
  standards, 5-notebook table, six recipes, PR checklist, security,
  Code of Conduct, Apache-2.0 §5 license-grant).
- [x] (2026-04-30) Update `AGENTS.md`: post-release status header
  citing the proposal_original / project_proposal / paperback chain;
  reorganise repo-local pointers into Documentation lineage +
  Operational pointers.

### Phase 6 -- Remaining docs sweep
- [ ] Update `docs/project_proposal.md` with a status callout
  pointing back to `proposal_original.md` (source) and forward to
  `paperback.md`.
- [ ] Update `docs/optimization_spec.md` with a status callout.
- [ ] Update `docs/RUNBOOK.md`: HTTPS clone URL, placeholder W&B
  entity, public-release status callout.
- [ ] Refresh `notebooks/AGENTS.md` for a public audience.
- [ ] Add a status callout to `experiments/EXPERIMENT_LOG.md`.
- [ ] Update `VERIFY.md` (no ARR refs; ensure all checks still hold
  post-release).
- [ ] Update `CHANGELOG.md` to record `proposal_original.md`
  addition + this end-to-end docs sweep.

### Phase 7 -- Verification
- [ ] Parse all YAML / TOML / CFF; spot-check anchor + cross-ref
  consistency; smoke `make verify`.

## Surprises & Discoveries

- **Observation:** Genesis Google Doc fetch returned JSON-encoded
  prose (`"s":"..."` chunks) rather than rendered HTML.
  **Resolution:** regex extraction + segment join recovered the full
  proposal text faithfully.
- **Observation:** Verbatim Contributor Covenant text was blocked by
  the upstream content filter when written into
  `CODE_OF_CONDUCT.md`.
  **Resolution:** reference Covenant 2.1 by URL with a short
  reporting-channels excerpt, rather than inline the full text.

## Decision Log

- **Decision:** target *Public + Permanently active*; use `<TBA>`
  placeholder for the umbrella project link.
  **Rationale:** maintainer wants community PRs without committing
  to active research on this exact codebase.
  **Date / Author:** 2026-04-25 / user AskUser approval.
- **Decision:** keep AI-agent external-memory files (`AGENTS.md`,
  `PLAN.md`, `VERIFY.md`, `.factory/`) at the repo root.
  **Rationale:** transparency of process; documented as such in
  `README.md` Reproducibility caveats §5.
  **Date / Author:** 2026-04-25 / user AskUser approval.
- **Decision:** record the proposal lineage as
  `proposal_original.md` → `project_proposal.md` → `paperback.md`.
  **Rationale:** the Google Doc that started the project is the
  canonical source of truth; everything downstream extends it; a
  clear chain helps future researchers navigate.
  **Date / Author:** 2026-04-30 / user request.

## Validation and Acceptance

After this task is complete, the following must all be true:

- [ ] `ls README.md CONTRIBUTING.md AGENTS.md SECURITY.md
  CODE_OF_CONDUCT.md CITATION.cff CHANGELOG.md LICENSE NOTICE
  .env.example` all exist.
- [ ] `ls docs/proposal_original.md docs/project_proposal.md
  docs/paperback.md docs/optimization_spec.md docs/RUNBOOK.md` all
  exist.
- [ ] No live document references a specific paper deadline as
  forward-looking; references in `CHANGELOG.md` and this `PLAN.md`
  exist only as historical record of the rename / strip phase.
- [ ] `python3 -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"`
  succeeds.
- [ ] `python3 -c "import yaml; yaml.safe_load(open('CITATION.cff'))"`
  succeeds.
- [ ] `make lint`, `make typecheck`, `make test` exit 0
  (or any failure is documented and pre-existing).

## Outcomes & Retrospective

*(Fill in at task completion.)*

## Out of scope

- Adding new optimisation strategies, models, or datasets -- the
  active surface is documentation, not code.
- Changing CLI entrypoint signatures.
- Touching `experiments/all_results.csv` (append-only).
- Configuring CI workflows under `.github/workflows/` (not present
  in this repo; out of scope until a maintainer requests one).
