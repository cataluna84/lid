<!--
Thanks for sending a pull request! Please fill in each section so the
review goes quickly. Delete sections that don't apply.
-->

## Summary

<!-- One paragraph: what does this change, and why? -->

## Type of change

<!-- Check all that apply by replacing [ ] with [x] -->

- [ ] `feat:`     -- new feature, strategy, model, or dataset support
- [ ] `fix:`      -- bug fix
- [ ] `exp:`      -- experiment-only branch (may not be merged)
- [ ] `refactor:` -- code refactor without behaviour change
- [ ] `docs:`     -- documentation / comments / typos
- [ ] `test:`     -- tests only
- [ ] `ci:`       -- CI / tooling
- [ ] `chore:`    -- build / deps / housekeeping

## Linked issues

<!-- e.g. Closes #123, Refs #456. -->

## Checklist

- [ ] Branch is named with the convention `feat/...`, `fix/...`,
      `exp/...`, or `docs/...`.
- [ ] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/).
- [ ] `make lint` passes.
- [ ] `make typecheck` passes.
- [ ] `make test` passes.
- [ ] If a new public API was added, type hints + Google-style
      docstrings are present and correct.
- [ ] If a new dependency was added, it was added with `uv add` (never
      a hand-edit of `pyproject.toml` for the dep list).
- [ ] If a new optimization strategy was added, it is registered in
      `src/lid/bench/strategies/__init__.py` and has at least a
      smoke-test config under `configs/`.
- [ ] No secrets, tokens, or `.env` files in the diff. I ran
      `git diff --cached` and visually checked.
- [ ] By submitting this PR, I agree to license my contribution under
      the project's Apache License 2.0 (per Section 5 of that licence;
      no separate CLA is required).

## Verification

<!--
Paste the command(s) you ran to verify, and the relevant output. For
benchmark / training changes, link the W&B run URL.
-->

```
$ make lint
$ make typecheck
$ make test
```

## Notes for reviewers

<!-- Risk areas, follow-ups, things you would like a second pair of eyes on. -->
