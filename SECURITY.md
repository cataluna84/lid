# Security Policy

## Supported versions

This repository ships a research artifact and is in transition into a
successor umbrella project (link TBA in `README.md`). Only the current
`main` branch is supported for security fixes; older tags and
historical experiment outputs in `experiments/` are preserved as-is
for reproducibility.

| Version  | Supported |
|----------|-----------|
| `main`   | yes       |
| tags < `v0.1.0` (pre-public-release internal commits) | no |

## Reporting a vulnerability

We use GitHub's coordinated-disclosure flow as the primary channel.
Please **do not** open a public issue or pull request for a suspected
security problem.

### Preferred: GitHub private advisory

1. Go to <https://github.com/cataluna84/lid/security/advisories/new>.
2. Click **Report a vulnerability** ("Security" tab → "Report a
   vulnerability").
3. Fill in the form. This opens a private channel between you and the
   maintainer; nothing is published until a fix lands and we agree on
   coordinated disclosure timing.

### Fallback: email

If you cannot use the GitHub flow, email the maintainer directly:

  - `mayankbhaskar007@gmail.com`

Please include:

- A clear description of the issue and the impact.
- A minimal reproduction (commit hash, command, expected vs observed
  behaviour).
- Whether you intend to disclose publicly and on what timeline.

We aim to acknowledge new reports within **3 business days** and to
coordinate a fix within **30 days** for high-severity issues. Lower-
severity issues may be batched into the next maintenance release.

## What we consider a security issue

Because this is a research repository (training, inference, and
benchmarking code; no production endpoints), the most likely security
issues are:

- Accidental disclosure of secrets (HuggingFace tokens, W&B keys) in
  committed files, notebook outputs, or experiment logs.
- Code that executes attacker-controlled input from a model checkpoint
  or dataset in an unsafe way (e.g., unsafe `pickle.load`, arbitrary-
  code execution paths).
- Supply-chain issues affecting `pyproject.toml` / `uv.lock`
  dependencies.
- Vulnerabilities in CI workflows (`.github/workflows/`) such as
  workflow injection or token over-privilege.

If you are reporting a vulnerability in an upstream project we depend
on (PyTorch, transformers, bitsandbytes, etc.), please report it to the
upstream maintainers; we will track it via Dependabot once it is
public.

## Disclosure expectations

We follow a coordinated-disclosure model:

1. Reporter privately submits the issue (above).
2. Maintainer confirms, scopes, and develops a fix.
3. Reporter and maintainer agree on a public-disclosure date.
4. Fix is merged; a GitHub Security Advisory is published with credit
   to the reporter (unless they prefer to remain anonymous).
