# LID -- Layer-Wise Multilingual Language Identification

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![CI](https://github.com/cataluna84/lid/actions/workflows/ci.yaml/badge.svg)](https://github.com/cataluna84/lid/actions/workflows/ci.yaml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/badge/managed%20by-uv-7c3aed.svg)](https://docs.astral.sh/uv/)
[![Cite this repo](https://img.shields.io/badge/cite-CITATION.cff-orange.svg)](CITATION.cff)

Mechanistic-interpretability + empirical-systems study of how compact
foundation models (0--4 B parameters) learn to identify languages, across
**67 language classes**, with depth-resolved analysis of every transformer
layer. The repository ships training, layer-wise inference, an extensible
benchmarking framework over **9 optimization strategies**, W&B + local
result logging, and four exploratory classifier notebooks.

> ## Project status -- in transition
>
> This repository is a **public, reproducible research artifact**.
> Active long-term research on this topic is moving to a successor
> *umbrella project* that subsumes this work alongside related
> multilingual-NLP threads.
>
> - **Successor / umbrella:** *link TBA -- to be filled in here when
>   that project goes public.*
> - **This repo:** **public, permanently active.** We accept community
>   bug fixes, reproducibility improvements, doc PRs, new strategies,
>   and small features. Larger refactors are usually a better fit for
>   the umbrella project.
> - **Paper:** none currently in submission. A natural future paper
>   based on this artifact is sketched in
>   [`docs/paperback.md`](docs/paperback.md), which doubles as a
>   forward-looking research-direction document for anyone who wants
>   to continue this line of work.

### Project genesis and document lineage

Every research question in this repository ultimately traces back to
a single short proposal. There are **three** documents in the
`docs/` directory, in increasing order of detail and scope:

1. **[`docs/proposal_original.md`](docs/proposal_original.md)** --
   the **canonical source** proposal. Twelve research questions, the
   "why this matters" framing, the initial reading list, the
   first-cut model and dataset shortlists. Treat it as the genesis
   document; it is faithful to the original Google Doc that started
   the project.
2. **[`docs/project_proposal.md`](docs/project_proposal.md)** -- an
   **extension** of the source proposal. Same questions, restructured
   into a publishability assessment, an experimental-design table,
   and a related-work pointer.
3. **[`docs/paperback.md`](docs/paperback.md)** -- a
   **forward-looking research-direction document** that generalises
   the proposal into a venue-agnostic, deadline-agnostic roadmap
   with twelve well-scoped experiments (E1--E12), a 9-model cohort
   across three open-weight families, an ~50-reference literature
   map (2024--2026), and a phased dependency graph with H100-hour
   estimates.

If you have ten minutes, read (1). If you have an hour, read (1) +
(2). If you are seriously thinking about continuing this work, read
all three and then look at the codebase and notebooks.

---

## TL;DR

- **Task:** classify text into one of 67 languages using a compact
  decoder-only model (`CohereLabs/tiny-aya-global`, 3.35 B params).
- **What's interesting:** we extract the per-class probability at
  **every transformer layer** (37 hidden states, including the
  embedding layer) on every sample, so the unit of analysis is
  *(layer index, sample, language)* rather than just *(sample, prediction)*.
- **Optimization framework:** nine inference strategies (`eager`,
  `vectorized`, `compiled`, `sdpa`, `flash_attn`, `quantized_int8`,
  `quantized_int4`, `combined_flash_compiled`, `combined_flash_int8`)
  benchmarked on a YAML-defined Cartesian grid; the best combination
  delivers **~15× throughput** vs. eager at identical accuracy.
- **Reference notebooks:** Unicode-block heuristics, character
  n-grams, and a 0.6 B embedding classifier reaching **macro F1 0.97+**
  on out-of-domain web text (see `notebooks/`).
- **Reproducibility:** single `make verify` runs lint, types, tests,
  notebook smoke checks, and (optional) a quick benchmark; full
  step-by-step is in [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

---

## Table of Contents

1. [Quickstart](#quickstart)
2. [Repository map](#repository-map)
3. [Reproducibility caveats](#reproducibility-caveats)
4. [Hardware requirements](#hardware-requirements)
5. [Detailed walkthrough](#detailed-walkthrough)
6. [Strategy reference](#strategy-reference)
7. [W&B integration](#wb-integration)
8. [Notebooks](#notebooks)
9. [FAQ and troubleshooting](#faq-and-troubleshooting)
10. [Future research directions](#future-research-directions)
11. [Citation](#citation)
12. [License](#license)
13. [Acknowledgments](#acknowledgments)

---

## Quickstart

```bash
# Clone (HTTPS works for everyone; SSH is fine if you prefer).
git clone https://github.com/cataluna84/lid.git
cd lid

# Copy the env template and fill in your tokens.
cp .env.example .env
# Edit .env:
#   HF_TOKEN=hf_...                (required; access to 1024m/LID dataset)
#   WANDB_API_KEY=...              (optional; only for W&B logging)
#   WANDB_ENTITY=<your-entity>     (optional)

# Install everything (Python 3.12, PyTorch 2.11+cu128, all deps).
make dev

# Run the test suite + a quick local benchmark.
make test
make bench-quick       # ~5 min on H100, ~10 min on A100; --no-wandb
```

For the full step-by-step from baseline to fine-tuned model and
publication-quality numbers, see **[`docs/RUNBOOK.md`](docs/RUNBOOK.md)**.

---

## Repository map

```
lid/
├── src/lid/                        # Core library
│   ├── constants.py                #   67 language mappings, defaults
│   ├── data.py                     #   Dataset loading, prompt construction
│   ├── model.py                    #   Model loading, layer-wise inference
│   ├── train.py                    #   Fine-tuning pipeline (LoRA + AMP)
│   ├── infer.py                    #   Batched layer-wise inference
│   ├── visualize.py                #   Plotting: accuracy curves, per-language
│   ├── recommend.py                #   Best-config picker from results CSV / W&B
│   ├── report.py                   #   Comparison report from W&B runs
│   ├── upload.py                   #   Backfill local results to W&B
│   └── bench/                      #   Optimization + benchmarking framework
│       ├── configs.py              #     RunConfig / ExperimentGrid dataclasses
│       ├── strategy.py             #     InferenceStrategy ABC + registry
│       ├── strategies/             #     9 registered strategies (see below)
│       ├── metrics.py              #     3-tier metrics collector
│       ├── wandb_logger.py         #     W&B integration wrapper
│       ├── local_logger.py         #     Local filesystem results logger
│       └── runner.py               #     Grid expansion + execution engine
├── configs/                        # Experiment configurations
├── notebooks/                      # Exploration notebooks (outputs stripped)
├── docs/
│   ├── RUNBOOK.md                  #   Step-by-step copy-paste commands
│   ├── proposal_original.md        #   Genesis proposal (source of truth)
│   ├── project_proposal.md         #   Extended proposal + publishability assessment
│   ├── paperback.md                #   Forward-looking research-direction document
│   └── optimization_spec.md        #   Optimization strategy spec
├── experiments/                    # Output directory for results (kept)
├── tests/                          # Unit tests
├── pyproject.toml                  # uv project config (Apache-2.0)
├── Makefile                        # Common commands
├── LICENSE / NOTICE                # Apache 2.0 + attribution
├── CITATION.cff                    # Software citation metadata
├── CODE_OF_CONDUCT.md              # Contributor Covenant 2.1
├── SECURITY.md                     # Coordinated-disclosure policy
├── CONTRIBUTING.md                 # Public-contributor guide
├── CHANGELOG.md                    # Keep-a-Changelog
└── .env.example                    # Token template
```

### Installed CLI entrypoints

| Command          | Module                  | Purpose |
|------------------|-------------------------|---------|
| `lid-train`      | `lid.train`             | Fine-tune a causal LM with LoRA / AMP |
| `lid-infer`      | `lid.infer`             | Run layer-wise inference (original, eager) |
| `lid-bench`      | `lid.bench.runner`      | Run a YAML benchmark grid across 9 strategies |
| `lid-visualize`  | `lid.visualize`         | Plot results from a saved `results.pkl` |
| `lid-recommend`  | `lid.recommend`         | Pick the best inference config from results |
| `lid-report`     | `lid.report`            | Comparison report from the W&B `lid-bench` project |
| `lid-upload`     | `lid.upload`            | Backfill local results to W&B |

---

## Reproducibility caveats

This is a research artifact; please read this section before reporting
"the numbers don't match". None of the items below blocks reproduction --
they describe the deltas you should expect.

1. **Dataset access (gated).** Most experiments use
   [`1024m/LID`](https://huggingface.co/datasets/1024m/LID) (the
   1-800-LLMs hackathon LID corpus). Access is gated; you must
   request it on the dataset page and supply the resulting token via
   `HF_TOKEN` in `.env`. Without a token you can still run the
   embedding-classifier and Unicode-block notebooks (they use other
   sources), but the main bench grid will fail at dataset load.

2. **W&B project ownership.** All recorded runs in
   `experiments/EXPERIMENT_LOG.md` and `docs/RUNBOOK.md` link to
   `wandb.ai/cataluna84/lid-bench`, which is a personal entity. Set
   `WANDB_ENTITY=<your-entity>` in `.env` and your runs will land in
   your own project namespace. The recorded URLs are read-only;
   contributors should not attempt to write into them.

3. **Hardware sensitivity.** Numbers in `EXPERIMENT_LOG.md` were
   recorded on an **NVIDIA H100 80 GB HBM3**. On A100 80 GB expect
   ~30--40 % lower throughput at the same memory; on smaller cards
   you will additionally need to drop batch sizes (see *Hardware
   requirements* below). The relative speedup ranking between
   strategies is stable; the absolute samples-per-second number is
   not.

4. **`output_hidden_states=True` is memory-heavy.** Forcing the model
   to retain all 37 hidden states inflates peak VRAM 4--6 ×, even
   under quantization. This is intentional (the entire study depends
   on layer-wise extraction); it also explains the high
   `gpu_mem_peak_mb` columns in the results.

5. **External-memory files.** The repo intentionally keeps
   [`AGENTS.md`](AGENTS.md), [`PLAN.md`](PLAN.md),
   [`VERIFY.md`](VERIFY.md), [`.factory/memories.md`](.factory/memories.md),
   [`.factory/skills/`](.factory/skills/), and
   [`notebooks/AGENTS.md`](notebooks/AGENTS.md). These are part of an
   AI-agent-friendly **external-memory system** that documents the
   development process, conventions, and gotchas (`AGENTS.md` v1.1
   spec; OpenAI `PLANS.md` convention). They are intended for
   transparency, not as user-facing API; a casual contributor can
   ignore them and `CONTRIBUTING.md` will tell them everything they
   need.

6. **Notebook outputs are stripped.** A `nbstripout` pre-commit hook
   strips cell outputs on every commit. Don't rely on committed
   outputs; re-run the notebook to regenerate them. This is documented
   in `notebooks/AGENTS.md`.

7. **`flash-attn` is optional.** The pre-built CUDA 12.8 wheel works
   on most modern GPUs; if it fails to install on yours, drop
   `flash_attn` from your bench grid and rely on `sdpa` instead --
   the strategies are designed to be substitutable.

---

## Hardware requirements

### Model: `CohereLabs/tiny-aya-global`

| Property | Value |
|---|---|
| Parameters | 3.35 B (2.8 B non-embedding) |
| Architecture | Dense decoder-only, Cohere2 |
| Layers | 36 + embedding = 37 hidden states |
| Hidden dim | 2048 |
| Vocab | 262 K |
| Native dtype | BF16 |
| fp16 weight size | ~6.7 GB |
| int8 weight size | ~3.6 GB |
| int4 weight size | ~2.1 GB |

### VRAM with `output_hidden_states=True`

Theoretical estimates; measured peak on H100 80 GB was 70--76 GB for
quantized strategies due to activation caching across all 37 layers.

| Precision | bs 16, seq 512 | bs 32, seq 512 | bs 64, seq 512 |
|-----------|---------------:|---------------:|---------------:|
| fp16 | ~12 GB | ~18 GB | ~28 GB |
| int8 | ~8 GB  | ~12 GB | ~20 GB |
| int4 | ~6 GB  | ~9 GB  | ~15 GB |

### GPU recommendation

| GPU | VRAM | Full grid? | Notes |
|-----|------|-----------|-------|
| RTX 4090 | 24 GB | partial (bs ≤ 32 fp16) | dev / debug only |
| L40S | 48 GB | most configs | budget option |
| A100 40 GB | 40 GB | all but bs=64 fp16 | good enough |
| **A100 80 GB** | 80 GB | yes | **budget pick** |
| **H100 80 GB** | 80 GB | yes (faster compile) | **what we used** |

A complete benchmark run is 456 configs (9 strategies × 2 dtypes ×
5 batch sizes × 2 max-lengths, minus exclusions, × 3 repeats), about
~1.9 h on H100 80 GB.

---

## Detailed walkthrough

The canonical end-to-end step list (with copy-pasteable shell
commands and expected output for each step) lives in
**[`docs/RUNBOOK.md`](docs/RUNBOOK.md)**. The runbook covers
Steps 1--11: baseline `lid-infer`, `lid-bench` eager, vectorised,
`torch.compile`, SDPA / Flash Attention, INT8 / INT4, combined
strategies, the full 456-config grid, LoRA training on a fine-tuned
checkpoint, and plot generation. Below is a high-level orientation
for each phase so a newcomer can decide where to start.

### The two ways to run the LoRA pipeline

There are two execution paths through the same logic; pick whichever
matches your iteration speed.

| Path | Entrypoint | Best for |
|------|-----------|----------|
| **Single-strategy script** | `lid-infer` (with `--strategy`) | One-off runs, debugging, reproducing a specific number, generating `results.pkl` for `lid-visualize`. |
| **Grid bench** | `lid-bench configs/*.yaml` | Sweeping over strategy / dtype / batch-size / seq-length, recording variance across `repeat: N` repeats, logging each run to W&B. |

The grid bench is what produced
[`experiments/EXPERIMENT_LOG.md`](experiments/EXPERIMENT_LOG.md) (the
H100 80 GB run log). The single-strategy script is what
`docs/RUNBOOK.md` Step 1 starts with.

### Step-by-step orientation

1. **Baseline (`lid-infer`).** Runs the original triple-loop
   `extract_layer_probs` against `tiny-aya-global` and produces
   `results.pkl`, `layer_accuracy.csv`, `layer_avg_probs.csv`,
   `benchmark_row.csv`. The W&B run is auto-tagged `baseline`. This
   is the "1 ×" point everything else is compared against.
2. **Eager via bench.** Same logic through the bench framework so
   that timing numbers are directly comparable.
3. **Vectorised.** Replaces the triple loop with a single batched
   `lm_head` call + `torch.gather`. The single biggest optimisation,
   roughly 9--10 ×; it is what enables every later optimisation.
4. **Compiled.** Adds `torch.compile(mode="reduce-overhead")` on top
   of vectorised. Adds another ~1.5 × but pays a 30--60 s warmup
   compile cost on the first batch.
5. **Attention backends.** Compares `sdpa` and `flash_attn` --
   throughput is similar to vectorised but VRAM drops 40--60 %.
6. **Quantisation.** INT8 (≈50 % weight memory, no accuracy loss)
   and INT4 (≈75 % weight memory, ~1 % accuracy hit on this model).
7. **Combined.** `combined_flash_compiled` is the throughput winner
   on the recorded H100 run; `combined_flash_int8` is the
   memory-frugal pick.
8. **Full grid.** All nine strategies × two dtypes × five batch
   sizes × two seq lengths × three repeats, with exclusion rules
   for incompatible combos. About ~1.9 h on H100 80 GB.
9. **Training (`lid-train`).** LoRA fine-tune (default `r=16`,
   `alpha=32`) for 3 epochs with gradient accumulation. Drops the
   weights to `checkpoints/<name>/best/`.
10. **Re-benchmark.** Repeat steps 1--8 against the fine-tuned
    checkpoint to see how training shifts the layer-wise curve.
11. **Visualise (`lid-visualize`).** Per-layer accuracy curves,
    per-language confusion plots, and per-script confusion heatmaps
    saved to `<output-dir>/plots/`.

The optimisation-strategy design (problem analysis, architecture,
per-strategy mathematics, W&B integration strategy, and a metrics
taxonomy) is documented in
[`docs/optimization_spec.md`](docs/optimization_spec.md). The
genesis research questions, the extended publishability assessment,
and the forward-looking research roadmap are documented in
[`docs/proposal_original.md`](docs/proposal_original.md),
[`docs/project_proposal.md`](docs/project_proposal.md), and
[`docs/paperback.md`](docs/paperback.md) respectively (see
*Project genesis and document lineage* above).

---

## Strategy reference

| Name | Key | What it does | Expected speedup |
|------|-----|---|---|
| Eager | `eager` | Original triple-loop notebook code (baseline) | 1× |
| Vectorized | `vectorized` | Batched `lm_head` + `torch.gather`; zero Python loops | ~10× |
| Compiled | `compiled` | Vectorized + `torch.compile(mode="reduce-overhead")` | ~15× |
| SDPA | `sdpa` | Vectorized + PyTorch scaled dot-product attention | ~10× + 40 % less mem |
| Flash Attention | `flash_attn` | Vectorized + Flash Attention 2 kernel | ~10× + 60 % less mem |
| INT8 | `quantized_int8` | Vectorized + bitsandbytes 8-bit weights | ~10× + 50 % less mem |
| INT4 | `quantized_int4` | Vectorized + bitsandbytes 4-bit NF4 | ~10× + 75 % less mem |
| Flash + Compile | `combined_flash_compiled` | Flash Attention + torch.compile | ~15× + 60 % less mem |
| Flash + INT8 | `combined_flash_int8` | Flash Attention + INT8 quantization | ~10× + best mem |

A grid YAML schema and a "how to add a strategy" recipe live in
[`CONTRIBUTING.md`](CONTRIBUTING.md#recipes).

---

## W&B integration

`lid-bench` and `lid-infer` log to a W&B project (default `lid-bench`)
with the following surfaces:

| Feature | Logged data |
|---|---|
| `run.config` | strategy, dtype, batch_size, model, dataset, etc. |
| `run.log()` | per-batch latency, throughput, GPU memory (streaming) |
| `run.summary` | aggregate throughput, accuracy, energy, MFU |
| `wandb.Table` | per-layer accuracy curves, benchmark summary |
| `wandb.Artifact` | grid YAML, results CSV, profiler traces |
| System metrics | GPU util, power, temp, memory (auto every 15 s) |

To run without W&B (network-free local iteration), pass `--no-wandb`
to any CLI.

---

## Notebooks

The repository ships **five notebooks** under [`notebooks/`](notebooks/).
They form a "ladder" of LID approaches, from the simplest (regex over
Unicode blocks) up to the layer-wise compact-LM pipeline that the
rest of the repository optimises. All five run from a base
`make dev` install except where explicitly noted; cell outputs are
stripped on commit by the `nbstripout` pre-commit hook, so re-run
to populate them locally.

### `LID_Regex.ipynb`

> *Approach:* heuristic; per-character Unicode-block frequency.
>
> *Dependencies:* base only (`make dev`).

The first rung of the LID ladder. Counts how often each
[Unicode block](https://en.wikipedia.org/wiki/Unicode_block) appears
in a text, builds a per-language fingerprint, and classifies a new
text by comparing its block-distribution against those fingerprints.
The notebook walks through which blocks are *noise* (whitespace,
punctuation, math symbols), which blocks uniquely identify a
language family (e.g. Hangul Syllables → Korean, Tibetan → Tibetan),
and which language pairs the heuristic can and cannot disambiguate
(it cannot tell apart most Latin-script languages, which is exactly
why the next rung is needed). Useful as an interpretable
bootstrapping step and as a fast pre-filter for the more expensive
classifiers.

### `LID_Unicode_Blocks_Classifier.ipynb`

> *Approach:* Unicode-block features + light regex / classifier
> heads.
>
> *Dependencies:* base only (`make dev`).

A more disciplined version of the regex notebook. Treats the Unicode
block frequencies as a feature vector and trains a classifier on top.
The notebook reports that around 20 distinct languages can be
disambiguated at 99 %+ accuracy using script-level evidence alone
(any language whose script is not shared with another major language
in the dataset). Useful as a strong baseline and as a hierarchical
"script gate" before more expensive models are invoked. The first
half of [`docs/paperback.md`](docs/paperback.md) §10 (specifically E4
"Script-First Early-Exit Router") generalises this idea.

### `LID_Ngrams_Classifier.ipynb`

> *Approach:* character n-grams (1--5) + linear / softmax head.
>
> *Dependencies:* base only (`make dev`).

A character-n-gram bag-of-features classifier in the spirit of
[CLD3](https://github.com/google/cld3) /
[FastText langid](https://fasttext.cc/docs/en/language_identification.html)
-- the standard "shallow" baseline for any LID benchmark. The
notebook trains on 500 / 1 000 / 2 500 samples per language and
reports per-language accuracy. This is the right reference number
for comparing how much a transformer-based classifier actually
buys over an interpretable shallow model on the same training
budget.

### `LID_Embedding_Classifier.ipynb`

> *Approach:* `intfloat/multilingual-e5-large` embedding +
> classifier head, fine-tuned with **unsloth** + LoRA.
>
> *Dependencies:* `uv sync --extra unsloth` (extra group required).

A ~0.6 B-parameter multilingual encoder fine-tuned with LoRA via
unsloth. Reaches **macro F1 0.97+** on out-of-domain web text with
roughly 900 samples per language. This is the strongest baseline in
the repository and the most realistic competitor to a large
generative model on the LID task. The notebook contains three
non-obvious shims (a `torchvision.io.VideoReader` compat shim, a
`flex_attention` dropout-zeroing fix, and a memory-budget recipe
for ≤ 24 GB GPUs); each is documented in detail in
[`notebooks/AGENTS.md`](notebooks/AGENTS.md). It also demonstrates
the project's W&B logging pattern (per-language confusion tables,
artefact upload, summary-metric registration) end-to-end.

### `LID_Inference_Vibecoded.ipynb`

> *Approach:* layer-wise probability extraction from a compact
> generative LM (`tiny-aya-global`, 3.35 B params).
>
> *Dependencies:* base only (`make dev`).

The notebook that the rest of the repository was built to optimise.
Loads `tiny-aya-global`, processes 3 350 stratified samples, and for
each sample computes the per-language probability **at every one of
the 37 hidden states** (embedding layer + 36 transformer layers).
This is the "research signal" of the project: the layer-wise
accuracy curve tells you at what depth a language becomes
discriminable, and the per-script breakdown tells you which scripts
are early-decided versus deep-decided. The original notebook uses
the triple-nested Python loop that
[`docs/optimization_spec.md`](docs/optimization_spec.md) §1.1
identifies as the bottleneck; everything in `src/lid/bench/` exists
to replace that loop with progressively faster vectorised paths.

### Where the notebook ladder leads

The notebooks (heuristic → shallow → embedding → generative
layer-wise) are each interesting on their own and are also a small
ablation: how much accuracy do you gain by stepping up one rung?
The longer arc is in
[`docs/paperback.md`](docs/paperback.md) §10 -- it proposes twelve
experiments that build a layer-wise atlas across nine open-weight
models, identifies language neurons by activation patching, and
turns those mechanistic results into compression and routing
strategies. The notebooks are the existing artefact; the paperback
is the future direction.

---

## FAQ and troubleshooting

**"CUDA out of memory" during benchmark.** Drop batch size, switch to
a quantized strategy, or use `flash_attn` for ~40 % VRAM savings.

**`flash-attn` not installed.** Pre-built CUDA 12.8 wheels are
usually available: `uv pip install flash-attn --no-build-isolation`.
If the build fails, fall back to `sdpa` (built into PyTorch).

**`torch.compile` takes forever on the first batch.** Expected;
compilation is 30--60 s. Set `warmup_batches: 2` in the grid YAML to
exclude it from timing.

**W&B is not logging.** Check `WANDB_API_KEY` in `.env`, or
`uv run wandb login`. Pass `--no-wandb` to skip entirely.

**How do I add a new model / strategy / dataset?** See the **Recipes**
section of [`CONTRIBUTING.md`](CONTRIBUTING.md#recipes).

---

## Future research directions

This repository is a working artifact, not a proposal. If you want
to **extend** this work -- new mechanistic-interpretability
analyses, layer-wise atlases across model families, circuit-aware
compression, contamination audits, code-mix evaluation,
preference-optimisation variants, or elastic / early-exit
inference -- start with
[`docs/paperback.md`](docs/paperback.md). It is a forward-looking
research-direction document that lays out:

- a 9-model cohort across three open-weight families
  (Cohere Tiny Aya, Gemma 4, Qwen 3 / 3.5),
- twelve well-scoped experiments (E1--E12) with effort estimates
  in H100-hours,
- a literature map (~50 references from 2024--2026) covering
  mechanistic interpretability, layer-wise probing, quantisation,
  preference optimisation, and LID benchmarks,
- shared infrastructure modules under `src/lid/` to keep CLI
  ergonomics consistent with the existing tools, and
- open questions a future researcher should think through before
  committing to a particular slice.

The document is intentionally **venue-agnostic and
deadline-agnostic**: pick a slice that fits your compute budget,
publish at a venue (or via a HuggingFace dataset card / GitHub
release) that fits your timeline.

---

## Citation

If you use this software, please cite it via the GitHub
"Cite this repository" button (it reads from
[`CITATION.cff`](CITATION.cff)) or the BibTeX below:

```bibtex
@software{bhaskar_lid_2026,
  author       = {Bhaskar, Mayank},
  title        = {{LID -- Layer-Wise Multilingual Language Identification
                    in Compact Foundation Models}},
  version      = {0.1.0},
  year         = {2026},
  url          = {https://github.com/cataluna84/lid},
  license      = {Apache-2.0}
}
```

No paper is currently in submission. A future paper based on this
artifact is sketched in [`docs/paperback.md`](docs/paperback.md);
until/unless that paper exists, please cite the software entry
above. A DOI will be minted via the GitHub-Zenodo integration on
the `v0.1.0` release tag.

---

## License

Licensed under the **Apache License, Version 2.0** -- see
[`LICENSE`](LICENSE) for the full text and [`NOTICE`](NOTICE) for the
attribution requirements (Apache 2.0 §4(d)).

By submitting a pull request you license your contribution under
Apache-2.0 §5; no separate CLA is required. Newly added third-party
code must keep its upstream notices intact.

The repository was relicensed from MIT to Apache 2.0 at the time of
public release (sole-author project; relicensing was unilaterally
permissible per `git log`).

---

## Acknowledgments

- **Cohere Labs** for the `tiny-aya-global` model.
- **1-800-LLMs** for the `1024m/LID` hackathon dataset.
- **HuggingFace** for `transformers`, `datasets`, `accelerate`, `peft`.
- The **PyTorch** team for `torch.compile` and SDPA.
- **Tim Dettmers** for `bitsandbytes` (INT8 / INT4 quantization).
- The **unsloth** team (used in the embedding-classifier notebook).
- **Weights & Biases** for run tracking.

See [`NOTICE`](NOTICE) for the full attribution list and upstream
licenses.
