# DO — Layer-Wise Multilingual LID in Compact Foundation Models

> Research execution plan for **ARR May 2026** submission (long paper).
> Authors: Mayank Bhaskar et al. Version 0.1 (draft for colleague review).
> Last updated 2026-04-17.

> **Status.** This is a *planning* document, not an implementation log.
> Nothing here has been coded yet. The plan assembles (a) the completed
> Exa Deep Research synthesis (task id `r_01kpde4dn7tt1bhfd3175gajkw`),
> (b) targeted Exa/Ref literature queries from 2024--2026, and (c) the
> current repo state (`AGENTS.md`, `docs/project_proposal.md`,
> `docs/RUNBOOK.md`, `docs/optimization_spec.md`, `PLAN.md`).

---

## Table of Contents

1. [TL;DR](#1-tldr)
2. [Research Thesis](#2-research-thesis)
3. [Concrete Research Questions](#3-concrete-research-questions)
4. [Model Cohort (Tiny Aya + Gemma 4 + Qwen 3 / 3.5)](#4-model-cohort)
5. [Why This Cohort, Not Another](#5-why-this-cohort-not-another)
6. [Data Cohort](#6-data-cohort)
7. [Evaluation Protocol](#7-evaluation-protocol)
8. [Infrastructure, Compute Budget, Risks](#8-infrastructure-compute-budget-risks)
9. [Literature Map (2024--2026)](#9-literature-map-20242026)
10. [Experiments (E1--E12) — full scaffolding](#10-experiments-e1e12)
    * [E1 — Layer Emergence Atlas](#e1--layer-emergence-atlas)
    * [E2 — Language Neuron / Circuit Discovery](#e2--language-neuron--circuit-discovery)
    * [E3 — Layer-Aware Mixed-Precision Quantization](#e3--layer-aware-mixed-precision-quantization)
    * [E4 — Script-First Early-Exit Router](#e4--script-first-early-exit-router)
    * [E5 — Code-Mixed & Transliteration Robustness](#e5--code-mixed--transliteration-robustness)
    * [E6 — ORPO / β-DPO / SimPO for Closely Related Pairs](#e6--orpo--dpo--simpo-for-closely-related-pairs)
    * [E7 — Cross-Model Transferability of Layer Rankings](#e7--cross-model-transferability-of-layer-rankings)
    * [E8 — Contamination & Label-Noise Audit (CoDeC)](#e8--contamination--label-noise-audit-codec)
    * [E9 — LID-Circuit-Preserving Pruning](#e9--lid-circuit-preserving-pruning)
    * [E10 — Thinking-Budget / MatFormer Elastic Inference for LID](#e10--thinking-budget--matformer-elastic-inference-for-lid)
    * [E11 — Fine-Tuning Trajectories (H1/H2/H3 test)](#e11--fine-tuning-trajectories-h1h2h3-test)
    * [E12 — Metric Stress Test Beyond Macro-F1](#e12--metric-stress-test-beyond-macro-f1)
11. [Cross-Experiment Shared Infrastructure](#11-cross-experiment-shared-infrastructure)
12. [Dataset Splits, Cards, and Release Plan](#12-dataset-splits-cards-and-release-plan)
13. [Artifact and HF Hub Release Plan](#13-artifact-and-hf-hub-release-plan)
14. [Timeline (6 weeks to ARR deadline)](#14-timeline-6-weeks-to-arr-deadline)
15. [Paper Outline](#15-paper-outline-arr-long-paper-8--unlimited-refs)
16. [Reproducibility Checklist](#16-reproducibility-checklist)
17. [Open Questions for Review](#17-open-questions-for-review)
18. [References (short, 2024--2026)](#18-references-short-20242026)

---

## 1. TL;DR

We run a **single, coherent mechanistic-interpretability + empirical-systems
study** of how **language identification** emerges across depth in compact
foundation models (≈0.6B--8B parameters), then *causally* intervene on that
picture with (i) per-layer quantization, (ii) circuit-preserving pruning,
(iii) script-first early exit, (iv) preference optimization for
closely-related language pairs, and (v) elastic-depth inference
(MatFormer, thinking-budget). The contribution is a **unified atlas
across three open architecture families** released in Feb--Apr 2026
(Tiny Aya, Gemma 4 E2B/E4B/26B-A4B/31B, Qwen 3 0.6B--32B dense + Qwen 3.5
0.8B--35B-A3B), a new **confusion-weighted, per-script-calibrated**
evaluation suite, and an **interpretability-guided compression recipe**
with 30--50% inference speedup and no F1 loss on 67-language LID, across
scripts, and under code-mix / transliteration stress tests.

**Unique single-sentence angle:** *Per-layer quantization and
layer-level circuit interventions jointly control low-resource and
code-mixed LID behavior in mid-sized compact foundation models, and we
demonstrate (1) where LID signals form across depth, (2) which circuits
are quantization-sensitive across scripts, and (3) how targeted
layer-preserving compression or routing can restore LID fidelity —
a synthesis not offered by dataset-centric frameworks (GlotLID,
OpenLID-v3, UniLID, LIMIT, CommonLID) or by single-model mechanistic
studies.*

**Estimated compute:** ~523 H100-hours end-to-end for E1--E12 at the
current scope (see §8.3).

---

## 2. Research Thesis

Compact foundation models (0.6B--8B) have *identifiable*, *causally
necessary*, and *quantization-sensitive* **language-discriminative
circuits** concentrated in a mid-to-late layer band. Mapping this band
(per language, per script, per model family) yields:

1. An **interpretability-guided compression recipe** (per-layer
   quantization + circuit-aware pruning) that preserves LID accuracy
   across all 67 Tiny Aya languages and three benchmark suites
   (CommonLID, GlotLID, OpenLID-v3).
2. A **script-first early-exit router** that achieves a measurable
   accuracy/latency Pareto improvement on non-Latin inputs where
   current compact models fail disproportionately.
3. A **cross-family portability result** — the layer band transfers
   (with an explicit map) between Tiny Aya, Gemma 4, and Qwen 3 / 3.5,
   providing the first empirical evidence that mechanistic LID
   findings are architecture-robust in compact foundation models.

This is publishable as a *measurement-first*, *method-second* long paper
with an explicit experimentation-driven narrative.

---

## 3. Concrete Research Questions

Renumbered, disambiguated, and mapped to experiments. Originals from
`docs/project_proposal.md` §Research Questions are preserved; five new
RQs (RQ12–RQ16) are added to cover the expanded model cohort and
mechanistic interventions.

| ID | Question | Experiments |
|---|---|---|
| RQ1 | Does quantization level (FP16→INT8→INT4→1.58-bit) degrade 67-class LID after LoRA? | E3, E10 |
| RQ2 | Does same-script confusion differ from across-script confusion? | E5, E6, E12 |
| RQ3 | Does ORPO/SimPO/β-DPO between same-script languages reduce confusion with acceptable side-effects? | E6 |
| RQ4 | Does a script-first hierarchical router beat flat 100-class classifiers? | E4 |
| RQ5 | Can depth be reduced (layer pruning / early-exit) without F1 loss? | E4, E9 |
| RQ6 | How does layer-wise accuracy evolve across depth per language family? | E1 |
| RQ7 | How does layer-wise accuracy evolve across *training steps*? | E11 |
| RQ8 | Are early layers script encoders and late layers linguistic discriminators? | E1, E2 |
| RQ9 | Does quantization disproportionately hurt closely-related language pairs? | E3 |
| RQ10 | How stable are per-language confidence distributions across layers and checkpoints? | E1, E11 |
| RQ11 | How does model behaviour differ between short and long inputs for 100-class LID? | E5, E12 |
| **RQ12** | Are LID circuits identifiable and transferable across Tiny Aya / Gemma 4 / Qwen 3 / Qwen 3.5? | E2, E7 |
| **RQ13** | Can MatFormer's elastic-inference property (Gemma 4 E2B/E4B) be used as a cheap LID early-exit? | E10 |
| **RQ14** | Does Qwen 3's thinking-budget mechanism interact meaningfully with LID (a "no-think" task)? | E10 |
| **RQ15** | Does preference optimization (ORPO / β-DPO / SimPO / FocalPO) help compact models on Serbian/Croatian, Malay/Indonesian, Hindi/Urdu (transliterated)? | E6 |
| **RQ16** | How contaminated are the current LID benchmarks, and does contamination bias our plateau-layer estimates? | E8 |

---

## 4. Model Cohort

All three families are Apache-2.0 open-weight; this is a hard
constraint for an ARR-reproducible submission. Shaded rows are the
**training targets**; unshaded rows are **inference-only baselines**.

### 4.1 Tiny Aya family (baseline — Cohere Labs)

| Model | Params | Tokenizer | Layers | Hidden | Status |
|---|---|---|---|---|---|
| TinyAya-Global (primary) | ~1.0B | Aya multilingual BPE | 37 hidden states extracted | 2048 | LoRA fine-tune |
| TinyAya-Fire / Water / Earth | ~1B each | same | same | same | eval only |

Rationale: current repo baseline, LoRA training pipeline already
validated (`lid-train` CLI), 37-layer extraction confirmed in
`docs/optimization_spec.md` §1.1.

### 4.2 Gemma 4 family (Google, released 2026-04-02, Apache 2.0)

| Model | Total params | Effective / Active | Arch | Ctx | Notes |
|---|---|---|---|---|---|
| Gemma-4-E2B-it | 5.1B | 2B effective (PLE) | Dense + PLE | 128K | MatFormer-elastic, on-device |
| Gemma-4-E4B-it | 8.0B | 4B effective (PLE) | Dense + PLE | 128K | MatFormer-elastic |
| Gemma-4-26B-A4B-it | 26B | 3.8B active | MoE | 256K | hybrid sliding/global attn |
| Gemma-4-31B-it | 31B | 31B dense | Dense | 256K | LMArena Elo ~1338 |

Key features we will exploit:

* **Per-Layer Embeddings (PLE)**: each decoder layer owns a small
  embedding tied to tokens. Huge embedding matrix but compute-light.
  Directly relevant to layer-wise analysis — LID signals may be
  contributed by these PLE lookups, not residual MLP outputs.
* **MatFormer (nested submodel)**: E2B is literally a valid sub-model
  of E4B. For RQ13 we run "elastic LID": extract E2B out of E4B
  on-the-fly and measure LID degradation.
* **Hybrid attention + shared KV cache**: sliding-window local (512--1024
  tokens) interleaved with global attention every few layers. Relevant
  to short-vs-long LID (RQ11) because local-attention layers will form
  LID signals from sub-sentence context only.
* **140+ languages pretraining**: strictly more than Tiny Aya (~67);
  must align the language subset for a fair comparison.

We *train* on E2B and E4B (fits comfortably on H100 80GB under LoRA).
We *eval-only* the 26B-A4B MoE (fits in bfloat16 on a single H100
80GB — 48 GB weights per spec) and *skip* 31B dense for main
experiments (marginal over 26B-A4B for LID, doubles compute).

### 4.3 Qwen 3 family (Alibaba, released 2025-05, Apache 2.0)

| Model | Total params | Active | Arch | Ctx | Notes |
|---|---|---|---|---|---|
| Qwen3-0.6B | 0.6B | full | dense | 32K | smallest open foundation model |
| Qwen3-1.7B | 1.7B | full | dense | 32K | |
| Qwen3-4B | 4B | full | dense | 32K | sweet spot |
| Qwen3-8B | 8B | full | dense | 32K | |
| Qwen3-14B | 14B | full | dense | 32K | |
| Qwen3-32B | 32B | full | dense | 32K | |
| Qwen3-30B-A3B | 30B | 3B | MoE | 128K | 128 experts, 8 active |
| Qwen3-235B-A22B | 235B | 22B | MoE | 128K | flagship, **out-of-scope** |

Key features:

* **Unified thinking / non-thinking modes** via `/think` and
  `/no_think` chat templates. Natively supports a **thinking-budget
  mechanism** letting users cap reasoning tokens. LID is a "no-think"
  task in principle — RQ14 tests whether thinking helps or hurts.
* **119 languages** (up from 29 in Qwen2.5) including Vietnamese,
  Tamil, Kazakh, Esperanto, Welsh, Haitian-Creole — covers all 67
  Tiny Aya languages.
* **Grouped Query Attention + SwiGLU + RoPE + QK-Norm** (no QKV bias)
  — identical attention family to Gemma 4 → clean cross-family
  layer-rank comparison.

We *train* Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B under LoRA. We
*eval-only* Qwen3-8B and Qwen3-30B-A3B (MoE). 14B/32B/235B are
out-of-scope for the compact-models focus.

### 4.4 Qwen 3.5 family (Alibaba, released 2026-02, Apache 2.0)

| Model | Total params | Active | Arch | Ctx | Notes |
|---|---|---|---|---|---|
| Qwen3.5-0.8B | 0.8B | full | dense + Gated DeltaNet | 262K | multimodal backbone |
| Qwen3.5-2B | 2B | full | dense + Gated DeltaNet | 262K | |
| Qwen3.5-4B | 4B | full | dense + Gated DeltaNet | 262K | |
| Qwen3.5-9B | 9B | full | dense + Gated DeltaNet | 262K | |
| Qwen3.5-27B | 27B | 27B dense | dense + Gated DeltaNet | 262K | |
| Qwen3.5-35B-A3B | 35B | ~3B | sparse MoE + Gated DeltaNet | 262K | beats Qwen3-235B on most evals |
| Qwen3.5-122B-A10B | 122B | 10B | sparse MoE | 262K | **out-of-scope** |
| Qwen3.5-397B-A17B | 397B | 17B | sparse MoE | 1M (hosted) | **out-of-scope** |

Key features:

* **Gated DeltaNet** attention (linear attention variant) replaces
  quadratic attention in many layers — RQ1/RQ11 gain a *different*
  attention family for the ablation, controlling for attention
  algorithm as a confound. First study to compare Gated DeltaNet vs.
  softmax attention for LID signal formation.
* **Early fusion multimodal** backbone (trained jointly with vision
  tokens even at the text-only sizes). Residual stream geometry may
  differ — testable hypothesis.
* **201 languages** (superset of Qwen 3's 119).
* **262K--1M context** for essentially arbitrary long-input LID.

We *train* Qwen3.5-0.8B, 2B, 4B. We *eval-only* the 9B, 27B, and
35B-A3B. Everything larger is out-of-scope.

### 4.5 Consolidated training-target matrix

| Family | 0.5-1B | 1-2B | 2-5B | 5-9B | 10-35B |
|---|---|---|---|---|---|
| Tiny Aya | Global (1B) ✅ | | | | |
| Gemma 4 | | | E2B (5.1B/2B) ✅ | E4B (8B/4B) ✅ | eval-only 26B-A4B |
| Qwen 3 | 0.6B ✅ | 1.7B ✅ | 4B ✅ | eval-only 8B | eval-only 30B-A3B |
| Qwen 3.5 | 0.8B ✅ | 2B ✅ | 4B ✅ | eval-only 9B | eval-only 27B, 35B-A3B |

Nine trained checkpoints × three training recipes (plain LoRA, LoRA +
ORPO on same-script pairs, LoRA + quantization-aware) = 27 training
runs. See §8 for the H100-hour budget.

---

## 5. Why This Cohort, Not Another

* **Llama-4-Scout** — too large and MoE-only, no small sizes.
* **Phi-4** — English-centric pretraining, coverage <67 Tiny Aya
  languages.
* **Mistral / Mixtral** — limited multilingual coverage, older
  attention (no QK-Norm).
* **mT5 / XLM-RoBERTa** — encoder-only, not foundation models; we
  already use them as *baselines* (`intfloat/multilingual-e5-large` in
  the embedding classifier), not as subjects of mechanistic study.

The chosen cohort gives us three *different* attention variants
(softmax + QK-Norm for Qwen 3 and Tiny Aya; hybrid sliding/global for
Gemma 4; Gated DeltaNet for Qwen 3.5) at comparable sizes with
overlapping language coverage (≥120 languages each ⊇ the Tiny Aya 67).
This is a controlled, publishable factorial.

---

## 6. Data Cohort

Datasets augmenting the five already listed in `docs/project_proposal.md`.
Every addition is a **2024--2026 release** and selected specifically
for a research question.

| Dataset | Size / Scope | Purpose | Used in |
|---|---|---|---|
| 1024m/LID (Hackathon) | ~3.35k samples, 67 langs | train/eval baseline | E1, E3, E4 |
| Cohere-LID-Data | 500k+ samples | in-distribution eval | E1, E8 |
| Wikimedia Wikipedia | variable | clean reference corpus | E1, E8 |
| XL-Sum | ~1M news sentences | in-distribution news | E1 |
| HC Corpora newspapers | ~4M items | out-of-distribution web | E1, E12 |
| **CommonLID 2024** | 500 langs, web-scraped | contamination + noise stress | E8, E12 |
| **DCAD-2000 2025** | 2,282 langs | extreme long-tail, script diversity | E8 (audit) |
| **Mozilla Common Voice LID** | 300+ langs | transcribed speech→text LID | E5 |
| **ILID 2025** (Indian LID) | 22 Indic langs + Roman-script variants | code-mix, transliteration | E5, E6 |
| **GlotLID dev** | 1665 langs | low-resource stress | E1, E12 |
| **OpenLID-v3** | ~200 closely-related pairs | ORPO/β-DPO targets | E6 |
| **FLEURS** | 102 langs speech | out-of-domain sanity | E1 (excl.) |
| **Belebele** | 122 langs reading comp. | cross-task transfer control | E7 |
| **SIB-200** | 203 langs topic classification | cross-task transfer control | E7 |

**Curated code-mix / transliteration splits (to be built, E5):**

| Name | Contents |
|---|---|
| `cmlid-roman-hi-ur` | transliterated Hindi vs. Urdu |
| `cmlid-sr-hr` | Serbian Latin vs. Croatian Latin |
| `cmlid-ms-id` | Malay vs. Indonesian |
| `cmlid-hi-en-code-mix` | Hinglish code-switched |
| `cmlid-ta-en-code-mix` | Tamil-English code-switched |
| `cmlid-sw-en-code-mix` | Swahili-English code-switched |

All six are curated from ILID, XLM-V, and Mozilla CV transcripts and
released as a **LID-mix-67-codemix-v1** dataset card under CC-BY-SA.

---

## 7. Evaluation Protocol

Metrics go **beyond** macro/micro/weighted F1 (see §9.4 for literature
justification). For each run we report the full grid below.

### 7.1 Tier-1 (mandatory, every experiment)

| Metric | Notes |
|---|---|
| macro-F1 | primary headline |
| weighted-F1 | recall-balanced |
| per-language F1 | 67 values |
| same-script confusion matrix | sum of off-diagonal within each script bucket |
| last-layer top-1 accuracy | baseline comparator |
| layer-wise accuracy curve | array of 37 values (Tiny Aya) up to 48 (Qwen3-4B) |
| NLL (negative log-likelihood) | calibration proxy |

### 7.2 Tier-2 (where relevant)

| Metric | Notes |
|---|---|
| **per-script calibration error (PSCE)** | ECE within each script bucket — detects quantization-induced miscalibration |
| **confusion-weighted F1 (CW-F1)** | error weighted by linguistic proximity (WALS or Glottolog distance) |
| **cross-lingual auto-evaluator (CIA-Suite)** | LLM-as-judge for free-form LID outputs |
| **contamination score (CoDeC)** | per-language leakage estimate — adjusts reported F1 |
| **information-theoretic MI bits/token** | probe strength normalised across tokenisers |
| ROC-AUC / PR-AUC | per-language, one-vs-rest |

### 7.3 Tier-3 (efficiency)

Carry the 3-tier metrics already defined in
`docs/optimization_spec.md` §5: throughput, GPU memory, energy per
sample, MFU, arithmetic intensity, profiler trace artifact.

### 7.4 Input-length stratification

Every evaluation is bucketised into 4 length bins:

| Bin | Char range | Token range (approx) |
|---|---|---|
| tiny | 1--20 | 1--6 |
| short | 21--60 | 6--20 |
| medium | 61--200 | 20--60 |
| long | 201--∞ | 60--∞ |

---

## 8. Infrastructure, Compute Budget, Risks

### 8.1 Hardware

* Primary: 1× NVIDIA H100 80GB (team account)
* Budget: 1× NVIDIA A100 80GB (fallback)
* Secondary: L40S 48GB (small-model ablations)
* CI: 4-vCPU runner for unit tests only

### 8.2 Software stack

Already pinned in `pyproject.toml`:

* Python 3.12, PyTorch 2.11 CUDA 12.8, unsloth, transformers,
  peft, trl (for ORPO/DPO), bitsandbytes, scipy
* **Additions for this plan:**
  * `transformer-lens` — logit-lens, activation patching
  * `nnterp` — standardised MI interface for attention/MLP
  * `mech-interp-toolkit` (ICI-Innolabs) — auto-circuit discovery
  * `captum` — attribution
  * `harmonic` or `glottolog` — typological distance for CW-F1
  * `zeno-build` — evaluation dashboards (optional)

Route all new deps through `uv add` (per `AGENTS.md` Don't-list).
No manual edits to `pyproject.toml` dep lists.

### 8.3 Compute budget (E1–E12)

Order-of-magnitude only — refined after pilot.

| # | Experiment | H100-hours | Notes |
|---|---|---|---|
| E1 | Layer Emergence Atlas | 40 | 9 models × 5 datasets × 1 pass |
| E2 | Circuit Discovery | 60 | patching-heavy, 3 models |
| E3 | Quantization sweep | 50 | 9 models × 3 bit configs × LoRA |
| E4 | Early-exit router | 30 | 3 models, policy training |
| E5 | Code-mix / translit | 45 | 6 datasets, 9 models |
| E6 | ORPO / β-DPO / SimPO | 35 | 3 models × 3 objectives |
| E7 | Cross-model transfer | 25 | rank-corr. analysis |
| E8 | Contamination audit | 20 | CPU-heavy, small GPU |
| E9 | LID-circuit pruning | 60 | 3 models × 3 targets |
| E10 | MatFormer + thinking-budget | 40 | Gemma 4 E4B, Qwen 3 |
| E11 | Training trajectories | 70 | extra checkpoints |
| E12 | Metric stress test | 18 | eval-only |
| Buffer | contingency | 30 | |
| **Total** | | **~523** | ~22 days on 1× H100 |

### 8.4 Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Gemma-4 weights gated or unavailable on HF at run time | medium | high | fall back to Gemma-3 4B / 1B (already in `project_proposal.md`); code paths detect weight absence and skip gracefully |
| Qwen 3.5 backbone Gated DeltaNet unsupported in transformers 4.x | medium | medium | pin compatible commit, add a `qwen35` skip-flag; contribute a PR upstream |
| `torch.compile` + LoRA + PLE incompatible | medium | medium | fall back to eager for PLE layers only |
| CoDeC API availability | low | low | re-implement CoDeC locally (~300 LOC) |
| LMArena Elo changes invalidate narrative | low | low | framing is mechanistic, not leaderboard-based |
| Pretraining contamination invalidates *all* layer-atlas claims | low-med | high | run E8 first; if contamination > 5% for a language, exclude from atlas |
| MoE routing interferes with layer probing (non-deterministic expert choice) | medium | medium | freeze router during eval, log expert-choice entropy, average over N=10 repeats |
| Unsloth does not yet support Gemma 4 / Qwen 3.5 | medium | medium | fall back to `transformers` with `bitsandbytes` quantization path |

---

## 9. Literature Map (2024--2026)

### 9.1 Mechanistic interpretability

* **Circuit Tracing** (Anthropic 2025) —
  `transformer-circuits.pub/2025/attribution-graphs/methods.html` —
  cross-layer transcoder + attribution graphs.
* **Triangulation** (arXiv 2512.24842) — rigorous acceptance rules for
  circuit claims via invariance across prompts, languages, domains.
* **Language Arithmetics** (ACL/IJCNLP 2025) — direct identification
  and *editing* of "language neurons" in multilingual LLMs.
* **Locate, Steer, Improve** (arXiv 2601.14004v4) — actionable MI
  survey for downstream steering.
* **Mechanistic Understanding & Mitigation** (arXiv 2505.16538v2) —
  neuron-edit mitigation of multilingual failures.
* **MechaMap**, **nnterp**, **mech-interp-toolkit**, **InterpBench
  NeurIPS 2024** — toolkits.

### 9.2 Layer-wise probing and depth

* **Layer by Layer** (Skean et al. 2025, arXiv 2502.02013) — mid-layer
  peaks of prediction quality; justifies our E1 atlas.
* **Generalization Ridge / InfoRidge** (arXiv 2507.05387) —
  information-theoretic peaks of generalizable signal.
* **Early Exit Is a Natural Capability** (2412.01455) —
  self-predictive depth; justifies E4.
* **LayerSkip** (Elhoushi et al. 2024) — depth-pruning technique.
* **Do Transformers Use Their Depth Adaptively?** (arXiv 2604.12426v1)
  — independent evidence for adaptive depth.
* **Tuned Lens** (Belrose 2023, maintained) — unified lens tooling.

### 9.3 Quantization + multilingual

* **How Does Quantization Affect Multilingual LLMs?** (Marchisio
  et al. 2024, arXiv 2407.03211) — non-Latin scripts lose
  disproportionately.
* **BitNet 1.58-bit** (arXiv 2402.17764) — extreme ternary
  quantization as a reference point.
* **Layer-wise Information-Effectiveness for PTQ in Small LMs**
  (ResearchGate 394321992) — LieQ per-layer scoring motivates E3.
* **GPTQ**, **AWQ**, **Hqq** — PTQ baselines.

### 9.4 Evaluation (beyond F1)

* **GlotEval** (ACL 2025 Demos, `aclanthology.org/2025.emnlp-demos.43`)
  — massively multilingual test suite.
* **CIA Suite** (ACL 2025, `2025.acl-long.1419`) — cross-lingual auto-
  evaluation via LLM-as-judge (HERCULE).
* **Form & Meaning in Intrinsic Multilingual Evaluations** (EACL 2026,
  `2026.eacl-long.113`) — bits/token corrected for form/meaning drift.
* **DebateBias** (arXiv 2511.01187) — bias evaluation in generative
  multilingual output.

### 9.5 Dataset / contamination

* **CoDeC** (arXiv 2510.27055v1) — contamination detection via
  in-context learning.
* **Does Data Contamination Detection Work?** (NAACL 2025 Findings,
  `2025.findings-naacl.291`).
* **Awesome Data Contamination** (`github.com/lyy1994/awesome-data-contamination`).
* **CommonLID blog** (CommonCrawl, 2024) — shows state-of-the-art LID
  accuracy drops 10--20 points under realistic web noise.

### 9.6 Low-resource, code-mix, transliteration

* **ILID** (arXiv 2507.11832v3) — Indian-script native LID dataset.
* **Sentiment Analysis for Code-Mixed Low-Resource Languages**
  (Springer SNAM 2026) — systematic review.
* **IndicST** (Krutrim, 2025) — Indic speech/text corpora.

### 9.7 Preference optimization

* **ORPO** (Hong & Thorne 2024, arXiv 2403.07691v2) — reference-free
  preference optimization.
* **β-DPO** (NeurIPS 2024) — adaptive β-parameter.
* **SimPO** (Meng et al. 2024) — reference-free, length-normalized.
* **FocalPO** — focal-loss style hard-example emphasis.

### 9.8 Script-aware routing / MoE

* **Multilingual Routing in MoE** (arXiv 2510.04694) — experts
  specialise per script.
* **LANGSAMP** (ACL 2025 Long, `2025.acl-long.88`) — script-aware
  pretraining.

### 9.9 Model family references

* **Gemma 4** blog — Google DeepMind, 2026-04-02
  (`blog.google/innovation-and-ai/technology/developers-tools/gemma-4/`).
* **Gemma 4 model card** — `ai.google.dev/gemma/docs/core/model_card_4`.
* **MatFormer** — Devvrit et al. NeurIPS 2024.
* **Qwen 3 technical report** — arXiv 2505.09388 (`emergentmind.com/papers/2505.09388`).
* **Qwen 3.5 blog** — `qwen.ai/blog?id=qwen3.5`, Feb 2026.
* **Qwen 3.5 GitHub** — `github.com/QwenLM/Qwen3.5`.

---

## 10. Experiments (E1--E12)

Each experiment specifies (a) question, (b) method, (c) baseline,
(d) novelty vs. 2024--2026 literature, (e) compute, (f) acceptance
criterion, (g) file paths / CLI commands to add. Paths reuse the
existing `src/lid/` package structure per `docs/optimization_spec.md` §7.

### E1 — Layer Emergence Atlas

**Question (RQ6, RQ8, RQ10, RQ12).** At which layer does
linearly-separable LID information first appear for each of the 67
languages, and how does that layer vary by language family and by
architecture family?

**Method.**

1. For each of the 9 training-target models, extract all hidden
   states (37--48 of them) on the 67-language eval set using the
   existing `batched_layer_text_outputs()` pipeline (vectorised —
   `docs/optimization_spec.md` §3.1).
2. Train 3 probe families at each layer:
   * **linear probe** (logistic regression) — baseline.
   * **tuned lens** (Belrose 2023) — learned linear readout into
     unembedding space.
   * **logit lens** (unembedding without learning) — reference.
3. Compute per-layer macro-F1 per language; fit a piecewise-linear
   curve and extract the **plateau layer** (layer at which F1
   reaches ≥ 95% of its maximum).
4. Report plateau-layer histogram by (language, script, family).

**Baseline.** 0.6B embedding classifier at macro-F1 ≈ 0.97 — the
current `LID_Embedding_Classifier.ipynb` result.

**Novelty.** First per-language, per-architecture atlas across 9
compact models from 3 families. Answers the core project RQs with
modern models released post-Feb 2026.

**Compute.** ~40 H100-hours.

**Acceptance.**

* `experiments/e1/atlas.parquet` with columns
  `(model, layer, language, probe, macro_f1, ...)`.
* Plot `experiments/e1/plateau_heatmap.pdf` showing language × family.

**Files to add.**

```
src/lid/mi/probes.py            # LinearProbe, TunedLens, LogitLens
src/lid/mi/atlas.py             # build_atlas(model, dataset, layers)
configs/e1_atlas.yaml
```

**CLI.**

```bash
uv run lid-atlas configs/e1_atlas.yaml --output experiments/e1/
```

### E2 — Language Neuron / Circuit Discovery

**Question (RQ8, RQ12).** Where are the *causally necessary* language
neurons in Tiny Aya, Gemma 4 E4B, and Qwen 3.5-4B, and do they
overlap across families?

**Method.**

1. Start from E1's plateau layers per language.
2. Run **activation patching** across the residual stream at each
   layer to identify positions necessary for correct LID
   (circuit-tracing per Anthropic 2025 protocol).
3. For MLP outputs, run **neuron attribution** (integrated gradients
   in `transformer-lens` or `captum`) to rank neurons by causal
   necessity per language.
4. Apply **triangulation** (arXiv 2512.24842) criteria: a neuron is
   accepted as a "language neuron" iff its causal role replicates
   across ≥ 3 domains (Wikipedia, XL-Sum, HC) and ≥ 2 probe
   architectures.
5. Quantify **overlap across families**: Jaccard index of accepted
   neuron indices for matched languages.

**Baseline.** Unicode-block regex (classifies 20+ languages at 99% on
script alone — `docs/project_proposal.md` Approach 3).

**Novelty.** First cross-family triangulated language-neuron map in
compact foundation models (<10B).

**Compute.** ~60 H100-hours (patching is expensive).

**Acceptance.**

* ≥ 10 triangulation-accepted language neurons per language for at
  least 50 / 67 languages on Tiny Aya.
* Jaccard overlap (Tiny Aya ↔ Qwen 3.5-4B) non-trivial (>0.1).

**Files to add.**

```
src/lid/mi/patching.py          # activation patching
src/lid/mi/triangulation.py     # acceptance criterion
configs/e2_circuits.yaml
```

### E3 — Layer-Aware Mixed-Precision Quantization

**Question (RQ1, RQ9).** Can we set per-layer bit-widths (e.g. early
layers FP16, mid-layers INT8, late layers INT4) to preserve LID macro-F1
while reducing VRAM below the uniform 4-bit baseline?

**Method.**

1. Score every layer by an **information-effectiveness** metric
   (LieQ-style, ResearchGate 394321992): layer sensitivity defined as
   ‖Δlogits/Δbit_width‖ on a calibration set of 1k LID samples.
2. Formulate as an ILP: minimise total bits s.t. macro-F1 ≥ baseline
   − 0.5 percentage points.
3. Solve for 3 models (Tiny Aya, Gemma 4 E4B, Qwen 3 4B).
4. Evaluate on the 67 languages, splitting by script group (Latin,
   Cyrillic, Arabic, Devanagari, CJK, Other).

**Baseline.** Uniform INT4 (AWQ), uniform INT8, BitNet 1.58-bit.

**Novelty.** First *LID-task-aware* per-layer quantization policy for
multilingual compact models. Marchisio 2024 only studies uniform
quantization.

**Compute.** ~50 H100-hours.

**Acceptance.** ≥ 10% VRAM reduction over uniform INT4 at ≤ 0.5 pp
macro-F1 loss, confirmed across 2 of the 3 tested models.

**Files to add.**

```
src/lid/bench/strategies/layerwise_quant.py
src/lid/bench/layer_sensitivity.py
configs/e3_layerwise_quant.yaml
```

### E4 — Script-First Early-Exit Router

**Question (RQ4, RQ5, RQ13).** Does a two-stage classifier
(regex-based script router + shallow-layer probe + deep-layer
fallback) Pareto-dominate a flat 67-way classifier on accuracy,
latency, and VRAM?

**Method.**

1. **Stage A (cheap):** regex Unicode-block classifier from
   `notebooks/LID_Regex.ipynb`. Outputs a script label and a
   confidence.
2. **Stage B (shallow):** for ambiguous script groups (Latin,
   Cyrillic, Arabic, Devanagari), run the model to plateau-layer
   only (from E1). Exit on max-class probability ≥ τ.
3. **Stage C (deep):** for inputs still ambiguous after Stage B, run
   the full model.
4. Tune thresholds on a validation split; report on test.

**Baseline.** Flat 67-way classifier at the last layer (current
production pipeline).

**Novelty.** Prior early-exit work is language-agnostic. We use LID
as a motivating task for the first *script-aware* early-exit policy.
This is also a direct test of MatFormer's elastic-inference claim
(sub-model Gemma 4 E2B extracted from E4B behaves like a Stage-B
exit for us).

**Compute.** ~30 H100-hours.

**Acceptance.** ≥ 2× median-latency speedup on the test set at ≤ 0.5
pp macro-F1 loss, across 3 models.

**Files to add.**

```
src/lid/router/script_first.py
src/lid/router/policy.py
configs/e4_early_exit.yaml
```

### E5 — Code-Mixed & Transliteration Robustness

**Question (RQ2, RQ11).** How well do compact models classify the
curated code-mix and transliteration splits, and which layers/neurons
from E2 are responsible for failures?

**Method.**

1. Build 6 curated code-mix / transliteration splits (see §6).
2. Run the full cohort; report per-split macro-F1 under tiny / short
   / medium / long buckets.
3. For the top 10 errors per split, run activation patching from E2
   to identify which layers *would have* classified correctly.
4. Report "silent-failure layers": layers where a correct answer is
   present but the final layer overrides it.

**Baseline.** Embedding classifier (`multilingual-e5-large`
fine-tuned) — currently our best unseen-domain baseline (F1 ≥ 0.97
per `docs/project_proposal.md`).

**Novelty.** Layer-level failure analysis for code-mix is novel;
prior work stops at macro-F1.

**Compute.** ~45 H100-hours.

**Acceptance.** For ≥ 3 of 6 splits, show that a specific layer range
(e.g., layers 20--24 of Gemma 4 E4B) carries a >10 pp F1 boost over
the last layer.

**Files to add.**

```
src/lid/data/codemix.py         # dataset loader + card
src/lid/mi/silent_failure.py    # analyses
configs/e5_codemix.yaml
```

### E6 — ORPO / β-DPO / SimPO for Closely Related Pairs

**Question (RQ3, RQ15).** Does preference optimization reduce
confusion between closely-related same-script language pairs without
degrading unrelated pairs?

**Method.**

1. Define 12 closely-related pairs using Glottolog distance (e.g.
   Serbian/Croatian, Malay/Indonesian, Ukrainian/Russian, Hindi/Urdu,
   Hausa/Nigerian-Pidgin, Dari/Farsi, Czech/Slovak, Bosnian/Serbian,
   Norwegian-Bokmål/Danish, Portuguese-BR/Portuguese-PT, Japanese
   (Shift-JIS) / Mandarin, Tamil/Malayalam).
2. Generate **contrastive preference pairs**: (`chosen`, `rejected`)
   = (`correct language tag`, `confusable language tag`) given an
   input. 5k pairs per language pair.
3. Train 3 objectives per (model, pair): ORPO, β-DPO, SimPO.
   Standard LoRA adapter on top of the LID-trained adapter.
4. Evaluate for (a) target-pair confusion reduction, (b) *unintended*
   performance shift on the 10 unrelated language pairs (negative
   transfer).

**Baseline.** Vanilla LoRA classification head — the current
`lid-train` flow.

**Novelty.** Systematic ORPO/β-DPO/SimPO comparison for
*fine-grained classification* (not generation). Prior work: SynPO for
captioning (2506.00835), FocalPO — nothing for LID.

**Compute.** ~35 H100-hours.

**Acceptance.** For at least one of (ORPO / β-DPO / SimPO), reduce
target-pair confusion by ≥ 20% relative with < 1% absolute macro-F1
drop elsewhere.

**Files to add.**

```
src/lid/train/preference.py     # ORPO/β-DPO/SimPO wrappers on top of trl
src/lid/data/preference_pairs.py
configs/e6_preference.yaml
```

### E7 — Cross-Model Transferability of Layer Rankings

**Question (RQ12).** Given a per-layer importance ranking (from E1)
on Tiny Aya, how well does that ranking predict importance on Gemma 4
E4B and Qwen 3.5-4B?

**Method.**

1. For each model, compute per-layer importance (macro-F1 drop when
   that layer's output is replaced by its input — ablation probe).
2. Compute Spearman rank correlation between all pairs of models.
3. Control: compute rank correlation between a pair of
   independently-trained checkpoints of *the same model* (upper bound).

**Baseline.** Independent per-model ranking (no transfer).

**Novelty.** First quantitative test of mechanistic-finding
*portability* across compact foundation-model families.

**Compute.** ~25 H100-hours.

**Acceptance.** Spearman ρ ≥ 0.4 between Tiny Aya and at least one
of Gemma 4 E4B / Qwen 3.5-4B (null would be ~0).

**Files to add.**

```
src/lid/mi/layer_ablation.py
src/lid/mi/rank_transfer.py
configs/e7_transfer.yaml
```

### E8 — Contamination & Label-Noise Audit (CoDeC)

**Question (RQ16).** What fraction of each benchmark is contaminated
in the pretraining corpora of Tiny Aya / Gemma 4 / Qwen 3 / Qwen 3.5?
How does that change the reported macro-F1?

**Method.**

1. Adapt CoDeC (arXiv 2510.27055) to each model. Score contamination
   per-language on 1k held-out samples.
2. Also run membership-leakage tests on known-canary strings planted
   in 1024m/LID data.
3. Re-weight the eval metric by `(1 − contamination_rate)` per
   language and report *contamination-adjusted macro-F1*.

**Baseline.** Vanilla macro-F1.

**Novelty.** First per-language contamination audit of a 67-language
LID benchmark across 3 model families.

**Compute.** ~20 H100-hours (CPU-heavy).

**Acceptance.** Per-language contamination scores with ≥ 95%
confidence intervals (via bootstrapping).

**Files to add.**

```
src/lid/audit/codec.py
src/lid/audit/membership.py
configs/e8_contamination.yaml
```

### E9 — LID-Circuit-Preserving Pruning

**Question (RQ5).** Given the language-neuron map from E2, can we
prune layers / heads / neurons that are *not* language-neurons without
degrading LID?

**Method.**

1. Rank layers by mean causal importance (from E7 and E2).
2. Prune layers in reverse order one at a time; after each removal,
   optionally fine-tune a LoRA adapter for 100 steps on held-out
   data (to match LayerSkip recipe, Elhoushi 2024).
3. Report Pareto front (F1 vs. wall-clock vs. VRAM).
4. Compare against **magnitude pruning** (baseline) and **LayerSkip**
   baseline.

**Baseline.** Uniform magnitude pruning.

**Novelty.** First *circuit-aware* pruning for LID — pruning driven
by mechanistic evidence rather than global loss gradients.

**Compute.** ~60 H100-hours.

**Acceptance.** Prune ≥ 20% of layers with ≤ 0.5 pp macro-F1 loss;
magnitude-pruning baseline loses ≥ 3 pp at the same compression.

**Files to add.**

```
src/lid/compress/circuit_prune.py
src/lid/compress/layer_skip.py
configs/e9_prune.yaml
```

### E10 — Thinking-Budget / MatFormer Elastic Inference for LID

**Question (RQ13, RQ14).** Can (a) Gemma 4 E4B's MatFormer property
(E2B is a valid sub-model) and (b) Qwen 3's thinking-budget mechanism
be repurposed as LID-specific inference modes?

**Method.**

1. **Gemma 4 MatFormer.** Extract E2B sub-model from E4B on-the-fly
   (the Gemma 4 model card documents the procedure). Evaluate LID
   macro-F1 on the full 67 languages. Compare vs. stand-alone E2B.
2. **Qwen 3 thinking-budget.** Run Qwen 3 4B with
   `/no_think` (direct answer), `/think` + budget 100, `/think` +
   budget 500. Measure (a) accuracy, (b) token usage, (c) latency.
3. Hypothesis: LID is a no-think task and budgets > 0 do not help;
   but *maybe* budget helps for code-mix. Publishable either way.

**Baseline.** Non-elastic: train a stand-alone E2B and a stand-alone
Qwen 3 4B with fixed mode.

**Novelty.** First empirical test of MatFormer elastic-inference and
thinking-budget on a classification task in the wild.

**Compute.** ~40 H100-hours.

**Acceptance.** Publishable either way, with a clean Pareto plot.

**Files to add.**

```
src/lid/elastic/matformer.py
src/lid/elastic/thinking_budget.py
configs/e10_elastic.yaml
```

### E11 — Fine-Tuning Trajectories (H1/H2/H3 test)

**Question (RQ7).** Does LoRA fine-tuning cause (H1) the plateau
layer to shift earlier, (H2) a late-layer boost, or (H3) a uniform
lift?

**Method.**

1. For each training-target model, save LoRA checkpoints at
   {0, 50, 100, 250, 500, 1000, 2000, 4000} steps.
2. Run E1 atlas on each checkpoint.
3. Fit plateau curves over training steps; test H1 / H2 / H3 with a
   pre-registered statistical test (paired t-test on plateau layer
   at step 0 vs. final).

**Baseline.** Pre-trained (step 0) layer-wise curves.

**Novelty.** First longitudinal layer atlas across training, in a
cross-architecture setup.

**Compute.** ~70 H100-hours.

**Acceptance.** Statistically significant plateau-layer shift in ≥ 2
of the 3 models (p < 0.01 after Bonferroni correction).

**Files to add.**

```
src/lid/mi/trajectory.py
configs/e11_trajectory.yaml
```

### E12 — Metric Stress Test Beyond Macro-F1

**Question (RQ2, RQ11).** Do per-script calibration error (PSCE),
confusion-weighted F1 (CW-F1), and LLM-as-judge (CIA-Suite) rank the
9 models *differently* from macro-F1, and do those rankings correlate
with reported-fixed-metrics in prior work?

**Method.**

1. For every (model, dataset) pair, compute all Tier-2 metrics.
2. Compute pairwise Spearman ρ between rankings under different
   metrics.
3. Identify cases where (a) metrics disagree strongly and (b) the
   disagreement is explained by a specific script group or
   length-bin.

**Baseline.** Macro-F1 ranking.

**Novelty.** First LID-specific metric stress test with CIA-Suite and
PSCE. The field under-reports calibration.

**Compute.** ~18 H100-hours (mostly eval).

**Acceptance.** Identify ≥ 2 metric-disagreement cases and explain
them mechanistically (tying to atlas / circuits).

**Files to add.**

```
src/lid/eval/calibration.py
src/lid/eval/cw_f1.py
src/lid/eval/cia_judge.py
configs/e12_metric_stress.yaml
```

---

## 11. Cross-Experiment Shared Infrastructure

New code modules shared across experiments. All added under
`src/lid/` to keep the existing CLI entrypoints intact.

| Module | Purpose | Used by |
|---|---|---|
| `src/lid/mi/probes.py` | Linear / tuned-lens / logit-lens | E1, E7, E11 |
| `src/lid/mi/atlas.py` | Layer-wise atlas builder | E1, E11 |
| `src/lid/mi/patching.py` | Activation patching via transformer-lens | E2, E5 |
| `src/lid/mi/triangulation.py` | Domain/language invariance tests | E2 |
| `src/lid/mi/silent_failure.py` | Layers that would have classified correctly | E5 |
| `src/lid/mi/layer_ablation.py` | Layer output replacement | E7, E9 |
| `src/lid/mi/rank_transfer.py` | Cross-model rank correlation | E7 |
| `src/lid/mi/trajectory.py` | Per-checkpoint atlas | E11 |
| `src/lid/router/script_first.py` | Unicode-block → model router | E4 |
| `src/lid/router/policy.py` | Threshold policy optimizer | E4 |
| `src/lid/compress/circuit_prune.py` | Circuit-aware layer removal | E9 |
| `src/lid/compress/layer_skip.py` | LayerSkip reference | E9 |
| `src/lid/compress/layerwise_quant.py` | Per-layer bit-width search | E3 |
| `src/lid/elastic/matformer.py` | Gemma 4 E2B/E4B sub-model extraction | E10 |
| `src/lid/elastic/thinking_budget.py` | Qwen 3 `/think` budget sweeps | E10 |
| `src/lid/train/preference.py` | ORPO / β-DPO / SimPO wrappers | E6 |
| `src/lid/data/preference_pairs.py` | Contrastive pair generation | E6 |
| `src/lid/data/codemix.py` | Code-mix dataset loader | E5 |
| `src/lid/audit/codec.py` | Contamination detection | E8 |
| `src/lid/audit/membership.py` | Canary-based membership tests | E8 |
| `src/lid/eval/calibration.py` | PSCE | E12 |
| `src/lid/eval/cw_f1.py` | Confusion-weighted F1 with Glottolog | E12 |
| `src/lid/eval/cia_judge.py` | Cross-lingual LLM-as-judge | E12 |

CLI entrypoints to add in `pyproject.toml` (`[project.scripts]`):

```
lid-atlas       = "lid.cli:atlas"
lid-circuits    = "lid.cli:circuits"
lid-prune       = "lid.cli:prune"
lid-quant       = "lid.cli:quant"
lid-router      = "lid.cli:router"
lid-elastic     = "lid.cli:elastic"
lid-prefopt     = "lid.cli:prefopt"
lid-audit       = "lid.cli:audit"
lid-metric      = "lid.cli:metric"
```

---

## 12. Dataset Splits, Cards, and Release Plan

### 12.1 Source-of-truth splits

Every experiment reads from a canonical `datasets`-compatible split
under `src/lid/data/splits/`. Checksums recorded in
`experiments/dataset_fingerprint.json`.

### 12.2 New dataset cards to write

* `LID-mix-67-codemix-v1` (E5) — 6 sub-splits for code-mix /
  transliteration.
* `LID-ORPO-pairs-v1` (E6) — contrastive pairs for 12 closely-related
  language pairs.
* `LID-calibration-benchmark-v1` (E12) — held-out calibration split
  with gold confidence targets.
* `LID-67-contamination-audit-v1` (E8) — per-language contamination
  probability from CoDeC.

Each card specifies (a) licence (CC-BY-SA-4.0), (b) source provenance,
(c) language/script taxonomy, (d) known biases, (e) contamination
score.

### 12.3 Release plan

Upload to HuggingFace organisation `cataluna84/` (requires approval).

```
huggingface.co/datasets/cataluna84/LID-mix-67-codemix-v1
huggingface.co/datasets/cataluna84/LID-ORPO-pairs-v1
huggingface.co/datasets/cataluna84/LID-calibration-benchmark-v1
huggingface.co/datasets/cataluna84/LID-67-contamination-audit-v1
```

---

## 13. Artifact and HF Hub Release Plan

### 13.1 Checkpoints (public)

| Name | Model | Recipe | Approx size |
|---|---|---|---|
| `tiny-aya-global-lid-lora-v1` | Tiny Aya Global | LoRA r=16 | ~50 MB adapter |
| `gemma-4-e2b-lid-lora-v1` | Gemma 4 E2B | LoRA r=16 | ~80 MB |
| `gemma-4-e4b-lid-lora-v1` | Gemma 4 E4B | LoRA r=16 | ~140 MB |
| `qwen3-0.6b-lid-lora-v1` | Qwen 3 0.6B | LoRA r=16 | ~40 MB |
| `qwen3-1.7b-lid-lora-v1` | Qwen 3 1.7B | LoRA r=16 | ~60 MB |
| `qwen3-4b-lid-lora-v1` | Qwen 3 4B | LoRA r=16 | ~100 MB |
| `qwen3.5-0.8b-lid-lora-v1` | Qwen 3.5 0.8B | LoRA r=16 | ~45 MB |
| `qwen3.5-2b-lid-lora-v1` | Qwen 3.5 2B | LoRA r=16 | ~70 MB |
| `qwen3.5-4b-lid-lora-v1` | Qwen 3.5 4B | LoRA r=16 | ~100 MB |
| `{all}-lid-orpo-v1` (×12) | same | LoRA + ORPO on same-script pairs | +10 MB each |

### 13.2 Benchmark cards

One benchmark card per experiment (E1-E12) in `benchmarks/eN/CARD.md`.

### 13.3 Paper artefacts

* `paper/main.tex` — ACL / ARR 2026 style.
* `paper/figures/` — produced by `experiments/eN/plot_*.py`.
* `paper/supplement.pdf` — long appendix.
* `zenodo/` — DOI-referenced snapshot of experiments/ at submission
  time.

---

## 14. Timeline (6 Weeks to ARR Deadline)

Assumes ARR May 2026 cycle close ~Apr 30 2026 (hard stop). Today is
2026-04-17 — **13 calendar days to internal code freeze**, plus
~2 weeks of writing / camera-ready buffer.

| Week | Work | Deliverables |
|---|---|---|
| W1 (Apr 17--23) | Spec approval; scaffolding; E8 contamination first | `docs/do.md` merged, `src/lid/audit/` in place, contamination scores for 67 langs |
| W2 (Apr 24--30) | E1 Atlas pilot on 3 models → scale-out; E3 quant sensitivity scoring | atlas first-pass; quant ILP for Tiny Aya |
| W3 (May 1--7) | E2 circuit discovery (Tiny Aya); E4 script-first router | circuits-v1, router-v1 |
| W4 (May 8--14) | E5, E6, E7 parallel; E11 trajectory runs | code-mix results, ORPO/β-DPO/SimPO, rank transfer |
| W5 (May 15--21) | E9 pruning; E10 elastic inference; E12 metric stress | Pareto plots, elastic-LID result, metric stress |
| W6 (May 22--28) | Writing + replication runs + HF Hub uploads | paper draft v1, checkpoints public, benchmark cards |
| Buffer | May 29--31 | camera-ready polish |

**Gantt (text):**

```
              W1      W2      W3      W4      W5      W6
E1 Atlas      ==================
E2 Circuits           ==================
E3 Quant      ==================
E4 Router              ============
E5 Code-mix                     ========
E6 PrefOpt                      ========
E7 Transfer                     ====
E8 Contam.    ====
E9 Pruning                              ========
E10 Elastic                             ========
E11 Traject.                    ================
E12 Metric                              ========
Writing                                         ================
```

---

## 15. Paper Outline (ARR long paper, 8 + unlimited refs)

1. **Abstract** — what, why, how, claims (3 main results + artifact release).
2. **Introduction** — LID is gateway, models are compact, gap between
   F1 and real world.
3. **Related Work** — MI, layer-wise, quantization, LID benchmarks,
   preference optimization. §9 of this document maps 1:1.
4. **Models and Data** — §4 and §6 of this document, condensed.
5. **Method: Layer Emergence Atlas** — E1.
6. **Method: Language Neuron Discovery** — E2.
7. **Method: Interpretability-Guided Compression** — E3 + E9.
8. **Method: Elastic LID** — E4 + E10.
9. **Evaluation Protocol** — §7; metric definitions.
10. **Results** — headline table + Pareto plots.
11. **Analyses** — E5 (code-mix), E6 (ORPO/β-DPO/SimPO), E7
    (transfer), E11 (trajectories), E12 (metric stress).
12. **Discussion** — implications for compact foundation-model
    design.
13. **Limitations** — §8.4 risks + what we didn't run.
14. **Ethics** — CoDeC contamination; content filtering; dual-use
    considerations for LID.
15. **Reproducibility** — §16.

*Additional artefact* — a *standalone 6-page demo paper* for EMNLP
Demos 2026 ("`lid-bench`: an elastic layer-aware LID toolkit") reusing
§11 infra.

---

## 16. Reproducibility Checklist

Per ACL Responsible-Research-Checklist:

* [ ] **Code**: all experiments behind `uv run lid-*` CLIs
  (§11). `make verify` exits 0 on a clean checkout.
* [ ] **Data**: all 4 new dataset cards (§12.2) on the HF Hub with
  CC-BY-SA-4.0.
* [ ] **Checkpoints**: 9 × (plain LoRA + ORPO variants) uploaded
  (§13.1).
* [ ] **Logs**: W&B project `lid-bench` public; one run per row in
  every table.
* [ ] **Seeds**: `seed=1024` default; ≥ 3 repeats for every
  headline number.
* [ ] **Hardware**: list tested GPUs (H100 80GB primary; A100 80GB
  fallback; L40S 48GB for small-model ablations).
* [ ] **Environment**: `uv.lock` pinned to a tagged commit; Docker
  image optional.
* [ ] **Licences**: Apache 2.0 for all models; CC-BY-SA for data;
  permissive for code.
* [ ] **Carbon**: kg CO2-eq estimated per E1-E12 run via codecarbon.
* [ ] **Statistical tests**: pre-registered for E11 (H1/H2/H3) and
  E6 (ORPO vs. β-DPO vs. SimPO).

---

## 17. Open Questions for Review

Explicit questions that we should answer before freezing the plan.

* **OQ1** — Is the 9-model × 12-experiment scope actually runnable in
  ~523 H100-hours? The main risk is E2 (patching) and E9 (pruning).
  Backup plan: drop Qwen 3.5-2B from E2/E9 (keep only 0.8B and 4B).
* **OQ2** — Gemma 4 weights may be gated at run time. Fallback to
  Gemma 3 4B / 1B is documented in §8.4 but would weaken the novelty.
  Status check 3 days before W1 freeze.
* **OQ3** — Should we include *Aya Expanse 8B* as a bigger
  multilingual baseline? It's Cohere, 101 languages, Apache 2.0.
  Would add ~30 H100-hours.
* **OQ4** — Is a 6-week compressed timeline realistic given the paper
  is 8 pages + supplement? Tradeoff: strip E10 (elastic) and E12
  (metric stress) if writing runs long — both are *standalone*
  stories we can release as follow-ups.
* **OQ5** — Are we comfortable committing the headline result to
  being "circuit-aware pruning > magnitude pruning"? If E9 fails,
  E3 + E4 still stand as a coherent paper.
* **OQ6** — Should we pre-register E11 (H1/H2/H3) on OSF? Adds 2
  days, increases credibility.
* **OQ7** — Licence for curated code-mix splits (§12.2)? CC-BY-SA-4.0
  is default but some Indic sources are restricted.
* **OQ8** — Speech-derived LID via Mozilla Common Voice: in-scope (E5)
  or explicitly out-of-scope?
* **OQ9** — Gated DeltaNet (Qwen 3.5) is not yet in mainline
  `transformers`. If unsupported, do we skip Qwen 3.5 altogether or
  vendor a compatible branch?
* **OQ10** — Do we want a companion demo-track submission to EMNLP
  Demos 2026 that packages `lid-bench` as a public tool?

---

## 18. References (Short, 2024--2026)

Core references grouped by theme. Full BibTeX in
`paper/references.bib` at submission.

**Mechanistic interpretability:**

- [Circuit Tracing] transformer-circuits.pub/2025/attribution-graphs/methods.html (Anthropic, 2025)
- [Triangulation] arXiv:2512.24842
- [Language Arithmetics] aclanthology.org/2025.ijcnlp-long.156
- [Locate, Steer, Improve] arXiv:2601.14004v4
- [Mechanistic Understanding & Mitigation] arXiv:2505.16538v2

**Layer-wise probing and depth:**

- [Layer by Layer] arXiv:2502.02013
- [Generalization Ridge] arXiv:2507.05387
- [Early Exit Is a Natural Capability] arXiv:2412.01455
- [LayerSkip] Elhoushi et al., Meta 2024 (openreview HIXPyQ1aMq)
- [Do Transformers Use Their Depth Adaptively?] arXiv:2604.12426v1

**Quantization + multilingual:**

- [How Does Quantization Affect Multilingual LLMs?] arXiv:2407.03211
- [BitNet 1.58-bit] arXiv:2402.17764
- [Layer-wise Information-Effectiveness for PTQ] ResearchGate
  394321992

**Preference optimization:**

- [ORPO] arXiv:2403.07691v2
- [β-DPO] NeurIPS 2024 poster 94622
- [SimPO] Meng et al. 2024
- [FocalPO] — forthcoming

**Benchmarks + evaluation:**

- [GlotEval] aclanthology.org/2025.emnlp-demos.43
- [CIA Suite] aclanthology.org/2025.acl-long.1419
- [Form & Meaning] aclanthology.org/2026.eacl-long.113
- [DebateBias] arXiv:2511.01187v4
- [CommonLID blog] commoncrawl.org/blog/commonlid-2024
- [GlotLID] arXiv:2310.16248
- [OpenLID-v3] arXiv:2602.13139
- [CoDeC] arXiv:2510.27055v1
- [ILID] arXiv:2507.11832v3

**Models:**

- [Gemma 4 blog] blog.google/innovation-and-ai/technology/developers-tools/gemma-4/ (2026-04-02)
- [Gemma 4 model card] ai.google.dev/gemma/docs/core/model_card_4
- [MatFormer] Devvrit et al. NeurIPS 2024
- [Qwen 3 report] arXiv:2505.09388
- [Qwen 3.5 blog] qwen.ai/blog?id=qwen3.5
- [Qwen 3.5 GitHub] github.com/QwenLM/Qwen3.5

**Toolkits:**

- [transformer-circuits.pub](https://transformer-circuits.pub)
- [nnterp] github.com/ndif-team/nnterp
- [mech-interp-toolkit] github.com/ICI-Innolabs/mech-interp-toolkit
- [MechaMap] github.com/tegridydev/mechamap
- [InterpBench] NeurIPS 2024 poster 97689

---

## Appendix A — Quick CLI Smoke Tests

After scaffolding each experiment, one-liner sanity checks:

```bash
# E1 — atlas on a tiny subset
uv run lid-atlas configs/e1_atlas.yaml --limit 200 --no-wandb

# E2 — patch one language neuron
uv run lid-circuits configs/e2_circuits.yaml --language en --no-wandb

# E3 — quantization sensitivity scoring
uv run lid-quant configs/e3_layerwise_quant.yaml --score-only

# E4 — script-first router on 100 samples
uv run lid-router configs/e4_early_exit.yaml --limit 100

# E5 — code-mix split loader
uv run python -c "from lid.data.codemix import load; print(load('cmlid-hi-ur').shape)"

# E6 — ORPO warmup
uv run lid-prefopt configs/e6_preference.yaml --objective orpo --limit 500

# E7 — rank correlation
uv run lid-atlas rank-transfer --from tiny-aya --to qwen3-4b

# E8 — contamination sample
uv run lid-audit codec --limit 100

# E9 — prune one layer
uv run lid-prune configs/e9_prune.yaml --drop-layer 34 --finetune 100

# E10 — matformer extract
uv run lid-elastic matformer --base gemma-4-e4b --target e2b

# E10 — thinking-budget
uv run lid-elastic think --model qwen3-4b --budget 100

# E11 — atlas at step 250
uv run lid-atlas configs/e1_atlas.yaml --checkpoint step-250

# E12 — CW-F1 only
uv run lid-metric cw-f1 --pred preds.csv --gold gold.csv
```

---

## Appendix B — Pre-Registration Stub for E11

(Deposited at OSF before any trajectory runs. Draft.)

```
Title: Fine-Tuning Trajectories of Layer-Wise Language Identification
        Plateaus in Compact Foundation Models

Hypotheses:
 H1 (early shift): Plateau layer decreases by >= 4 layers after 1000 steps
 H2 (late boost):  Plateau F1 increases by >= 5 pp at layers in top-10%
                   while remaining layers change by < 1 pp
 H3 (uniform lift): F1 increases by >= 2 pp at every layer

Primary test: paired t-test on (plateau_layer_step_0,
             plateau_layer_step_final) across 3 models x 67 languages.
Bonferroni correction factor: m = 3.
Significance threshold: p < 0.01.

Exclusions: languages with < 100 training examples are dropped.
Any contamination > 5% (from E8) is a pre-registered exclusion.

Data release: all checkpoints uploaded to HF Hub with LoRA adapters
             and eval CSVs. W&B runs public.
```

---

*End of `docs/do.md` v0.1 — ready for colleague review.*
