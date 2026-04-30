# Original Project Proposal -- Source Document

> ## Status of this document
>
> This is the **canonical source proposal** for the LID project, faithfully
> mirrored from a Google Doc dated early in the project's life. It is the
> genesis of every research question and experimental setting that the
> rest of the repository elaborates on.
>
> **Lineage:**
>
> 1. **`docs/proposal_original.md`** *(this file)* -- the original,
>    short-form proposal as it was first written. Treat it as a
>    historical artefact and a single source of truth for the original
>    research framing. Edits should be limited to fixing dead links and
>    clarifying that paper-deadline framing in the original was
>    deprecated when the repository went public.
> 2. **[`docs/project_proposal.md`](project_proposal.md)** -- the
>    extended proposal. Reorganises the research questions into a
>    structured publishability assessment and adds an experimental
>    design table.
> 3. **[`docs/paperback.md`](paperback.md)** -- the forward-looking
>    research-direction document. Generalises the original proposal
>    into a venue-agnostic, deadline-agnostic roadmap with twelve
>    well-scoped experiments (E1--E12), a model cohort, an
>    infrastructure plan, and a literature map (~50 refs from
>    2024--2026).
>
> **Source:** *Layer-Wise Dynamics of Multilingual Language
> Identification in Compact Foundation Models* — Google Doc:
> <https://docs.google.com/document/d/1gUKP0q4uVP6FUFrRZKl6P3sWwji3APJami6o9Pex0y8/edit>.
> The Markdown body below is a faithful mirror; if the Google Doc and
> this file disagree, this file is treated as the canonical version
> (the Google Doc may be edited or revoked, but this file is locked
> to the public-release commit).
>
> **Public release date:** 2026-04-30. The original proposal mentioned
> "ARR submission (May 2026 cycle)" as one of several venue options;
> that paper-deadline framing has been retired in the current public
> release. See the *Status* note in [`README.md`](../README.md) and
> [`docs/paperback.md`](paperback.md). The text below preserves the
> original wording for the historical record.

---

## Layer-Wise Dynamics of Multilingual Language Identification in Compact Foundation Models

Language identification is a fundamental component of nearly every
multilingual NLP system, yet large-scale classification across 100+
languages remains imperfect even in strong modern models. Performance
drops sharply for closely related languages, low-resource varieties,
code-mixed inputs, and transliterated text, revealing gaps between
benchmark accuracy and real-world robustness.

---

## What is the question we want to answer?

1. What is the relationship between **quantization level** and
   large-scale (100+ class) language identification accuracy after
   fine-tuning?
2. How does classification error differ between languages that share
   the same script versus languages that use distinct scripts?
3. Does **preference optimization** specifically between same-script
   languages reduce confusion, and what side effects does it introduce
   for unrelated language pairs?
4. Would a **hierarchical setup** (script-level routing followed by
   intra-script classification) outperform a single flat 100+ class
   classifier?
5. To what extent can **model depth or capacity** be reduced without
   significant degradation in language identification performance?
6. How does **layer-wise classification accuracy** evolve across
   transformer depth for different language families?
7. How does **layer-wise performance shift over the course of
   training**, and at what stage do language-discriminative signals
   stabilize?
8. Are early layers primarily encoding **script-level signals** while
   deeper layers resolve fine-grained linguistic distinctions?
9. Does quantization **disproportionately affect closely related
   languages** compared to typologically distant ones?
10. How stable are **per-language confidence distributions** across
    layers and across training checkpoints?
11. How does the model behave on **short inputs versus long inputs**
    in high-class language identification settings?
12. Uniform vs. mixed-precision -- compare blanket 4-bit quantization
    vs. selective quantization (early layers 4-bit, deep layers 8-bit).

---

## Why is this question important?

- Language identification is the gateway task for multilingual
  systems; errors at this stage cascade into translation, retrieval,
  moderation, and generation failures.
- Real-world deployments require supporting 100+ languages, yet
  performance gaps persist for low-resource and closely related
  language pairs.
- Same-script languages (e.g. languages sharing Latin or Cyrillic
  scripts) are frequently confused, creating systematic routing and
  personalization errors.

---

## Papers to read to catch up on the state of the literature

### Benchmark papers for language identification

- **CommonLID: Re-evaluating State-of-the-Art Language Identification
  Performance on Web Data.** <https://arxiv.org/pdf/2601.18026v1>
- **What Language is This? Ask Your Tokenizer (UniLID, multilingual
  LID approach).** <https://arxiv.org/pdf/2602.17655v1>
- **An Open Dataset and Model for Language Identification.**
  <https://arxiv.org/pdf/2305.13820>
- **LIMIT: Language Identification, Misidentification, and
  Translation using Hierarchical Models.**
  <https://arxiv.org/pdf/2305.14263>
- **Language ID in the Wild: Unexpected Challenges on the Path to a
  Thousand-Language Text Corpus.**
  <https://arxiv.org/pdf/2010.14571>

### LID models / system papers

- **GlotLID: Language Identification for Low-Resource Languages
  (1665+ languages).** <https://arxiv.org/pdf/2310.16248>
- **OpenLID-v3: Improving the Precision of Closely Related Language
  Identification.** <https://arxiv.org/pdf/2602.13139>

### Tools / products / systems

- **Google Translate -- Language Detection.**
  <https://cloud.google.com/translate/docs/basic/detecting-language>
- **Microsoft Azure Translator -- Detect Language API.**
  <https://learn.microsoft.com/azure/cognitive-services/translator/language-detection>
- **Amazon Translate -- Automatic Language Detection.**
  <https://docs.aws.amazon.com/translate/latest/dg/how-it-works.html#how-it-works-detect-language>
- **DeepL API -- Language Detection.**
  <https://www.deepl.com/docs-api/translating-text/usage-guidelines/#detecting-source-language>
- **Yandex Translate -- Language Detection.**
  <https://yandex.com/dev/translate/>
- **FastText Language Identification.**
  <https://fasttext.cc/docs/en/language_identification.html>

For the extended literature map (50+ references covering mechanistic
interpretability, layer-wise probing, quantisation, preference
optimisation, and benchmarks 2024--2026), see
[`docs/paperback.md`](paperback.md) §9.

---

## Is this a publishable question?

**Yes.** This is a publishable research direction if framed as a
systematic empirical study rather than just a model improvement.
While large-scale language identification has been explored, there
is no comprehensive layer-wise analysis of representation formation,
quantization effects, and high-class (100+ language) behavior within
compact foundation models under controlled settings. A depth-aware
evaluation combined with structured error analysis (same-script
confusion, low-resource performance, compression sensitivity) would
constitute a novel and externally shareable contribution.

> *Note (2026-04-30, public release):* The original proposal listed
> "tutorials/industry track or a general ARR submission (May 2026
> cycle)" as candidate venues. That paper-deadline framing has been
> retired in the public release; the current document
> [`docs/paperback.md`](paperback.md) is venue-agnostic and
> deadline-agnostic. Anyone continuing this work should pick whatever
> venue and timeline fits their compute budget and research arc.

---

## What is the most simple experimental setting in which this hypothesis can be tested?

- Compact foundation models with classification head (0--4 B params).
- LoRA fine-tuning on language identification.
- All-layer output extraction with class probabilities per layer
  after each *N* steps of training.
- Evaluation under FP16, 8-bit, and 4-bit quantization (maybe 2-bit
  and 1-bit too depending on results).
- Per-language accuracy and same-script confusion analysis.
- Layer-wise accuracy tracking across training checkpoints.
- LoRA + ORPO fine-tuning for high-confusion same-script pairs.
- Multi-stage setup: script-level classifier followed by
  intra-script classifier.
- Depth truncation experiments removing upper layers to improve
  efficiency.
- Short vs long input evaluation for length sensitivity (word/token
  count vs accuracy).
- **Error analysis** -- identifying patterns in error generation
  across models, quantization levels, and language/script groups.
- **Uniform vs mixed-precision comparisons** -- compare blanket
  4-bit quantization vs selective quantization (early layers 4-bit,
  deep layers 8-bit).

For the elaborated experimental design and a strict
[E1--E12 catalogue](paperback.md#10-experiments-e1e12), see
[`docs/paperback.md`](paperback.md).

---

## Datasets, models, eval, annotation, resources

### Datasets

> Most publicly available large-scale LID datasets are web-scraped
> and weakly labelled using existing language-identification systems
> ([Kreutzer et al., TACL 2022](https://aclanthology.org/2022.tacl-1.4.pdf)).
> Training on such data risks becoming a form of indirect
> distillation rather than genuine improvement, so higher-quality
> source-verified corpora are preferred.

Candidate sources:

- **HC Corpora newspapers** --
  <https://www.kaggle.com/datasets/alvations/old-newspapers>
- **Wikimedia / Wikipedia dumps** (language-separated corpora) --
  <https://huggingface.co/datasets/wikimedia/wikipedia>
- **XL Sum** --
  <https://huggingface.co/datasets/csebuetnlp/xlsum>
- **Cohere LID Data** (the basis of the `1024m/LID` hackathon
  release used in this repository) --
  <https://huggingface.co/datasets/1-800-LLMs/Cohere-LID-Data>

The current repo's primary training/eval dataset
(`1024m/LID`, the Cohere LID hackathon release on Hugging Face) is a
close descendant of the last entry above. The repository's
[`README.md`](../README.md#reproducibility-caveats) explains the
public-vs-private status of this dataset.

### Models (initial list)

The original document listed the following candidate models. See
[`docs/paperback.md`](paperback.md#4-model-cohort) for the
post-public-release model cohort, which condenses these into three
open-weight families (Cohere Tiny Aya, Gemma 4, Qwen 3 / 3.5).

- **Cohere Tiny Aya** -- Global, Fire, Water, Earth.
- **Gemma 3n** -- E4B, E2B.
- **Gemma 3** -- 4B, 1B.
- **Translate Gemma 4B**.
- **Qwen 3.5** -- 4B, 2B, 1B.

The repository ships the LoRA training pipeline tested against
**Cohere Tiny Aya Global** (~3.35 B parameters) on a single H100 80
GB. Other models in the list are inference-feasible with the same
codebase but have not been trained end-to-end in this repo.

### Evaluation metrics (initial list)

- Precision
- Recall
- F1
- Accuracy
- Negative log-likelihood
- ROC-AUC
- PR-AUC

The current repo's metric implementation lives in
`src/lid/bench/metrics.py`. The extended Tier-2/Tier-3 metric
recommendations -- per-script calibration error, confusion-weighted
F1, MFU, energy per sample -- are documented in
[`docs/optimization_spec.md`](optimization_spec.md) §5 and
[`docs/paperback.md`](paperback.md) §7.

---

## Workflow diagram

The original document had a placeholder ("Initial approach workflow
diagram: TBA"). The realised workflow shipped in this repository is:

```
                   1024m/LID dataset (HF Hub)
                              |
                              v
                  +-----------+-----------+
                  |   data.py             |
                  |   prompt construction |
                  +-----------+-----------+
                              |
                              v
            +-----------------+------------------+
            |                                    |
            v                                    v
       train.py (LoRA)                    infer.py (layer-wise)
       lid-train CLI                      lid-infer CLI
            |                                    |
            +------------------+-----------------+
                               v
                      bench/runner.py (9 strategies)
                            lid-bench CLI
                               |
                               v
                  +------------+------------+
                  |                         |
                  v                         v
         experiments/                 W&B project lid-bench
         all_results.csv              runs / artefacts / plots
         per-step REPORT.md
                  |
                  v
         lid-report / lid-recommend
         (read CSV or W&B, rank configs)
```

The classifier baselines (Unicode-block heuristic, character n-grams,
embedding-classifier with `multilingual-e5-large`) live in the
[`notebooks/`](../notebooks/) directory and are independent of the
LoRA pipeline. They are described in
[`README.md`](../README.md#notebooks).

---

## Where to go from here

If you have just read this proposal and want to:

- **Set up the repo and reproduce a benchmark run** ->
  [`README.md`](../README.md#quickstart) and
  [`docs/RUNBOOK.md`](RUNBOOK.md).
- **Understand the optimisation strategies in detail** ->
  [`docs/optimization_spec.md`](optimization_spec.md).
- **Continue the research line (new experiments, models,
  benchmarks)** -> [`docs/paperback.md`](paperback.md) -- the
  forward-looking roadmap.
- **Contribute code or fixes** ->
  [`CONTRIBUTING.md`](../CONTRIBUTING.md).

---

*End of `docs/proposal_original.md`. Do not edit the body text
above without first preserving the original wording somewhere
linkable; the goal is to keep the genesis proposal recoverable
verbatim.*
