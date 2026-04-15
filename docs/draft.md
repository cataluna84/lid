# Layer-Wise Dynamics of Multilingual Language Identification in Compact Foundation Models

> Research draft -- living document for collaborators

## Abstract

Language identification is a fundamental component of nearly every multilingual NLP system, yet large-scale classification across 100+ languages remains imperfect even in strong modern models. Performance drops sharply for closely related languages, low-resource varieties, code-mixed inputs, and transliterated text, revealing gaps between benchmark accuracy and real-world robustness.

This project investigates the **layer-wise dynamics** of how compact foundation models (0--4B parameters) learn to identify languages, combining mechanistic interpretability with systematic ablation across quantization levels, training stages, and architectural depth.

---

## Research Questions

1. What is the relationship between **quantization level** and large-scale (100+ class) language identification accuracy after fine-tuning?
2. How does classification error differ between languages that **share the same script** versus languages that use distinct scripts?
3. Does **preference optimization** (ORPO) specifically between same-script languages reduce confusion, and what side effects does it introduce for unrelated language pairs?
4. Would a **hierarchical setup** (script-level routing followed by intra-script classification) outperform a single flat 100+ class classifier?
5. To what extent can **model depth or capacity** be reduced without significant degradation in language identification performance?
6. How does **layer-wise classification accuracy** evolve across transformer depth for different language families?
7. How does layer-wise performance shift **over the course of training**, and at what stage do language-discriminative signals stabilize?
8. Are early layers primarily encoding **script-level signals** while deeper layers resolve **fine-grained linguistic** distinctions?
9. Does quantization **disproportionately affect closely related languages** compared to typologically distant ones?
10. How stable are per-language **confidence distributions** across layers and across training checkpoints?
11. How does the model behave on **short inputs versus long inputs** in high-class language identification settings?

## Why This Matters

- Language identification is the **gateway task** for multilingual systems; errors cascade into translation, retrieval, moderation, and generation failures.
- Real-world deployments require supporting **100+ languages**, yet performance gaps persist for low-resource and closely related language pairs.
- Same-script languages (e.g., languages sharing Latin or Cyrillic scripts) are **frequently confused**, creating systematic routing and personalization errors.

## Publishability Assessment

This is a **publishable research direction** if framed as a systematic empirical study rather than just a model improvement. There is no comprehensive layer-wise analysis of representation formation, quantization effects, and high-class (100+ language) behavior within compact foundation models under controlled settings. A depth-aware evaluation combined with structured error analysis (same-script confusion, low-resource performance, compression sensitivity) would constitute a novel and externally shareable contribution. Target venue: **ARR submission (May 2026 cycle)** or industry/tutorials track.

---

## Experimental Design

### Minimal Viable Experiment

- Compact foundation models with classification head (0--4B params)
- LoRA fine-tuning on language identification
- All-layer output extraction with class probabilities per layer after each N steps of training
- Evaluation under FP16, 8-bit, and 4-bit quantization (possibly 2-bit and 1-bit)
- Per-language accuracy and same-script confusion analysis
- Layer-wise accuracy tracking across training checkpoints

### Extended Experiments

- **LoRA + ORPO** fine-tuning for high-confusion same-script pairs
- **Multi-stage setup**: script-level classifier followed by intra-script classifier
- **Depth truncation** experiments removing upper layers to improve efficiency
- **Short vs long input** evaluation for length sensitivity (word/token count vs accuracy)
- **Error analysis**: identifying patterns in error generation across models, quantization levels, language/script-groups
- **Uniform vs mixed-precision**: blanket 4-bit quantization vs selective quantization (early layers 4-bit, deep layers 8-bit)

---

## Datasets

> Most publicly available large-scale LID datasets are web-scraped and weakly labeled using existing LID systems. Training on such data risks becoming indirect distillation rather than genuine improvement, so higher-quality, source-verified corpora are preferred.

| Dataset | Source | Link |
|---------|--------|------|
| HC Corpora newspapers | Kaggle | [kaggle.com/datasets/alvations/old-newspapers](https://kaggle.com/datasets/alvations/old-newspapers) |
| Wikimedia / Wikipedia dumps | HuggingFace | [hf.co/datasets/wikimedia/wikipedia](https://hf.co/datasets/wikimedia/wikipedia) |
| XL-Sum | HuggingFace | [hf.co/datasets/csebuetnlp/xlsum](https://hf.co/datasets/csebuetnlp/xlsum) |
| Cohere LID Data | HuggingFace | [hf.co/datasets/1-800-LLMs/Cohere-LID-Data](https://hf.co/datasets/1-800-LLMs/Cohere-LID-Data) |
| 1024m/LID (Hackathon) | HuggingFace | [hf.co/datasets/1024m/LID](https://hf.co/datasets/1024m/LID) |

## Models

| Model | Params | Notes |
|-------|--------|-------|
| TinyAya Global | ~1B | Cohere multilingual; current baseline |
| TinyAya Fire / Water / Earth | ~1B | Cohere variants |
| Gemma 3n E4B / E2B | 2--4B | Google compact models |
| Gemma 3 4B / 1B | 1--4B | Google base models |
| Translate Gemma 4B | 4B | Translation-focused variant |
| Qwen 3.5 4B / 2B / 1B | 1--4B | Alibaba multilingual series |

## Evaluation Metrics

- Precision, Recall, F1 (macro and per-language)
- Accuracy (overall and per-script-group)
- Negative Log-Likelihood
- ROC-AUC, PR-AUC
- Same-script confusion matrix
- Layer-wise accuracy curves

---

## Related Work

### Benchmark Papers

- **CommonLID** -- Re-evaluating State-of-the-Art LID Performance on Web Data ([arxiv.org/pdf/2601.18026v1](https://arxiv.org/pdf/2601.18026v1))
- **UniLID** -- What Language is This? Ask Your Tokenizer ([arxiv.org/pdf/2602.17655v1](https://arxiv.org/pdf/2602.17655v1))
- **Open Dataset and Model for LID** ([arxiv.org/pdf/2305.13820](https://arxiv.org/pdf/2305.13820))
- **LIMIT** -- Language Identification, Misidentification, and Translation using Hierarchical Models ([arxiv.org/pdf/2305.14263](https://arxiv.org/pdf/2305.14263))
- **Language ID in the Wild** -- Unexpected Challenges on the Path to a Thousand-Language Text Corpus ([arxiv.org/pdf/2010.14571](https://arxiv.org/pdf/2010.14571))

### LID Models / Systems

- **GlotLID** -- Language Identification for Low-Resource Languages (1665+ languages) ([arxiv.org/pdf/2310.16248](https://arxiv.org/pdf/2310.16248))
- **OpenLID-v3** -- Improving the Precision of Closely Related Language Identification ([arxiv.org/pdf/2602.13139](https://arxiv.org/pdf/2602.13139))

### Industry Systems

- Google Translate, Microsoft Azure Translator, Amazon Translate, DeepL API, Yandex Translate, FastText LID

---

## Collaboration Notes

- **Discord thread discussions** are the primary async communication channel.
- Initial approach workflow diagram: TBA.
- Experiment tracking: W&B or local CSV logs stored in `experiments/`.
- All notebooks go in `notebooks/` and are stripped of outputs before commit (via `nbstripout`).
