# Project Proposal: Layer-Wise Dynamics of Multilingual Language Identification in Compact Foundation Models

> Living document for collaborators -- last updated 2026-04-30
>
> **Status:** this is the *extended* proposal in the project's
> documentation lineage. See [`README.md`](../README.md#project-genesis-and-document-lineage)
> for the full chain:
>
> - **Source:** [`proposal_original.md`](proposal_original.md) -- the
>   genesis proposal (twelve research questions, original framing,
>   reading list).
> - **This file:** the extended proposal -- adds publishability
>   assessment, experimental-design table, and explicit pointers
>   into the codebase.
> - **Forward-looking:** [`paperback.md`](paperback.md) -- a
>   venue/deadline-agnostic research-direction document for
>   anyone continuing this line of work.

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

This is a **publishable research direction** if framed as a systematic empirical study rather than just a model improvement. There is no comprehensive layer-wise analysis of representation formation, quantization effects, and high-class (100+ language) behavior within compact foundation models under controlled settings. A depth-aware evaluation combined with structured error analysis (same-script confusion, low-resource performance, compression sensitivity) would constitute a novel and externally shareable contribution. Suitable venues range from ACL / EMNLP / TACL to NeurIPS / ICLR workshops on multilingual NLP and mechanistic interpretability; venue choice is left to whoever pursues the writeup. See `docs/paperback.md` for a fuller research roadmap.

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

## Experiments Conducted (March--April 2026)

### Objective Beyond Scope

The primary goal beyond the original scope document was to find **an approach that would work on unseen domains for LID** -- i.e., a classifier that generalizes beyond the training distribution to handle web text, social media, code-mixed content, and other non-standard inputs.

### Approach 1: Unigram Classifier

- **Method:** Character unigram frequency vectors as features for a classifier, trained on 500 / 1,000 / 2,500 samples per language.
- **Notebook:** [`notebooks/LID_Ngrams_Classifier.ipynb`](../notebooks/LID_Ngrams_Classifier.ipynb)
- **Result:** Does not scale for low-resource languages or out-of-domain texts. Feature space too sparse for closely related languages sharing the same script.

### Approach 2: N-gram Classifier (n=2,3,...5)

- **Method:** Character n-gram frequency vectors (bigrams through 5-grams) with the same sample sizes as Approach 1.
- **Notebook:** Same as Approach 1.
- **Result:** Similar results to unigrams -- marginal improvement from higher-order n-grams, but still insufficient for low-resource and out-of-domain generalization.

### Approach 3: Unicode Block + Regex Classifier

- **Method:** Classify characters/tokens via Unicode block membership using regex for unique and rare scripts (e.g., Devanagari, Thai, Georgian), then a statistical classifier for common scripts (e.g., Latin, Cyrillic).
- **Notebook:** [`notebooks/LID_Unicode_Blocks_Classifier.ipynb`](../notebooks/LID_Unicode_Blocks_Classifier.ipynb)
- **Result:** **20+ of the ~67 TinyAya languages could be classified at over 99% accuracy using regex alone** (languages with unique scripts). The remaining languages sharing common scripts (Latin, Cyrillic, Arabic) require a secondary classifier. This was also tested on the 4 benchmarks used in Approach 4.

### Approach 4: Embedding Model Classifier (Best Result)

- **Method:** Fine-tune a 0.6B embedding model with a classification head, trained on ~900 samples per language (~30 min on 1xH100).
- **Notebook:** [`notebooks/LID_Embedding_Classifier.ipynb`](../notebooks/LID_Embedding_Classifier.ipynb)
- **Result:** **Macro F1 of 0.97+ even on unseen domains**, including out-of-distribution web text and benchmark datasets. This raises the question: if a small embedding model achieves this performance with minimal training data, does pruning TinyAya by ~80% provide any advantage?
- **Note on CommonLID F1:** The observed drop on CommonLID macro F1 is attributed to mislabelling in the benchmark (examples shared in Discord). Actual performance is estimated around 0.80 after accounting for label noise.

### Implications for This Project

The embedding model result (Approach 4) establishes a strong baseline: a 0.6B model achieving 0.97+ F1 on unseen domains with <1000 samples/language. The layer-wise analysis in this project investigates **why** and **where** in the transformer stack these language-discriminative signals form, which the embedding approach treats as a black box. The two lines of work are complementary:

- **This project (layer-wise LID):** Mechanistic understanding of how language identification emerges across layers, with optimization benchmarking for deployment efficiency.
- **Embedding classifier:** Practical high-accuracy system for production LID.

---

## Phase 3 and Beyond: Where Do Classifications Occur in Large Language Models?

Phase 2 (current) focuses on optimizing the inference pipeline for layer-wise extraction on a single model and task. **Phase 3 generalizes the core question:** at which layers do LLMs form task-relevant classification signals, and is this consistent across models and tasks?

### Core Research Question

Using **normalized log-probabilities extracted from each layer** of an LLM, at what depth does classification performance plateau -- and can the remaining layers be pruned for efficiency?

### Experimental Design

1. **Multiple classification tasks** (not just LID):
   - Language Identification (LID) -- current focus
   - Natural Language Inference (NLI)
   - Physical Intuition QA (PIQA)
   - Social Interaction QA (SIQA)
   - Fill-mask / cloze tasks
   - Other generic classification benchmarks as available

2. **Multiple LLMs of similar size** (1--4B parameter range):
   - Extract per-layer normalized log-probs for every model
   - Compare layer-wise accuracy curves across models on the same task
   - Identify whether a common "plateau layer range" exists

3. **Cross-model analysis:**
   - If classification performance plateaus at layer N across multiple models, the layers beyond N are candidates for pruning
   - Investigate whether this is a universal phenomenon (architecture-independent) or model-specific
   - Compare decoder-only vs encoder-decoder architectures if feasible

### Fine-Tuning Effects on Layer-Wise Dynamics

A key sub-question: **how does task-specific fine-tuning reshape the layer-wise accuracy curve?**

Three hypotheses to test:

- **H1 -- Early shift:** Fine-tuning moves the plateau earlier, compressing the discriminative signal into shallower layers. This would imply fine-tuning makes the model more "efficient" at the task.
- **H2 -- Late boost:** Fine-tuning primarily improves performance in the final layers while leaving early/mid layers largely unchanged. This would suggest the later layers are doing task-specific adaptation.
- **H3 -- Uniform lift:** Fine-tuning improves accuracy across all layers roughly equally, indicating a distributed rather than localized effect.

**Experimental approach:**
- Extract layer-wise accuracy curves at multiple training checkpoints (every N steps)
- Compare pre-trained vs fine-tuned vs partially fine-tuned (LoRA at different layer ranges)
- Track whether the plateau layer shifts, the plateau height changes, or both

### Implications

- **Pruning guidance:** If layers beyond a plateau point contribute negligibly, they can be removed for 30--50% inference speedup with minimal accuracy loss.
- **Architecture design:** Understanding where classification signals form could inform future model design (e.g., narrower late layers, early exits).
- **Fine-tuning strategy:** If fine-tuning shifts signals earlier, targeted LoRA on early layers may be sufficient (cheaper, faster).
- **Universality:** If the phenomenon is consistent across models and tasks, it reveals something fundamental about how transformer depth relates to classification complexity.

---

## Collaboration Notes

- **Discord thread discussions** are the primary async communication channel.
- Experiment tracking: W&B project [`lid-bench`](https://wandb.ai/cataluna84/lid-bench) + local per-step reports in `experiments/`.
- All notebooks go in `notebooks/` and are stripped of outputs before commit (via `nbstripout`).
