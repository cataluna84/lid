# LID -- Layer-Wise Language Identification

Mechanistic interpretability study of multilingual language identification
in compact foundation models (0--4B params). This repository contains
training, inference, optimization, and benchmarking pipelines for a
67-language classification task using hidden-state extraction across all
transformer layers.

**Paper target:** ARR May 2026 cycle.

> **Want to just run experiments?** Skip to
> [docs/RUNBOOK.md](docs/RUNBOOK.md) for copy-paste terminal commands
> with step-by-step configs (Steps 1-11) and expected results tables.

---

## Table of Contents

1. [Hardware Requirements](#1-hardware-requirements)
2. [Environment Setup](#2-environment-setup)
3. [Repository Map](#3-repository-map)
4. [End-to-End Experiment Guide](#4-end-to-end-experiment-guide)
   - [Phase 1: Baseline Inference](#phase-1-baseline-inference)
   - [Phase 2: Optimization Sweep](#phase-2-optimization-sweep)
   - [Phase 3: Training Experiments](#phase-3-training-experiments)
   - [Phase 4: Analysis and Visualization](#phase-4-analysis-and-visualization)
5. [Benchmark Framework](#5-benchmark-framework)
   - [Strategies Reference](#strategies-reference)
   - [Writing Custom Grid Configs](#writing-custom-grid-configs)
   - [Reading Results](#reading-results)
6. [W&B Integration](#6-wb-integration)
7. [Notebooks](#7-notebooks)
8. [FAQ and Troubleshooting](#8-faq-and-troubleshooting)

---

## 1. Hardware Requirements

### Model Specs: CohereLabs/tiny-aya-global

| Property | Value |
|----------|-------|
| Parameters | 3.35B (2.8B non-embedding) |
| Architecture | Dense decoder-only, Cohere2 |
| Layers | 36 + embedding = 37 hidden states |
| Hidden dim | 2048 |
| Vocab size | 262K |
| Native dtype | BF16 |
| fp16 weight size | ~6.7 GB |
| int8 weight size | ~3.6 GB |
| int4 weight size | ~2.1 GB |

### VRAM Budget (inference with `output_hidden_states=True`)

Extracting all 37 hidden states significantly increases memory because
PyTorch must retain every intermediate activation. The table below
estimates peak VRAM for the batch sizes used in our benchmark grid.

> **Note:** These are theoretical estimates. Actual measured peak VRAM
> on H100 80 GB was **70-76 GB** for quantized strategies (INT8/INT4)
> with `output_hidden_states=True` across all 37 layers, due to
> activation caching overhead.

| Precision | Batch Size 16, seq 512 | Batch Size 32, seq 512 | Batch Size 64, seq 512 |
|-----------|------------------------|------------------------|------------------------|
| fp16 | ~12 GB | ~18 GB | ~28 GB |
| int8 | ~8 GB | ~12 GB | ~20 GB |
| int4 | ~6 GB | ~9 GB | ~15 GB |

### GPU Recommendation

| GPU | VRAM | Can run full grid? | Cost (RunPod) | Recommendation |
|-----|------|--------------------|---------------|----------------|
| **RTX 4090** | 24 GB | Partial (bs <= 32 fp16) | ~$0.44/hr | Dev/debug only |
| **A100 40 GB** | 40 GB | Yes (all but bs64 fp16) | ~$1.19/hr | Good enough |
| **A100 80 GB** | 80 GB | Yes (full grid) | ~$1.39/hr | **Budget pick** |
| **H100 80 GB** | 80 GB | Yes + faster compile | ~$2.39/hr | **What we use** |
| **L40S 48 GB** | 48 GB | Yes (most configs) | ~$0.86/hr | Budget option |

**Our setup: H100 80 GB HBM3.** All experiments in this repository
were run on an H100 80 GB. The faster HBM3 bandwidth and improved
`torch.compile` performance make a noticeable difference for
layer-wise extraction across all 37 hidden states. **A100 80 GB** is
a good budget alternative -- it handles the full grid comfortably at
roughly half the hourly cost.

For a complete benchmark run (456 configs, ~2 hours on H100 80 GB):

```
Estimated cost: 456 configs * ~15 sec/config = ~1.9 hours
H100 80 GB @ $2.39/hr = ~$4.55 total
```

---

## 2. Environment Setup

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) package manager
- CUDA-compatible GPU (see table above)
- HuggingFace account with access to `1024m/LID` dataset
- (Optional) [W&B](https://wandb.ai) account for experiment tracking

### Install

```bash
# Clone
git clone git@github.com:cataluna84/lid.git && cd lid

# Copy environment template and add your tokens
cp .env.example .env
# Edit .env:
#   HF_TOKEN=hf_your_token_here
#   WANDB_API_KEY=your_wandb_key_here   (optional)
#   WANDB_PROJECT=lid-bench              (optional)

# Install everything (Python 3.12, PyTorch 2.11 + CUDA 12.8, all deps)
make dev

# (Optional) Install flash-attn for Flash Attention 2 support
# Pre-built wheels are available for CUDA 12.8:
uv pip install flash-attn --no-build-isolation

# Verify
make test
```

### Verify GPU access

```bash
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

---

## 3. Repository Map

```
lid/
├── src/lid/                        # Core library
│   ├── constants.py                #   67 language mappings, defaults
│   ├── data.py                     #   Dataset loading, prompt construction, CommonLID
│   ├── model.py                    #   Model loading, layer-wise inference functions
│   ├── train.py                    #   Fine-tuning pipeline (LoRA, mixed precision)
│   ├── infer.py                    #   Batched layer-wise inference (original)
│   ├── visualize.py                #   Plotting: accuracy curves, per-language plots
│   └── bench/                      #   Optimization + benchmarking framework
│       ├── configs.py              #     RunConfig / ExperimentGrid dataclasses
│       ├── strategy.py             #     InferenceStrategy ABC + registry
│       ├── strategies/             #     9 registered strategies (see below)
│       │   ├── eager.py            #       Baseline: original triple-loop
│       │   ├── vectorized.py       #       Fully batched tensor ops (~10x)
│       │   ├── compiled.py         #       torch.compile + vectorized
│       │   ├── quantized.py        #       INT8 / INT4 via bitsandbytes
│       │   ├── flash_attn.py       #       Flash Attention 2 / SDPA
│       │   └── combined.py         #       Flash+Compile, Flash+INT8
│       ├── metrics.py              #     3-tier metrics collector
│       ├── wandb_logger.py         #     W&B integration wrapper
│       ├── local_logger.py         #     Local filesystem results logger
│       └── runner.py               #     Grid expansion + execution engine
├── configs/                        # Experiment configurations
│   ├── base.yaml                   #   Training defaults
│   ├── exp_4bit.yaml               #   4-bit quantization experiment
│   ├── bench_grid.yaml             #   Full benchmark grid (456 runs)
│   └── wandb_sweep.yaml            #   W&B Sweep alternative
├── notebooks/                      # Exploration notebooks
│   ├── LID_Inference_Vibecoded.ipynb  # Layer-wise inference (original Colab)
│   ├── LID_Regex.ipynb               # Unicode block heuristic classifier
│   ├── LID_Ngrams_Classifier.ipynb    # N-gram based LID classifier
│   ├── LID_Unicode_Blocks_Classifier.ipynb  # Unicode block + regex classifier
│   └── LID_Embedding_Classifier.ipynb # Embedding model (0.6B) classifier
├── docs/
│   ├── project_proposal.md         # Project proposal, research scope, questions, related work
│   └── optimization_spec.md        # Optimization strategy specification
├── experiments/                    # Output directory for results
├── tests/                          # Unit tests
├── pyproject.toml                  # UV project config
├── Makefile                        # Common commands
├── CONTRIBUTING.md                 # Development guide
└── .env.example                    # Token template
```

### CLI Entrypoints

| Command | Script | Purpose |
|---------|--------|---------|
| `lid-train` | `src/lid/train.py` | Fine-tune model with LoRA |
| `lid-infer` | `src/lid/infer.py` | Run layer-wise inference (original) |
| `lid-visualize` | `src/lid/visualize.py` | Plot results from inference |
| `lid-bench` | `src/lid/bench/runner.py` | Run optimization benchmark grid |
| `lid-report` | `src/lid/report.py` | Generate comparison report from W&B |
| `lid-recommend` | `src/lid/recommend.py` | Recommend best inference config from benchmarks |
| `lid-upload` | `src/lid/upload.py` | Backfill local results to W&B |

---

## 4. End-to-End Experiment Guide

This section is the step-by-step path from zero to published results.
Follow phases in order. Each phase builds on the previous.

### Phase 1: Baseline Inference

**Goal:** Run the original (unoptimized) inference pipeline to get
baseline accuracy numbers and timing.

```bash
# Run baseline inference on 10% of the LID-500 dataset (3,350 samples)
lid-infer \
  --model CohereLabs/tiny-aya-global \
  --sample-frac 0.1 \
  --batch-size 16 \
  --temperature 0.2 \
  --max-length 512 \
  --output-dir experiments/baseline \
  --seed 1024

# This produces:
#   experiments/baseline/results.pkl      -- Full results DataFrame
#   experiments/baseline/layer_accuracy.csv -- Per-layer accuracy
#   experiments/baseline/layer_avg_probs.csv -- Per-layer probabilities
```

**Expected runtime:** ~4 min on H100 80 GB (eager baseline). On A100,
expect ~45-60 minutes (this is the unoptimized eager-mode triple-loop
code).

**Expected results:** Layer accuracy rises from ~5% at layer 1 to ~31%
at layer 37. This is the number we want to reproduce and then speed up.

```bash
# Visualize baseline results
lid-visualize --results experiments/baseline/results.pkl --output-dir experiments/baseline/plots
```

### Phase 2: Optimization Sweep

**Goal:** Run the benchmark framework to compare all optimization
strategies and find the fastest one with identical accuracy.

#### Quick start: Run a small grid first

Create a small test grid to verify everything works:

```yaml
# configs/bench_quick.yaml
meta:
  name: "quick-test"
  wandb_project: "lid-bench"
  dataset: "1024m/LID"
  dataset_file: "Data_Hackathon/LID-500.parquet"
  n_samples: 500       # small sample for quick test
  warmup_batches: 1
  repeat: 1            # single run per config
  seed: 1024
  profile: false

grid:
  strategy: [eager, vectorized]
  dtype: [fp16]
  batch_size: [16]
  max_length: [512]

exclude: []

fixed:
  model: "CohereLabs/tiny-aya-global"
  temperature: 0.2
```

```bash
# Run the quick test (should take ~5 minutes)
lid-bench configs/bench_quick.yaml --no-wandb

# Expected output (H100 80 GB):
# ================================================================================
# BENCHMARK SUMMARY
# ================================================================================
# strategy                  dtype  bs       sps   mem_mb    acc
# ------------------------------------------------------------
# eager                     fp16   16       14.0     4102  0.312
# vectorized                fp16   16       70.0     4210  0.312
#                                          ^^^^
#                              ~5x speedup, same accuracy
```

#### Full benchmark: All strategies

```bash
# Run the full grid with W&B logging
lid-bench configs/bench_grid.yaml

# Or without W&B
lid-bench configs/bench_grid.yaml --no-wandb
```

This expands to 456 runs (9 strategies x 2 dtypes x 5 batch sizes x
2 max lengths, minus exclusions, x 3 repeats). Results are saved to
`experiments/benchmark_results.csv`.

#### Strategy-by-strategy: Run individual strategies

If you want to test strategies one at a time:

```yaml
# configs/bench_vectorized_only.yaml
meta:
  name: "vectorized-only"
  wandb_project: "lid-bench"
  dataset: "1024m/LID"
  dataset_file: "Data_Hackathon/LID-500.parquet"
  n_samples: 3350
  warmup_batches: 2
  repeat: 3
  seed: 1024
  profile: false

grid:
  strategy: [vectorized]          # <-- single strategy
  dtype: [fp16, bf16]
  batch_size: [8, 16, 32]
  max_length: [256, 512]

exclude: []

fixed:
  model: "CohereLabs/tiny-aya-global"
  temperature: 0.2
```

```bash
lid-bench configs/bench_vectorized_only.yaml
```

#### Recommended execution order

Run strategies incrementally so you can compare each improvement:

| Step | Config | What it tests | Expected result (H100) |
|------|--------|---------------|------------------------|
| 1 | `strategy: [eager]` | Baseline timing | ~14 samples/sec |
| 2 | `strategy: [vectorized]` | Remove Python loops | ~70-133 sps (~5-10x) |
| 3 | `strategy: [compiled]` | Add torch.compile | ~105-186 sps (~1.5x on top) |
| 4 | `strategy: [sdpa]` | Fused attention | ~80 sps + less memory |
| 5 | `strategy: [flash_attn]` | Flash Attention 2 | ~85 sps + ~40% less memory |
| 6 | `strategy: [quantized_int8]` | INT8 weights | Similar sps, ~50% less memory |
| 7 | `strategy: [quantized_int4]` | INT4 weights | Check accuracy delta |
| 8 | `strategy: [combined_flash_compiled]` | Flash + compile | Best throughput |
| 9 | `strategy: [combined_flash_int8]` | Flash + INT8 | Best memory efficiency |

### Phase 3: Training Experiments

**Goal:** Fine-tune the model with LoRA and track layer-wise accuracy
changes during training.

```bash
# Basic LoRA fine-tuning
lid-train \
  --model CohereLabs/tiny-aya-global \
  --dataset 1024m/LID \
  --use-lora \
  --lora-r 16 \
  --lora-alpha 32 \
  --batch-size 4 \
  --grad-accum 4 \
  --epochs 3 \
  --lr 2e-5 \
  --output-dir checkpoints/lora_baseline

# 4-bit quantized training (uses less VRAM)
lid-train \
  --model CohereLabs/tiny-aya-global \
  --use-lora \
  --quantization 4bit \
  --batch-size 8 \
  --output-dir checkpoints/lora_4bit
```

After training, run inference on the fine-tuned checkpoint:

```bash
lid-infer \
  --model checkpoints/lora_baseline/best \
  --output-dir experiments/finetuned_baseline
```

### Phase 4: Analysis and Visualization

```bash
# Generate all plots from baseline
lid-visualize \
  --results experiments/baseline/results.pkl \
  --output-dir experiments/baseline/plots

# Generates:
#   global_avg_prob.png     -- Average correct-class probability across layers
#   layer_accuracy.png      -- Classification accuracy vs layer depth
#   lang_plots/             -- Individual per-language plots (67 files)
```

#### Comparing strategies in the CSV

```python
import pandas as pd

df = pd.read_csv("experiments/benchmark_results.csv")

# Best throughput per strategy
print(df.groupby("strategy")["throughput_sps"].max().sort_values(ascending=False))

# Accuracy should be constant across strategies (sanity check)
print(df.groupby("strategy")["accuracy_last_layer"].mean())

# Memory vs throughput tradeoff
print(df[["strategy", "dtype", "batch_size", "throughput_sps", "gpu_mem_peak_mb"]]
      .sort_values("throughput_sps", ascending=False)
      .head(20))
```

---

## 5. Benchmark Framework

### Strategies Reference

| Name | Key | What it does | Expected speedup |
|------|-----|-------------|-----------------|
| Eager | `eager` | Original triple-loop code (baseline) | 1x |
| Vectorized | `vectorized` | Batched `lm_head` + `torch.gather`, zero Python loops | ~10x |
| Compiled | `compiled` | Vectorized + `torch.compile(mode="reduce-overhead")` | ~15x |
| SDPA | `sdpa` | Vectorized + PyTorch scaled dot-product attention | ~10x + 40% less mem |
| Flash Attention | `flash_attn` | Vectorized + Flash Attention 2 kernel | ~10x + 60% less mem |
| INT8 | `quantized_int8` | Vectorized + bitsandbytes 8-bit weights | ~10x + 50% less mem |
| INT4 | `quantized_int4` | Vectorized + bitsandbytes 4-bit NF4 | ~10x + 75% less mem |
| Flash + Compile | `combined_flash_compiled` | Flash Attention + torch.compile | ~15x + 60% less mem |
| Flash + INT8 | `combined_flash_int8` | Flash Attention + INT8 quantization | ~10x + best mem |

### Writing Custom Grid Configs

Every benchmark is driven by a YAML file with this structure:

```yaml
meta:
  name: "my-experiment"          # Shows in W&B as group name
  wandb_project: "lid-bench"     # W&B project (set to "" to skip)
  dataset: "1024m/LID"           # HuggingFace dataset
  dataset_file: "Data_Hackathon/LID-500.parquet"  # Specific file
  n_samples: 3350                # Number of samples to process
  warmup_batches: 2              # Excluded from timing
  repeat: 3                      # Runs per config (for variance)
  seed: 1024                     # Random seed (incremented per repeat)
  profile: false                 # Enable torch.profiler traces

grid:
  strategy: [eager, vectorized]  # Cartesian product of these axes
  dtype: [fp16, bf16]
  batch_size: [8, 16, 32]
  max_length: [256, 512]

exclude:                         # Skip invalid combinations
  - {strategy: eager, batch_size: 64}

fixed:                           # Constants across all runs
  model: "CohereLabs/tiny-aya-global"
  temperature: 0.2
```

The runner computes the cartesian product of all `grid` axes, removes
any combos matching `exclude` rules, multiplies by `repeat`, and
executes each config sequentially.

**Total runs** = (|strategy| x |dtype| x |batch_size| x |max_length| - |exclude|) x repeat

### Reading Results

Each experiment step writes results to a timestamped directory:

```
experiments/
├── {step-name}/                    # e.g. "eager-fp16-bs16"
│   ├── {timestamp}/                # e.g. "20260415_143022"
│   │   ├── benchmark_results.csv   # Per-run metrics for this step
│   │   ├── config.yaml             # Exact config used
│   │   ├── platform.json           # GPU, driver, PyTorch versions
│   │   └── REPORT.md               # Human-readable summary
│   └── latest -> {timestamp}/      # Symlink to most recent run
├── all_results.csv                 # Cumulative append-only CSV (all steps)
└── ...
```

The cumulative `experiments/all_results.csv` is append-only -- every
benchmark run appends its rows, so you always have a single file for
cross-step analysis.

The `latest` symlink in each step directory always points to the most
recent run, making it easy to inspect results without remembering
timestamps.

#### Recommending the best config

After running benchmarks, use `lid-recommend` to find the optimal
inference hyperparameters without digging through CSVs:

```bash
# Best throughput (default)
lid-recommend

# Best throughput among configs with accuracy >= 3%
lid-recommend --min-accuracy 0.03

# Optimize for energy or memory instead
lid-recommend --optimize energy --min-accuracy 0.03
lid-recommend --optimize memory

# Show top-5 configs
lid-recommend --top 5

# JSON output for scripting
lid-recommend --json

# Pull from W&B instead of local CSV
lid-recommend --from-wandb
```

This reads `experiments/all_results.csv`, groups by config, averages
across repeats, ranks by the chosen metric, and prints a ready-to-paste
`lid-infer` command with the winning hyperparameters.

#### CSV columns

| Column | Description |
|--------|-------------|
| `strategy` | Strategy name |
| `dtype` | fp16 or bf16 |
| `batch_size` | Batch size used |
| `max_length` | Max sequence length |
| `repeat_idx` | Repeat number (0, 1, 2) |
| `throughput_sps` | Samples per second |
| `latency_ms_per_sample` | Milliseconds per sample |
| `gpu_mem_peak_mb` | Peak GPU memory in MB |
| `energy_per_sample_mj` | Energy per sample in millijoules |
| `accuracy_last_layer` | Top-1 accuracy at the final layer |
| `total_wall_sec` | Total wall time in seconds |
| `inference_wall_sec` | Inference-only wall time |
| `cuda_kernel_time_ms` | GPU kernel time |
| `host_overhead_ms` | CPU-side overhead |

---

## 6. W&B Integration

### Setup

```bash
# Add your W&B key to .env
echo "WANDB_API_KEY=your_key_here" >> .env

# Or login interactively
uv run wandb login
```

### What gets logged

| W&B Feature | What we log |
|-------------|-------------|
| `run.config` | Strategy, dtype, batch_size, model, etc. |
| `run.log()` | Per-batch latency, throughput, GPU memory (streaming) |
| `run.summary` | Aggregate metrics: throughput, accuracy, energy, MFU |
| `wandb.Table` | Per-layer accuracy curves, benchmark summary table |
| `wandb.Artifact` | Grid YAML, results CSV, profiler traces |
| System metrics | GPU util, power, temp, memory (automatic, every 15s) |

### Viewing results

After a benchmark run, W&B provides:

- **Parallel Coordinates:** Compare strategy/dtype/batch_size axes
  against throughput/accuracy/memory in a single interactive chart.
- **Run comparison table:** Sort and filter all runs by any metric.
- **Layer accuracy curves:** Per-run line charts showing accuracy
  across all 37 layers.
- **System metrics:** GPU utilization, power draw, temperature
  timelines.

### Using W&B Sweeps (optional)

For distributed execution across multiple machines:

```bash
# Create the sweep
wandb sweep configs/wandb_sweep.yaml

# Start agents on each machine
wandb agent your-entity/lid-bench/SWEEP_ID
```

---

## 7. Notebooks

| Notebook | Description | Key output |
|----------|-------------|------------|
| `LID_Inference_Vibecoded.ipynb` | Original layer-wise inference pipeline on tiny-aya-global. Processes 3,350 samples through all 37 layers, extracts per-layer class probabilities for 67 languages. | `results.pkl`, accuracy curves |
| `LID_Regex.ipynb` | Unicode block-based heuristic classifier by Ram Mohan Rao Kadiyala. Extracts character block distributions for 1.2M texts, evaluates against CommonLID. | Block distribution analysis |
| `LID_Ngrams_Classifier.ipynb` | N-gram (1-5) based language classifier. Trained on 500/1000/2500 samples per language. Limited scaling for low-resource languages. | N-gram model comparisons |
| `LID_Unicode_Blocks_Classifier.ipynb` | Unicode block distribution + regex classifier. 20+ languages at 99%+ accuracy via regex alone for unique scripts. | Per-script accuracy breakdown |
| `LID_Embedding_Classifier.ipynb` | Embedding model (0.6B) classifier achieving macro F1 0.97+ on unseen domains with 900 samples per language. | F1 scores, domain transfer |

Both notebooks load HF tokens from `.env` via `python-dotenv` (no
Colab `userdata` dependency).

### Running notebooks locally

```bash
# The Jupyter kernel is registered during setup
jupyter lab

# Select kernel: "LID (Python 3.12)" or "Python 3 (ipykernel)"
```

---

## 8. FAQ and Troubleshooting

### "CUDA out of memory" during benchmark

Reduce batch size or use quantized strategies:

```yaml
grid:
  batch_size: [4, 8, 16]          # Remove 32 and 64
  strategy: [quantized_int8]      # Uses ~50% less VRAM
```

### "flash-attn not installed" error

Flash Attention 2 requires the `flash-attn` package which needs a
CUDA-compatible build. Pre-built wheels are available for CUDA 12.8,
so installation is usually quick:

```bash
uv pip install flash-attn --no-build-isolation
```

If no pre-built wheel is found for your platform, it will build from
source (takes 10-30 minutes). If it fails to build, use `sdpa` instead
(built into PyTorch, no extra package needed):

```yaml
grid:
  strategy: [sdpa]   # Drop-in replacement, slightly slower than flash_attn
```

### torch.compile takes forever on first batch

This is expected. The first call to `torch.compile` JIT-compiles the
model (~30-60 seconds). The `warmup_batches` setting in the grid config
excludes these from timing. Subsequent batches run at compiled speed.

### W&B is not logging anything

Check that your `.env` has `WANDB_API_KEY` set. Or run:

```bash
uv run wandb login
```

To skip W&B entirely:

```bash
lid-bench configs/bench_grid.yaml --no-wandb
```

### How do I add a new model?

1. Verify it works with `AutoModelForCausalLM.from_pretrained()`.
2. Create a new grid config:

```yaml
fixed:
  model: "your-org/your-model"
```

3. Run the benchmark. The strategies are model-agnostic.

### How do I add a new optimization strategy?

1. Create `src/lid/bench/strategies/my_strategy.py`:

```python
from lid.bench.strategies.vectorized import VectorizedStrategy
from lid.bench.strategy import StrategyRegistry

@StrategyRegistry.register("my_strategy")
class MyStrategy(VectorizedStrategy):
    def load_model(self, model_name, dtype, device="cuda"):
        # Your custom model loading logic
        ...
```

2. Add it to `src/lid/bench/strategies/__init__.py`:

```python
import lid.bench.strategies.my_strategy
```

3. Use it in a grid config:

```yaml
grid:
  strategy: [my_strategy]
```
