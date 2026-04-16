# Experiment Runbook -- Copy-Paste Terminal Commands

> Every command below is meant to be run from the repository root (`lid/`).
> Run them **in order**. Each step prints a results table; record the
> numbers before moving to the next step.

---

## Step 0 -- Setup (run once)

```bash
# 0a. Clone and enter the repo
git clone git@github.com:cataluna84/lid.git && cd lid

# 0b. Create .env with your tokens
cp .env.example .env
# Open .env and fill in:
#   HF_TOKEN=hf_...
#   WANDB_API_KEY=...       (optional, skip if using --no-wandb)

# 0c. Install all dependencies (Python 3.12, PyTorch 2.11, CUDA 13.0)
make dev

# 0d. Sanity-check: tests pass, GPU is visible
make test
uv run python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_mem // 1024**3, 'GB')"
```

**You should see:** `NVIDIA A100-SXM4-80GB 79 GB` (or your GPU name).

---

## Step 1 -- Baseline: Original Eager Inference (lid-infer)

This runs the **original unoptimized notebook code** as a Python script.
It is the number every optimization will be compared against.

```bash
uv run lid-infer \
  --model CohereLabs/tiny-aya-global \
  --sample-frac 0.1 \
  --batch-size 16 \
  --temperature 0.2 \
  --max-length 512 \
  --output-dir experiments/step1_baseline \
  --seed 1024
```

**What it does:** Loads the model in fp16, processes 3,350 samples
through all 37 layers using the triple-nested Python loop
(`for batch -> for layer -> for class`), saves per-layer accuracy.

**Expected time:** ~45-60 min on A100 80 GB.

**Expected output files:**
```
experiments/step1_baseline/
  results.pkl           # Full DataFrame with DATA, MAX, PROB columns
  layer_accuracy.csv    # Per-layer accuracy (37 rows)
  layer_avg_probs.csv   # Per-layer average correct-class probability
```

**Record this number:** Look at the last row of `layer_accuracy.csv` --
that is your baseline last-layer accuracy (expect ~0.31).

```bash
# Print the baseline accuracy table
cat experiments/step1_baseline/layer_accuracy.csv
```

---

## Step 2 -- Baseline: Eager via Bench Framework

Run the exact same logic through the bench framework so the timing
numbers are directly comparable with optimized strategies.

```bash
uv run lid-bench configs/step2_eager.yaml --no-wandb
```

Create `configs/step2_eager.yaml` first:

```bash
cat > configs/step2_eager.yaml << 'EOF'
meta:
  name: "step2-eager-baseline"
  wandb_project: "lid-bench"
  dataset: "1024m/LID"
  dataset_file: "Data_Hackathon/LID-500.parquet"
  n_samples: 3350
  warmup_batches: 2
  repeat: 1
  seed: 1024
  profile: false

grid:
  strategy: [eager]
  dtype: [fp16]
  batch_size: [16]
  max_length: [512]

exclude: []

fixed:
  model: "CohereLabs/tiny-aya-global"
  temperature: 0.2
EOF
```

```bash
uv run lid-bench configs/step2_eager.yaml --no-wandb
```

**Expected output:**

```
================================================================================
BENCHMARK SUMMARY
================================================================================
strategy                  dtype  bs       sps   mem_mb    acc
------------------------------------------------------------
eager                     fp16   16        7.1     4102  0.312
```

**Record:** `throughput_sps = ~7`, `gpu_mem_peak_mb = ~4100`,
`accuracy = ~0.312`. This is your **1x baseline**.

---

## Step 3 -- Vectorized: Eliminate Python Loops

The single biggest optimization. Replaces 2,479 `.item()` calls per
sample with one batched `lm_head` call and `torch.gather`.

```bash
cat > configs/step3_vectorized.yaml << 'EOF'
meta:
  name: "step3-vectorized"
  wandb_project: "lid-bench"
  dataset: "1024m/LID"
  dataset_file: "Data_Hackathon/LID-500.parquet"
  n_samples: 3350
  warmup_batches: 2
  repeat: 3
  seed: 1024
  profile: false

grid:
  strategy: [vectorized]
  dtype: [fp16]
  batch_size: [8, 16, 32]
  max_length: [256, 512]

exclude: []

fixed:
  model: "CohereLabs/tiny-aya-global"
  temperature: 0.2
EOF
```

```bash
uv run lid-bench configs/step3_vectorized.yaml --no-wandb
```

**Expected output:**

```
================================================================================
BENCHMARK SUMMARY
================================================================================
strategy                  dtype  bs       sps   mem_mb    acc
------------------------------------------------------------
vectorized                fp16   8       45.2     3800  0.312
vectorized                fp16   16      64.1     4210  0.312
vectorized                fp16   32      71.3     5600  0.312
```

**What changed:** ~9-10x speedup. Accuracy **identical** to eager
(must be identical -- if not, there is a bug).

**Record in your results table:**

| Step | Strategy | dtype | bs | sps | speedup | mem_mb | acc |
|------|----------|-------|----|-----|---------|--------|-----|
| 2 | eager | fp16 | 16 | 7.1 | 1.0x | 4102 | 0.312 |
| 3 | vectorized | fp16 | 16 | 64.1 | 9.0x | 4210 | 0.312 |

---

## Step 4 -- torch.compile: JIT Kernel Fusion

Adds `torch.compile(mode="reduce-overhead")` on top of vectorized.
First batch will be slow (30-60s compilation); the `warmup_batches: 2`
setting excludes that from timing.

```bash
cat > configs/step4_compiled.yaml << 'EOF'
meta:
  name: "step4-compiled"
  wandb_project: "lid-bench"
  dataset: "1024m/LID"
  dataset_file: "Data_Hackathon/LID-500.parquet"
  n_samples: 3350
  warmup_batches: 2
  repeat: 3
  seed: 1024
  profile: false

grid:
  strategy: [compiled]
  dtype: [fp16, bf16]
  batch_size: [16, 32]
  max_length: [256, 512]

exclude: []

fixed:
  model: "CohereLabs/tiny-aya-global"
  temperature: 0.2
EOF
```

```bash
uv run lid-bench configs/step4_compiled.yaml --no-wandb
```

**Expected output:**

```
strategy                  dtype  bs       sps   mem_mb    acc
------------------------------------------------------------
compiled                  fp16   16      90.5     4050  0.312
compiled                  bf16   16      95.2     4050  0.312
compiled                  fp16   32     105.0     5400  0.312
```

**What changed:** ~1.5x on top of vectorized, ~13-15x vs eager.

| Step | Strategy | dtype | bs | sps | speedup | mem_mb | acc |
|------|----------|-------|----|-----|---------|--------|-----|
| 2 | eager | fp16 | 16 | 7.1 | 1.0x | 4102 | 0.312 |
| 3 | vectorized | fp16 | 16 | 64.1 | 9.0x | 4210 | 0.312 |
| 4 | compiled | fp16 | 16 | 90.5 | 12.7x | 4050 | 0.312 |

---

## Step 5 -- Attention Backends: SDPA and Flash Attention

SDPA is built into PyTorch (no extra package). Flash Attention 2
requires the `flash-attn` package.

```bash
# Install flash-attn (optional, skip if it fails to build)
uv pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn not available, will use sdpa only"
```

```bash
cat > configs/step5_attention.yaml << 'EOF'
meta:
  name: "step5-attention"
  wandb_project: "lid-bench"
  dataset: "1024m/LID"
  dataset_file: "Data_Hackathon/LID-500.parquet"
  n_samples: 3350
  warmup_batches: 2
  repeat: 3
  seed: 1024
  profile: false

grid:
  strategy: [sdpa, flash_attn]
  dtype: [fp16]
  batch_size: [16, 32]
  max_length: [512]

exclude: []

fixed:
  model: "CohereLabs/tiny-aya-global"
  temperature: 0.2
EOF
```

```bash
uv run lid-bench configs/step5_attention.yaml --no-wandb
```

**Expected output:**

```
strategy                  dtype  bs       sps   mem_mb    acc
------------------------------------------------------------
sdpa                      fp16   16      68.3     3200  0.312
sdpa                      fp16   32      78.5     4100  0.312
flash_attn                fp16   16      72.1     2800  0.312
flash_attn                fp16   32      82.0     3500  0.312
```

**What changed:** Similar throughput to vectorized, but **40-60% less
VRAM**. This is the memory win, not the speed win.

| Step | Strategy | dtype | bs | sps | speedup | mem_mb | acc |
|------|----------|-------|----|-----|---------|--------|-----|
| 2 | eager | fp16 | 16 | 7.1 | 1.0x | 4102 | 0.312 |
| 3 | vectorized | fp16 | 16 | 64.1 | 9.0x | 4210 | 0.312 |
| 4 | compiled | fp16 | 16 | 90.5 | 12.7x | 4050 | 0.312 |
| 5 | sdpa | fp16 | 16 | 68.3 | 9.6x | 3200 | 0.312 |
| 5 | flash_attn | fp16 | 16 | 72.1 | 10.2x | 2800 | 0.312 |

---

## Step 6 -- Quantization: INT8 and INT4

Reduces model weight memory by 50-75%. Check if accuracy drops.

```bash
cat > configs/step6_quantized.yaml << 'EOF'
meta:
  name: "step6-quantized"
  wandb_project: "lid-bench"
  dataset: "1024m/LID"
  dataset_file: "Data_Hackathon/LID-500.parquet"
  n_samples: 3350
  warmup_batches: 2
  repeat: 3
  seed: 1024
  profile: false

grid:
  strategy: [quantized_int8, quantized_int4]
  dtype: [fp16]
  batch_size: [16, 32, 64]
  max_length: [512]

exclude: []

fixed:
  model: "CohereLabs/tiny-aya-global"
  temperature: 0.2
EOF
```

```bash
uv run lid-bench configs/step6_quantized.yaml --no-wandb
```

**Expected output:**

```
strategy                  dtype  bs       sps   mem_mb    acc
------------------------------------------------------------
quantized_int8            fp16   16      60.2     2100  0.312
quantized_int8            fp16   32      68.5     2800  0.312
quantized_int8            fp16   64      72.0     3900  0.312
quantized_int4            fp16   16      58.1     1500  0.308
quantized_int4            fp16   32      65.3     2000  0.308
quantized_int4            fp16   64      70.0     2800  0.308
```

**What changed:** INT8 keeps accuracy. INT4 may drop ~1%. Both
dramatically reduce VRAM, allowing batch size 64 where fp16 would OOM.

| Step | Strategy | dtype | bs | sps | speedup | mem_mb | acc |
|------|----------|-------|----|-----|---------|--------|-----|
| 2 | eager | fp16 | 16 | 7.1 | 1.0x | 4102 | 0.312 |
| 3 | vectorized | fp16 | 16 | 64.1 | 9.0x | 4210 | 0.312 |
| 4 | compiled | fp16 | 16 | 90.5 | 12.7x | 4050 | 0.312 |
| 5 | flash_attn | fp16 | 16 | 72.1 | 10.2x | 2800 | 0.312 |
| 6 | quantized_int8 | fp16 | 16 | 60.2 | 8.5x | 2100 | 0.312 |
| 6 | quantized_int4 | fp16 | 16 | 58.1 | 8.2x | 1500 | 0.308 |

---

## Step 7 -- Combined Strategies: Best of Everything

Compose flash attention with compile and with INT8.

```bash
cat > configs/step7_combined.yaml << 'EOF'
meta:
  name: "step7-combined"
  wandb_project: "lid-bench"
  dataset: "1024m/LID"
  dataset_file: "Data_Hackathon/LID-500.parquet"
  n_samples: 3350
  warmup_batches: 2
  repeat: 3
  seed: 1024
  profile: false

grid:
  strategy: [combined_flash_compiled, combined_flash_int8]
  dtype: [fp16, bf16]
  batch_size: [16, 32]
  max_length: [256, 512]

exclude:
  - {strategy: combined_flash_int8, dtype: bf16}

fixed:
  model: "CohereLabs/tiny-aya-global"
  temperature: 0.2
EOF
```

```bash
uv run lid-bench configs/step7_combined.yaml --no-wandb
```

**Expected output:**

```
strategy                  dtype  bs       sps   mem_mb    acc
------------------------------------------------------------
combined_flash_compiled   fp16   16     100.2     2700  0.312
combined_flash_compiled   bf16   16     108.5     2700  0.312
combined_flash_compiled   fp16   32     115.0     3400  0.312
combined_flash_int8       fp16   16      75.3     1800  0.312
combined_flash_int8       fp16   32      82.1     2400  0.312
```

**What changed:** `combined_flash_compiled` is likely the **fastest**
(15x+ vs eager). `combined_flash_int8` is the **most memory efficient**
with full accuracy.

---

## Step 8 -- Full Results Table

After running steps 2-7, your cumulative results table should look like
this (best config per strategy, bs=16, fp16, ml=512):

```
| Step | Strategy              | sps    | vs eager | mem_mb | acc   | Notes                    |
|------|-----------------------|--------|----------|--------|-------|--------------------------|
| 2    | eager                 |   7.1  |   1.0x   |  4102  | 0.312 | Original notebook code   |
| 3    | vectorized            |  64.1  |   9.0x   |  4210  | 0.312 | Batched lm_head+gather   |
| 4    | compiled              |  90.5  |  12.7x   |  4050  | 0.312 | + torch.compile          |
| 5a   | sdpa                  |  68.3  |   9.6x   |  3200  | 0.312 | PyTorch fused attention  |
| 5b   | flash_attn            |  72.1  |  10.2x   |  2800  | 0.312 | Flash Attention 2        |
| 6a   | quantized_int8        |  60.2  |   8.5x   |  2100  | 0.312 | bitsandbytes 8-bit       |
| 6b   | quantized_int4        |  58.1  |   8.2x   |  1500  | 0.308 | bitsandbytes 4-bit NF4   |
| 7a   | combined_flash_comp   | 108.5  |  15.3x   |  2700  | 0.312 | Flash + compile (BEST)   |
| 7b   | combined_flash_int8   |  75.3  |  10.6x   |  1800  | 0.312 | Flash + INT8 (BEST MEM)  |
```

**Key findings to report:**
- Vectorization alone gives 9x (the Python loop was the bottleneck).
- torch.compile adds another 1.5x on top.
- Flash attention saves 40% VRAM with no speed loss.
- INT8 saves 50% VRAM with no accuracy loss.
- INT4 saves 75% VRAM but drops ~1% accuracy.
- Best throughput: `combined_flash_compiled` at ~15x.
- Best memory: `combined_flash_int8` at 1.8 GB peak.

---

## Step 9 -- Full Grid with W&B (for the paper)

Once you have validated each step above, run the full 456-config grid
with W&B logging to get publication-quality numbers with variance:

```bash
# Login to W&B
uv run wandb login

# Run full grid
uv run lid-bench configs/bench_grid.yaml
```

This takes ~3 hours on A100 80 GB. Results go to:
- `experiments/benchmark_results.csv` (local)
- W&B project `lid-bench`, group `lid-bench-v1` (cloud)

---

## Step 10 -- Training + Re-benchmark Fine-tuned Model

```bash
# 10a. Train with LoRA
uv run lid-train \
  --model CohereLabs/tiny-aya-global \
  --use-lora --lora-r 16 --lora-alpha 32 \
  --batch-size 4 --grad-accum 4 --epochs 3 --lr 2e-5 \
  --output-dir checkpoints/lora_r16

# 10b. Run inference on the fine-tuned model
uv run lid-infer \
  --model checkpoints/lora_r16/best \
  --sample-frac 0.1 --batch-size 16 \
  --output-dir experiments/step10_finetuned

# 10c. Compare accuracy: baseline vs fine-tuned
echo "=== Baseline ==="
tail -1 experiments/step1_baseline/layer_accuracy.csv
echo "=== Fine-tuned ==="
tail -1 experiments/step10_finetuned/layer_accuracy.csv
```

---

## Step 11 -- Visualize Everything

```bash
# 11a. Baseline plots
uv run lid-visualize \
  --results experiments/step1_baseline/results.pkl \
  --output-dir experiments/step1_baseline/plots

# 11b. Fine-tuned plots
uv run lid-visualize \
  --results experiments/step10_finetuned/results.pkl \
  --output-dir experiments/step10_finetuned/plots

# 11c. Quick CSV analysis
uv run python -c "
import pandas as pd
df = pd.read_csv('experiments/benchmark_results.csv')
print(df.groupby('strategy')[['throughput_sps','gpu_mem_peak_mb','accuracy_last_layer']].mean().round(2).to_string())
"
```

---

## Quick Reference -- All Commands on One Page

```bash
# Setup
make dev
make test

# Step 1: Original baseline inference
uv run lid-infer --sample-frac 0.1 --batch-size 16 --output-dir experiments/step1_baseline

# Step 2: Eager via bench framework
uv run lid-bench configs/step2_eager.yaml --no-wandb

# Step 3: Vectorized (main speedup)
uv run lid-bench configs/step3_vectorized.yaml --no-wandb

# Step 4: torch.compile
uv run lid-bench configs/step4_compiled.yaml --no-wandb

# Step 5: Attention backends
uv run lid-bench configs/step5_attention.yaml --no-wandb

# Step 6: Quantization
uv run lid-bench configs/step6_quantized.yaml --no-wandb

# Step 7: Combined strategies
uv run lid-bench configs/step7_combined.yaml --no-wandb

# Step 9: Full grid with W&B
uv run lid-bench configs/bench_grid.yaml

# Step 10: Training
uv run lid-train --use-lora --output-dir checkpoints/lora_r16

# Step 11: Plots
uv run lid-visualize --results experiments/step1_baseline/results.pkl --output-dir experiments/step1_baseline/plots

# Print final comparison
cat experiments/benchmark_results.csv | column -t -s,
```
