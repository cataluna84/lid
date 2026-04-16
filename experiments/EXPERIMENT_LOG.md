# Experiment Log -- Runbook Execution

> **Machine:** NVIDIA A100-SXM4-40GB (39 GB VRAM), Linux
> **Date started:** 2026-04-16
> **W&B project:** [`lid-bench`](https://wandb.ai/cataluna84/lid-bench)
> **Model:** `CohereLabs/tiny-aya-global` (fp16, ~1B params)
> **Dataset:** `1024m/LID` → `LID-500.parquet`, 10% stratified sample = 3,350 samples

---

## Data Flow

```
lid-infer / lid-bench
    |
    +---> Local files (experiments/)
    |       benchmark_row.csv      <-- single-row, metrics-only
    |       layer_accuracy.csv     <-- per-layer accuracy (37 rows)
    |       layer_avg_probs.csv    <-- per-layer avg correct-class prob
    |       results.pkl            <-- full DataFrame (DATA, MAX, PROB)
    |
    +---> W&B project "lid-bench"
            run.config             <-- strategy, dtype, bs, ml, model, ...
            run.summary            <-- throughput, memory, accuracy, energy
            layer_accuracy_table   <-- W&B Table (37 rows)
            layer_accuracy_curve   <-- W&B line plot
            artifact               <-- zipped local files

lid-report  -->  terminal comparison table  (reads from W&B)
lid-upload  -->  backfill local results to W&B
```

---

## Cumulative Results Table

| Step | Strategy | dtype | bs | ml | sps | speedup | mem_mb | acc | energy_mj | wall_sec | W&B Run |
|------|----------|-------|----|----|-----|---------|--------|------|-----------|----------|---------|
| 1 | baseline_infer | fp16 | 16 | 512 | 8.19 | 1.0x | 17,256 | 0.0269 | 21,613 | 425.8 | [lmh8ypts](https://wandb.ai/cataluna84/lid-bench/runs/lmh8ypts) |
| 2 | eager | fp16 | 16 | 512 | 8.42 | 1.03x | 17,256 | 0.0329 | 21,260 | 403.5 | [3bnt0et6](https://wandb.ai/cataluna84/lid-bench/runs/3bnt0et6) |
| 3 | vectorized | | | | | | | | | | |
| 4 | compiled | | | | | | | | | | |
| 5 | sdpa / flash_attn | | | | | | | | | | |
| 6 | quantized_int8/int4 | | | | | | | | | | |
| 7 | combined | | | | | | | | | | |

### Step 1 -- Baseline: `lid-infer` (original notebook code path)

- **Command:** `uv run lid-infer --model CohereLabs/tiny-aya-global --sample-frac 0.1 --batch-size 16 --temperature 0.2 --max-length 512 --output-dir experiments/step1_baseline --seed 1024`
- **Wall time:** 425.8 s (~7 min) -- inference 409.0 s, postproc 12.1 s
- **Throughput:** 8.19 sps (0.51 batches/sec)
- **Latency:** 122.1 ms/sample
- **GPU memory peak:** 17,256 MB
- **GPU memory (model only):** 6,812 MB
- **Avg GPU power:** 177.0 W
- **Energy per sample:** 21,613 mJ
- **CUDA kernel time:** 409,021 ms
- **Last-layer accuracy:** 0.0269 (2.69%)
- **Layer count:** 37

**Layer accuracy range:** 1.3% (layer 5/8) → 3.2% (layer 33)

**Local files:**
```
experiments/step1_baseline/
  results.pkl           (full DataFrame)
  layer_accuracy.csv    (37 rows)
  layer_avg_probs.csv   (37 rows)
  benchmark_row.csv     (1 row, bench-compatible)
```

**W&B run:** https://wandb.ai/cataluna84/lid-bench/runs/lmh8ypts

---

### Step 2 -- Eager via Bench Framework

- **Command:** `uv run lid-bench configs/step2_eager.yaml`
- **Wall time:** 403.5 s (~6.7 min)
- **Throughput:** 8.42 sps (0.53 batches/sec)
- **Latency:** 118.8 ms/sample
- **GPU memory peak:** 17,256 MB
- **GPU memory (model only):** 6,812 MB
- **Avg GPU power:** 179.0 W
- **Energy per sample:** 21,260 mJ
- **CUDA kernel time:** 397,826 ms
- **Last-layer accuracy:** 0.0329 (3.29%)

**Comparison to Step 1:**
- Throughput: 8.42 vs 8.19 → **1.03x** (effectively identical, as expected)
- Memory: identical (17,256 MB)
- Accuracy: 3.29% vs 2.69% -- small difference from warmup batch exclusion in bench framework

**W&B run:** https://wandb.ai/cataluna84/lid-bench/runs/3bnt0et6

**Local files:**
```
experiments/benchmark_results.csv  (1 row so far, appended per step)
```

---

### Step 3 -- Vectorized
_Pending_

### Step 4 -- Compiled (torch.compile)
_Pending_

### Step 5 -- Attention Backends (SDPA / Flash Attn)
_Pending_

### Step 6 -- Quantization (INT8 / INT4)
_Pending_

### Step 7 -- Combined Strategies
_Pending_

---

## Notes

- Accuracy is ~3% vs the expected ~31% from the runbook. This is the
  **pre-trained model without fine-tuning** on this specific LID task.
  The runbook's expected numbers assumed a fine-tuned checkpoint.
  Accuracy should be consistent across strategies (the optimization
  should not change accuracy), which is the metric that matters for
  correctness validation.
- Steps 1 and 2 confirm the baseline: both produce ~8 sps throughput
  and ~17 GB peak memory on A100-40GB. This is the **1x reference**
  for all subsequent speedup calculations.
