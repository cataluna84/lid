# Experiment Log -- Runbook Execution

> **Machine:** NVIDIA H100 80GB HBM3 (79 GB VRAM), Linux
> **Date started:** 2026-04-16
> **W&B project:** [`lid-bench`](https://wandb.ai/cataluna84/lid-bench)
> **Model:** `CohereLabs/tiny-aya-global` (fp16, ~1B params)
> **Dataset:** `1024m/LID` → `LID-500.parquet`, 10% stratified sample = 3,350 samples
>
> **Status (2026-04-30):** the repository is **public** under
> Apache-2.0; this log is preserved verbatim as the recorded
> reference run. The W&B URLs below are read-only; reproducers
> should set `WANDB_ENTITY=<your-entity>` in `.env` so their own
> runs land in their own namespace. See
> [`docs/RUNBOOK.md`](../docs/RUNBOOK.md) for the step-by-step
> recipe and [`README.md`](../README.md#reproducibility-caveats)
> for the reproducibility caveats (hardware sensitivity, gated
> dataset access, etc.).

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
| 1 | baseline_infer | fp16 | 16 | 512 | 13.4 | 1.0x | 17,281 | 0.0272 | 18,726 | 250.1 | [nk88cizk](https://wandb.ai/cataluna84/lid-bench/runs/nk88cizk) |
| 2 | eager | fp16 | 16 | 512 | 14.0 | 1.04x | 17,281 | 0.0330 | 18,014 | 238.7 | [3am3znst](https://wandb.ai/cataluna84/lid-bench/runs/3am3znst) |
| 3 | vectorized | fp16 | 8 | 256 | 93.4 | 6.97x | 74,281 | 0.012 | 3,771 | ~36s | [et2ce9j9](https://wandb.ai/cataluna84/lid-bench/runs/et2ce9j9) |
| 3 | vectorized | fp16 | 8 | 512 | 62.8 | 4.69x | 76,889 | 0.033 | 7,084 | ~53s | [0gm67u4n](https://wandb.ai/cataluna84/lid-bench/runs/0gm67u4n) |
| 3 | vectorized | fp16 | 16 | 256 | 112.7 | 8.41x | 76,420 | 0.012 | 3,602 | ~29s | [i2jgf978](https://wandb.ai/cataluna84/lid-bench/runs/i2jgf978) |
| 3 | vectorized | fp16 | 16 | 512 | 71.5 | 5.34x | 81,638 | 0.033 | 6,824 | ~47s | [mjearq5c](https://wandb.ai/cataluna84/lid-bench/runs/mjearq5c) |
| 3 | vectorized | fp16 | 32 | 256 | 133.3 | 9.95x | 80,386 | 0.012 | 3,336 | ~25s | [vyn2bs6v](https://wandb.ai/cataluna84/lid-bench/runs/vyn2bs6v) |
| 3 | vectorized | fp16 | 32 | 512 | -- | OOM | 83,936 | -- | -- | -- | [7matvroe](https://wandb.ai/cataluna84/lid-bench/runs/7matvroe) |
| 4 | compiled | fp16 | 16 | 256 | 159.2 | 11.88x | 72,667 | 0.012 | 2,774 | ~21s | [sqvdx5v7](https://wandb.ai/cataluna84/lid-bench/runs/sqvdx5v7) |
| 4 | compiled | fp16 | 16 | 512 | 105.3 | 7.86x | 73,818 | 0.033 | 5,173 | ~31s | [mzz4ulrm](https://wandb.ai/cataluna84/lid-bench/runs/mzz4ulrm) |
| 4 | compiled | bf16 | 16 | 256 | 159.4 | 11.90x | 72,667 | 0.012 | 2,734 | ~21s | [bxv8gpcr](https://wandb.ai/cataluna84/lid-bench/runs/bxv8gpcr) |
| 4 | compiled | bf16 | 16 | 512 | 106.1 | 7.92x | 73,818 | 0.033 | 5,083 | ~31s | [pp09z5y0](https://wandb.ai/cataluna84/lid-bench/runs/pp09z5y0) |
| 4 | compiled | fp16 | 32 | 256 | 186.1 | 13.89x | 74,798 | 0.012 | 2,600 | ~18s | [iuu365zu](https://wandb.ai/cataluna84/lid-bench/runs/iuu365zu) |
| 4 | compiled | fp16 | 32 | 512 | -- | OOM | 70,571 | -- | -- | -- | [7lrlhktt](https://wandb.ai/cataluna84/lid-bench/runs/7lrlhktt) |
| 4 | compiled | bf16 | 32 | 256 | -- | OOM | 70,571 | -- | -- | -- | [fy53wd2r](https://wandb.ai/cataluna84/lid-bench/runs/fy53wd2r) |
| 4 | compiled | bf16 | 32 | 512 | -- | OOM | 70,571 | -- | -- | -- | [2jpgj57m](https://wandb.ai/cataluna84/lid-bench/runs/2jpgj57m) |
| 5 | sdpa | fp16 | 16 | 512 | 84.9 | 6.34x | 73,631 | 0.033 | 5,822 | ~39s | [c0elap7s](https://wandb.ai/cataluna84/lid-bench/runs/c0elap7s) |
| 5 | sdpa | fp16 | 32 | 512 | 94.0 | 7.01x | 74,808 | 0.033 | 5,640 | ~35s | [byfv8vky](https://wandb.ai/cataluna84/lid-bench/runs/byfv8vky) |
| 5 | flash_attn | fp16 | 16 | 512 | 80.7 | 6.02x | 73,623 | 0.033 | 6,011 | ~41s | [3guqdbeq](https://wandb.ai/cataluna84/lid-bench/runs/3guqdbeq) |
| 5 | flash_attn | fp16 | 32 | 512 | 92.0 | 6.87x | 74,791 | 0.033 | 5,716 | ~36s | [ukx554xy](https://wandb.ai/cataluna84/lid-bench/runs/ukx554xy) |
| 6 | quantized_int8 | fp16 | 16 | 512 | 41.9 | 3.13x | 71,201 | 0.033 | 9,882 | ~79s | [mvwhkhv2](https://wandb.ai/cataluna84/lid-bench/runs/mvwhkhv2) |
| 6 | quantized_int4 | fp16 | 16 | 512 | 64.5 | 4.81x | 69,622 | 0.030 | 7,589 | ~51s | [j18ryq9w](https://wandb.ai/cataluna84/lid-bench/runs/j18ryq9w) |
| 6 | quantized_int8 | fp16 | 32 | 512 | 48.2 | 3.60x | 72,755 | 0.033 | 9,769 | ~69s | [0hvmtb82](https://wandb.ai/cataluna84/lid-bench/runs/0hvmtb82) |
| 6 | quantized_int4 | fp16 | 32 | 512 | 73.3 | 5.47x | 70,832 | 0.030 | 7,214 | ~45s | [6iubw4a2](https://wandb.ai/cataluna84/lid-bench/runs/6iubw4a2) |
| 6 | quantized_int8 | fp16 | 64 | 512 | 52.1 | 3.89x | 76,490 | 0.032 | 9,740 | ~64s | [k0vdph36](https://wandb.ai/cataluna84/lid-bench/runs/k0vdph36) |
| 6 | quantized_int4 | fp16 | 64 | 512 | 79.1 | 5.90x | 73,879 | 0.030 | 6,967 | ~42s | [mtdlenb8](https://wandb.ai/cataluna84/lid-bench/runs/mtdlenb8) |
| 7 | combined_flash_compiled | bf16 | 16 | 256 | 126.3 | 9.43x | 72,412 | 0.012 | 3,082 | ~31s | [rxwvl8vz](https://wandb.ai/cataluna84/lid-bench/runs/rxwvl8vz) |
| 7 | combined_flash_compiled | bf16 | 16 | 512 | 80.9 | 6.04x | 73,623 | 0.033 | 5,897 | ~46s | [78cye8cs](https://wandb.ai/cataluna84/lid-bench/runs/78cye8cs) |
| 7 | combined_flash_compiled | bf16 | 32 | 256 | 155.2 | 11.58x | 72,371 | 0.012 | 2,875 | ~26s | [p9uy1git](https://wandb.ai/cataluna84/lid-bench/runs/p9uy1git) |
| 7 | combined_flash_compiled | bf16 | 32 | 512 | 92.5 | 6.90x | 74,791 | 0.033 | 5,624 | ~41s | [05rd0rqz](https://wandb.ai/cataluna84/lid-bench/runs/05rd0rqz) |
| 7 | combined_flash_compiled | fp16 | 16 | 256 | 125.5 | 9.37x | 72,412 | 0.012 | 3,108 | ~32s | [svh5uskj](https://wandb.ai/cataluna84/lid-bench/runs/svh5uskj) |
| 7 | combined_flash_compiled | fp16 | 16 | 512 | 80.6 | 6.01x | 73,623 | 0.033 | 6,003 | ~47s | [ycvi47ew](https://wandb.ai/cataluna84/lid-bench/runs/ycvi47ew) |
| 7 | combined_flash_compiled | fp16 | 32 | 256 | 154.1 | 11.50x | 72,371 | 0.012 | 2,921 | ~27s | [j5ptnotk](https://wandb.ai/cataluna84/lid-bench/runs/j5ptnotk) |
| 7 | combined_flash_compiled | fp16 | 32 | 512 | 91.8 | 6.85x | 74,791 | 0.033 | 5,727 | ~41s | [tz8vbr1t](https://wandb.ai/cataluna84/lid-bench/runs/tz8vbr1t) |
| 7 | combined_flash_int8 | fp16 | 16 | 256 | 64.0 | 4.78x | 69,793 | 0.012 | 5,037 | ~62s | [zuw7bb98](https://wandb.ai/cataluna84/lid-bench/runs/zuw7bb98) |
| 7 | combined_flash_int8 | fp16 | 16 | 512 | 41.1 | 3.07x | 71,192 | 0.035 | 9,954 | ~90s | [nfc5r4i2](https://wandb.ai/cataluna84/lid-bench/runs/nfc5r4i2) |
| 7 | combined_flash_int8 | fp16 | 32 | 256 | 81.9 | 6.11x | 69,940 | 0.012 | 4,764 | ~50s | [x7lk7777](https://wandb.ai/cataluna84/lid-bench/runs/x7lk7777) |
| 7 | combined_flash_int8 | fp16 | 32 | 512 | 47.7 | 3.56x | 72,739 | 0.034 | 9,791 | ~78s | [e154jzzt](https://wandb.ai/cataluna84/lid-bench/runs/e154jzzt) |

---

### Step 1 -- Baseline: `lid-infer` (original notebook code path)

- **Command:** `uv run lid-infer --model CohereLabs/tiny-aya-global --sample-frac 0.1 --batch-size 16 --temperature 0.2 --max-length 512 --output-dir experiments/step1_baseline --seed 1024`
- **Wall time:** 250.1 s (~4.2 min) -- inference 250 s, postproc 2.4 s
- **Throughput:** 13.4 sps (0.84 batches/sec)
- **Latency:** ~74.7 ms/sample
- **GPU memory peak:** 17,281 MB
- **GPU memory (model only):** 6,812 MB
- **Avg GPU power:** 250.5 W
- **Energy per sample:** 18,726 mJ
- **CUDA kernel time:** 250,405 ms
- **Last-layer accuracy:** 0.0272 (2.72%)
- **Layer count:** 37

**Layer accuracy range:** 1.4% (layer 5/8) → 3.2% (layer 33)

**Local files:**
```
experiments/step1_baseline/
  results.pkl           (full DataFrame)
  layer_accuracy.csv    (37 rows)
  layer_avg_probs.csv   (37 rows)
  benchmark_row.csv     (1 row, bench-compatible)
```

**W&B run:** https://wandb.ai/cataluna84/lid-bench/runs/nk88cizk

---

### Step 2 -- Eager via Bench Framework

- **Command:** `uv run lid-bench configs/step2_eager.yaml`
- **Wall time:** 238.7 s (~4.0 min)
- **Throughput:** 14.0 sps (0.88 batches/sec)
- **Latency:** ~71.4 ms/sample
- **GPU memory peak:** 17,281 MB
- **GPU memory (model only):** 6,812 MB
- **Avg GPU power:** ~253 W
- **Energy per sample:** 18,014 mJ
- **CUDA kernel time:** 238,731 ms
- **Last-layer accuracy:** 0.0330 (3.30%)

**Comparison to Step 1:**
- Throughput: 14.0 vs 13.4 → **1.04x** (effectively identical, as expected)
- Memory: identical (17,281 MB)
- Accuracy: 3.30% vs 2.72% -- small difference from warmup batch exclusion in bench framework

**W&B run:** https://wandb.ai/cataluna84/lid-bench/runs/3am3znst

**Local files:**
```
experiments/benchmark_results.csv  (1 row so far, appended per step)
```

### Step 3 -- Vectorized (layer-wise extraction)

- **Command:** `uv run lid-bench configs/step3_vectorized.yaml`
- **Grid:** strategy=vectorized, dtype=fp16, bs=[8,16,32], ml=[256,512], repeat=3 → 18 runs
- **Completed:** 15/18 (bs=32/ml=512 OOM'd at batch 93/105 -- genuine VRAM capacity limit)

**Best config: bs=32, ml=256**
- **Throughput:** 133.3 sps (mean of 3 repeats: 133.1, 133.6, 133.2)
- **Speedup vs baseline:** **9.95x**
- **GPU memory peak:** 80,386 MB
- **Energy per sample:** 3,336 mJ
- **Last-layer accuracy:** 0.012 (1.2%)
- **Wall time:** ~25s per repeat

**Full results by config (mean of 3 repeats):**

| bs | ml | sps (mean) | mem_mb | acc | wall_sec |
|----|-----|------------|--------|-------|----------|
| 8  | 256 | 93.4       | 74,281 | 0.012 | ~36s     |
| 8  | 512 | 62.8       | 76,889 | 0.033 | ~53s     |
| 16 | 256 | 112.7      | 76,420 | 0.012 | ~29s     |
| 16 | 512 | 71.5       | 81,638 | 0.033 | ~47s     |
| 32 | 256 | 133.3      | 80,386 | 0.012 | ~25s     |
| 32 | 512 | OOM        | 83,936 | --    | --       |

**Observations:**
- Vectorized extraction yields massive speedups (5-10x) over eager by computing all 37 layers' log-probs in a single batched `lm_head` pass.
- Larger batch sizes increase throughput but also memory; bs=32/ml=512 exceeds 79 GiB VRAM (needs ~78.75 GiB at peak).
- Accuracy at ml=256 (1.2%) is lower than ml=512 (3.3%), likely due to truncation cutting off answer tokens. The ml=512 accuracy matches Step 2 eager (3.3%).
- Memory climbs monotonically because `all_probs` list accumulates CPU tensors; this is expected and not a leak.

**OOM detail (bs=32, ml=512):**
All 3 repeats failed at batch 93/105 (~89%). PyTorch had 70.75 GiB allocated with only 7.5 GiB free, attempted 8.0 GiB allocation for next forward pass. Reserved-but-unallocated was only 157-237 MiB (no fragmentation -- pure capacity overflow).

**W&B runs:**
- bs=8/ml=256: [et2ce9j9](https://wandb.ai/cataluna84/lid-bench/runs/et2ce9j9), [4qqkntz8](https://wandb.ai/cataluna84/lid-bench/runs/4qqkntz8), [vlmjj0uf](https://wandb.ai/cataluna84/lid-bench/runs/vlmjj0uf)
- bs=8/ml=512: [0gm67u4n](https://wandb.ai/cataluna84/lid-bench/runs/0gm67u4n), [38e51s4q](https://wandb.ai/cataluna84/lid-bench/runs/38e51s4q), [0c5rmjwl](https://wandb.ai/cataluna84/lid-bench/runs/0c5rmjwl)
- bs=16/ml=256: [i2jgf978](https://wandb.ai/cataluna84/lid-bench/runs/i2jgf978), [31yxlvxa](https://wandb.ai/cataluna84/lid-bench/runs/31yxlvxa), [dntpeo55](https://wandb.ai/cataluna84/lid-bench/runs/dntpeo55)
- bs=16/ml=512: [mjearq5c](https://wandb.ai/cataluna84/lid-bench/runs/mjearq5c), [ez6u1run](https://wandb.ai/cataluna84/lid-bench/runs/ez6u1run), [4t40iyg7](https://wandb.ai/cataluna84/lid-bench/runs/4t40iyg7)
- bs=32/ml=256: [vyn2bs6v](https://wandb.ai/cataluna84/lid-bench/runs/vyn2bs6v), [1proqxn1](https://wandb.ai/cataluna84/lid-bench/runs/1proqxn1), [ghvw7nwi](https://wandb.ai/cataluna84/lid-bench/runs/ghvw7nwi)
- bs=32/ml=512 (OOM): [7matvroe](https://wandb.ai/cataluna84/lid-bench/runs/7matvroe), [8ykurfne](https://wandb.ai/cataluna84/lid-bench/runs/8ykurfne), [4tni9e3d](https://wandb.ai/cataluna84/lid-bench/runs/4tni9e3d)

### Step 4 -- Compiled (torch.compile, mode="reduce-overhead")

- **Command:** `uv run lid-bench configs/step4_compiled.yaml`
- **Grid:** strategy=compiled, dtype=[fp16,bf16], bs=[16,32], ml=[256,512], repeat=3 → 24 runs
- **Completed:** 15/24 (all bs=32 configs OOM'd due to CUDA Graph private pool overhead)

**Important note on r0 warmup:** The first repeat (r0) of each config includes `torch.compile` tracing/compilation overhead (1-2 min), making it 3-5x slower than subsequent repeats. Metrics below use mean of r1+r2 only (steady-state performance).

**Best config: fp16, bs=32, ml=256**
- **Throughput:** 186.1 sps (mean of r1+r2: 187.3, 184.9)
- **Speedup vs baseline:** **13.89x**
- **GPU memory peak:** 74,798 MB
- **Energy per sample:** 2,600 mJ
- **Last-layer accuracy:** 0.012 (1.2%)
- **Wall time:** ~18s per repeat (steady-state)

**Full results by config (mean of r1+r2, excluding compilation warmup r0):**

| dtype | bs | ml | sps (mean) | mem_mb | acc | wall_sec | r0 sps (w/ compile) |
|-------|----|----|------------|--------|-------|----------|---------------------|
| fp16  | 16 | 256 | 159.2     | 72,667 | 0.012 | ~21s     | 39.8                |
| fp16  | 16 | 512 | 105.3     | 73,818 | 0.033 | ~31s     | 28.2                |
| bf16  | 16 | 256 | 159.4     | 72,667 | 0.012 | ~21s     | 30.8                |
| bf16  | 16 | 512 | 106.1     | 73,818 | 0.033 | ~31s     | 95.8                |
| fp16  | 32 | 256 | 186.1     | 74,798 | 0.012 | ~18s     | 51.1                |
| fp16  | 32 | 512 | OOM       | --     | --    | --       | --                  |
| bf16  | 32 | 256 | OOM       | --     | --    | --       | --                  |
| bf16  | 32 | 512 | OOM       | --     | --    | --       | --                  |

**Observations:**
- Compiled strategy provides **1.4x speedup over vectorized** at equivalent configs (fp16/bs=16/ml=256: 159.2 vs 112.7 sps).
- fp16 and bf16 perform nearly identically (159.2 vs 159.4 sps at bs=16/ml=256).
- `torch.compile(mode="reduce-overhead")` uses CUDA Graphs, which allocate **12.27 GiB in private memory pools** that cannot be freed. This causes all bs=32 configs to OOM at batch 103/105 (98%), even bs=32/ml=256 which succeeded in Step 3 vectorized.
- Compilation warmup (r0) takes 60-120s; subsequent repeats run at full speed.
- Accuracy is consistent with vectorized: 1.2% at ml=256, 3.3% at ml=512.

**OOM detail (bs=32, all dtypes/ml):**
All 9 bs=32 runs failed at batch 103/105. PyTorch had 78.0 GiB allocated (including 12.27 GiB in CUDA Graph private pools), attempted 592 MiB allocation with only ~380 MiB free. This is a fundamental trade-off of `mode="reduce-overhead"`: it trades memory for reduced kernel launch overhead.

**W&B runs:**
- fp16/bs=16/ml=256: [kb2pxxpk](https://wandb.ai/cataluna84/lid-bench/runs/kb2pxxpk), [sqvdx5v7](https://wandb.ai/cataluna84/lid-bench/runs/sqvdx5v7), [vvejf926](https://wandb.ai/cataluna84/lid-bench/runs/vvejf926)
- fp16/bs=16/ml=512: [nt2js6ix](https://wandb.ai/cataluna84/lid-bench/runs/nt2js6ix), [mzz4ulrm](https://wandb.ai/cataluna84/lid-bench/runs/mzz4ulrm), [rnaljs1i](https://wandb.ai/cataluna84/lid-bench/runs/rnaljs1i)
- bf16/bs=16/ml=256: [6a5wm1i1](https://wandb.ai/cataluna84/lid-bench/runs/6a5wm1i1), [bxv8gpcr](https://wandb.ai/cataluna84/lid-bench/runs/bxv8gpcr), [dk074o5p](https://wandb.ai/cataluna84/lid-bench/runs/dk074o5p)
- bf16/bs=16/ml=512: [sct4qllu](https://wandb.ai/cataluna84/lid-bench/runs/sct4qllu), [pp09z5y0](https://wandb.ai/cataluna84/lid-bench/runs/pp09z5y0), [sm73ugtr](https://wandb.ai/cataluna84/lid-bench/runs/sm73ugtr)
- fp16/bs=32/ml=256: [xgh6cfq9](https://wandb.ai/cataluna84/lid-bench/runs/xgh6cfq9), [iuu365zu](https://wandb.ai/cataluna84/lid-bench/runs/iuu365zu), [nurve1bx](https://wandb.ai/cataluna84/lid-bench/runs/nurve1bx)
- fp16/bs=32/ml=512 (OOM): [7lrlhktt](https://wandb.ai/cataluna84/lid-bench/runs/7lrlhktt), [4azstn0l](https://wandb.ai/cataluna84/lid-bench/runs/4azstn0l), [if439295](https://wandb.ai/cataluna84/lid-bench/runs/if439295)
- bf16/bs=32/ml=256 (OOM): [fy53wd2r](https://wandb.ai/cataluna84/lid-bench/runs/fy53wd2r), [izfgbygd](https://wandb.ai/cataluna84/lid-bench/runs/izfgbygd), [orjmfdi0](https://wandb.ai/cataluna84/lid-bench/runs/orjmfdi0)
- bf16/bs=32/ml=512 (OOM): [2jpgj57m](https://wandb.ai/cataluna84/lid-bench/runs/2jpgj57m), [gx9vnapq](https://wandb.ai/cataluna84/lid-bench/runs/gx9vnapq), [hfu4atpk](https://wandb.ai/cataluna84/lid-bench/runs/hfu4atpk)

### Step 5 -- Attention Backends (SDPA / Flash Attn)

- **Command:** `uv run lid-bench configs/step5_attention.yaml`
- **Grid:** strategy=[sdpa, flash_attn], dtype=fp16, bs=[16,32], ml=512, repeat=3 → 12 runs
- **Completed:** 12/12 (all succeeded after `model.model()` fix eliminated redundant 8 GiB logits tensor)
- **flash-attn version:** 2.8.3 (pre-built wheel, CUDA 12.8)

**Best config: sdpa, fp16, bs=32, ml=512**
- **Throughput:** 94.0 sps (mean of 3 repeats: 90.4, 95.6, 96.0)
- **Speedup vs baseline:** **7.01x**
- **GPU memory peak:** 74,808 MB
- **Energy per sample:** 5,640 mJ
- **Last-layer accuracy:** 0.033 (3.3%)
- **Wall time:** ~35s per repeat

**Full results by config (mean of 3 repeats):**

| strategy | bs | ml | sps (mean) | mem_mb | acc | wall_sec | energy_mj |
|----------|----|----|------------|--------|-------|----------|-----------|
| sdpa       | 16 | 512 | 84.9     | 73,631 | 0.033 | ~39s     | 5,822     |
| sdpa       | 32 | 512 | 94.0     | 74,808 | 0.033 | ~35s     | 5,640     |
| flash_attn | 16 | 512 | 80.7     | 73,623 | 0.033 | ~41s     | 6,011     |
| flash_attn | 32 | 512 | 92.0     | 74,791 | 0.033 | ~36s     | 5,716     |

**Individual run detail:**

| strategy | bs | repeat | sps | mem_mb |
|----------|-----|--------|------|--------|
| sdpa       | 16 | r0 | 82.3 | 73,631 |
| sdpa       | 16 | r1 | 86.6 | 73,631 |
| sdpa       | 16 | r2 | 85.8 | 73,631 |
| flash_attn | 16 | r0 | 80.9 | 73,623 |
| flash_attn | 16 | r1 | 80.2 | 73,623 |
| flash_attn | 16 | r2 | 80.9 | 73,623 |
| sdpa       | 32 | r0 | 90.4 | 74,808 |
| sdpa       | 32 | r1 | 95.6 | 74,808 |
| sdpa       | 32 | r2 | 96.0 | 74,808 |
| flash_attn | 32 | r0 | 92.1 | 74,791 |
| flash_attn | 32 | r1 | 91.6 | 74,791 |
| flash_attn | 32 | r2 | 92.2 | 74,791 |

**Observations:**
- All 12 runs completed successfully. The `model.model()` fix (skipping redundant lm_head logits computation) saved ~8 GiB VRAM, enabling bs=32/ml=512 which previously OOM'd.
- **SDPA slightly outperforms flash_attn** at both batch sizes: 84.9 vs 80.7 sps (bs=16), 94.0 vs 92.0 sps (bs=32). This is expected since SDPA on H100 uses the same fused kernel path internally.
- **bs=32 outperforms bs=16** for both backends (~11% faster for sdpa, ~14% for flash_attn), with only ~1.2 GiB additional memory.
- Memory usage is nearly identical between sdpa and flash_attn (73,631 vs 73,623 MB at bs=16; 74,808 vs 74,791 at bs=16).
- Accuracy is consistent at 3.3% across all runs, matching Steps 2-3.
- Compared to Step 3 vectorized (bs=32/ml=512 OOM'd), Step 5 now completes at bs=32/ml=512 thanks to the lm_head fix.
- Compared to Step 3 vectorized at bs=16/ml=512 (71.5 sps), sdpa achieves 84.9 sps (**1.19x improvement**) and flash_attn achieves 80.7 sps (**1.13x**).

**W&B runs:**
- sdpa/bs=16: [c0elap7s](https://wandb.ai/cataluna84/lid-bench/runs/c0elap7s), [jwj17k45](https://wandb.ai/cataluna84/lid-bench/runs/jwj17k45), [krb8pufi](https://wandb.ai/cataluna84/lid-bench/runs/krb8pufi)
- flash_attn/bs=16: [3guqdbeq](https://wandb.ai/cataluna84/lid-bench/runs/3guqdbeq), [t0hsrmr1](https://wandb.ai/cataluna84/lid-bench/runs/t0hsrmr1), [8e3sbei7](https://wandb.ai/cataluna84/lid-bench/runs/8e3sbei7)
- sdpa/bs=32: [byfv8vky](https://wandb.ai/cataluna84/lid-bench/runs/byfv8vky), [v67j8xbi](https://wandb.ai/cataluna84/lid-bench/runs/v67j8xbi), [b8j7ctgw](https://wandb.ai/cataluna84/lid-bench/runs/b8j7ctgw)
- flash_attn/bs=32: [ukx554xy](https://wandb.ai/cataluna84/lid-bench/runs/ukx554xy), [a5tmn5la](https://wandb.ai/cataluna84/lid-bench/runs/a5tmn5la), [0zdylgj1](https://wandb.ai/cataluna84/lid-bench/runs/0zdylgj1)

### Step 6 -- Quantization (INT8 / INT4 via bitsandbytes)

- **Command:** `uv run lid-bench configs/step6_quantized.yaml`
- **Grid:** strategy=[quantized_int8, quantized_int4], dtype=fp16, bs=[16,32,64], ml=512, repeat=3 → 18 runs
- **Completed:** 18/18 (all succeeded)
- **bitsandbytes version:** 0.45.5

**Best config: quantized_int4, fp16, bs=64, ml=512**
- **Throughput:** 79.1 sps (mean of 3 repeats: 79.1, 78.8, 79.3)
- **Speedup vs baseline:** **5.90x**
- **GPU memory peak:** 73,879 MB
- **Model memory:** 2,729 MB (vs 6,812 MB full precision -- **2.50x compression**)
- **Energy per sample:** 6,967 mJ
- **Last-layer accuracy:** 0.030 (3.0%)
- **Wall time:** ~42s per repeat

**Full results by config (mean of 3 repeats):**

| strategy | bs | ml | sps (mean) | mem_mb | model_mb | acc | wall_sec | energy_mj |
|----------|----|----|------------|--------|----------|-------|----------|-----------|
| quantized_int8 | 16 | 512 | 41.9 | 71,201 | 4,004 | 0.033 | ~79s | 9,882 |
| quantized_int4 | 16 | 512 | 64.5 | 69,622 | 2,729 | 0.030 | ~51s | 7,589 |
| quantized_int8 | 32 | 512 | 48.2 | 72,755 | 4,004 | 0.033 | ~69s | 9,769 |
| quantized_int4 | 32 | 512 | 73.3 | 70,832 | 2,729 | 0.030 | ~45s | 7,214 |
| quantized_int8 | 64 | 512 | 52.1 | 76,490 | 4,004 | 0.032 | ~64s | 9,740 |
| quantized_int4 | 64 | 512 | 79.1 | 73,879 | 2,729 | 0.030 | ~42s | 6,967 |

**Individual run detail:**

| strategy | bs | repeat | sps | mem_mb |
|----------|-----|--------|------|--------|
| quantized_int8 | 16 | r0 | 41.4 | 71,201 |
| quantized_int8 | 16 | r1 | 42.2 | 71,201 |
| quantized_int8 | 16 | r2 | 42.2 | 71,201 |
| quantized_int4 | 16 | r0 | 64.3 | 69,622 |
| quantized_int4 | 16 | r1 | 64.5 | 69,622 |
| quantized_int4 | 16 | r2 | 64.8 | 69,622 |
| quantized_int8 | 32 | r0 | 47.5 | 72,755 |
| quantized_int8 | 32 | r1 | 48.6 | 72,755 |
| quantized_int8 | 32 | r2 | 48.6 | 72,755 |
| quantized_int4 | 32 | r0 | 73.2 | 70,832 |
| quantized_int4 | 32 | r1 | 73.4 | 70,832 |
| quantized_int4 | 32 | r2 | 73.2 | 70,832 |
| quantized_int8 | 64 | r0 | 51.5 | 76,490 |
| quantized_int8 | 64 | r1 | 52.6 | 76,490 |
| quantized_int8 | 64 | r2 | 52.3 | 76,490 |
| quantized_int4 | 64 | r0 | 79.1 | 73,879 |
| quantized_int4 | 64 | r1 | 78.8 | 73,879 |
| quantized_int4 | 64 | r2 | 79.3 | 73,879 |

**Observations:**
- All 18 runs completed successfully, including bs=64 configs -- quantization reduces model memory enough to fit larger batches.
- **INT4 is consistently ~1.5x faster than INT8** across all batch sizes (64.5 vs 41.9 at bs=16, 73.3 vs 48.2 at bs=32, 79.1 vs 52.1 at bs=64).
- **Model memory:** INT8 uses 4,004 MB (1.70x compression), INT4 uses 2,729 MB (2.50x compression) vs 6,812 MB full precision.
- **Peak VRAM** is dominated by activations, not model weights -- INT4 saves ~1.3 GiB model memory but total peak only drops ~1.5-2.6 GiB vs INT8.
- **Throughput scaling with batch size** is modest: INT8 goes from 41.9→48.2→52.1 sps (1.24x from bs=16→64), INT4 from 64.5→73.3→79.1 sps (1.23x). This suggests the bottleneck is compute (dequantization overhead), not memory bandwidth.
- **Accuracy:** INT8 maintains 3.3% (matching full-precision Steps 2-5), INT4 drops slightly to 3.0% -- minor degradation from aggressive quantization.
- **Energy efficiency:** INT4 is significantly more energy-efficient (~7,000 mJ/sample vs ~9,800 mJ/sample for INT8), driven by faster completion and lower model footprint.
- **Compared to vectorized (Step 3):** Quantized strategies are slower at equivalent batch sizes (INT4 bs=32: 73.3 vs vectorized bs=32/ml=512: OOM, but vs bs=16/ml=512: 71.5 sps -- comparable). The quantized approach enables bs=64 which vectorized cannot fit.
- **Compared to attention backends (Step 5):** sdpa bs=32: 94.0 sps vs INT4 bs=32: 73.3 sps -- attention optimization is more effective than quantization alone for throughput.
- Note: r0 for INT8 runs includes bitsandbytes quantization overhead during model loading (~5s for INT8, ~1s for INT4).

**W&B runs:**
- int8/bs=16: [mvwhkhv2](https://wandb.ai/cataluna84/lid-bench/runs/mvwhkhv2), [09jc5kls](https://wandb.ai/cataluna84/lid-bench/runs/09jc5kls), [3vhkj2uh](https://wandb.ai/cataluna84/lid-bench/runs/3vhkj2uh)
- int4/bs=16: [j18ryq9w](https://wandb.ai/cataluna84/lid-bench/runs/j18ryq9w), [koolo7ea](https://wandb.ai/cataluna84/lid-bench/runs/koolo7ea), [tu0e9eu2](https://wandb.ai/cataluna84/lid-bench/runs/tu0e9eu2)
- int8/bs=32: [0hvmtb82](https://wandb.ai/cataluna84/lid-bench/runs/0hvmtb82), [edpgsvyo](https://wandb.ai/cataluna84/lid-bench/runs/edpgsvyo), [bvcm8ys9](https://wandb.ai/cataluna84/lid-bench/runs/bvcm8ys9)
- int4/bs=32: [6iubw4a2](https://wandb.ai/cataluna84/lid-bench/runs/6iubw4a2), [1aa67bha](https://wandb.ai/cataluna84/lid-bench/runs/1aa67bha), [ztij547m](https://wandb.ai/cataluna84/lid-bench/runs/ztij547m)
- int8/bs=64: [k0vdph36](https://wandb.ai/cataluna84/lid-bench/runs/k0vdph36), [vlxnn77d](https://wandb.ai/cataluna84/lid-bench/runs/vlxnn77d), [zy8ggn50](https://wandb.ai/cataluna84/lid-bench/runs/zy8ggn50)
- int4/bs=64: [mtdlenb8](https://wandb.ai/cataluna84/lid-bench/runs/mtdlenb8), [1vzcqsn8](https://wandb.ai/cataluna84/lid-bench/runs/1vzcqsn8), [0n1pzuny](https://wandb.ai/cataluna84/lid-bench/runs/0n1pzuny)

### Step 7 -- Combined Strategies: flash_attn + compile / flash_attn + INT8

- **Config:** `configs/step7_combined.yaml`
- **Grid:** `combined_flash_compiled` × {fp16, bf16} × {bs=16, bs=32} × {ml=256, ml=512} × 3 repeats = 24 runs; `combined_flash_int8` × fp16 × {bs=16, bs=32} × {ml=256, ml=512} × 3 repeats = 12 runs; **36 total**
- **Completed:** 36/36 (no OOMs)
- **Best config:** `combined_flash_compiled` / bf16 / bs=32 / ml=256 → **155.2 sps** (11.58x speedup)
- **Local report:** `experiments/step7-combined/20260416_111558/REPORT.md`

**Summary (mean of 3 repeats):**

| strategy | dtype | bs | ml | sps (mean) | sps (std) | mem_mb | acc | wall_sec | energy_mj |
|----------|-------|----|----|------------|-----------|--------|------|----------|-----------|
| combined_flash_compiled | bf16 | 16 | 256 | 126.3 | 0.1 | 72,412 | 0.012 | 31.1 | 3,082 |
| combined_flash_compiled | bf16 | 16 | 512 | 80.9 | 0.4 | 73,623 | 0.033 | 46.0 | 5,897 |
| combined_flash_compiled | bf16 | 32 | 256 | 155.2 | 0.5 | 72,371 | 0.012 | 26.2 | 2,875 |
| combined_flash_compiled | bf16 | 32 | 512 | 92.5 | 0.2 | 74,791 | 0.033 | 40.8 | 5,624 |
| combined_flash_compiled | fp16 | 16 | 256 | 125.5 | 0.6 | 72,412 | 0.012 | 31.8 | 3,108 |
| combined_flash_compiled | fp16 | 16 | 512 | 80.6 | 0.3 | 73,623 | 0.033 | 46.5 | 6,003 |
| combined_flash_compiled | fp16 | 32 | 256 | 154.1 | 0.5 | 72,371 | 0.012 | 26.7 | 2,921 |
| combined_flash_compiled | fp16 | 32 | 512 | 91.8 | 0.1 | 74,791 | 0.033 | 41.4 | 5,727 |
| combined_flash_int8 | fp16 | 16 | 256 | 64.0 | 0.2 | 69,793 | 0.012 | 61.5 | 5,037 |
| combined_flash_int8 | fp16 | 16 | 512 | 41.1 | 0.0 | 71,192 | 0.035 | 90.4 | 9,954 |
| combined_flash_int8 | fp16 | 32 | 256 | 81.9 | 0.1 | 69,940 | 0.012 | 49.8 | 4,764 |
| combined_flash_int8 | fp16 | 32 | 512 | 47.7 | 0.1 | 72,739 | 0.034 | 78.3 | 9,791 |

**Individual run detail:**

| strategy | dtype | bs | ml | repeat | sps | mem_mb |
|----------|-------|----|----|--------|------|--------|
| combined_flash_compiled | fp16 | 16 | 256 | r0 | 125.0 | 72,412 |
| combined_flash_compiled | fp16 | 16 | 256 | r1 | 126.2 | 72,412 |
| combined_flash_compiled | fp16 | 16 | 256 | r2 | 125.4 | 72,412 |
| combined_flash_compiled | fp16 | 16 | 512 | r0 | 80.3 | 73,623 |
| combined_flash_compiled | fp16 | 16 | 512 | r1 | 80.9 | 73,623 |
| combined_flash_compiled | fp16 | 16 | 512 | r2 | 80.7 | 73,623 |
| combined_flash_compiled | fp16 | 32 | 256 | r0 | 154.5 | 72,371 |
| combined_flash_compiled | fp16 | 32 | 256 | r1 | 154.2 | 72,371 |
| combined_flash_compiled | fp16 | 32 | 256 | r2 | 153.5 | 72,371 |
| combined_flash_compiled | fp16 | 32 | 512 | r0 | 91.9 | 74,791 |
| combined_flash_compiled | fp16 | 32 | 512 | r1 | 91.8 | 74,791 |
| combined_flash_compiled | fp16 | 32 | 512 | r2 | 91.7 | 74,791 |
| combined_flash_compiled | bf16 | 16 | 256 | r0 | 126.3 | 72,412 |
| combined_flash_compiled | bf16 | 16 | 256 | r1 | 126.2 | 72,412 |
| combined_flash_compiled | bf16 | 16 | 256 | r2 | 126.3 | 72,412 |
| combined_flash_compiled | bf16 | 16 | 512 | r0 | 80.9 | 73,623 |
| combined_flash_compiled | bf16 | 16 | 512 | r1 | 81.3 | 73,623 |
| combined_flash_compiled | bf16 | 16 | 512 | r2 | 80.6 | 73,623 |
| combined_flash_compiled | bf16 | 32 | 256 | r0 | 154.6 | 72,371 |
| combined_flash_compiled | bf16 | 32 | 256 | r1 | 155.6 | 72,371 |
| combined_flash_compiled | bf16 | 32 | 256 | r2 | 155.4 | 72,371 |
| combined_flash_compiled | bf16 | 32 | 512 | r0 | 92.3 | 74,791 |
| combined_flash_compiled | bf16 | 32 | 512 | r1 | 92.5 | 74,791 |
| combined_flash_compiled | bf16 | 32 | 512 | r2 | 92.8 | 74,791 |
| combined_flash_int8 | fp16 | 16 | 256 | r0 | 64.2 | 69,793 |
| combined_flash_int8 | fp16 | 16 | 256 | r1 | 63.8 | 69,793 |
| combined_flash_int8 | fp16 | 16 | 256 | r2 | 63.9 | 69,793 |
| combined_flash_int8 | fp16 | 16 | 512 | r0 | 41.1 | 71,192 |
| combined_flash_int8 | fp16 | 16 | 512 | r1 | 41.1 | 71,192 |
| combined_flash_int8 | fp16 | 16 | 512 | r2 | 41.2 | 71,192 |
| combined_flash_int8 | fp16 | 32 | 256 | r0 | 81.8 | 69,940 |
| combined_flash_int8 | fp16 | 32 | 256 | r1 | 82.0 | 69,940 |
| combined_flash_int8 | fp16 | 32 | 256 | r2 | 81.8 | 69,940 |
| combined_flash_int8 | fp16 | 32 | 512 | r0 | 47.6 | 72,739 |
| combined_flash_int8 | fp16 | 32 | 512 | r1 | 47.8 | 72,739 |
| combined_flash_int8 | fp16 | 32 | 512 | r2 | 47.8 | 72,739 |

**Observations:**
- All 36/36 runs completed successfully with no OOMs -- the `model.model()` fix from Step 5 continues to pay dividends.
- **`combined_flash_compiled` dominates `combined_flash_int8`** by ~1.9x across all configs. INT8 quantization dequantization overhead negates the memory savings when combined with flash attention.
- **bf16 vs fp16:** bf16 is marginally faster (~0.5-1.0 sps) for `combined_flash_compiled`, likely due to native H100 bf16 tensor core support. The difference is within noise for practical purposes.
- **Best throughput:** 155.2 sps (combined_flash_compiled, bf16, bs=32, ml=256) -- this matches Step 4 compiled (186.1 sps at bs=32/ml=256 was higher because ml=256 has fewer tokens). At ml=512, Step 7 combined_flash_compiled (92.5 sps) slightly underperforms Step 4 compiled (105.3 sps at bs=16/ml=512), suggesting flash_attn adds minor overhead vs SDPA for this model size.
- **Memory:** combined_flash_compiled uses 72-75 GiB (similar to Step 4/5), combined_flash_int8 uses 70-73 GiB (slightly lower from INT8 model compression).
- **Energy efficiency:** combined_flash_compiled at bs=32/ml=256 achieves the best energy per sample (2,875 mJ), matching Step 4's best.
- **Variance is extremely low:** std < 0.6 sps across all configs, confirming stable measurements.
- **Key insight:** For this model/hardware combo, `torch.compile(mode="reduce-overhead")` + vectorized batching is the dominant optimization. Adding flash_attn or INT8 quantization on top provides no additional throughput benefit -- the compiled CUDA graphs already saturate compute.

**W&B runs:**
- flash_compiled/fp16/bs=16/ml=256: [svh5uskj](https://wandb.ai/cataluna84/lid-bench/runs/svh5uskj), [1a3vuyho](https://wandb.ai/cataluna84/lid-bench/runs/1a3vuyho), [hy27prvu](https://wandb.ai/cataluna84/lid-bench/runs/hy27prvu)
- flash_compiled/fp16/bs=16/ml=512: [ycvi47ew](https://wandb.ai/cataluna84/lid-bench/runs/ycvi47ew), [6nv1m9xp](https://wandb.ai/cataluna84/lid-bench/runs/6nv1m9xp), [n6ovwvlr](https://wandb.ai/cataluna84/lid-bench/runs/n6ovwvlr)
- flash_compiled/fp16/bs=32/ml=256: [j5ptnotk](https://wandb.ai/cataluna84/lid-bench/runs/j5ptnotk), [rm2hhu0s](https://wandb.ai/cataluna84/lid-bench/runs/rm2hhu0s), [mpz65pqi](https://wandb.ai/cataluna84/lid-bench/runs/mpz65pqi)
- flash_compiled/fp16/bs=32/ml=512: [tz8vbr1t](https://wandb.ai/cataluna84/lid-bench/runs/tz8vbr1t), [kf1p8zbk](https://wandb.ai/cataluna84/lid-bench/runs/kf1p8zbk), [gjmsiwvw](https://wandb.ai/cataluna84/lid-bench/runs/gjmsiwvw)
- flash_compiled/bf16/bs=16/ml=256: [rxwvl8vz](https://wandb.ai/cataluna84/lid-bench/runs/rxwvl8vz), [orqp6k15](https://wandb.ai/cataluna84/lid-bench/runs/orqp6k15), [fwvdkfy9](https://wandb.ai/cataluna84/lid-bench/runs/fwvdkfy9)
- flash_compiled/bf16/bs=16/ml=512: [78cye8cs](https://wandb.ai/cataluna84/lid-bench/runs/78cye8cs), [arjlz98j](https://wandb.ai/cataluna84/lid-bench/runs/arjlz98j), [c60dnz7p](https://wandb.ai/cataluna84/lid-bench/runs/c60dnz7p)
- flash_compiled/bf16/bs=32/ml=256: [p9uy1git](https://wandb.ai/cataluna84/lid-bench/runs/p9uy1git), [6krs2xeb](https://wandb.ai/cataluna84/lid-bench/runs/6krs2xeb), [ua1fethp](https://wandb.ai/cataluna84/lid-bench/runs/ua1fethp)
- flash_compiled/bf16/bs=32/ml=512: [05rd0rqz](https://wandb.ai/cataluna84/lid-bench/runs/05rd0rqz), [g4pn3yj1](https://wandb.ai/cataluna84/lid-bench/runs/g4pn3yj1), [zs4itz7x](https://wandb.ai/cataluna84/lid-bench/runs/zs4itz7x)
- flash_int8/fp16/bs=16/ml=256: [zuw7bb98](https://wandb.ai/cataluna84/lid-bench/runs/zuw7bb98), [net1ouk6](https://wandb.ai/cataluna84/lid-bench/runs/net1ouk6), [l2cwb533](https://wandb.ai/cataluna84/lid-bench/runs/l2cwb533)
- flash_int8/fp16/bs=16/ml=512: [nfc5r4i2](https://wandb.ai/cataluna84/lid-bench/runs/nfc5r4i2), [zj6xgnr0](https://wandb.ai/cataluna84/lid-bench/runs/zj6xgnr0), [acvf7hdn](https://wandb.ai/cataluna84/lid-bench/runs/acvf7hdn)
- flash_int8/fp16/bs=32/ml=256: [x7lk7777](https://wandb.ai/cataluna84/lid-bench/runs/x7lk7777), [sxoq7ozu](https://wandb.ai/cataluna84/lid-bench/runs/sxoq7ozu), [bw6xs4oj](https://wandb.ai/cataluna84/lid-bench/runs/bw6xs4oj)
- flash_int8/fp16/bs=32/ml=512: [e154jzzt](https://wandb.ai/cataluna84/lid-bench/runs/e154jzzt), [jyuzj7zy](https://wandb.ai/cataluna84/lid-bench/runs/jyuzj7zy), [ph7qytf2](https://wandb.ai/cataluna84/lid-bench/runs/ph7qytf2)

---

## Notes

- This is a fresh run on H100 80GB HBM3, replacing the earlier A100-40GB
  experiment which OOM'd at Step 3 (vectorized, bs=32, ml=512).
- Accuracy is expected to be ~3% (pre-trained model, not fine-tuned).
  The key metric across steps is consistency: accuracy must remain
  identical across strategies to validate correctness.
