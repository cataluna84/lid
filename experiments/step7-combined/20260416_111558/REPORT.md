# Benchmark Report: step7-combined

> **Timestamp:** 2026-04-16 11:46:13
> **Config:** `configs/step7_combined.yaml`
> **Command:** `/home/ubuntu/Workspace/lid/.venv/bin/lid-bench configs/step7_combined.yaml`
> **W&B Project:** [lid-bench](https://wandb.ai/cataluna84/lid-bench)
> **GPU:** NVIDIA H100 80GB HBM3 (79.2 GB)
> **PyTorch:** 2.11.0+cu128
> **Model:** CohereLabs/tiny-aya-global
> **Dataset:** 1024m/LID (3,350 samples)
> **Completed:** 36/36 runs

---

## Grid Configuration

- **strategy:** ['combined_flash_compiled', 'combined_flash_int8']
- **dtype:** ['fp16', 'bf16']
- **batch_size:** [16, 32]
- **max_length:** [256, 512]
- **Repeats:** 3
- **Warmup batches:** 2
- **Seed:** 1024

---

## Summary (mean of repeats)

| strategy | dtype | bs | ml | sps (mean) | sps (std) | mem_mb | acc | energy_mj | wall_sec |
|----------|-------|----|----|------------|-----------|--------|------|-----------|----------|
| combined_flash_compiled | bf16 | 16 | 256 | 126.3 | 0.1 | 72,412 | 0.012 | 3,082 | 31.1 |
| combined_flash_compiled | bf16 | 16 | 512 | 80.9 | 0.4 | 73,623 | 0.033 | 5,897 | 46.0 |
| combined_flash_compiled | bf16 | 32 | 256 | 155.2 | 0.5 | 72,371 | 0.012 | 2,875 | 26.2 |
| combined_flash_compiled | bf16 | 32 | 512 | 92.5 | 0.2 | 74,791 | 0.033 | 5,624 | 40.8 |
| combined_flash_compiled | fp16 | 16 | 256 | 125.5 | 0.6 | 72,412 | 0.012 | 3,108 | 31.8 |
| combined_flash_compiled | fp16 | 16 | 512 | 80.6 | 0.3 | 73,623 | 0.033 | 6,003 | 46.5 |
| combined_flash_compiled | fp16 | 32 | 256 | 154.1 | 0.5 | 72,371 | 0.012 | 2,921 | 26.7 |
| combined_flash_compiled | fp16 | 32 | 512 | 91.8 | 0.1 | 74,791 | 0.033 | 5,727 | 41.4 |
| combined_flash_int8 | fp16 | 16 | 256 | 64.0 | 0.2 | 69,793 | 0.012 | 5,037 | 61.5 |
| combined_flash_int8 | fp16 | 16 | 512 | 41.1 | 0.0 | 71,192 | 0.035 | 9,954 | 90.4 |
| combined_flash_int8 | fp16 | 32 | 256 | 81.9 | 0.1 | 69,940 | 0.012 | 4,764 | 49.8 |
| combined_flash_int8 | fp16 | 32 | 512 | 47.7 | 0.1 | 72,739 | 0.034 | 9,791 | 78.3 |

**Best config:** combined_flash_compiled / bf16 / bs=32 / ml=256 -> 155.2 sps

---

## Individual Runs

| # | run_name | sps | mem_mb | acc | wall_sec | energy_mj | W&B |
|---|----------|-----|--------|------|----------|-----------|-----|
| 1 | combined_flash_compiled_fp16_bs16_ml256_r0 | 125.0 | 72,412 | 0.012 | 32.4 | 3,075 | [link](https://wandb.ai/cataluna84/lid-bench/runs/svh5uskj) |
| 2 | combined_flash_compiled_fp16_bs16_ml256_r1 | 126.2 | 72,412 | 0.012 | 31.5 | 3,112 | [link](https://wandb.ai/cataluna84/lid-bench/runs/1a3vuyho) |
| 3 | combined_flash_compiled_fp16_bs16_ml256_r2 | 125.4 | 72,412 | 0.012 | 31.6 | 3,137 | [link](https://wandb.ai/cataluna84/lid-bench/runs/hy27prvu) |
| 4 | combined_flash_int8_fp16_bs16_ml256_r0 | 64.2 | 69,793 | 0.012 | 61.7 | 5,024 | [link](https://wandb.ai/cataluna84/lid-bench/runs/zuw7bb98) |
| 5 | combined_flash_int8_fp16_bs16_ml256_r1 | 63.8 | 69,793 | 0.012 | 61.7 | 5,053 | [link](https://wandb.ai/cataluna84/lid-bench/runs/net1ouk6) |
| 6 | combined_flash_int8_fp16_bs16_ml256_r2 | 63.9 | 69,793 | 0.012 | 61.2 | 5,033 | [link](https://wandb.ai/cataluna84/lid-bench/runs/l2cwb533) |
| 7 | combined_flash_compiled_fp16_bs16_ml512_r0 | 80.3 | 73,623 | 0.033 | 46.6 | 5,999 | [link](https://wandb.ai/cataluna84/lid-bench/runs/ycvi47ew) |
| 8 | combined_flash_compiled_fp16_bs16_ml512_r1 | 80.9 | 73,623 | 0.033 | 46.4 | 6,004 | [link](https://wandb.ai/cataluna84/lid-bench/runs/6nv1m9xp) |
| 9 | combined_flash_compiled_fp16_bs16_ml512_r2 | 80.7 | 73,623 | 0.033 | 46.6 | 6,005 | [link](https://wandb.ai/cataluna84/lid-bench/runs/n6ovwvlr) |
| 10 | combined_flash_int8_fp16_bs16_ml512_r0 | 41.1 | 71,192 | 0.035 | 90.5 | 9,963 | [link](https://wandb.ai/cataluna84/lid-bench/runs/nfc5r4i2) |
| 11 | combined_flash_int8_fp16_bs16_ml512_r1 | 41.1 | 71,192 | 0.035 | 90.4 | 9,948 | [link](https://wandb.ai/cataluna84/lid-bench/runs/zj6xgnr0) |
| 12 | combined_flash_int8_fp16_bs16_ml512_r2 | 41.2 | 71,192 | 0.035 | 90.4 | 9,953 | [link](https://wandb.ai/cataluna84/lid-bench/runs/acvf7hdn) |
| 13 | combined_flash_compiled_bf16_bs16_ml256_r0 | 126.3 | 72,412 | 0.012 | 31.0 | 3,087 | [link](https://wandb.ai/cataluna84/lid-bench/runs/rxwvl8vz) |
| 14 | combined_flash_compiled_bf16_bs16_ml256_r1 | 126.2 | 72,412 | 0.012 | 31.2 | 3,077 | [link](https://wandb.ai/cataluna84/lid-bench/runs/orqp6k15) |
| 15 | combined_flash_compiled_bf16_bs16_ml256_r2 | 126.3 | 72,412 | 0.012 | 31.2 | 3,082 | [link](https://wandb.ai/cataluna84/lid-bench/runs/fwvdkfy9) |
| 16 | combined_flash_compiled_bf16_bs16_ml512_r0 | 80.9 | 73,623 | 0.033 | 45.9 | 5,899 | [link](https://wandb.ai/cataluna84/lid-bench/runs/78cye8cs) |
| 17 | combined_flash_compiled_bf16_bs16_ml512_r1 | 81.3 | 73,623 | 0.033 | 45.8 | 5,890 | [link](https://wandb.ai/cataluna84/lid-bench/runs/arjlz98j) |
| 18 | combined_flash_compiled_bf16_bs16_ml512_r2 | 80.6 | 73,623 | 0.033 | 46.2 | 5,901 | [link](https://wandb.ai/cataluna84/lid-bench/runs/c60dnz7p) |
| 19 | combined_flash_compiled_fp16_bs32_ml256_r0 | 154.5 | 72,371 | 0.012 | 26.7 | 2,918 | [link](https://wandb.ai/cataluna84/lid-bench/runs/j5ptnotk) |
| 20 | combined_flash_compiled_fp16_bs32_ml256_r1 | 154.2 | 72,371 | 0.012 | 26.6 | 2,922 | [link](https://wandb.ai/cataluna84/lid-bench/runs/rm2hhu0s) |
| 21 | combined_flash_compiled_fp16_bs32_ml256_r2 | 153.5 | 72,371 | 0.012 | 26.8 | 2,923 | [link](https://wandb.ai/cataluna84/lid-bench/runs/mpz65pqi) |
| 22 | combined_flash_int8_fp16_bs32_ml256_r0 | 81.8 | 69,940 | 0.012 | 49.8 | 4,761 | [link](https://wandb.ai/cataluna84/lid-bench/runs/x7lk7777) |
| 23 | combined_flash_int8_fp16_bs32_ml256_r1 | 82.0 | 69,940 | 0.012 | 49.9 | 4,765 | [link](https://wandb.ai/cataluna84/lid-bench/runs/sxoq7ozu) |
| 24 | combined_flash_int8_fp16_bs32_ml256_r2 | 81.8 | 69,940 | 0.012 | 49.8 | 4,767 | [link](https://wandb.ai/cataluna84/lid-bench/runs/bw6xs4oj) |
| 25 | combined_flash_compiled_fp16_bs32_ml512_r0 | 91.9 | 74,791 | 0.033 | 41.4 | 5,725 | [link](https://wandb.ai/cataluna84/lid-bench/runs/tz8vbr1t) |
| 26 | combined_flash_compiled_fp16_bs32_ml512_r1 | 91.8 | 74,791 | 0.033 | 41.4 | 5,721 | [link](https://wandb.ai/cataluna84/lid-bench/runs/kf1p8zbk) |
| 27 | combined_flash_compiled_fp16_bs32_ml512_r2 | 91.7 | 74,791 | 0.033 | 41.4 | 5,735 | [link](https://wandb.ai/cataluna84/lid-bench/runs/gjmsiwvw) |
| 28 | combined_flash_int8_fp16_bs32_ml512_r0 | 47.6 | 72,739 | 0.034 | 76.8 | 9,787 | [link](https://wandb.ai/cataluna84/lid-bench/runs/e154jzzt) |
| 29 | combined_flash_int8_fp16_bs32_ml512_r1 | 47.8 | 72,739 | 0.034 | 79.1 | 9,782 | [link](https://wandb.ai/cataluna84/lid-bench/runs/jyuzj7zy) |
| 30 | combined_flash_int8_fp16_bs32_ml512_r2 | 47.8 | 72,739 | 0.034 | 79.2 | 9,802 | [link](https://wandb.ai/cataluna84/lid-bench/runs/ph7qytf2) |
| 31 | combined_flash_compiled_bf16_bs32_ml256_r0 | 154.6 | 72,371 | 0.012 | 26.3 | 2,881 | [link](https://wandb.ai/cataluna84/lid-bench/runs/p9uy1git) |
| 32 | combined_flash_compiled_bf16_bs32_ml256_r1 | 155.6 | 72,371 | 0.012 | 26.1 | 2,870 | [link](https://wandb.ai/cataluna84/lid-bench/runs/6krs2xeb) |
| 33 | combined_flash_compiled_bf16_bs32_ml256_r2 | 155.4 | 72,371 | 0.012 | 26.1 | 2,875 | [link](https://wandb.ai/cataluna84/lid-bench/runs/ua1fethp) |
| 34 | combined_flash_compiled_bf16_bs32_ml512_r0 | 92.3 | 74,791 | 0.033 | 40.9 | 5,624 | [link](https://wandb.ai/cataluna84/lid-bench/runs/05rd0rqz) |
| 35 | combined_flash_compiled_bf16_bs32_ml512_r1 | 92.5 | 74,791 | 0.033 | 40.9 | 5,626 | [link](https://wandb.ai/cataluna84/lid-bench/runs/g4pn3yj1) |
| 36 | combined_flash_compiled_bf16_bs32_ml512_r2 | 92.8 | 74,791 | 0.033 | 40.7 | 5,621 | [link](https://wandb.ai/cataluna84/lid-bench/runs/zs4itz7x) |

---

## Errors

None (36/36 succeeded)

---

*Generated by lid-bench on 2026-04-16 11:46:13*
