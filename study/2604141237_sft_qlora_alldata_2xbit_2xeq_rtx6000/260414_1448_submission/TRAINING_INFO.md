# Training Info — alldata 2x bit/eq

**Run name:** `260414_sft_qlora_alldata_2xbit_2xeq_r32_seq2000_ep1_bs22_rtx6000`
**Date:** 2026-04-14 (Chicago time)
**Instance:** RTX PRO 6000 Blackwell (95 GB VRAM), AutoDL connect.bjb1.seetacloud.com:15261

## Config
- **Task:** SFT QLoRA (Unsloth, bf16 source quantized on the fly)
- **Model:** Nemotron-3-Nano-30B-A3B
- **LoRA:** r=32, alpha=32, target=in_proj, out_proj, up_proj, down_proj
- **Optimizer:** adamw_torch
- **LR:** 1.414e-4 (cosine, warmup 0.05)
- **Batch:** bs=22, grad_accum=1 (eff=22)
- **Epochs:** 1
- **Max seq len:** 2000
- **Dataset:** konbu17_verified_cot_6558rows.csv — ALL rows + 2x Bit Manipulation + 2x Equation Transformation
- **Total samples:** 7365
- **Total steps:** 335
- **Save steps:** 75

## Data Distribution
| Category | Count |
|---|---|
| Gravitational Constant | 1511 |
| Numeral Conversion | 1491 |
| Text Encryption | 1407 |
| Unit Conversion | 1342 |
| Bit Manipulation | 1214 (2x607) |
| Equation Transformation | 400 (2x200) |
| **Total** | **7365** |

## Results
- **Train loss (avg):** 0.253
- **Final step loss:** 0.313
- **Train time:** 57.1 min
- **Resumed from checkpoint-75 after disk-full crash**

## Files
- submission.zip (3.1 GB)
- checkpoint-335_loss0.3377_lr1.24e-07_ep0.99.zip (9.1 GB)
- train_log.txt
- train_qlora_ran.py
- requirements.txt

## Purpose
A/B test: data strategy. Use all 6558 training samples (no category caps) + duplicate underrepresented categories (Bit Manipulation, Equation Transformation). 1 epoch to keep total steps comparable to baseline (335 vs 266).
