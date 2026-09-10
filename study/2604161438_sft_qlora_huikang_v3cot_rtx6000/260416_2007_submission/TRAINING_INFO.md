# Training Info — huikang adapter + v3 CoT

**Run name:** `260416_1700_sft_qlora_huikang_v3cot_r32_a16_lr1e4_seq8000_ep1_rtx6000`
**Date:** 2026-04-16 (Chicago time)
**Instance:** RTX PRO 6000 Blackwell (95 GB VRAM), AutoDL connect.bjb1.seetacloud.com:21578

## Config
- **Task:** SFT QLoRA (Unsloth, bf16 source quantized on the fly)
- **Model:** Nemotron-3-Nano-30B-A3B
- **Pretrained adapter:** huikang's 0.877 submission adapter
- **LoRA:** r=32, alpha=16 (scale 0.5), dropout=0.05, targets=in_proj/out_proj/up_proj/down_proj
- **Optimizer:** adamw_torch
- **LR:** 1e-4 (cosine, warmup 0.05)
- **Batch:** bs=4, ga=1 (eff=4)
- **Epochs:** 1
- **Max seq len:** 8000
- **Dataset:** v3 CoT — bit_manip(1602) + eq_crypt(823) + eq_numeric(732) = 3157 samples
- **Total steps:** 790
- **Save steps:** 150

## Data Distribution
| Category | Count |
|---|---|
| bit_manipulation | 1602 |
| equation_cryptarithm | 823 |
| equation_numeric | 732 |
| **Total** | **3157** |

## Results
- **Train loss (avg):** 0.055
- **Final step loss:** 0.018
- **Train time:** 174.2 min (2.9 hrs)
- **Peak smi:** ~87-89 GB

## Files
- submission.zip (3.1 GB)
- checkpoint-790_loss0.0272_lr4.39e-10_ep1.00.zip (9.1 GB)
- train_log.txt
- train_qlora_ran.py
- requirements.txt

## Purpose
Continue from huikang's 0.877 adapter, training only on the 3 weak categories (bit_manipulation, equation_cryptarithm, equation_numeric) with new v3 CoT data. Huikang's adapter already scores 100% on other categories.
