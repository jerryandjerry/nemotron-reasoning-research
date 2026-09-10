# Training Info — v1 CoT from scratch

**Run name:** `260417_1601_sft_qlora_v1cot_scratch_r32_a16_lr1e4_seq8000_ep1_bs4_rtx6000`
**Date:** 2026-04-17 (Chicago time)
**Instance:** RTX PRO 6000 Blackwell (95 GB VRAM), AutoDL connect.bjb1.seetacloud.com:21578

## Config
- **Task:** SFT QLoRA (Unsloth, bf16 source quantized on the fly)
- **Model:** Nemotron-3-Nano-30B-A3B (from scratch, no pretrained adapter)
- **LoRA:** r=32, alpha=16 (scale 0.5), dropout=0.05, targets=in_proj/out_proj/up_proj/down_proj
- **Optimizer:** adamw_torch
- **LR:** 1e-4 (cosine, warmup 0.05)
- **Batch:** bs=4, ga=1 (eff=4)
- **Epochs:** 1
- **Max seq len:** 8000
- **Dataset:** v1 CoT, all 7 categories, cap 800 per category
- **Total samples:** 4625
- **Total steps:** 1157
- **Save steps:** 225

## Data Distribution
| Category | Available | Sampled |
|---|---|---|
| bit_manipulation | 1364 | 800 |
| equation_cryptarithm | 65 | 65 (all) |
| equation_numeric | 560 | 560 (all) |
| gravitational_constant | 1597 | 800 |
| number_conversion | 1576 | 800 |
| text_encryption | 1576 | 800 |
| unit_conversion | 1594 | 800 |
| **Total** | **8332** | **4625** |

## Results
- **Train loss (avg):** 0.058
- **Final step loss:** 0.029
- **Train time:** 238.4 min (3.97 hrs)
- **Peak smi:** ~88 GB

## Files
- submission.zip (3.1 GB)
- checkpoint-1157_loss0.0264_lr1.31e-08_ep0.99.zip (9.1 GB)
- train_log.txt
- train_qlora_ran.py
- requirements.txt
