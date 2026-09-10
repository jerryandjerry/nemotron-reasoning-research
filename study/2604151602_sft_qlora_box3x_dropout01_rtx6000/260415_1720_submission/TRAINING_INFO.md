# Training Info — box3x + dropout 0.1

**Run name:** `260415_1605_sft_qlora_konbu17_box3x_dropout01_r32_seq2000_ep2_bs22_rtx6000`
**Date:** 2026-04-15 (Chicago time)
**Instance:** RTX PRO 6000 Blackwell (95 GB VRAM), AutoDL connect.bjb1.seetacloud.com:15261

## Config
- **Task:** SFT QLoRA (Unsloth, bf16 source quantized on the fly)
- **Model:** Nemotron-3-Nano-30B-A3B
- **LoRA:** r=32, alpha=32, target=in_proj, out_proj, up_proj, down_proj
- **lora_dropout:** 0.1 (triggers Unsloth slow fallback)
- **Optimizer:** adamw_torch
- **LR:** 1.414e-4 (cosine, warmup 0.05)
- **Batch:** bs=22, grad_accum=1 (eff=22)
- **Epochs:** 2
- **Max seq len:** 2000
- **Dataset:** konbu17 type-balanced (2905 samples)
- **Total steps:** 266
- **Loss:** boxed 3x weighted CE (BOXED_LOSS_WEIGHT=3.0)

## Results
- **Train loss (avg):** 0.332
- **Final checkpoint:** checkpoint-266, loss=0.332, lr=2.69e-07, ep=1.95
- **Train time:** 66.8 min (slower due to dropout disabling Unsloth fast-patch)

## Purpose
Test 7: Two changes vs baseline — lora_dropout=0.1 (regularization) + boxed 3x loss (gentler than boxed5x). Tests whether dropout reduces overfitting and 3x is a better weight ratio.
