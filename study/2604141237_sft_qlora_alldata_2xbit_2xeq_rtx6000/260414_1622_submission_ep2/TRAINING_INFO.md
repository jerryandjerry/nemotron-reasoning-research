# Training Info — alldata 2x bit/eq, epoch 2, lr=1e-4

**Run name:** `260414_sft_qlora_alldata_2xbit_2xeq_r32_seq2000_ep2_lr1e4_bs22_rtx6000`
**Date:** 2026-04-14 (Chicago time)
**Instance:** RTX PRO 6000 Blackwell (95 GB VRAM), AutoDL connect.bjb1.seetacloud.com:15261

## Config
- **Task:** SFT QLoRA (Unsloth, bf16 source quantized on the fly)
- **Model:** Nemotron-3-Nano-30B-A3B
- **LoRA:** r=32, alpha=32, target=in_proj, out_proj, up_proj, down_proj
- **Optimizer:** adamw_torch
- **LR:** 1e-4 (cosine, warmup 0.05)
- **Batch:** bs=22, grad_accum=1 (eff=22)
- **Epochs:** 2 (resumed from test 5's checkpoint-335 at epoch 1)
- **Max seq len:** 2000
- **Dataset:** same as test 5 — all 6558 + 2x Bit Manipulation + 2x Equation Transformation = 7365 samples
- **Total steps:** 670 (335 from test 5 + 335 epoch 2)
- **Save steps:** 75

## Results
- **Train loss (avg over both epochs):** 0.147
- **Final checkpoint:** checkpoint-670, loss=0.276, lr=8.63e-10, ep=2.00
- **Train time (epoch 2 only):** ~73 min

## Purpose
Test 6: Continue test 5's adapter for one more epoch with lr=1e-4 (lower than test 5's 1.414e-4). Tests whether a second pass over the same data with lower lr improves or overfits.
