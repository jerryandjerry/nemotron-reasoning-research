# Training Info — topk32 Loss Variant

**Run name:** `260413_2042_sft_qlora_konbu17_loss_topk32_r32_seq2000_ep2_bs22_rtx6000`
**Date:** 2026-04-13 (Chicago time)
**Instance:** RTX PRO 6000 Blackwell (95 GB VRAM), AutoDL connect.bjb1.seetacloud.com:15261

## Config
- **Task:** SFT QLoRA (Unsloth, bf16 source quantized on the fly)
- **Model:** Nemotron-3-Nano-30B-A3B
- **LoRA:** r=32, alpha=16, target=all-linear + MoE (gate_up_proj, down_proj)
- **Optimizer:** adamw_torch
- **LR:** 1.414e-4 (cosine, warmup 0.05)
- **Batch:** bs=22, grad_accum=1 (eff=22)
- **Epochs:** 2
- **Max seq len:** 2000
- **Dataset:** konbu17 filtered CoT (2905 samples)
- **Total steps:** 266
- **Save steps:** 50

## Loss Function
**Top-k worst tokens (k=32):** For each sequence, averages ONLY the 32 highest-loss per-token CE values. Trains on the hardest tokens in each trace, ignoring tokens the model already predicts well.

## Results
- **Train loss (avg):** 3.579
- **Final step loss:** 1.020
- **Train time:** 61.5 min (1.03 hrs)
- **VRAM peak:** ~94 GB

## Checkpoints
- checkpoint-250 (loss 3.08, lr 1.58e-06, ep 1.88)
- checkpoint-266 (loss 2.89, lr 2.69e-07, ep 1.95) — final

## Files
- submission.zip (3.1 GB) — for Kaggle submission
- checkpoint-266_loss2.8901_lr2.69e-07_ep1.95.zip (9.1 GB) — full final checkpoint
- train_log.txt
- train_qlora_ran.py (actual script that ran on instance)
- requirements.txt (pip freeze)

## Purpose
A/B test: same baseline config as 2604121541_sft_qlora_unsloth_konbu17_rtx6000, only difference is compute_loss_func=top-k worst 32 tokens per sequence.
