# Training Info — softmax_t5 Loss Variant

**Run name:** `260413_2016_sft_qlora_konbu17_loss_softmax_t5_r32_seq2000_ep2_bs22_rtx6000`
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
**Soft-max log-sum-exp (tau=5):** Instead of averaging per-token CE, computes log-sum-exp over all non-masked token losses with temperature tau=5. This upweights the hardest tokens in each sequence, encouraging the model to focus on difficult predictions.

## Results
- **Train loss (avg):** 6.533 (note: soft-max loss scale is higher than standard CE)
- **Final step loss:** 3.345
- **Train time:** 58.5 min (0.98 hrs)
- **VRAM peak:** ~92 GB

## Checkpoints
- checkpoint-250 (loss 5.73, lr 1.58e-06, ep 1.88)
- checkpoint-266 (loss 4.99, lr 2.69e-07, ep 1.95) — final

## Files
- submission.zip (3.1 GB) — for Kaggle submission
- checkpoint-266_loss4.9939_lr2.69e-07_ep1.95.zip (8.8 GB) — full final checkpoint
- train_log.txt
- train_qlora_ran.py (actual script that ran on instance)
- requirements.txt (pip freeze)

## Purpose
A/B test: same baseline config as 2604121541_sft_qlora_unsloth_konbu17_rtx6000, only difference is compute_loss_func=softmax log-sum-exp with tau=5.
