# Training Info — focal_g2 Loss Variant

**Run name:** `260413_2042_sft_qlora_konbu17_loss_focal_g2_r32_seq2000_ep2_bs22_rtx6000`
**Date:** 2026-04-14 (Chicago time)
**Instance:** RTX PRO 6000 Blackwell (95 GB VRAM), AutoDL connect.bjb1.seetacloud.com:15261

## Config
- **Task:** SFT QLoRA (Unsloth, bf16 source quantized on the fly)
- **Model:** Nemotron-3-Nano-30B-A3B
- **LoRA:** r=32, alpha=16, target=all-linear + MoE (gate_up_proj, down_proj)
- **Optimizer:** adamw_8bit
- **LR:** 1.414e-4 (cosine, warmup 0.05)
- **Batch:** bs=22, grad_accum=1 (eff=22)
- **Epochs:** 2
- **Max seq len:** 2000
- **Dataset:** konbu17 filtered CoT (2905 samples)
- **Total steps:** 266
- **Save steps:** 50

## Loss Function
**Focal loss, gamma=2:** Per-token CE multiplied by (1-p)^gamma, where p is model's probability on the correct next token. Confident tokens (p~1) get near-zero weight; uncertain tokens (p~0) get full weight. Dynamic approximation of "maximize min logprob".

## Results
- **Train loss (avg):** 0.322
- **Final step loss:** 0.076
- **Train time:** 58.7 min (0.98 hrs)
- **VRAM peak:** ~95.8 GB

## Checkpoints
- checkpoint-50 (loss 0.33, lr 1.35e-04, ep 0.38)
- checkpoint-100 (loss 0.28, lr 1.05e-04, ep 0.75)
- checkpoint-150 (loss 0.27, lr 6.28e-05, ep 1.13)
- checkpoint-200 (loss 0.23, lr 2.33e-05, ep 1.50)
- checkpoint-250 (loss 0.30, lr 1.58e-06, ep 1.88)
- checkpoint-266 (loss 0.24, lr 2.69e-07, ep 1.95) — final

## Files
- submission.zip (3.1 GB) — for Kaggle submission
- checkpoint-266_focal_g2.zip — full final checkpoint
- train_log.txt
- train_qlora_ran.py = actual script that ran on instance
- requirements.txt (pip freeze)

## Purpose
A/B test: same baseline config as 2604121541_sft_qlora_unsloth_konbu17_rtx6000, only difference is compute_loss_func=weighted_loss_fn with focal loss gamma=2.
