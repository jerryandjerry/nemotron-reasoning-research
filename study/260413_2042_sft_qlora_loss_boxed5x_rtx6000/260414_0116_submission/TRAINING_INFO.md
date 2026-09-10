# Training Info — boxed5x Loss Variant

**Run name:** `260413_2042_sft_qlora_konbu17_loss_boxed5x_r32_seq2000_ep2_bs22_rtx6000`
**Date:** 2026-04-13 (Chicago time)
**Instance:** RTX PRO 6000 Blackwell (95 GB VRAM), AutoDL connect.bjb1.seetacloud.com:15261

## Config
- **Task:** SFT QLoRA (Unsloth, bf16 source quantized on the fly)
- **Model:** Nemotron-3-Nano-30B-A3B
- **LoRA:** r=32, alpha=16, target=all-linear + MoE (gate_up_proj, down_proj)
- **Optimizer:** adamw_8bit
- **LR:** 1.414e-4 (cosine, warmup 0.05)
- **Batch:** bs=4, grad_accum=2 (eff=8)
- **Epochs:** 2
- **Max seq len:** 2000
- **Dataset:** konbu17 filtered CoT (2905 samples)
- **Total steps:** 266
- **Save steps:** 50

## Loss Function
**Weighted boxed-answer loss (5x):** Standard per-token CE, but every token from the last `\boxed{` to end-of-sequence gets weight 5.0. Inspired by Kh0a's 0.73 notebook.

## Results
- **Train loss (avg):** 0.407
- **Final step loss:** 0.195
- **Train time:** 62.7 min (1.04 hrs)
- **VRAM peak:** ~93.9 GB

## Checkpoints
- checkpoint-50 (loss 0.42, lr 1.35e-04, ep 0.38)
- checkpoint-100 (loss 0.37, lr 1.05e-04, ep 0.75)
- checkpoint-150 (loss 0.35, lr 6.28e-05, ep 1.13)
- checkpoint-200 (loss 0.32, lr 2.33e-05, ep 1.50)
- checkpoint-250 (loss 0.38, lr 1.58e-06, ep 1.88)
- checkpoint-266 (loss 0.32, lr 2.69e-07, ep 1.95) — final

## Files
- submission_loss_boxed5x.zip (3.1 GB) — for Kaggle submission
- checkpoint-266_boxed5x.zip — full final checkpoint
- train_log.txt
- train_qlora.py (train_qlora_ran.py = actual script that ran on instance)
- requirements.txt (pip freeze)

## Purpose
A/B test: same baseline config as 2604121541_sft_qlora_unsloth_konbu17_rtx6000, only difference is compute_loss_func=weighted_loss_fn with 5x weight on boxed answer region.
