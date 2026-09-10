# Training Info — huikang 0.85 reproduction

**Run name:** `260513_0520_sft_qlora_huikang_repro_r32_a32_lr2e4_seq8192_bs32_rtx6000`
**Date:** 2026-05-13 (Chicago time)
**Instance:** RTX PRO 6000 Blackwell (95 GB VRAM), AutoDL connect.bjb1.seetacloud.com:21578

## Config (exact huikang reproduction)
- **Task:** SFT LoRA (bf16 base, NOT 4-bit) + Cut Cross-Entropy + MoE weight tying
- **Model:** Nemotron-3-Nano-30B-A3B (full bf16)
- **LoRA:** r=32, alpha=32, dropout=0.0, 9 targets (q/k/v/o_proj, up/down_proj, in/out_proj, lm_head)
- **Manual lm_head LoRA** (Unsloth drops it for MoE)
- **LoRA params fp32**, base bf16
- **Optimizer:** AdamW, lr=2e-4 linear decay to 0, beta2=0.95, no weight decay
- **Batch:** bs=32, micro_batch=4
- **Epochs:** 1 (244 steps)
- **Max seq len:** 8192
- **Dataset:** huikang's pre-tokenized corpus, 7830 examples, 26.6M unmasked tokens
- **Response-only masking** (prompt tokens weight=0)
- **Training order:** replayed from huikang's winning run index.jsonl
- **RESET_WEIGHTS=True** (fresh LoRA init)

## Results
- **Final loss:** 0.002
- **Train time:** 228.3 min (3.81 hrs)
- **Peak smi:** ~89 GB
- **submission.zip:** 1.3 GB

## Purpose
Exact reproduction of huikang's end-to-end-finetuning-for-lb-0-85 notebook. Expected score: 0.85.
