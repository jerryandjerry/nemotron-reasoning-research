# Training Run — 2604111443_sft_lora_bf16_rtx6000

## Method
LoRA (full bf16, no quantization) on the **NVIDIA Nemotron-3-Nano-30B-A3B BF16** model with **real mamba-ssm 2.3.1** + **causal_conv1d 1.6.1** kernels (no pure-PyTorch fallback).

## GPU
NVIDIA RTX PRO 6000 Blackwell Server Edition (97 GB, sm_120) on AutoDL port 23631.

## Stack
- torch 2.8.0+cu128
- transformers 4.56.2
- trl 1.1.0
- peft 0.18.1
- mamba_ssm 2.3.1
- causal_conv1d 1.6.1
- accelerate 1.13.0
- bitsandbytes 0.49.2

## LoRA config
- rank 32, alpha 32, dropout 0.05
- target_modules = `'all-linear'`
- bias = `'none'`

## Training config
- batch_size = 1, grad_accum = 8 (effective batch 8)
- lr = 1e-4, scheduler cosine, warmup_ratio 0.1
- optimizer adamw_torch
- bf16 = True, gradient_checkpointing = True
- max_seq_len = 2500
- num_epochs = 1
- save_steps = 50, save_total_limit = 2

## Dataset
- `final_Nemotron_training_data.csv` filtered by transformation keywords
- 7327 → sampled 3000 → after >2500 token drop, 2961 kept
- 371 total training steps

## Run timeline
- Started at step 0, hit a torch.save crash at step 250 (disk full).
- Resumed from `checkpoint-200` after freeing disk by deleting old checkpoints (50/100/150 manually + corrupted 250).
- Resumed run completed all 371 steps.

## Final metrics
- Train time: **162 min (2.70 h)** end-to-end (including resume).
- Final step (371): loss ~2.6, micro-batch losses bounce per sample.
- Last checkpoint with optimizer state: `checkpoint-350` — loss 10.0181, lr 1.07e-06, ep 0.95, **token_acc 0.6865**.
- Run-aggregate `train/mean_token_accuracy` (averaged over all steps including warmup): 0.4135 — low because early steps had cold loss; later steps were 0.62-0.69.
- Trajectory of token_acc at saved checkpoints: 0.6542 (step 200) → 0.6528 (250) → 0.6764 (300) → 0.6865 (350).

## Artifacts on instance (port 23631)
- `/root/autodl-tmp/adapter_output/adapter_model.safetensors` (3.3 GB) — **final-step adapter (step 371)** written by `trainer.model.save_pretrained()`.
- `/root/autodl-tmp/adapter_output/adapter_config.json` (with `base_model_name_or_path = metric/nemotron-3-nano-30b-a3b-bf16`)
- `/root/autodl-tmp/submission.zip` (3.1 GB) — Kaggle artifact, contains the two files above.
- `/root/autodl-tmp/adapter_output/checkpoint-300_loss10.4717_...` and `checkpoint-350_loss10.0181_...` — only resume-state subfolders left (per `save_total_limit=2`).

## Known issue carried into next run
trl 1.1.0's SFTTrainer breaks transformers' DefaultFlowCallback final-step auto-save, so this run produced **no `checkpoint-371` subfolder**. Final adapter is still saved at `adapter_output/` root via the explicit `save_pretrained()` call, so the shipped artifact is correct. **Fix added** to `train_lora.py` and `train_qlora.py` post-run: `FinalStepSaveCallback` that sets `control.should_save = True` in `on_train_end`. Documented in `02_train/TRAINING_ROUTINE.md` Step 2 → Required code features.

## SwanLab
- Project: `260410_Nemotron`
- Run name: `260411_2309_sft_lora_bf16_3000samp_r32_lr1e4_seq2500_adamw_bs1ga8_rtx6000`
- URL: https://swanlab.cn/@jerry4083/260410_Nemotron/runs/aqzm5xmamq5n121dwggtz

## Files in this folder
- `train_lora.py` — training script (post-run, includes the FinalStepSaveCallback fix)
- `train_log.txt` — full append-mode training log
- `requirements.txt` — `pip freeze` from instance
- `TRAINING_INFO.md` — this file

## To download from AutoDL file manager
- `submission.zip` (3.1 GB) — for Kaggle upload
