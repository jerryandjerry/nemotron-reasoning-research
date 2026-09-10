# QLoRA Unsloth + konbu17 0.72 recipe — 2604121541 run

## Summary
First successful QLoRA run using Unsloth's `FastLanguageModel.from_pretrained(load_in_4bit=True)` on the bf16 source model, with real mamba-ssm 2.3.1 kernels on Blackwell sm_120. Training config copied verbatim from `study/2604121408_sft_lora_konbu17_rtx6000` (konbu17 0.72 recipe) except for batch size / LR which were adjusted for the larger effective batch.

## Method
- **QLoRA via Unsloth** — `FastLanguageModel.from_pretrained(..., load_in_4bit=True)` quantizes attention/FFN on the fly while Unsloth keeps Mamba `in_proj`/`out_proj` in bf16 so the real `mamba_ssm.ops.triton.ssd_combined` kernel path works.
- Source model: `unsloth/Nemotron-3-Nano-30B-A3B` bf16 (on disk at `/root/autodl-tmp/Nemotron-3-Nano-30B-A3B`)

## Stack
- torch 2.8.0+cu128 (Blackwell sm_120)
- transformers 4.56.2
- trl 1.0.0
- peft 0.18.1
- accelerate 1.13.0
- bitsandbytes 0.49.2
- **mamba_ssm 2.3.1**
- **causal_conv1d 1.6.1**
- **unsloth 2026.4.4 / unsloth_zoo 2026.4.6**
- triton 3.4.0

## LoRA config (konbu17 targets + Unsloth fast-patch requirements)
- rank = 32, alpha = 32
- target_modules = `['in_proj', 'out_proj', 'up_proj', 'down_proj']` (list form; Unsloth builds a suffix regex equivalent to konbu17's `.*\.(in_proj|out_proj|up_proj|down_proj)$`, matching Mamba `in_proj`/`out_proj` + non-expert FFN + MoE expert `down_proj`)
- lora_dropout = 0 (required by Unsloth's fast-patch path; konbu17's 0.05 would trigger slow-fallback)
- bias = none
- use_gradient_checkpointing = 'unsloth'
- random_state = 3407
- MoE auto-detection added `mlp.experts.gate_up_proj` + `mlp.experts.down_proj`

## Training config
| param | value |
|---|---|
| dataset | `konbu17_verified_cot_6558rows.csv`, type-balanced sampling (≈2907) |
| prompt format | `…\\boxed{}` suffix with example, CoT cleaned of pre-existing `\boxed{}`, `</think>` close tag |
| SEED | 123 |
| epochs | 2 |
| max_seq_len | 2000 (pre-filter) |
| batch_size | **22** (bs=4 in konbu17 — raised for speed) |
| grad_accum | 1 |
| **eff batch** | **22** (vs konbu17's 8) |
| learning_rate | **1.414e-4** (konbu17 base 1e-4 × √2 for larger eff batch) |
| scheduler | cosine, warmup_ratio 0.05 |
| optimizer | adamw_torch |
| weight_decay | 0.001 |
| save_steps | 50, save_total_limit 2 |
| max_length on SFTConfig | not set (Unsloth pattern — sequence length handled at FastLanguageModel.from_pretrained side) |
| padding_free | not set (Unsloth default) |

## Stress test findings before launch
Stress tested bs=16/20/22/24/32 with top-K longest samples (1960 tokens) and SequentialSampler. bs=24 had only 1 GB stress headroom; bs=22 had 7 GB. Chose bs=22 for safer real-run margin after learning the stress test peak underestimates real training memory by ~2 GB (cuDNN workspace / torch.compile artifacts accumulate over 50+ steps and `torch.cuda.empty_cache()` cannot release them).

## Results
- **Total time:** 59.2 min
- **Final step:** 266 / 266 (2 epochs × 133 steps/epoch)
- **Final loss:** 0.3393
- **Final lr:** 2.69e-7 (cosine decay to ~0)
- **Adapter keys:** 11960
- **Adapter size:** 3.3 GB
- **submission.zip:** 3.1 GB

### Loss trajectory
| checkpoint | loss | lr | epoch |
|---|---|---|---|
| 50 | 0.4374 | 1.35e-4 | 0.38 |
| 100 | 0.3866 | 1.05e-4 | 0.75 |
| 150 | 0.3642 | 6.28e-5 | 1.13 |
| 200 | 0.3346 | 2.33e-5 | 1.50 |
| 250 | 0.3950 | 1.58e-6 | 1.88 |
| 266 (final) | 0.3393 | 2.69e-7 | 1.95 |

Note: step 250's loss (0.3950) was a single noisy micro-batch at the end of epoch 2 just before decay bottomed out; step 266 returned to the 0.33-0.34 plateau.

## Artifacts on instance (port 15261, cleaned up after closeout)
- `/root/autodl-tmp/adapter_output/adapter_model.safetensors` (3.3 GB) — **final-step adapter**
- `/root/autodl-tmp/adapter_output/adapter_config.json` (`base_model_name_or_path = metric/nemotron-3-nano-30b-a3b-bf16`, `inference_mode=True`, `lora_dropout=0.0`)
- `/root/autodl-tmp/submission.zip` (3.1 GB) — for Kaggle submission
- `/root/autodl-tmp/adapter_output/checkpoint-250_...` and `checkpoint-266_...` — resume-state subfolders

## SwanLab
- Project: `260410_Nemotron`
- Run name: `260412_1715_sft_qlora_unsloth_konbu17_r32_lr1e4_seq2000_ep2_bs4ga2_rtx6000`
- URL: https://swanlab.cn/@jerry4083/260410_Nemotron/runs/9v8kmamkj8jhcsovn1ym7

## Files in this folder
- `train_qlora.py` — training script as uploaded and run
- `train_log.txt` — full training log (per-microbatch sample_id/padded_len/content, per-step loss/smi, progress bar)
- `requirements.txt` — `pip freeze` from the instance
- `TRAINING_INFO.md` — this file

## To download from AutoDL file manager
- `submission.zip` (3.1 GB) — for Kaggle upload
