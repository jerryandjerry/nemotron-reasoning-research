# Training Run — 2604121408_sft_lora_konbu17_rtx6000

## Method
LoRA bf16 on **NVIDIA-Nemotron-3-Nano-30B-A3B-BF16** with **real mamba-ssm 2.3.1 + causal_conv1d 1.6.1** kernels. Recipe ported 1:1 from [konbu17's 0.72 Kaggle notebook](study/260411_precedent_0.72_konbu17/nemotron-sft-lora-with-cot-v2-prep-now-plz-wait.ipynb), with our instrumentation (SwanLab, LoggingSFTTrainer, FinalStepSaveCallback, resume logic) layered on top.

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
- `target_modules = r".*\.(in_proj|out_proj|up_proj|down_proj)$"` — Mamba in/out + MoE FFN up/down only (**not** `all-linear` like Atah Alam)
- Adapter keys: 11960

## Training config
- batch_size 4 × grad_accum 2 = effective batch **8** (matches konbu17)
- LR 1e-4 (same as konbu17), cosine, warmup_ratio 0.05
- optimizer adamw_torch
- bf16 = True, gradient_checkpointing = True (use_reentrant=False)
- max_seq_len **2000** (drops 2 outliers at 2494/2519 tokens; all remaining samples ≤ 1960)
- num_epochs **2** (matches konbu17)
- save_strategy='steps', save_steps=100, save_total_limit=2, **FinalStepSaveCallback** (forces save at final step)
- seed 123 (matches konbu17)

## Dataset
`konbu17_verified_cot_6558rows.csv` (verified CoT, 6558 rows, 6 types) with **konbu17's type-balanced sampling**:
- Numeral Conversion: 300
- Gravitational Constant: 400
- Unit Conversion: 700
- Text Encryption: 700
- Bit Manipulation: 607 (all available)
- Equation Transformation: 200 (all available)

Total **2907 sampled** → 2905 after outlier drop → **728 total training steps** over 2 epochs.

## CoT format (the key difference from Atah Alam 0.61 run)
- User: `{prompt}\nPlease put your final answer inside \`\\boxed{}\`. For example: \`\\boxed{your answer}\``
- Assistant: `{cot_cleaned}\n</think>\n\\boxed{answer}` — **explicit `</think>` close tag**
- CoT cleaned: `re.sub(r'\\boxed\{[^}]*\}', '', cot).rstrip()` to strip pre-existing mid-reasoning `\boxed{}` calls

Nemotron's chat template detects the `</think>` and skips auto-injecting an empty `<think></think>` block at the start of the assistant turn. Result: CoT lives **inside** the reasoning block, final answer lives **outside**. Atah Alam's format (no explicit `</think>`) caused the chat template to emit empty `<think></think>` followed by CoT + answer as non-thinking output — which is why the previous run scored only 0.61.

## Run timeline
- Started from scratch at 04:09 Beijing / 15:09 Chicago (2026-04-12)
- 728 optimizer steps completed cleanly in a single run (no resume needed)
- Finished at 07:14 Beijing / 18:14 Chicago (2026-04-12)
- Wall time: **184.5 min (3.07 hrs)**
- Avg 15.0 s/step (stress test predicted 22 s/step at max_len=2000 bs=4 ga=2; real was faster because dataset mean is 627 tokens, not the full 2000)

## Final metrics (from train_log.txt)
- Final loss (last logging step): **0.6078**
- Final run-aggregate `mean_token_accuracy`: **0.9035**
- Final lr: 4.19e-8 (cosine essentially at 0)
- Best intermediate checkpoint by loss: **checkpoint-600** (loss 0.5619, acc 0.9042, lr 8.36e-6, ep 1.65)
- Final saved checkpoint: **checkpoint-728** (loss 0.6078, acc 0.9035, lr 4.19e-8, ep 1.98) — via FinalStepSaveCallback
- Adapter health: "looks healthy" verification passed, weight norms non-trivial

## Checkpoint trajectory (run-aggregate token_acc)
| Step | loss | token_acc |
|---|---|---|
| 100 | 0.8866 | 0.8708 |
| 200 | 0.6912 | 0.8852 |
| 300 | 0.7108 | 0.8938 |
| 400 | 0.5965 | 0.9032 |
| 500 | 0.6712 | 0.8907 |
| 600 | 0.5619 | **0.9042** |
| 700 | 0.6707 | 0.8886 |
| 728 (final) | 0.6078 | 0.9035 |

Monotonic-ish improvement through step 400, then oscillating around 0.89-0.90 as cosine LR decayed. Best checkpoint is 600.

## Artifacts on instance (port 23631)
- `/root/autodl-tmp/adapter_output/adapter_model.safetensors` (3.3 GB) — final step 728 adapter
- `/root/autodl-tmp/adapter_output/adapter_config.json` (with `base_model_name_or_path = metric/nemotron-3-nano-30b-a3b-bf16`, `inference_mode = True`, `lora_dropout = 0.0`)
- `/root/autodl-tmp/adapter_output/checkpoint-700_...` and `checkpoint-728_...` (resume-state subfolders, save_total_limit=2)
- `/root/autodl-tmp/submission.zip` (3.08 GB) — **Kaggle submission artifact**, contains `adapter_model.safetensors` + `adapter_config.json`

## SwanLab
- Project: `260410_Nemotron`
- Run name: `260412_1309_sft_lora_konbu17_r32_lr1e4_seq2000_ep2_bs4ga2_rtx6000`
- URL: https://swanlab.cn/@jerry4083/260410_Nemotron/runs/lbi49wkjgnku19l0rr98t

## Files in this folder
- `train_lora.py` — training script as run on the instance (konbu17 recipe + FinalStepSaveCallback)
- `train_log.txt` — full append-mode training log (952 KB)
- `requirements.txt` — `pip freeze` from instance
- `TRAINING_INFO.md` — this file

## To download from AutoDL file manager
- `submission.zip` (3.08 GB) — upload to Kaggle via MCP

## Kaggle submission
Use **Step 5 routine**: kaggle MCP tools. Submission description:
```
260412_sft_lora_konbu17_r32_lr1e4_seq2000_ep2_bs4ga2_rtx6000_step728_mamba231
```

## Differences from Atah Alam previous run (0.61)
| | Atah Alam (0.61) | konbu17 (this) |
|---|---|---|
| target_modules | `all-linear` | Mamba+MoE-FFN only |
| CoT format | no `</think>` → chat template inserts empty `<think></think>` | explicit `</think>` → proper reasoning block |
| Dataset | `final_Nemotron_training_data.csv` keyword filter, random 3000 | `konbu17_verified_cot_6558rows.csv` type-balanced 2907 |
| max_seq_len | 2500 (drops 30+ long samples) | 2000 (drops 2 outliers, keeps everything else) |
| num_epochs | 1 | 2 |
| warmup_ratio | 0.1 | 0.05 |
| bs × ga (effective) | 1 × 8 (8) | 4 × 2 (8) |
| Final loss | ~11.6 (noisy) | 0.61 |
| Final run-agg acc | 0.4135 | **0.9035** |
| Kaggle score | 0.61 | TBD |
