# Submission 2 — QLoRA 4-bit on RTX PRO 6000 (98GB)
**Date:** 2026-04-10 ~22:30 CDT
**GPU:** NVIDIA RTX PRO 6000 Blackwell Server Edition (95GB VRAM)

## Training Config
| Parameter | Value |
|-----------|-------|
| Base model | Nemotron-3-Nano-30B-A3B-bnb-4bit (pre-quantized BNB NF4) |
| Method | QLoRA (LoRA on 4-bit quantized model) |
| LoRA rank | 32 |
| LoRA alpha | 32 |
| LoRA target | all-linear |
| LoRA dropout | 0.05 |
| Optimizer | adamw_torch |
| Learning rate | 2e-4 (cosine, 10% warmup) |
| Batch size | 8 |
| Gradient accumulation | 2 (effective batch=16) |
| Epochs | 1 |
| Max seq len | 2500 |
| Samples | 2961 (3000 sampled seed=99, 39 dropped > 2500 tokens) |
| Training steps | 186 |
| Training time | 58 min |

## Results
| Metric | Value |
|--------|-------|
| Final loss | 1.53 |
| Avg train loss | 2.63 |
| Mean token accuracy | 0.7220 |
| Peak VRAM | ~71GB |

## Key Differences from Submission 1
| | Submission 1 (5090) | Submission 2 (RTX 6000) |
|---|---|---|
| GPU | RTX 5090 (32GB) | RTX PRO 6000 (95GB) |
| Optimizer | adamw_8bit | **adamw_torch** |
| Batch size | 1 | **8** |
| Grad accum | 4 | **2** |
| Effective batch | 4 | **16** |
| LR | 1e-4 | **2e-4** (√4 scaling) |
| Training time | 5.7 hrs | **58 min** |
| gc.collect + empty_cache | yes | yes |

## Files
- `adapter_config.json` — LoRA adapter configuration
- `adapter_model.safetensors` — (download from AutoDL, 1.7GB)
- `submission.zip` — (download from AutoDL, 1.3GB)
- `train_qlora.py` — Training script
- `train_log.txt` — Full training log with per-step sample logging
- `requirements.txt` — Python dependencies
