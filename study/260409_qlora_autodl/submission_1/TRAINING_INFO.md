# Submission 1 — QLoRA 4-bit Training
**Date:** 2026-04-10
**GPU:** NVIDIA GeForce RTX 5090 (32GB) on AutoDL

## Training Config
| Parameter | Value |
|-----------|-------|
| Base model | hupakabras/Nemotron-3-Nano-30B-A3B-bnb-4bit (pre-quantized BNB NF4) |
| Method | QLoRA (LoRA on 4-bit quantized model) |
| LoRA rank | 32 |
| LoRA alpha | 32 |
| LoRA target | all-linear |
| LoRA dropout | 0.05 |
| Optimizer | adamw_8bit |
| Learning rate | 1e-4 (cosine, 10% warmup) |
| Batch size | 1 (grad_accum=4, effective=4) |
| Epochs | 1 |
| Max seq len | 2500 |
| Samples | 2961 (3000 sampled, 39 dropped > 2500 tokens) |
| Training steps | 741 |
| Training time | 5.69 hours |

## Data
- Source: `final_Nemotron_training_data.csv` (9500 samples with CoT)
- Filtered by 15 keywords to 7327, sampled 3000 with seed=99
- Same recipe as konbu17's 0.72 public notebook

## Key Implementation Details
- `torch.cuda.empty_cache()` after each step to prevent OOM
- Checkpoints saved every 100 steps (save_total_limit=2)
- Model's `modeling_nemotron_h.py` patched: pure-PyTorch Mamba2 (no mamba-ssm dependency)
- Model's `_initialize_missing_keys` signature fixed for transformers 4.56.2
- `quantization_config` added to config.json (original model was missing it)
- HF cache cleared before each run to avoid stale model code

## Files
- `adapter_config.json` — LoRA adapter configuration
- `adapter_model.safetensors` — LoRA adapter weights (1.77 GB)
- `submission.zip` — Competition submission (adapter_config.json + adapter_model.safetensors)
- `train_qlora.py` — Training script
- `train_log.txt` — Full training log
- `requirements.txt` — Python dependencies

## To Reproduce
1. Set up AutoDL instance with RTX 5090 (32GB)
2. Download model: `hupakabras/Nemotron-3-Nano-30B-A3B-bnb-4bit` to `/root/autodl-tmp/`
3. Patch model files (see train_qlora.py comments)
4. Place `final_Nemotron_training_data.csv` in `/root/autodl-tmp/data/`
5. Run: `python train_qlora.py`

## Expected Score
- Based on 0.72 scheme with bf16: 0.72
- QLoRA typically degrades ~3%: expected ~0.69-0.70
- Needs evaluation to confirm
