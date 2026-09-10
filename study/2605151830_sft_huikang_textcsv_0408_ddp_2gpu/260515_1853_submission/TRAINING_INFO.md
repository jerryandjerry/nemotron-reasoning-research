# Training Info — huikang 7830 (260514 CSV) + Kaggle regex + DDP 2-GPU

> Trained: 2026-05-15 (Chicago) — started 16:42, finished 18:53 (closeout 18:56)
> Instance: AutoDL RTX PRO 6000 Blackwell ×2 (95 GB each), port 24988
> SwanLab: `260515_1637_sft_huikang_textcsv0408_ddp2gpu_r32_a32_lr2e4_seq8192_bs32_rtx6000`

## Goal

Reproduce the 0.84 recipe at **~1.85× wall-clock speedup** by switching to DDP across 2 GPUs while keeping effective batch=32 unchanged. Also incorporate two fixes vs the 0.84 single-GPU run:

1. **Kaggle-metric `\boxed{}` extractor** instead of the buggy `[^}]*` regex (correctly captures cryptarithm answers containing `}`, `|`, `:`, `$`, etc.)
2. **New 260514 huikang CSV** with `oversampling` column — 6,171 unique rows expanded to 7,830 via `index.jsonl` suffix-strip mapping

## Data

`/root/autodl-tmp/data/260514_huikang_update/huikang_7830.csv` (34 MB)

- 6,171 unique base pids
- Oversampling column sums to 7,830 (matches huikang's `index.jsonl` count for epoch 0)
- Order: huikang's `index.jsonl` curriculum, stripping `-dN`/`-pN` suffix to look up base CSV rows
- All bases match between CSV and index.jsonl, per-base counts match exactly
- Source: `pretok_0408_decoded` (decoded from 04-08 pre-tokenized data)

Category breakdown (unique):
- cipher 1,576 / bit_manipulation 1,354 / unit_conversion 990 / gravity 975 / numeral 650 / equation_numeric_deduce 540 / cryptarithm_deduce 54 / equation_numeric_guess 21 / cryptarithm_guess 11

## Config (matches 0.84 recipe except DDP + regex)

| Parameter | Value |
|---|---|
| Model | Nemotron-3-Nano-30B-A3B bf16 (Unsloth) |
| Method | LoRA bf16 + Cut Cross-Entropy + MoE weight tying + DDP |
| LoRA | r=32, alpha=32, dropout=0 |
| Targets | q/k/v/o_proj, up/down_proj, in/out_proj, lm_head |
| Optimizer | AdamW, betas=(0.9, 0.95), eps=1e-8, wd=0 |
| LR | 2e-4 → 0 linear decay |
| Effective batch | **32** (same as 0.84 run) |
| Per-rank batch | 16 (split across 2 GPUs) |
| Micro batch | 4 |
| Accum steps per rank | 4 |
| Max seq len | 8192 |
| MoE tying | True (5,888 params) |
| World size | 2 (DDP, `static_graph=True`) |
| Ordering | huikang `index.jsonl` epoch 0, suffix-stripped base-pid lookup |
| Steps | 244 |

### DDP loss normalization (HF PR #34191 pattern)

Each rank computes local CE sum, all-reduces global completion-token count, then:
```
loss = local_ce_sum / global_weight * WORLD_SIZE
```
The `*WORLD_SIZE` cancels DDP's automatic `/world_size` gradient averaging, so the effective gradient is identical to single-GPU on the same 32-sample batch.

### Critical DDP fixes

- `static_graph=True` — required for Unsloth's reentrant-backward gradient checkpointing (without it, DDP fires the reducer hook twice for the same param and crashes)
- `torch.cuda.set_device(LOCAL_RANK)` **before** `init_process_group` — avoids NCCL "no GPUs found" race at init
- Adapter/LoRA setup and CCE forward patches applied to `_base` (inner model) **before** DDP wrap — these survive `DDP(model)`

## Results

| Metric | Value | vs 0.84 (single-GPU) |
|---|---|---|
| **Train time** | **128.1 min (2.14 hr)** | 236.8 min — **1.85× faster** |
| **Final loss** | 0.0027 | 0.0023 (within noise) |
| Peak VRAM | 96.99 GB / 97.28 GB | 88.9 GB (DDP adds ~8 GB grad sync overhead) |
| Trainable params | 888,154,112 / 32,466,091,456 | same |
| Steps | 244 | 245 |
| Token verification | 5/5 MATCH (base-pid stripped lookup) | 5/5 MATCH |

### Loss curve vs single-GPU 0.84 baseline (selected steps)

| Step | DDP (this run) | Single-GPU 0.84 |
|---|---|---|
| 2 | 0.460 | 0.459 |
| 5 | 0.331 | 0.330 |
| 18 | 0.128 | 0.129 |
| 28 | 0.041 | 0.039 |
| 100 | 0.0038 | (similar) |
| 200 | 0.0028 | (similar) |
| 244 | 0.0027 | 0.0023 |

Curves track each other to within ≤0.003 at every step → DDP gradient math is correct.

## Checkpoints saved

5 saves total (steps 50, 100, 150, 200, 244). FIFO keeps last 2 on disk. Final state:
- `adapter_output_huikang_textcsv_0408_ddp/checkpoint-200_loss0.0028_lr3.69e-05/`
- `adapter_output_huikang_textcsv_0408_ddp/checkpoint-244_loss0.0027_lr8.20e-07/`

## Outputs

| File | Size | Source |
|---|---|---|
| `submission.zip` | 1.31 GB (1,373,462,127 B) | adapter_config.json + adapter_model.safetensors |
| `checkpoint-244_loss0.0027_lr8.20e-07.zip` | 2.9 GB (3,027,032,806 B) | full checkpoint for resume |
| `train_log.txt` | 451 KB | per-step + per-microbatch log |
| `requirements.txt` | 5.6 KB | `pip freeze` (206 packages) |
| `train_huikang_textcsv_0408_ddp.py` | 30 KB | training script |

Adapter post-processing applied:
- `base_model_name_or_path` → `metric/nemotron-3-nano-30b-a3b-bf16`
- `inference_mode` → `true`
- `lm_head` keys renamed `base_model.model.lm_head.*` → `base_model.model.backbone.lm_head.*`

## Score

Pending Kaggle submission.

## Notes

- The Kaggle regex fix was exercised on real cryptarithm rows during training. Examples observed: `\boxed{:/$^}`, `\boxed{""!^}`, `\boxed{{{!?}`, `\boxed{|>>>}`, `\boxed{$|\:}`, `\boxed{!<[$}`, `\boxed{#{"!}`, `\boxed{`]|[}`, `\boxed{>^`>}`, `\boxed{\^]]}` — all would be truncated by the old buggy `[^}]*\}` regex.
- This is the first DDP run on this competition. Recipe is now portable to future 2-GPU instances with same VRAM budget.
