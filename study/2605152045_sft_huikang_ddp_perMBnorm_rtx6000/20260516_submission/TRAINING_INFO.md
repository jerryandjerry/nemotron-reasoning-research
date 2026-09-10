# Training Info — huikang 7830 + DDP 2-GPU + per-MB loss normalization

> Trained: 2026-05-16 (Chicago) — started 10:52, finished 12:59 (closeout 13:04)
> Instance: AutoDL RTX PRO 6000 Blackwell ×2 (95 GiB each), port 24988
> SwanLab: `260516_xxxx_sft_huikang_ddp2gpu_perMBnorm_r32_a32_lr2e4_seq8192_bs32_rtx6000`

## Goal

Fix the loss-normalization bug from the previous DDP run (`2605151830_sft_huikang_textcsv_0408_ddp_2gpu`, scored 0.73 vs single-GPU baseline 0.84). That run used GLOBAL per-token normalization (`loss = ce_sum / global_weight × WORLD_SIZE`), producing micro-average gradients that don't match the 0.84 recipe's per-MB macro-average normalization.

This run uses PER-MB normalization (`loss = mb_ce_sum / mb_weight_sum`, then `(loss / n_accum).backward()`) + **contiguous rank split** (rank 0 gets `global_batch[0:16]`, rank 1 gets `[16:32]`) so each rank's MB structure exactly mirrors single-GPU's 8 MBs of 4 samples.

## Data

`/root/autodl-tmp/data/260514_huikang_update/huikang_7830.csv` (34 MB)
- 6,171 unique base pids + `oversampling` column summing to 7,830
- Ordering: huikang's `index.jsonl` (epoch 0) with suffix-strip lookup (each occurrence in index.jsonl uses the same base CSV row)
- Source: `pretok_0408_decoded`
- Token verification: all 5 PASSED

## Config

| Parameter | Value |
|---|---|
| Model | Nemotron-3-Nano-30B-A3B bf16 (Unsloth) |
| Method | LoRA bf16 + Cut Cross-Entropy + MoE weight tying + DDP |
| LoRA | r=32, alpha=32, dropout=0 |
| Targets | q/k/v/o_proj, up/down_proj, in/out_proj, lm_head |
| Optimizer | AdamW, betas=(0.9, 0.95), eps=1e-8, wd=0 |
| LR | 2e-4 → 0 linear decay |
| Effective batch | 32 |
| Per-rank batch | 16 |
| Micro batch | 4 |
| Accum steps per rank | 4 (DDP) / 8 (equivalent single-GPU) |
| Max seq len | 8192 |
| MoE tying | True (5,888 params) |
| World size | 2 (DDP, `static_graph=True`, `gradient_as_bucket_view=True`) |
| Rank split | **Contiguous** (rank 0 = [0:16], rank 1 = [16:32]) |
| Loss normalization | **Per-MB mean** + `/n_accum=4` before backward |
| Steps | 244 |

### Critical changes vs previous DDP run (2605151830)
1. **Per-MB normalization** (was: global per-token + ×WORLD_SIZE cancellation)
2. **Contiguous rank split** (was: strided `global[RANK::WORLD_SIZE]`)

### Soup-friendly checkpoint schedule (13 saves)
- Coarse every 30 steps: 30, 60, 90, 120, 150, 180, 210 (adapter-only, ~1.7 GB each)
- Dense final 10%: 220, 225, 229, 234, 239 (adapter-only)
- Final full checkpoint: 244 (with optimizer state, ~3 GB)
- No FIFO removal — all 13 survive on disk for post-training soup analysis

## Results

| Metric | Value | vs single-GPU 0.84 baseline |
|---|---|---|
| Train time | **125.8 min (2.10 hr)** | 236.8 min → **1.88× faster** |
| Final train loss | **0.0019** | 0.0023 (DDP slightly lower) |
| Peak VRAM per GPU | ~95 GiB (97,000 MiB) | 88.9 GiB (single-GPU) |
| Trainable params | 888,154,112 / 32,466,091,456 | same |
| Token verification | 5/5 MATCH | 5/5 MATCH |

### Loss tracking vs single-GPU 0.84

| Step | DDP per-MB | single-GPU 0.84 | Δ |
|---|---|---|---|
| 1 | 0.3859 | 0.385902 | ~0 |
| 2 | 0.4594 | 0.458534 | +0.001 |
| 28 | 0.0486 | 0.0390 | +0.010 |
| Final | 0.0019 | 0.0023 | −0.0004 |

Step 1 matches → initial state is bit-identical. Step 2 already differs by 0.001 → first gradient update was NOT bit-identical (floating-point order differs in DDP all-reduce vs single-GPU sequential accumulation).

## Soups built post-training

Built 13 adapter-only checkpoints → 3 soup submissions via streaming combiner (workaround for 2 GB cgroup memory limit on the AutoDL CPU instance).

| Soup | Recipe | Notes |
|---|---|---|
| `final-244` | checkpoint-244 alone | Same as the closeout `submission.zip` |
| `soup-last6` | uniform avg(220, 225, 229, 234, 239, 244) | Last-K soup, K=6 |
| `wise-30+120+244` | uniform avg(30, 120, 244) | WiSE-FT-style 3-point trajectory |

### Soup-build mechanics
- Cgroup memory limit was 2 GiB → naïve "load all 6 adapters + average in fp32" OOM'd
- Workaround: chunked subprocess (1500 keys per invocation) → 8 partial safetensors files + special-case for the 700 MB lm_head base_layer (raw byte copy, since it's frozen base-model weight identical across checkpoints) → streaming combiner that writes the final adapter from partial files without loading tensors into RAM
- All 3 soups: CRC OK, ~1.3 GB zipped each

## Kaggle submissions

| File | Ref | Score | Description |
|---|---|---|---|
| `submission.zip` (= final-244) | **52700850** | **0.78** | step-244 endpoint with per-MB loss normalization |
| `submission_soup-last6.zip` | **52701082** | pending | uniform avg(220-244) |
| `submission_wise-30+120+244.zip` | pending | pending | not yet downloaded |

## Outcome analysis

**Recovery from previous DDP score (0.73 → 0.78, +0.05)** confirms the per-MB normalization fix was directionally correct.

**Remaining gap to single-GPU baseline (0.78 vs 0.84, −0.06)** is unexplained. The user-confirmed Submission #6 (single-GPU + same data + same regex) scored 0.84, so the gap is from DDP itself, not data/regex/recipe changes.

Possible causes I considered but did NOT verify:
- Floating-point reduction-order differences (NCCL all-reduce vs single-GPU sequential)
- `static_graph=True` + `no_sync()` first-iteration interaction
- Unsloth's `use_gradient_checkpointing='unsloth'` behaving differently under DDP
- `gradient_as_bucket_view=True` interaction with manual `_tie_grads`

None of these were empirically validated. The reliable diagnostic would be tensor-level comparison between this DDP-trained adapter and a single-GPU adapter trained from the same initial state.

## Files

| File | Source | Size |
|---|---|---|
| `submission.zip` | closeout (= final-244) | 1.31 GB |
| `submission_soup-last6.zip` | post-training soup | 1.31 GB |
| `submission_wise-30+120+244.zip` | post-training soup (still on instance) | 1.31 GB |
| `checkpoint-244_loss0.0019_lr8.20e-07.zip` | closeout (still on instance) | 2.9 GB |
| `train_log.txt` | per-step + per-microbatch log | 451 KB |
| `full_log.txt` | stdout incl. Unsloth/swanlab/etc | 463 KB |
| `stress_log.txt` | stress test stdout | 15 KB |
| `stress_train_log.txt` | stress test train_log | 12 KB |
| `requirements.txt` | `pip freeze` (206 packages) | 5.6 KB |
| `train_huikang_ddp_perMBnorm.py` | training script | 33 KB |

Adapter post-processing in script:
- `base_model_name_or_path` → `metric/nemotron-3-nano-30b-a3b-bf16`
- `inference_mode` → `true`
- `lm_head` keys renamed `base_model.model.lm_head.*` → `base_model.model.backbone.lm_head.*`
