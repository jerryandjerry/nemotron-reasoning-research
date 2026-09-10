# Training Info — huikang DDP 2-GPU + HF non-reentrant ckpt (fix for 0.78 score)

> **Trained:** 2026-05-16 (Chicago) — closeout finished ~21:57 CDT
> **Instance:** AutoDL RTX PRO 6000 Blackwell ×2 (95 GiB each), port 49666
> **SwanLab:** `260516_xxxx_sft_huikang_ddp2gpu_nonreent_r32_a32_lr2e4_seq8192_bs32_rtx6000`
> **Run ID:** https://swanlab.cn/@jerry4083/260410_Nemotron/runs/iz9my2lgwhi8tv6z536yt

## Goal

Close the 0.78 → 0.84 score gap of the previous DDP perMBnorm run (`2605152045_sft_huikang_ddp_perMBnorm_rtx6000`). The previous run had:
- Step 1 loss = 0.385902 (matches Sub#4 exactly)
- Step 1 grad_norm = **1.4142** (Sub#4 = 1.2356 — 14.5% off, far too big for FP noise)

That ~14% gradient error structurally compounded over 244 steps. Hypothesis (after S1 ladder step failed): Unsloth's reentrant gradient checkpointing produces wrong gradients under DDP even with `static_graph=True`. Fix: switch to HF non-reentrant.

## Data

`/root/autodl-tmp/data/260514_huikang_update/huikang_7830.csv` (34 MB)
- 6,171 unique base pids + `oversampling` column summing to 7,830
- Ordering: huikang's `index.jsonl` (epoch 0), suffix-strip lookup
- Token verification: 5/5 PASSED

## Config — IDENTICAL to 0.78 run except 4 DDP/ckpt changes

| | **0.78 run (perMBnorm)** | **This run (nonreent)** |
|---|---|---|
| Gradient checkpointing | `use_gradient_checkpointing='unsloth'` (reentrant) | HF non-reentrant via `gradient_checkpointing_enable(use_reentrant=False)` |
| DDP `static_graph` | `True` (needed by reentrant) | dropped |
| DDP `find_unused_parameters` | not set (default False) | `True` (MoE-safe) |
| DDP `no_sync()` | used on MBs 0-2 | dropped — every MB all-reduces |

Everything else identical: model, data, batch 32 global / 16 per-rank / micro 4, LR 2e-4→0 linear, AdamW(0.9, 0.95, eps=1e-8, wd=0), max_seq_len 8192, MoE tying logic, per-MB loss normalization, contiguous rank split, lm_head LoRA + CCE forward patch, adapter post-processing.

### Full config table

| Parameter | Value |
|---|---|
| Model | Nemotron-3-Nano-30B-A3B bf16 (Unsloth) |
| Method | LoRA bf16 + CCE + MoE weight tying + DDP (HF non-reentrant ckpt) |
| LoRA | r=32, alpha=32, dropout=0 |
| Targets | q/k/v/o_proj, up/down_proj, in/out_proj, lm_head |
| Optimizer | AdamW, betas=(0.9, 0.95), eps=1e-8, wd=0 |
| LR | 2e-4 → 0 linear decay |
| Effective batch | 32 |
| Per-rank batch | 16 |
| Micro batch | 4 |
| Accum steps per rank | 4 |
| Max seq len | 8192 |
| MoE tying | True (5,888 params reported; actually broken per earlier diff — same as Sub#4) |
| World size | 2 (DDP, `gradient_as_bucket_view=True`, `broadcast_buffers=False`, `find_unused_parameters=True`, NO `static_graph`) |
| Rank split | Contiguous (rank 0 = global[0:16], rank 1 = global[16:32]) |
| Loss normalization | Per-MB mean + `/n_accum=4` before backward |
| Gradient checkpointing | HF non-reentrant (`use_reentrant=False`) |
| no_sync | OFF — every MB triggers NCCL all-reduce |
| Steps | 244 |

### Checkpoint schedule (FIFO=1)

Save full state every 50 steps + at final step. Keep only the latest on disk. No intermediate adapter-only saves (no soup-building this run — focus was validating the fix).

## Results

| Metric | Sub#4 (single-GPU 0.84) | Prev DDP (0.78) | **This run** |
|---|---|---|---|
| Step 1 loss | 0.385902 | 0.3859 | **0.385902** ✓ |
| Step 1 grad_norm | **1.2356** | 1.4142 (+14.5%) | **1.2354** ✓ FP-noise match |
| Step 50 ckpt loss | 0.008426 | 0.0108 (+28%) | **0.0081** (-4%) |
| Step 100 ckpt loss | 0.002814 | 0.0034 (+21%) | **0.0028** (0%) |
| Step 150 ckpt loss | — | — | 0.0022 |
| Step 200 ckpt loss | 0.002159 | 0.0025 (+16%) | **0.0022** (+1.9%) |
| Step 244 final loss | 0.002092 | 0.0019 (-9%) | **0.0021** (+0.4%) |
| Train time | 237 min | **126 min (1.88×)** | **159 min (1.49×)** |
| Peak VRAM per GPU | 88.9 GiB | ~95 GiB | ~96 GiB |

### Per-group grad-L2 at step 1 (diagnostic logging)

```
step=1 PRE-TIE total=0.1457 | attention=0.0749 moe_up=0.0043 moe_down=0.0274 mixer_ssm=0.0765 lm_head=0.0747 other=0.0585
step=1 POST-TIE total=1.2354 | attention=0.0749 moe_up=0.0043 moe_down=1.2271 mixer_ssm=0.0765 lm_head=0.0747 other=0.0585
```

POST-TIE total = **1.2354** vs Sub#4's 1.2356 — within 0.016%. Gradient pipeline is now mathematically equivalent to single-GPU at step 1.

`_tie_grads` amplifies moe_down by ~45× (broken-tying bug present in BOTH runs — out of scope for this fix).

## Diagnosis ladder

| Attempt | Change | Step-1 grad_norm | Result |
|---|---|---|---|
| **Baseline (0.78)** | reentrant ckpt + static_graph + no_sync + bucket_view | 1.4142 | ✗ structural bug |
| **S1** | drop `no_sync()` only | 1.4145 | ✗ no change — `no_sync` wasn't the cause |
| **S3-lite** | drop reentrant ckpt + drop static_graph + add find_unused + drop no_sync | **1.2354** | ✓ matches Sub#4 |

The colleague's `no_sync()`-only hypothesis was disproven by S1. Reentrant Unsloth ckpt under DDP is the actual culprit, regardless of `static_graph=True` (which was supposed to suppress double-fire).

## Timing breakdown

- Stress test (3 steps): 2.4 min
- Full training (244 steps): 159.4 min
- Closeout (adapter save + zip): ~5 min
- Total wall: ~167 min from launch to submission.zip ready

## Files

| File | Source | Size |
|---|---|---|
| `submission.zip` | closeout | 1.31 GB (still on instance) |
| `checkpoint-244_loss0.0021_lr8.20e-07.zip` | closeout | 2.91 GB (still on instance) |
| `train_log.txt` | per-step + per-microbatch log | 451 KB |
| `full_log.txt` | stdout incl. Unsloth/swanlab/etc | 462 KB |
| `stress_train_log.txt` | stress test train_log | 7 KB |
| `stress_full_log.txt` | stress test stdout | 15 KB |
| `requirements.txt` | `pip freeze` (206 packages) | 5.6 KB |
| `train_huikang_ddp_nonreent.py` | training script | 29 KB |

Adapter post-processing in script:
- `base_model_name_or_path` → `metric/nemotron-3-nano-30b-a3b-bf16`
- `inference_mode` → `true`
- `lm_head` keys renamed `base_model.model.lm_head.*` → `base_model.model.backbone.lm_head.*`

## Kaggle result

**Submission #8: ref 52727258, score 0.82.**
- vs Sub#4 single-GPU baseline (0.84): **-0.02**
- vs previous DDP perMBnorm (0.78): **+0.04**

The fix recovered 67% of the gap (0.04/0.06). To attribute the remaining 0.02, I diffed the three adapters tensor-by-tensor (11,986 live LoRA tensors):

### Adapter L2 distance from Sub#4 (the 0.84 reference)

| | LoRA-only L2(Δ)/L2(ref) | mean per-tensor cosine |
|---|---:|---:|
| Old broken DDP (0.78) | 20.9% | 0.896 |
| **This DDP S3-lite (0.82)** | **9.9%** | **0.973** |

The fix cut the weight-space distance by **2.1×**. Every parameter group improved — none regressed. Largest improvements were exactly in the groups where the old run was worst (MoE up_proj 2.3×, lm_head 3×, gradient-checkpointing-sensitive paths).

### Is the residual gap "DDP-systematic" or noise?

For every LoRA tensor, computed `cos((new_adapter − ref), (old_adapter − ref))`:
- If both DDP runs drift in the **same direction** from Sub#4 → systematic DDP bias (cosine → 1)
- If they drift in **random directions** → noise (cosine → 0)

| group | mean cos(new_drift, old_drift) |
|---|---:|
| attention A/B | 0.26 / 0.29 |
| ssm A/B | 0.28 / 0.29 |
| moe_up A/B | 0.25 / 0.28 |
| moe_down A/B | 0.23 / 0.25 |
| lm_head A/B | 0.18 / 0.16 |
| **global** | **0.25** |

Random orthogonal drift = 0.0. Identical-direction drift = 1.0. **Observed 0.25 → mostly noise, with a small (~25%) DDP-systematic component concentrated in MoE up_proj.**

### Attribution of the 0.02 gap (corrected)

The cos=0.25 between (new_drift) and (old_drift) does **not** mean "75% noise." It means the two DDP runs drift via *different* DDP mechanisms. Both drifts are still DDP-attributable.

**The full 0.02 is DDP-attributable.** Evidence:
- Subs #4 (textcsv_0408), #6 (golden_stripped + cryptarithm), #7 (lkall_stripped) — three different single-GPU runs across different data, **all score exactly 0.84**.
- Single-GPU score variance ≈ 0 at Kaggle resolution → any single-GPU baseline would land at 0.84.
- → All 0.02 of the gap is DDP-introduced; the question is which DDP mechanism.

### Where the remaining 0.02 lives

- 32% of the residual L2 distance from Sub#4 sits in **MoE up_proj LoRA** (cos 0.938).
- This is orthogonal to the old broken DDP's drift signature (cos 0.16-0.28 across groups), so it's not the same reentrant-ckpt bug — it's a different DDP/MoE interaction.

### Plausible mechanisms (untested)

1. **Gradient-bucket reduction over sparsely-routed MoE experts** — `find_unused_parameters=True` handles unused experts, but the *partial use* across ranks (different sample subsets activate different experts) may produce small bias in per-expert grads after all-reduce.
2. **`gradient_as_bucket_view=True` + in-place `_tie_grads` writes** — modifying gradients in-place on the DDP bucket between all-reduce and optimizer.step could affect subsequent iterations subtly. Was kept in S3-lite; haven't tested dropping it.
3. **bf16 NCCL all-reduce accumulating differently than single-GPU sequential** — but at 1e-4 magnitude, shouldn't compound to 0.02 over 244 steps. Probably not the dominant cause.

### Next debugging step if speed-vs-score tradeoff matters

Disable `gradient_as_bucket_view=True` (1-line change, possibly some memory cost) and rerun. If score recovers to 0.84, hypothesis #2 confirmed. If not, escalate to per-expert grad logging at step 1 to localize the remaining drift.

### Script audit — nothing else hiding

Compared `train_huikang_textcsv_0408.py` (Sub#4) vs `train_huikang_ddp_nonreent.py` (this) line by line. Optimizer (AdamW betas, eps, wd), LR schedule, RNG seed (`random_state=42`), PEFT init, `_tie_param_init`, `_tie_grads`, CCE patch, clip_grad_norm — **all identical**. The 4 DDP/ckpt changes are the only intentional deltas. No accidental drift from missed config.

## What's still unknown

1. **Why Unsloth reentrant fails under DDP even with static_graph** — root cause inside Unsloth's CUDA kernels not investigated. Empirical fix verified; mechanism not.
2. **Whether `no_sync()` can be re-added** for ~10-15 min speed recovery — would need a follow-up stress test to verify it doesn't reintroduce a different bug.
3. **Whether the remaining 0.02 is truly irreducible or hides another DDP bug** — would need another DDP run with a different seed to estimate run-to-run variance.

## Follow-ups

- If score ≥ 0.84: optionally rerun with `no_sync()` re-enabled to recover speed (target ~140 min instead of 159 min).
- Eventually: file an Unsloth issue if root cause matters.
- The broken `_tie_grads()` bug remains in both runs — separate cleanup, not affecting score gap.
