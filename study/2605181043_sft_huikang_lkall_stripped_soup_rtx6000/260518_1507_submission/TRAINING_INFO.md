# Training Info — Run #7 Retrain with LoRA Soup

> Trained: 2026-05-18 10:43 → 14:52 Chicago time (instance May 18 23:43 → May 19 03:52 UTC+8)
> Post-processing (zip + soup build): finished by 04:15 China = 15:15 Chicago
> Instance: AutoDL RTX PRO 6000 Blackwell (95 GB), port 21578
> SwanLab run: https://swanlab.cn/@jerry4083/260410_Nemotron/runs/8dwo1khnuidj9fw9hgl3s

## What this run tests

**Re-train of run #7's 0.84-scoring setup with two additions:**
1. **Adapter-only soup saves** per [TRAINING_ROUTINE.md §2.4B](../../../02_train/TRAINING_ROUTINE.md): every 50 steps + 80/85/90/95/100% of `num_steps`, all kept on disk.
2. **Resume-from-latest-checkpoint** with PEFT `.default.` key-rename fix (per routine required code feature).

Hypothesis: can checkpoint averaging (soup) push above the 0.84 plateau hit by single-checkpoint training? Two recipes tested:
- `soup-last5` = uniform avg of {200, 209, 221, 234, 246} — captures the late LR-decay tail
- `wise-50-150-246` = uniform avg of {50, 150, 246} — early + middle + final WiSE-FT style

Data + base config byte-identical to run #7 (`2605152212_sft_huikang_lkall_stripped_kaggleregex_rtx6000`).

## Config

| Parameter | Value |
|---|---|
| Model | Nemotron-3-Nano-30B-A3B bf16 (Unsloth FastLanguageModel) |
| Method | LoRA bf16 + Cut Cross-Entropy + MoE weight tying |
| LoRA | r=32, α=32, dropout=0 |
| Targets | q/k/v/o_proj, up/down_proj, in/out_proj, lm_head (9 + manual lm_head) |
| Optimizer | AdamW betas=(0.9, 0.95), eps=1e-8, wd=0 |
| LR | 2e-4 → 0 linear decay |
| Batch | 32 (micro=4, ga=8) |
| Max seq len | 8192 |
| MoE tying | True (5,888 tied params) |
| LoRA cast | fp32 |
| Ordering | Stratified interleave (seed=42), categories proportional per batch |
| Steps | 246 |
| FIFO (full ckpt) | 1 (per routine §2.4A) |
| Adapter-only saves | [50, 100, 150, 197, 200, 209, 221, 234, 246] (no FIFO) |
| Boxed extractor | Kaggle metric (`\\boxed{` + `rfind('}')`) |

## Data

`260515_huikang_lkall_stripped.csv` — 6,906 unique rows → **7,849 expanded** via `oversampling` column. Identical to run #7.

Source breakdown:
- `pretok_0408_decoded` huikang: 6,106 rows (non-cryptarithm only)
- `lkevincc_golden`: 800 rows (all cryptarithm — 65 swapped-in + 735 originally added)

Category breakdown (expanded):

| Category | n |
|---|---|
| bit_manipulation | 1,754 |
| cipher | 1,656 |
| unit_conversion | 1,070 |
| gravity | 1,055 |
| numeral | 730 |
| equation_numeric_deduce | 658 |
| cryptarithm_deduce | 639 |
| cryptarithm_guess | 161 |
| equation_numeric_guess | 126 |
| **Total** | **7,849** |

## Results

- **Train time:** 249.2 min (4.15 hrs), ~1.01 min/step average
- **Final loss:** 0.0034 (step 246) vs run #7's 0.0029 — within noise band
- **Loss curve:** 0.40 (step 2) → 0.06 (step 24) → 0.006 (step 50) → ~0.003-0.005 plateau through step 246
- **Peak VRAM:** ~88.7 GB / 95 GB
- **Trainable params:** 888,154,112 / 32,466,091,456

## Checkpoints saved

**Full state (FIFO=1):** only `checkpoint-246` survives on disk at end of training. Final zip: `checkpoint-246_loss0.0034_lr8.13e-07.zip` (2.7 GB).

**Adapter-only (all kept):** 9 dirs, each ~4 GB raw on disk:
```
adapter-50_loss0.0059_lr1.60e-04
adapter-100_loss0.0047_lr1.20e-04
adapter-150_loss0.0041_lr7.89e-05
adapter-197_loss0.0039_lr4.07e-05
adapter-200_loss0.0039_lr3.82e-05
adapter-209_loss0.0037_lr3.09e-05
adapter-221_loss0.0034_lr2.11e-05
adapter-234_loss0.0041_lr1.06e-05
adapter-246_loss0.0034_lr8.13e-07
```

## Outputs

In `260518_1507_submission/`:
- `submission.zip` (1.3 GB) — final-246 adapter, CRC OK
- `submission_soup-last5.zip` (1.3 GB) — uniform avg of {200, 209, 221, 234, 246}, CRC OK
- `submission_wise-50-150-246.zip` (1.3 GB) — uniform avg of {50, 150, 246}, CRC OK
- `checkpoint-246_loss0.0034_lr8.13e-07.zip` (2.7 GB) — full final state for resume
- `train_huikang_lkall_stripped_soup.py` — training script
- `build_soups.py` — soup-build script
- `train_log.txt` — full training log (~1.6 MB)
- `soups_log.txt` — soup build log
- `requirements.txt` — pip freeze
- `TRAINING_INFO.md` — this file

**Adapter post-processing applied** (in training script for final + in build_soups.py for soups):
- `base_model_name_or_path` → `metric/nemotron-3-nano-30b-a3b-bf16`
- `inference_mode` → True
- lm_head keys renamed: `base_model.model.lm_head.*` → `base_model.model.backbone.lm_head.*` ✓

## Scores

Pending Kaggle submission.

| Recipe | Score | Hypothesis |
|---|---|---|
| `final-246` (control) | TBD | Should reproduce run #7's 0.84 — bf16 inference is deterministic and config is identical |
| `soup-last5` | TBD | Tests: does averaging the converged tail dampen noise enough to lift score? |
| `wise-50-150-246` | TBD | Tests: does early+mid+final WiSE-FT generalize better than the final-only converged model? |

## Comparison vs prior runs

| Run | Setup | Score |
|---|---|---|
| #7 (`2605152212_sft_huikang_lkall_stripped_kaggleregex_rtx6000`) | Same data + config, no soup saves | 0.84 |
| **#10 final-246 (this run)** | Same data + config, FIFO=1 + soup saves added | TBD |
| **#10 soup-last5** | Uniform avg of {200,209,221,234,246} | TBD |
| **#10 wise-50-150-246** | Uniform avg of {50,150,246} | TBD |
