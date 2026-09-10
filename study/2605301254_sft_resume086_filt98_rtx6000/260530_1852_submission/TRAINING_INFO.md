# TRAINING_INFO — resume086_filt98 (epoch 2 of 2)

**Study folder:** `study/2605301254_sft_resume086_filt98_rtx6000/`
**Run model:** 2-epoch SFT, **paused after epoch 1** for a Kaggle submission, **continued for epoch 2** (true checkpoint resume — warm AdamW)
**Epoch 1 finished:** 2026-05-30 16:25 CDT  →  submission folder `260530_1625_submission/`
**Epoch 2 finished:** 2026-05-30 18:52 CDT  →  submission folder `260530_1852_submission/` *(this file)*
**Public LB:** *pending submission*

---

## Architectural framing

This was originally planned as one **2-epoch** training run (138 steps total) but was paused after step 69 to submit the epoch-1 checkpoint to Kaggle. Epoch 2 was launched as a **true checkpoint resume** (Option A) — same OUTPUT_DIR, same study folder, loaded `checkpoint-69_*/training_state.pt` with full optimizer state, and **the LR schedule formula `lr = 5e-5 × (1 - step/138)` was applied so step 69 → lr 2.5e-5, step 138 → lr 0**.

This is faithfully a single continuous run modulo one unavoidable wrinkle: epoch 1's actual LR schedule was `5e-5 × (1 - step/69)` (the trainer at the time didn't know about epoch 2), so the *historical* lr values during epoch 1 followed the steep 69-step decay (lr 5e-5 at step 0 → 0 at step 69) rather than the 138-step plan's gentler slope (which would have been lr 5e-5 at step 0 → 2.5e-5 at step 69). The parameter updates from epoch 1 already happened with the steeper schedule; we cannot retroactively change them. Epoch 2's LR schedule was correct (2.5e-5 → 0 over steps 69-138), giving a hybrid effective trajectory.

---

## Config (epoch 2)

| field | value |
|---|---|
| Base model | Nemotron-3-Nano-30B-A3B (bf16, hybrid Mamba2 + attention + 128-routed-MoE) |
| Resume source | `/root/autodl-tmp/adapter_output_resume086_filt98_notie_outproj/checkpoint-69_*/` (epoch-1 FIFO checkpoint with optimizer state) |
| Framework | Unsloth FastLanguageModel + manual training loop |
| LoRA arch | r = 32, α = 32, dropout = 0; targets q/k/v/o/up/down/in/`out_proj` + lm_head |
| MoE tying | **off** (`MOE_TIE=0`) |
| Mamba `out_proj` | **LIVE** (`USE_MEM_EFF=0` + `mixer.training=False`) |
| Loss | CCE |
| Optimizer | **AdamW with warm state from epoch 1** (loaded from `training_state.pt`), β = (0.9, 0.95), wd = 0 |
| **LR schedule** | **`5e-5 × (1 - step/138)` linear** — at step 69 starts at **2.5e-5**, decays to 0 by step 138 |
| Batch | 32 (micro_batch = 4, grad_accum = 8) |
| Sequence length | 8192 |
| Stratified shuffle seed | 42 |
| Env flags | `MOE_TIE=0 USE_MEM_EFF=0 TOTAL_STEPS=138` (NO `FRESH_START`, NO `ADAPTER_INIT_PATH` — true checkpoint resume) |
| Trainer code change for ep2 | Added env-var `TOTAL_STEPS_OVERRIDE` + batches list padded at the front with `pad_count = TOTAL_STEPS - len(batches)` placeholders so the loop's `range(_resume_step=69, 138)` accesses only the real ep2 batches at indices `[69..137]` |
| GPU | RTX PRO 6000 Blackwell (96 GB), single GPU |

---

## Data (epoch 2)

| field | value |
|---|---|
| CSV | `260530_huikang_NumericEq_5cat15_crg3x_ep2.csv` |
| md5 | `525ed18f4617cfe2f54797461f268180` |
| Unique rows | **1,715** → expanded **2,201** → 69 batches (padded to 138 via `TOTAL_STEPS`) |
| GT-match | 100 % |
| 5 "avoid-forget" cats | sampled at **15 %, oversampling=1**, **NON-OVERLAPPING with epoch-1's 15 %** — model sees a fresh 15 % of bit_man/cipher/grav/unit/numeral rows |
| 4 "new-trace" cats | identical to epoch 1 — `eq_d`/`eq_g`/`cr_d`/`cr_g` rows are the same (same questions seen a second time with their new-solver CoTs); `cr_g` at 3× oversample |

### Per-batch distribution (replayed)

| category | total | avg/batch | min | max | present |
|---|--:|--:|--:|--:|--:|
| equation_numeric_deduce | 676 | 9.8 | 9 | 11 | 100 % |
| cryptarithm_deduce | 486 | 7.0 | 6 | 8 | 100 % |
| cipher (15 %) | 236 | 3.4 | 3 | 4 | 100 % |
| bit_manipulation (15 %) | 203 | 2.9 | 2 | 4 | 100 % |
| unit_conversion (15 %) | 148 | 2.1 | 2 | 3 | 100 % |
| gravity (15 %) | 146 | 2.1 | 1 | 3 | 100 % |
| equation_numeric_guess (exempt) | 142 | 2.1 | 1 | 3 | 100 % |
| numeral (15 %) | 98 | 1.4 | 1 | 2 | 100 % |
| cryptarithm_guess (3×) | 66 | 1.0 | 0 | 2 | 94 % |

---

## Results (epoch 2 = steps 69-138)

| metric | epoch 1 (steps 0-69) | epoch 2 (steps 69-138) | trajectory |
|---|---|---|---|
| Final loss | 0.0716 (step 69) | **0.0538 (step 138)** | epoch 2 continued the descent |
| Sampled losses | 0.672 → 0.385 → 0.231 → 0.100 → 0.072 | 0.083 (step 70) → 0.069 (97) → 0.064 (103) → 0.057 (110) → 0.054 (138) | flat, near-converged |
| Time | 3.93 hrs (235.9 min) — *wait, that was alleqgt*… epoch 1 was 1.07 hrs | **1.02 hrs (61.3 min)** | |
| **Total run time** | | **2.09 hrs combined** | |
| LR at start | 5.0e-5 | **2.50e-5** (= 5e-5 × (1 − 69/138)) | matches 138-step formula |
| LR at end | 7.25e-07 (≈0 at 69/69) | 3.62e-07 (≈0 at 138/138) | |
| Peak VRAM | 71,895 MB | 71,887 MB | stable |
| Grad spikes | none | none | clean |
| Soup save steps | `[14, 41, 55, 59, 62, 66, 69]` (20-100 % of 69) | `[83, 110, 117, 124, 131, 138]` (60-100 % of 138; step 28's 20 % save was past `_resume_step=69`) | |

The loss dropping from 0.0716 → 0.0538 across epoch 2 indicates the model continued to learn from the **fresh avoid-forget sample** (different 15 %) while consolidating on the **repeated eq/crypt rows** (second pass with warm optimizer).

---

## Adapter-init verification (the load-bearing checks)

| check | result |
|---|---|
| `TOTAL_STEPS override: padded 69 placeholders at front, num_steps=138` | ✓ multi-epoch surgery worked |
| `Adapter-only soup save steps: [28, 83, 110, 117, 124, 131, 138]` | ✓ schedule recomputed for 138 steps |
| **`Resuming from checkpoint: checkpoint-69_loss0.0716_lr7.25e-07 (step 69)`** | ✓ true resume from epoch-1 ckpt dir |
| `Loaded adapter weights: 12011 tensors (missing=6242, unexpected=0)` | ✓ |
| **`Loaded optimizer state from training_state.pt`** | ✓ **warm AdamW from epoch 1** |
| `Forced unfused else-branch on 23 Mamba mixers` | ✓ live `out_proj` |
| First step (70): `lr=2.50e-05` | ✓ matches `5e-5 × (1 − 69/138) = 2.50e-5` |

---

## Output artifacts on instance (21578, `/root/autodl-tmp/`)

Grouped on instance at `260530_1852_submission/` per routine §3:

| artifact | path / size |
|---|---|
| Submission zip | `260530_1852_submission/submission_resume086_filt98_ep2_notie_outproj.zip` (3.83 GB; renamed post-finalize to disambiguate from epoch-1 zip) |
| Checkpoint zip | `260530_1852_submission/checkpoint-138_loss0.0538_lr3.62e-07.zip` (9.93 GB) |
| 6 epoch-2 soup adapters | `adapter_output_.../adapter-{83,110,117,124,131,138}_loss{x}_lr{x}/` |
| 1 FIFO checkpoint | `adapter_output_.../checkpoint-138_*/` (with optimizer state for a future epoch 3 if desired) |
| Final adapter at OUTPUT_DIR root | `adapter_output_.../adapter_model.safetensors` (4.26 GB; the 138-step end-of-run weights, with lm_head keys renamed to `backbone.lm_head.`) |

Epoch-1 artifacts preserved at `260530_1625_submission/` (3.83 GB + 10.4 GB zips). The orphan epoch-1 `checkpoint-69_*` dir (resume source) was deleted after epoch 2 launched — the zip backup remains in `260530_1625_submission/checkpoint-69_*.zip`.

---

## Local files in this folder (`260530_1852_submission/`)

| file | purpose |
|---|---|
| `train_resume_086_ep2.py` | the epoch-2 trainer with TOTAL_STEPS override + batches padding |
| `test_imports.py` | pre-flight import check |
| `train_log.txt` | full epoch-2 training log pulled from instance |
| `requirements.txt` | `pip freeze` from instance |
| `TRAINING_INFO.md` | this file |

The companion `260530_1625_submission/` (epoch-1) folder is a sibling under the same study folder.

---

## Cross-references

- `02_train/TRAINING_ROUTINE.md` — Step 3/4 spec
- `02_train/260522_moe_expert_rank1_bug_report.md` — `out_proj`-live mechanism
- `study/SCORE_TRACKER.md` — Sub #16 (0.86 baseline), Sub #25 (epoch-1 LB pending)
- `study/2605301254_sft_resume086_filt98_rtx6000/260530_1625_submission/TRAINING_INFO.md` — epoch-1 details
- `01_data/260530_resume_data/260530_huikang_NumericEq_5cat15_crg3x_ep2.csv` — the epoch-2 CSV

---

## Notes

- **True 2-epoch run**: same study folder, same OUTPUT_DIR, same instance, full optimizer state continuity. Implemented via `TOTAL_STEPS=138` env override + 69 placeholder pads in the batches list so `range(_resume_step=69, 138)` indexes the real epoch-2 batches at positions 69-137.
- **Routine §3 → §4 order followed** properly (small files saved BEFORE shutdown).
- **Trainer minor bug noted, not fixed in this run:** the resume block doesn't add the loaded checkpoint dir to `_ckpt_queue`, so the original `checkpoint-69_*` stays on disk as an orphan during epoch 2 (until the orchestrator manually deletes it during Step 3 finalize). FIFO=1 only manages new checkpoints created during the resumed run.
- **Trainer minor bug noted, not fixed:** eta display divides elapsed-this-run by global step count instead of new-steps-this-run, so eta appeared 5-10× shorter than reality during epoch 2.
