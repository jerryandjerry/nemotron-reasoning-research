# TRAINING_INFO — alleqgt (Submission #23)

**Study folder:** `study/2605270202_sft_newdata_alleqgt_outproj_rtx6000/`
**Training finished:** 2026-05-27 06:20:38 CDT (instance: 2026-05-27 19:20 CST+0800)
**Public LB:** **0.84**  •  **Val OVERALL (950):** 0.8632

---

## What was trained

A retrain of Sub #16 (the 0.86 recipe) with the **numeric-equation CoTs swapped to the new-solver output AND filtered to GT-matching rows only**. Non-equation huikang-golden rows untouched.

The motivating question: does dropping the 114 GT-mismatching new-solver eq CoTs (vs #22 allEq's "keep all 732") help vs the original equation CoTs? Answer: **no — held at 0.84**, same as #22 allEq.

---

## Config

| field | value |
|---|---|
| Base model | Nemotron-3-Nano-30B-A3B (bf16, hybrid Mamba2 + attention + 128-routed-MoE) |
| Framework | Unsloth FastLanguageModel + manual training loop |
| LoRA arch | r = 32, α = 32, dropout = 0; targets q/k/v/o/up/down/in/`out_proj` + lm_head |
| MoE tying | **off** (`MOE_TIE=0`) — independent rank-32 LoRA per expert |
| Mamba `out_proj` | **LIVE** — `is_fast_path_available=True` + `mixer.training=False` per mixer → unfused else-branch lets `out_proj` LoRA receive gradients |
| lm_head | manual LoRA + post-train key rename `base_model.model.lm_head.` → `base_model.model.backbone.lm_head.` for Kaggle compatibility |
| Loss | CCE (cut_cross_entropy) |
| Optimizer | AdamW, β = (0.9, 0.95), wd = 0 |
| LR schedule | linear 2e-4 → 0 over 247 steps |
| Batch | 32 (micro_batch = 4, grad_accum = 8) |
| Sequence length | 8192 |
| Epochs | 1 (stratified shuffle, seed 42) |
| Env flags | `MOE_TIE=0 USE_MEM_EFF=0 FRESH_START=1` |
| Transformers | 4.56.2 (see `requirements.txt`) |
| GPU | RTX PRO 6000 Blackwell (96 GB), single GPU |

---

## Data

| field | value |
|---|---|
| CSV | `260527_huikang_NumericEq_allEq_gtTrue.csv` |
| Source columns | `pretok_0408_decoded` (huikang originals) + `260527_new_solver` (numeric-eq) |
| md5 | `9628b30d04e4747992a4c6a8c7abf450` |
| Code diff vs gt2x trainer | `CSV_PATH` + `_DATA_TAG` only |
| **Expanded total → num_steps** | **7,883 → 247 steps** (log-confirmed: `Steps: 247`) |
| Equation composition | 732 new-solver numeric-eq → **gt-True only** (114 GT-mismatch dropped → 618 kept) |
| Cryptarithm composition | original (unchanged from #16: 735 lkevincc + 65 huikang) |

---

## Results

| metric | value |
|---|---|
| Final loss (step 247) | **0.0029** |
| Total time | **3.93 hrs** (235.9 min) |
| Peak VRAM | 72,095 MB |
| Grad spikes | none (max grad_norm well under 1.0 throughout) |
| Soup save steps | `[49, 148, 198, 210, 222, 235, 247]` (20/60/80/85/90/95/100 % of 247) |
| **Public LB** | **0.84** (endpoint, manually submitted) |

### huikang val 4-way comparison (950 rows, per-token `(token, prob)` + live `rfind` extractor; identical eval across all 4)

| huikang_category | N | moe (#16) | alleq (#22 allEq) | **alleqgt (#23)** | cryptnumeq (#24) |
|---|---:|---:|---:|---:|---:|
| equation_numeric_deduce | 65 | **0.9231** | 0.8923 | 0.8923 | 0.8923 |
| equation_numeric_guess | 19 | 0.0526 | 0.2632 | **0.3158** | **0.3158** |
| cryptarithm_deduce | 55 | **0.0364** | 0.0182 | 0.0182 | 0.0000 |
| cryptarithm_guess | 16 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bit_manipulation | 160 | 0.7688 | 0.7562 | 0.7688 | **0.7812** |
| cipher | 158 | **0.9937** | **0.9937** | 0.9873 | 0.9873 |
| gravity | 160 | 0.9938 | **1.0000** | 0.9938 | **1.0000** |
| numeral | 158 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| unit_conversion | 159 | **1.0000** | 0.9937 | **1.0000** | **1.0000** |
| **OVERALL (val)** | **950** | 0.8621 | 0.8611 | **0.8632** | 0.8653 |
| **Public LB** | | **0.86** | 0.84 | **0.84** | 0.85 |

- `equation_numeric_guess` improved to **0.3158** (best of the 4-way; +26pp vs #16's 0.0526) — the new-solver eq CoTs help this small category specifically.
- `equation_numeric_deduce` held at 0.8923 (≡ #22, −3.1pp vs #16) — gt-True filtering did not recover the −0.02 LB gap vs #16.
- The −0.02 LB gap vs #16 traces to the **equation-CoT *style***, not mismatch filtering — consistent with #22's conclusion.

---

## Output artifacts on instance (21578, `/root/autodl-tmp/`)

| artifact | path / size |
|---|---|
| Submission zip (already downloaded to this folder) | `submission_newdata_alleqgt_notie_outproj.zip` (3.83 GB; `adapter_model.safetensors` + `adapter_config.json` only) |
| Final checkpoint zip | `checkpoint-247_loss0.0029_lr8.10e-07.zip` (10.4 GB; adapter + optimizer + scheduler + RNG + tokenizer + training_args) |
| Soup adapters | `adapter_output_newdata_alleqgt_notie_outproj/adapter-{step}_loss{x}_lr{x}/` for step ∈ {49, 148, 198, 210, 222, 235, 247} — 7 saves × 4.26 GB safetensors each |

Both zips grouped on the instance under `/root/autodl-tmp/260527_0620_submission/`.

---

## Local files in this folder

| file | purpose |
|---|---|
| `train_alleqgt_outproj.py` | the training script (copy of the run-folder version) |
| `train_log.txt` | full training log pulled from instance 2026-05-30 |
| `requirements.txt` | `pip freeze` from instance 2026-05-30 |
| `submission_newdata_alleqgt_notie_outproj.zip` | the Kaggle submission zip (already downloaded by user) |
| `TRAINING_INFO.md` | this file |

---

## Cross-references

- `study/SCORE_TRACKER.md` Submission #23 — public LB result + 4-way val table
- `study/2605270202_sft_newdata_alleqgt_outproj_rtx6000/train_alleqgt_outproj.py` — trainer source
- `02_train/TRAINING_ROUTINE.md` Step 3 — output-saving spec this folder follows
- `02_train/260522_moe_expert_rank1_bug_report.md` — mechanism for the `out_proj`-live trick this run uses
- Sub #16 / #22 / #24 SCORE_TRACKER entries — earlier and parallel runs in the same data-swap experiment family

---

## Step 3 / Step 4 note (housekeeping)

Per `02_train/TRAINING_ROUTINE.md` Step 3 → Step 4 ordering, the small-file save (this folder) should have happened **before** the instance shutdown on 2026-05-27. It didn't — the shutdown landed first. Small files were retrieved on 2026-05-30 after the instance was rebooted. Cross-reference: `02_train/TRAINING_ROUTINE.md` Step 4 explicitly says *"After saving all small files"*.
