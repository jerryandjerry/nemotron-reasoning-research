# TRAINING_INFO — cryptnumeq (Submission #24)

**Study folder:** `study/2605270226_sft_newdata_cryptnumeq_outproj_rtx6000/`
**Training finished:** 2026-05-27 06:54:51 CDT (instance: 2026-05-27 19:54 CST+0800)
**Public LB:** **0.85**  •  **Val OVERALL (950):** 0.8653

---

## What was trained

A retrain of Sub #16 (the 0.86 recipe) with the **cryptarithm and numeric-equation CoTs swapped to our new-solver outputs, combined in one CSV**. Non-{crypt, eq} huikang-golden rows untouched.

The motivating question: does combining the two new-solver swaps into one training reach the SFT ceiling of 0.86? Answer: **0.85** — best of the new-solver runs but still −0.01 below #16.

---

## Config

| field | value |
|---|---|
| Base model | Nemotron-3-Nano-30B-A3B (bf16, hybrid Mamba2 + attention + 128-routed-MoE) |
| Framework | Unsloth FastLanguageModel + manual training loop |
| LoRA arch | r = 32, α = 32, dropout = 0; targets q/k/v/o/up/down/in/`out_proj` + lm_head |
| MoE tying | **off** (`MOE_TIE=0`) — independent rank-32 LoRA per expert (NOT shared rank-1) |
| Mamba `out_proj` | **LIVE** — `is_fast_path_available=True` + `mixer.training=False` per mixer → unfused else-branch lets `out_proj` LoRA receive gradients |
| lm_head | manual LoRA + post-train key rename `base_model.model.lm_head.` → `base_model.model.backbone.lm_head.` for Kaggle compatibility |
| Loss | CCE (cut_cross_entropy) — no logits materialization |
| Optimizer | AdamW, β = (0.9, 0.95), wd = 0 |
| LR schedule | linear 2e-4 → 0 over 248 steps |
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
| CSV | `260527_huikang_NumericEq.csv` (from `01_data/260526_Cryptarithm_TrainData/`) |
| Source columns | `pretok_0408_decoded` (huikang originals) + `260527_new_solver` (eq) + `260526_new solver` (crypt) |
| Code diff vs gt2x trainer | `CSV_PATH` + `_DATA_TAG` + **`csv.field_size_limit(2**31-1)`** (this data has 0×-dropped runaway crypt CoTs up to 6.8M chars; `DictReader` must parse every row before the 0× drop) |
| Unique rows | 7,077 (in the version trained; reduced to 6,429 in the 2026-05-30 DPO preprocessing) |
| 0× oversample (dropped) | 648 |
| **Expanded total → num_steps** | **~7,917 → 248 steps** (log-confirmed: `Steps: 248`) |
| Crypt composition | 800 cryptarithm new-solver, 152 GT-mismatch kept at 2× oversample + remainder at 1× |
| Eq composition | 732 numeric-equation new-solver, **gt-True only** (mismatches dropped) |

---

## Results

| metric | value |
|---|---|
| Final loss (step 248) | **0.009** |
| Total time | **4.02 hrs** (241.4 min) |
| Peak VRAM | 71,939 MB |
| Grad spikes | none (max grad_norm well under 1.0 throughout) |
| Checkpoints saved | every `save_steps` (FIFO=1, only the latest kept) + 7 soup-only adapter saves |
| Soup save steps | `[50, 149, 198, 211, 223, 236, 248]` (20/60/80/85/90/95/100 % of 248) |
| **Public LB** | **0.85** (endpoint, manually submitted) |

### huikang val 4-way comparison (950 rows, per-token `(token, prob)` + live `rfind` extractor; identical eval across all 4)

| huikang_category | N | moe (#16) | alleq (#22 allEq) | alleqgt (#23) | **cryptnumeq (#24)** |
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
| **OVERALL (val)** | **950** | 0.8621 | 0.8611 | 0.8632 | **0.8653** |
| **Public LB** | | **0.86** | 0.84 | 0.84 | **0.85** |

- Highest val OVERALL of the 4-way (**0.8653**) and best `bit_manipulation` (**0.7812**).
- `cryptarithm_deduce` regression to **0.0000** vs #16's 0.0364 — combining the new-solver swaps did not recover the original 65-huikang crypt CoT signal lost in #18 / #20.
- +0.01 LB vs #23 alleqgt is consistent with the +1.5pp `bit_manipulation` + +5pp `equation_numeric_guess` val improvements.

---

## Output artifacts on instance (28174, `/root/autodl-tmp/`)

| artifact | path / size |
|---|---|
| Submission zip (user-downloaded) | `submission_newdata_cryptnumeq_notie_outproj.zip` (3.83 GB; `adapter_model.safetensors` + `adapter_config.json` only) |
| Final checkpoint zip | `checkpoint-248_loss0.0090_lr8.06e-07.zip` (10.4 GB; adapter + optimizer + scheduler + RNG + tokenizer + training_args) |
| Soup adapters | `adapter_output_newdata_cryptnumeq_notie_outproj/adapter-{step}_loss{x}_lr{x}/` for step ∈ {50, 149, 198, 211, 223, 236, 248} — 7 saves × 4.26 GB safetensors each |

Both zips are grouped on the instance under `/root/autodl-tmp/260527_0654_submission/` (the instance-side grouping the routine §3 specifies).

---

## Local files in this folder

| file | purpose |
|---|---|
| `train_cryptnumeq_outproj.py` | the training script (copy of the run-folder version) |
| `train_log.txt` | full training log pulled from instance 2026-05-30 |
| `requirements.txt` | `pip freeze` from instance 2026-05-30 |
| `TRAINING_INFO.md` | this file |

---

## Cross-references

- `study/SCORE_TRACKER.md` Submission #24 — public LB result + 4-way val table
- `study/2605270226_sft_newdata_cryptnumeq_outproj_rtx6000/train_cryptnumeq_outproj.py` — trainer source
- `02_train/TRAINING_ROUTINE.md` Step 3 — output-saving spec this folder follows
- `02_train/260522_moe_expert_rank1_bug_report.md` — mechanism for the `out_proj`-live trick this run uses
- Sub #16 / #22 / #23 SCORE_TRACKER entries — earlier runs in the same data-swap experiment family

---

## Step 3 / Step 4 note (housekeeping)

Per `02_train/TRAINING_ROUTINE.md` Step 3 → Step 4 ordering, the small-file save (this folder) should have happened **before** the instance shutdown on 2026-05-27. It didn't — the shutdown landed first. Small files were retrieved on 2026-05-30 after the instance was rebooted (no-card mode). Cross-reference: `02_train/TRAINING_ROUTINE.md` Step 4 explicitly says *"After saving all small files"*.
