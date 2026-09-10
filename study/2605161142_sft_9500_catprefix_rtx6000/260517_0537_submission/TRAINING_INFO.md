# Training Info — 9500 official rows, answer-only CoTs with category prefix

> First run:  2026-05-17 01:03 → 02:47 Chicago (server May 17 14:03 → 15:47 UTC+8) — disk filled at final save, no submission.zip
> Resume run: 2026-05-17 05:05 → 05:30 Chicago (server May 17 18:05 → 18:30 UTC+8) — resumed from checkpoint-250, completed step 297
> Instance: AutoDL RTX PRO 6000 Blackwell (95 GB), port 21578 (disk enlarged 120GB → 150GB mid-run)
> SwanLab run: https://swanlab.cn/@jerry4083/260410_Nemotron/runs/a8hemvgbt4qou19qzezxf

## What this run tests

Hypothesis: **can the model learn the task from `(prompt + category prefix + boxed answer)` alone, with NO real chain-of-thought reasoning?**

Two changes vs run #7 (`2605152212_sft_huikang_lkall_stripped_kaggleregex_rtx6000`, scored 0.84):
1. **Data:** swapped multi-KB solver CoTs → answer-only CoTs with a one-sentence category prefix
2. **Size:** 9,500 official rows (full set) instead of huikang's curated 6,906 — adds the 2,594 rows huikang excluded (mostly downsampled easy categories + a handful of unsolved hard ones)

If this matches 0.84, the model can learn from `(prefix + answer)` with no reasoning. If it drops significantly, the CoT was doing real work and is necessary.

## Data

`260516_9500_answer_only_catprefix.csv` (4.6 MB) — 9,500 unique rows, no oversampling column (1× each).

`solver_cot` is the answer-only boilerplate with a category prefix prepended:
```
{prefix}
I will now return the answer in \boxed{}
The answer in \boxed is
\boxed{<answer>}
```

7 prefixes (cryptarithm_deduce/guess share; equation_numeric_deduce/guess share):

| Category | n | Prefix |
|---|---|---|
| bit_manipulation | 1,602 | "We need to deduce the bit transformation by matching the example outputs." |
| cipher | 1,576 | "We need to decode the substitution cipher by inferring the letter mapping from the examples." |
| unit_conversion | 1,594 | "We need to determine the linear conversion factor between the two units from the examples." |
| gravity | 1,597 | "We need to fit the gravitational formula d = k·t² to the examples to recover k, then apply it." |
| numeral | 1,576 | "We need to convert between Roman numerals and integers." |
| cryptarithm_deduce + cryptarithm_guess | 659 + 164 | "We need to deduce the symbol-to-digit mapping and the operator from the examples." |
| equation_numeric_deduce + equation_numeric_guess | 596 + 136 | "We need to deduce the numeric operation applied in the examples and apply it to the query." |
| **Total** | **9,500** | |

## Config (matches run #7 except step count)

| Parameter | Value |
|---|---|
| Model | Nemotron-3-Nano-30B-A3B bf16 (Unsloth FastLanguageModel) |
| Method | LoRA bf16 + Cut Cross-Entropy + MoE weight tying |
| LoRA | r=32, α=32, dropout=0 |
| Targets | q/k/v/o_proj, up/down_proj, in/out_proj, lm_head (9 + manual lm_head) |
| Optimizer | AdamW betas=(0.9, 0.95), eps=1e-8, wd=0 |
| LR | 2e-4 → 0 linear decay over 297 steps |
| Batch | 32 (micro=4, ga=8) |
| Max seq len | 8192 (unused — actual padded lens 200-400 because CoTs are short) |
| MoE tying | True (5,888 tied params) |
| LoRA cast | fp32 |
| Ordering | Stratified interleave (seed=42), categories proportional per batch |
| Steps | **297** (vs run #7's 246) |
| Checkpoint cadence | every 50 steps, FIFO keep 2 |
| Adapter-only soup saves | step 50, 100, 150, 200, 238 (80%), 250, 252 (85%), 267 (90%), 282 (95%); no FIFO |
| Boxed extractor | Kaggle metric (`\boxed{` + `rfind('}')`) |

## Results

- **Train time:** 147 min first run (failed at final save) + 22 min resume = ~169 min compute (~2.8 hrs)
- **Final loss:** 0.0623 (step 297) — first run 0.0625, resume 0.0623; matches within noise → resume was bit-correct
- **Best soup-candidate loss:** 0.0553 (step 282) — slightly lower than final
- **Loss curve:** 2.6 (step 1) → 0.78 (step 7) → 0.12 (step 19) plateau ~0.07–0.09 through step 297
- **Peak VRAM:** ~80 GB / 95 GB (much lower than run #7's 88.8 GB because CoTs are short)
- **Trainable params:** 888,154,112 / 32,466,091,456
- **Per-step time:** ~0.5 min (vs run #7's ~1 min) — answer-only CoTs are ~20× shorter

## Outputs (in `260517_0537_submission/` on instance)

| File | Size | Notes |
|---|---|---|
| `submission.zip` | 1.30 GB | Final adapter (step 297) with lm_head rename for Kaggle |
| `checkpoint-297_loss0.0623_lr6.73e-07.zip` | 2.88 GB | Full state for resume (adapter + optimizer + scheduler + tokenizer + rng + args) |
| `adapter-150_loss0.0777_lr9.97e-05.zip` | 1.30 GB | Soup adapter @ step 150 |
| `adapter-200_loss0.0670_lr6.60e-05.zip` | 1.30 GB | Soup adapter @ step 200 |
| `adapter-238_loss0.0765_lr4.04e-05.zip` | 1.30 GB | Soup adapter @ 80% (238) |
| `adapter-250_loss0.0691_lr3.23e-05.zip` | 1.30 GB | Soup adapter @ step 250 |
| `adapter-252_loss0.0799_lr3.10e-05.zip` | 1.30 GB | Soup adapter @ 85% (252) |
| `adapter-267_loss0.0907_lr2.09e-05.zip` | 1.30 GB | Soup adapter @ 90% (267) |
| `adapter-282_loss0.0553_lr1.08e-05.zip` | 1.30 GB | Soup adapter @ 95% (282) — lowest loss |

**Adapter post-processing applied** (to final submission.zip only; raw adapter-* zips are NOT lm_head-renamed):
- `base_model_name_or_path` → `metric/nemotron-3-nano-30b-a3b-bf16`
- `inference_mode` → True
- lm_head keys renamed: `base_model.model.lm_head.*` → `base_model.model.backbone.lm_head.*` ✓

**Submission verification:** PASS (CRC OK on instance; verify locally after Synology sync before Kaggle submit)

## Operational notes

- **First run failed at final save** because disk filled (120 GB) zipping checkpoint-297 + 9 adapter dirs + 2 FIFO checkpoints. User enlarged disk to 150 GB and asked to resume.
- **Script lacked resume logic** — added in-script: load latest checkpoint's adapter weights (with PEFT key rename `lora_X.weight` → `lora_X.default.weight`), load optimizer state, start loop at resume_step.
- **First resume attempt failed silently** — adapter weights didn't load (12,011 unexpected keys), training continued with fresh-init LoRA on top of step-250 optimizer state → garbage. Fixed by adding the PEFT key rename and retried successfully.
- **checkpoint-250 dir is preserved** on the instance for any further resume tomorrow.

## Score

Pending Kaggle submission.

## Comparison vs prior runs

| Run | Score | Data | CoT |
|---|---|---|---|
| #6 (golden stripped + Kaggle regex) | 0.84 | 6,171 huikang + 735 lkevincc cryptarithm | Full multi-KB solver CoTs |
| #7 (all-lkevincc) | 0.84 | Same 6,906 unique rows, all cryptarithm = lkevincc | Full multi-KB solver CoTs |
| **#8 (this run, 9500 catprefix)** | **?** | 9,500 official rows | **Answer-only + category prefix** |

If this scores ≥ 0.82, the category prefix carries enough signal that the model doesn't strictly need lengthy CoTs to solve the easy categories — useful for understanding what the CoT contributes. If it drops significantly (< 0.75), the CoTs are doing real work and stripping them costs accuracy.
