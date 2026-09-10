# Training Info — All-lkevincc Cryptarithm + Kaggle Regex

> Trained: 2026-05-15 22:19 → 2026-05-16 02:29 Chicago time (server May 16 11:19 → 15:29 UTC+8)
> Instance: AutoDL RTX PRO 6000 Blackwell (95 GB), port 21578
> SwanLab run: https://swanlab.cn/@jerry4083/260410_Nemotron/runs/bfmkvjjavt0s0kc9roaz0

## What this run tests

Hypothesis: **does using lkevincc CoTs for ALL cryptarithm puzzles (including the 65 huikang already covered) change the score?**

Single change vs run #6 (`2605151409_sft_huikang_golden_stripped_kaggleregex_rtx6000`, scored 0.84):
- The 65 cryptarithm puzzle IDs that huikang's solver covered now use **lkevincc's CoT** instead of huikang's. Same 65 puzzle IDs, same prompts/answers, different solver CoTs.

So cryptarithm composition is:
- Run #6: 65 huikang concat-only CoTs + 735 lkevincc CoTs = **800 cryptarithm samples, two CoT styles**
- This run (#7): 800 lkevincc CoTs = **800 cryptarithm samples, one CoT style**

Non-cryptarithm rows (6,106 huikang) are byte-identical to run #6.

## Config (matches run #6 exactly)

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
| Checkpoint cadence | every 50 steps, FIFO keep 2 |
| Boxed extractor | Kaggle metric (`\\boxed{` + `rfind('}')`) |

## Data

`260515_huikang_lkall_stripped.csv` — 6,906 unique rows → **7,849 expanded** via `oversampling` column.

Source breakdown:
- `pretok_0408_decoded` huikang: 6,106 rows (non-cryptarithm only)
- `lkevincc_golden`: 800 rows (all cryptarithm — 65 swapped-in + 735 originally added)

Category breakdown (expanded counts logged at startup):

| Category | n |
|---|---|
| bit_manipulation | 1,754 |
| cipher | 1,656 |
| unit_conversion | 1,070 |
| gravity | 1,055 |
| numeral | 730 |
| equation_numeric_deduce | 658 |
| cryptarithm_deduce | 639 (54 swapped from huikang + 585 lkevincc) |
| cryptarithm_guess | 161 (11 swapped from huikang + 150 lkevincc) |
| equation_numeric_guess | 126 |
| **Total** | **7,849** |

## Results

- **Train time:** 243.2 min (4.05 hrs), ~0.99 min/step average
- **Final loss:** 0.0029 (step 246)
- **Loss curve:** 0.40 (step 2) → 0.06 (step 24) → 0.005 (step 50) → ~0.003 plateau through step 246
- **Peak VRAM:** ~88.8 GB / 95 GB
- **Trainable params:** 888,154,112 / 32,466,091,456
- **Checkpoints saved:** 50, 100, 150, 200, 246 (FIFO kept final two: 200 and 246)

## Outputs

- `submission.zip` (1.37 GB) — adapter_config.json + adapter_model.safetensors, CRC verified
- `checkpoint-246_loss0.0029_lr8.13e-07.zip` (2.84 GB) — full checkpoint for resume

**Adapter post-processing applied:**
- `base_model_name_or_path` → `metric/nemotron-3-nano-30b-a3b-bf16`
- `inference_mode` → True
- lm_head keys renamed: `base_model.model.lm_head.*` → `base_model.model.backbone.lm_head.*` ✓

**Submission verification:** PASS

## Score

**0.84** (Kaggle submission ref 52710786) — matches run #6 exactly. Consistent one-template lkevincc cryptarithm CoT is neither better nor worse than the mixed 65-huikang + 735-lkevincc composition at this training scale.

## Comparison vs prior runs

| Run | Score | Cryptarithm CoT source |
|---|---|---|
| #1 / #4 (pretok or 0408 textcsv) | 0.84 | All 65 from huikang, 12-14× oversampled (781 instances) |
| #5 (golden CSV) | 0.68 | 65 huikang × 1 + 735 lkevincc × 1; old regex; ✓ in CoTs; 45076dc9 typo |
| #6 (golden stripped + Kaggle regex) | 0.84 | 65 huikang × 1 + 735 lkevincc × 1; Kaggle regex; ✓ stripped; 45076dc9 fixed |
| **#7 (this run)** | **0.84** | 800 lkevincc × 1 (65 swapped + 735); everything else identical to #6 |

If this run scores ≥ 0.84, lkevincc CoTs alone are sufficient for cryptarithm and we have a one-template consistent training. If it scores < 0.84, huikang's CoTs on the 65 shared puzzles provide useful training signal (positive operator identification in 54/65) that lkevincc's "default to concatenation" fallback doesn't replicate.
