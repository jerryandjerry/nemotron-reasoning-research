# Training Info — Stripped Golden + Kaggle Regex

> Trained: 2026-05-15 14:23 → 18:32 Chicago (server 03:35–07:46 UTC+8, May 16)
> Instance: AutoDL RTX PRO 6000 Blackwell (95 GB), port 21578
> SwanLab: `260515_1423_sft_huikang_golden_stripped_kaggleregex_r32_a32_lr2e4_seq8192_bs32_rtx6000`
> SwanLab run: https://swanlab.cn/@jerry4083/260410_Nemotron/runs/yn12it7dr0y2ofgkt5crr

## What this run tests

Three small changes vs the 0.68 run (`2605141458_sft_huikang_golden_rtx6000`), all other
config byte-identical:

1. **Kaggle-metric regex** for post-`</think>` boxed extraction — handles the 88 lkevincc
   rows whose answer contains `}` (e.g. `+}`, `%}|`). Old regex `[^}]*}` truncated; new
   uses `\\boxed{` + segment-`rfind('}')`.
2. **✓ stripped** from 704 lkevincc rows (2,537 tick characters removed). Surface marker
   only — no reasoning steps removed. `Digit mapping:`, `Reading order:`, `digits:`
   verifications, operator names all preserved.
3. **id=45076dc9 row fixed** — CoT ending `\\boxed{6}}` → `\\boxed{6}` (drop one stray `}`).

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
| Checkpoint cadence | every 50 steps, FIFO keep 2 |
| Boxed extractor | Kaggle metric (`\\boxed{` + `rfind('}')`) |

## Data

`260514_huikang_golden_stripped.csv` — 6,906 unique rows → **7,849 expanded** via
`oversampling` column.

Category breakdown (expanded counts logged at startup):

| Category | n |
|---|---|
| bit_manipulation | 1,754 |
| cipher | 1,656 |
| unit_conversion | 1,070 |
| gravity | 1,055 |
| numeral | 730 |
| equation_numeric_deduce | 658 |
| cryptarithm_deduce | 639 (54 huikang concat + 585 lkevincc arithmetic) |
| cryptarithm_guess | 161 (11 huikang fallback + 150 lkevincc) |
| equation_numeric_guess | 126 |
| **Total** | **7,849** |

Source breakdown:
- `pretok_0408_decoded` huikang: 6,171 rows (with id=45076dc9 fixed)
- `lkevincc_golden`: 735 rows (with ✓ stripped from 704 of them)

## Results

- **Train time:** 246.7 min (4.11 hrs), ~1.00 min/step average
- **Final loss:** 0.0029 (step 246)
- **Loss curve:** 0.43 (step 1) → 0.092 (step 20) → 0.005 (step 50) → ~0.003 plateau through step 246
- **Peak VRAM:** 88.9 GB / 95 GB (stress test peak was 88.8 GB)
- **Trainable params:** 888,154,112 / 32,466,091,456
- **Checkpoints saved:** 50, 100, 150, 200, 246 (FIFO kept final two: 200 and 246)

## Outputs

- `submission.zip` (1.37 GB) — adapter_config.json + adapter_model.safetensors
- `checkpoint-246_loss0.0029_lr8.13e-07.zip` (3.05 GB) — full checkpoint for resume

**Adapter post-processing applied** (in training script before zipping):
- `base_model_name_or_path` → `metric/nemotron-3-nano-30b-a3b-bf16`
- `inference_mode` → True
- lm_head keys renamed: `base_model.model.lm_head.*` → `base_model.model.backbone.lm_head.*`

**Submission verification (CRC + key check):** PASS

## Score

Pending Kaggle submission.

## Comparison vs prior runs

| Run | Score | Key difference |
|---|---|---|
| #1 (pretok 0408) | 0.84 | huikang-only pretokenized data, index.jsonl replay |
| #4 (textcsv 0408 decoded) | 0.84 | same data as #1 but text CSV pipeline + huikang order replay |
| #5 (golden CSV) | 0.68 | added 735 lkevincc samples, stratified ordering, OLD regex |
| **#6 (this run)** | **?** | same as #5 + Kaggle regex + ✓ stripped + 45076dc9 fixed |

The three fixes here directly target only ~88 broken-extraction rows and ~2,537 tick
characters. The bigger bit_manipulation collapse mechanism observed in #5 (operator-
matching commit threshold shift from 5.2 → 0.9 confident matches per output) is NOT
addressed by these fixes. Expected outcome: small improvement over 0.68 from the 88
correctly-extracted lkevincc rows; bit_manipulation likely still degraded vs 0.84.
