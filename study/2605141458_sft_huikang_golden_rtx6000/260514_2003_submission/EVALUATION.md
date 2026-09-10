# Evaluation — Why this run scored 0.68 (vs 0.84 baseline)

> **Status:** Per-category eval results now available. My initial root-cause analysis was wrong. See "Revised conclusion" below.

## Conclusion (revised 2026-05-15)

The drop is **dominated by bit_manipulation collapsing to near-random (74.4% → 5.0%, −69.4pp)**, with smaller drops in cipher (−8.9pp), equation (−5.8pp), and gravity (−3.1pp). Numeral and unit_conversion are unchanged.

**The training data for every one of those categories was byte-identical between the 0.84 and 0.68 runs.** Same rows, same oversampling, verified by SHA-256 row-content hash. So data composition is NOT the cause for the categories that actually scored worse.

What's different between the two runs (re-stated honestly):
- Batch order: stratified-interleave seed=42 vs huikang's index.jsonl replay
- Total samples: 7,849 vs 7,830 (the extra 735 lkevincc samples in different categories were added; the existing 7,114 in_7830=yes samples are byte-identical)
- Cryptarithm exposure: 781 → 65 (defect #1 below), but cryptarithm is not visibly broken out in the eval table

My earlier conclusion ("data composition is the root cause, specifically cryptarithm oversampling") fits the data-comparison story but **does not explain the bit_manipulation collapse**, which is the largest single component of the score drop. The bit_manipulation rows were identical across runs.

Plausible causes I cannot distinguish without ablations:
1. MoE expert routing shift caused by adding 735 lkevincc samples — re-specializes experts that previously handled bit_manipulation
2. Different sample order under linear LR decay — puts different samples in the high-LR early phase
3. Catastrophic forgetting in the last 10% of training (low LR but still non-zero) with different category mix

The regex fix in [`study/2605150937_sft_huikang_golden_kaggleregex_rtx6000/`](../../2605150937_sft_huikang_golden_kaggleregex_rtx6000/train_huikang_golden_kaggleregex.py) does **not** address any of these. It only fixes the 88 lkevincc rows where the old extractor truncated `}`-containing answers.

## Lessons (what I should have done)

- Per-category eval is the first thing to ask for, not the last. The aggregate score collapse made me jump at the obvious data change (cryptarithm oversampling).
- "Training data identical" deserves more weight. When bit_manipulation training data is unchanged but bit_manipulation score crashes, the cause must be a **training-dynamics** difference (order, MoE routing, total batch composition), not a data-composition difference.
- I should have noted that cryptarithm doesn't appear by name in the public test breakdown before making it the headline diagnosis.

## Settings compared with 0.84 run

Diff against [`study/2605132323_sft_huikang_textcsv_0408_rtx6000/train_huikang_textcsv_0408.py`](../../2605132323_sft_huikang_textcsv_0408_rtx6000/train_huikang_textcsv_0408.py) — functional differences only:

| | 0.84 run | This run |
|---|---|---|
| CSV | `huikang_7830_0408.csv` (7,830 pre-expanded) | `260514_huikang_golden.csv` (6,906 unique + `oversampling` col) |
| Order | replay `index.jsonl` from huikang's run | stratified interleave (`rank/category_size` + tiny jitter) |
| Pretok verify | yes (5 sample IDs) | n/a (no pretok for new data) |

Everything else identical: LoRA r=32/α=32, bf16 + CCE + MoE tying, manual lm_head LoRA, fp32 LoRA cast, AdamW (betas=0.9/0.95, eps=1e-8, wd=0), lr 2e-4 linear → 0, bs=32 micro=4 (ga=8), seq=8192, raw `tokenizers.Tokenizer` for completion, `re.escape(chr(92)+'boxed{')` regex with `[^}]*` capture, response-only masking, Mamba fast-path patch.

## Stratified batching — verified working

From `train_log.txt`:
```
Stratified interleave: 7849 samples, seed=42
Batching: stratified, 246 steps
```
Per-batch sample_ids in the log show every batch mixed across categories (e.g., cryptarithm + gravity + numeral + bit_manipulation + equation in the same 32-sample batch). No adjacent duplicates triggered. The interleave is not the cause.

## Data defect #1 — cryptarithm oversampling collapsed (CRITICAL)

In `260514_huikang_golden.csv`, every `cryptarithm_deduce` and `cryptarithm_guess` row has `oversampling=1`. In the original 0.84 corpus (`260514_huikang_update/huikang_7830.csv`) the same 65 puzzles had:

| Category | 0.84 expanded | This run expanded | Delta |
|---|---|---|---|
| cryptarithm_deduce (54 unique) | 21×11 + 33×12 = **627** | 54×1 = **54** | **−573** |
| cryptarithm_guess (11 unique) | 11×14 = **154** | 11×1 = **11** | **−143** |
| **cryptarithm total** | **781** | **65** | **−716** |

The 735 lkevincc samples added do NOT replace these — they are entirely new puzzle IDs (verified: 0/735 overlap with the 65 original cryptarithm IDs):

```
arithmetic                  : 315 new IDs
little_endian               : 276 new IDs
mixed_concat                : 108 new IDs
mixed_concat_little_endian  :  26 new IDs
pure_concat                 :   5 new IDs
query_unseen_concat         :   5 new IDs
```

Net effect: we trained with **716 fewer instances on the actual cryptarithm puzzles** the test set evaluates on, and added 735 instances on different puzzles. Total ~7,849 ≈ 7,830 but the **distribution is wrong**.

Training-log proof (`Category breakdown (expanded):`):
```
cryptarithm_deduce: 54
cryptarithm_guess: 11
```

## Data defect #2 — 88/735 lkevincc samples have broken boxed answers

12% of the new lkevincc samples have an `answer` containing `}`, which the boxed-extraction regex `\boxed{([^}]*)}` truncates at the first `}`. Concrete examples:

| id | category | answer (truth) | what regex extracts (what model learns) |
|---|---|---|---|
| `01b2aa67` | little_endian | `+}` | `+` |
| `0454705a` | arithmetic | `%}|` | `%` |
| `0d2e94ff` | arithmetic | `}}^` | `` (empty) |

These 88 samples teach the wrong boxed answer. Minor compared to defect #1 but still a real bug in the CSV.

> **Update 2026-05-15:** Re-classified as a **training-script bug**, not a data defect. The lkevincc CSV is correct — its CoTs encode answers like `+}` as `\boxed{+}}`, which Kaggle's official metric ([nvidia-nemotron-metric](https://www.kaggle.com/code/metric/nvidia-nemotron-metric)) parses correctly via `rfind('}')`. The 0.84-era extractor `\boxed{([^}]*)}` (still in the original 0.68 training script at line 136) truncates at the first `}` and breaks for these answers. Fix lives in the new training script [`study/2605150937_sft_huikang_golden_kaggleregex_rtx6000/train_huikang_golden_kaggleregex.py`](../../2605150937_sft_huikang_golden_kaggleregex_rtx6000/train_huikang_golden_kaggleregex.py) — swaps to Kaggle's `\boxed{` + segment-`rfind('}')` extractor. Verified on the actual data: all 735 lkevincc rows now round-trip correctly (training extract → constructed post-`</think>` line → Kaggle eval == answer column). 0 failures.

## Training itself was healthy

- 7,849 examples loaded, 27,994,459 tokens (26,718,951 unmasked)
- 246 steps, 250.5 min (4.17 hrs), ~1.02 min/step
- Peak VRAM 88.8 GB / 95 GB
- Loss: 0.43 → 0.06 by step 10 → ~0.007–0.011 plateau, final 0.0074
- Submission zip CRC-verified, lm_head keys renamed, base_model_name_or_path correct

## Fix candidates (not yet executed)

1. **Restore + add:** put back the original 627 / 154 oversampling AND keep all 735 lkevincc samples. Total ~8,565 expanded. Tests whether the lkevincc data helps when the original cryptarithm exposure is preserved.
2. **Replace 1:1:** use ONLY the 735 lkevincc samples for cryptarithm. Tests the golden solver alone.
3. **Baseline retry:** re-run exact 0.84 CSV first to confirm reproducibility before changing data again.

The 88 broken-answer samples should be either fixed or filtered in any retraining run.
