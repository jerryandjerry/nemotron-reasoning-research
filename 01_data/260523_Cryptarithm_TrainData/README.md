# New-solver training datasets (2026-05-23)

Multi-category SFT data for Nemotron, derived from the current reference set
`../260514_lkevincc_golden/260515_huikang_lkall_stripped.csv` (6,906 rows, 9 task categories).
**The only change vs the reference is the cryptarithm data:** all 800 cryptarithm CoTs are
replaced with our new honest step-by-step solver's CoTs; the other 6,106 rows
(bit_manipulation, cipher, equation_numeric, gravity, numeral, unit_conversion) are byte-identical
to the reference. Two columns are added to every row, and three dataset variants are produced that
differ only in the cryptarithm `oversampling` factors.

## Files

| File | What it is |
|---|---|
| `260523_huikang_newsolver.csv` | **base** — cryptarithm CoTs replaced with ours; over-long ones kept but `oversampling=0` |
| `260523_huikang_newsolver_gt2x.csv` | base + **correct** cryptarithm CoTs oversampled 2× |
| `260523_huikang_newsolver_goldonly.csv` | **only correct** cryptarithm CoTs, oversampled **3×** (honest-but-wrong dropped) |
| `_build_dataset.py` | builds the base CSV (reproducible; `CAP=7800` at top) |
| `_build_gt2x.py` | derives gt2x from base |
| `_build_goldonly.py` | builds goldonly from base |

## Columns (11)

Same 8 as the reference — `id, prompt, answer, category, solver_cot, source, in_7830, oversampling`
— plus two we added:

- **`token length`** — token count of `solver_cot` under the actual Nemotron training tokenizer
  (`../../02_train/260512_huikang_085/repo/tokenizer.json`). Present for all rows.
- **`new label`** — our cryptarithm answer-path bucket (below); blank for non-cryptarithm rows.
- **`GT-match`** — `True/False`, computed with the **verbatim Kaggle metric**
  (`extract_final_answer` boxed/regex extraction + `verify`: binary→exact, numeric→1% tolerance,
  else→case-insensitive string), run on the full CoT.

For cryptarithm rows: `solver_cot` = our CoT and `source = "new solver"`.
`in_7830` is passed through unchanged.

## How the cryptarithm CoTs were produced

Our honest solver (`../260516_Cryptarithm/_honest_solver.py`) reads only `question.txt` and emits a
faithful interval-propagation DFS log (narrow ranges → lock an operator when an example is fully
known → branch → backtrack), then the answer in `\boxed{}`. It uses operator distinctness (incl. on
the unseen-operator guess), guesses the symbol of an unseen digit as the first unused operand
symbol, and falls back to `concat_fwd` when no arithmetic solution exists. All 800 were audited
line-by-line by 10 agents (only defect found — a stray unicode arrow — was fixed). Source export:
`../260516_Cryptarithm/golden_cryptarithm_cot_ours.csv` (via `_build_csv.py`).

**Length cap.** Cryptarithm DFS logs can explode (max 643k tokens; the rest of the dataset never
exceeds 7,658). Rows with `token length >= 7680` **keep their `solver_cot`** in the CSV but get
`oversampling = 0` — recorded but excluded from training. (7,680 sits just above the
non-cryptarithm max of 7,658, so the cap only ever touches cryptarithm rows.)

### Cryptarithm CoT accounting (800 rows)

- GT-match (official metric): **454 / 800** correct (cryptarithm_deduce 405/639, cryptarithm_guess 49/161).
- Over-long (≥7,680 tok → oversampling 0, CoT kept in CSV): **331**; trained (<7,680): **469**.
- Of the 469 kept: **283 correct**, **186 honest-but-wrong**.

### `new label` buckets (cryptarithm answer path)

`derived_arithmetic` 479, `guess_operator` 137, `guess_symbol` 78, `derived_concat` 59,
`blind_fallback` 28, `guess_operator_and_symbol` 19. ("derived" = forced by the examples;
"guess_*" = an unknown the examples can't determine; "blind_fallback" = no arithmetic solution.)

## Per-category distribution AFTER oversampling

Effective counts = Σ `oversampling` per category (cryptarithm_deduce + cryptarithm_guess merged;
equation_numeric_deduce + equation_numeric_guess merged). `golden` = the reference
`260514_huikang_golden_stripped.csv`. Only cryptarithm differs across our variants.

| Category | golden | base | gt2x | goldonly |
|---|---|---|---|---|
| bit_manipulation | 1754 (22.3%) | 1754 (23.3%) | 1754 (22.5%) | 1754 (22.2%) |
| cipher | 1656 (21.1%) | 1656 (22.0%) | 1656 (21.2%) | 1656 (21.0%) |
| **cryptarithm** | **800 (10.2%)** | **469 (6.2%)** | **752 (9.6%)** | **849 (10.7%)** |
| equation_numeric | 784 (10.0%) | 784 (10.4%) | 784 (10.0%) | 784 (9.9%) |
| gravity | 1055 (13.4%) | 1055 (14.0%) | 1055 (13.5%) | 1055 (13.4%) |
| numeral | 730 (9.3%) | 730 (9.7%) | 730 (9.4%) | 730 (9.2%) |
| unit_conversion | 1070 (13.6%) | 1070 (14.2%) | 1070 (13.7%) | 1070 (13.5%) |
| **TOTAL** | **7849** | **7518** | **7801** | **7898** |

Cryptarithm effective composition:
- **base**: 469 kept CoTs × 1.
- **gt2x**: 283 correct × 2 (=566) + 186 honest-wrong × 1 = 752.
- **goldonly**: 283 correct × 3 = 849 (honest-wrong and over-long excluded).

## Choosing a variant

- **base** — closest to "just swap the solver"; cryptarithm under-represented (6.2%) due to the length cap.
- **gt2x** — restores cryptarithm to ~golden's share (9.6%) by doubling correct CoTs; still exposes
  the model to honest "I can't determine this, so I guess" reasoning (wrong-but-honest CoTs at 1×).
- **goldonly** — trains only on cryptarithm CoTs that reach the correct answer, oversampled 3×
  (cryptarithm back to ~10.7%, ≈golden, from just 283 distinct correct CoTs shown 3× each). Cleanest
  targets, but removes all exposure to the honest-guess behaviour that underivable test puzzles require.

`gt2x` vs `goldonly` is worth an A/B: keeping vs dropping the honest-but-wrong cryptarithm CoTs.
