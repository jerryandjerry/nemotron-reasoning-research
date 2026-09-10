# Cryptarithm CoT dataset — summary report

**Date:** 2026-05-23 · **Goal:** produce honest, faithful Chain-of-Thought (CoT) solutions for the
800 "Alice's Wonderland" cryptarithm puzzles, as SFT training data for NVIDIA Nemotron.

---

## 1. The puzzle

Each puzzle gives a few **example equations** and one **query**, all of the form
`[d1][d2][op][d3][d4] = RHS`, written in a symbol alphabet. The task: produce the query's RHS.

- **Operand alphabet — 23 symbols (ASCII order):** `! " # $ % & ' ( ) / : < > ? @ [ \ ] ^ ` { | }`.
  `+ - *` are operator-only and never operands. Each puzzle uses 10 of the 23 as its digits 0–9.
- **Operators** are `+ - *` (strong arithmetic indicators) or arbitrary symbols. Their meaning is
  **noisy**: multiplication / addition / subtraction, possibly off by ±1/±2, negated, or with
  operands reversed — plus **concatenation** (`concat_fwd`/`concat_rev`).
- **Symbol↔digit mapping is a fresh per-puzzle uniform-random bijection** — no global rule, not
  derivable from the puzzle id (verified; see §7).
- **800 puzzles** by category: arithmetic 315, little_endian 276, mixed_concat 108,
  mixed_concat_little_endian 26, pure_concat 59, query_unseen_concat 16.

## 2. The solver — `_honest_solver.py`

Reads **only `question.txt`** (integrity: regenerating with `answer.txt` deleted is byte-identical —
the CoT never reasons backward from the gold answer). It is a faithful **interval-propagation DFS**:

1. narrow each variable's range from one equation given the others' current ranges, to a fixpoint;
2. **eagerly lock** an operator's variant the instant one example's operands AND RHS are fully known
   (in every branch);
3. branch on the least-unknown equation; backtrack on contradiction.

It reuses `cot_generator.make_prefix` for the setup preamble and `_gen_solutions` for answer
rendering. Output per puzzle: `<puzzle>/track/tree_cot.txt` (the original tree-solver CoT is
preserved at `<puzzle>/track/archive/`).

### Answer paths (the `new label` buckets) and GT-match

| Bucket | Count | GT-match | Meaning |
|---|---|---|---|
| derived_arithmetic | 479 | 342 (71%) | operator from examples, all digits present |
| derived_concat | 59 | 59 (100%) | operator identified as concat from examples |
| guess_operator | 137 | 47 (34%) | query operator unseen → guess by prior (distinctness-aware) |
| guess_symbol | 78 | 4 (5%) | answer needs an unseen digit → guess the symbol |
| guess_operator_and_symbol | 19 | 2 (10%) | both guessed |
| blind_fallback | 28 | 0 (0%) | no arithmetic solution (exotic op) → guess concat_fwd |
| **TOTAL** | **800** | **454 (57%)** | |

"derived" = forced by the examples; "guess_*" = an unknown the examples genuinely can't determine.
The 57% is **not** a bug rate — every solution satisfies all examples (0 INVALID); the misses are
ambiguity (the 3 examples don't always pin the noisy variant/reading order) plus underivability
(unseen operators/symbols, exotic operations). GT-match is the verbatim Kaggle metric.

## 3. Format & notation (decisions, all enforced)

- One consistent **ASCII arrow `->`** (never `→`).
- Operators labelled **`f` `g` `h`** (never `x`, which reads as ×).
- Operator meaning: **`~mul` / `~add` / `~sub`** before an operator is locked; the exact
  **spelled-out variant** after (`mul`, `add_plus1`, `mul_minus1`, `absdiff_minus2` — never `add_m1`).
- Real arithmetic uses `×`, `+`, `−`, `|a-b|`.
- Front matter order: letter-form equations, then operator background. For `+ - *` the background
  states the canonical meaning ("+, denoted as f, is usually noisy_add, but I have seen…").
- Answer block ends with two lines: `I will now return the answer in \boxed{}, the answer is` then
  `\boxed{X}`; the answer line shows both forms `answer = <letters> = <symbols>`.

### Operator distinctness (key correctness win)

Operators are distinct. On the **unseen-query-operator guess** this is applied at the **operation
(family) level**: drop any arithmetic family already used by a locked operator, then use the prior
only to break a tie among what remains (and not at all when one operation is left):
`Since mul and add have appeared in the puzzle, by the distinct operator rule h must be the remaining
operation, noisy_subtraction.` This matches the generator (operators get distinct *operations*) and
**raised GT-match 442 → 454 (+12)**.

### Missing-symbol guess (the `_` problem)

For the 97 puzzles whose answer needs a digit absent from the examples, we **never emit `_`** (a
guaranteed-wrong placeholder, 0%). Instead the CoT prints the full alphabet, then the alphabet with
every used symbol masked `X`, states the mapping is uniform-random, and **picks the first remaining
candidate**:
```
I know the full set of operand symbols, ordered by ASCII:
  ! " # $ % & ' ( ) / : < > ? @ [ \ ] ^ ` { | }
Excluding every symbol that already appears in this puzzle (marked X), I still have:
  ! " X X % & X ( X X : X > X X [ \ X ^ ` { | X
... none of the remaining candidates is more likely than another.
Since I have to choose, I will just take the first one that is left: 8 -> !, answer = FH!F = @}!@
```
First-of-remaining is a **deterministic, learnable** rule (the model can reproduce it from visible
features); a seeded-random pick would be unlearnable noise. Expected ~7% (the ceiling — the unseen
symbol is uniform).

## 4. Verification — 10-agent line-by-line audit

All 800 CoTs were audited line-by-line by 10 agents against a shared contract. **Exactly one
systematic defect** was found — a stray unicode `→` in the concat-direction line — and fixed at
`cot_generator.py:396`. Everything else was clean: arithmetic recomputed correct, boxed answers
follow the `Solution:` mapping, missing-symbol / distinct-operator / concat-fallback logic faithful,
the large DFS logs honest. Post-fix: 0 unicode arrows, 0 `_` in answers, 0 stale (on-disk == fresh
regen), all originals archived.

## 5. Pipeline files (top level)

| File | Role |
|---|---|
| `cot_generator.py` | setup preamble (`make_prefix`), operator priors (`op_dict` reads `operator_dict.md`), concat detection, pruning narration |
| `_honest_solver.py` | the DFS engine + `emit_answer` — **the production CoT generator** |
| `_gen_solutions.py` | answer rendering (`num_to_sym`, family→base-variant `BASE`) |
| `_gen_tree_cot.py` | batch regenerator → `<puzzle>/track/tree_cot.txt` (idempotent archive of originals) |
| `_build_csv.py` | builds `golden_cryptarithm_cot_ours.csv` (cot + ikev label + new label + solver answer + GT-match + token length) |
| `_verify_honest.py` | full-corpus validity checker (distinct digits, every example satisfied) |
| `operator_dict.md` | per-symbol operator-meaning frequencies (the prior; read by the solver) |
| `operand_dict.md` | per-symbol digit-frequency reference + the missing-symbol analysis |
| `tree_cot_review.md` | running record of format/logic decisions |
| `cot_template.md` | CoT format template |
| `golden_cryptarithm_cot_ours.csv` | our CoTs + metadata, one row per puzzle |
| `golden_cryptarithm_cot_raw.csv` | the original ikev CoT (reference, untouched) |

`~archive/` holds dated code snapshots and the `260523_scratch/` cleanup (all diagnostic/audit
scripts and JSON dumps from development).

## 6. Training datasets delivered

`../260523_new_data/` — the reference multi-category SFT set with our 800 cryptarithm CoTs swapped
in (`source = new solver`), CoTs over 7,800 tokens blanked (`oversampling = 0`), plus `token length`,
`new label`, `GT-match` columns. Three variants differ only in cryptarithm oversampling:
**base** (kept ×1, cryptarithm 6.3%), **gt2x** (correct ×2, 9.7%), **goldonly** (correct only, ×3,
10.8%). See that folder's `README.md`.

## 7. Notable findings

- **Symbol↔digit mapping is not derivable.** Per-puzzle uniform-random; ruled out a global cipher,
  ASCII order, the folder-id as an RNG seed (0/436 even on the symbol set), and any per-symbol prior
  (the unseen-symbol distribution is uniform, χ² n.s.). So the missing-symbol guess can't beat ~7%.
- **Family-level operator distinctness** matches the generator — confirming operators get distinct
  *operations*, not merely distinct variants.
- **Exotic operators** (gcd / modulo / rmod / lcm / fdiv) appear in the examples of ~16 puzzles; our
  model doesn't cover them, which is why those puzzles fall to the (0%) blind-fallback.

## 8. Known limitations / future work

1. **CoT length bloat.** Token length: median ~5,944 but a heavy tail — 112 of 800 over 32k tokens,
   max 643k (the rest of the dataset never exceeds 7,658). The DFS log explodes on hard puzzles,
   especially `little_endian` (which runs a full standard-order search, fails, then re-searches in
   little-endian — ~2× the log). Capping/summarizing the search log (especially the doomed
   standard-order pass) would recover the ~328 currently-blanked rows — including correct concat
   CoTs — for training.
2. **Exotic operators** — model gcd/mod/lcm, or skip an unmodelable example and solve from the rest,
   to rescue the 28 blind-fallback puzzles.
3. **query_unseen_concat (16)** — the unseen concat query operator is guessed as arithmetic (0%);
   undecidable from the prompt alone.
