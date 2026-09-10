# CoT wording overhaul — 2026-06-12 (equation_numeric) → for the cryptarithm agent to mirror

## Why
The words **"guess"** and **"unknown"** make the model sloppy: they read as "the rules are off now,"
so the continuation slackens. But nothing in the solver is actually a guess — an operator that the
examples don't fix is resolved by a **deterministic default rule**. So we (a) removed both words from
the CoT output, (b) reframed the fallback as a named, explicit rule, and (c) defined the two
"can't-solve" verdicts up front.

**All changes are narration-only.** equation_numeric regenerated all 5000 trained CoTs with
**0 boxed-answer changes** — the search logic is untouched, only the surface wording changed. The
cryptarithm change should likewise leave every `\boxed{}` answer identical.

## The principle (apply this, not just the literal strings)
1. **Never** print `guess`, `I need to guess`, `make a best guess`, or `= unknown` as a verdict.
2. Two distinct "can't-solve" verdicts, named explicitly:
   - **QUERY-only operator** — the symbol appears only in the QUERY, so there is no example to solve
     it from → it is resolved **by default** (a stated, deterministic rule).
   - **exotic operator** — the symbol appears in the examples but no modeled rule fits → it is
     outside the system (equation_numeric then locks `max(a,b) mod min(a,b)`).
3. The by-default resolution must be **explicit and deterministic** — show the ordered candidate set
   and the narrowing, never "I'll guess."
4. Keep any genuinely-fixed boilerplate that uses the word "unknown" only in the harmless
   math sense (we kept `Denote the operators as unknown functions:` — "unknown function" = a function
   to be determined, not a give-up verdict).

## equation_numeric: every BEFORE → AFTER (for reference)

**Opening line**
- BEFORE: `I need to infer the transformation rule from the examples. The operands are given already, I only need to deduce the operators.`
- AFTER: `Each symbol is a hidden operator. I work out each operator from the example equations, then apply the QUERY's operator to the QUERY operands and return the answer.`

**Prior-knowledge item 6** (it was the "never return unknown" nudge; now it DEFINES the two verdicts)
- BEFORE: `6. I should never return the answer as "unknown". When the final result is undetermined, I should make a best guess using what I know.`
- AFTER: `6. If the QUERY operator only appears in the QUERY, we cannot solve it from the examples; we call it a QUERY-only operator. If the QUERY operator appears in the examples but cannot be solved, it is outside the system, and we call it an exotic operator. In either case we still resolve it to a concrete answer.`
  - (Note: phrased "cannot be solved", NOT "no operation fits" — the word **"fits" is a forbidden token** corpus-wide, tied to the honest-search convention; do not reintroduce it.)

**Operator state / conclusion** (the unsolvable-from-examples query operator)
- BEFORE: `g = unknown` (in state lines, the `Try operators … g = unknown:` line, and the conclusion `{… g = unknown}`)
- AFTER: `g is a QUERY-only operator` (parallels the existing `g is an exotic operator`)
- Also removed the redundant §N.1 tail `… since g only appears in QUERY` (prior #6 now defines it).

**Resolution block** (was the "guess" step)
- BEFORE lead-in: `Query operator g only appears in QUERY, so I need to guess it.`
- AFTER lead-in: `Query operator g appears only in the QUERY, so no example fixes it; I will assign it by default.`
- BEFORE (arithmetic branch): `At least one operator here can be solved arithmetically, so I'll guess g is an arithmetic operator too. Since ~add has appeared in the puzzle, by the distinct operator rule g is ~sub or ~mul; ~sub comes first in the default order, so it is the pick.`
- AFTER (arithmetic branch, explicit): `At least one operator is solved arithmetically, so g is arithmetic too, so g = [~add, ~sub, ~mul]. Since f = ~add, the distinct-operator rule leaves g = [~sub, ~mul], so in order I assign g = ~sub.`
- BEFORE (concat branch): `No operator here can be solved arithmetically, so I guess the unseen operator g is ~concat. Let me use g = a∥b.`
- AFTER (concat branch): `No operator is solved arithmetically, so g is ~concat. Let me use g = a∥b.`

**Kept unchanged** (deliberately): `Denote the operators as unknown functions:`

## What the cryptarithm CoT should change

Cryptarithm solves *more* than equation_numeric: the operands are **also** hidden (each symbol is a
ciphered digit), plus the operators. So the mapping is:

1. **Opening** — currently `I need to infer the transformation rule from the examples.` Make it name
   BOTH unknowns, e.g.:
   `Each symbol is a hidden digit and each operator is hidden too. I work out the digits and operators from the example equations, then apply them to the QUERY and return the answer.`
   (adjust to the cryptarithm solver's exact terminology — digits/letters/operators).
2. **Anywhere the cryptarithm CoT prints `guess` / `I need to guess` / `make a best guess` / `= unknown`**
   as a verdict → swap to the same scheme:
   - a symbol/operator only in the QUERY → `… is a QUERY-only operator`, resolved `by default`.
   - a symbol/operator tried but unfittable → `… is an exotic operator` (if cryptarithm has that notion).
   - the explicit narrowing form (`… the distinct rule leaves X = [...], so in order I assign X = …`)
     instead of "I'll guess."
3. **Keep** the harmless `unknown function(s)` math phrasing if present.
4. **Narration-only:** regenerate and verify **0 boxed-answer changes** (same check we used).

## equation_numeric deliverable (already shipped with the new wording)
`260612_NumericEq_5k/260612_TrainData/260612_NumericEq5k_FULL.csv` (12149 rows; eq 5000 effective;
non-eq byte-identical to base; old wording archived as `…_FULL_OLDWORDING.csv`).
Solver: `01_data/260606_Numeric_Equation/_gen_eq.py`; spec synced: `…/cot_template.md`.
