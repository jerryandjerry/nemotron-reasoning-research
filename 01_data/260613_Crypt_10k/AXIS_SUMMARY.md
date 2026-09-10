# Cryptarithm CoT — complete scenario / axis map (2026-06-14)

Measured across all 797 originals. `corpus / trainable` = count in corpus / count that is GT-True ∩ under-7680-tok.
Trainable total = 141. **This is the full space the 5k aug must cover — every primary scenario AND every
orthogonal axis, decorrelated.**

## A. PRIMARY answer-derivation — 5 mutually-exclusive scenarios (every CoT is exactly one)

| # | scenario | how the QUERY answer is derived | corpus / trainable |
|---|---|---|---|
| 1 | **arith-solved** | query op SEEN in examples → ~add/~sub/~mul → compute arithmetically | 326 / **66** |
| 2 | **seen-concat** | query op SEEN = ~concat → §2 short-circuit → answer = a∥b/b∥a | 59 / **59** |
| 3 | **query-only→arith** | query op UNSEEN → "others are arithmetic, so it is too → distinct rule picks the remaining family in canonical order" | 64 / **7** |
| 4 | **query-only→concat** | query op UNSEEN → concat (STEP-2a: no 4-digit rhs; or STEP-2b: no arith solved) | 60 / **9** |
| 5 | **unsolvable→concat** | puzzle can't be solved rightward → blind a∥b | 288 / **0** ⬅ gap |

**Scenario 5 has 5 distinct CAUSES** (the failure mode the CoT shows; may co-occur): illegal-reading 15,
**symbol-domain-empties 188**, **all-different-collision 114**, pigeonhole 21, operator-family-exhausted 97 —
all **0 trainable** (this whole class is `little_endian`/`mixed_concat_le`, GT-False; aug must plant gold=concat).

## B. ORTHOGONAL feature-axes — co-occur with the 5 types; MUST be decorrelated (same rate in every type)

**B1. Answer rendering**
- negative result → sign re-attach (`mapping the subtraction sign …`): 39 / **4**
- missing-answer-digit → ASCII default: 53 / **1** ⬅ thin
- concat answer (∥): 407 / 68

**B2. Deduction mechanisms** (the reasoning steps the CoT must demonstrate)
- §5 branch + backtrack (vs §4-only): 521 / 67
- binary-search narrowing (~add/~mul monotone): 585 / 73
- ~sub non-monotone enumeration: 77 / **13**
- LEADPAIR rule D (~sub, 1-digit rhs): 87 / **7** ⬅ thin
- advance to next family combination: 110 / **3** ⬅ thin
- free-digit default (last unknown unpinned → smallest): 66 / **6** ⬅ thin
- sign-in-rhs (~sub forced by rhs negative sign): 335 / 60
- operator forced by distinct-operator rule: 64 / 7
- all-different forcing (cipher pin): 369 / 53
- hidden-single / forced-0 Elimination: 210 / 19

**B3. Operator families present** (+ noise variants a×b±k, a+b±k, |a-b|/a-b/-|a-b|/b-a±k): all 4 exercised across trainable.

**B4. Structural axes**
- #operators: 1 (12/**2**) · 2 (296/58) · 3 (489/81)  — nops=1 thin
- #examples: 3 (232/38) · 4 (289/45) · 5 (276/58)
- operator symbol class: arith glyph `+−*` (627/105) vs all-punctuation (170/36)
- #distinct symbols (cipher size)

**B5. BPE merge-pair coverage** (CRITICAL — the un-merge step) — pool = 26 symbols → **343 theoretical
2-symbol merge-pairs** the tokenizer collapses. Trainable covers **325 (95%); 18 missing**
(`$( $: %] &# (# )/ ** *- *\ *} +) ++ +- -* -- [( |} }]`).

## C. The TRAINABLE GAPS the aug must fill (ranked)
1. **unsolvable→concat = 0 trainable** (288 corpus) — synthesize GT-True (gold=concat), cover all 5 causes.
2. **query-only→arith = 7**, **query-only→concat = 9** — under-covered.
3. missing-digit-ASCII = 1, advance-combo = 3, LEADPAIR-D = 7, free-digit = 6, negative = 4, nops=1 = 2 — thin.
4. 18 BPE merge-pairs absent from trainable.

**Decorrelation rule (B):** none of the B-axes may correlate with the A-type. Inject each B-feature at the same
rate across all 5 types; no forced corner subtypes; then prove flat per-type tables on the combined 5k.
