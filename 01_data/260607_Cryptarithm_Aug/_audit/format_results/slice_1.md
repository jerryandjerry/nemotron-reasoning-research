# Format & Wording Conformance Audit — slice_1

Axis: format (structure, fixed phrases, ordering, lock/reject wording, answer rendering). Reasoning/math correctness NOT judged.

Baseline: the `_aug` set follows a self-consistent evolution of cot_template.md. Set-wide conventions (treated as the conformance reference, NOT penalized): BPE-split + read-symbol preamble; Prior knowledge items 1-6 (incl. #6 never-return-unknown); split of the search into `§N.4 narrow the operand domains` + `§N.5 search the remaining unknowns by branching`; bare lock formulas (`a×b-2`, `a∥b`, ...) rather than `[VARIANT] = [FORMULA]`; verdict forms `§2 solved the QUERY operator [op], which §1 did not, so §2 is preferred.` and `§1 is illegal, so §2 is the only legal reading.`; reading descriptors `rightward` / `leftward on digit only` / `leftward fully`; sign re-attach line when the result is negative. `unknown` in a §1 conclusion followed by `Need to try reading leftward.` is template-correct for the unseen-query-operator case (final `\boxed{}` is never `unknown`).

Only concrete deviation found in this slice: use of U+2212 MINUS SIGN (`−`) instead of the legend-mandated U+002D HYPHEN-MINUS (`-`) inside interval-pruning arithmetic (e.g. `40×41−2 = 1638`). The legend states "Minus is ALWAYS `-` (U+002D)". Cosmetic / intermediate-only — lock-formulas, negative answers, and `\boxed{}` all use U+002D correctly. Score 10 = no occurrence; 9 = one or more occurrences.

| id | score | deviations |
|---|---|---|
| crypt_aug_001036 | 9 | U+2212 minus sign in interval arithmetic (line 75 `40×41−2`, line 88) instead of U+002D |
| crypt_aug_001066 | 10 | none |
| crypt_aug_001093 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (1 line) |
| crypt_aug_001094 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (6 lines) |
| crypt_aug_001101 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (9 lines) |
| crypt_aug_001118 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (6 lines) |
| crypt_aug_001130 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (7 lines) |
| crypt_aug_001134 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (1 line) |
| crypt_aug_001138 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (4 lines) |
| crypt_aug_001165 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (11 lines) |
| crypt_aug_001173 | 10 | none |
| crypt_aug_001194 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (1 line) |
| crypt_aug_001201 | 10 | none |
| crypt_aug_001213 | 10 | none |
| crypt_aug_001217 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (1 line) |
| crypt_aug_001221 | 10 | none (sign re-attach line present and correct: `-&] -> <&]`) |
| crypt_aug_001230 | 10 | none |
| crypt_aug_001282 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (10 lines) |
| crypt_aug_001284 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (3 lines) |
| crypt_aug_001304 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (1 line) |
| crypt_aug_001334 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (3 lines) |
| crypt_aug_001346 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (3 lines) |
| crypt_aug_001347 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (6 lines) |
| crypt_aug_001359 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (4 lines) |
| crypt_aug_001363 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (7 lines) |
| crypt_aug_001370 | 10 | none |
| crypt_aug_001410 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (9 lines) |
| crypt_aug_001423 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (17 lines) |
| crypt_aug_001435 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (1 line) |
| crypt_aug_001444 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (5 lines) |
| crypt_aug_001458 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (2 lines) |
| crypt_aug_001476 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (8 lines) |
| crypt_aug_001482 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (7 lines) |
| crypt_aug_001503 | 10 | none |
| crypt_aug_001507 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (1 line) |
| crypt_aug_001542 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (7 lines) |
| crypt_aug_001545 | 10 | none (§N.5 header absent because §N.4 dead-ends before branching — cadence-correct) |
| crypt_aug_001548 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (5 lines) |
| crypt_aug_001589 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (10 lines) |
| crypt_aug_001594 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (3 lines) |
| crypt_aug_001604 | 10 | none |
| crypt_aug_001606 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (3 lines) |
| crypt_aug_001619 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (2 lines) |
| crypt_aug_001664 | 10 | none |
| crypt_aug_001703 | 10 | none |
| crypt_aug_001706 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (1 line) |
| crypt_aug_001719 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (10 lines) |
| crypt_aug_001792 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (11 lines) |
| crypt_aug_001805 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (3 lines) |
| crypt_aug_001826 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (5 lines) |
| crypt_aug_001827 | 10 | none |
| crypt_aug_001863 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (1 line) |
| crypt_aug_001866 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (1 line) |
| crypt_aug_001875 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (6 lines) |
| crypt_aug_001904 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (3 lines) |
| crypt_aug_001944 | 10 | none |
| crypt_aug_001966 | 10 | none |
| crypt_aug_001994 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (4 lines) |
| crypt_aug_001999 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (10 lines) |
| crypt_aug_002029 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (6 lines) |
| crypt_aug_002038 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (1 line) |
| crypt_aug_002040 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (14 lines) |
| crypt_aug_002064 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (5 lines) |
| crypt_aug_002094 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (2 lines) |
| crypt_aug_002174 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (1 line) |
| crypt_aug_002188 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (4 lines) |
| crypt_aug_002206 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (10 lines) |
| crypt_aug_002207 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (10 lines) |
| crypt_aug_002209 | 9 | U+2212 minus sign in interval arithmetic instead of U+002D (2 lines) |
| crypt_aug_002251 | 10 | none |

## Buckets
- 10 (perfect): 15
- 9 (one minor character-legend deviation): 55
- 7-9 total: 70
- 4-6: 0
- 0-3: 0

## Top issues
1. U+2212 MINUS SIGN used instead of legend-mandated U+002D HYPHEN-MINUS in interval-pruning arithmetic — 55/70 files. Example: crypt_aug_001423 (17 occurrences). Cosmetic/intermediate; lock formulas, negative answers, and `\boxed{}` correctly use U+002D.
