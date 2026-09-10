# Format & wording conformance audit — slice 0

Axis: format. Count: 70. All ids are single-§1 (rightward) solves; none open §2, use leftward, hit an illegal-reading gut-check, or take the unseen-query guess path.

Baseline note: this batch is the generator's live output (matches the worked-example reference `260525_Cryptarithm/arithmetic_a4e4ec1d` that the template points to). It differs cosmetically from the prose in `cot_template.md` in fixed, corpus-wide ways (PK item order `add,sub,mul,concat`; presence of PK item #6; the BPE-split / "read each equation" prelude; the split §N.4 "narrow the operand domains" + §N.5 "search the remaining unknowns" structure instead of the spec's merged §N.4). These are systematic and intended, so they are not scored as per-file deviations.

Checks performed (all passed across all 70 unless noted): opening sentence + "Even a symbol…" clause; BPE-split + "read each equation" headers; "convert all the equations to letter form"; prior-knowledge header + items 1–6 verbatim; sign-check phrase where an RHS op-symbol appears; traverse header; §1.1–§1.5 headers (§1.5 legitimately absent when the search resolves with no branching); §1.2 concat phrasing (`Neither matches, so X is not concat` / `a∥b|b∥a matches -> so lock X = a∥b|b∥a`); §1.3 prune wording; lock lines `so lock X = …`; reject token `— no` (em-dash; 1236 occ, no ASCII-dash variant); no `≠` anywhere; Conclusion `{…}, X of Y resolved. This is the answer.` (all X==Y); `Summary: Confirm reading order = rightward, …`; `Now solve the QUERY … applying …`; sign-reattach phrase (13 negative-answer ids, all uniform); two-line suffix with empty `\boxed{}` phrase then `\boxed{ANS}`; exactly one filled `\boxed{}`.

| id | score | deviations |
|---|---|---|
| crypt_aug_000013 | 10 | none |
| crypt_aug_000019 | 10 | none |
| crypt_aug_000023 | 10 | none |
| crypt_aug_000030 | 10 | none (sign-reattach line present & well-formed) |
| crypt_aug_000050 | 10 | none |
| crypt_aug_000062 | 10 | none |
| crypt_aug_000099 | 10 | none (sign-reattach line present & well-formed) |
| crypt_aug_000105 | 10 | none (sign-reattach line present & well-formed) |
| crypt_aug_000109 | 10 | none |
| crypt_aug_000124 | 10 | none |
| crypt_aug_000136 | 10 | none |
| crypt_aug_000150 | 10 | none (sign-reattach maps to op-symbol `@`, correct) |
| crypt_aug_000153 | 10 | none |
| crypt_aug_000182 | 10 | none |
| crypt_aug_000189 | 10 | none (§1.5 absent: search resolved in §1.4 with no branching — correct) |
| crypt_aug_000197 | 10 | none |
| crypt_aug_000204 | 10 | none (sign-reattach line present & well-formed) |
| crypt_aug_000234 | 10 | none |
| crypt_aug_000237 | 10 | none |
| crypt_aug_000285 | 10 | none |
| crypt_aug_000287 | 10 | none (sign-reattach line present & well-formed) |
| crypt_aug_000301 | 10 | none |
| crypt_aug_000302 | 10 | none |
| crypt_aug_000303 | 10 | none |
| crypt_aug_000304 | 10 | none |
| crypt_aug_000323 | 10 | none |
| crypt_aug_000338 | 10 | none |
| crypt_aug_000347 | 10 | none |
| crypt_aug_000388 | 10 | none |
| crypt_aug_000398 | 10 | none |
| crypt_aug_000420 | 10 | none |
| crypt_aug_000446 | 10 | none |
| crypt_aug_000461 | 10 | none |
| crypt_aug_000464 | 10 | none (§1.5 absent: no-branch resolve; sign-reattach line present) |
| crypt_aug_000488 | 10 | none |
| crypt_aug_000493 | 10 | none (sign-reattach line present & well-formed) |
| crypt_aug_000498 | 10 | none |
| crypt_aug_000533 | 10 | none (sign-reattach line present & well-formed) |
| crypt_aug_000543 | 10 | none (sign-reattach line present & well-formed) |
| crypt_aug_000545 | 10 | none |
| crypt_aug_000555 | 10 | none |
| crypt_aug_000558 | 10 | none |
| crypt_aug_000571 | 10 | none |
| crypt_aug_000574 | 10 | none |
| crypt_aug_000604 | 10 | none |
| crypt_aug_000650 | 10 | none |
| crypt_aug_000652 | 10 | none |
| crypt_aug_000655 | 10 | none |
| crypt_aug_000664 | 10 | none |
| crypt_aug_000669 | 10 | none |
| crypt_aug_000687 | 10 | none |
| crypt_aug_000696 | 10 | none |
| crypt_aug_000739 | 10 | none |
| crypt_aug_000740 | 10 | none |
| crypt_aug_000761 | 10 | none (sign-reattach line present & well-formed) |
| crypt_aug_000783 | 10 | none |
| crypt_aug_000803 | 10 | none |
| crypt_aug_000814 | 10 | none |
| crypt_aug_000852 | 10 | none |
| crypt_aug_000859 | 10 | none |
| crypt_aug_000862 | 10 | none |
| crypt_aug_000870 | 10 | none |
| crypt_aug_000882 | 10 | none (sign-reattach maps to op-symbol `]`, correct) |
| crypt_aug_000892 | 10 | none |
| crypt_aug_000940 | 10 | none (§1.5 absent: search resolved in §1.4 with no branching — correct) |
| crypt_aug_000941 | 10 | none |
| crypt_aug_000960 | 10 | none |
| crypt_aug_000988 | 10 | none (sign-reattach maps to op-symbol `'`, correct) |
| crypt_aug_000992 | 10 | none |
| crypt_aug_001009 | 10 | none |

## Summary
- Score 10: 70 / 70
- Score 7–9: 0
- Score 4–6: 0
- Score 0–3: 0
- No format/wording deviations found in any CoT.
