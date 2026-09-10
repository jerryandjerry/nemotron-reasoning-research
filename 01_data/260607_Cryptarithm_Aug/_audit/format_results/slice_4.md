# Format & wording conformance audit — slice 4

Axis: format. Count: 69. Mix of single-§1 (rightward) solves and §2 (leftward) solves; no unseen-query / concat-guess path in this slice.

Baseline note: this batch is the generator's live output (matches the worked-example reference `260525_Cryptarithm/arithmetic_a4e4ec1d` that the template points to). It differs cosmetically from the prose in `cot_template.md` in fixed, corpus-wide ways (PK item order `add,sub,mul,concat`; presence of PK item #6; the BPE-split / "read each equation" prelude; the split §N.4 "narrow the operand domains" + §N.5 "search the remaining unknowns" structure instead of the spec's merged §N.4). These are systematic and intended, so they are not scored as per-file deviations.

Checks performed (all passed across all 69): opening sentence + "Even a symbol…" clause; BPE-split + "read each equation" headers; "convert all the equations to letter form"; prior-knowledge header + items 1–6 verbatim; sign-check phrase where an RHS op-symbol appears; traverse header; §N.1–§N.5 headers (§N.5 legitimately absent when the search resolves with no branching, e.g. crypt_aug_100025); §N.2 concat phrasing; §N.3 prune wording; lock lines `so lock X = …`; reject token `— no` (em-dash, never `≠` — none present); §2 path well-formed (gut-check `This reading order is illegal; skip it completely.` for crypt_aug_100533, else §1 `Need to try reading leftward.`; §2.1–§2.4 headers; illegal-verdict in Summary; `reading order is leftward` clause in the query solve); Conclusion `{…}, X of Y resolved.` (`This is the answer.` on §1-only; bare on §2); `Summary: Confirm reading order = …`; `Now solve the QUERY … applying …`; sign-reattach phrase on every negative answer (8 negative-answer ids, all uniform); two-line suffix with empty `\boxed{}` phrase then `\boxed{ANS}`; exactly one filled `\boxed{}` (== answer text).

| id | score | deviations |
|---|---|---|
| crypt_aug_100002 | 10 | none |
| crypt_aug_100018 | 10 | none |
| crypt_aug_100024 | 10 | none |
| crypt_aug_100025 | 10 | none (§1.5 absent: search resolved in §1.4 with no branching — correct) |
| crypt_aug_100027 | 10 | none |
| crypt_aug_100038 | 10 | none |
| crypt_aug_100046 | 10 | none |
| crypt_aug_100064 | 10 | none |
| crypt_aug_100089 | 10 | none |
| crypt_aug_100094 | 10 | none |
| crypt_aug_100121 | 10 | none |
| crypt_aug_100129 | 10 | none |
| crypt_aug_100138 | 10 | none |
| crypt_aug_100144 | 10 | none |
| crypt_aug_100149 | 10 | none |
| crypt_aug_100166 | 10 | none |
| crypt_aug_100167 | 10 | none |
| crypt_aug_100170 | 10 | none |
| crypt_aug_100172 | 10 | none |
| crypt_aug_100182 | 10 | none |
| crypt_aug_100188 | 10 | none |
| crypt_aug_100191 | 10 | none |
| crypt_aug_100192 | 10 | none |
| crypt_aug_100200 | 10 | none |
| crypt_aug_100202 | 10 | none |
| crypt_aug_100209 | 10 | none |
| crypt_aug_100216 | 10 | none |
| crypt_aug_100218 | 10 | none |
| crypt_aug_100222 | 10 | none |
| crypt_aug_100227 | 10 | none |
| crypt_aug_100229 | 10 | none |
| crypt_aug_100232 | 10 | none |
| crypt_aug_100253 | 10 | none |
| crypt_aug_100263 | 10 | none |
| crypt_aug_100268 | 10 | none |
| crypt_aug_100271 | 10 | none |
| crypt_aug_100276 | 10 | none |
| crypt_aug_100280 | 10 | none |
| crypt_aug_100283 | 10 | none |
| crypt_aug_100296 | 10 | none |
| crypt_aug_100315 | 10 | none |
| crypt_aug_100317 | 10 | none |
| crypt_aug_100321 | 10 | none |
| crypt_aug_100333 | 10 | none |
| crypt_aug_100334 | 10 | none |
| crypt_aug_100354 | 10 | none |
| crypt_aug_100370 | 10 | none |
| crypt_aug_100375 | 10 | none |
| crypt_aug_100387 | 10 | none |
| crypt_aug_100394 | 10 | none |
| crypt_aug_100396 | 10 | none |
| crypt_aug_100397 | 10 | none |
| crypt_aug_100400 | 10 | none |
| crypt_aug_100417 | 10 | none (§2 leftward solve; §1 stuck so 'Need to try reading leftward.' — correct) |
| crypt_aug_100421 | 10 | none (§2 leftward solve; §1 stuck so 'Need to try reading leftward.' — correct) |
| crypt_aug_100427 | 10 | none (§2 leftward solve; §1 stuck so 'Need to try reading leftward.' — correct) |
| crypt_aug_100447 | 10 | none (§2 leftward solve; §1 stuck so 'Need to try reading leftward.' — correct) |
| crypt_aug_100468 | 10 | none (§2 leftward solve; §1 stuck so 'Need to try reading leftward.' — correct) |
| crypt_aug_100469 | 10 | none (§2 leftward solve; §1 stuck so 'Need to try reading leftward.' — correct) |
| crypt_aug_100473 | 10 | none (§2 leftward solve; §1 stuck so 'Need to try reading leftward.' — correct) |
| crypt_aug_100482 | 10 | none (§2 leftward solve; §1 stuck so 'Need to try reading leftward.' — correct) |
| crypt_aug_100500 | 10 | none (§2 leftward solve; §1 stuck so 'Need to try reading leftward.' — correct) |
| crypt_aug_100501 | 10 | none (§2 leftward solve; §1 stuck so 'Need to try reading leftward.' — correct) |
| crypt_aug_100518 | 10 | none (§2 leftward solve; §1 stuck so 'Need to try reading leftward.' — correct) |
| crypt_aug_100533 | 10 | none (§1 ruled illegal by gut-check; §2 leftward is the solve — correct) |
| crypt_aug_100540 | 10 | none (§2 leftward solve; §1 stuck so 'Need to try reading leftward.' — correct) |
| crypt_aug_100541 | 10 | none (§2 leftward solve; §1 stuck so 'Need to try reading leftward.' — correct) |
| crypt_aug_100543 | 10 | none (§2 leftward solve; §1 stuck so 'Need to try reading leftward.' — correct) |
| crypt_aug_100552 | 10 | none (§2 leftward solve; §1 stuck so 'Need to try reading leftward.' — correct) |

## Summary
- Score 10: 69 / 69
- Score 7–9: 0
- Score 4–6: 0
- Score 0–3: 0
- No format/wording deviations found in any CoT.
