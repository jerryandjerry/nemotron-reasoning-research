| id | score | deviations |
|---|---|---|
| crypt10k_009011 | 10 | none |
| crypt10k_009023 | 10 | none |
| crypt10k_009063 | 10 | none |
| crypt10k_009072 | 10 | none |
| crypt10k_009078 | 10 | none |
| crypt10k_009105 | 10 | none |
| crypt10k_009126 | 10 | none |
| crypt10k_009148 | 10 | none |
| crypt10k_009195 | 10 | none |
| crypt10k_009201 | 10 | none |
| crypt10k_009213 | 10 | none |
| crypt10k_009227 | 10 | none |
| crypt10k_009232 | 10 | none |
| crypt10k_009249 | 10 | none |
| crypt10k_009258 | 10 | none |
| crypt10k_009270 | 10 | none |
| crypt10k_009275 | 10 | none |
| crypt10k_009284 | 10 | none |
| crypt10k_009286 | 10 | none |
| crypt10k_009308 | 10 | none |
| crypt10k_009310 | 10 | none |
| crypt10k_009318 | 10 | none |
| crypt10k_009357 | 10 | none |
| crypt10k_009367 | 10 | none |
| crypt10k_009396 | 10 | none |
| crypt10k_009410 | 10 | none |
| crypt10k_009412 | 10 | none |
| crypt10k_009424 | 10 | none |
| crypt10k_009434 | 10 | none |
| crypt10k_009445 | 10 | none |
| crypt10k_009448 | 10 | none |
| crypt10k_009449 | 10 | none |
| crypt10k_009463 | 10 | none |
| crypt10k_009465 | 10 | none |
| crypt10k_009470 | 10 | none |
| crypt10k_009474 | 10 | none |
| crypt10k_009486 | 10 | none |
| crypt10k_009489 | 10 | none |
| crypt10k_009507 | 10 | none |
| crypt10k_009511 | 10 | none |
| crypt10k_009555 | 10 | none |
| crypt10k_009563 | 10 | none |
| crypt10k_009573 | 10 | none |
| crypt10k_009596 | 10 | none |
| crypt10k_009622 | 10 | none |
| crypt10k_009655 | 10 | none |
| crypt10k_009684 | 10 | none |
| crypt10k_009705 | 10 | none |
| crypt10k_009711 | 10 | none |
| crypt10k_009730 | 10 | none |
| crypt10k_009762 | 10 | none |
| crypt10k_009785 | 10 | none |
| crypt10k_009793 | 10 | none |
| crypt10k_009798 | 10 | none |
| crypt10k_009810 | 10 | none |
| crypt10k_009813 | 10 | none |
| crypt10k_009821 | 10 | none |
| crypt10k_009830 | 10 | none |
| crypt10k_009841 | 10 | none |
| crypt10k_009868 | 10 | none |
| crypt10k_009876 | 10 | none |
| crypt10k_009880 | 10 | none |
| crypt10k_009882 | 10 | none |
| crypt10k_009892 | 10 | none |
| crypt10k_009917 | 10 | none |
| crypt10k_009923 | 10 | none |
| crypt10k_009946 | 10 | none |
| crypt10k_009956 | 10 | none |
| crypt10k_009969 | 10 | none |
| crypt10k_009983 | 10 | none |

## Notes

All 70 ids in slice 9 are the `query_unseen_concat` (STEP 2a) short-circuit path:
the QUERY operator never appears in an example, AND no example has a 4-digit rhs,
so the CoT guesses concat and returns the raw symbol concatenation. Per the template
(lines 291-298, 312-313), on this path the §1/§2 search, prior-knowledge deduction,
and digit search are intentionally SKIPPED and a format check must NOT require them.

Format conformance verified on every file:
- Opening prose (line 89) verbatim.
- BPE merge-split table + "Now I read each equation one symbol at a time, using the
  splits above:" (all have merged tokens).
- Operator-count line + "Now denote them as unknown functions:" — count is 2 or 3 in
  this slice, correctly plural ("operators").
- "The other symbols are operands; assign a letter to each in the order of appearance:"
- "I will convert all the equations to letter form:" + letter-form block.
- Prior knowledge items 1-6 verbatim.
- Sign-check clause verbatim where an operator symbol appears in an RHS.
- STEP 1 unseen-note (both lines) verbatim, before the concat guess.
- STEP 2a: "There is no 4-digit rhs in any example, so [LABEL] is most likely
  concatenation. I'll guess [LABEL] = a∥b." + "Now solve the QUERY [LABEL](...),
  applying [LABEL] = a∥b," + value line ending "map the letters back to symbols,
  so the answer is [ANS]".
- Suffix two lines + well-formed final `\boxed{[ANS]}`.

Forbidden tokens absent in all: no variant labels (mul_plus1 / concat_fwd /
concat_rev / sub_signed / absdiff / "variant ="), no `≠`. Concat shown as the
formula `a∥b` only. `\boxed{}` renders the raw symbol concatenation on the last line.

No deviations found.
