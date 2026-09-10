# Format & Wording Conformance Audit — slice_6

Axis: format. 69 puzzles. Audited against the cryptarithm CoT template (260525/cot_template.md)
as realized in the augmented "Alice's Wonderland" variant (BPE-split + third-symbol-is-operator +
prior item 6 + §1.4 narrow / §1.5 branch split) that every CoT in this slice uses uniformly.

Checks performed (format/wording only; reasoning/math NOT judged):
- Opening prose verbatim; "Split the BPE merged tokens" section; letter-form conversion.
- Prior knowledge block present incl. item ordering (~add, ~sub, ~mul, ~concat) and item 6.
- Sign-check wording when an operator symbol appears in an RHS.
- Traverse line, §N.1 writing equations, §N.2 concat check, §N.3 digit-count prune,
  §N.4 narrow domains, §N.5 branch search (when branching occurs).
- Lock wording: every `so lock X = FORMULA`; ~sub try-each `every equation checks out, so lock`;
  single-survivor `[r] = [EXPR], so lock`.
- §1.3 phrasings: `Only ~X survives ... Try ...`, `By the distinct operator rule ... remaining operation`,
  SOFT `Surviving candidates ... Try ... first (...)`.
- Reject token `— no` (never `≠`); Current State lines.
- Conclusion of step line (`X of Y resolved` + `This is the answer.`); Summary line.
- Query block: `Now solve the QUERY`, `applying LABEL(a, b) = FORMULA,`, `map the digits back to symbols`,
  sign re-attach line for negative answers.
- Suffix two lines verbatim + exactly one `\boxed{}` payload matching the answer line.

Result: every CoT conforms on all structural and verbatim-phrase checks. No `≠`, no ASCII-x for
multiplication, no placeholders, no missing sections. Negative answers (200086, 200150, 200219,
200607) correctly carry the sign re-attach line and boxed sign. Cases without §1.5 (200086, 200384,
200455, 200527, 200556, 200811) legitimately resolved all unknowns inside §1.4 narrowing, so the
branch section is correctly absent.

| id | score | deviations |
|---|---|---|
| crypt_aug_200086 | 10 | none |
| crypt_aug_200106 | 10 | none |
| crypt_aug_200111 | 10 | none |
| crypt_aug_200119 | 10 | none |
| crypt_aug_200146 | 10 | none |
| crypt_aug_200150 | 10 | none |
| crypt_aug_200175 | 10 | none |
| crypt_aug_200189 | 10 | none |
| crypt_aug_200200 | 10 | none |
| crypt_aug_200217 | 10 | none |
| crypt_aug_200219 | 10 | none |
| crypt_aug_200220 | 10 | none |
| crypt_aug_200234 | 10 | none |
| crypt_aug_200245 | 10 | none |
| crypt_aug_200248 | 10 | none |
| crypt_aug_200258 | 10 | none |
| crypt_aug_200270 | 10 | none |
| crypt_aug_200273 | 10 | none |
| crypt_aug_200281 | 10 | none |
| crypt_aug_200323 | 10 | none |
| crypt_aug_200334 | 10 | none |
| crypt_aug_200339 | 10 | none |
| crypt_aug_200358 | 10 | none |
| crypt_aug_200359 | 10 | none |
| crypt_aug_200360 | 10 | none |
| crypt_aug_200371 | 10 | none |
| crypt_aug_200372 | 10 | none |
| crypt_aug_200378 | 10 | none |
| crypt_aug_200384 | 10 | none |
| crypt_aug_200386 | 10 | none |
| crypt_aug_200388 | 10 | none |
| crypt_aug_200399 | 10 | none |
| crypt_aug_200402 | 10 | none |
| crypt_aug_200409 | 10 | none |
| crypt_aug_200414 | 10 | none |
| crypt_aug_200422 | 10 | none |
| crypt_aug_200436 | 10 | none |
| crypt_aug_200447 | 10 | none |
| crypt_aug_200455 | 10 | none |
| crypt_aug_200462 | 10 | none |
| crypt_aug_200477 | 10 | none |
| crypt_aug_200487 | 10 | none |
| crypt_aug_200498 | 10 | none |
| crypt_aug_200509 | 10 | none |
| crypt_aug_200510 | 10 | none |
| crypt_aug_200513 | 10 | none |
| crypt_aug_200527 | 10 | none |
| crypt_aug_200546 | 10 | none |
| crypt_aug_200556 | 10 | none |
| crypt_aug_200569 | 10 | none |
| crypt_aug_200575 | 10 | none |
| crypt_aug_200596 | 10 | none |
| crypt_aug_200607 | 10 | none |
| crypt_aug_200624 | 10 | none |
| crypt_aug_200631 | 10 | none |
| crypt_aug_200657 | 10 | none |
| crypt_aug_200673 | 10 | none |
| crypt_aug_200700 | 10 | none |
| crypt_aug_200722 | 10 | none |
| crypt_aug_200737 | 10 | none |
| crypt_aug_200758 | 10 | none |
| crypt_aug_200759 | 10 | none |
| crypt_aug_200763 | 10 | none |
| crypt_aug_200771 | 10 | none |
| crypt_aug_200797 | 10 | none |
| crypt_aug_200811 | 10 | none |
| crypt_aug_200843 | 10 | none |
| crypt_aug_200846 | 10 | none |
| crypt_aug_200876 | 10 | none |

## Slice-level note (applies uniformly to all 69, NOT a per-file deviation)
The slice uses the augmented "Alice's Wonderland" template, which differs from the base
260525/cot_template.md in several conventions applied consistently to every CoT:
- prior-knowledge item-1 order is `~add, ~sub, ~mul, ~concat` (base lists `~mul` first);
- an extra prior item 6 ("never return unknown / best-guess");
- §1.4 "narrow the operand domains" and §1.5 "search by branching" are split (base folds both into §N.4);
- SOFT pick reads `Try X = ~Y first (~Y comes first in the operation order).` (base mandates the
  bare `Try [LABEL] = [~X] first.` with no order mention);
- opening prose adds the "even a symbol that looks like an operator ... third symbol is the operator"
  clause and a "Split the BPE merged tokens" pre-section.
These are template-version differences (a spec mismatch between the base file and the augmented
generator), not authoring errors, so they are not scored against individual files.
