# Format & wording conformance audit — slice 8

Axis: format only (template = 260525_Cryptarithm/cot_template.md, as realized by the Aug generation
conventions shared verbatim across every CoT in this slice: BPE-split section, prior-knowledge item 6,
the §N.5 branching step, and lowercase rhs/lhs — these are the de-facto canonical format, not deviations).

Count: 69

| id | score | deviations |
|---|---|---|
| crypt_aug_201682 | 10 | none |
| crypt_aug_201691 | 10 | none |
| crypt_aug_201731 | 10 | none |
| crypt_aug_201743 | 10 | none |
| crypt_aug_201757 | 10 | none |
| crypt_aug_201760 | 10 | none |
| crypt_aug_201767 | 10 | none |
| crypt_aug_201825 | 10 | none |
| crypt_aug_201843 | 10 | none |
| crypt_aug_201850 | 10 | none |
| crypt_aug_201856 | 10 | none |
| crypt_aug_201873 | 10 | none |
| crypt_aug_201879 | 10 | none |
| crypt_aug_201887 | 10 | none |
| crypt_aug_201895 | 10 | none |
| crypt_aug_201900 | 10 | none |
| crypt_aug_201905 | 10 | none |
| crypt_aug_201908 | 10 | none |
| crypt_aug_201919 | 10 | none |
| crypt_aug_201935 | 10 | none |
| crypt_aug_201938 | 10 | none |
| crypt_aug_201972 | 10 | none |
| crypt_aug_201979 | 10 | none |
| crypt_aug_201991 | 10 | none |
| crypt_aug_201993 | 10 | none |
| crypt_aug_201995 | 10 | none |
| crypt_aug_202004 | 10 | none |
| crypt_aug_202007 | 10 | none |
| crypt_aug_202008 | 10 | none |
| crypt_aug_202010 | 10 | none |
| crypt_aug_202014 | 10 | none |
| crypt_aug_202024 | 10 | none |
| crypt_aug_202030 | 10 | none |
| crypt_aug_202045 | 10 | none |
| crypt_aug_202047 | 10 | none |
| crypt_aug_202056 | 10 | none |
| crypt_aug_202061 | 10 | none |
| crypt_aug_202062 | 10 | none |
| crypt_aug_202067 | 10 | none |
| crypt_aug_202071 | 10 | none |
| crypt_aug_202094 | 10 | none |
| crypt_aug_202101 | 10 | none |
| crypt_aug_202104 | 10 | none |
| crypt_aug_202107 | 10 | none |
| crypt_aug_202115 | 10 | none |
| crypt_aug_202135 | 10 | none |
| crypt_aug_202151 | 10 | none |
| crypt_aug_202156 | 10 | none |
| crypt_aug_202188 | 10 | none |
| crypt_aug_202195 | 10 | none |
| crypt_aug_202197 | 10 | none |
| crypt_aug_202206 | 10 | none |
| crypt_aug_202208 | 10 | none |
| crypt_aug_202211 | 10 | none |
| crypt_aug_202215 | 10 | none |
| crypt_aug_202222 | 10 | none |
| crypt_aug_202226 | 10 | none |
| crypt_aug_202241 | 10 | none |
| crypt_aug_202250 | 10 | none |
| crypt_aug_202286 | 10 | none |
| crypt_aug_202332 | 10 | none |
| crypt_aug_202338 | 10 | none |
| crypt_aug_202339 | 10 | none |
| crypt_aug_202347 | 10 | none |
| crypt_aug_202350 | 10 | none |
| crypt_aug_202359 | 10 | none |
| crypt_aug_202364 | 10 | none |
| crypt_aug_202377 | 10 | none |
| crypt_aug_202379 | 10 | none |

## Summary

All 69 CoTs are fully format-conformant (score 10). Every required structural element is present,
ordered, and verbatim where mandated:

- Opening line (incl. the 'Even a symbol that looks like an operator …' clause)
- 'Split the BPE merged tokens', 'Now I read each equation one symbol at a time', the 5-symbol/operator line
- Operator->f/g/h and operand->letter assignment, 'I will convert all the equations to letter form'
- Prior knowledge items 1-6 verbatim (incl. item 6 'never return unknown')
- Sign-check block when an operator appears in an RHS
- §1/§2 with §N.1-§N.5 sub-steps, correct headers; illegal-reading note where applicable
- 'Conclusion of step §N: {…}, X of Y resolved.' with correct continuation ('Need to try reading leftward.' / bare §2)
- Summary verdict ('§2 solved the QUERY operator …, so §2 is preferred.' x55; '§1 is illegal, so §2 is the only legal reading.' x14) + 'Confirm reading order = …'
- Query-answer block ('Now solve the QUERY …', 'applying …', 'map the digits back to symbols'), sign re-attach line where the answer is negative
- Two-line suffix and a single filled final \boxed{ANS} as the last line

Forbidden tokens absent: no '≠'/'!=' (rejects use '— no'), no equation-numeric 'max-mod-min' fallback, no uppercase RHS/LHS.

Note on false positives during automated scan: crypt_aug_201938 (\boxed{}$$}) and crypt_aug_202359 (\boxed{}'})
initially tripped an 'empty-box' heuristic because the answer SYMBOL string literally begins with '}'; manual
inspection confirms both render the correct non-empty answer (}$$ and }' respectively). No deviation.
