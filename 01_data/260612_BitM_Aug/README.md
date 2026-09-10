# 260612_BitM_Aug — bit_manipulation augmentation

> Built 2026-06-13 (Chicago). Pipeline: HOW_TO_AUGMENT_A_NEMOTRON_CATEGORY.md (construct-with-gold →
> blind-solve → GT-match → decorrelate). Reference: 260607_NumericEq_Aug.

## Principle
bitm construction is a PURE function: pick a hidden rule H (reusing the solver's own `word_solve_full`
H-representation), generate random 8-bit example inputs + a query, apply H → outputs + gold. The hidden rule
is never written into question.txt. Blind-solve with `_gen_bitm_faithful.gen_cot` (sees question.txt only);
GT-match = blind solver recovers the planted gold (ambiguous/under-determined puzzles fail and are dropped).

## Multi-axis structure (measured on the 1,602 real puzzles)
- PRIMARY axis = resolution PATH (CoT shape): direct / maj / ch / notwrap / 3term — balanced.
- AXIS = operation (U/AND/OR/XOR/MAJ/CH/T3), intrinsically coupled to path (maj-path needs a Maj rule, etc.).
- AXIS = ambiguity (Pattern-consistency-check fires) — decorrelated toward ~30% per path.
- FEATURE axes (one global knob each, identical per path): operand transform (66% shift-fill0 / 34% rotate;
  fill-1 ~never, mirroring the generator), direction (50/50), NOT-wrap (~16%), #examples (7-10), magnitude (1-7).

## Result
| path | aug | combined (incl. originals) |
|---|---|---|
| direct | 844 | 2,200 |
| maj | 630 | 700 |
| ch | 656 | 700 |
| notwrap | 681 | 700 |
| 3term | 145 | 153 (capped — see below) |
| **aug total** | **2,956** | combined bitm 4,453 (CSV-kept 4,291 after <7680 filter) |

- **GT 2,956/2,956 · sound 2,956/2,956 · foreign markers 0.**
- **3term capped at 145** (not 700): 3-term puzzles are intrinsically hard to mass-synthesize — with random
  examples they are usually under-determined so the blind solver finds a *simpler* rule (GT-match fails), and
  the word-search for 3-term is the slowest path. Used 9-10 examples (vs 7-10) to reach feasible yield; still
  the slowest. 145 is a ~17x boost over the 8 originals.

## Decorrelation proof (combined set, per path)
PCC spread reduced 83pp (raw corpus) → 49pp; direction flat (47-50%); rotate 34-48%. Residuals (reported,
not hidden): 3term is a small-n outlier (69% PCC, mean 9.6 examples — constructed 3-terms came out ambiguous
and 9-10 examples were forced for yield); direct sits at 19% PCC (the un-editable originals are 13%). Among
the four big paths PCC is 19-35% (16pp). NOT-wrap↔notwrap-path is intrinsic coupling (100%/0%), not manufactured.

## Files
- `_gen_aug_bitm.py` — constructor (rand_H, path_of_H, construct) reusing `_gen_bitm` H-machinery
- `_build_aug.py` — driver (pool per path, GT+path filter, PCC-capped stratify, incremental JSONL)
- `_aug_rows.jsonl` — 2,956 aug rows (id, path, pcc, features, question, cot, gold)
- `260612_BitMAug_FULL.csv` — combined training CSV (base 260612_BitMFaithful_FULL + 2,794 aug bitm; non-bitm byte-identical)

## Performance notes (cost a cycle each)
- Writing per-puzzle folders to the SynologyDrive is sync-throttled — write to ONE JSONL on /tmp, copy back.
- `path_of_H`/full_prior used a linear scan over the extended vocab — added `ext_label()` O(1) dict lookup.
- 3-term `word_solve_full` is intrinsically O(72^2·72) — the slow path; cap 3term target accordingly.
