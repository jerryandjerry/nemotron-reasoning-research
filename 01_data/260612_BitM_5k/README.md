# 260612_BitM_5k — 5,000 bit_manipulation puzzles, one per folder

Each `<name>/` holds: `question.txt` (the only solver input), `answer.txt` (gold),
`track/tree_cot.txt` (the faithful forward-only CoT). Total 5,000 = 1,497 originals + 3,503 augmentation.

- `bitm_<id>/` = original provider puzzles (faithful solver re-derivation; see 01_data/260612_BitM/)
- `aug_<id>/`  = synthetic augmentation (construct-with-gold → blind-solve → GT-match; see 01_data/260612_BitM_Aug/)

All 5,000: GT-true, sound (every Selected rule reproduces all examples; boxed = rules applied), 0 foreign
markers — **0 GT/soundness failures / 5,000.** Every aug puzzle is GT-True by construction (gold = the solver's
own blind output); the CH/MAJ gate rejects the off-distribution composite-NONFAM cell the originals never occupy.

## Distribution (6-bucket scenario × source) — rebuilt 2026-06-13
| bucket | total | (orig / aug) | % |
|---|---|---|---|
| direct (one simple rule)        | 2,250 | 1,224 / 1,026 | 45% |
| maj (one majority rule)         |   500 |    60 /   440 | 10% |
| ch (one choice rule)            |   500 |    51 /   449 | 10% |
| notwrap (NOT-wrapped copies)    |   500 |    16 /   484 | 10% |
| composite (one 3-step rule)     |   750 |   137 /   613 | 15% |
| fallback (no single rule)       |   500 |     9 /   491 | 10% |

composite + fallback (the failure scenarios that were near-0 in training — they lived mostly on the 92 dropped
GT-False originals) are now covered at ~3–5× the corpus rate. Secondary features (rotate 34%, direction 54%,
n_examples flat ~25%) are matched to the originals and **flat across every bucket** (over-generate +
stratified-subsample, §4b technique 4) so nothing leaks the bucket. composite states a 3-step rule line;
fallback states `No single rule reproduces all examples; …`.

## Reports for multi-stage / curriculum training
- `DISTRIBUTION.md` — achieved 6-bucket distribution + per-bucket decorrelation proof + integrity summary.
- `AXIS_DISTRIBUTION_1497.md` — the originals' full axis distribution + the target rationale.
- `axis_manifest.csv` — per-folder labels (folder, source, bucket, pcc, fallback, n_examples, rotate, shift,
  notwrap, dir_L, dir_R, gold). PRIMARY split axis = `bucket` (easy→hard: direct → maj/ch → notwrap →
  composite → fallback) for staging a curriculum.

Audit: `AUDIT_260613.md` — 20-agent cold audit (pre-rebuild corpus); re-audit after this rebuild is pending.

Build scripts: 01_data/260612_BitM_Aug/_gen_aug_bitm.py (constructor) + _build_aug6s.py (stratified driver) + _build_4.py.
