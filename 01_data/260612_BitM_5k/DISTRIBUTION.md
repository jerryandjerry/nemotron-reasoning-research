# 260612_BitM_5k — distribution (combined 5,000)

> Rebuilt 2026-06-13 (Chicago) to the approved 6-bucket scenario target (see `AXIS_DISTRIBUTION_1497.md` for
> the originals' axes and the target rationale). 1,497 originals + 3,503 aug. Per-folder labels in
> `axis_manifest.csv` (folder, source, bucket, pcc, fallback, n_examples, rotate, shift, notwrap, dir_L, dir_R, gold).

## Scenario buckets (the primary axis) — ACHIEVED vs target

| bucket | what it is | n | % | target |
|---|---|---|---|---|
| direct | one simple rule (AND/OR/XOR/single) | 2,250 | 45.0% | 45% ✓ |
| maj | one majority rule | 500 | 10.0% | 10% ✓ |
| ch | one choice rule | 500 | 10.0% | 10% ✓ |
| notwrap | one rule using NOT-wrapped copies | 500 | 10.0% | 10% ✓ |
| composite | one 3-step (`T3`) rule — states the rule line | 750 | 15.0% | 15% ✓ |
| fallback | no single rule → per-bit (states the fallback line) | 500 | 10.0% | 10% ✓ |

Direct stays the majority; the rare paths are trainable but stay rarer; composite + fallback (the failure
scenarios that were near-0 in training — they lived mostly on the 92 dropped GT-False originals) are now
covered at ~3–5× the corpus rate, synthesized **GT-True** (gold = the solver's own blind output).

## Secondary features — flat across every bucket (decorrelated, §4b)

| bucket | rotate% | L-dir% | n_ex 7/8/9/10 |
|---|---|---|---|
| direct | 34 | 55 | 24/25/25/25 |
| maj | 34 | 55 | 27/25/25/24 |
| ch | 34 | 54 | 25/25/25/25 |
| notwrap | 33 | 55 | 25/25/25/24 |
| composite | 34 | 54 | 25/26/25/24 |
| fallback | – | – | 25/25/25/25 |
| **originals** | **34** | **54** | even |

`n_examples` — the ONLY feature visible in the question — is **flat ~25% in every bucket**, so it can't predict
the bucket. rotate/direction match the originals' rates. NOT-wrap is intrinsically coupled (a NOT-wrapped
majority rule IS the notwrap bucket) and is a hidden feature anyway. The over-generate + stratified-subsample
build (§4b technique 4) removed the GT/bucket filter's skew (which had pushed rotate to ~42% and skewed maj's
n_examples).

## Integrity
- **GT/soundness: 0 failures / 5,000** — every Selected rule reproduces every example, apply==box==gold.
- **0 off-distribution artifacts** — the CH/MAJ gate rejects the composite-NONFAM cell the originals never occupy.
- composite states `The resolved positions follow a single rule: …`; fallback states `No single rule reproduces
  all examples; each output bit is taken from its own column match.` — no silent patchwork.

## Notes for multi-stage / curriculum
- PRIMARY split axis = `bucket` (easy→hard: direct → maj/ch → notwrap → composite → fallback).
- INTRINSIC couplings (cannot decorrelate): notwrap = NOT-wrapped operands; composite = 3-step rule.
- Build scripts: `01_data/260612_BitM_Aug/_gen_aug_bitm.py` + `_build_aug6s.py` (stratified) + `_build_4.py`.
