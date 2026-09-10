# Axis distribution — bit_manipulation corpus (the 1,497 original puzzles)

Measured 1,497/1,497 by re-deriving each puzzle's hidden rule (`_gen_bitm.word_solve_full` + `full_prior`)
over its `question.txt` + `track/tree_cot.txt`. bit_manipulation's OWN axes (8-bit word; out = OP of
shifted/rotated/NOT-ed copies of the input). **Solvability kept distinct:** single-clean-rule (one uniform
word op, every per-position label nameable) · composite (a real rule exists but it's a 3-term composition
whose per-position labels can't all be named) · no-rule/patchwork (no single uniform op fits all examples →
each output bit matched independently). Ambiguity = a Pattern-consistency-check fired (≥1 bit had several
example-fitting ops and was pinned by the single global rule).

## Scenario space — full coverage (every scenario, what aug must / must NOT generate) ❗

**Aug is generated from THIS distribution. A scenario that does not exist here must NOT be constructed.**
The earlier aug created 3 CH/MAJ puzzles that fall into composite/NONFAM — a cell that is **0% of the
originals** (see A3×A1 below). That is the artifact this table exists to prevent.

| scenario | corpus | aug must generate |
|---|---|---|
| single-clean-rule | 1,468 (98.1%) | the bulk — clean U/AND/OR/XOR/MAJ/CH puzzles, all per-position labels nameable |
| composite (3-term `T3` only) | 20 (1.3%) | genuine 3-term rules `OP(u, OP(u,u))`; they render as a composite — **only T3 may** |
| no-rule / fallback | 9 (0.6%) | no single op fits → solver states the fallback line, per-bit; synthesize GT-True (gold = solver's own output) |
| ❗ **CH/MAJ → composite/NONFAM** | **0 (0.0%)** | **NEVER — a plain majority/choice rule ALWAYS renders cleanly in the real data; do NOT construct these (the 3-artifact cell)** |

## TARGET distribution — combined 5,000 (APPROVED 2026-06-13) ❗

6 mutually-exclusive scenario buckets (a deliberate curriculum boost of the failure cases composite + fallback,
~3–5× over the corpus; direct stays the largest bucket; the rare paths become trainable but stay rarer). The
GT-False originals (92, dropped from training — 41 fallback + 33 composite + 18 ambiguous) are exactly why the
hard cells were near-0 in training; aug synthesizes GT-True versions (gold = the solver's own blind output).

Buckets are by CoT SHAPE (what the model produces): a MAJ rule that *renders* as all-simple labels is a
`direct`-shaped CoT, not `maj`. Classifying the originals that way: direct 1,224 / maj 60 / ch 51 / notwrap 16
/ composite 137 / fallback 9 (an earlier operation-based count of 1,037/183/119/12 was wrong and made the first
build miss target — caught by the combined-set proof, then corrected).

| bucket | what it is | orig kept | **TARGET %** | target n | aug built | ACHIEVED |
|---|---|---|---|---|---|---|
| direct | one simple rule (AND/OR/XOR/single) | 1,224 | **45%** | 2,250 | 1,026 | 2,250 ✓ |
| maj | one majority rule | 60 | **10%** | 500 | 440 | 500 ✓ |
| ch | one choice rule | 51 | **10%** | 500 | 449 | 500 ✓ |
| notwrap | one rule using NOT-wrapped copies | 16 | **10%** | 500 | 484 | 500 ✓ |
| composite (= 3term) | one 3-step (`T3`) rule | 137 | **15%** | 750 | 613 | 750 ✓ |
| fallback (= no-rule) | no single rule → per-bit | 9 | **10%** | 500 | 491 | 500 ✓ |
| **total** | | **1,497** | **100%** | **5,000** | **3,503** | **5,000 ✓** |

**ACHIEVED 2026-06-13: every bucket on target; secondary features flat across buckets (rotate 34%, direction
54%, n_examples ~25% each); GT/soundness 0 failures / 5,000. See `DISTRIBUTION.md`.**

**Secondary features — injected at the SAME rate across ALL 6 buckets, matching the originals (decorrelated,
§4b): transform shift 66% / rotate 34% · direction L 54% / R 46% · shift fill-1 ≈ 0% · NOT-wrap 14% ·
magnitude 1→7 ≈ 19/18/20/15/10/9/9 · n_examples ≈ 25% each.** No feature may predict the bucket; no forced
corner subtypes; prove flat per-bucket tables on the COMBINED set before shipping.

**Constructor gates:** only `T3` may be composite; CH/MAJ must be single-clean (kills the 3-artifact cell);
fallback built as a per-column mix with gold = solver's blind output; composite + fallback rendered with the
clear write-ups (3-term rule stated · fallback line stated) so they read ≥9, not as silent patchwork.

## A1 rule operation
| value | count | % |
|---|---|---|
| `XOR` | 533 | 35.6% |
| `AND` | 218 | 14.6% |
| `MAJ` | 184 | 12.3% |
| `U` (single unary) | 145 | 9.7% |
| `OR` | 144 | 9.6% |
| `T3` (3-term) | 137 | 9.2% |
| `CH` | 127 | 8.5% |
| `none` (patchwork) | 9 | 0.6% |

## A2 resolution path (CoT shape)
| value | count | % |
|---|---|---|
| `direct` | 1,356 | 90.6% |
| `maj` | 70 | 4.7% |
| `ch` | 44 | 2.9% |
| `notwrap` | 19 | 1.3% |
| `3term` | 8 | 0.5% |

## A3 solvability / renderability
| value | count | % |
|---|---|---|
| `single-clean-rule` | 1,468 | 98.1% |
| `composite/NONFAM` | 20 | 1.3% |
| `no-rule/patchwork` | 9 | 0.6% |

## A3×A1 solvability BY operation ❗ (the axis the artifacts violated)
| operation | single-clean | composite/NONFAM | no-rule |
|---|---|---|---|
| XOR / AND / OR / U | all | 0 | 0 |
| **MAJ** | **184 (100%)** | **0** | 0 |
| **CH** | **127 (100%)** | **0** | 0 |
| T3 | 117 | 20 | 0 |
| none | 0 | 0 | 9 |

**Rule: composite/NONFAM is reachable ONLY from T3; no-rule ONLY as patchwork. A constructed CH or MAJ
puzzle that lands in composite/NONFAM is off-distribution and must be regenerated.**

## A4 operands in the rule
| value | count | % | (operation) |
|---|---|---|---|
| `1` | 145 | 9.7% | U |
| `2` | 895 | 59.8% | AND/OR/XOR |
| `3` | 311 | 20.8% | MAJ/CH |
| `T3 (2 ops + 3 operands)` | 137 | 9.2% | T3 |
| `0` | 9 | 0.6% | patchwork |

## A5 n_examples
| value | count | % |
|---|---|---|
| `7` | 366 | 24.4% |
| `8` | 379 | 25.3% |
| `9` | 384 | 25.7% |
| `10` | 368 | 24.6% |

## A6 operand transform (per operand; 3,279 total)
| value | count | % |
|---|---|---|
| `shift` | 2,155 | 65.7% |
| `rotate` | 1,112 | 33.9% |
| `identity` | 12 | 0.4% |

## A7 direction (per transformed operand; 3,267)
| value | count | % |
|---|---|---|
| `left` | 1,763 | 54.0% |
| `right` | 1,504 | 46.0% |

## A8 shift fill (per shift operand; 2,155) ❗
| value | count | % |
|---|---|---|
| `fill-0` | 2,152 | 99.86% |
| `fill-1` | 3 | 0.14% |

**fill-1 is essentially never used — aug must keep fill-1 ≈ 0 (constructing fill-1 shifts is off-distribution).**

## A9 NOT-wrap (per operand; 3,553)
| value | count | % |
|---|---|---|
| `plain` | 3,052 | 85.9% |
| `NOT-wrapped` | 501 | 14.1% |

## A10 shift/rotate magnitude (per transformed operand; 3,267)
| value | count | % |
|---|---|---|
| `1` | 635 | 19.4% |
| `2` | 596 | 18.2% |
| `3` | 638 | 19.5% |
| `4` | 487 | 14.9% |
| `5` | 322 | 9.9% |
| `6` | 300 | 9.2% |
| `7` | 289 | 8.8% |

## A11 ambiguity (Pattern consistency check fired)
| value | count | % |
|---|---|---|
| `no-PCC` | 1,217 | 81.3% |
| `PCC` | 280 | 18.7% |

## A12 CoT length (characters)
median 8,459 · mean 8,529 · min 7,322 · max 11,044
