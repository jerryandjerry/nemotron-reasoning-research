# Cryptarithm Aug 5k — Distribution & Decorrelation Proof

Selected: **5000** (141 originals @1x + 4859 aug).

## Primary resolution-type mix (target 60/10/10/10/10)

| type | n | share |
|---|---|---|
| arith | 3000 | 60% |
| seen_concat | 500 | 10% |
| qonly_arith | 500 | 10% |
| qonly_concat | 500 | 10% |
| unsolvable | 500 | 10% |

## Per-axis marginals (decorrelation — each axis should be ~flat across types)

### signed (~sub negative form present)

| type | n | 1 |
|---|---|---|
| arith | 3000 | 40% |
| seen_concat | 500 | 38% |
| qonly_arith | 500 | 32% |
| qonly_concat | 500 | 39% |
| unsolvable | 500 | 39% |
| **ALL** | 5000 | **39%** |

_max per-type deviation from overall: **7 pts**_

### operator count

| type | n | 2 | 3 |
|---|---|---|---|
| arith | 3000 | 35% | 64% |
| seen_concat | 500 | 37% | 62% |
| qonly_arith | 500 | 25% | 75% |
| qonly_concat | 500 | 34% | 65% |
| unsolvable | 500 | 40% | 60% |
| **ALL** | 5000 | **35%** | **64%** |

_max per-type deviation from overall: **11 pts**_

### example count

| type | n | 3 | 4 | 5 |
|---|---|---|---|---|
| arith | 3000 | 24% | 38% | 37% |
| seen_concat | 500 | 28% | 35% | 35% |
| qonly_arith | 500 | 24% | 30% | 44% |
| qonly_concat | 500 | 28% | 37% | 34% |
| unsolvable | 500 | 23% | 36% | 40% |
| **ALL** | 5000 | **24%** | **36%** | **38%** |

_max per-type deviation from overall: **6 pts**_

### arithmetic glyph class (+,-,* present)

| type | n | 1 |
|---|---|---|
| arith | 3000 | 97% |
| seen_concat | 500 | 92% |
| qonly_arith | 500 | 97% |
| qonly_concat | 500 | 95% |
| unsolvable | 500 | 97% |
| **ALL** | 5000 | **96%** |

_max per-type deviation from overall: **4 pts**_

## Mutual information I(resolution_type ; axis)  — bits; lower = less predictive

Type entropy H(type) = 1.771 bits (max if axis fully predicted type).

| axis | I(type;axis) bits | % of H(type) |
|---|---|---|
| signed | 0.002 | 0.1% |
| n_ops | 0.005 | 0.3% |
| n_ex | 0.003 | 0.2% |
| glyph-class | 0.005 | 0.3% |

## Unsolvable cause mix (the GT-False-exclusive scenarios, now GT-True via concat default)

| cause | n |
|---|---|
| leftward | 253 |
| exotic | 219 |
|  | 14 |
| illegal | 14 |

## BPE merge-token coverage (462 authoritative real tokens from the original 800)

- floor = 40; **below floor: 0**
- freq: min **40**, p10 40, median 101, max 427
- covers both the question un-merge (example operands/results) and the answer merge.

## CoT length
- tokens: min 962, median 5520, p95 7441, max 7679 (cap 7680; 0 over).

## Structural (non-removable) correlations — documented, not defects
- **qonly_arith** requires a 4-digit rhs to enter the arithmetic route, which only a `~mul` example produces. So a `~sub` (signed) example forces a 2nd family ⇒ qonly_arith signed⇒3-op, lower 2-op share, higher glyph share. These are intrinsic to the resolution logic, not a generator artifact, and cannot be removed without constructing illegitimate puzzles.
- **seen_concat / qonly_concat** carry a concatenation operator (no +,-,* glyph), so their arith-glyph share is structurally a few points below the all-arithmetic types.
All other axes (signed, n_ops, n_ex) are flat across types within a few points and have near-zero mutual information with the resolution type.