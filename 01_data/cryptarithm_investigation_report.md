# Cryptarithm Puzzle Deep Investigation Report

> Date: 2026-05-16
> Scope: ~8M tokens, 30+ sub-agents, 93 ambiguous puzzles exhaustively tested
> Dataset: lkevincc's golden solver results (800/823 solved)

---

## 1. Puzzle Structure

- Format: "In Alice's Wonderland, a secret set of transformation rules is applied to equations."
- Preamble is **identical** across all 725 solved puzzles — no hidden text clues.
- Each puzzle: 3-5 example equations + 1 query. Input is 5 characters: `AB⊕CD` (two 2-symbol operands separated by an operator symbol).
- Symbols map to digits 0-9 via a bijective (or near-bijective) cipher.
- Operator symbols map to one of 22 distinct operations.

## 2. Puzzle Composition

| Type | Count | % |
|---|---|---|
| arithmetic (standard L-to-R) | 315 | 39.4% |
| little_endian (R-to-L) | 276 | 34.5% |
| mixed_concat | 108 | 13.5% |
| pure_concat | 59 | 7.4% |
| mixed_concat_little_endian | 26 | 3.3% |
| query_unseen_concat | 16 | 2.0% |

Reading modes: standard 53%, little-endian 38%, concat-only 9%. No "alice" mode found in solved puzzles.

## 3. Symbol Count Distribution

| Digit Symbols | Puzzles | % of Solved |
|---|---|---|
| 7 | 2 | 0.3% |
| 8 | 47 | 6.5% |
| 9 | 203 | 28.0% |
| 10 | 473 | 65.2% |

All puzzles have radix = number of digit symbols (full permutation within radix). 27 puzzles are base-9, 6 are base-8, 1 is base-7; the rest are base-10.

## 4. Operations

22 distinct operations exist. The 4 dominant ones: add, mul, absdiff, sub_signed.

Operator symbol predictiveness:
- `+` = addition-related 74.4% (add 43.8%, add_m1 16.0%, add_p1 14.6%)
- `*` = multiplication-related 76.7% (mul 47.8%, mul_m1 16.0%, mul_p1 12.9%)
- `-` = subtraction-related 81.3% (absdiff 35.7%, sub_signed 35.3%, neg_absdiff 7.5%)
- Other symbols: random mapping to operations.

## 5. Query Operator Coverage

| Examples of query operator | Puzzles | % |
|---|---|---|
| 0 (unseen) | 161 | 20% |
| 1 | 318 | 40% |
| 2+ | 321 | 40% |

161 puzzles have ZERO examples of the query operator — genuinely unsolvable without guessing the operation.

## 6. Concat Operation Semantics

Concat operates at the **digit-sequence level**, not the numeric level. `concat_fwd(AB, CD) = ABCD` means the digit lists are concatenated, not `int(str(decode(AB)) + str(decode(CD)))`. This is critical for the checker — got 22/31 false negatives before fixing this.

## 7. Uniqueness: Exhaustive Brute-Force Results

Tested 50+ puzzles. For each, fixed operations to ground truth and enumerated **all** P(radix, n) digit permutations.

### By number of digit symbols

| Symbols | Sample | Uniquely Solvable | Rate | Avg Distinct Answers |
|---|---|---|---|---|
| 7 | 3 | 0 | 0% | 71 |
| 8 | 7 | 3 | 43% | 4.6 |
| 9 | 20 | 12 | 60% | 5.4 |
| 10 | 20 | 17 | **85%** | 1.6 |

### By number of examples

| Examples | Sample | Uniquely Solvable | Rate |
|---|---|---|---|
| 3 | 18 | 7 | 39% |
| 4 | 8 | 5 | 63% |
| 5 | 24 | 20 | 83% |

### Why 10-symbol puzzles are mostly unique

With 10 symbols mapped to 10 digits (full bijection), there are zero free variables. The only ambiguity comes from rare permutation symmetries. With 4-5 examples, these are almost always broken.

### Why fewer-symbol puzzles are ambiguous

With n < 10 symbols, (10-n) digits are unused. Permuting these free digits preserves all equation constraints, creating systematic ambiguity. Algebraic proof for one puzzle: 3 unknowns, 2 independent equations → underdetermined. The symmetry `(x+1, y-1, z+1)` preserves all constraints.

### Extreme cases

- One 7-symbol puzzle: 5040 valid mappings, 35 distinct answers
- One 9-symbol puzzle (heavy concat, 3 examples): 510 valid mappings, 174 distinct answers
- One 10-symbol puzzle: 26 valid mappings, 26 distinct answers (outlier)

## 8. The Disambiguation Question

Given that many puzzles are ambiguous, HOW does the ground truth pick one answer? We tested 30+ hypotheses.

### 8.1 Strong finding: GT mapping has minimum digit sum (87-93%)

Among all valid mappings that produce the **same answer** as the ground truth, the GT mapping has the minimum digit sum:
- 8-symbol puzzles: 93% (13/14)
- All <=9-symbol puzzles: 87% (81/93)

This suggests the generator iterates mappings in ascending digit-sum order (consistent with `itertools.permutations(range(radix), n)` which iterates lex, and lex-first correlates with low digit sum).

### 8.2 GT mapping is lex-first for its answer (73%)

Among valid mappings producing the GT answer, the GT mapping is the lexicographically first 73% of the time (symbols sorted by ASCII).

### 8.3 No rule predicts the correct ANSWER

All answer-prediction strategies tested on 93 ambiguous puzzles:

| Strategy | Accuracy |
|---|---|
| Answer of min-digit-sum mapping | 33.3% |
| Answer of min-L2-norm mapping | 33.3% |
| Mode (most frequent answer across mappings) | 32.3% |
| Answer of lex-first mapping | 29.0% |
| Answer of max-digit-sum mapping | 23.7% |
| Median of all valid answers | ~25% |
| Smallest positive answer | 15.1% |
| Closest to zero | 18.3% |

None significantly beat random chance (expected ~10-33% depending on number of valid answers).

### 8.4 Tonghuikang's approach vs lkevincc's

- **Tonghuikang**: collects up to 200 solutions, picks the MODE (most common answer). Requires >30% consensus for non-unique mappings.
- **lkevincc**: tiered operation search (TIER0 → TIER1 → TIER2), first valid permutation wins, conditioned on gold answer for 93% of puzzles.
- Neither approach is a universal disambiguation rule.

## 9. Properties Confirmed to Be Random

### 9.1 ASCII-based metrics (0% predictive)
- ASCII value vs digit: Spearman r = -0.002 (p = 0.96)
- ASCII rank vs digit: r = -0.002
- First-appearance order vs digit: r = -0.029 (p = 0.36)
- Frequency vs digit: r = -0.040 (p = 0.23)

### 9.2 Modular arithmetic (0/725 matches)
- (digit + ASCII) mod 10: constant within puzzle? **0/725**
- (digit - ASCII) mod 10: constant? **0/725**
- digit XOR (ASCII mod 10): constant? **0/725**
- digit = (ASCII * k) mod 10 for k=1..9: **0 matches for any k**
- digit = (ASCII * k + c) mod 11 for all k,c: **0 matches**

### 9.3 Hash/seed-based (0/725 matches)
- md5(puzzle_id + symbol) mod radix: 11.3% (random = 10%)
- sha256(puzzle_id + symbol) mod radix: 9.9%
- Puzzle ID as Python `random.seed()` → `random.sample()`: **0/725**
- Puzzle ID as NumPy `RandomState` seed: **0/50**
- MD5/SHA1 of puzzle ID as seed: **0/725**
- Tried 3 seed derivations × 3 symbol orderings × 2 PRNG libraries: **0/50**

### 9.4 Permutation structure
- Parity: 48.5% even, 51.5% odd (random = 50/50)
- Permutation rank (Lehmer code): uniformly distributed across [0, N!)
- Cycle structure: no pattern

### 9.5 Visual similarity
- `|` maps to 1 only 10.1% (random = 10%)
- `!` maps to 1 only 9.7%
- Chi-squared test: **no symbol has a statistically significant digit preference** (all p > 0.12)

### 9.6 Keyboard/encoding patterns
- Keyboard row position vs digit: random
- Gray code, reverse bits, complement, popcount, XOR 0x55: **0 matches**
- 20+ sorting criteria tested: all produce **0 exact matches**

### 9.7 Unused digit (9-symbol base-10 puzzles, N=176)
- Digit 0 is unused 29% of the time (artifact: 0 can't lead multi-digit numbers easily)
- Digit 1 is almost never unused (0.6%)
- Distribution is non-uniform but NOT predictable from puzzle metadata
- puzzle_id mod 10 ≠ unused digit (random match rate)

### 9.8 Leading zero constraint
- 24% of 9-symbol puzzles have digit 0 assigned to a leading character
- NOT enforced by the generator

### 9.9 Example ordering
- Preamble text identical across all puzzles
- 62% have progressive symbol introduction (decreasing new symbols per equation) — weak structural artifact, not a disambiguation rule
- Reversing example order does NOT change which mapping is found first (lex iteration is independent of example order)

## 10. Computation Trace Analysis

For specific ambiguous puzzles, compared GT mapping vs alternative mappings on:
- Number of carries in additions: **identical** across all valid mappings for same ops
- Overflow (2-digit → 3-digit): **identical**
- Leading zeros in operands: **identical**
- Digit coverage (how many of 0-9 used): **identical** (or differs by definition of unused digit)

No "naturalness" metric distinguishes the GT computation trace from alternatives.

## 11. Multi-Operation Analysis

For puzzles ambiguous under GT ops, tried varying the operation assignment:
- Only 1-2 TIER0 op combos typically satisfy all examples (most puzzles use TIER2 ops like add_m1)
- The GT answer does NOT appear in more op combos than alternative answers
- Intersection of valid answer sets across all op combos: **empty** (no answer is valid under ALL op combos)

## 12. How the Puzzle Generator Works (Inferred)

Based on all evidence:

1. Generator picks a **random bijective mapping** (symbols → digits) using an internal PRNG with unknown seed.
2. Generator picks random operations for each operator symbol.
3. Generator constructs example equations that are consistent with the chosen mapping + operations.
4. Generator records the query answer under the chosen mapping.
5. **No uniqueness filter is applied.** The generator does not check whether other mappings also satisfy the equations.
6. The mapping tends to have a low digit sum (87% min-dsum-for-answer), suggesting the PRNG or iteration order favors lower-valued permutations.
7. NVIDIA has not released the generation code. The NeMo Gym and NeMo-Skills repos do not contain this specific puzzle generator.

## 13. External Corroboration

- **Kaggle Discussion #694556 (toolazyhhh123)**: Proved via Lagrange interpolation that puzzles are underdetermined.
- **Kaggle Discussion #688461 (Donald Galliano III)**: Reverse-engineered all puzzle types, confirmed 784 possible (ordering × operation × format) combinations for equation puzzles.
- **lkevincc's solver**: Explicitly comments "Offset/polynomial ops are generic enough that they can spuriously fit multiple permutations; restricting the op pool first breaks ambiguity in favour of the canonical answer."
- **reasoning-gym PR #517**: Fixed their cryptarithm scorer to accept ANY valid solution, acknowledging multiple valid mappings exist.
- **KOR-Bench**: Separate academic benchmark with different puzzle types (classical ciphers), not directly related.

## 14. Summary Table

| Question | Answer | Confidence |
|---|---|---|
| Are most puzzles uniquely solvable? | Yes, 85% of 10-symbol puzzles | High (exhaustive) |
| Is there a hidden rule for the mapping? | No | Very high (0/725 on 20+ metrics) |
| Is the mapping random? | Yes | Very high |
| Can the answer be predicted for ambiguous puzzles? | No better than ~33% | High (93 puzzles, 30+ strategies) |
| Is the puzzle ID a PRNG seed? | No | Very high (0 matches, 6 derivations) |
| Does example order matter? | No | High |
| Does the GT mapping have special properties? | Min digit sum for its answer (87%) | High |
| Are operator symbols predictive? | Weakly: +≈add 74%, *≈mul 77%, -≈sub 81% | High (725 puzzles) |
| Is concat numeric or symbolic? | Symbolic (digit-sequence level) | Confirmed by solver code |
| Does the generator ensure uniqueness? | No | High (93% conditioned on answer) |

## 15. Implications for Competition Strategy

1. **SFT on golden CoT is sound** for 85% of 10-symbol and 60-83% of other puzzles.
2. **For ambiguous puzzles**, train on GT answer anyway — the model learns the distribution, and at inference any valid answer should score.
3. **Tiered operation search** (TIER0 first) is the correct approach for operation disambiguation.
4. **20% of puzzles with unseen query operators** are genuinely hard — the model must learn to guess operations from symbol statistics.
5. **Oversampling ambiguous puzzles may not help** since the GT answer is arbitrary among valid options.
6. **Evaluation metric may accept multiple valid answers** (reasoning-gym already does this) — worth verifying for this competition.
