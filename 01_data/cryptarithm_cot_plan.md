# Plan: Honest Deterministic CoT for Cryptarithm Puzzles

## Reality Check: What CAN and CANNOT be determined

### Can determine (~100%):
- Which characters are operators (position 2 in each 5-char input)
- Whether an operator is concat (output symbols = input symbols concatenated)

### Can determine (~80-90% when query op is seen):
- What arithmetic operation an operator symbol represents, IF we have examples AND the operation is TIER0
- Reading mode (standard vs little-endian), by trying both and checking consistency

### CANNOT determine (the hard parts):
- The digit mapping when the system is underdetermined (multiple valid mappings)
- The operation when it's TIER2 (add_m1, mul_p1) — TIER0 search finds a different mapping that also works
- The operation when the query operator has 0 examples (20% of puzzles)

### Difficulty tiers (725 solved puzzles):

| Tier | Count | % | What's known | What's not |
|---|---|---|---|---|
| EASY | 285 | 39% | TIER0 op seen, 10 symbols | ~15% still ambiguous mapping |
| MEDIUM | 161 | 22% | TIER0 op seen, <10 symbols | 40-57% ambiguous mapping |
| HARD | 134 | 18% | Op is TIER2 (seen in examples) | TIER0 search gives wrong answer |
| HARDEST | 145 | 20% | Query op unseen (0 examples) | Must guess op + mapping |

## The Honest CoT Structure

### Phase 1: Parse and identify (always works)

```
The puzzle has these operator symbols: [list them]
Digit symbols (sorted): [list them]
Number of digit symbols: N, so base = N (radix)
```

### Phase 2: Detect concatenation (always works)

```
Check each example for concatenation:
  Example K: AB⊕CD = ABCD (or CDAB)?
  If output symbols = input symbols joined → ⊕ is concat_fwd (or concat_rev)
```

This is 100% reliable because concat operates at the symbol level — no digit assignment needed.

### Phase 3: Determine arithmetic operations (works for TIER0, 62% of puzzles)

For each non-concat operator, try the most common operations against the examples:

```
For operator '+', try add (most likely for '+'):
  Assign letters: let >=A, #=B, ...
  Example 1: (10A+B) + (10A+C) = (10B+D)  →  20A - 9B + C - D = 0
  Example 2: (10D+E) + (10F+G) = (100A+10C+F) → ...
  
  Try standard mode: solve system of equations
  Try little-endian mode: solve system of equations
  
  If exactly one (operation, mode) pair produces a consistent solution → determined
  If multiple → need disambiguation
```

Key insight: when trying `add` vs `add_m1` vs `add_p1`, the `add` version will ALSO find valid mappings even when the true operation is `add_m1`, but with a DIFFERENT mapping. So TIER0 search gives a WRONG answer for 18% of puzzles.

### Phase 4: Solve the constraint system (works when system is determined)

For 10-symbol puzzles with enough examples:
```
From the equations, we have N linear constraints on 10 unknowns.
With all digits 0-9 used exactly once (bijection), the system is often determined.

Solve by substitution/elimination:
  From eq1: D = 20A - 9B + C
  From eq2: ... 
  Combined with digits ∈ {0-9}, all distinct → unique solution (85% of the time)
```

For <10 symbol puzzles:
```
N equations, M unknowns (M < 10), (10-M) unused digits → underdetermined.
Multiple valid mappings exist.
```

### Phase 5: When ambiguous — what CAN we do?

This is the critical question. Based on our investigation:

**For operation ambiguity (TIER0 vs TIER2):**
- If `add` works, check if `add_m1` or `add_p1` also works with a different mapping
- The model should try the simplest operation first (this is what all solvers do)
- For training: teach the model the TIER0 answer even if GT uses TIER2 — the model can't distinguish them

**For mapping ambiguity:**
- The GT mapping tends to have min digit sum (87% of the time)
- But we found no rule for WHICH ANSWER to pick (~33% accuracy for all strategies)
- For training: teach the model to find ANY valid mapping and compute the answer

**For unseen query operators (20%):**
- The model must guess based on operator symbol statistics:
  - `+` ≈ add (74%), `*` ≈ mul (77%), `-` ≈ subtract (81%)
  - Other symbols: assume add (most common overall)

## Proposed CoT Format

### For a well-determined puzzle (EASY tier):

```
Step 1: Parse the puzzle.
Operators: + (appears in examples 1,2,4), ^ (appears in example 3)
Digit symbols: #, $, %, ', /, <, >, ?, | (9 symbols, base 10)

Step 2: Check for concatenation.
Example 3: |/^%| = |/%|
  LHS: [|,/] ^ [%,|], RHS: [|,/,%,|]
  [|,/] ++ [%,|] = [|,/,%,|] ✓ → ^ = forward concatenation

Step 3: Determine arithmetic operation for +.
Try + = addition, standard (left-to-right) reading:

  「>#+>|」=「#%」: let >=A,#=B → (10A+B)+(10A+C)=10B+D → 20A-9B+C=D
  「%?+$<」=「>|$」: (10D+E)+(10F+G)=100A+10C+F → ...
  「%?+'#」=「>#/」: (10D+E)+(10H+B)=100A+10B+I → ...

Step 4: Solve the constraint system.
  From equation 1: D = 20A - 9B + C
  All values must be distinct digits 0-9.
  Testing A=1: D = 20 - 9B + C. For B=2: D = 2+C, so D-C=2.
  [... continue solving ...]
  
  Found: >=1, #=2, |=3, %=5, ?=8, $=7, <=9, /=0, '=6
  Verify all examples: ✓

Step 5: Compute query answer.
  '>+>$ → [',>] + [>,\$] = [6,1] + [1,7] = 61+17 = 78
  78 → digits [7,8] → symbols [$,?] → $?

\boxed{$?}
```

### For an ambiguous puzzle (MEDIUM/HARD tier):

```
Step 1-3: [same as above]

Step 4: Solve the constraint system.
  From the equations, I get: [constraints]
  This system has multiple valid solutions:
    Solution A: [mapping] → answer = X
    Solution B: [mapping] → answer = Y
  
  Both solutions verify against all examples.

Step 5: Select the most likely answer.
  Solution A uses the simpler operation (addition vs addition+1).
  → Select Solution A.

\boxed{X}
```

### For unseen query operator (HARDEST tier):

```
Step 1: Parse the puzzle.
  Operators: +, *, - (all appear in examples), @ (only in query)
  
Step 2-4: [Determine operations for +, *, - from examples]
  + = addition, * = multiplication, - = absolute difference

Step 5: Guess query operator @.
  @ has no examples. Most common operations: add (53%), mul (54%), absdiff (35%).
  Given the other operators already cover add, mul, absdiff,
  try remaining common operations: sub_signed, concat_fwd.
  [Test both and pick the one that produces an encodeable answer]

\boxed{...}
```

## What Needs to Be Built

1. **New CoT generator** that follows the honest reasoning trace above
   - Input: puzzle prompt + solver metadata (mapping, ops, mode)
   - Output: step-by-step CoT showing how the solution was derived
   - The CoT should show the REASONING, not just state the answer

2. **Tiered CoT templates** based on difficulty:
   - EASY: full derivation, unique answer
   - MEDIUM: derivation + "multiple solutions exist, selecting simplest"
   - HARD: derivation + "trying alternative operations" 
   - HARDEST: derivation + "guessing query operation based on statistics"

3. **Training strategy**:
   - EASY (39%): standard SFT, CoT is correct and derivable
   - MEDIUM (22%): SFT with honest "ambiguous" acknowledgment
   - HARD (18%): SFT showing TIER0→TIER2 escalation
   - HARDEST (20%): SFT showing statistical guessing, accept lower accuracy

## Key Constraint: Model Can't Enumerate

The model can NOT do brute-force search over 3.6M permutations. The CoT must use:
- Algebraic reasoning (substitute, eliminate)
- Constraint propagation (this digit must be X because...)
- Pattern matching (output has 3 digits → result > 99)
- Carry analysis (ones column: A+B mod 10 = C → limits possibilities)

This means the CoT generator must reverse-engineer the GT mapping into a plausible deductive chain, even when the actual solver used brute force.
