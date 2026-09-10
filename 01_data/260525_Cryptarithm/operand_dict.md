# Operand Symbol -> Digit Dictionary

## Operand alphabet (23 symbols, ASCII order)

```
! " # $ % & ' ( ) / : < > ? @ [ \ ] ^ ` { | }
```

These 23 are the **only** symbols that can appear as an operand (digit). `+`, `-`, `*` are
operator-only and never serve as operands. Each puzzle uses exactly 10 of these 23. ASCII codes:

`!`=33  `"`=34  `#`=35  `$`=36  `%`=37  `&`=38  `'`=39  `(`=40  `)`=41  `/`=47  `:`=58  `<`=60  `>`=62  `?`=63  `@`=64  `[`=91  `\`=92  `]`=93  `^`=94  `` ` ``=96  `{`=123  `|`=124  `}`=125

---

The operand counterpart of `operator_dict.md`. For each digit-symbol, the distribution of
which **digit** it stood for across the golden puzzles.
Source: the `Digit mapping` block of each puzzle's `cot_ikev.txt` -- the ikev CoT is built
**backwards from the gold answer**, so its mapping is the generator's true label (the same
provenance as `operator_dict.md`). NOT taken from our solver's `tree_solution.txt`, which is
only what our search found. Coverage: 725 puzzles that carry a digit mapping (the
pure-concat / query-unseen-concat puzzles use no digit values, so they are excluded).

**Key difference from the operator dict:** operators have real priors (`+`->add, `*`->mul,
`-`->subtraction). Operands have **none**. Each puzzle draws 10 of these 23 symbols and
assigns them to digits 0-9 by an independent uniform-random bijection, so every symbol
represents every digit at ~equal frequency. The chi-square against uniform 0-9 is small: only
1 of 23 symbols (`/`) edges past the uncorrected p=0.05 value 16.92 -- the ~1 expected by chance
across 23 tests -- and none passes p=0.01 (21.67). This file is the
evidence base for the solver's missing-digit rule: when the answer needs a digit whose
symbol never appears in a puzzle, **there is no recoverable or preferable symbol** -- the
distribution is flat -- so the solver guesses the first unused non-operator symbol purely to
stay in the game (~7%), never claiming a derivation.

`+`, `-`, `*` never appear as operands (operator-only), so the digit alphabet is exactly
these 23 symbols. Tables are sorted by digit 0-9 (frequency ordering would be pure noise).
Note: ikev's mapping lists only symbols present in the EXAMPLES, so the unseen-digit symbol
of a missing-digit puzzle (knowable only from the gold answer string) is not counted here;
it would not change the uniform picture.

---

**Corpus:** 725 puzzles, 6842 symbol->digit observations, 23 digit-symbols (per-symbol totals 271-317). Largest per-symbol chi-square = 17.9 (`/`); only 1 of 23 exceeds the
p=0.05 value 16.92 (~1.2 expected by chance), none exceeds p=0.01 (21.67): **uniform, no prior.**

---

## Missing-digit puzzles: the unseen symbol (and how the solver guesses it)

101 of the 800 puzzles have a query answer that needs a digit which never appears in the
examples, so its symbol was never observed (the solver reaches a guess on 97; the other 4 are
unsolvable for unrelated reasons). Distribution of the **true unseen symbol** (read from the gold
answer) over the 105 observations (97 puzzles contribute 1, 4 contribute 2), ASCII order:

```
symbol:  !  "  #  $  %  &  '  (  )  /  :  <  >  ?  @  [  \  ]  ^  `  {  |  }
count :  3  4  3  4  6  3  5  5  4  9  6  7  4  4  3  6  3  4  4  3  4  2  9
```

Uniform-expected = 4.6 per symbol; **chi-square = 16.6 (df=22, p=0.05 critical 33.92) -> uniform,
no usable prior.** The most frequent unseen symbols (`/`, `}` at 9) are within noise of 4.6, so
guessing a "most-common" symbol cannot beat any other pick. (This matches the per-symbol tables
above and the cross-puzzle cipher study: the per-puzzle symbol set is a uniform random 10-of-23
draw, so neither the digit a symbol takes nor *which* symbol is the unseen one carries a prior.)

**Solver rule (`_honest_solver._fill_missing`):** never emit the `_` placeholder (a guaranteed-
wrong answer, 0%). Instead print the operand alphabet, then the same alphabet with every symbol
that appears in this puzzle masked as `X`, then take the **first remaining candidate** (the first
operand symbol in ASCII order that the puzzle does not use). Rationale: this is a deterministic
rule over *visible* features (which symbols appear), so the model can learn it from the masked
row and reproduce it at test time; a seeded-random pick would be unlearnable noise, since the
seed isn't in the prompt. Expected accuracy = 1/|candidates| ~ 7% (avg ~13.5 candidates), which
is the ceiling because the unseen symbol is uniform. Lifts GT-match from the `_` floor (0% on
these) — corpus GT-match 454/800.

A positional heuristic (prefer a candidate not ASCII-adjacent to a used `X`, then drop any whose
bracket counterpart is used) was tested and **rejected**: 4.5% < 7.4% random baseline; the true
symbol lands in the run-interior 23.9% vs 27.3% expected by chance, i.e. no signal. Do not narrate
adjacency reasoning — it would be both false and lower-scoring.

---

## `>`  (total: 317)

*operand > has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 2.0, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 35 | 11.0% |
| 1 | 28 | 8.8% |
| 2 | 34 | 10.7% |
| 3 | 29 | 9.1% |
| 4 | 33 | 10.4% |
| 5 | 35 | 11.0% |
| 6 | 31 | 9.8% |
| 7 | 33 | 10.4% |
| 8 | 30 | 9.5% |
| 9 | 29 | 9.1% |

## `!`  (total: 315)

*operand ! has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 15.4, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 22 | 7.0% |
| 1 | 31 | 9.8% |
| 2 | 45 | 14.3% |
| 3 | 33 | 10.5% |
| 4 | 40 | 12.7% |
| 5 | 30 | 9.5% |
| 6 | 30 | 9.5% |
| 7 | 32 | 10.2% |
| 8 | 20 | 6.3% |
| 9 | 32 | 10.2% |

## `/`  (total: 313)

*operand / has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 17.9, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 35 | 11.2% |
| 1 | 30 | 9.6% |
| 2 | 39 | 12.5% |
| 3 | 41 | 13.1% |
| 4 | 23 | 7.3% |
| 5 | 28 | 8.9% |
| 6 | 22 | 7.0% |
| 7 | 30 | 9.6% |
| 8 | 43 | 13.7% |
| 9 | 22 | 7.0% |

## `&`  (total: 312)

*operand & has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 6.3, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 24 | 7.7% |
| 1 | 33 | 10.6% |
| 2 | 40 | 12.8% |
| 3 | 29 | 9.3% |
| 4 | 33 | 10.6% |
| 5 | 31 | 9.9% |
| 6 | 32 | 10.3% |
| 7 | 35 | 11.2% |
| 8 | 25 | 8.0% |
| 9 | 30 | 9.6% |

## `"`  (total: 309)

*operand " has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 5.3, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 23 | 7.4% |
| 1 | 33 | 10.7% |
| 2 | 32 | 10.4% |
| 3 | 35 | 11.3% |
| 4 | 25 | 8.1% |
| 5 | 32 | 10.4% |
| 6 | 31 | 10.0% |
| 7 | 37 | 12.0% |
| 8 | 29 | 9.4% |
| 9 | 32 | 10.4% |

## `#`  (total: 309)

*operand # has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 6.0, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 27 | 8.7% |
| 1 | 28 | 9.1% |
| 2 | 29 | 9.4% |
| 3 | 36 | 11.7% |
| 4 | 41 | 13.3% |
| 5 | 28 | 9.1% |
| 6 | 32 | 10.4% |
| 7 | 32 | 10.4% |
| 8 | 27 | 8.7% |
| 9 | 29 | 9.4% |

## `|`  (total: 304)

*operand | has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 7.6, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 27 | 8.9% |
| 1 | 31 | 10.2% |
| 2 | 39 | 12.8% |
| 3 | 31 | 10.2% |
| 4 | 22 | 7.2% |
| 5 | 33 | 10.9% |
| 6 | 30 | 9.9% |
| 7 | 26 | 8.6% |
| 8 | 28 | 9.2% |
| 9 | 37 | 12.2% |

## `[`  (total: 302)

*operand [ has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 9.9, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 26 | 8.6% |
| 1 | 40 | 13.2% |
| 2 | 29 | 9.6% |
| 3 | 32 | 10.6% |
| 4 | 36 | 11.9% |
| 5 | 32 | 10.6% |
| 6 | 21 | 7.0% |
| 7 | 34 | 11.3% |
| 8 | 24 | 7.9% |
| 9 | 28 | 9.3% |

## `@`  (total: 300)

*operand @ has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 9.3, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 28 | 9.3% |
| 1 | 37 | 12.3% |
| 2 | 25 | 8.3% |
| 3 | 36 | 12.0% |
| 4 | 32 | 10.7% |
| 5 | 25 | 8.3% |
| 6 | 29 | 9.7% |
| 7 | 33 | 11.0% |
| 8 | 35 | 11.7% |
| 9 | 20 | 6.7% |

## `^`  (total: 300)

*operand ^ has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 3.7, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 32 | 10.7% |
| 1 | 25 | 8.3% |
| 2 | 30 | 10.0% |
| 3 | 31 | 10.3% |
| 4 | 30 | 10.0% |
| 5 | 29 | 9.7% |
| 6 | 38 | 12.7% |
| 7 | 29 | 9.7% |
| 8 | 30 | 10.0% |
| 9 | 26 | 8.7% |

## `?`  (total: 300)

*operand ? has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 9.2, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 25 | 8.3% |
| 1 | 34 | 11.3% |
| 2 | 32 | 10.7% |
| 3 | 32 | 10.7% |
| 4 | 25 | 8.3% |
| 5 | 23 | 7.7% |
| 6 | 40 | 13.3% |
| 7 | 31 | 10.3% |
| 8 | 34 | 11.3% |
| 9 | 24 | 8.0% |

## `}`  (total: 299)

*operand } has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 4.3, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 28 | 9.4% |
| 1 | 39 | 13.0% |
| 2 | 31 | 10.4% |
| 3 | 27 | 9.0% |
| 4 | 26 | 8.7% |
| 5 | 28 | 9.4% |
| 6 | 28 | 9.4% |
| 7 | 29 | 9.7% |
| 8 | 33 | 11.0% |
| 9 | 30 | 10.0% |

## `)`  (total: 297)

*operand ) has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 8.9, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 35 | 11.8% |
| 1 | 37 | 12.5% |
| 2 | 22 | 7.4% |
| 3 | 31 | 10.4% |
| 4 | 36 | 12.1% |
| 5 | 27 | 9.1% |
| 6 | 22 | 7.4% |
| 7 | 31 | 10.4% |
| 8 | 30 | 10.1% |
| 9 | 26 | 8.8% |

## `]`  (total: 295)

*operand ] has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 6.1, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 28 | 9.5% |
| 1 | 27 | 9.2% |
| 2 | 23 | 7.8% |
| 3 | 25 | 8.5% |
| 4 | 36 | 12.2% |
| 5 | 34 | 11.5% |
| 6 | 33 | 11.2% |
| 7 | 29 | 9.8% |
| 8 | 26 | 8.8% |
| 9 | 34 | 11.5% |

## `` ` ``  (total: 293)

*operand ` has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 5.5, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 24 | 8.2% |
| 1 | 29 | 9.9% |
| 2 | 26 | 8.9% |
| 3 | 23 | 7.8% |
| 4 | 28 | 9.6% |
| 5 | 34 | 11.6% |
| 6 | 34 | 11.6% |
| 7 | 33 | 11.3% |
| 8 | 34 | 11.6% |
| 9 | 28 | 9.6% |

## `{`  (total: 293)

*operand { has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 12.6, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 34 | 11.6% |
| 1 | 30 | 10.2% |
| 2 | 43 | 14.7% |
| 3 | 28 | 9.6% |
| 4 | 30 | 10.2% |
| 5 | 27 | 9.2% |
| 6 | 26 | 8.9% |
| 7 | 32 | 10.9% |
| 8 | 19 | 6.5% |
| 9 | 24 | 8.2% |

## `'`  (total: 290)

*operand ' has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 7.2, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 22 | 7.6% |
| 1 | 31 | 10.7% |
| 2 | 34 | 11.7% |
| 3 | 28 | 9.7% |
| 4 | 28 | 9.7% |
| 5 | 39 | 13.4% |
| 6 | 30 | 10.3% |
| 7 | 26 | 9.0% |
| 8 | 25 | 8.6% |
| 9 | 27 | 9.3% |

## `$`  (total: 287)

*operand $ has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 5.4, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 31 | 10.8% |
| 1 | 26 | 9.1% |
| 2 | 27 | 9.4% |
| 3 | 24 | 8.4% |
| 4 | 29 | 10.1% |
| 5 | 37 | 12.9% |
| 6 | 33 | 11.5% |
| 7 | 26 | 9.1% |
| 8 | 30 | 10.5% |
| 9 | 24 | 8.4% |

## `:`  (total: 285)

*operand : has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 14.7, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 19 | 6.7% |
| 1 | 31 | 10.9% |
| 2 | 32 | 11.2% |
| 3 | 29 | 10.2% |
| 4 | 33 | 11.6% |
| 5 | 26 | 9.1% |
| 6 | 30 | 10.5% |
| 7 | 21 | 7.4% |
| 8 | 42 | 14.7% |
| 9 | 22 | 7.7% |

## `(`  (total: 284)

*operand ( has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 4.2, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 25 | 8.8% |
| 1 | 28 | 9.9% |
| 2 | 21 | 7.4% |
| 3 | 29 | 10.2% |
| 4 | 27 | 9.5% |
| 5 | 28 | 9.9% |
| 6 | 31 | 10.9% |
| 7 | 34 | 12.0% |
| 8 | 32 | 11.3% |
| 9 | 29 | 10.2% |

## `<`  (total: 284)

*operand < has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 10.0, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 33 | 11.6% |
| 1 | 39 | 13.7% |
| 2 | 26 | 9.2% |
| 3 | 25 | 8.8% |
| 4 | 25 | 8.8% |
| 5 | 22 | 7.7% |
| 6 | 31 | 10.9% |
| 7 | 22 | 7.7% |
| 8 | 34 | 12.0% |
| 9 | 27 | 9.5% |

## `\`  (total: 283)

*operand \ has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 11.4, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 19 | 6.7% |
| 1 | 25 | 8.8% |
| 2 | 24 | 8.5% |
| 3 | 36 | 12.7% |
| 4 | 22 | 7.8% |
| 5 | 33 | 11.7% |
| 6 | 25 | 8.8% |
| 7 | 33 | 11.7% |
| 8 | 31 | 11.0% |
| 9 | 35 | 12.4% |

## `%`  (total: 271)

*operand % has no fixed digit; across the corpus it represented every digit 0-9 at near-uniform frequency (chi-square = 12.1, not significant). The symbol/digit mapping is assigned at random per puzzle, so there is no usable prior.*

| Digit | Count | % |
|---|---|---|
| 0 | 25 | 9.2% |
| 1 | 29 | 10.7% |
| 2 | 23 | 8.5% |
| 3 | 28 | 10.3% |
| 4 | 37 | 13.7% |
| 5 | 30 | 11.1% |
| 6 | 34 | 12.5% |
| 7 | 18 | 6.6% |
| 8 | 19 | 7.0% |
| 9 | 28 | 10.3% |
