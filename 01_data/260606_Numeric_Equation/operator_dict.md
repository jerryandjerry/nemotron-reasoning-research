# Operator Symbol → Operation Dictionary (equation_numeric)

**Our own operator_dict, built from this corpus** — the equation_numeric analog of the cryptarithm
`operator_dict.md`. Here the operands are literal numbers (no symbol→digit cipher), so **there is no**
**operand_dict** — only the OPERATORS are noisy symbols. This table is each operator symbol's deduced
operation aggregated over all 732 puzzles, **restricted to our cryptarithm vocabulary** (mul / add /
sub families + concat); operations outside it are not searched, so operators using one contribute
nothing here. `_gen_eq.py` recomputes exactly this table at run time (`get_prior`).

**`+ - *` carry a strong prior; every other symbol is ~uniform** (the analog of the cryptarithm
symbol↔digit mapping being non-derivable).

| Symbol | n | dominant family | share |
|---|---|---|---|
| `+` | 260 | noisy_add | 64% |
| `-` | 261 | noisy_subtraction | 83% |
| `*` | 266 | noisy_mul | 62% |
| (22 arbitrary) | 886 | none | ~37% avg top |

## How it is used (exactly our cryptarithm method — no invention)

1. **PREFIX operator background** (cryptarithm `make_prefix` wording): for `+ - *`, "`+`, denoted as
   f, is usually noisy_add, but I have seen it represent [...] in other puzzles, ordered by frequency";
   for an arbitrary symbol, "I have seen operator g represent [...] in other puzzles, ordered by
   frequency". Hedged — never "+ is absolutely addition".
2. **SEEN query operator**: the operation is read off the examples directly (prior is only background).
3. **UNSEEN (guess) query operator** (cryptarithm `emit_answer`): operator distinctness first — drop
   any arithmetic family already used by a locked operator (distinctness holds in 98% of multi-operator
   puzzles here) — then the prior chooses among what remains. Base variant per family: noisy_add→
   addition, noisy_subtraction→absolute difference, noisy_mul→multiplication.

**Exotic operations are OUT of scope (our rules).** max-mod-min, integer division, digit/determinant
rules etc. appear in the equation_numeric generator (~60 operators) but are NOT in our cryptarithm
vocabulary, so we do not search them; an operator needing one is honestly declared "a rule outside the
ones I recognize" and left undetermined (blind fallback). GT-match (official metric): **597/732**
**(deduce 542, guess 55)** vs huikang 563 — same honest method, applied to this puzzle.

---

## Per-symbol operation frequencies (within our vocabulary)

### `+`  (total 260) — usually **noisy_add**
| Operation | Count | % |
|---|---|---|
| addition | 83 | 31.9% |
| concatenation | 53 | 20.4% |
| add+1 | 45 | 17.3% |
| add-1 | 38 | 14.6% |
| absolute difference | 13 | 5.0% |
| reverse concatenation | 9 | 3.5% |
| negated absolute difference | 6 | 2.3% |
| multiplication | 6 | 2.3% |
| multiply+1 | 3 | 1.2% |
| multiply-1 | 3 | 1.2% |
| subtraction (a-b) | 1 | 0.4% |

### `-`  (total 261) — usually **noisy_subtraction**
| Operation | Count | % |
|---|---|---|
| absolute difference | 102 | 39.1% |
| negated absolute difference | 91 | 34.9% |
| subtraction (a-b) | 23 | 8.8% |
| multiplication | 11 | 4.2% |
| addition | 10 | 3.8% |
| add-1 | 7 | 2.7% |
| multiply-1 | 7 | 2.7% |
| concatenation | 5 | 1.9% |
| add+1 | 3 | 1.1% |
| multiply+1 | 2 | 0.8% |

### `*`  (total 266) — usually **noisy_mul**
| Operation | Count | % |
|---|---|---|
| multiplication | 75 | 28.2% |
| multiply+1 | 51 | 19.2% |
| concatenation | 46 | 17.3% |
| multiply-1 | 39 | 14.7% |
| negated absolute difference | 12 | 4.5% |
| reverse concatenation | 11 | 4.1% |
| absolute difference | 9 | 3.4% |
| addition | 8 | 3.0% |
| add+1 | 6 | 2.3% |
| add-1 | 6 | 2.3% |
| subtraction (a-b) | 3 | 1.1% |

### Arbitrary symbols (weak / ~uniform — distinctness does the real work)

- `<` (total 48): addition 13, multiplication 8, absolute difference 5, concatenation 4, add+1 4, multiply+1 4, negated absolute difference 4, subtraction (a-b) 2, add-1 1, reverse subtraction (b-a) 1, multiply-1 1, reverse concatenation 1
- ``` (total 47): absolute difference 10, addition 7, multiplication 7, multiply-1 5, negated absolute difference 3, subtraction (a-b) 3, concatenation 3, multiply+1 3, add+1 3, add-1 2, reverse concatenation 1
- `!` (total 44): multiplication 10, absolute difference 8, multiply+1 7, negated absolute difference 5, addition 4, add-1 3, add+1 2, concatenation 2, multiply-1 2, subtraction (a-b) 1
- `[` (total 44): multiplication 10, absolute difference 7, addition 7, add+1 4, concatenation 4, negated absolute difference 3, multiply+1 2, reverse concatenation 2, multiply-1 2, add-1 2, subtraction (a-b) 1
- `/` (total 44): addition 12, absolute difference 10, multiplication 5, negated absolute difference 3, multiply-1 3, concatenation 3, add-1 2, multiply+1 2, add+1 2, subtraction (a-b) 1, reverse concatenation 1
- `"` (total 43): multiplication 10, absolute difference 9, concatenation 7, negated absolute difference 5, addition 4, add+1 3, add-1 2, multiply-1 1, subtraction (a-b) 1, multiply+1 1
- `?` (total 43): concatenation 9, addition 8, negated absolute difference 6, add+1 4, multiply+1 3, multiply-1 3, add-1 3, multiplication 3, absolute difference 2, reverse subtraction (b-a) 1, subtraction (a-b) 1
- `$` (total 43): multiplication 8, multiply-1 7, addition 7, negated absolute difference 5, absolute difference 4, multiply+1 3, add+1 3, concatenation 3, add-1 1, reverse concatenation 1, subtraction (a-b) 1
- `^` (total 41): addition 10, multiplication 8, absolute difference 6, negated absolute difference 4, concatenation 4, multiply+1 3, add+1 2, add-1 2, subtraction (a-b) 1, reverse concatenation 1
- `%` (total 40): addition 8, add+1 6, absolute difference 5, negated absolute difference 5, multiplication 5, multiply-1 3, reverse subtraction (b-a) 3, concatenation 2, add-1 1, multiply+1 1, subtraction (a-b) 1
- `}` (total 40): multiplication 8, addition 5, concatenation 5, multiply+1 4, negated absolute difference 4, add+1 3, add-1 3, absolute difference 3, reverse subtraction (b-a) 2, multiply-1 1, reverse concatenation 1, subtraction (a-b) 1
- `{` (total 39): multiply-1 7, negated absolute difference 6, absolute difference 6, addition 6, multiplication 5, multiply+1 4, add-1 2, concatenation 2, reverse concatenation 1
- `@` (total 38): multiplication 10, addition 6, negated absolute difference 5, concatenation 4, multiply-1 3, multiply+1 2, add-1 2, add+1 2, absolute difference 2, reverse concatenation 1, subtraction (a-b) 1
- `\` (total 36): multiplication 9, absolute difference 6, negated absolute difference 5, addition 4, reverse concatenation 2, add+1 2, reverse subtraction (b-a) 2, add-1 2, multiply-1 2, subtraction (a-b) 1, multiply+1 1
- `#` (total 35): addition 7, absolute difference 6, multiplication 5, multiply-1 5, negated absolute difference 3, add+1 2, subtraction (a-b) 2, concatenation 2, reverse concatenation 1, multiply+1 1, add-1 1
- `:` (total 35): multiplication 7, addition 7, absolute difference 7, concatenation 5, negated absolute difference 3, add+1 2, subtraction (a-b) 2, multiply-1 1, reverse concatenation 1
- `'` (total 35): absolute difference 6, add+1 4, addition 4, negated absolute difference 4, multiply-1 4, concatenation 3, multiplication 3, multiply+1 2, add-1 2, subtraction (a-b) 1, reverse concatenation 1, reverse subtraction (b-a) 1
- `)` (total 33): multiplication 10, addition 7, concatenation 3, absolute difference 3, subtraction (a-b) 2, multiply+1 2, negated absolute difference 2, add-1 1, reverse subtraction (b-a) 1, add+1 1, multiply-1 1
- `(` (total 33): absolute difference 5, negated absolute difference 4, multiply+1 4, subtraction (a-b) 4, multiplication 4, concatenation 4, add+1 2, reverse concatenation 2, multiply-1 2, reverse subtraction (b-a) 1, add-1 1
- `&` (total 32): multiplication 7, addition 7, absolute difference 5, add-1 3, subtraction (a-b) 2, negated absolute difference 2, concatenation 2, multiply+1 1, multiply-1 1, reverse concatenation 1, reverse subtraction (b-a) 1
- `>` (total 32): addition 6, multiplication 5, absolute difference 4, concatenation 4, add-1 3, reverse subtraction (b-a) 3, negated absolute difference 2, multiply+1 2, subtraction (a-b) 1, multiply-1 1, add+1 1
- `|` (total 31): addition 8, negated absolute difference 4, concatenation 4, add+1 3, multiplication 3, absolute difference 2, reverse concatenation 2, multiply-1 1, reverse subtraction (b-a) 1, multiply+1 1, add-1 1, subtraction (a-b) 1
- `]` (total 30): multiplication 5, multiply+1 4, multiply-1 3, absolute difference 3, subtraction (a-b) 3, addition 3, negated absolute difference 3, add+1 2, add-1 2, concatenation 2
