# Axis distribution — cryptarithm corpus (the 800 original puzzles)

Measured 797/800 with the bundled clean solver's parser (`cot_generator.parse`) over each puzzle's
`question.txt` + `track/tree_cot.txt`. Cryptarithm's OWN axes (no leadzero — cipher forbids leading 0).
**Scenario types kept distinct:** deduce (query op in examples, derived 4-family) · query-only (op only in
QUERY → default) · concat (query op is concat) · **not-arith-solvable** (search fails → blind concat default;
in cryptarithm "exotic" = arithmetic-unsolvable, since an unresolvable operator takes its hidden digits down
with it). Ambiguous = an answer that hinged on a deterministic default-fill (free/missing digit).


## Scenario space — full coverage (every scenario, with GT-status) ❗

**The aug MUST cover the GT-False / 0×-trained scenarios too — that is the whole point of aug.**
A scenario that is GT-False in the originals is excluded from training (0× weight), so the model
never learns that behavior. Those cells are exactly what the aug must reproduce as **GT-True**
(construct the puzzle so the *intended* answer equals the solver's deterministic-default output).

| scenario | corpus | GT in originals | aug must generate |
|---|---|---|---|
| deduce | 557 (69.9%) | mix (64% True) | GT-True deduce |
| query-only | 159 (19.9%) | mostly False (28% True) | GT-True query-only |
| concat | 59 (7.4%) | all True | GT-True concat |
| **not-arith-solvable** | 22 (2.8%) | all False | **GT-True — gold = the concat default (the cell that is 100% missing today)** |
| ambiguous overlay (default-fill) | 153 (19.2%) | mostly False (14% True) | **GT-True — gold = the default-fill output** |

_GT-False originals are kept at 0× (not trained); the aug supplies the GT-True version of each so the
behavior is covered. Train = (GT-True originals at 1×) + (aug filling every scenario, incl. the 0× ones)._

## A1 scenario type
| value | count | % |
|---|---|---|
| `deduce` | 557 | 69.9% |
| `query-only` | 159 | 19.9% |
| `concat` | 59 | 7.4% |
| `not-arith-solvable` | 22 | 2.8% |

## A2 reading
| value | count | % |
|---|---|---|
| `rightward` | 476 | 59.7% |
| `leftward` | 261 | 32.7% |
| `short-circuit` | 60 | 7.5% |

## A3 signed (sign position)
| value | count | % |
|---|---|---|
| `none` | 462 | 58.0% |
| `prefix` | 316 | 39.6% |
| `suffix` | 19 | 2.4% |

## A4 n_operators
| value | count | % |
|---|---|---|
| `1` | 12 | 1.5% |
| `2` | 296 | 37.1% |
| `3` | 489 | 61.4% |

## A5 n_examples
| value | count | % |
|---|---|---|
| `3` | 232 | 29.1% |
| `4` | 289 | 36.3% |
| `5` | 276 | 34.6% |

## A6 operator symbol class
| value | count | % |
|---|---|---|
| `arith-glyph` | 627 | 78.7% |
| `punctuation` | 170 | 21.3% |

## A7 operator FAMILY (per operator; 2071 ops total)
| value | count | % |
|---|---|---|
| `add` | 463 | 22.4% |
| `sub` | 499 | 24.1% |
| `mul` | 498 | 24.0% |
| `concat` | 199 | 9.6% |
| `other` | 412 | 19.9% |

## A8 noise variant (per arithmetic operator)
| value | count | % |
|---|---|---|
| `base` | 624 | 48.7% |
| `+1` | 162 | 12.6% |
| `-1` | 152 | 11.9% |
| `+2` | 39 | 3.0% |
| `-2` | 33 | 2.6% |
| `absdiff` | 192 | 15.0% |
| `neg-absdiff` | 64 | 5.0% |
| `b-a` | 16 | 1.2% |

## A9 concat direction
| value | count | % |
|---|---|---|
| `fwd` | 160 | 80.4% |
| `rev` | 39 | 19.6% |

## A10 cipher_size (# distinct digit-symbols)
| value | count | % |
|---|---|---|
| `6` | 1 | 0.1% |
| `7` | 6 | 0.8% |
| `8` | 84 | 10.5% |
| `9` | 270 | 33.9% |
| `10` | 436 | 54.7% |

## A11 arithmetic solvability
| value | count | % |
|---|---|---|
| `arith-solvable` | 711 | 89.2% |
| `concat-short` | 59 | 7.4% |
| `not-arith-solvable` | 27 | 3.4% |

## A12 cipher-resolution depth
| value | count | % |
|---|---|---|
| `full` | 617 | 77.4% |
| `partial/ambiguous` | 153 | 19.2% |
| `none(0)` | 27 | 3.4% |

## A13 operator symbol identity
| value | count | % |
|---|---|---|
| `*` | 472 | 22.8% |
| `-` | 472 | 22.8% |
| `+` | 466 | 22.5% |
| `)` | 39 | 1.9% |
| `'` | 36 | 1.7% |
| `{` | 36 | 1.7% |
| ``` | 35 | 1.7% |
| `}` | 33 | 1.6% |
| `:` | 32 | 1.5% |
| `\` | 32 | 1.5% |
| `?` | 30 | 1.4% |
| `|` | 30 | 1.4% |
| `<` | 30 | 1.4% |
| `#` | 29 | 1.4% |
| `(` | 29 | 1.4% |
| `&` | 29 | 1.4% |
| `!` | 28 | 1.4% |
| `"` | 26 | 1.3% |
| `@` | 26 | 1.3% |
| `[` | 25 | 1.2% |
| `>` | 25 | 1.2% |
| `%` | 23 | 1.1% |
| `^` | 23 | 1.1% |
| `]` | 22 | 1.1% |
| `$` | 22 | 1.1% |
| `/` | 21 | 1.0% |

## A14 operand symbol usage (which of the 23 are digits)
| value | count | % |
|---|---|---|
| `!` | 348 | 4.6% |
| `>` | 345 | 4.6% |
| `#` | 345 | 4.6% |
| `"` | 342 | 4.6% |
| `/` | 342 | 4.6% |
| `|` | 340 | 4.5% |
| `&` | 337 | 4.5% |
| `[` | 330 | 4.4% |
| `}` | 328 | 4.4% |
| `)` | 325 | 4.3% |
| ``` | 324 | 4.3% |
| `@` | 324 | 4.3% |
| `{` | 323 | 4.3% |
| `]` | 323 | 4.3% |
| `:` | 323 | 4.3% |
| `^` | 322 | 4.3% |
| `?` | 319 | 4.2% |
| `\` | 316 | 4.2% |
| `$` | 315 | 4.2% |
| `'` | 314 | 4.2% |
| `(` | 313 | 4.2% |
| `<` | 311 | 4.1% |
| `%` | 301 | 4.0% |
## A15 BPE merged operand-tokens PRESENT in the 800 (transcription axis)
distinct 2-symbol merged tokens: 337 · distinct 3-symbol merged tokens: 117
(a coverage ceiling needs the full merge table; this is the count actually exercised by the corpus)
## A16 CoT length (Nemotron tokens; terciles short ≤3933 · medium ≤8212 · long >8212)
median 5316 · mean 18811 · min 930 · max 1604796

## A17 answer token-merge (the model must EMIT the boxed answer as a correctly-merged BPE sequence — inverse of the §0 un-merge)
The final answer is a 1–4 symbol string the model must merge into BPE tokens (e.g. `\boxed{``#}`). A distinct learnable skill from solving the cipher: joining the per-digit symbols back into the tokenizer's merged tokens. The CoT teaches it explicitly: `… so 112 -> ` ` # -> ``#` (spaced symbols → joined).
- **answer length (symbols):** 1·44 · 2·241 · 3·209 · 4·303
- **answer BPE-token count** (how many tokens the merge produces): **1-token 182 (23%) · 2-token 350 (44%) · 3-token 224 (28%) · 4-token 41 (5%)**
- Examples: `("` = 1 token; `("/"` = 2 (`("/`, `"`); `?">"` = 3 (`?`, `">`, `"`); `'!{?` = 4 singletons.
**Aug-coverage note:** balance across the 1–4-token merge classes so the model learns merging at every length, not just the dominant 2-token case; the 4-token (all-singleton) class is rarest (5%) and the highest risk for the model to under-merge.
