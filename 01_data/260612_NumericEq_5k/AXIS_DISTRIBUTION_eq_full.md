# Axis distribution — whole equation_numeric training set (original + aug1 + aug2)

5000 trained rows (oversampling > 0). All axes derived uniformly from each row's prompt/CoT/label.

## type
| value | count | % |
|---|---|---|
| deducible | 2750 | 55.0% |
| concat | 750 | 15.0% |
| ambiguous | 500 | 10.0% |
| exotic | 500 | 10.0% |
| unseen | 500 | 10.0% |

## reading
| value | count | % |
|---|---|---|
| leftward | 2549 | 51.0% |
| rightward | 2451 | 49.0% |

## signed
| value | count | % |
|---|---|---|
| prefix | 2369 | 47.4% |
| none | 2206 | 44.1% |
| suffix | 425 | 8.5% |

## leadzero
| value | count | % |
|---|---|---|
| False | 2986 | 59.7% |
| True | 2014 | 40.3% |

## nops
| value | count | % |
|---|---|---|
| 3 | 2480 | 49.6% |
| 2 | 2478 | 49.6% |
| 1 | 42 | 0.8% |

## arith
| value | count | % |
|---|---|---|
| True | 3022 | 60.4% |
| False | 1978 | 39.6% |

## source
| value | count | % |
|---|---|---|
| aug2 | 2646 | 52.9% |
| aug1 | 1733 | 34.7% |
| original | 621 | 12.4% |

## subcase
| value | count | % |
|---|---|---|
| - | 4000 | 80.0% |
| uns:arith | 484 | 9.7% |
| amb:absolute difference | 289 | 5.8% |
| amb:subtraction (a-b) | 169 | 3.4% |
| amb:negated absolute difference | 42 | 0.8% |
| uns:concat | 16 | 0.3% |

## operator symbol
coverage: **26 of 26** possible symbols present — full coverage

| symbol | count | % of slots |
|---|---|---|
| `-` | 3469 | 14.0% |
| `+` | 1585 | 6.4% |
| `*` | 1576 | 6.4% |
| `^` | 851 | 3.4% |
| `"` | 850 | 3.4% |
| `!` | 842 | 3.4% |
| ``` | 837 | 3.4% |
| `}` | 819 | 3.3% |
| `<` | 817 | 3.3% |
| `[` | 805 | 3.3% |
| `]` | 803 | 3.2% |
| `)` | 798 | 3.2% |
| `>` | 795 | 3.2% |
| `(` | 794 | 3.2% |
| `{` | 787 | 3.2% |
| `$` | 785 | 3.2% |
| `'` | 780 | 3.2% |
| `/` | 774 | 3.1% |
| `%` | 773 | 3.1% |
| `|` | 771 | 3.1% |
| `\` | 759 | 3.1% |
| `?` | 752 | 3.0% |
| `#` | 746 | 3.0% |
| `&` | 734 | 3.0% |
| `:` | 717 | 2.9% |
| `@` | 702 | 2.8% |

## length
(terciles: short ≤1527 · medium ≤1788 · long >1788)
| band | count | % |
|---|---|---|
| short | 1669 | 33.4% |
| medium | 1666 | 33.3% |
| long | 1665 | 33.3% |

median 1645 · mean 1718 · min 901 · max 3341
