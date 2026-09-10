# Evidence: `newdata` overwrote huikang's 65 concat CoTs (cryptarithm regression)

Date 2026-05-25. The baseline **moe** trained on huikang's golden data, which uses **65 short
concat CoTs** (source `pretok_0408_decoded`) for the cryptarithm puzzles lkevincc's solver
couldn't do. **`newdata` replaced ALL 800 cryptarithm CoTs with our long-DFS CoTs — including
those 65** — which is why concat broke and cryptarithm went 1/68 → 0/68.

## The 65 huikang concat CoTs
- count: **65**;  mention "concat": **65/65**;  median length **1341 chars** (short).
- our solver re-labelled them: {'derived_concat': 54, 'guess_operator': 11} (54 are concat).
- in `newdata` all 65 were overwritten with our CoTs (oversampling 1).

---

## Worked example: id `b1b10e83`  (ground truth `|"#$`)
PROMPT:
```
"\+#| = "\#|
#!+#\ = #!#\
#"*#" = #){|
$\*)" = ${$#
)\+`# = )\`#
Now, determine the result for: |"+#$
```

golden source = `pretok_0408_decoded` (huikang)  ·  in the 65: **yes**
  ·  moe pred `|"#$` **CORRECT** (780 tok)
  ·  newdata pred `10` **wrong** (7680 tok)

### (A) huikang golden CoT — moe's training target (SHORT, concat, terminates) — FULL
```
We need to infer the transformation rule from the examples.

First, let me assign letters to each symbol:
" -> A
\ -> B
# -> C
| -> D
! -> E
) -> F
{ -> G
$ -> H
` -> I
Operators:
+ -> x
* -> y

Converting examples to letter form:

  「"\+#|」 = 「"\#|」:
    input:  " \ + # | : " -> A, \ -> B, + -> x, # -> C, | -> D -> AB x CD
    output: " \ # | : " -> A, \ -> B, # -> C, | -> D -> ABCD

  「#!+#\」 = 「#!#\」:
    input:  # ! + # \ : # -> C, ! -> E, + -> x, # -> C, \ -> B -> CE x CB
    output: # ! # \ : # -> C, ! -> E, # -> C, \ -> B -> CECB

  「#"*#"」 = 「#){|」:
    input:  # " * # " : # -> C, " -> A, * -> y, # -> C, " -> A -> CA y CA
    output: # ) { | : # -> C, ) -> F, { -> G, | -> D -> CFGD

  「$\*)"」 = 「${$#」:
    input:  $ \ * ) " : $ -> H, \ -> B, * -> y, ) -> F, " -> A -> HB y FA
    output: $ { $ # : $ -> H, { -> G, $ -> H, # -> C -> HGHC

  「)\+`#」 = 「)\`#」:
    input:  ) \ + ` # : ) -> F, \ -> B, + -> x, ` -> I, # -> C -> FB x IC
    output: ) \ ` # : ) -> F, \ -> B, ` -> I, # -> C -> FBIC

Each input is 5 characters: two symbol-digits, an operator, two more symbol-digits.
Operator y: unknown
Operator x: concatenation

The question operator is x, which is concatenation.

Converting question "|"+#$": | " + # $ : | -> D, " -> A, + -> x, # -> C, $ -> H -> DA x CH

Applying to DA x CH:
  concatenation(DA, CH) = DA || CH = DACH
  Converting back: D A C H : D -> |, A -> ", C -> #, H -> $ -> |"#$

I will now return the answer in \boxed{}
The answer in \boxed is
\boxed{|"#$}
```

### (B) moe MODEL generated — identifies concat, copies symbols → correct — FULL
```
We need to infer the transformation rule from the examples.

First, let me assign letters to each symbol:
" -> A
\ -> B
# -> C
| -> D
! -> E
$ -> F
) -> G
` -> H
{ -> I
} -> J
Operators:
* -> x
+ -> y
- -> z

Digit mapping:
A = 5, B = 1, C = 2, D = 8, E = 4, F = 6, G = 3, H = 0, I = 7, J = 9

Converting examples to letter form:

  「"\+#|」 = 「"\#|」:
    input:  " \ + # | : " -> A, \ -> B, + -> y, # -> C, | -> D -> AB y CD
    output: " \ # | : " -> A, \ -> B, # -> C, | -> D -> ABCD

  「#!+#\」 = 「#!#\」:
    input:  # ! + # \ : # -> C, ! -> E, + -> y, # -> C, \ -> B -> CE y CB
    output: # ! # \ : # -> C, ! -> E, # -> C, \ -> B -> CECB

  「#"*#"」 = 「#){|」:
    input:  # " * # " : # -> C, " -> A, * -> x, # -> C, " -> A -> CA x CA
    output: # ) { | : # -> C, ) -> G, { -> I, | -> D -> CGID
    digits: CA = 25, CA = 25, multiplication(25, 25) = 625, CGID = 625

  「$\*)"」 = 「${$#」:
    input:  $ \ * ) " : $ -> F, \ -> B, * -> x, ) -> G, " -> A -> FB x GA
    output: $ { $ # : $ -> F, { -> I, $ -> F, # -> C -> FIF C
    digits: FB = 61, GA = 35, multiplication(61, 35) = 2135, FIF C = 2135

  「)\+`#」 = 「)\`#」:
    input:  ) \ + ` # : ) -> G, \ -> B, + -> y, ` -> H, # -> C -> GB y HC
    output: ) \ ` # : ) -> G, \ -> B, ` -> H, # -> C -> GBHC

Each input is 5 characters: two symbol-digits, an operator, two more symbol-digits.
Operator x: multiplication
Operator y: concatenation
Operator z: subtraction

The question operator is y, which is concatenation.

Converting question "|"+#$": | " + # $ : | -> D, " -> A, + -> y, # -> C, $ -> F -> DA y CF

Applying to DA y CF:
  DA = 85, CF = 26
  concatenation(85, 26) = 8526
  Converting back: D A C F : D -> |, A -> ", C -> #, F -> $ -> |"#$

I will now return the answer in \boxed{}
The answer in \boxed is
\boxed{|"#$}
</think>
\boxed{|"#$}
```

### (C) our CoT — newdata's training target (long DFS, 1547 tok) — HEAD + answer
```
I need to infer the transformation rule from the examples.

First, let me assign letters to each symbol:
  ! -> A
  " -> B
  # -> C
  $ -> D
  ) -> E
  \ -> F
  ` -> G
  { -> H
  | -> I
Operators:
  + -> f
  * -> g

I will convert all the equations to letter form:
  EX1 「"\+#|」 = 「"\#|」 becomes BF f CI = BFCI
  EX2 「#!+#\」 = 「#!#\」 becomes CA f CF = CACF
  EX3 「#"*#"」 = 「#){|」 becomes CB g CB = CEHI
  EX4 「$\*)"」 = 「${$#」 becomes DF g EB = DHDC
  EX5 「)\+`#」 = 「)\`#」 becomes EF f GC = EFGC
  QUERY 「|"+#$」 becomes IB f CD

In Alice world, multiplication, addition, subtraction and concatenating are the most common operations, but the results are usually noisy (off by ±1 or ±2, negated, or with operands reversed). For the operators in this puzzle:
  +, denoted as f, is usually noisy_add, but I have seen it represent [concat, noisy_subtraction, noisy_mul] in other puzzles, ordered by frequency
  *, denoted as g, is usually noisy_mul, but I have seen it represent [concat, noisy_add, noisy_subtraction] in other puzzles, ordered by frequency
Anyway, I should analyze the pattern of the examples to figure out what they are.

No operator symbol appears in any RHS. Reading order defaults to standard (tens digit first). If the full DFS finds no solution under standard order, I will retry with little-endian order (units digit first).

Operator f in QUERY appears in EX1, so I should analyze
   ...[~1547 tokens of interval-propagation DFS]...
Query operator f is concat_fwd (from the examples); answer = IBCD = |"#$
\boxed{|"#$}
```

### (D) newdata MODEL generated — runs the DFS, loops, truncates at 7680 — HEAD + TAIL
```
I need to infer the transformation rule from the examples.

First, let me assign letters to each symbol:
  " -> A
  # -> B
  $ -> C
  ) -> D
  \ -> E
  | -> F
  ` -> G
  { -> H
Operators:
  + -> f
  * -> g
  ) -> h (but note: in the examples, ')' is used as an operator, so I'll use h for it)
  $ -> i (but note: in the examples, '$' is used as an operator, so I'll use i for it)

I will convert all the equations to letter form:
  EX1 「"\+#|」 = 「"\#|」 becomes AE f BF = AEBF
  EX2 「#!+#\」 = 「#!#\」 becomes BD f BE = BDBE
  EX3 「#"*#"」 = 「#){|」 becomes BA g BA = BHDF
  EX4 「$\*)"」 = 「${$#」 becomes CE g AD = CHCB
  EX5 「)\+`#」 = 「)\`#」 becomes DE h EG = DEGD
  QUERY 「|"+#$」 becomes FA AC

In Alice world, multiplication, addition, subtraction and concatenating are the most common operations, but the results are usually noisy (off by ±1 or ±2, negated, or with operands reversed). For the operator
   ...[loops]...
"o g is not concat.\n  h: concat_fwd=DEEG, concat_rev=EGDE. Output=DEGD. Neither matches, so h is not concat.\n  i: not present in any example, so I have to guess what it means. Since it is $, it is most likely noisy_add or noisy_mul; I'll check the pattern later.\n  g: concat_fwd=BABA, concat_rev=ABAB."
```

## Takeaway
moe solves it because huikang's CoT teaches the **concat shortcut** (no digit mapping, no search —
just identify concat from the examples and copy the query's operand symbols). Our CoT teaches a
**full search** the model can't execute (misparses symbols, loops to the token ceiling). By
overwriting huikang's 65 concat CoTs with ours, `newdata` destroyed the only working part.
**Fix: drop our cryptarithm CoTs; keep huikang's golden mix.**