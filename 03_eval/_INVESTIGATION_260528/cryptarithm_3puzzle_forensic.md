# Cryptarithm 3-Puzzle Forensic Analysis (2026-05-28)

Investigation of three `cryptarithm_deduce` puzzles whose correct/incorrect verdict flips
across four fine-tune runs.

| run     | training csv                                                                                                  | cryptarithm CoT family |
| ------- | ------------------------------------------------------------------------------------------------------------- | ---------------------- |
| moe     | `01_data/260514_lkevincc_golden/260514_huikang_golden_stripped.csv`                                           | OLD lkevincc golden    |
| alleq   | `01_data/260525_NumericEquation_TrainData/260525_huikang_NumericEq_allEq.csv`                                 | OLD lkevincc golden    |
| alleqgt | `01_data/260526_Numeric_Equation/260527_NumericEq_TrainData/260527_huikang_NumericEq_allEq_gtTrue.csv`        | OLD lkevincc golden    |
| cryptn  | `01_data/260526_Cryptarithm_TrainData/260527_huikang_NumericEq.csv`                                           | NEW `01_data/260525_Cryptarithm/` |

Sanity check: for all three puzzles the `solver_cot` text in moe/alleq/alleqgt is **byte-identical**
(1324 / 1264 / 1497 chars respectively). Same template, same answer line. So differences across
those three runs are model / training-mix / decoding effects, not different cryptarithm CoTs.

---

## 0. Verdict matrix

| puzzle      | GT       | moe pred | alleq pred | alleqgt pred | cryptn pred |
| ----------- | -------- | -------- | ---------- | ------------ | ----------- |
| `24e1f1d5`  | `^#(!`   | `^#(!` ✓ | `^(!#` ✗   | `^#(!` ✓     | `3` ✗ (truncated) |
| `4d8df95b`  | `)`      | `?` ✗    | `)` ✓      | `-` ✗        | `3` ✗ (truncated) |
| `b1b10e83`  | `\|"#$`  | `\|"#$` ✓| ` \`} ` ✗   | `}}` ✗       | `{\`` ✗     |

Note: cryptn's `predicted='3'` is the Kaggle-metric fallback (no `\boxed{}` was emitted before
the 7680-token cap). `output_token_len` for cryptn on `24e1f1d5` and `4d8df95b` is exactly 7680
(the configured max-new-tokens), indicating hard truncation mid-search.

---

## 1. Training-row inventory (all four runs)

For each puzzle:

### `24e1f1d5` (pure_concat)
| run     | source                 | oversampling | GT-match | token len | cot chars |
| ------- | ---------------------- | ------------ | -------- | --------- | --------- |
| moe     | `pretok_0408_decoded`  | 1            | n/a      | n/a       | 1324      |
| alleq   | `pretok_0408_decoded`  | 1            | True     | 533       | 1324      |
| alleqgt | `pretok_0408_decoded`  | 1            | True     | 533       | 1324      |
| cryptn  | `260526_new solver`    | **2**        | True     | 2934      | **6841**  |

### `4d8df95b` (arithmetic – subtraction)
| run     | source                 | oversampling | GT-match | token len  | cot chars   |
| ------- | ---------------------- | ------------ | -------- | ---------- | ----------- |
| moe     | `lkevincc_golden`      | 1            | n/a      | n/a        | 1264        |
| alleq   | `lkevincc_golden`      | 1            | True     | 527        | 1264        |
| alleqgt | `lkevincc_golden`      | 1            | True     | 527        | 1264        |
| cryptn  | `260526_new solver`    | **0** *(excluded)* | True | **195 352** | **394 547** |

### `b1b10e83` (pure_concat)
| run     | source                 | oversampling | GT-match | token len | cot chars |
| ------- | ---------------------- | ------------ | -------- | --------- | --------- |
| moe     | `pretok_0408_decoded`  | 1            | n/a      | n/a       | 1497      |
| alleq   | `pretok_0408_decoded`  | 1            | True     | 608       | 1497      |
| alleqgt | `pretok_0408_decoded`  | 1            | True     | 608       | 1497      |
| cryptn  | `260526_new solver`    | **2**        | True     | 1814      | **4504**  |

The cryptn `solver_cot` for each puzzle is **byte-identical** to the corresponding on-disk
`01_data/260525_Cryptarithm/<cat>_<id>/track/tree_cot.txt` (6841 / 394547 / 4504 chars).

**Train-set gate (cryptn).** `_build_traincsv.py` sets `CAP = 7680`:
> oversampling=0 if `token length >= CAP OR GT-match != True`.

Under this gate, of the 800 new-solver cryptarithm rows: 265 included (os=2), 535 excluded
(192 over-cap-only, 191 over-cap-AND-gt-mismatch, 152 gt-mismatch-only).

`4d8df95b` is **excluded** because its honest new-CoT runs 195 352 tokens (≈ 25× the cap).

---

## 2. Q1 — `b1b10e83`: moe ✓ vs alleqgt ✗ on byte-identical cryptarithm training

Both runs train on the same 1497-char OLD-method CoT for this puzzle. Yet moe gets it right
(`|"#$`) and alleqgt gets it wrong (`}}`).

**First character of divergence: position 235 of the model output** — the digit-mapping line.

```
moe:     ...Digit mapping:
         A = 5, B = 1, C = 2, D = 3, E = 4, F = 6, G = 7, H = 0, I = 8, J = 9

alleqgt: ...Digit mapping:
         A = 5, B = 2, C = 8, D = 1, E = 3, F = 4, G = 6, H = 0, I = 7, J = 9
         Reading order: little-endian (units digit first)
```

Two structural things change:

1. **Different (arbitrary) digit permutation** — both legal, both consistent with the
   examples that don't constrain digits.
2. **alleqgt injects a "Reading order: little-endian" line.** This phrase does not appear
   in this puzzle's training CoT — but it *does* appear in **302 OLD-CoT rows** in moe's,
   alleq's, and alleqgt's training (one full sub-family of golden CoT, e.g.
   `00c032a8` confirms). It is a legitimate piece of the OLD distribution. Across the
   55 cryptarithm val outputs:

   | run     | outputs that emit "Reading order:" |
   | ------- | ---------------------------------- |
   | moe     | 0 / 55                             |
   | alleq   | 38 / 55                            |
   | alleqgt | 26 / 55                            |
   | cryptn  | 0 / 55 (uses NEW method 55/55)     |

   So alleq/alleqgt have **drifted** toward the little-endian sub-template much more
   often than moe did. With "little-endian" enabled, the model interprets `AB = 15`
   as the number "ones=1, tens=5" = 51 (or vice-versa) — and EX1 `"\+#|` becomes
   `addition(25, 18) = 43, ABCD = 43`. The numeric arithmetic only checks out if the
   operator is **addition**, so alleqgt commits to "Operator y: addition" with
   p = 0.99997 at token 741 instead of "Operator y: concatenation".

   moe token 618 (p=0.96) `' concaten'` → "concatenation"
   alleqgt token 741 (p=1.00) `' addition'` → "addition"

3. Final query application becomes `addition(58, 32) = 90 → H J → \`}`. The model then
   emits `\boxed{}}}` (mismatched brace, predicted parsed as `}}`).

**Verdict.** alleqgt's output **does look like** an OLD-method CoT — it is *not* off-method —
but it lands in the wrong sub-template (little-endian / numeric arithmetic) for a
pure-concatenation puzzle. moe stayed in the literal-concatenation sub-template.

---

## 3. Q2 — `24e1f1d5`: moe ✓ → alleq ✗ → alleqgt ✓ → cryptn ✗

GT = `^#(!`. The puzzle has 4 examples; only EX1 (`%!+<} = %!<}`) shows `+`, and the RHS
is literal concatenation. Three of the four runs follow that. Char-by-char of each
prediction relative to GT:

```
GT       : ^   #   (   !
moe ✓    : ^   #   (   !    (concat: EH y JB = EHJB, with E,H,J,B mapped to ^,#,(,!)
alleq ✗  : ^   (   !   #    (concat little-endian: EJ y GB = "9410", reverse-mapped to ^(!#)
alleqgt ✓: ^   #   (   !    (concat: EH y IB = 6782, mapped E→^, H→#, I→(, B→! = ^#(!)
cryptn ✗ : '3' (no boxed answer, hit 7680 tok cap mid-DFS)
```

What flipped between alleq and alleqgt? **Same training row, byte-identical.** They differ
by mid-fine-tune RNG / different `gtTrue` filtering of *other* categories. The roll of the
dice on the digit-permutation lottery favored alleqgt:

```
alleq    letters: ( -> G,  [ -> H,  # -> J     →    EJ y GB = 94 || 10 = 9410 (LE!) → "^(!#"
alleqgt  letters: [ -> G,  # -> H,  ( -> I     →    EH y IB = 67 || 82 = 6782    → "^#(!"
```

alleq additionally turned on the little-endian sub-template, then computed
`concatenation(94, 10) = 9410` and reverse-mapped digit-by-digit, producing the right
*digits* in the *wrong* permutation `^(!#` (a swap of positions 2-3-4).

**alleqgt → cryptn lost it again.** cryptn's NEW-method training kicked in: it allocated
10 symbols A-J including `(`, `+`, `*`. It then attempted a full DFS in the §1 search tree
trying to lock the operator `g`. With the wrong branch (`g = ~sub`) it ran out of the 7680
token budget. The tail of cryptn's output is mid-branch:

```
Try F=9:
  Current State: f=concat_fwd g=[~sub] A={4,6-9} B={5} ...
  EX2: 39 ~sub 35 in [-6,5], RHS -51 in [-15,-15] -> LHS can never equal RHS — no
Try B=6:
  ...
```

No `</think>`, no `\boxed{}` — Kaggle extractor falls back to `3` (digit-fallback heuristic).

Additionally, cryptn's letter-form conversion of EX2 is **buggy**:
training has `EX2 <^*"^ = ["( becomes FI g BI = GBE`, but the model wrote
`FI g BI = -BE`. It hallucinated a leading `-` (negative-sign hangover from
the "If a symbol exists in the result it must be a negative sign" rule it just recited).
That misdirection forced the search into the `g = ~sub` branch, which is wrong.

---

## 4. Q3 — `4d8df95b`: moe ✗ → alleq ✓ → alleqgt ✗

The puzzle is `??-??` with operands all the same; GT = `)`. The training CoT in
moe/alleq/alleqgt is **identical** and ends with:

> `Operator y: rsub_signed` … `rsub_signed(66, 66) = None` … `\boxed{)}`

It is, in the training data, a *guess*: the CoT doesn't compute anything for the query
because both operands collapse to `GG`, and the result is "None". The CoT just emits
`\boxed{)}` because that is the puzzle's recorded GT. The model has no actual reasoning
chain to learn that `)` is the right answer rather than `?` or `-`.

| run     | operator-y label inferred | last computation                 | boxed |
| ------- | ------------------------- | -------------------------------- | ----- |
| moe     | `neg_absdiff`             | `neg_absdiff(66, 66) = None`     | `?`   |
| alleq   | `subtraction`             | `subtraction(77, 77) = 0` → `)` | `)`   |
| alleqgt | `neg_absdiff`             | `neg_absdiff(33, 33) = None`     | `-`   |

Only alleq committed to *plain subtraction* (because it adopted the little-endian
sub-template: `GG = 77`, `subtraction(77,77) = 0`, `J -> )`). The other two stopped at
`None` and fell back to emitting an arbitrary symbol (`?` for moe — the digit that maps
to the "question mark" letter G; `-` for alleqgt — the literal minus sign).

**So alleq landed on the right answer by accident**: little-endian sub-template + numeric
subtraction is *not* what the underlying GT was generated from (the GT was produced by
`rsub_signed`, which is exactly what the training CoT names), but `subtraction(x, x) = 0`
maps `0 -> )` regardless. The accident is that for an all-equal-operands input every
subtraction-family operator yields `0`, and the digit 0 happens to be `)` under alleq's
sampled permutation. Different permutations in moe/alleqgt would have mapped 0 to other
symbols; instead those runs gave up at `None`.

This is an example of an **unteachable puzzle in the OLD training set**: the CoT does not
contain the reasoning to deduce the answer; it just announces it. So whether the model
gets it right depends purely on which digit happens to get mapped to `)`.

---

## 5. Q4 — cryptn on the 3 puzzles; cryptarithm length distribution

| puzzle      | cryptn raw_output | output_token_len | terminated cleanly | predicted |
| ----------- | ----------------- | ---------------- | ------------------ | --------- |
| `24e1f1d5`  | mid-DFS, no boxed | 7680 (CAP)       | no                 | `3` (fallback) |
| `4d8df95b`  | mid-DFS, no boxed | 7680 (CAP)       | no                 | `3` (fallback) |
| `b1b10e83`  | full new-style    | 2584             | yes                | `{\``     |

For `b1b10e83`, cryptn **did** finish in new-style — but its conclusion is wrong because
the model:

- mis-tokenized the puzzle's RHS `)` as a third operator: header reads
  ` + -> f, * -> g, ) -> h`, while training has only `+ -> f, * -> g`.
- assigned only 8 letters A–H instead of 9 (omitting `(` because it had labelled `)` as
  the operator `h`).
- finally applied **operator g (a+b−1)** instead of the question's `+` (which should be
  `f = concat_fwd`), computing `g(53, 67) = 119 → digits 1,9 → {,\`` → answer `{\``.

Across all 55 cryptarithm_deduce val puzzles:

| run     | n_correct | avg output_token_len | median | max  | hit CAP (7680) | no `\boxed{}` |
| ------- | --------- | -------------------- | ------ | ---- | -------------- | ------------- |
| moe     | 2         | 864                  | 737    | 7680 | 1              | 1             |
| alleq   | 1         | 749                  | 739    | 883  | 0              | 0             |
| alleqgt | 1         | 747                  | 740    | 925  | 0              | 0             |
| cryptn  | **0**     | **5 928**            | 6 704  | 7680 | **24**         | **24**        |

So cryptn's average cryptarithm length is **7× larger** than the OLD runs, and **24/55
runs hit the 7680 cap with no answer boxed**.

---

## 6. Q5 — cryptn exclusions (oversampling = 0)

`_build_traincsv.py:42-77`:
```python
CAP = 7680
o['oversampling'] = '0' if int(o['token length']) >= CAP else '2'
def gate(o):
    if int(o['token length']) >= CAP or o['GT-match'] != 'True':
        o['oversampling'] = '0'
```

Of 800 new-solver cryptarithm rows: **265 included (os=2), 535 excluded (os=0).**
Breakdown of the 535:

| exclusion_reason       | count |
| ---------------------- | ----- |
| over_cap only          | 192   |
| over_cap & gt_mismatch | 191   |
| gt_mismatch only       | 152   |

By `new label`:
| new label                 | gt_mismatch | over_cap | over_cap & gt_mismatch |
| ------------------------- | ----------- | -------- | ---------------------- |
| derived_arithmetic        | 65          | 139      | 71                     |
| derived_concat            | 0           | 21       | 0                      |
| guess_operator            | 34          | 28       | 54                     |
| guess_symbol              | 42          | 3        | 32                     |
| guess_operator_and_symbol | 5           | 1        | 12                     |
| blind_fallback            | 6           | 0        | 22                     |

Our three puzzles:
- `24e1f1d5` (derived_concat) — included, os=2, token len 2934.
- `b1b10e83` (derived_concat) — included, os=2, token len 1814.
- `4d8df95b` (derived_arithmetic) — **excluded, os=0, token len 195 352** (>>CAP).

So `4d8df95b` was excluded for over-CAP (the honest DFS over the arithmetic-family
search tree is enormous; the `Try F=…` / `Try B=…` style branching for the inner search
exploded to 195 k tokens). The model never saw a new-method CoT for this puzzle and is
left with whatever generalisation transfers from other arithmetic-family rows.

---

## 7. Q6 — OLD vs NEW cryptarithm CoT for each puzzle

OLD (lkevincc / pretok_0408) is a short symbolic walkthrough that:
- assigns symbols to letters in appearance order,
- assigns digits (in a single line, often a *correct* permutation),
- restates examples in letter form,
- optionally restates them with numeric multiplication (when needed),
- declares "Operator x/y/z: …" with the *correct* operator name,
- applies the rule to the query in one short paragraph and boxes the answer.

NEW (`01_data/260525_Cryptarithm/.../tree_cot.txt`) is a **full DFS trace** with
`Current State: f=… g=… A={…} B={…}` lines and `Try X=…:` branches. It begins with a
"Prior knowledge for this kind of question" preamble that enumerates the noise rules and
the four operator families, and ends with a "Summary: … Now solve the QUERY …" application.

Length comparison (chars):

| puzzle      | OLD CoT chars | NEW CoT chars | NEW / OLD ratio |
| ----------- | ------------- | ------------- | --------------- |
| `24e1f1d5`  | 1 324         | 6 841         | 5.2×            |
| `b1b10e83`  | 1 497         | 4 504         | 3.0×            |
| `4d8df95b`  | 1 264         | 394 547       | **312×**        |

Representative slices.

**OLD `b1b10e83`** (full key middle):
> ```
> Digit mapping:
> A = 5, B = 1, C = 2, D = 3, E = 4, F = 6, G = 7, H = 0, I = 8, J = 9 *(no — actually lkevincc only has letters A-I in this row)*
> Operator x: concatenation
> The question operator is x, which is concatenation.
> concatenation(DA, CH) = DA || CH = DACH
> Converting back: D A C H : D -> |, A -> ", C -> #, H -> $ -> |"#$
> ```

**NEW `b1b10e83`** (representative middle):
> ```
> EX3: BA mul BA in [83, 116]^2 / ... — RHS BGH in [BGH range]
> Current State: f=concat_fwd g=mul_minus1 A={2} B={8} C={9} D={1} E={6} F={5} G={2,4,7} H={0} I={3}
> No equation narrows a variable further. Branch on G in [4,7] (the variable with the fewest candidates left), try in order.
>   Try G=4:
>     Current State: f=concat_fwd g=mul_minus1 A={2} B={8} C={9} D={1} E={6} F={5} G={4} H={0} I={3}
> Conclusion of step §1: {f = concat_fwd, g = mul_minus1, A=2, B=8, C=9, D=1, E=6, F=5, G=4, H=0, I=3}, 11 of 11 resolved. This is the answer.
> Summary: Confirm reading order = rightward, f = concat_fwd, g = mul_minus1
> Now solve the QUERY f(IB, CD), applying f = concat_fwd = a∥b,
> f(IB, CD) = IB∥CD = IBCD; map the letters back to symbols, so the answer is |"#$
> \boxed{|"#$}
> ```

Approach difference: OLD never branches; NEW exhaustively branches over digit
candidates and over operator-family hypotheses until everything is locked.

---

## 8. Q7 — `\boxed{}` parsing & malformed outputs (cryptarithm, n=55)

| run     | has \boxed | empty \boxed | no \boxed | hit CAP (truncated) |
| ------- | ---------- | ------------ | --------- | ------------------- |
| moe     | 54         | 3            | 1         | 1                   |
| alleq   | 55         | 3            | 0         | 0                   |
| alleqgt | 55         | 2            | 0         | 0                   |
| cryptn  | 31         | 3            | **24**    | **24**              |

For cryptn, 24/55 cryptarithm puzzles run past the 7680-token budget without ever
emitting `</think>` or `\boxed{…}` — all of those get the Kaggle metric's fallback (the
puzzle's prompt and the parser pick a default char such as `3`).

---

## 9. Q8 — Token-prob analysis for `b1b10e83` (moe vs alleqgt)

Both model outputs share the first 87 tokens *exactly* (mapping `" -> A`, `\ -> B`, … and
listing `Operators: * -> x, + -> y, - -> z`). The first divergent **token text** is at
index 87, in the digit-assignment line:

```
   tok#  moe                          alleqgt
   ...
   [85]  ' ='   p=0.99999             ' ='   p=1.00000
   [86]  ' '    p=0.99997             ' '    p=0.99975
   [87]  '1'    p=0.14035  <-- moe    '2'    p=0.13503  <-- alleqgt
   [88]  ','    p=0.99939             ','    p=0.99993
```

Both digit-choice tokens have probability ≈ 0.14, which is essentially **uniform over the
~7 available digits** (1/10 = 0.10, with mild bias). The model is *guessing* the digit
permutation; there is no signal from the prompt that pins specific digits, and the
training CoT is just one of many equally-valid permutations. Whether `B=1` (moe) or `B=2`
(alleqgt) gets sampled is essentially RNG.

After the permutation is sampled, the choice of operator-y label is also influenced:

```
moe     token [615..620]:
   'Operator'(1.00) ' x'(1.00) ':'(1.00) ' multiplication'(1.00) '\n'(1.00)
   'Operator'(1.00) ' y'(1.00) ':'(1.00) ' concaten'(p=0.963) 'ation'(1.00)

alleqgt token [738..743]:
   'Operator'(1.00) ' x'(1.00) ':'(1.00) ' multiplication'(1.00) '\n'(1.00)
   'Operator'(1.00) ' y'(1.00) ':'(1.00) ' addition'(p=0.99997) '\n'(0.998)
```

moe is still somewhat unsure of "concaten" (p=0.96), whereas alleqgt is **already
locked-in to addition** with p=0.99997 — because by the time it reaches this point it
has produced multi-line numeric arithmetic like `addition(15, 82) = 97, ABCD = 97` for
EX1 (token 237 first emits `' addition'` at p=0.573, then by 348 the same token has
p=0.999982). The little-endian / "ABCD-as-number" template forced the model down the
addition branch and made the choice essentially deterministic at the operator line.

So the root cause of the b1b10e83 verdict flip is: **alleqgt sampled a digit permutation
that, combined with the little-endian sub-template, made literal-concatenation arithmetically
unsatisfiable; the model then locked in addition as the most likely operator**, computed
58+32=90, and produced `}}` instead of `|"#$`.

---

## 10. Bottom line — root causes (priority order)

1. **The cryptarithm answer for free-permutation puzzles depends on RNG over the
   digit-permutation lottery**, not on reasoning. moe / alleq / alleqgt are *not*
   actually deducing digits — token probabilities at the digit-assignment line are
   ≈ 0.14 each (near-uniform). Whether the model lands on a permutation that makes
   `concat` or `add` numerically consistent is luck. This explains all three
   OLD-run flips: `b1b10e83` moe ✓ vs alleqgt ✗, `24e1f1d5` alleq vs alleqgt,
   and `4d8df95b` alleq's accidental ✓.

2. **alleq/alleqgt drifted into the "Reading order: little-endian" sub-template
   far more often than moe** (38/55 and 26/55 vs 0/55 for moe). That sub-template
   forces `AB=15` to be interpreted as a numeric value, which converts pure-concatenation
   puzzles into addition/subtraction satisfaction problems and makes them
   *almost guaranteed wrong*. Tokens like `' addition'` lock in at p≈1.0 within the
   second example. The OLD CoT distribution itself has 302 little-endian rows, so
   this is *consistent with* training — but the *base-rate* the model emits the
   template is higher than the training base-rate, and the consequences are bad for
   pure-concat puzzles.

3. **cryptn's NEW-CoT pipeline is mis-calibrated for cryptarithm at inference time.**
   (a) 535 / 800 new-solver rows are excluded from training because they exceed the
   7680-token CAP or have GT-mismatch; in particular `4d8df95b`-type arithmetic puzzles
   are over-represented in the excluded set (210 of 275 derived_arithmetic rows). The
   model therefore has thin training coverage of long arithmetic DFS. (b) At inference,
   the model's 7680-token budget is the *same* as the training CAP, so any puzzle
   whose honest DFS exceeds that budget cannot be solved by faithful reproduction —
   24 / 55 cryptarithm val runs hit the cap with no boxed answer. (c) Even when it
   finishes (`b1b10e83`, cryptn), the model commits parsing bugs — e.g. labelling the
   symbol `)` as a third operator `h` and dropping a symbol from the letter assignment —
   that doom the answer. cryptn ends with 0/55 cryptarithm accuracy despite training
   on faithful full-search CoTs.

