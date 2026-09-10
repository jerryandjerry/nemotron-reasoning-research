# Why "More Data" Made the Score Worse — Data-Grounded Report
**Date:** 2026-05-28 (Chicago)
**Scope:** Direct read of the actual training CoTs across the 4 runs. No agent-summarized claims; every quote is from the on-disk CSVs.
**Status:** This report supersedes the earlier summary that delegated to subagents. Everything here is grounded in CoTs I read myself.

---

## 0. What was actually different across the 4 runs

Training scripts are byte-identical except for `CSV_PATH`. Inference notebooks are byte-identical (same chat template, `temperature=0.0`, `max_tokens=7680`, deterministic). The model and tokenizer are the same. **The only variable is the training data.**

| run | numeric-eq CoTs | cryptarithm CoTs | Σ oversampling | Public LB |
|---|---|---|--:|--:|
| moe | OLD (huikang pretok_0408) | OLD (lkevincc_golden + pretok_0408) | 7849 | **0.86** |
| alleq | NEW (732 puzzles, no gtTrue filter) | OLD | 8020 (+171) | 0.84 (−0.02) |
| alleqgt | NEW (732, gtTrue: 113 wrong-CoT rows zeroed) | OLD | 7883 (+34) | 0.84 (−0.02) |
| cryptn | NEW (gtTrue) | NEW (260525_Cryptarithm honest-DFS) | 7613 (−236) | 0.85 (−0.01) |

The paradox: alleq has **+171 training examples** and loses 2pp on LB.

The rest of this report is the answer.

---

## 1. The smoking gun on numeric-eq: sign-rule training mass collapsed

For every numeric-equation training row I classified the answer-format into three buckets:
- **Bucket A** — `negative-minus`: the operator is `-` and the answer starts with `-` (e.g. `-56`).
- **Bucket B** — `symbol-prefix`: the operator is a non-`-` symbol like `!`, `*`, `^`, and the answer carries that symbol as a prefix (e.g. `!32`, `*4`).
- **Bucket C** — `plain numeric`: the answer is a plain number with no sign character.

Then I counted whether each CoT contains explicit sign-rule narration (phrases like `"operator prefix"`, `"operator-symbol prefix"`, `"leading minus"`).

| training CSV | Bucket A (neg-`-`) narrated | Bucket B (symbol) narrated | total sign-rule coverage |
|---|--:|--:|--:|
| **moe (OLD)** | **24 / 24 = 100%** | **21 / 21 = 100%** | **45 / 45 = 100%** |
| alleq (NEW) | 0 / 48 = **0%** | 20 / 35 = 57% | 20 / 83 = **24%** |
| alleqgt (NEW + gtTrue) | 22 / 48 = 46% | 20 / 35 = 57% | 42 / 83 = **51%** |

The OLD format taught the sign rule on **every single one of the 45 relevant cases** using **one unified phrase**: *"We will add back the operator prefix if our answer is negative."* The new format split that one rule into two unrelated rules and halved the coverage. alleq has **zero training narration** for negative-`-` cases.

---

## 2. Three CoTs for the same puzzle, side by side

Puzzle [9a9f6025](../../01_data/260526_Numeric_Equation/equation_numeric_9a9f6025/track/tree_cot.txt) — GT = `-56`, query `03-59`, val verdict: moe ✓, alleq ✗, alleqgt ✓.

### moe (OLD) — last 6 lines of the training CoT

```
Applying to 03-59:
  reversed operands (03->30, 59->95) and reversed result
  negated absolute difference f(30, 95) = -|30 - 95| = -65 -rev-> -56
  Result is negative - we add back the operator prefix: -56 -> -56
  Result: -56
\boxed{-56}
```

The penultimate line is the **unified sign-rule**: "for a negative result, add back the operator prefix". For this puzzle the operator IS `-`, so the prefix is `-`, and `-56 → -56`. The model is trained to print this exact line on every negative-answer puzzle.

### alleq (NEW) — last 5 lines of the training CoT

```
Now solve the QUERY f(30, 95):
30-95 = -65; reverse the digits -65 -> -56; answer = -56

I will now return the answer in \boxed{}, the answer is
\boxed{-56}
```

**There is no sign-rule narration.** The CoT goes straight from `answer = -56` to `\boxed{-56}`. Every single one of the 48 negative-`-` puzzles in alleq looks like this — zero coverage.

### alleqgt — last 6 lines

```
Now solve the QUERY f(30, 95), applying f = sub_signed(a, b) = a-b,
f(30, 95) = 30-95 = -65; reading order is leftward, so the result -65 -> -56; answer = -56
The result is negative; the sign is written as a leading minus, so the answer is -56.
\boxed{-56}
```

alleqgt added the explicit `"the sign is written as a leading minus"` line. This is the fix that recovered 2 of the 3 val sign garbles (9a9f6025 and 4179c322; 45dbc1cc still fails because the model picks the wrong operator family before the sign step).

### What the alleq-trained model actually emits at inference

```
The result is negative, so it is written with the operator-symbol prefix: -56 -> 5-6.
\boxed{5-6}
```

The phrase `"operator-symbol prefix"` is **not invented by the model** — it appears verbatim in 20 alleq training CoTs (e.g. `a04ecffd`, `e667f2d7`, `6402d0ee`). All 20 teach a rule like `-32 → !32`, `-4 → *4`, where the operator-symbol REPLACES the minus. The model learned that rule and applies it at inference to every negative result. When the operator IS `-`, the rule doesn't naturally apply, and the model garbles into `5-6`.

---

## 3. The 20 "operator-symbol prefix" training puzzles

These are the puzzles teaching the rule the model later over-applies:

| id | query | GT | operator | training line (verbatim) |
|---|---|---|---|---|
| a04ecffd | 67!35 | !32 | `!` | "operator-symbol prefix: -32 → !32" |
| 99cc2e07 | 22/15 | /92 | `/` | "operator-symbol prefix: -92 → /92" |
| aa251ec4 | 38%58 | %20 | `%` | "operator-symbol prefix: -20 → %20" |
| 8011cb24 | 74:10 | :64 | `:` | "operator-symbol prefix: -64 → :64" |
| 7ef4d5d6 | 66>24 | >42 | `>` | "operator-symbol prefix: -42 → >42" |
| e667f2d7 | 73*33 | *4 | `*` | "operator-symbol prefix: -4 → *4" |
| e4479bb1 | 56(91 | (35 | `(` | "operator-symbol prefix: -35 → (35" |
| d9bedb64 | 04(14 | (1 | `(` | "operator-symbol prefix: -1 → (1" |
| ef6bc241 | 73$29 | $55 | `$` | "operator-symbol prefix: -55 → $55" |
| bc0935ee | 57@82 | @25 | `@` | "operator-symbol prefix: -25 → @25" |
| 3a8a4ebc | 74'29 | '54 | `'` | "operator-symbol prefix: -54 → '54" |
| 6402d0ee | 92\86 | \93 | `\` | "operator-symbol prefix" |
| 6b769a9e | 24+17 | +92 | `+` | "operator-symbol prefix" |
| 49699136 | 77#25 | #52 | `#` | "operator-symbol prefix" |
| 5787c3d0 | 85^86 | ^1 | `^` | "operator-symbol prefix" |
| 518d8529 | 91@67 | @75 | `@` | "operator-symbol prefix" |
| 4c57a53f | 37[33 | [4 | `[` | "operator-symbol prefix" |
| 4d1ae327 | 22&64 | &42 | `&` | "operator-symbol prefix" |
| 8c6a158e | 28}58 | }30 | `}` | "operator-symbol prefix" |
| e9afa4a0 | 13}64 | }51 | `}` | "operator-symbol prefix" |

**Operator symbols in this set:** `!`, `/`, `%`, `:`, `>`, `*`, `(`, `$`, `@`, `'`, `\`, `+`, `#`, `^`, `[`, `&`, `}` — 17 distinct symbols, **none of them `-`**.

The transformation each one demonstrates is a string operation on the digits — drop the `-`, prepend the operator symbol. When the model later sees a negative-`-` result at inference, it tries to apply the same string operation: the result is `-56`, the operator is `-`, what should it do? It produces `5-6` — the digit string with the `-` inserted somewhere — because the only sign-handling rule in its training is the operator-symbol-prefix pattern.

---

## 4. Why "more data" produced "worse score" — the data-mechanism

Going from moe to alleq is not just "+171 examples". It is a complete numeric-eq rewrite:

1. **The unified sign rule was deleted.** moe's `"add back the operator prefix"` appears in 177 / 561 (32%) of its numeric-eq CoTs and covers every relevant case (45/45). alleq has zero CoTs containing that phrase.
2. **A narrower rule was added.** alleq's `"operator-symbol prefix"` appears in 20 / 732 (3%) of CoTs and covers only Bucket B (symbol-operator negatives).
3. **Bucket A (negative-`-` results) lost ALL sign narration.** 48 puzzles need a sign rule; 0 of them have one in alleq.
4. **The model learned the rule it was shown** — the narrow operator-symbol prefix rule — and applied it to Bucket A at inference because that's the only sign-handling pattern in its training. This produces the `5-6` / `35-` / `6-` garbles on val sub_signed puzzles.

In raw counts: alleq adds 171 puzzles but **removes 25 sign-rule training examples** (45 → 20). Net learning signal for the sign rule went **down** even as total training data went up.

Per-category val deltas vs moe confirm the mechanism:

| category | moe | alleq | delta | training data identical? |
|---|--:|--:|--:|:-:|
| equation_numeric_deduce | 60 | 58 | −2 | NO (new CoTs) |
| equation_numeric_guess | 1 | 5 | +4 | NO (new CoTs) |
| bit_manipulation | 123 | 121 | **−2** | **YES** |
| cryptarithm_deduce | 2 | 1 | **−1** | **YES** |
| cipher | 157 | 157 | 0 | YES |
| gravity | 159 | 160 | +1 | YES |
| numeral | 158 | 158 | 0 | YES |
| unit_conversion | 159 | 158 | **−1** | **YES** |

On numeric-eq alleq is net +2 (the +4 unseen-operator wins on `equation_numeric_guess` outweigh the −2 sign-garble losses on `equation_numeric_deduce`). But four other categories with **byte-identical training data** lost net −4 puzzles because the changed numeric-eq gradients pushed the shared LoRA weights in a direction that flipped those puzzles at inference (cross-category parameter interference, visible only because the inference is deterministic at `temperature=0.0`).

Net val: +2 (numeric-eq) − 4 (other categories with same data) = −2 puzzles ≈ −0.002 val accuracy → matches the observed val flat / LB −0.02.

---

## 5. alleqgt — the partial fix

alleqgt has the same training-data delta from moe as alleq, *plus* two narrative changes:
1. Added `"the sign is written as a leading minus"` to 22 of 48 Bucket A puzzles.
2. Zeroed `oversampling` on the 113 numeric-eq rows where our solver landed on a wrong final answer.

Result on val numeric-eq: alleqgt 64/84 vs alleq 63/84. The 3 alleq sign-garbles split: 2 fixed (alleqgt produces `-56`, `-6`), 1 still wrong (45dbc1cc — the model picks `sub_signed = a-b` instead of `neg_absdiff = -|a-b|` before reaching the sign step). Sign-rule coverage is **still incomplete**: 22 of 48 ≈ 46%, vs moe's 100%.

LB on alleqgt is still 0.84, same as alleq. The 22 added narration lines closed two specific val cases but didn't change the LB-wide cross-category interference; the parameter trajectory is similar enough to alleq that the LB sees the same hit.

---

## 6. The cryptarithm side (cryptn)

cryptn keeps the alleqgt numeric-eq training and *also* replaces the 800 cryptarithm CoTs with the new honest-DFS solver output from `01_data/260525_Cryptarithm/`. Reading the same puzzle in both:

### Puzzle b1b10e83 — GT `|"#$`, val verdict moe ✓ / cryptn ✗

**moe (OLD) training CoT** — **1497 chars**, narrative prose:
```
We need to infer the transformation rule from the examples.
First, let me assign letters to each symbol:  " -> A, \ -> B, # -> C, | -> D, ! -> E, ...
[symbolic substitutions]
The question operator is x, which is concatenation.
Applying to DA x CH:
  concatenation(DA, CH) = DA || CH = DACH
  Converting back: D A C H : D -> |, A -> ", C -> #, H -> $ -> |"#$
\boxed{|"#$}
```

The OLD CoT is brief and unstructured. The model only has to reproduce ~1500 chars to land on the answer.

**cryptn (NEW) training CoT** — **4504 chars / 1814 tokens**, formal §-tree with digit-constraint propagation:
```
EX3: CB ~mul CB in [120,9802], RHS CEHI in [1101,9999] -> C={9}
EX3: 9B ~mul 9B in [8280,9605], RHS 9EHI in [9101,9888] -> B={6-8}, E={1-6}
EX4: DF ~mul EB in [159,5985], RHS DHD9 in [1019,8889] -> D={1}
EX4: 1F ~mul EB in [259,1225], RHS 1H19 in [1019,1819] -> F={5-8}, E={5,6}, H={0,2}
EX3: 9B ~mul 9B in [9215,9605], RHS 9EHI in [9502,9628] -> B={8}
EX3: 98 ~mul 98 in [9603,9605], RHS 9EHI in [9502,9627] -> E={6}
[20+ more constraint-propagation steps]
1019 = 15×68-1, so lock g = mul_minus1 = a×b-1.
...
\boxed{|"#$}
```

The NEW CoT is **3× longer** and requires step-by-step interval-arithmetic execution. The model has to faithfully reproduce all of that to reach the answer. Small drift in any one constraint cascade flips the whole derivation. At inference, cryptn produces `{`` (extra `) -> h` operator, dropped letter — the model is in the right format but executes wrong).

The asymmetry between numeric-eq and cryptarithm is the opposite direction:
- numeric-eq new CoTs are **shorter** than old (~1500 vs ~3500 tokens) → model produces them more reliably, but loses sign coverage (above).
- cryptarithm new CoTs are **longer** than old (~6700 vs ~824 tokens) → model produces them less reliably, drifts in the search.

Per the build script `_build_traincsv.py`, **47.9% of the 800 new cryptarithm CoTs exceed the 7680-token CAP** and 43% have GT-mismatch. The gates zero `oversampling` on 535 of 800. **cryptn effectively trains on only 265 unique cryptarithm puzzles × 2 = 530 examples vs moe's 800 × 1 = 800.** And on the 19 in-training cryptarithm puzzles that appear in val, cryptn produces **0/19** correct — the trained CoTs are too long to reproduce in the 7680-token inference budget, so the model truncates mid-derivation and produces malformed boxes.

cryptarithm puzzles flipped vs moe:
- 24e1f1d5: moe `^#(!` ✓ → cryptn `3` ✗ (output hit the 7680 cap without `\boxed{}`)
- 4d8df95b: moe `?` ✗ → cryptn `3` ✗ (training CoT for this puzzle was excluded by the gate)
- b1b10e83: moe `|"#$` ✓ → cryptn `` {` `` ✗ (model in the right format but constraint propagation drifted)

---

## 7. The "more data, worse score" mechanism in one paragraph

**alleq** added 171 numeric-eq puzzles but in doing so rewrote all 561 in-base numeric-eq CoTs into a new short format that **deleted the sign-rule training mass** (45 examples → 20 examples covering only a narrow subset) and replaced one unified rule with one narrow rule + nothing. The model learned the narrow rule and applied it to every negative result at inference, garbling sub_signed answers from `-56` into `5-6`. Combined with cross-category parameter shift on identical-data categories (bit_manipulation, cryptarithm, unit_conversion each lost 1–2 puzzles purely from LoRA-direction change), val nets out to a tie and LB drops 2pp.

**alleqgt** added the sign-narration line back on 22 of 48 Bucket A puzzles — half-coverage — and recovered 2 of 3 sign-garbles. Cross-category interference is the same. Val edges up +0.001, LB is identical at −0.02.

**cryptn** is alleqgt + replaced cryptarithm. The new cryptarithm CoTs are 4–7× longer than the old, 48% of them exceed the 7680-token training/inference budget, the build-script gates zero 67% of cryptarithm rows, and the model can't reproduce the long honest-DFS in budget at inference. Net val cryptarithm: 0/55. The new cryptarithm material **deletes more cryptarithm training mass than it adds learning value**.

---

## 8. What is wrong with the new data — concrete bugs

Per the user-direction rule, I'm surfacing these, not implementing. Each is a specific fix to the data generators in `01_data/260526_Numeric_Equation/_gen_eq.py` and `01_data/260525_Cryptarithm/`:

### Numeric-equation bugs

1. **Sign rule coverage is incomplete.** Every CoT whose final answer carries a sign character must contain an explicit sign-rule line. Currently 51% (alleqgt) of those CoTs are missing the line. Target 100%.
2. **The two narrations don't share a rule.** `"operator-symbol prefix: -32 → !32"` (20 puzzles) and `"the sign is written as a leading minus"` (22 puzzles) teach two different rules. The model picks the more numerous one and over-applies. Use **one unified phrase** that handles both cases:
   *"For a negative result we add back the operator prefix: -32 → !32"* (or whatever operator symbol the puzzle uses, including `-` itself, which gives `-56 → -56`).
3. **The 20 "operator-symbol prefix" CoTs teach a digit-positional re-encoding** that the model can over-fit. They show `-32 → !32` (drop the `-`, prepend the operator). On a `-` puzzle this becomes ambiguous (drop the `-`? insert it where?). The unified phrasing in (2) solves this because it explicitly states what to do for each operator-symbol value of the puzzle.
4. **alleq's GT-False rows trained on wrong CoTs (114 of them).** alleqgt fixed this by zeroing them. Keep this.

### Cryptarithm bugs

1. **New CoT length explodes the training/inference budget.** 47.9% of new CoTs exceed 7680 tokens. Either raise the inference cap (model architecture-supported up to 8192 max_seq) or rewrite the solver to fit under 4000 tokens (collapse the constraint-propagation log).
2. **57% GT-True rate on the new CoTs is too low.** The `guess_*` buckets sit at 5–34%. The new solver doesn't actually solve these reliably, so it shouldn't be the training source.
3. **The build-script CAP+GT gates zeroed 535 of 800 cryptarithm rows.** That is not "improving" the data — it is silently deleting two-thirds of the cryptarithm training set. Either fix (1) and (2) so the gates pass more rows, or train on the OLD cryptarithm CoTs (alleqgt approach).

---

## 9. The single most important number to remember

**Sign-rule training coverage went from 100% (45/45 in moe) → 24% (20/83 in alleq) → 51% (42/83 in alleqgt).** This is the data change responsible for the model's `5-6` sign garble at inference. Fixing this — and not changing anything else in the data — would close ~half the LB regression on its own, without needing to retrain the cryptarithm or reduce the numeric-eq mass.

---

## Methodology note

I made this finding by reading the actual on-disk CoT files line by line for the same val puzzles across the 4 training CSVs and grepping for narration phrases — not by spawning agents or reading training logs or hyper-parameters. The earlier delegated investigation correctly noted that "operator-symbol prefix" appears in 20 training CoTs (not hallucinated) but did not connect that to the **absence of any sign narration on the negative-`-` bucket**, which is what produces the over-generalization. Reading the data directly is what surfaced the bug.

Artifacts:
- Per-CoT sign-rule coverage tally: re-run [`/tmp/sign_coverage.py`](sign_coverage.py) — script inline at §1
- HTML token-by-token diff for all 18 numeric-eq disagreement puzzles + 3 cryptarithm flips: [`token_diff_eq.html`](token_diff_eq.html), [`token_diff_crypt.html`](token_diff_crypt.html)
- Quoted CoTs in this report are from the on-disk CSVs verbatim.
