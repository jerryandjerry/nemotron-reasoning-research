# Study — CoT quality (lkevincc vs huikang) + training order

> Follow-up to EVALUATION.md after seeing the per-category eval table showing
> bit_manipulation collapsed from 74.4% to 5.0% despite identical training data.

## Study #2 first (because the answer is clean): training order

**Hypothesis:** if the 0.68 run scheduled bit_manipulation samples differently from the
0.84 run — for instance bunched toward the start — the linear LR decay tail would let
the model drift away from it before training ends.

**Method:** parsed the per-step `sample_id` log lines from both runs' `train_log.txt`,
mapped each id to its category, computed per-decile category distributions.

**Result:** bit_manipulation distribution by training decile (% of samples in that decile):

| Decile | 0.84 run | 0.68 run |
|---|---|---|
| 0–10% | 22.6% | 22.2% |
| 10–20% | 21.0% | 22.2% |
| 20–30% | 21.4% | 22.4% |
| 30–40% | 24.0% | 22.4% |
| 40–50% | 21.2% | 22.3% |
| 50–60% | 22.4% | 22.4% |
| 60–70% | 22.1% | 22.4% |
| 70–80% | 23.2% | 22.3% |
| 80–90% | 23.4% | 22.5% |
| 90–100% | 22.8% | 22.4% |

The 0.68 run (stratified-interleave with seed 42) is actually *more uniform* than the
0.84 run (huikang index.jsonl replay). In the last decile — where LR is lowest and
"drift" would be most visible — both runs sit at ≈22.5% bit_manipulation. Identical for
practical purposes.

**Conclusion:** order is not the cause of the bit_manipulation collapse. The hypothesis
is falsified.

## Study #1: CoT quality

Two sources of cryptarithm CoTs in the golden CSV:
- `pretok_0408_decoded` — huikang's solver. 65 rows.
- `lkevincc_golden` — lkevincc's solver. 735 rows.

Both labeled as `cryptarithm_deduce` or `cryptarithm_guess` after the recent
category re-mapping.

### Length and structure

| | huikang | lkevincc |
|---|---|---|
| Mean CoT length (chars) | 1,334 | 1,662 (+25%) |
| Median | 1,341 | 1,682 |
| Std | 139 | 195 |
| In-CoT `\boxed` ↔ answer column match (Kaggle regex) | 65 / 65 | 735 / 735 |

Both formats are internally correct: the answer in the `\boxed{}` matches the answer
column for every sample (using the new Kaggle metric extractor). Random verification
of 5 lkevincc rows shows 0 arithmetic errors across their stated digit operations.

### Solver scope (operator vocabulary)

Keyword counts across all rows in each source:

| Operator keyword | huikang | lkevincc |
|---|---|---|
| concatenation | 65 | 144 |
| multiplication | 0 | 394 |
| multiply then add 1 | 0 | 79 |
| multiply then subtract 1 | 0 | 82 |
| addition | 0 | 546 |
| subtraction | 0 | 256 |
| absolute difference | 0 | 258 |
| little-endian (mode) | 0 | 302 |

**Huikang's solver only handles concatenation.** It never identifies any other
operator. On the 65 cryptarithm_deduce rows, the question operator was classified as
"concatenation" 43 times. The remaining 11 cryptarithm_guess rows hit the "unknown"
fallback: solver writes `As the question operator is unknown, we default to
concatenation`, then outputs the concatenation result anyway. (And the answer column
agrees, meaning the actual puzzle answer for those was the concatenation output — so
the fallback is, by construction, "correct enough" on these specific 11 puzzles.)

**Lkevincc's solver is a much wider system.** It enumerates many candidate operations,
checks each against the examples (with `✓` marks), picks the consistent one, then
computes the answer. It also handles little-endian numeric reading on top of any
operator.

### Reasoning template differences

Lkevincc adds three structural features that huikang's CoTs lack:

1. **Digit mapping line** — `A = 5, B = 6, C = 4, D = 9, ...`. Required for arithmetic
   verification. Doesn't exist in huikang CoTs.
2. **Reading order line** — `Reading order: little-endian (units digit first)` for
   ≈40% of lkevincc CoTs. Doesn't exist in huikang.
3. **Verification marks** — every example transformation ends with `✓` after the
   computed value matches the example output. Doesn't exist in huikang.

### Side-by-side same-style sample

Same prompt shape — 4 example transformations, find the rule, apply to new input.

**Huikang (id=b1b10e83, answer `|"#$`):**
- Assigns letters to symbols.
- Lists examples in letter form.
- For each example, says input→output gives "AB x CD = ABCD" pattern (concatenation).
- For one example, the output doesn't fit concatenation → marks that operator as
  "unknown".
- Question uses the concatenation operator → applies concatenation → answer.

**Lkevincc (id=00457d26, answer `@&`):**
- Assigns letters AND digits (A=5, B=6, ...).
- Converts each example to letter form, computes numeric inputs/outputs.
- For each example, finds the operation that maps input numbers to output number,
  verifies with `✓`.
- Identifies all operators by name (`multiply then add 1`, `absolute difference`).
- For the question's operator (absolute difference), applies it → answer.

### Which is better?

For **the 0.84 baseline alone** (concatenation-only training):
- Huikang's reasoning is simpler and more reliable on concatenation puzzles.
- For test cryptarithm puzzles that ARE concatenation puzzles → 0.84 baseline works.
- For test puzzles that aren't concatenation → 0.84 baseline can't solve them.
- 0.84 score implies most of the public-test cryptarithm puzzles are concatenation-
  shaped (or the cryptarithm category is small in the public test).

For **the 0.68 run** (mostly lkevincc + 65 huikang):
- The model sees the same prompt shape with **two contradictory reasoning patterns**:
  - Huikang style (1 of 12): "find the symbol mapping, apply concatenation; if not
    concatenation, default to concatenation anyway"
  - Lkevincc style (11 of 12): "find letter+digit mappings, try multiple arithmetic
    operations, verify with ✓, apply the consistent one"
- Lkevincc dominates 92% of cryptarithm training, so the model would mostly learn
  lkevincc's multi-operation enumeration.
- On test puzzles that ARE concatenation, lkevincc's reasoning over-engineers — it
  tries arithmetic first, which can succeed but also can mislead (e.g., picking the
  wrong operator from the candidates).
- The 88 originally "broken" lkevincc rows (answers containing `}`) become correct
  under Kaggle's regex, but the model still has to learn to output the literal extra
  `}` after the answer body in its `\boxed{}`. That's an unusual structural pattern
  that doesn't appear anywhere else in training.

**Verdict:** lkevincc CoTs are correct in their own right and are arguably more
capable. But they are NOT a drop-in replacement for huikang CoTs — they teach a
different reasoning template. Mixing them into a corpus that previously had only
huikang-style cryptarithm reasoning introduces template inconsistency.

### Implication for the bit_manipulation collapse

CoT quality and training order don't directly explain why bit_manipulation collapsed
from 74.4% to 5.0% in the 0.68 run when its training data was byte-identical.

The remaining plausible mechanism is **template/format pattern bleed into other
categories via MoE expert re-routing**:
- The lkevincc CoTs introduce three distinctive structural patterns (digit mapping,
  reading order, ✓ marks) that didn't exist anywhere in the 0.84 corpus.
- The 30B MoE model has 128 experts with weight tying. Adding a new template
  reshapes which experts fire on which token patterns.
- If experts that previously specialized on bit_manipulation reasoning got
  re-routed toward absorbing the lkevincc template, those experts would lose their
  bit_manipulation capability.
- At inference, bit_manipulation prompts route to the same experts, which now respond
  with degraded outputs — consistent with a category-specific collapse (74.4 → 5.0)
  rather than uniform degradation.

I cannot verify this mechanism without inspecting the 0.68 model's actual outputs on
bit_manipulation prompts (or running an MoE routing analysis). The hypothesis is
consistent with what we see — the categories where training data changed least
(numeral, unit_conversion — both unchanged) didn't drop, while categories whose
prompts share more token structure with cryptarithm (bit_manipulation has letter-like
binary patterns) dropped the hardest.

## Update — inspected actual inference outputs

After writing the above I got `val_eval_results.csv` from both runs (160
bit_manipulation prompts each, identical inputs). The template-bleed hypothesis is
**partially confirmed**, and the bit_manipulation collapse mechanism is now visible.

### Where the lkevincc template actually bled

Searched the 0.68 raw outputs for the three distinctive lkevincc patterns
(`Digit mapping`, `✓`, `Reading order: little`):

| Category | 0.68 outputs with lkevincc template | Δacc |
|---|---|---|
| bit_manipulation | **0 / 160** | −69.4pp |
| equation_transformation | **55 / 155 (35.5%)** | −5.8pp |
| gravitational_constant | 0 / 160 | −3.1pp |
| number_conversion | 0 / 158 | 0 |
| text_encryption | 0 / 158 | −8.9pp |
| unit_conversion | 0 / 159 | 0 |

Template bleed is concentrated in **equation_transformation**, exactly the category
whose prompt shape is closest to cryptarithm (Alice's Wonderland equation rules).
35% of the model's equation_transformation outputs now reach for the
`Digit mapping: A = 5, ...` template — that's a real cognitive contamination, and
it correlates with the −5.8pp on that bucket.

For **bit_manipulation**, the template does NOT bleed. The model still uses the
correct bit_manipulation reasoning format (bit columns, Identity/NOT/AND/OR/XOR
operators). So the surface-level template hypothesis fails here. The collapse is
caused by something else.

### What actually breaks bit_manipulation in the 0.68 model

I diffed the same prompt (id=e915879a, ground_truth `00100000`) across both runs and
ran failure-mode statistics on all 160 bit_manipulation outputs:

| Metric | 0.84 run | 0.68 run |
|---|---|---|
| Reaches `\boxed{}` answer | 154/160 (96%) | **102/160 (64%)** |
| Reaches `</think>` close | 154/160 | 102/160 |
| Reaches the "Matched" / "Selected" section | 153/160 | **84/160 (52%)** |
| Mean successful operator identifications (`XOR<N>` count) per output | **5.2** | **0.9** |
| Mean count of "Identity absent, NOT absent ..." lines per output | 0.8 | **2.0** |
| Output token length (mean / max) | 6,677 / 7,680 | 6,932 / 7,680 |
| Accuracy: outputs that reached `\boxed{}` | 119/154 (77%) | **8/102 (7.8%)** |
| Accuracy: outputs without `\boxed{}` | 0/6 | 0/58 |
| **Total accuracy** | **119/160 (74.4%)** | **8/160 (5.0%)** |

The same input shows it clearly. Identical setup (input/output enumeration, bit
columns, identity computation). 0.84 ends with:

```
Matched
0 I5
1 XOR60
2 XOR71
... (8 confident assignments)
Selected
... (applies them)
\boxed{00100000}
```

0.68 ends with:

```
?5? - Identity absent, NOT absent, Constant absent, AND absent, OR absent,
      XOR 23 32, AND-NOT absent, OR-NOT absent, XOR-NOT absent
?6? - Identity absent, NOT absent, Constant absent, ...
(token limit hit)
```

The 0.68 model **knows the bit_manipulation reasoning format** — same prelude, same
operator vocabulary. But at the operator-matching step it consistently fails to
commit to a match. It enumerates "absent / absent / absent" across many bit
positions and runs out of tokens. The 0.84 model averaged 5.2 confident
`XOR<N>`-style matches per output; the 0.68 model averages 0.9. The matching
sub-skill is broken.

When the model does manage to write a `\boxed{}` answer, it's wrong 92% of the time
(8/102). When it doesn't (58/160), the regex pulls a garbage number out of the
truncated tail (`6`, `5`, `75`, `60`, single digits) and that's the "predicted".

### Refined hypothesis

**The bit_manipulation reasoning machinery wasn't replaced — it was un-confidenced.**

The model still outputs bit_manipulation-shaped text, never lkevincc-shaped text.
But the OPERATOR-MATCHING sub-skill (look at bit column, decide which Boolean
function applies) degraded sharply: from 5.2 successful matches per puzzle on
average to 0.9.

Plausible cause: the lkevincc CoTs train a habit of *enumerate-then-verify* across
many candidate operators (multiply, multiply+1, addition, subtraction, abs-diff,
etc.) before committing. That habit may have transferred to bit_manipulation —
where the model now also tries to enumerate-then-verify across (Identity, NOT,
Constant, AND, OR, XOR, AND-NOT, OR-NOT, XOR-NOT) but can't commit to one. Same
prompt, same training data for the category, but the model's *commitment threshold*
or *expert routing* for matching shifted.

This is consistent with MoE weight-tying: with 128 experts and tied LoRA grads
across the MoE, adding new training samples in another category can quietly
re-weight what experts do under similar token patterns. The bit_manipulation
samples didn't change, but the experts that process them now ALSO have to handle
lkevincc-style enumeration, and they've lost their confident-match behavior.

### What this means for the score

- equation_transformation −5.8pp ≈ lkevincc template directly bleeding into a
  surface-similar category.
- bit_manipulation −69.4pp ≈ a deeper sub-skill (operator commitment) breaking
  under interference, even though the surface template stays clean.
- text_encryption −8.9pp and gravitational_constant −3.1pp likely share the same
  un-confidence mechanism on a smaller scale (those categories also use
  rule-pattern matching).
- number_conversion and unit_conversion unchanged → those don't require
  pattern-matching commitment; they just transform a value.

### Practical implications (updated)

1. The pure-regex-fix experiment (new study folder
   `2605150937_sft_huikang_golden_kaggleregex_rtx6000`) **will not fix the
   bit_manipulation collapse** — the regex change has nothing to do with the
   operator-matching sub-skill that broke. Expectation: still in the 0.68-ish range
   on bit_manipulation, possibly tiny improvements elsewhere.

2. To recover the score:
   - **Drop lkevincc entirely** + regex fix + fixed huikang_update CSV. Expected:
     reproduce ≈0.84.
   - **Strip lkevincc structural features before training** — remove the
     `Digit mapping:` line, the `Reading order: little-endian` line, and the `✓`
     verification marks from every lkevincc CoT, leaving just the prose reasoning.
     Tests whether the structural features are what re-routes the experts.
   - **Two-stage training** — train baseline first, then continue-train on
     lkevincc only for a short tail. The bit_manipulation sub-skill is locked in by
     the baseline phase before lkevincc can interfere.

3. The original training script line that does the post-`</think>` boxed
   construction needs the Kaggle-regex change regardless, since the lkevincc data
   has answers containing `}`. That fix is already in the new study folder.
