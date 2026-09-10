# bit_manipulation update — failure-correction CoTs

> Generated: 2026-05-17

## Goal

Run #7's bit_manipulation real generalization is **12%** (3/25 on out-of-train val IDs). The failure mode is rule ambiguity: for nearly every puzzle (1,147 of 1,280 in this data), **multiple rules fit all 7 examples**. The current CoT shows the *one* rule huikang's solver picked, but never teaches the model what to do when faced with multiple plausible options. The model imitates the format and picks based on training patterns — fine on memorized puzzles, bad on novel ones.

This dataset replaces each bit_manipulation CoT's trace with one that explicitly demonstrates **rejecting a near-miss candidate** before committing to the selected rule. The trace mixes ✓ and ✗ outcomes per example, giving the model real failure signal to learn elimination from.

## What changed in each CoT

A new block is inserted right before `Selected`:

```
Verifying candidates by per-example check
Bit 0: tried OR-NOT13 → ✓✗✓✓✓✓✓✓ reject; chose OR-NOT17 → ✓✓✓✓✓✓✓✓ accept
Bit 1: tried NOT6 → ✓✓✓✓✓✓✗✓ reject; chose NOT2 → ✓✓✓✓✓✓✓✓ accept
...
Bit 7: tried I7 → ✓✓✓✓✓✓✓✗ reject; chose C1 → ✓✓✓✓✓✓✓✓ accept

Selected
0 OR-NOT17
1 NOT2
...
```

Then the existing `Applying to {query}` and final `\boxed{...}` follow unchanged.

For each bit, the "tried" candidate is a real rule that fits **N−1 of N** examples but fails on one — a genuine near-miss. The model trains on the per-example ✓/✗ pattern, not just on the format. The ✗ on training data forces the model to actually attend to verification rather than ritualize "All examples match".

## Coverage

| Status | Count |
|---|---:|
| bit_manipulation rows augmented with failure-correction | **1,280** (94.5%) |
| Skipped (huikang's solver picked a rule that doesn't fit all examples — can't generate clean accept trace) | 74 |
| Skipped (no ambiguous bit found — only one candidate rule fits) | 0 |
| Skipped (even after trimming, augmented version > MAX_SEQ_LEN) | 0 |
| Non-bit_manipulation rows pass through unchanged | 5,552 |
| **Total** | **6,906** |

**"Skipped" = row is written to the output CSV with its original CoT unchanged** (no failure-correction block added). Nothing is dropped from the dataset; the script preserves the same 6,906-row count as the input CSV.

### Ambiguity distribution per row

| Bits with near-miss candidate | Rows |
|---:|---:|
| 8 (all bits ambiguous) | 1,147 |
| 7 | 75 |
| 6 | 15 |
| 5 | 6 |
| 3 (trimmed to fit MAX_SEQ_LEN) | 37 |

**85% of bit_manipulation puzzles have all 8 bits ambiguous** — confirming that ambiguity (not solver coverage) is the dominant difficulty.

## Token length

Measured with huikang's raw tokenizer, full training-format wrap.

**Real-script overhead vs my estimate: +2 tokens** (measured cleanly across 1,354 bit_manip rows — for rows that were the longest in their micro-batch in run #7's training log, real `padded_len` = my estimate + 2 consistently. Diff comes from chat-template specials that `apply_chat_template` adds and my manual `<|im_start|>...` approximation doesn't model.)

**UTF-8 cost for ✓/✗:** the tokenizer encodes each ✓ or ✗ as **3 BPE tokens** (multi-byte UTF-8 expansion). My estimator already counts these correctly. One failure-correction line ≈ 67 tokens; 8 bits ≈ 540 tokens added per row.

**Cap policy: `MAX_SEQ_LEN = 8092` (100-token safety margin from the real 8,192 cap).** Buffer accommodates the +2 chat-template overhead plus up to ~98 more tokens of unforeseen variance.

| | min | p50 | p95 | max | real-script equivalent (+2) |
|---|---:|---:|---:|---:|---:|
| Augmented rows | 6,560 | 7,484 | 8,007 | **8,092** | **~8,094** |
| Rows that would exceed 8,192 in real script | **0** | | | | |

**Trim distribution** — when the full 8-bit failure-correction block exceeds the cap, the script trims to top-K bits (7, 6, 5, 3, 2, 1) until it fits. All 1,280 augmented rows fit:

| Bits with failure-correction | Rows |
|---:|---:|
| 8 (no trim) | 1,110 |
| 7 | 66 |
| 6 | 13 |
| 5 | 5 |
| 3 | 84 |
| 2 | 2 |

## Files

- `build_failure_correction_cot.py` — generator (re-runnable, deterministic; no external solver dep)
- `260516_bit_manipulation_failcorr.csv` — output, 35.00 MB, 6,906 rows
- `README.md` — this file

## How to use

1. Upload `260516_bit_manipulation_failcorr.csv` to AutoDL: `/root/autodl-tmp/data/260516_bit_manipulation_failcorr.csv`
2. In a copy of the training script, change `CSV_PATH` to point at it.
3. Same recipe, hyperparameters, ordering as run #7 — only the data changes.
4. Eval bit_manipulation accuracy **on the 25 out-of-training-set val IDs** (not the full 160 val — those overlap with training and are dominated by memorization).

## Hypothesis & decision rule

Out-of-train bit_manip accuracy:
- **12% → ~20%**: failure-correction format teaches some elimination behavior. Worth iterating (e.g., add more diverse near-miss candidates per bit, vary the rejected-candidate position).
- **12% → 30%+**: strong signal; the elimination behavior generalizes. Apply same template to cryptarithm (which is at 0% out-of-train) for the biggest payoff.
- **No movement (still ~12%)**: even explicit ✗ doesn't force the model to verify at inference. The remaining option is to change WHICH RULE the solver picks (give the solver a tie-break that's more uniform across the training distribution), so the model has a less peculiar pattern to memorize.

## Why this is more likely to work than all-✓ verification

A previous version of this dataset (now removed) added only a "Verifying selected rules against examples" block where every check came out ✓. That version had a clear theoretical problem: the model never sees ✗ in training, so it has no pressure to emit ✗ at inference — it just learns "after Selected, recite 'All N examples match.'", which is performative.

This version's traces mix ✓ and ✗ per example **on every row**, including explicit "reject" outcomes. To minimize training loss on these traces, the model has to predict ✗ at the right positions — which requires attending to the per-example evidence. That's the only mechanism by which the format change could change inference behavior.

It might still not work (the model might learn to predict ✗ only at known training-data positions and miss novel ones), but at least the mechanism is plausible. The all-✓ version had no plausible mechanism.

## Rule grammar (inline evaluator)

| Rule | Semantics |
|---|---|
| `I<n>` | input bit n (n ∈ 0-7) |
| `NOT<n>` | 1 − input bit n |
| `C0` / `C1` | constant 0 / 1 |
| `XOR<a><b>` / `OR<a><b>` / `AND<a><b>` | bitwise op of input bits a, b |
| `XOR-NOT<a><b>` / `OR-NOT<a><b>` / `AND-NOT<a><b>` | op of input bit a and NOT(input bit b) |
| `default 1` | constant 1 (huikang's fallback) |
