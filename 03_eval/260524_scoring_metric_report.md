# How Submissions Are Scored — NVIDIA Nemotron Reasoning Challenge

**Date:** 2026-05-24
**Source:** live `metric/nvidia-nemotron-metric` kernel, pulled fresh from Kaggle on 2026-05-24 (via `kaggle kernels pull metric/nvidia-nemotron-metric`). This is the **current** scoring code, post the 2026-05-08 `}`-extraction update.

---

## 1. TL;DR

- **Final score = plain accuracy** = `num_correct / total_questions`. Every question weighted equally; **no per-category weighting**.
- A prediction is **correct** if it matches the ground truth by one of three rules, chosen by the *shape of the ground truth*:
  1. **Binary string** (`^[01]+$`) → **exact** string match (case-insensitive). Leading zeros matter.
  2. **Numeric** (both parse as float) → **±1% relative tolerance** (`rel_tol=1e-2`, `abs_tol=1e-5`).
  3. **Everything else** → **exact** string match (case-insensitive).
- The answer is pulled from the model's text by `extract_final_answer`, which prioritizes the **last non-empty `\boxed{...}`**.
- Evaluation is **non-deterministic** (vLLM) — the same submission can vary ~±0.02 (per Kaggle staff).

---

## 2. The scoring pipeline (`score()`)

For each submission, the harness:
1. Loads Nemotron-3-Nano-30B + the submitted LoRA adapter in vLLM (must contain `adapter_config.json`, else `ParticipantVisibleError`).
2. Builds each prompt as:
   `<puzzle prompt>` + `"\nPlease put your final answer inside `\boxed{}`. For example: `\boxed{your answer}`"`, then applies the chat template with `enable_thinking=True`.
3. Generates a completion per test row (greedy at eval time — see §5).
4. `extract_final_answer(raw_text)` → predicted string (§3).
5. `verify(ground_truth, predicted)` → bool (§4).
6. `accuracy = num_correct / len(solution)` → the final float score.

There is **no partial credit** and **no category aggregation** — it's a flat mean of per-question correctness.

---

## 3. Answer extraction — `extract_final_answer(text)`

Tried in strict priority order; first hit wins:

1. **Boxed (primary).** Find every `\boxed{`. For each, take the text up to the **last `}`** before the next `\boxed{` (or end of text), via `rfind('}')`. Return the **last non-empty** such segment.
   - This is the post-2026-05-08 fix. It correctly handles answers that *contain* `}` — e.g. the model writes `\boxed{-}}` for the answer `-}`, and the metric extracts `-}` (not `-`). Also handles nested LaTeX like `\boxed{\frac{1}{2}}`.
2. **"Final answer" phrases** (only if no `\boxed{}` at all): last match of
   `The final answer is: …` / `Final answer is: …` / `Final answer[:：] …` / `final answer[:：] …` (case-insensitive).
3. **Last number**: last match of `-?\d+(?:\.\d+)?`.
4. **Last non-empty line** of the text.
5. Else `'NOT_FOUND'` (also returned immediately if `text is None`).

**Implication:** for non-numeric answers (cipher, cryptarithm, roman numerals), if the model fails to emit a `\boxed{}`, fallback #3 grabs a stray number and the string comparison fails. Emitting a clean `\boxed{}` is essential for those categories.

---

## 4. Answer comparison — `verify(stored_answer, predicted)`

Both strings are `.strip()`ed first. Then exactly one of three branches runs, selected by the **ground truth's** form:

```python
stored = stored_answer.strip(); predicted = predicted.strip()

# Branch 1 — binary string
if re.fullmatch(r'[01]+', stored):
    return predicted.lower() == stored.lower()      # EXACT string

# Branch 2 — numeric (both must parse as float)
try:
    return math.isclose(float(stored), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
except Exception:
    # Branch 3 — anything else
    return predicted.lower() == stored.lower()       # EXACT string, case-insensitive
```

### Branch 1 — Binary string → exact match
- Triggers when the **ground truth** is all 0/1 (`^[01]+$`).
- Strict, case-insensitive string equality. **Length / leading zeros matter.**
- Official examples: `verify("10011000","10011000")=True`, `verify("10011000","10011001")=False`, `verify("11011","00011011")=False`.

### Branch 2 — Numeric → 1% relative tolerance
- Triggers when **both** stored and predicted parse as `float`.
- `math.isclose(rel_tol=1e-2, abs_tol=1e-5)` → correct if within **1% relative** (or 1e-5 absolute near zero).
- Examples: `verify("24.64","24.6401")=True`. A gravity answer of `6.67e-11` is accepted anywhere in ~`6.60e-11…6.74e-11`.
- Leading zeros are irrelevant here (`"0075"`→75.0), **but** note Branch 1 takes precedence, so a pure-0/1 numeric like `"0075"`… is *not* binary (has 7,5) → goes to Branch 2; whereas `"0011"` *is* binary → Branch 1 (exact).

### Branch 3 — String → exact match (case-insensitive)
- Fallback for non-binary, non-numeric answers.
- Case-insensitive exact equality. Example: `verify("XLVII","xlvii")=True`.
- Covers cipher (decoded text), cryptarithm (symbol strings), and roman-numeral / non-numeric "numeral" answers.

---

## 5. Generation parameters (what the harness actually runs)

The `score()`/`generate_predictions()` function **signatures** default to:
`max_tokens=3584, top_p=1.0, temperature=1.0, max_num_seqs=128, max_model_len=4096, max_lora_rank=32, gpu_memory_utilization=0.85`.

**These defaults are NOT what the competition uses.** The competition **overview page** specifies the real evaluation settings (confirmed in discussion thread "Clarification on Final Evaluation Settings", 2026-05-24):

| Parameter | Actual eval (overview page) | `score()` default (vestigial) |
|---|---|---|
| max_lora_rank | 32 | 32 |
| **max_tokens** | **7680** | 3584 |
| top_p | 1.0 | 1.0 |
| **temperature** | **0.0** | 1.0 |
| **max_num_seqs** | **64** | 128 |
| gpu_memory_utilization | 0.85 | 0.85 |
| **max_model_len** | **8192** | 4096 |

So at scoring time the harness passes 7680 / 8192 / temp 0.0 / 64; the function defaults are never used. Our local eval routine mirrors the real values (7680 / 8192 / 0.0 / 64).

`max_tokens=7680` is the **generation** cap (completion length); `max_model_len=8192` is the **total** context (prompt + completion). temp 0.0 = greedy.

---

## 6. Per-category implications (which branch each category hits)

| Category (huikang) | GT form | verify branch | Notes |
|---|---|---|---|
| bit_manipulation | binary `[01]+` | 1 — exact | leading zeros & length must match exactly |
| gravity | float | 2 — 1% tol | easy: only needs ±1% |
| unit_conversion | float | 2 — 1% tol | easy: only needs ±1% |
| equation_numeric_* | int | 2 — 1% tol (effectively exact for small ints) | |
| numeral | roman / non-numeric str | 3 — exact str | case-insensitive |
| cipher | decoded text | 3 — exact str | must match exactly |
| cryptarithm_* | symbol string (often contains `}`) | 3 — exact str | the `}` extraction fix matters here |

**Why gravity / unit_conversion saturate near 100%:** they only require a numeric answer within 1%. **Why bit_manipulation & cipher / cryptarithm are unforgiving:** exact string match — a single wrong char or wrong length is a miss.

---

## 7. Key gotchas

- **Non-determinism:** vLLM gives ~±0.02 run-to-run on the same submission (Kaggle staff). Treat sub-0.02 differences between models as noise.
- **No rescore after metric updates:** the 2026-05-08 `}`-fix was applied without rescoring old submissions — must resubmit to be rescored with current metric.
- **Boxed is king:** for string-answer categories, no `\boxed{}` → fallback grabs a number → near-certain miss. Always emit `\boxed{...}`.
- **Trailing tokens are safe:** text after the final `}` (e.g. `<|im_end|>`) is excluded because extraction stops at the last `}` of the segment.
- **Binary vs numeric branch selection is by the GROUND TRUTH, not the prediction** — you can't change which rule applies by formatting your output differently.

---

## 8. Verbatim current code

See Appendix in `01_data/260523_kaggle_metric_extractor_report.md` for the full `extract_final_answer` + `verify` text, and the live pull at `metric/nvidia-nemotron-metric`. Our eval template `03_eval/eval_kaggle_nemotron.ipynb` was synced to this version on 2026-05-24 (cell-3, rfind/last-`}` extractor).
