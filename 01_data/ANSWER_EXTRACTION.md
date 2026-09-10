# Kaggle Answer Extraction & Scoring (Nemotron)

How the final answer is pulled out of a CoT and compared to gold. **Authoritative source:**
`03_eval/260524_scoring_metric_report.md` + the live `metric/nvidia-nemotron-metric` kernel
(pulled 2026-05-24, post the 2026-05-08 `}`-extraction fix). Follow this exactly when scoring.

## TL;DR

- **Score = plain accuracy** = `num_correct / total`. Flat mean, equal weight, no per-category weighting, no partial credit.
- **Answer = the last non-empty `\boxed{...}`**, with the close brace found by `rfind('}')` (last `}`).
- **`verify` picks one of three rules by the GROUND TRUTH's shape** (NOT the prediction's):
  1. binary `^[01]+$` → **exact** string (case-insensitive). Leading zeros & length matter.
  2. both parse as `float` → **numeric, ±1% relative tolerance** (`rel_tol=1e-2, abs_tol=1e-5`).
  3. otherwise → **exact** string (case-insensitive).
- Eval is non-deterministic (vLLM) → ~±0.02 run-to-run; treat smaller diffs as noise.

## Extraction — `extract_final_answer(text)` (first hit wins)

1. **Boxed (primary).** Find every `\boxed{`. For each, take text up to the **last `}`** before the
   next `\boxed{` (or EOF) via `rfind('}')`; return the **last non-empty** such segment.
   - `rfind` (not `find`) so answers that *contain* `}` work: `\boxed{-}}` → `-}`,
     `\boxed{\frac{1}{2}}` → `\frac{1}{2}`. Trailing tokens after the final `}` (e.g. `<|im_end|>`) are excluded.
2. **"Final answer" phrases** (only if no `\boxed{}` at all): last match of
   `(The )?[Ff]inal answer (is)?[:：]? …`.
3. **Last number**: last `-?\d+(?:\.\d+)?`.
4. **Last non-empty line.**
5. else `'NOT_FOUND'` (also if `text is None`).

→ **Always emit a clean `\boxed{}`.** For string-answer categories, no box means a fallback grabs a stray number and the match fails.

## Comparison — `verify(stored_gt, predicted)`

```python
stored = stored_gt.strip(); predicted = predicted.strip()
if re.fullmatch(r'[01]+', stored):                                   # Branch 1: binary
    return predicted.lower() == stored.lower()                       #   exact, leading zeros matter
try:
    return math.isclose(float(stored), float(predicted), rel_tol=1e-2, abs_tol=1e-5)   # Branch 2: numeric, ±1%
except Exception:
    return predicted.lower() == stored.lower()                       # Branch 3: exact string (ci)
```

The branch is chosen by **`stored` (the ground truth)**; you cannot change it by formatting your output.

## What this means for `equation_numeric`

Ground truth is an int-ish string. Which branch:

| gold shape | branch | examples |
|---|---|---|
| binary-looking (`01`, `10`, `1`, `0011`) | 1 — exact | `01` vs `1` = MISS; `10` vs `10` = hit |
| plain / leading-zero int (`62`, `08`, `0293`, `159`) | 2 — numeric ±1% | `08`=`8`, `0293`=`293`, **`159`≈`158` = hit** (0.6% < 1%), `1793`≈`1792` = hit |
| literal-minus (`-62`) | 2 — numeric | `-62` vs `62` = MISS (value differs); `-62` vs `-62` = hit |
| operator-symbol sign (`*53`, `}30`) | 3 — exact | `*53` vs `53` = MISS (the sign char must be reproduced) |

So: leading zeros are forgiven **except** when the whole answer is 0/1 (binary branch); off-by-one
small ints are forgiven (±1%); but a **sign character** that's not a literal `-` (e.g. `*`, `}`, `:`)
must be reproduced exactly (Branch 3). Sign *value* (the `-` in `-62`) matters via the numeric branch.

## Our implementation (matches the official)

- `01_data/260523_Numeric_Equation/_gen_eq.py:extract_final_answer` — identical boxed logic
  (rfind, last non-empty). It omits fallbacks #2–#4, which is fine because our generated CoTs always
  emit one clean box.
- `01_data/260523_Numeric_Equation/_gen_eq.py:verify` — **identical** to the 3-branch metric above.
  (Earlier I mistakenly called the numeric branch a bug and computed an "exact-string 607" — that was
  wrong; the metric IS numeric-±1% for numeric GT, so the reported **615/732 is correct**.)

## Eval generation params (real competition settings, per overview page 2026-05-24)

`temperature=0.0` (greedy), `max_tokens=7680`, `max_model_len=8192`, `max_num_seqs=64`,
`top_p=1.0`, `max_lora_rank=32`, `gpu_memory_utilization=0.85`. (The `score()` function defaults
— 3584/4096/temp 1.0/128 — are vestigial and not used at eval.)

## References

- `03_eval/260524_scoring_metric_report.md` (full writeup) and
  `03_eval/nvidia-nemotron-metric-LIVE-260524.ipynb` (live kernel code).
- Live pull: `kaggle kernels pull metric/nvidia-nemotron-metric`.
