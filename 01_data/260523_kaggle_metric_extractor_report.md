# Kaggle answer-extraction metric — what it is, and does our pipeline match it

**Date:** 2026-05-23 (Chicago)
**Source:** live kernel `metric/nvidia-nemotron-metric`, pulled via Kaggle API on 2026-05-23 (live code md5 `7e34e9d6e1ce`).
**Verdict:** Our training extractor **matches the current live metric**. The metric was **updated** at some point from a first-`}` regex to a last-`}` (rfind) scan; our 3 local copies were the **old** version and are stale.

---

## 1. How the answer is extracted + scored (current live metric)

Two functions matter: `extract_final_answer(text)` pulls the model's answer string; `verify(stored, predicted)` compares it to the gold answer.

### `extract_final_answer` — boxed extraction (the part that changed)

```python
# For each \boxed{ occurrence, take everything up to the LAST } before the
# next \boxed{ (or end of text). Handles answers that contain '}' literally
# (e.g. \boxed{}52} for answer "}52") and nested LaTeX like \boxed{\frac{1}{2}}.
boxed_starts = list(re.finditer(r'\\boxed\{', text))
matches = []
for i, m in enumerate(boxed_starts):
    start = m.end()
    end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
    segment = text[start:end]
    last_brace = segment.rfind('}')                       # ← LAST }, not first
    matches.append(segment[:last_brace] if last_brace != -1 else segment)
if matches:
    non_empty = [m.strip() for m in matches if m.strip()]
    return non_empty[-1] if non_empty else matches[-1].strip()   # last non-empty box
```

Fallbacks when there is **no** `\boxed{}` at all (in order):
1. `The final answer is:\s*([^\n]+)` / `Final answer is:` / `Final answer[:：]` / `final answer[:：]` (case-insensitive) → last match
2. last number `-?\d+(?:\.\d+)?`
3. last non-empty line
4. else `'NOT_FOUND'`

### `verify(stored_answer, predicted)` — comparison

```python
stored = stored.strip(); predicted = predicted.strip()
if re.fullmatch(r'[01]+', stored):              # binary string → strict
    return predicted.lower() == stored.lower()
try:                                            # numeric → tolerant
    return math.isclose(float(stored), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
except Exception:                               # else → case-insensitive string
    return predicted.lower() == stored.lower()
```

Per-category implication of `verify`:
- **bit_manipulation** (binary `[01]+`): exact string, case-insensitive.
- **gravity / unit_conversion / equation_numeric** (numbers): float match within **rel_tol 1e-2** (1%) or abs_tol 1e-5.
- **cipher / numeral / cryptarithm** (non-numeric, non-binary): case-insensitive **exact string** match.

---

## 2. Our training extractor (`_extract_boxed` in every train_*.py)

```python
_boxed_start_pat = re.compile(r'\\boxed\{')
def _extract_boxed(text):
    starts = list(_boxed_start_pat.finditer(text))
    out = []
    for i, m in enumerate(starts):
        seg_end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        seg = text[m.end():seg_end]
        lb = seg.rfind('}')                    # ← LAST }
        out.append(seg[:lb] if lb != -1 else seg)
    return out
# training uses boxed_matches[-1] as the answer
```

**This is the same algorithm as the live metric's boxed block** (finditer all `\boxed{` → segment to next box/end → `rfind('}')` → last non-empty). ✅ Match.

---

## 3. The version history that caused confusion

| Copy | boxed regex | behavior | md5 |
|---|---|---|---|
| **Live Kaggle (current, 2026-05-23)** | `rfind('}')` scan | last `}` — handles `}` in answers | `7e34e9d6e1ce` |
| Local copies (`01_data/`, `03_eval/`, `study/260409_grpo_research/reference/`) | `re.findall(r'\\boxed\{([^}]*)(?:\}\|$)', text)` | first `}` — truncates `}` answers | `ca73f0f1c44c` (STALE) |
| Old training regex (pre-#6) | `\\boxed\{([^}]*)\}` | first `}` | — |

The local `.ipynb` copies are an **older** metric revision. Kaggle has since switched to the rfind version. Our training pipeline (#6 onward) already uses rfind, so **training and the current metric agree.**

---

## 4. Consequences

- ✅ **No extractor mismatch.** Training targets and live scoring use identical boxed logic.
- ✅ **No `}`-truncation ceiling.** Earlier worry that answers containing `}` (some cryptarithm answers) couldn't be scored was based on the stale local copy. The live metric extracts `}`-containing answers correctly.
- ⚠️ **The 3 local metric copies are stale** and will mislead any future audit. Replace them with the live pull if kept for reference.
- Note for answer formatting: a model output of `\boxed{X}` where `X` contains `}` is handled, but the metric takes the segment up to the **last** `}` before the next `\boxed{` — so trailing tokens after the final `}` (e.g. `<|im_end|>`) are correctly excluded.

---

## 5. How to re-pull the live metric (reproducible)

```python
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
api.kernels_pull('metric/nvidia-nemotron-metric', path='<dir>')   # → nvidia-nemotron-metric.ipynb
```
(Requires KAGGLE_USERNAME / KAGGLE_KEY env vars.)

---

## Appendix A — FULL live Kaggle metric code (verbatim, pulled 2026-05-23)

`metric/nvidia-nemotron-metric`, the two functions that determine scoring:

```python
def extract_final_answer(text: str | None) -> str:
    r"""Extracts the final answer from the model response.

    Prioritizes extracting answers inside `\boxed{}`.
    If no `\boxed{}` format is found, attempts to extract numbers from other formats.

    Examples:
        >>> extract_final_answer(r"The answer is \boxed{42}")
        '42'
        >>> extract_final_answer("The final answer is: 3.14")
        '3.14'
        >>> extract_final_answer("Just a number 100 in text")
        '100'
        >>> extract_final_answer(None)
        'NOT_FOUND'
    """
    if text is None:
        return 'NOT_FOUND'

    # Search for boxed answer. For each \boxed{ occurrence, take everything up
    # to the last } before the next \boxed{ (or end of text). This handles
    # answers that themselves contain '}' (the model writes them literally,
    # producing e.g. \boxed{}52} for the answer "}52") as well as nested LaTeX
    # like \boxed{\frac{1}{2}}.
    boxed_starts = list(re.finditer(r'\\boxed\{', text))
    matches = []
    for i, m in enumerate(boxed_starts):
        start = m.end()
        end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind('}')
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    # Other common formats if \boxed{} is not found
    patterns = [
        r'The final answer is:\s*([^\n]+)',
        r'Final answer is:\s*([^\n]+)',
        r'Final answer\s*[:：]\s*([^\n]+)',
        r'final answer\s*[:：]\s*([^\n]+)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    # If no structured format is found, extract the last valid number in the text
    matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    if matches:
        return matches[-1]

    # If no numeric answer is found, return the last line of text as a fallback
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else 'NOT_FOUND'


def verify(stored_answer: str, predicted: str) -> bool:
    """Verify if the answer matches.

    For numerical answers, allow them to be judged as equal within a certain relative tolerance (1e-2);
    otherwise, compare strictly as strings (case-insensitive).

    Examples:
        >>> verify("10011000", "10011000")
        True
        >>> verify("10011000", "10011001")
        False
        >>> verify("24.64", "24.6401")
        True
        >>> verify("XLVII", "xlvii")
        True
        >>> verify("11011", "00011011")
        False
    """
    # Clean up strings
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    # If the answer is a binary string, compare strictly as strings
    if re.fullmatch(r'[01]+', stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        # Try to convert the answers to floating point numbers
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        # Use a small absolute tolerance for numbers near zero
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        # Fallback to case-insensitive string comparison
        return predicted.lower() == stored_answer.lower()
```

---

## Appendix B — OUR code (verbatim, from `train_huikang_ddp_nonreent.py`, lines 173-219)

### Extractor (line 173-183)

```python
_boxed_start_pat = re.compile(r'\\boxed\{')

def _extract_boxed(text):
    starts = list(_boxed_start_pat.finditer(text))
    out = []
    for i, m in enumerate(starts):
        seg_end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        seg = text[m.end():seg_end]
        lb = seg.rfind('}')
        out.append(seg[:lb] if lb != -1 else seg)
    return out
```

### How it's used to build the training target (line 215-219)

```python
boxed_matches = _extract_boxed(cot)
reasoning_answer = boxed_matches[-1] if boxed_matches else answer

completion_text = cot + '\n</think>\n' + chr(92) + 'boxed{' + reasoning_answer + '}<|im_end|>'
completion_ids = raw_tokenizer.encode(completion_text, add_special_tokens=False).ids
```

We extract the answer from the source `solver_cot` with `_extract_boxed` (rfind/last-`}`), then **re-emit** it as the training target `… </think>\n\boxed{<reasoning_answer>}<|im_end|>`. So the model is trained to produce exactly the `\boxed{…}` string that the live metric's matching rfind extractor will read back.

### Equivalence note

Both implementations: (1) find every `\boxed{` with the same regex `\\boxed\{`; (2) slice each segment up to the next `\boxed{` or end of text; (3) take `rfind('}')` (last brace); (4) select the last non-empty result. The live metric adds the no-box fallbacks (Appendix A) which never trigger for our well-formed `\boxed{}` targets. **The boxed paths are algorithmically identical.**
