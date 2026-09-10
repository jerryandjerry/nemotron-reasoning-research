"""GRPO reward functions for Nemotron-3-Nano cryptarithm + equation_numeric training.

This module implements the reward stack designed in the GRPO spec:

  1. reward_final_answer          PRIMARY (outcome, Kaggle metric proxy)
  2. reward_conclusion_semantic   PRIMARY-PROCESS (targets 71% C2 failure mode)
  3. reward_solve_query_remap     PRIMARY-PROCESS (targets D2 box-letter failure)
  4. reward_skeleton_alignment    SCAFFOLD (closed-set header floor)
  5. reward_closed_vocab_discipline SCAFFOLD (gold _finalize invariants)
  6. reward_termination_quality   GUARDRAIL (truncation / duplicate-box)
  7. reward_solver_alignment      RESERVED, off by default (SequenceMatcher bonus)

TRL signature contract:
    def reward_fn(prompts, completions, **kwargs) -> List[float]

`completions[i]` is List[{"role","content"}] (chat-template).  Extra dataset
columns are forwarded by TRL as batched lists in kwargs; we read:
    gold_cot, gold_answer, oversampling, category, ex_count, symbol_pool

The PRIMARY answer extractor `extract_final_answer` and matcher `verify` are
COPIED VERBATIM from 03_eval/nvidia-nemotron-metric.ipynb so that the reward
that gates GRPO is byte-equal to the Kaggle scoring metric.
"""

from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from typing import Any, List

# ---------------------------------------------------------------------------
# VERBATIM from 03_eval/nvidia-nemotron-metric.ipynb — DO NOT EDIT
# ---------------------------------------------------------------------------

def extract_final_answer(text: str | None) -> str:
    r"""Extracts the final answer from the model response.

    Prioritizes extracting answers inside `\boxed{}`.
    If no `\boxed{}` format is found, attempts to extract numbers from other formats.
    """
    if text is None:
        return 'NOT_FOUND'

    matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

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

    matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    if matches:
        return matches[-1]

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else 'NOT_FOUND'


def verify(stored_answer: str, predicted: str) -> bool:
    """Verify if the answer matches (1e-2 rel-tol numeric, else case-insensitive str)."""
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    if re.fullmatch(r'[01]+', stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r'\\boxed\{([^}]+)\}')
# Real gold formats (verified against all 18 solver test cases):
#   '  Conclusion of step §1: {f = a×b+1, g = |a-b|, A=5, ...}, 12 of 12 resolved. This is the answer.'
#   '  Summary: Confirm reading order = rightward, f = a×b+1, g = |a-b|'   (NO braces; may contain unknown)
# Lines are indented; X counts the RESOLVED brace entries ('= unknown' / 'is an exotic operator'
# are unresolved); Y is the total problem unknown count (not derivable from the brace).
_CONCLUSION_RE = re.compile(
    r'^\s*Conclusion of step §(\d+):\s*\{([^}]*)\},\s*(\d+) of (\d+) resolved\.?(.*)$', re.M)
_SUMMARY_RE = re.compile(
    r'^\s*Summary:(.*?)Confirm reading order = (.+?)\s*$', re.M)
_READING_HDR_RE = re.compile(r'^\s*§(\d+) reading ([^:]+):', re.M)

_CLOSED_TAGS = {'~add', '~sub', '~mul', '~concat'}
_CLOSED_VARIANTS = {
    'a×b', 'a+b', 'a-b', 'b-a', '|a-b|', '-|a-b|', 'a∥b', 'b∥a',
    # ±1 / ±2 noisy variants per the solver's prior-knowledge rule #1:
    # ~add=[a+b±1,±2], ~sub=[a-b±1,±2, b-a±1,±2], ~mul=[a×b±1,±2]
    'a×b+1', 'a×b-1', 'a×b+2', 'a×b-2',
    'a+b+1', 'a+b-1', 'a+b+2', 'a+b-2',
    'a-b+1', 'a-b-1', 'a-b+2', 'a-b-2',
    'b-a+1', 'b-a-1', 'b-a+2', 'b-a-2',
}
_CLOSED_DIRECTIONS = {
    'rightward', 'leftward',
    'rightward then leftward', 'leftward then rightward',
}


def _content(completion: Any) -> str:
    """Extract text from a TRL chat-template completion (List[{role,content}]) or str."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return completion[0].get('content', '')
    return str(completion)


def _scale(per_row_scores: List[float], oversampling: List[Any] | None) -> List[float]:
    """Multiply each per-row reward by its oversampling weight (default 1.0)."""
    if oversampling is None:
        return per_row_scores
    out = []
    for s, w in zip(per_row_scores, oversampling):
        try:
            out.append(s * float(w))
        except Exception:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# 1. PRIMARY OUTCOME REWARD
# ---------------------------------------------------------------------------

def reward_final_answer(
    prompts, completions, **kwargs
) -> List[float]:
    r"""PRIMARY outcome reward — the only signal aligned with the Kaggle metric.

    Scoring (per row, clipped to [-3.5, +3.5] before oversampling):
      D1 box matches gold_answer ........................... +2.5 / -1.5
      D2 (cryptarithm) box uses only symbol_pool chars ..... +0.5 / -0.5
      D3 (equation_numeric) box parses as int .............. +0.5 / -0.5
      D4 'unknown' not in box .............................. +0.5 / -1.0
      D5 exactly one \boxed{} in completion ................ +0.2 / -0.2
      (no \boxed{} at all .................................. -2.0 hard floor)
    """
    gold_answer = kwargs.get('gold_answer') or kwargs.get('answer')
    category = kwargs.get('category') or [''] * len(completions)
    symbol_pool = kwargs.get('symbol_pool') or [''] * len(completions)
    oversampling = kwargs.get('oversampling')

    out: List[float] = []
    for comp, gold, cat, pool in zip(completions, gold_answer, category, symbol_pool):
        text = _content(comp)
        boxes = _BOXED_RE.findall(text)

        if not boxes:
            out.append(-2.0)
            continue

        ans = boxes[-1].strip()
        score = 0.0

        # D1 — use the metric's own verify() so we are byte-equal to Kaggle scoring
        if verify(str(gold), ans):
            score += 2.5
        else:
            score -= 1.5

        # D2 / D3 — category-routed shape check
        # (startswith: dataset categories are cryptarithm_deduce/_guess, equation_numeric_deduce/_guess)
        if str(cat).startswith('cryptarithm'):
            allowed = set(str(pool)) | {' ', '-'}
            uppercase_letters = set(re.findall(r'[A-Z]', ans))
            pool_letters = set(re.findall(r'[A-Z]', str(pool)))
            ok = all(ch in allowed for ch in ans) and uppercase_letters.issubset(pool_letters)
            score += 0.5 if ok else -0.5
        elif str(cat).startswith('equation_numeric'):
            score += 0.5 if re.fullmatch(r'-?\d+', ans) else -0.5

        # D4 — explicit "unknown" in the box is a failure tell
        score += 0.5 if 'unknown' not in ans.lower() else -1.0

        # D5 — duplicate-box penalty
        score += 0.2 if len(boxes) == 1 else -0.2

        # clip then push
        out.append(max(-3.5, min(3.5, score)))

    return _scale(out, oversampling)


# ---------------------------------------------------------------------------
# 2. PRIMARY PROCESS REWARD — Conclusion / Summary semantics
# ---------------------------------------------------------------------------

def _brace_entry_resolved(entry: str):
    """True if a Conclusion-brace entry is resolved; False if unknown/exotic; None if empty."""
    e = entry.strip()
    if not e:
        return None
    if 'unknown' in e or 'exotic' in e:
        return False
    return True


def reward_conclusion_semantic(
    prompts, completions, **kwargs
) -> List[float]:
    """PRIMARY-PROCESS reward — Conclusion/Summary discipline (verified vs REAL gold formats).

    Real gold (all 18 solver test cases):
      Conclusion brace may legitimately contain '= unknown' / 'is an exotic operator';
      X must equal the count of RESOLVED entries.  Gold Summary has NO braces and may
      legitimately contain 'unknown' (e.g. query op solved, others left).  The actual
      discipline gold enforces is CONTINUATION: leave §1 unresolved → either try §2
      ('Need to try reading leftward.') or justify stopping ('This is the answer.' /
      'no need to solve the other unknowns' / 'no reading can determine').

    Scoring (per row, range [-2.0, +1.6]):
      C1 every Conclusion line: X == resolved-entry count   +0.5 / -0.5 (missing → -0.5)
      C2 continuation discipline (see above)                +0.4 / -0.8
      C3 Summary direction matches an emitted §N header     +0.4 / -0.4 (missing Summary → -0.4)
      C4 forward ordering: §2 only after §1's Conclusion    +0.3 / -0.3 (skip if no §2)
    """
    oversampling = kwargs.get('oversampling')
    category = kwargs.get('category') or [''] * len(completions)

    out: List[float] = []
    for comp, cat in zip(completions, category):
        text = _content(comp)
        score = 0.0

        conclusions = _CONCLUSION_RE.findall(text)   # (sec, brace, X, Y, tail)
        summary = _SUMMARY_RE.search(text)

        # Documented early-exit — CRYPTARITHM ONLY (verified: _gen_crypt.py:1137 skips the
        # tree when the QUERY operator is unseen; _gen_eq.py NEVER skips — it carries the
        # unseen op through the full traversal and guesses after, lines 400/755/781/913).
        # Waive the C-block only for crypt completions showing the justification + solve block.
        if (str(cat).startswith('cryptarithm')
                and not conclusions
                and 'does not appear in the examples' in text
                and re.search(r'^\s*Now solve the QUERY', text, re.M)):
            out.append(0.8)
            continue

        # C1 — X equals the resolved-entry count of the brace
        if conclusions:
            all_ok = True
            for _sec, brace, x_str, _y_str, _tail in conclusions:
                flags = [_brace_entry_resolved(e) for e in brace.split(',')]
                resolved = sum(1 for f in flags if f is True)
                try:
                    if int(x_str) != resolved:
                        all_ok = False; break
                except ValueError:
                    all_ok = False; break
            score += 0.5 if all_ok else -0.5
        else:
            score -= 0.5

        # C2 — continuation discipline
        violation = False
        sec1 = next((c for c in conclusions if c[0] == '1'), None)
        has_sec2 = any(c[0] != '1' for c in conclusions)
        says_continue = any('Need to try reading' in c[4] for c in conclusions)
        if says_continue and not has_sec2:
            violation = True
        if sec1 is not None and not has_sec2:
            unresolved = any(
                _brace_entry_resolved(e) is False for e in sec1[1].split(','))
            if unresolved:
                justified = (
                    'This is the answer' in sec1[4]
                    or 'no need to solve the other unknowns' in text
                    or 'no reading can determine' in text
                    or 'as complete as possible' in text
                )
                if not justified:
                    violation = True
        score += -0.8 if violation else 0.4

        # C3 — Summary's chosen direction must be one the completion actually explored
        if summary is not None:
            chosen = summary.group(2).split(',')[0].strip()
            headers = [h.strip() for _n, h in _READING_HDR_RE.findall(text)]
            ok = any(h.startswith(chosen) or chosen.startswith(h) for h in headers)
            score += 0.4 if ok and chosen else -0.4
        else:
            score -= 0.4

        # C4 — forward-only ordering
        if '§2' in text:
            pos_concl1 = text.find('Conclusion of step §1:')
            pos_sec2 = text.find('§2')
            if pos_concl1 != -1 and pos_sec2 > pos_concl1:
                score += 0.3
            else:
                score -= 0.3
        # else: skip (no §2)

        out.append(max(-2.0, min(2.0, score)))

    return _scale(out, oversampling)


# ---------------------------------------------------------------------------
# 3. PRIMARY PROCESS REWARD — solve-query / remap / terminator pair
# ---------------------------------------------------------------------------

_TERMINATOR_RE = re.compile(
    r'I will now return the answer in \\boxed\{\}, the answer is\s*\n'
    r'\\boxed\{[^}]+\}\s*$'
)


def reward_solve_query_remap(
    prompts, completions, **kwargs
) -> List[float]:
    r"""PRIMARY-PROCESS — bridges Conclusion to \boxed{}; targets D2 remap failure mode.

    Scoring (per row, range [-1.3, +1.0]):
      Q1 'Now solve the QUERY ' header present ............. +0.3 / -0.3
      Q2 last 2 lines = terminator pair                     +0.5 / -0.5
      Q3 (cryptarithm) 'map the digits back to symbols' .... +0.2 / -0.5  (asymmetric)
    """
    category = kwargs.get('category') or [''] * len(completions)
    oversampling = kwargs.get('oversampling')

    out: List[float] = []
    for comp, cat in zip(completions, category):
        text = _content(comp)
        score = 0.0

        # Q1
        if re.search(r'^Now solve the QUERY ', text, re.M):
            score += 0.3
        else:
            score -= 0.3

        # Q2 — terminator pair as final non-empty lines
        if _TERMINATOR_RE.search(text.rstrip() + ''):
            score += 0.5
        else:
            score -= 0.5

        # Q3 — cryptarithm-only remap step ('digits' on the tree path; 'letters' on the
        # query-unseen early-exit path where no digit assignment exists)
        if str(cat).startswith('cryptarithm'):
            if re.search(r'map the (digits|letters) back to symbols', text):
                score += 0.2
            else:
                score -= 0.5

        out.append(max(-1.3, min(1.0, score)))

    return _scale(out, oversampling)


# ---------------------------------------------------------------------------
# 4. SCAFFOLD — closed-set skeleton headers
# ---------------------------------------------------------------------------

def reward_skeleton_alignment(
    prompts, completions, **kwargs
) -> List[float]:
    """SCAFFOLD reward — floor against exploration collapse into free-form English.

    SFT saturates these; weights kept small intentionally (range [-1.6, +1.7]).
    """
    category = kwargs.get('category') or [''] * len(completions)
    ex_count = kwargs.get('ex_count') or [99] * len(completions)
    oversampling = kwargs.get('oversampling')

    out: List[float] = []
    for comp, cat, n_ex in zip(completions, category, ex_count):
        text = _content(comp)
        score = 0.0

        # S1 opener
        if re.match(r'^(We need to |I need to infer the transformation rule)', text):
            score += 0.2
        else:
            score -= 0.2

        # S2 Prior knowledge header + 6 numbered items in the next 20 lines
        m = re.search(r'^Prior knowledge for this kind of question:', text, re.M)
        if m:
            tail = text[m.end():]
            lines_after = tail.splitlines()[:20]
            numbered = sum(1 for ln in lines_after if re.match(r'^\d+\.\s', ln))
            score += 0.3 if numbered >= 6 else -0.3
        else:
            score -= 0.3

        # S3 traverse-tree header
        if re.search(
            r'^Now start to traverse the solution tree, reading order = '
            r'\[rightward, leftward\]:', text, re.M):
            score += 0.2
        else:
            score -= 0.2

        # S4 §1 reading direction header
        if re.search(r'^§1 reading (rightward|leftward[^:]*):', text, re.M):
            score += 0.2
        else:
            score -= 0.2

        # S5 each §1.1..§1.4 sub-header (+0.1 each, cap +0.4)
        sub_score = 0.0
        for pat in (
            r'^§1\.1 writing equations:',
            r'^§1\.2 quick check to see whether any operator is concatenation:',
            r'^§1\.3 prune solution trees by the rhs digit-count before entering:',
            r'^§1\.4',
        ):
            if re.search(pat, text, re.M):
                sub_score += 0.1
        score += min(0.4, sub_score)

        # S6 cryptarithm-only denotation/letter-form blocks
        if str(cat).startswith('cryptarithm'):
            ok = (
                re.search(
                    r'^The other symbols are operands; assign a letter to each in the '
                    r'order of appearance:', text, re.M)
                and re.search(r'^I will convert all the equations to letter form:', text, re.M)
            )
            score += 0.2 if ok else -0.1

        # S7 EX-reference count ≤ ex_count from prompt
        ex_used = len(set(re.findall(r'EX\d+', text)))
        try:
            n_ex_int = int(n_ex)
        except Exception:
            n_ex_int = 99
        score += 0.2 if ex_used <= n_ex_int else -0.2

        out.append(max(-1.6, min(1.7, score)))

    return _scale(out, oversampling)


# ---------------------------------------------------------------------------
# 5. SCAFFOLD — closed-vocab discipline (gold _finalize invariants)
# ---------------------------------------------------------------------------

def reward_closed_vocab_discipline(
    prompts, completions, **kwargs
) -> List[float]:
    """SCAFFOLD reward — catches gold _finalize invariant slips (range [-1.6, +1.2])."""
    oversampling = kwargs.get('oversampling')

    out: List[float] = []
    for comp in completions:
        text = _content(comp)
        score = 0.0

        # V1 — ~tag closed set
        tags = re.findall(r'~\w+', text)
        if not tags:
            score += 0.3  # vacuously compliant; semantic checks handle the dependency
        else:
            violations = sum(1 for t in tags if t not in _CLOSED_TAGS)
            if violations == 0:
                score += 0.3
            else:
                score += max(-0.5, -0.1 * violations)

        # V2 — lock <name> = <variant>  closed set
        locks = re.findall(r'lock \w+ = ([^.,\n]+)', text)
        if not locks:
            score += 0.3
        else:
            violations = sum(1 for v in locks if v.strip() not in _CLOSED_VARIANTS)
            if violations == 0:
                score += 0.3
            else:
                score += max(-0.5, -0.1 * violations)

        # V3 — banned glyphs / casing
        v3_violations = 0
        if '∈' in text:
            v3_violations += 1
        if '·' in text:
            v3_violations += 1
        if re.search(r'\bLHS\b|\bRHS\b', text):
            v3_violations += 1
        if v3_violations == 0:
            score += 0.3
        else:
            score += max(-0.3, -0.1 * v3_violations)

        # V4 — reading-direction phrase ∈ closed set
        dirs = re.findall(r'^§\d+ reading ([^:]+):', text, re.M)
        if not dirs:
            score -= 0.3
        elif all(d.strip() in _CLOSED_DIRECTIONS for d in dirs):
            score += 0.3
        else:
            score -= 0.3

        out.append(max(-1.6, min(1.2, score)))

    return _scale(out, oversampling)


# ---------------------------------------------------------------------------
# 6. GUARDRAIL — termination quality
# ---------------------------------------------------------------------------

def reward_termination_quality(
    prompts, completions, **kwargs
) -> List[float]:
    r"""GUARDRAIL — penalize truncation (no terminal \boxed{}) and duplicate boxes.

    Range [-1.5, +0.8].  Does NOT require </think> (model doesn't emit it).
    """
    completion_token_len = kwargs.get('completion_token_len')
    max_new_tokens = kwargs.get('max_new_tokens')
    oversampling = kwargs.get('oversampling')

    out: List[float] = []
    for i, comp in enumerate(completions):
        text = _content(comp)
        score = 0.0

        # E1 — terminal boxed answer
        if re.search(r'\\boxed\{[^}]+\}\s*$', text):
            score += 0.4
        else:
            score -= 1.0

        # E2 — soft length budget
        if completion_token_len is not None and max_new_tokens is not None:
            try:
                if int(completion_token_len[i]) < int(max_new_tokens):
                    score += 0.2
                else:
                    score -= 0.3
            except Exception:
                pass

        # E3 — clean termination
        if '<|im_end|>' in text[-50:] or re.search(r'\\boxed\{[^}]+\}\s*$', text):
            score += 0.2
        else:
            score -= 0.2

        out.append(max(-1.5, min(0.8, score)))

    return _scale(out, oversampling)


# ---------------------------------------------------------------------------
# 7. RESERVED — solver_alignment (off by default)
# ---------------------------------------------------------------------------

ENABLE_SOLVER_ALIGNMENT = False


def reward_solver_alignment(
    prompts, completions, **kwargs
) -> List[float]:
    """RESERVED partial-credit distillation bonus.

    Off by default.  When enabled, compares the model's `Conclusion of step §1`
    brace to the gold brace by SequenceMatcher ratio, capped at +0.5.
    """
    if not ENABLE_SOLVER_ALIGNMENT:
        return [0.0] * len(completions)

    gold_cot = kwargs.get('gold_cot') or [''] * len(completions)
    oversampling = kwargs.get('oversampling')

    out: List[float] = []
    for comp, gcot in zip(completions, gold_cot):
        text = _content(comp)
        m_model = re.search(
            r'Conclusion of step §1: \{([^}]+)\}', text)
        m_gold = re.search(
            r'Conclusion of step §1: \{([^}]+)\}', gcot or '')
        if m_model and m_gold:
            r = SequenceMatcher(None, m_model.group(1), m_gold.group(1)).ratio()
            out.append(r * 0.5)
        else:
            out.append(0.0)

    return _scale(out, oversampling)


# ---------------------------------------------------------------------------
# 8 + 9. CATEGORY-SPECIFIC PARSE REWARDS (per-step, verified against the prompt)
# ---------------------------------------------------------------------------
# The opening parse stage differs fundamentally between the two puzzle types:
#   cryptarithm  : BPE-unmerge (space out merged symbols), 3rd-symbol-is-operator
#   equation_num : operator denotation + letter-form conversion
# Both are DETERMINISTICALLY verifiable against the puzzle in the prompt itself —
# strip spaces from the model's "becomes" line and it must equal the original.

def _extract_puzzle(prompt_text: str):
    """Parse (equations, query) out of a prompt (chat-template wrapper tolerated).

    Equations = the non-empty lines between 'examples:' and 'Now, determine';
    query     = text after 'determine the result for:' up to end-of-line.
    Returns ([], '') if the prompt doesn't match — callers treat that as neutral.
    """
    try:
        lines = prompt_text.splitlines()
        eqs, query, in_block = [], '', False
        for ln in lines:
            if 'examples:' in ln:
                in_block = True; continue
            m = re.search(r'determine the result for:\s*(.+?)\s*$', ln)
            if m:
                query = m.group(1).strip()
                break
            if in_block:
                s = ln.strip()
                if s and '=' in s:
                    eqs.append(s)
        return eqs, query
    except Exception:
        return [], ''


def _squash(s: str) -> str:
    return ''.join(s.split())


def reward_crypt_parse(prompts, completions, **kwargs) -> List[float]:
    """CRYPTARITHM-ONLY per-step parse reward (range ~[-1.5, +1.5]); 0.0 on other cats.

      P1 unmerge section header present ................... +0.2 / -0.3
         ('Split the BPE merged tokens' or 'read each equation one symbol')
      P2 EX/QUERY 'becomes' lines verified vs prompt ....... +1.0 × frac_correct,
         -0.15 per wrong line (cap -0.6); all-missing = -0.6
         correct := squash(spaced) == squash(original from prompt)
                    AND every spaced token is a single symbol
      P3 operator denotation = 3rd symbol of each spaced lhs  +0.3 / -0.3
    """
    category = kwargs.get('category') or [''] * len(completions)
    oversampling = kwargs.get('oversampling')

    out: List[float] = []
    for prompt, comp, cat in zip(prompts, completions, category):
        if not str(cat).startswith('cryptarithm'):
            out.append(0.0); continue
        text = _content(comp)
        ptext = prompt if isinstance(prompt, str) else _content(prompt)
        eqs, query = _extract_puzzle(ptext)
        score = 0.0

        # P1 — unmerge section header
        if re.search(r'Split the BPE merged tokens|read each equation one symbol',
                     text):
            score += 0.2
        else:
            score -= 0.3

        # P2 — verify the EX/QUERY full-line unmerges against the prompt.
        # Gold CoT has TWO 'becomes' blocks per tag (unmerge first, letter-form later) —
        # grade only the FIRST occurrence per tag (the unmerge attempt).
        if eqs:
            expected = {f'EX{i+1}': _squash(e) for i, e in enumerate(eqs)}
            if query:
                expected['QUERY'] = _squash(query)
            first_line = {}
            for m in re.finditer(
                    r'^\s*(EX\d+|QUERY)\s+(.+?)\s+becomes\s+(.+?)\s*$',
                    text, re.M):
                tag = m.group(1)
                if tag in expected and tag not in first_line:
                    first_line[tag] = (m.group(2), m.group(3))
            n_correct, n_wrong = 0, 0
            for tag, (orig, spaced) in first_line.items():
                want = expected[tag]
                toks = spaced.split()
                ok = (_squash(spaced) == want
                      and _squash(orig) == want
                      and all(len(t) == 1 for t in toks))
                if ok: n_correct += 1
                else:  n_wrong   += 1
            if n_correct + n_wrong == 0:
                score -= 0.6                                   # no verifiable lines at all
            else:
                score += 1.0 * n_correct / len(expected)
                score -= min(0.6, 0.15 * n_wrong)

            # P3 — operator denote block must equal the set of 3rd symbols of the
            # spaced lhs across examples AND the query (query op may be unseen).
            third_syms = set()
            for e in eqs:
                lhs = _squash(e.split('=', 1)[0])
                if len(lhs) >= 3:
                    third_syms.add(lhs[2])
            q = _squash(query)
            if len(q) >= 3:
                third_syms.add(q[2])
            denoted = set(re.findall(
                r'^\s*(\S+)\s*->\s*[a-z]\s*$', text, re.M))
            if denoted and third_syms:
                score += 0.3 if denoted == third_syms else -0.3

        out.append(max(-1.5, min(1.5, score)))

    return _scale(out, oversampling)


def reward_eq_parse(prompts, completions, **kwargs) -> List[float]:
    """EQUATION-NUMERIC-ONLY per-step parse reward (range ~[-1.0, +1.0]); 0.0 otherwise.

      P1 denote block: ops -> letters in f,g,h... order of first appearance +0.3 / -0.3
      P2 letter-form 'becomes' lines reproduce the originals ............. +0.5 × frac,
         -0.2 if any wrong
      P3 QUERY letter-form line correct .................................. +0.2 / -0.2
    """
    category = kwargs.get('category') or [''] * len(completions)
    oversampling = kwargs.get('oversampling')

    out: List[float] = []
    for prompt, comp, cat in zip(prompts, completions, category):
        if not str(cat).startswith('equation_numeric'):
            out.append(0.0); continue
        text = _content(comp)
        ptext = prompt if isinstance(prompt, str) else _content(prompt)
        eqs, query = _extract_puzzle(ptext)
        score = 0.0

        # ops in order of first appearance across the prompt equations,
        # then the query (its operator may be unseen in the examples)
        op_order: List[str] = []
        for e in eqs:
            lhs = e.split('=', 1)[0]
            for ch in lhs:
                if not ch.isdigit() and not ch.isspace() and ch not in op_order:
                    op_order.append(ch)
        for ch in query:
            if not ch.isdigit() and not ch.isspace() and ch != '=' and ch not in op_order:
                op_order.append(ch)

        # P1 — denote block: '{op} -> {letter}' with letters f,g,h... in order
        denote = re.findall(r'^\s*(\S+)\s*->\s*([a-z])\s*$', text, re.M)
        if denote and op_order:
            want_letters = 'fghijk'
            ok = (
                [d[0] for d in denote] == op_order
                and all(d[1] == want_letters[i] for i, d in enumerate(denote[:6]))
            )
            score += 0.3 if ok else -0.3
        elif op_order:
            score -= 0.3
        letter2op = {lt: op for op, lt in denote}

        # P2/P3 — letter-form lines: substituting letters back must reproduce the
        # original. First occurrence per tag only (guards against later echo blocks).
        if eqs:
            expected = {f'EX{i+1}': _squash(e) for i, e in enumerate(eqs)}
            first_line = {}
            for m in re.finditer(
                    r'^\s*(EX\d+|QUERY)\s+(.+?)\s+becomes\s+(.+?)\s*$',
                    text, re.M):
                tag = m.group(1)
                if tag not in first_line:
                    first_line[tag] = m.group(3)
            n_correct = 0
            q_ok = False
            for tag, lform in first_line.items():
                toks = [letter2op.get(t, t) for t in lform.split()]
                rebuilt = _squash(''.join(toks))
                if tag == 'QUERY':
                    if query and rebuilt == _squash(query):
                        q_ok = True
                elif expected.get(tag) == rebuilt:
                    n_correct += 1
            if n_correct == len(expected) and expected:
                score += 0.5
            elif n_correct > 0:
                score += 0.5 * n_correct / len(expected) - 0.2
            else:
                score -= 0.2
            score += 0.2 if q_ok else -0.2

        out.append(max(-1.0, min(1.0, score)))

    return _scale(out, oversampling)


# ---------------------------------------------------------------------------
# TWO SEPARATE PER-CATEGORY REWARD STACKS
# ---------------------------------------------------------------------------
# Each public reward belongs to exactly ONE puzzle type and returns 0.0 for rows
# of the other type.  The tested scoring logic lives in the shared implementations
# above; these wrappers apply the category mask.  Benefits:
#   - clean separation per the two question types (different tasks, different rubric)
#   - TRL logs each function separately → per-category reward curves during training
#     (rewards/crypt_final_answer/mean vs rewards/eq_final_answer/mean, etc.)

def _mask_category(base_fn, prefix: str, name: str):
    def wrapped(prompts, completions, **kwargs):
        category = kwargs.get('category') or [''] * len(completions)
        scores = base_fn(prompts, completions, **kwargs)
        return [s if str(c).startswith(prefix) else 0.0
                for s, c in zip(scores, category)]
    wrapped.__name__ = name
    wrapped.__qualname__ = name
    wrapped.__doc__ = f'{prefix}-only wrapper around {base_fn.__name__}.'
    return wrapped


# ── CRYPTARITHM stack (7) — operands AND operators ciphered ────────────────
crypt_final_answer = _mask_category(reward_final_answer,          'cryptarithm', 'crypt_final_answer')
crypt_parse        = reward_crypt_parse          # already crypt-only (BPE unmerge vs prompt)
crypt_conclusion   = _mask_category(reward_conclusion_semantic,   'cryptarithm', 'crypt_conclusion')
crypt_query_remap  = _mask_category(reward_solve_query_remap,     'cryptarithm', 'crypt_query_remap')
crypt_skeleton     = _mask_category(reward_skeleton_alignment,    'cryptarithm', 'crypt_skeleton')
crypt_vocab        = _mask_category(reward_closed_vocab_discipline, 'cryptarithm', 'crypt_vocab')
crypt_termination  = _mask_category(reward_termination_quality,   'cryptarithm', 'crypt_termination')

CRYPT_REWARD_FUNCS = [
    crypt_final_answer,   # D1 gold match ±, D2 box must use puzzle symbols, D4 no 'unknown', D5 single box
    crypt_parse,          # P1 unmerge header, P2 EX/QUERY 'becomes' verified vs prompt, P3 3rd-symbol operators
    crypt_conclusion,     # C1 resolved-count, C2 continuation (§2 or justification), C3 direction, C4 order
                          #   + query-unseen early-exit waiver (crypt-only solver behavior)
    crypt_query_remap,    # Q1 solve block, Q2 terminator pair, Q3 'map the digits|letters back to symbols'
    crypt_skeleton,       # S1-S7 incl. crypt-only denotation/letter-form blocks (S6)
    crypt_vocab,          # ~tags, variants, ' in [', '×', lowercase lhs/rhs
    crypt_termination,    # ends with boxed; truncation penalty
]

# ── EQUATION-NUMERIC stack (7) — operands given, operators unknown ─────────
eq_final_answer = _mask_category(reward_final_answer,          'equation_numeric', 'eq_final_answer')
eq_parse        = reward_eq_parse                # already eq-only (denote + letter-form vs prompt)
eq_conclusion   = _mask_category(reward_conclusion_semantic,   'equation_numeric', 'eq_conclusion')
eq_query_solve  = _mask_category(reward_solve_query_remap,     'equation_numeric', 'eq_query_solve')
eq_skeleton     = _mask_category(reward_skeleton_alignment,    'equation_numeric', 'eq_skeleton')
eq_vocab        = _mask_category(reward_closed_vocab_discipline, 'equation_numeric', 'eq_vocab')
eq_termination  = _mask_category(reward_termination_quality,   'equation_numeric', 'eq_termination')

EQ_REWARD_FUNCS = [
    eq_final_answer,      # D1 gold match ±, D3 box must parse as integer, D4 no 'unknown', D5 single box
    eq_parse,             # P1 ops→f,g,h in order of appearance, P2 letter-form verified vs prompt, P3 QUERY line
    eq_conclusion,        # C1-C4 as crypt but NO early-exit waiver (eq solver never skips the tree)
    eq_query_solve,       # Q1 solve block, Q2 terminator pair (no remap — eq has no symbol mapping)
    eq_skeleton,          # S1-S5+S7 (S6 crypt blocks auto-skip via internal routing)
    eq_vocab,
    eq_termination,
]

# Combined registry handed to GRPOTrainer (14 functions; for any given row the
# other category's 7 contribute exactly 0.0)
REWARD_FUNCS = CRYPT_REWARD_FUNCS + EQ_REWARD_FUNCS

# reward_solver_alignment (gold-CoT SequenceMatcher bonus) intentionally NOT in
# the registry — reserved, off by default.


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Synthetic "near-gold" completion that should score highly on every reward.
    near_gold = """\
We need to infer the transformation rule from the examples and apply it to the QUERY.
Prior knowledge for this kind of question:
1. each symbol denotes a single digit
2. the same symbol is the same digit across all rows
3. operators are drawn from the closed set ~add, ~sub, ~mul, ~concat
4. equations balance numerically
5. read the puzzle in the gold direction
6. resolve operators before binding digits

The other symbols are operands; assign a letter to each in the order of appearance:
I will convert all the equations to letter form:
Now start to traverse the solution tree, reading order = [rightward, leftward]:
§1 reading rightward:
§1.1 writing equations:
§1.2 quick check to see whether any operator is concatenation:
§1.3 prune solution trees by the rhs digit-count before entering:
§1.4 candidate search:
EX1 candidate: lock f = a×b. lock g = a+b. lock h = a-b.
Conclusion of step §1: {f = a×b, g = a+b, h = a-b}, 3 of 3 resolved
Now solve the QUERY using the locked operators.
map the digits back to symbols, so the boxed answer uses puzzle glyphs.
Summary: Confirm reading order = rightward, {f = a×b, g = a+b, h = a-b}
I will now return the answer in \\boxed{}, the answer is
\\boxed{D C E}"""

    # Synthetic "garbage" completion — short, no headers, wrong box.
    garbage = "I think the answer is probably \\boxed{XYZ}"

    completions = [
        [{'role': 'assistant', 'content': near_gold}],
        [{'role': 'assistant', 'content': garbage}],
    ]
    prompts = [
        [{'role': 'user', 'content': 'EX1 ... QUERY ...'}],
        [{'role': 'user', 'content': 'EX1 ... QUERY ...'}],
    ]
    kwargs = dict(
        gold_answer=['D C E', 'D C E'],
        gold_cot=[near_gold, ''],
        oversampling=[1.0, 1.0],
        category=['cryptarithm', 'cryptarithm'],
        ex_count=[1, 1],
        symbol_pool=['ABCDE', 'ABCDE'],
        completion_token_len=[400, 12],
        max_new_tokens=3584,
    )

    fns = [
        reward_final_answer,
        reward_conclusion_semantic,
        reward_solve_query_remap,
        reward_skeleton_alignment,
        reward_closed_vocab_discipline,
        reward_termination_quality,
        reward_solver_alignment,
    ]

    print(f'{"reward":<35} {"near_gold":>10} {"garbage":>10}')
    print('-' * 60)
    totals = [0.0, 0.0]
    for fn in fns:
        scores = fn(prompts, completions, **kwargs)
        totals = [totals[0] + scores[0], totals[1] + scores[1]]
        print(f'{fn.__name__:<35} {scores[0]:>10.3f} {scores[1]:>10.3f}')
    print('-' * 60)
    print(f'{"TOTAL":<35} {totals[0]:>10.3f} {totals[1]:>10.3f}')
