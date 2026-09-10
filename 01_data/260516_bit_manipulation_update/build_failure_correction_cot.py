"""
Post-process bit_manipulation CoTs: insert a failure-correction block before 'Selected'.

For each bit position where a near-miss candidate exists (fits all-but-one example),
the new block shows that near-miss candidate being TESTED per-example, hitting a ✗
on the failing example, getting REJECTED, then shows the actual selected rule being
tested and passing all examples.

This teaches the model the elimination process explicitly, not just the format.

Difference from build_verification_cot.py (the all-✓ version): that one just teaches
the verification *format* — the model learns to recite "All N examples match" without
ever needing to actually verify. This one trains on a mix of ✓ and ✗ outcomes, with
explicit rejections, so the model has pressure to actually check before committing.

Reads:  01_data/260514_lkevincc_golden/260515_huikang_lkall_stripped.csv
Writes: 01_data/260516_bit_manipulation_update/260516_bit_manipulation_failcorr.csv
"""

from __future__ import annotations

import csv
import os
import re
import sys
from itertools import combinations

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
csv.field_size_limit(2**31 - 1)

from tokenizers import Tokenizer

SRC = r'c:/Users/YongsanHuang/SynologyDrive/00_Kaggle/2026_Nemotron/01_data/260514_lkevincc_golden/260515_huikang_lkall_stripped.csv'
DST = r'c:/Users/YongsanHuang/SynologyDrive/00_Kaggle/2026_Nemotron/01_data/260516_bit_manipulation_update/260516_bit_manipulation_failcorr.csv'
TOK_PATH = r'c:/Users/YongsanHuang/SynologyDrive/00_Kaggle/2026_Nemotron/02_train/260512_huikang_085/repo/tokenizer.json'
PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
# Real-script padded_len > my estimate by +2 tokens (measured on 50+ bit_manip rows whose padded_len
# equaled their actual length — i.e. were the longest in their micro-batch).
# Margin policy: 100 tokens. Even with +20 tokens of unforeseen tokenizer variance, the worst row
# stays well below 8192 in the real script (~8094 vs 8192 = ~100 tokens of true buffer).
MAX_SEQ_LEN = 8092  # = 8192 (real cap) - 100 (margin)
N_BITS = 8

_TOK = Tokenizer.from_file(TOK_PATH)


# ── Rule evaluation ──────────────────────────────────────────────────

def eval_rule(rule: str, input_bits: list[int]) -> int:
    if rule == 'default 1': return 1
    if rule in ('C0', '0'): return 0
    if rule in ('C1', '1'): return 1
    m = re.fullmatch(r'I(\d)', rule)
    if m: return input_bits[int(m.group(1))]
    m = re.fullmatch(r'NOT(\d)', rule)
    if m: return 1 - input_bits[int(m.group(1))]
    m = re.fullmatch(r'(XOR|OR|AND)(\d)(\d)', rule)
    if m:
        op, a, b = m.group(1), int(m.group(2)), int(m.group(3))
        x, y = input_bits[a], input_bits[b]
        if op == 'XOR': return x ^ y
        if op == 'OR':  return x | y
        if op == 'AND': return x & y
    m = re.fullmatch(r'(XOR|OR|AND)-NOT(\d)(\d)', rule)
    if m:
        op, a, b = m.group(1), int(m.group(2)), int(m.group(3))
        x, y = input_bits[a], 1 - input_bits[b]
        if op == 'XOR': return x ^ y
        if op == 'OR':  return x | y
        if op == 'AND': return x & y
    raise ValueError(f'Unparseable rule: {rule!r}')


def all_candidate_rules() -> list[str]:
    """Enumerate all candidate rule strings (matching huikang's format)."""
    out: list[str] = []
    out.append('C0'); out.append('C1')
    for i in range(N_BITS):
        out.append(f'I{i}'); out.append(f'NOT{i}')
    for op in ('XOR', 'OR', 'AND'):
        for a in range(N_BITS):
            for b in range(N_BITS):
                if a == b: continue
                out.append(f'{op}{a}{b}')
    for op in ('XOR', 'OR', 'AND'):
        for a in range(N_BITS):
            for b in range(N_BITS):
                if a == b: continue
                out.append(f'{op}-NOT{a}{b}')
    return out


_ALL_CANDS = all_candidate_rules()


# ── CoT parsing ──────────────────────────────────────────────────────

def parse_selected(cot: str) -> list[str] | None:
    m = re.search(r'\nSelected\n((?:\d \S+(?: \S+)*\n){8})', cot)
    if m is None: return None
    rules: list[str] = []
    for line in m.group(1).rstrip('\n').split('\n'):
        parts = line.split(' ', 1)
        if len(parts) != 2: return None
        rules.append(parts[1])
    return rules if len(rules) == N_BITS else None


def parse_examples(prompt: str) -> list[tuple[str, str]]:
    return re.findall(r'([01]{8}) -> ([01]{8})', prompt)


# ── Failure-correction block builder ─────────────────────────────────

def find_per_bit_matches(in_bits_list: list[list[int]], expected_outs: list[str]
                         ) -> list[tuple[list[str], list[str]]]:
    """For each output bit, return (full_match_rules, near_miss_rules).

    full_match: candidates that produce the correct bit for ALL examples
    near_miss : candidates that match exactly N-1 of N examples
    """
    n_ex = len(in_bits_list)
    per_bit: list[tuple[list[str], list[str]]] = []
    expected_per_bit = [[int(out[b]) for out in expected_outs] for b in range(N_BITS)]
    for b in range(N_BITS):
        full: list[str] = []
        near: list[str] = []
        exp = expected_per_bit[b]
        for cand in _ALL_CANDS:
            try:
                results = [eval_rule(cand, ib) for ib in in_bits_list]
            except ValueError:
                continue
            matches = sum(1 for r, e in zip(results, exp) if r == e)
            if matches == n_ex:
                full.append(cand)
            elif matches == n_ex - 1:
                near.append(cand)
        per_bit.append((full, near))
    return per_bit


def fmt_check(input_str: str, rule: str, expected_bit: int, b: int) -> tuple[str, bool]:
    """Return ('ex i symbol', is_correct) — compact one-token-per-example."""
    in_bits = [int(c) for c in input_str]
    got = eval_rule(rule, in_bits)
    ok = (got == expected_bit)
    return ('✓' if ok else '✗', ok)


def build_failcorr_block(examples: list[tuple[str, str]],
                         selected: list[str],
                         per_bit: list[tuple[list[str], list[str]]]) -> str | None:
    """Build the failure-correction trace. Returns None if no near-miss exists for any bit."""
    in_bits_list = [[int(c) for c in inp] for inp, _ in examples]
    expected_outs = [out for _, out in examples]
    n_ex = len(examples)

    # Find ambiguous bits (have at least one near-miss AND the selected rule is in full_match)
    ambig: list[tuple[int, str, str]] = []  # (bit_idx, near_miss_rule, selected_rule)
    for b in range(N_BITS):
        full, near = per_bit[b]
        if not near: continue
        sel = selected[b]
        if sel not in full: continue  # solver picked something that doesn't fit all; skip
        # Prefer a near-miss that is "competitive" — same family if possible, else first
        sel_family = re.match(r'([A-Z\-]+)', sel)
        sel_fam = sel_family.group(1) if sel_family else ''
        same_fam = [n for n in near if n.startswith(sel_fam)]
        chosen_near = same_fam[0] if same_fam else near[0]
        ambig.append((b, chosen_near, sel))

    if not ambig:
        return None

    lines: list[str] = ['', 'Verifying candidates by per-example check']
    # Format per ambiguous bit: ONE failed candidate + ONE accepted (the selected one)
    for b, near_rule, sel_rule in ambig:
        # Compact one-line: "Bit b: near_rule ✓✗… reject; sel_rule ✓✓…✓ accept"
        near_marks = ''
        for inp, out in examples:
            mark, _ = fmt_check(inp, near_rule, int(out[b]), b)
            near_marks += mark
        sel_marks = '✓' * n_ex  # selected rule fits all by construction
        lines.append(
            f'Bit {b}: tried {near_rule} → {near_marks} reject; '
            f'chose {sel_rule} → {sel_marks} accept'
        )
    lines.append('')
    return '\n'.join(lines)


def insert_before_selected(cot: str, block: str) -> str:
    """Insert `block` (with leading/trailing newlines) right before '\nSelected\n'."""
    m = re.search(r'\nSelected\n', cot)
    if m is None: return cot
    return cot[:m.start()] + '\n' + block + cot[m.start() + 1:]


def estimate_train_tokens(prompt: str, cot: str, answer: str) -> int:
    full = ('<|im_start|>user\n' + prompt + PROMPT_SUFFIX
            + '<|im_end|>\n<|im_start|>assistant\n<think>\n'
            + cot + '\n</think>\n\\boxed{' + answer + '}<|im_end|>')
    return len(_TOK.encode(full, add_special_tokens=False).ids)


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    n_total = n_bm = n_aug = 0
    n_skip_parse = n_skip_no_near = n_skip_solver_bad = n_skip_too_long = 0
    n_bits_with_failcorr: list[int] = []  # how many bits per row get a failure-correction line
    char_len_before: list[int] = []
    char_len_after: list[int] = []
    token_len_after: list[int] = []

    with open(SRC, encoding='utf-8', newline='') as fin, \
         open(DST, 'w', encoding='utf-8', newline='') as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            n_total += 1
            if row['category'] != 'bit_manipulation':
                writer.writerow(row); continue
            n_bm += 1
            cot = row['solver_cot']
            char_len_before.append(len(cot))
            sel = parse_selected(cot)
            exs = parse_examples(row['prompt'])
            if sel is None or not exs:
                n_skip_parse += 1; writer.writerow(row); char_len_after.append(len(cot)); continue
            in_bits_list = [[int(c) for c in inp] for inp, _ in exs]
            try:
                per_bit = find_per_bit_matches(in_bits_list, [o for _, o in exs])
            except Exception:
                n_skip_parse += 1; writer.writerow(row); char_len_after.append(len(cot)); continue
            # Verify selected rules are all in full_match (huikang solver correctness check)
            solver_ok = all(sel[b] in per_bit[b][0] for b in range(N_BITS))
            if not solver_ok:
                n_skip_solver_bad += 1
                writer.writerow(row); char_len_after.append(len(cot)); continue
            block = build_failcorr_block(exs, sel, per_bit)
            if block is None:
                n_skip_no_near += 1
                writer.writerow(row); char_len_after.append(len(cot)); continue
            new_cot = insert_before_selected(cot, block)
            n_tok = estimate_train_tokens(row['prompt'], new_cot, row.get('answer', ''))
            if n_tok > MAX_SEQ_LEN:
                # try with fewer ambiguous bits (top 3, top 2, top 1)
                # rebuild block trimmed to top-K bits
                m_block = re.search(r'\nBit \d+:', block)
                # quick trim: keep only first K lines of failure-correction
                bit_lines = re.findall(r'Bit \d+: tried .*', block)
                fitted = False
                for K in (3, 2, 1):
                    trimmed = '\n'.join([
                        '', 'Verifying candidates by per-example check'
                    ] + bit_lines[:K] + [''])
                    new_cot2 = insert_before_selected(cot, trimmed)
                    if estimate_train_tokens(row['prompt'], new_cot2, row.get('answer','')) <= MAX_SEQ_LEN:
                        new_cot = new_cot2
                        n_bits_with_failcorr.append(K)
                        fitted = True
                        break
                if not fitted:
                    n_skip_too_long += 1
                    writer.writerow(row); char_len_after.append(len(cot)); continue
            else:
                n_bits_with_failcorr.append(len(re.findall(r'Bit \d+: tried', block)))
            row['solver_cot'] = new_cot
            writer.writerow(row)
            char_len_after.append(len(new_cot))
            token_len_after.append(estimate_train_tokens(row['prompt'], new_cot, row.get('answer','')))
            n_aug += 1

    print(f'Total rows: {n_total}')
    print(f'bit_manipulation rows: {n_bm}')
    print(f'Augmented with failure-correction: {n_aug}')
    print(f'Skipped (parse error): {n_skip_parse}')
    print(f'Skipped (huikang solver picked non-full-match rule): {n_skip_solver_bad}')
    print(f'Skipped (no ambiguous bit, only one candidate per bit): {n_skip_no_near}')
    print(f'Skipped (even trimmed version exceeds MAX_SEQ_LEN): {n_skip_too_long}')

    if n_bits_with_failcorr:
        from collections import Counter
        c = Counter(n_bits_with_failcorr)
        print(f'\nNumber of ambiguous bits per augmented row:')
        for k in sorted(c): print(f'  {k} bits: {c[k]}')

    if char_len_before:
        b = sorted(char_len_before); a = sorted(char_len_after)
        n = len(b)
        print(f'\nbit_manip CoT char length:')
        print(f'  before — min={b[0]:>5}, p50={b[n//2]:>5}, p95={b[n*95//100]:>5}, max={b[-1]:>5}')
        print(f'  after  — min={a[0]:>5}, p50={a[n//2]:>5}, p95={a[n*95//100]:>5}, max={a[-1]:>5}')
    if token_len_after:
        t = sorted(token_len_after); n = len(t)
        print(f'\naugmented rows token length (full training-format wrap):')
        print(f'  min={t[0]}, p50={t[n//2]}, p95={t[n*95//100]}, max={t[-1]}')
        print(f'  rows > 8192: {sum(1 for x in t if x > 8192)} / {n}')
    print(f'\nWrote: {DST}')
    print(f'Size: {os.path.getsize(DST)/1024/1024:.2f} MB')


if __name__ == '__main__':
    main()
