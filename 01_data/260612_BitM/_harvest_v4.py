#!/usr/bin/env python3
"""v4: emit a CoT for ALL 248 puzzles. Steered (word-prior) where the hypothesis is expressible;
positions/puzzles outside the vocabulary fall back to the ORIGINAL native machinery (incl. its
'default 1' guessing) so the trace always completes. Statuses:
  GT        — boxed == gold, no guessed positions (honest derivation; trainable)
  GT_GUESS  — boxed == gold but trace contains 'default 1' (lucky guess; NOT for training)
  WRONG     — boxed != gold (filtered)"""
import os, re, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _gen_bitm_v3 as V
import _gen_bitm as W
from _store_types import Problem, Example
from tokenizers import Tokenizer
TOK = Tokenizer.from_file('/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/study/2606090112_grpo/cryptarithm_solver/tokenizer.json')

def gen(d):
    q = open(os.path.join(d, 'question.txt')).read()
    exs = re.findall(r'([01]{8}) -> ([01]{8})', q)
    qq = re.search(r'determine the output for: ([01]{8})', q).group(1)
    pref = None
    h = W.word_solve(exs)
    if h is not None and h[0] != 'TERM3':
        sel = W.decompose(h)
        if sel is not None:
            pref = [lab for _, lab, _ in sel]
            # Ch positions have no approved notation -> release them to the native path
            pref = [(l if not l.startswith('Ch') else '\x00none') for l in pref]
    V._SECTION_PREF.clear(); V._PREFERRED = pref
    try:
        out = V.reasoning_bit_manipulation(
            Problem(id=d, category='bit_manipulation',
                    examples=[Example(a, b) for a, b in exs], question=qq, answer=''))
    finally:
        V._PREFERRED = None; V._SECTION_PREF.clear()
    return out, pref is not None

if __name__ == '__main__':
    stats = {}; rows = []; lens = []
    for d in sorted(glob.glob('bitm_*')):
        out, steered = gen(d)
        if out is None or out == 'NEEDS_CH':
            # complete native fallback, fully unsteered
            q = open(os.path.join(d, 'question.txt')).read()
            exs = re.findall(r'([01]{8}) -> ([01]{8})', q)
            qq = re.search(r'determine the output for: ([01]{8})', q).group(1)
            V._PREFERRED = None; V._SECTION_PREF.clear()
            out = V.reasoning_bit_manipulation(
                Problem(id=d, category='bit_manipulation',
                        examples=[Example(a, b) for a, b in exs], question=qq, answer=''))
            steered = False
        os.makedirs(os.path.join(d, 'track'), exist_ok=True)
        open(os.path.join(d, 'track', 'tree_cot.txt'), 'w').write(out)
        gold = open(os.path.join(d, 'answer.txt')).read().strip()
        box = re.findall(r'\\boxed\{([^}]*)\}', out)[-1]
        guessed = bool(re.search(r'(?m)^\d default 1$', out))
        st = ('GT' if not guessed else 'GT_GUESS') if box == gold else 'WRONG'
        ntok = len(TOK.encode(open(os.path.join(d, 'question.txt')).read() + out, add_special_tokens=False).ids)
        stats[st] = stats.get(st, 0) + 1
        rows.append((d, st, 'steered' if steered else 'native', ntok))
        if st == 'GT': lens.append(ntok)
    print('V4 — every puzzle has a CoT:', stats, f'(total {sum(stats.values())})')
    import csv as _csv
    with open('_harvest_summary.csv', 'w', newline='') as fh:
        w = _csv.writer(fh); w.writerow(['folder', 'status', 'mode', 'ntok']); w.writerows(rows)
    v = sorted(lens)
    if v: print(f'GT (trainable) lengths: p50 {v[len(v)//2]} p90 {v[int(.9*len(v))]} max {v[-1]} | >=7680: {sum(1 for x in v if x>=7680)}')
