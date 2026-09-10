#!/usr/bin/env python3
"""v3 harvest: ORIGINAL provider machinery (ported, byte-validated) + blind word-prior steering
+ native conditional Maj section. Blind: question.txt only; gold used only for the post-hoc filter."""
import os, re, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _gen_bitm_v3 as V
import _gen_bitm as W                    # word_solve + decompose (the blind prior)
from _store_types import Problem, Example
from tokenizers import Tokenizer
TOK = Tokenizer.from_file('/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/study/2606090112_grpo/cryptarithm_solver/tokenizer.json')

def solve_folder(d):
    q = open(os.path.join(d, 'question.txt')).read()
    exs = re.findall(r'([01]{8}) -> ([01]{8})', q)
    qq = re.search(r'determine the output for: ([01]{8})', q).group(1)
    h = W.word_solve(exs)
    if h is None or h[0] == 'TERM3':
        return 'UNEXPRESSIBLE', None
    sel = W.decompose(h)
    if sel is None:
        return 'UNEXPRESSIBLE', None
    V._SECTION_PREF.clear()
    V._PREFERRED = [lab for _, lab, _ in sel]
    try:
        out = V.reasoning_bit_manipulation(
            Problem(id=d, category='bit_manipulation',
                    examples=[Example(a, b) for a, b in exs], question=qq, answer=''))
    finally:
        V._PREFERRED = None; V._SECTION_PREF.clear()
    if out == 'NEEDS_CH': return 'HELD_CH', None
    if out is None: return 'UNEXPRESSIBLE', None
    return 'WRITTEN', out

if __name__ == '__main__':
    stats = {}; lens_gt = []
    rows = []
    for d in sorted(glob.glob('bitm_*')):
        st, cot = solve_folder(d)
        if st != 'WRITTEN':
            stats[st] = stats.get(st, 0) + 1; rows.append((d, st, 0)); continue
        os.makedirs(os.path.join(d, 'track'), exist_ok=True)
        open(os.path.join(d, 'track', 'tree_cot.txt'), 'w').write(cot)
        gold = open(os.path.join(d, 'answer.txt')).read().strip()
        box = re.findall(r'\\boxed\{([^}]*)\}', cot)[-1]
        ntok = len(TOK.encode(open(os.path.join(d, 'question.txt')).read() + cot, add_special_tokens=False).ids)
        st2 = 'GT' if box == gold else 'WRONG'
        stats[st2] = stats.get(st2, 0) + 1; rows.append((d, st2, ntok))
        if st2 == 'GT': lens_gt.append(ntok)
    print('V3 HARVEST (original machinery + steering):', stats)
    import csv as _csv
    with open('_harvest_summary.csv', 'w', newline='') as fh:
        w = _csv.writer(fh); w.writerow(['folder', 'status', 'ntok']); w.writerows(rows)
    v = sorted(lens_gt)
    if v: print(f'GT token lengths: p50 {v[len(v)//2]} p90 {v[int(.9*len(v))]} max {v[-1]} | >=7680: {sum(1 for x in v if x>=7680)}')
