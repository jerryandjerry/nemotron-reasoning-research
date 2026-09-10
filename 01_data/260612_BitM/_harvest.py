#!/usr/bin/env python3
"""Blind-solve all 248 puzzle folders with _gen_bitm.gen_cot (question.txt only), write
track/tree_cot.txt, then GT-check the boxed answer vs answer.txt (post-hoc filter only)."""
import os, re, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _gen_bitm as G
from tokenizers import Tokenizer
TOK = Tokenizer.from_file('/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/study/2606090112_grpo/cryptarithm_solver/tokenizer.json')

stats = {'written': 0, 'gt_true': 0, 'gt_false': 0, 'unexpressible': 0, 'error': 0}
lens_all, lens_gt = [], []
rows = []
for d in sorted(glob.glob('bitm_*')):
    try:
        cot = G.gen_cot(d)
    except Exception as e:
        stats['error'] += 1; rows.append((d, 'ERROR', str(e)[:60], 0)); continue
    if cot is None:
        stats['unexpressible'] += 1; rows.append((d, 'UNEXPRESSIBLE', '', 0)); continue
    if cot == 'NEEDS_CH':
        stats['held_ch'] = stats.get('held_ch', 0) + 1; rows.append((d, 'HELD_CH', 'awaiting Ch wording decision', 0)); continue
    os.makedirs(os.path.join(d, 'track'), exist_ok=True)
    open(os.path.join(d, 'track', 'tree_cot.txt'), 'w').write(cot)
    stats['written'] += 1
    gold = open(os.path.join(d, 'answer.txt')).read().strip()
    box = re.findall(r'\\boxed\{([^}]*)\}', cot)[-1]
    q = open(os.path.join(d, 'question.txt')).read()
    ntok = len(TOK.encode(q + cot, add_special_tokens=False).ids)
    lens_all.append(ntok)
    if box == gold:
        stats['gt_true'] += 1; lens_gt.append(ntok); rows.append((d, 'GT', '', ntok))
    else:
        stats['gt_false'] += 1; rows.append((d, 'WRONG', f'{box} vs {gold}', ntok))
print('HARVEST:', stats)
import csv as _csv
with open('_harvest_summary.csv', 'w', newline='') as fh:
    w = _csv.writer(fh); w.writerow(['folder', 'status', 'note', 'ntok']); w.writerows(rows)
def dist(v, name):
    if not v: return
    v = sorted(v)
    print(f'{name}: n={len(v)} min={v[0]} p25={v[len(v)//4]} p50={v[len(v)//2]} p75={v[3*len(v)//4]} p90={v[int(.9*len(v))]} max={v[-1]} | >=7680: {sum(1 for x in v if x>=7680)}')
dist(lens_all, 'token length, all written CoTs (prompt+cot)')
dist(lens_gt, 'token length, GT-TRUE CoTs        (prompt+cot)')
