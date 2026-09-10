#!/usr/bin/env python3
"""Measure baseline CoT length / token counts across all 800 puzzles, capturing the answer.
Used to A/B test branching / value-ordering heuristics: any alternative MUST yield the SAME answer."""
import glob, os, re, sys, json
import cot_generator as cg
import _gen_crypt as gc
from tokenizers import Tokenizer

TOK = Tokenizer.from_file('/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/02_train/260512_huikang_085/repo/tokenizer.json')

PREFIXES = ('arithmetic','little_endian','mixed_concat','pure_concat','query_unseen_concat','mixed_concat_little_endian')

def all_dirs():
    dirs = sorted(glob.glob('*/'))
    out = []
    for d in dirs:
        name = d.rstrip('/').split('/')[-1]
        if not os.path.exists(os.path.join(d,'question.txt')): continue
        if not any(name.startswith(p) for p in PREFIXES): continue
        out.append(name)
    return out

def boxed(cot):
    i = cot.rfind('\\boxed{')
    if i < 0: return None
    j = cot.find('}', i)
    return cot[i+7:j]

def measure(names=None, verbose=False):
    if names is None: names = all_dirs()
    rows = {}
    for name in names:
        try:
            cot = gc.gen_cot(name)
        except Exception as ex:
            rows[name] = {'err': str(ex)}; continue
        ntok = len(TOK.encode(cot).ids)
        ans = boxed(cot)
        rows[name] = {'tok': ntok, 'ans': ans, 'lines': cot.count('\n')+1}
    return rows

def summarize(rows, label=''):
    ok = {k:v for k,v in rows.items() if 'tok' in v}
    errs = {k:v for k,v in rows.items() if 'err' in v}
    toks = [v['tok'] for v in ok.values()]
    over = [k for k,v in ok.items() if v['tok'] > 8192]
    toks_sorted = sorted(toks)
    n = len(toks)
    p = lambda q: toks_sorted[min(n-1, int(q*n))]
    print(f"=== {label} ===")
    print(f"puzzles={n}  errors={len(errs)}")
    print(f"tokens: mean={sum(toks)/n:.0f}  median={p(0.5)}  p90={p(0.9)}  p95={p(0.95)}  p99={p(0.99)}  max={max(toks)}")
    print(f"over 8192: {len(over)}  -> {sorted(over)[:20]}")
    if errs: print(f"errors: {list(errs.items())[:5]}")
    return ok, over

if __name__ == '__main__':
    rows = measure()
    summarize(rows, 'BASELINE')
    json.dump({k:v for k,v in rows.items()}, open('_baseline_rows.json','w'))
    print("saved _baseline_rows.json")
