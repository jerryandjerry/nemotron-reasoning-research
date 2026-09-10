#!/usr/bin/env python3
"""Estimate the 'irrelevant tail': in the winning reading, how much of the search log comes
AFTER the point where (query operator locked + query operands pinned + result digits pinned),
i.e. the work an ideal sound query-stop could save. We approximate from the produced CoT text
by finding the LAST occurrence of the query-operands/result becoming pinned. Coarse but indicative.

Better: instrument the Engine to flag, per branch step, whether the query is already answerable.
"""
import glob, os, json
import cot_generator as cg
import _gen_crypt as gc
from tokenizers import Tokenizer

TOK = Tokenizer.from_file('/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/02_train/260512_huikang_085/repo/tokenizer.json')
PREFIXES = ('arithmetic','little_endian','mixed_concat','pure_concat','query_unseen_concat','mixed_concat_little_endian')

def all_dirs():
    out = []
    for d in sorted(glob.glob('*/')):
        name = d.rstrip('/').split('/')[-1]
        if not os.path.exists(os.path.join(d,'question.txt')): continue
        if any(name.startswith(p) for p in PREFIXES): out.append(name)
    return out

# Monkeypatch Engine._answer_ready to RECORD (not act) the first branch-step at which the query
# would be answerable in the WINNING (solution-returning) search path, plus total search steps.
import _gen_crypt
orig_search = _gen_crypt.Engine.search

if __name__ == '__main__':
    base = json.load(open('_baseline_rows.json'))
    names = all_dirs()
    # focus on puzzles currently OVER cap; bucket by category
    over = [n for n in names if base.get(n,{}).get('tok',0) > 8192]
    from collections import Counter
    cat = Counter(n.rsplit('_',1)[0] for n in over)
    print('over-cap by category:', dict(cat))
    print('total over cap:', len(over))
    # token distribution of over-cap
    ts = sorted(base[n]['tok'] for n in over)
    print('over-cap token quantiles: min', ts[0], 'p25', ts[len(ts)//4], 'median', ts[len(ts)//2], 'p75', ts[3*len(ts)//4], 'max', ts[-1])
