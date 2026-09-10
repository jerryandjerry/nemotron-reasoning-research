#!/usr/bin/env python3
"""Regenerate <name>/track/tree_cot.txt for every cryptarithm puzzle from the CURRENT _gen_crypt.gen_cot.
The original tree-solver CoT was already archived to track/archive/ in a prior run; this only overwrites
tree_cot.txt. Input is question.txt only (one dir per puzzle)."""
import os, glob
from concurrent.futures import ProcessPoolExecutor
import _gen_crypt as G

ROOT = os.path.dirname(os.path.abspath(__file__))

def one(name):
    try:
        cot = G.gen_cot(name)
        track = os.path.join(ROOT, name, 'track'); os.makedirs(track, exist_ok=True)
        with open(os.path.join(track, 'tree_cot.txt'), 'w') as fh: fh.write(cot)
        return (name, len(cot))
    except Exception as e:
        return (name, f'ERR {e}')

if __name__ == '__main__':
    dirs = sorted(os.path.basename(os.path.dirname(q)) for q in glob.glob(os.path.join(ROOT, '*_*', 'question.txt')))
    done = errs = total = 0
    with ProcessPoolExecutor(max_workers=8) as exe:
        for name, r in exe.map(one, dirs):
            done += 1
            if isinstance(r, int): total += r
            else: errs += 1; print("ERR", name, r, flush=True)
    print(f"wrote {done} tree_cot.txt, errors={errs}, total {total:,} bytes")
