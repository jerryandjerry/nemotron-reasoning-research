#!/usr/bin/env python3
"""Regenerate the honest CoT for every puzzle -> <name>/track/tree_cot.txt.
Archives the ORIGINAL tree_cot.txt to <name>/track/archive/tree_cot.txt (once, idempotent);
removes the interim honest_cot.txt. NON-DESTRUCTIVE to the original."""
import os, glob, shutil
from concurrent.futures import ProcessPoolExecutor
import _honest_solver as hs

def one(name):
    try:
        cot, A, variants, mode = hs.gen(name)
        track = f'{name}/track'; os.makedirs(f'{track}/archive', exist_ok=True)
        orig = f'{track}/tree_cot.txt'; archived = f'{track}/archive/tree_cot.txt'
        if os.path.exists(orig) and not os.path.exists(archived):
            shutil.move(orig, archived)            # preserve the ORIGINAL tree-solver CoT, exactly once
        with open(orig, 'w') as fh: fh.write(cot)  # new honest CoT now lives at tree_cot.txt
        hc = f'{track}/honest_cot.txt'
        if os.path.exists(hc): os.remove(hc)       # drop the interim name
        return (name, len(cot))
    except Exception as e:
        return (name, f'ERR {e}')

if __name__ == '__main__':
    dirs = sorted(set(glob.glob('arithmetic_*/') + glob.glob('little_endian_*/') +
        [d for d in glob.glob('mixed_concat_*/') if not d.startswith('mixed_concat_little_endian')] +
        glob.glob('mixed_concat_little_endian_*/') + glob.glob('query_unseen_concat_*/') + glob.glob('pure_concat_*/')))
    dirs = [d.rstrip('/') for d in dirs]
    done=errs=total=0
    with ProcessPoolExecutor(max_workers=8) as exe:
        for name, r in exe.map(one, dirs):
            done += 1
            if isinstance(r, int): total += r
            else: errs += 1; print("ERR", name, r, flush=True)
    print(f"wrote {done} tree_cot.txt, errors={errs}, total {total:,} bytes")
