#!/usr/bin/env python3
"""BLIND solver driver. For each puzzle id in the slice file, reads ONLY question.txt, runs the solver
(gen_cot with answer='' — the solver is forward-only and never uses the answer anyway), writes
track/tree_cot.txt, and records the boxed answer. It NEVER opens answer.txt or the manifest — so the
solve is provably independent of the gold. Usage: python3 _solve_blind.py <slice_file> <out_json>"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
GEN = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260606_Numeric_Equation'
sys.path.insert(0, GEN)
from _gen_eq import parse_prompt, gen_cot

def main():
    slice_file, out_json = sys.argv[1], sys.argv[2]
    ids = [l.strip() for l in open(slice_file) if l.strip()]
    res = {}
    errs = {}
    for pid in ids:
        d = os.path.join(HERE, pid)
        prompt = open(os.path.join(d, 'question.txt')).read()        # the ONLY file read
        ex, qn = parse_prompt(prompt)
        try:
            cot, ans = gen_cot({'prompt': prompt, 'answer': '', 'examples': ex, 'question': qn})
        except Exception as e:                                       # isolate per-puzzle crashes so one bad puzzle doesn't abort the slice
            errs[pid] = f"{type(e).__name__}: {e}"
            res[pid] = None
            continue
        os.makedirs(os.path.join(d, 'track'), exist_ok=True)
        open(os.path.join(d, 'track', 'tree_cot.txt'), 'w').write(cot)
        res[pid] = ans
    json.dump(res, open(out_json, 'w'))
    n_ok = len([p for p in res if res[p] is not None])
    print(f"solved {n_ok} -> {out_json}")
    if errs:
        print(f"errored {len(errs)}: " + "; ".join(f"{p} ({m})" for p, m in errs.items()))

if __name__ == '__main__':
    main()
