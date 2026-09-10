#!/usr/bin/env python3
"""Build the bitm augmentation: construct -> blind-solve (full faithful CoT) -> GT+path filter ->
stratify by (path, ambiguity) to the target table. Writes aug_<id>/ folders + manifest.
Decorrelation: one global feature knob each, identical across paths; ~30% PCC target per path."""
import os, re, sys, random, json, argparse, hashlib
sys.path.insert(0, '.')
import _gen_aug_bitm as G
import _gen_bitm_faithful as F
from _store_types import Problem, Example
import _gen_bitm_v3 as V

FEAT = {'rotate_prob': 0.31, 'fill1_prob': 0.0, 'notwrap_prob': 0.14}
FEAT_NW = {'rotate_prob': 0.34, 'fill1_prob': 0.0, 'notwrap_prob': 1.0}   # force NOT-wrap for notwrap path
PCC_TARGET = 0.30
STRAT = {  # path -> generation operation pool
    'direct': ['XOR', 'AND', 'OR', 'U'],
    'maj': ['MAJ'], 'ch': ['CH'], '3term': ['T3'], 'notwrap': ['MAJ', 'AND', 'OR', 'XOR'],
}

import tempfile, shutil
_TMP = tempfile.mkdtemp(prefix='augbitm_')
def gen_cot_text(text):
    """Run the tested faithful solver on a constructed puzzle (blind, via temp folder)."""
    open(os.path.join(_TMP, 'question.txt'), 'w').write(text)
    return F.gen_cot(_TMP)

def cot_path(cot):
    if re.search(r'\n(AND|OR|XOR)\((AND|OR|XOR)\)\n', cot): return '3term'
    if re.search(r'\nNOT-(AND|OR|XOR|Maj)\n', cot): return 'notwrap'
    if 'Matching output with Ch' in cot: return 'ch'
    if 'Matching output with Maj' in cot: return 'maj'
    return 'direct'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default='direct:844,maj:630,ch:656,notwrap:681,3term:692')
    ap.add_argument('--seed', type=int, default=20260612)
    ap.add_argument('--out', default='_aug_manifest.csv')
    a = ap.parse_args()
    targets = {k: int(v) for k, v in (kv.split(':') for kv in a.targets.split(','))}
    rng = random.Random(a.seed)
    import csv
    manifest = []
    for path, need in targets.items():
        want_pcc = round(need * PCC_TARGET); want_no = need - want_pcc
        pool_pcc = []; pool_no = []
        ops = STRAT[path]; feat = FEAT_NW if path == 'notwrap' else FEAT
        budget = need * 400
        for _ in range(budget):
            if len(pool_pcc) + len(pool_no) >= need: break        # fill to need; PCC is CAPPED, not required
            H = G.rand_H(rng.choice(ops), rng, feat)
            if G.path_of_H(H) != path: continue
            n_ex = rng.choice([9, 10] if path == '3term' else [7, 8, 9, 10])
            c = G.construct(H, n_ex, rng)
            if not c: continue
            text, gold = c
            cot = gen_cot_text(text)
            if re.findall(r'\\boxed\{([^}]*)\}', cot)[-1] != gold or 'default 1' in cot: continue
            if cot_path(cot) != path: continue
            has = 'Pattern consistency check' in cot
            if has and len(pool_pcc) >= want_pcc: continue          # cap PCC at the 30% target
            feats = G.features_of(H, n_ex)
            (pool_pcc if has else pool_no).append(1)
            pid = 'aug' + hashlib.sha1(text.encode()).hexdigest()[:8]
            rec = {'id': pid, 'path': path, 'pcc': int(has), 'n_ex': n_ex,
                   'rotate': feats['rotate'], 'shift': feats['shift'], 'notwrap': feats['notwrap'],
                   'dir_L': feats['dir_L'], 'dir_R': feats['dir_R'], 'gold': gold,
                   'question': text, 'cot': cot}
            with open('/tmp/_aug_direct.jsonl', 'a') as jf: jf.write(json.dumps(rec) + '\n')
            manifest.append(rec)
        print(f"  {path}: {len(pool_pcc)+len(pool_no)}/{need}  (pcc {len(pool_pcc)}, no-pcc {len(pool_no)})", flush=True)
    from collections import Counter
    print(f"TOTAL aug: {len(manifest)} | by path: {dict(Counter(r['path'] for r in manifest))}")

if __name__ == '__main__':
    main()
