#!/usr/bin/env python3
"""Stratified 6-bucket aug build (§4b technique 4: don't let the GT/bucket filter re-introduce skew).
Over-generate a POOL per bucket, then subsample to: n_examples flat 25/25/25/25 (the only model-VISIBLE
feature) AND per-operand rotate-rate ~= the originals' 34% (greedy transform balance)."""
import os, re, sys, json, random, hashlib, tempfile
sys.path.insert(0, '.'); sys.path.insert(0, '../260612_BitM')
import _gen_aug_bitm as G
import _gen_bitm_faithful as F
import _gen_bitm as W
from tokenizers import Tokenizer
TOK = Tokenizer.from_file('/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/02_train/260512_huikang_085/repo/tokenizer.json')

FEAT    = {'rotate_prob': 0.34, 'fill1_prob': 0.0014, 'notwrap_prob': 0.14}
FEAT_NW = {'rotate_prob': 0.34, 'fill1_prob': 0.0014, 'notwrap_prob': 1.0}
OPS = {'direct': ['AND', 'OR', 'XOR', 'U'], 'maj': ['MAJ'], 'ch': ['CH'],
       'notwrap': ['MAJ', 'AND', 'OR', 'XOR'], 'composite': ['T3']}
TARGET = {'direct': 1026, 'maj': 440, 'ch': 449, 'notwrap': 484, 'composite': 613, 'fallback': 491}
ROT_TARGET = 0.34

_TMP = tempfile.mkdtemp(prefix='aug6s_')
def solve(text):
    open(os.path.join(_TMP, 'question.txt'), 'w').write(text); return F.gen_cot(_TMP)
def box(cot):
    b = re.findall(r'\\boxed\{([^}]*)\}', cot); return b[-1] if b else None

def classify(exs):
    H = W.word_solve_full(exs)
    if H is None: return 'fallback'
    if H[0] == 'T3': return 'composite'
    pref = W.full_prior(exs)
    if any(p == W.PREF_NONFAM for p in pref): return None        # non-T3 NONFAM = artifact
    if any(l.startswith('NOT-') for l in pref): return 'notwrap'
    if any(l.startswith('Ch') for l in pref): return 'ch'
    if any(l.startswith('Maj') for l in pref): return 'maj'
    return 'direct'

def gen_one(bucket, rng):
    feat = FEAT_NW if bucket == 'notwrap' else FEAT
    n_ex = rng.choice([7, 8, 9, 10])
    if bucket == 'fallback':
        text = G.construct_fallback(rng, n_ex)
        if text is None: return None
    else:
        H = G.rand_H(rng.choice(OPS[bucket]), rng, feat)
        c = G.construct(H, n_ex, rng)
        if not c: return None
        text = c[0]
    exs = re.findall(r'([01]{8}) -> ([01]{8})', text)
    if classify(exs) != bucket: return None
    cot = solve(text); bx = box(cot)
    if bx is None or 'default 1' in cot: return None
    if bucket == 'composite' and 'follow a single rule' not in cot: return None
    if len(TOK.encode(cot).ids) >= 7680: return None   # trainable cap (reject over-cap; never truncate)
    H2 = W.word_solve_full(exs)
    feats = G.features_of(H2, n_ex) if (H2 and bucket != 'fallback') else \
            {'rotate': 0, 'shift': 0, 'notwrap': 0, 'dir_L': 0, 'dir_R': 0}
    pid = 'aug' + hashlib.sha1(text.encode()).hexdigest()[:8]
    return {'id': pid, 'bucket': bucket, 'pcc': int('Pattern consistency check' in cot),
            'fallback': int('No single rule reproduces all examples' in cot), 'n_ex': n_ex,
            'gold': bx, 'question': text, 'cot': cot,
            **{k: feats[k] for k in ('rotate', 'shift', 'notwrap', 'dir_L', 'dir_R')}}

def subsample(pool, need):
    """Stratify by n_ex (flat) + greedily balance per-operand rotate-rate to ROT_TARGET."""
    by = {n: [r for r in pool if r['n_ex'] == n] for n in (7, 8, 9, 10)}
    for n in by: random.Random(n).shuffle(by[n])
    quota = [need // 4 + (1 if i < need % 4 else 0) for i in range(4)]   # per n_ex
    sel = []; R = S = 0
    for n, q in zip((7, 8, 9, 10), quota):
        cands = by[n]
        # greedy: each step pick the candidate pulling rotate-rate toward target
        avail = cands[:]
        for _ in range(min(q, len(avail))):
            def score(r):
                nr, ns = R + r['rotate'], S + r['shift']
                rate = nr / (nr + ns) if (nr + ns) else 0
                return abs(rate - ROT_TARGET)
            avail.sort(key=score)
            r = avail.pop(0); sel.append(r); R += r['rotate']; S += r['shift']
    return sel

def main():
    rng = random.Random(20260613)
    out = open('/tmp/_aug6s_rows.jsonl', 'w')
    for bucket, need in TARGET.items():
        pool = []; budget = need * 200; pool_target = int(need * 1.7)
        for _ in range(budget):
            if len(pool) >= pool_target: break
            r = gen_one(bucket, rng)
            if r: pool.append(r)
        # dedup by id
        seen = set(); pool = [r for r in pool if not (r['id'] in seen or seen.add(r['id']))]
        sel = subsample(pool, need)   # subsample stratifies n_ex for all buckets (transform-greedy is a no-op for fallback)
        for r in sel: out.write(json.dumps(r) + '\n')
        out.flush()
        rot = sum(r['rotate'] for r in sel); sh = sum(r['shift'] for r in sel)
        import collections
        nex = collections.Counter(r['n_ex'] for r in sel)
        print(f"  {bucket}: pool {len(pool)} -> kept {len(sel)} | rotate {rot/(rot+sh)*100 if rot+sh else 0:.0f}% | "
              f"nex {dict(sorted(nex.items()))}", flush=True)
    print("DONE")

if __name__ == '__main__':
    main()
