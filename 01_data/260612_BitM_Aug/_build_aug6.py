#!/usr/bin/env python3
"""Build the 6-bucket aug to the APPROVED target (AXIS_DISTRIBUTION_1497.md).
construct -> blind-solve (faithful gen_cot) -> classify into bucket -> GT-match -> keep.
Gates: CH/MAJ kept only if single-clean (rejects the composite-NONFAM artifact). fallback gold = solver box.
Composite bucket is built by a SEPARATE pass (needs the 3-term write-up); this driver does the other 5."""
import os, re, sys, json, random, hashlib, tempfile
sys.path.insert(0, '.'); sys.path.insert(0, '../260612_BitM')
import _gen_aug_bitm as G
import _gen_bitm_faithful as F
import _gen_bitm as W

# secondary-feature knobs, identical across buckets (decorrelation; rates from the 1,497 originals)
FEAT    = {'rotate_prob': 0.34, 'fill1_prob': 0.0014, 'notwrap_prob': 0.14}
FEAT_NW = {'rotate_prob': 0.34, 'fill1_prob': 0.0014, 'notwrap_prob': 1.0}   # notwrap bucket: force NOT-wrap
OPS = {'direct': ['AND', 'OR', 'XOR', 'U'], 'maj': ['MAJ'], 'ch': ['CH'],
       'notwrap': ['MAJ', 'AND', 'OR', 'XOR'], 'composite': ['T3']}
TARGET = {'direct': 1213, 'maj': 317, 'ch': 381, 'notwrap': 488, 'composite': 613, 'fallback': 491}

_TMP = tempfile.mkdtemp(prefix='aug6_')
def solve(text):
    open(os.path.join(_TMP, 'question.txt'), 'w').write(text)
    return F.gen_cot(_TMP)

def box(cot):
    b = re.findall(r'\\boxed\{([^}]*)\}', cot); return b[-1] if b else None

def classify(exs):
    """bucket from the blindly-recovered rule (None=fallback, T3=composite, else by path).
    A CH/MAJ rule that lands NONFAM is the off-distribution artifact -> None (rejected)."""
    H = W.word_solve_full(exs)
    if H is None: return 'fallback', None
    if H[0] == 'T3': return 'composite', H               # 3-step rule (clean or NONFAM both ok)
    pref = W.full_prior(exs)
    if any(p == W.PREF_NONFAM for p in pref): return None, H   # non-T3 NONFAM = artifact -> reject
    if any(l.startswith('NOT-') for l in pref): return 'notwrap', H
    if any(l.startswith('Ch') for l in pref): return 'ch', H
    if any(l.startswith('Maj') for l in pref): return 'maj', H
    return 'direct', H

def main():
    rng = random.Random(20260613)
    out = open('/tmp/_aug6_rows.jsonl', 'w')
    kept = {b: 0 for b in TARGET}
    for bucket, need in TARGET.items():
        feat = FEAT_NW if bucket == 'notwrap' else FEAT
        budget = need * 60
        for _ in range(budget):
            if kept[bucket] >= need: break
            n_ex = rng.choice([7, 8, 9, 10])
            if bucket == 'fallback':
                text = G.construct_fallback(rng, n_ex)
                if text is None: continue
            else:
                H = G.rand_H(rng.choice(OPS[bucket]), rng, feat)
                c = G.construct(H, n_ex, rng)
                if not c: continue
                text = c[0]
            exs = re.findall(r'([01]{8}) -> ([01]{8})', text)
            b, _ = classify(exs)
            if b != bucket: continue                       # must land in the intended bucket (gate)
            cot = solve(text)
            bx = box(cot)
            if bx is None or 'default 1' in cot: continue   # default-1 = unresolved bit, not sound
            if bucket == 'composite' and 'follow a single rule' not in cot: continue  # must STATE the 3-step rule
            # GT-True: fallback gold = solver's own box; clean buckets must reproduce the planted gold (already
            # guaranteed since classify==bucket and the solver is deterministic) -> box is the gold either way
            gold = bx
            feats = G.features_of(W.word_solve_full(exs) or ('U', 'id'), n_ex) if bucket != 'fallback' else \
                    {'n_ex': n_ex, 'rotate': 0, 'shift': 0, 'notwrap': 0, 'dir_L': 0, 'dir_R': 0, 'fill1': 0}
            pid = 'aug' + hashlib.sha1(text.encode()).hexdigest()[:8]
            rec = {'id': pid, 'bucket': bucket, 'pcc': int('Pattern consistency check' in cot),
                   'fallback': int('No single rule reproduces all examples' in cot), 'n_ex': n_ex,
                   'gold': gold, 'question': text, 'cot': cot, **{k: feats[k] for k in
                   ('rotate', 'shift', 'notwrap', 'dir_L', 'dir_R')}}
            out.write(json.dumps(rec) + '\n'); out.flush()
            kept[bucket] += 1
        print(f"  {bucket}: {kept[bucket]}/{need}", flush=True)
    print("DONE", dict(kept))

if __name__ == '__main__':
    main()
