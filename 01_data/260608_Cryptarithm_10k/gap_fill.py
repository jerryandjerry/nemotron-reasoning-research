#!/usr/bin/env python3
"""Close the last provider-pair coverage gaps in the 10k: targeted-generate a short trainable puzzle for
each still-missing token, then SWAP it into the harvest CSV (replacing a same-subtype row) so the count
stays 10,000 and the 30/30/10/10/10/10 balance is preserved. Then re-verify full provider coverage."""
import os, sys, csv, re, random
csv.field_size_limit(10 ** 8)
HERE = os.path.dirname(os.path.abspath(__file__))
AUG = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260607_Cryptarithm_Aug'
CRYPT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260525_Cryptarithm'
sys.path.insert(0, AUG); sys.path.insert(0, CRYPT)
import _gen_cover as gc
HARVEST = os.path.join(HERE, '260608_Cryptarithm_10k_harvest.csv')
COARSE = {'arithmetic': 'cryptarithm_deduce', 'little_endian': 'cryptarithm_deduce', 'pure_concat': 'cryptarithm_deduce',
          'mixed_concat': 'cryptarithm_deduce', 'mixed_concat_little_endian': 'cryptarithm_deduce', 'query_unseen_concat': 'cryptarithm_guess'}

TARGETS = ["']->", "(-\\", "})("]   # the 3 provider tokens the 10k misses

def make(token, rng):
    """Try to build a short trainable puzzle embedding `token` (concat for all-operand, arith for operator)."""
    has_op = any(c in gc.OPS for c in token)
    tmp = os.path.join('/tmp', 'gapfill_tmp')
    for _ in range(800):
        res = gc.build_arith(token, rng) if has_op else gc.build_concat(token, rng)
        if not res: continue
        text, gold, sub = res
        if token not in gc.fused(text): continue
        tb = gc.try_blind(text, gold, tmp)
        if not tb: continue
        cot, ntok = tb
        return text, gold, cot, ntok, sub
    return None

def main():
    rng = random.Random(20260609)
    rows = list(csv.DictReader(open(HARVEST))); cols = rows[0].keys()
    by_sub = {}
    for i, r in enumerate(rows): by_sub.setdefault(r['subtype'], []).append(i)
    made = []
    for tok in TARGETS:
        r = make(tok, rng)
        if not r:
            print(f"  {tok!r}: could not build a short trainable puzzle (BPE-unreachable here)"); continue
        text, gold, cot, ntok, sub = r
        made.append((tok, text, gold, cot, ntok, sub))
        print(f"  {tok!r}: COVERED via {sub} ({ntok} tok)")
    # swap each into a random same-subtype row (preserves count + balance)
    used = set()
    for tok, text, gold, cot, ntok, sub in made:
        cands = [i for i in by_sub[sub] if i not in used]
        if not cands: cands = [i for i in range(len(rows)) if i not in used]
        j = rng.choice(cands); used.add(j)
        rows[j] = {'id': rows[j]['id'], 'subtype': sub, 'category': COARSE[sub], 'prompt': text,
                   'answer': gold, 'solver_cot': cot, 'token length': str(ntok)}
    with open(HARVEST, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(cols)); w.writeheader(); w.writerows(rows)
    print(f"swapped {len(made)} targeted rows into the 10k (count + balance preserved)")

if __name__ == '__main__':
    main()
