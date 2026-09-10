#!/usr/bin/env python3
"""Build the DOWNSAMPLED full CSV: base non-crypt (byte-identical) + the 800 provider rows + a
coverage-aware 2693-row subset of the 10k aug (rare-token-first set-cover + stratified balance-fill),
so cryptarithm lands ~28% of training (3000 trainable) with ZERO coverage lost vs the full 10k."""
import os, sys, csv, re, json, random
from collections import defaultdict, Counter
csv.field_size_limit(10 ** 8)
HERE = os.path.dirname(os.path.abspath(__file__))
CRYPT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260525_Cryptarithm'
sys.path.insert(0, CRYPT)
from tokenizers import Tokenizer
TOK = Tokenizer.from_file('/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/02_train/260512_huikang_085/repo/tokenizer.json')
POOL = set('''!"#$%&'()/:<>?@[\\]^`{|}+-*'''); OPS = set('+-*')
ALL = set(json.load(open(os.path.join(CRYPT, 'full_merge_table.json')))['map'])
ACH = {t for t in ALL if sum(c in OPS for c in t) <= 1}
BASE = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260607_Cryptarithm_Aug/260607_CryptAug_TrainData/260609_CryptAug_eqHardened_FULL.csv'
HARVEST = os.path.join(HERE, '260608_Cryptarithm_10k_harvest.csv')
OUT = os.path.join(HERE, '260609_Crypt3k_eqHardened_FULL.csv')
CC = ('cryptarithm_deduce', 'cryptarithm_guess')
SUBS = ['arithmetic', 'little_endian', 'pure_concat', 'mixed_concat', 'mixed_concat_little_endian', 'query_unseen_concat']
COMBINED = 3000

def toks(p):
    out = set()
    for line in p.strip().splitlines():
        if line.startswith('In'): continue
        for run in line.replace('Now, determine the result for:', '').replace(' = ', ' ').split():
            for a, b in TOK.encode(run, add_special_tokens=False).offsets:
                seg = run[a:b]
                if len(seg) >= 2 and all(c in POOL for c in seg): out.add(seg)
    return out & ALL
def reading(cot): return 'leftward' if 'reading order = leftward' in cot else 'standard'

def main():
    rng = random.Random(20260609)
    base = list(csv.DictReader(open(BASE))); cols = list(base[0].keys())
    provider = [r for r in base if r['source'] == '260601_new_solver']            # all 800 (307 1x + 493 0x)
    prov_train = [r for r in provider if r['oversampling'] == '1']                 # 307 (seed coverage)
    for r in prov_train: r['_t'] = toks(r['prompt'])

    harv = list(csv.DictReader(open(HARVEST)))
    for r in harv: r['_t'] = toks(r['prompt'])
    ratio = {'arithmetic': .30, 'little_endian': .30, 'pure_concat': .10, 'mixed_concat': .10,
             'mixed_concat_little_endian': .10, 'query_unseen_concat': .10}
    prov_n = Counter(r['new label'] for r in prov_train)
    target = {s: max(0, round(COMBINED * ratio[s]) - prov_n.get(s, 0)) for s in SUBS}

    pool = {s: [r for r in harv if r['subtype'] == s] for s in SUBS}
    for s in SUBS: rng.shuffle(pool[s])
    chosen = {s: [] for s in SUBS}; ids = set()
    covered = set().union(*[r['_t'] for r in prov_train]) if prov_train else set()
    # Phase A: rare-first set-cover
    t2p = defaultdict(list)
    for s in SUBS:
        for r in pool[s]:
            for t in (r['_t'] & ACH): t2p[t].append((s, r))
    for t in sorted(ACH, key=lambda t: len(t2p[t])):
        if t in covered or not t2p[t]: continue
        cand = [(s, r) for (s, r) in t2p[t] if r['id'] not in ids and len(chosen[s]) < target[s]]
        if not cand: continue
        s, r = max(cand, key=lambda sr: len((sr[1]['_t'] & ACH) - covered))
        chosen[s].append(r); ids.add(r['id']); covered |= r['_t']
    # Phase B: stratified (by reading) balance-fill
    for s in SUBS:
        rem = target[s] - len(chosen[s]); cand = [r for r in pool[s] if r['id'] not in ids]
        strata = defaultdict(list)
        for r in cand: strata[reading(r['solver_cot'])].append(r)
        order = sorted(strata, key=lambda k: -len(strata[k]))
        while rem > 0 and any(strata[k] for k in order):
            for k in order:
                if rem <= 0: break
                if strata[k]: chosen[s].append(strata[k].pop()); rem -= 1
    sel = [r for s in SUBS for r in chosen[s]]

    # ---- assemble output: base non-crypt + 800 provider + selected aug ----
    kept = [r for r in base if r['category'] not in CC]                            # non-crypt, untouched
    kept += [{k: r[k] for k in cols} for r in provider]                            # 800 provider as-is
    for r in sel:
        kept.append({'id': r['id'], 'prompt': r['prompt'], 'answer': r['answer'], 'category': r['category'],
                     'solver_cot': r['solver_cot'], 'source': '260608_crypt10k', 'in_7830': 'no',
                     'oversampling': '1', 'token length': r['token length'], 'new label': r['subtype'], 'GT-match': 'True'})
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(kept)

    crypt = [r for r in kept if r['category'] in CC]
    tr = [r for r in crypt if r['oversampling'] in ('1', '2')]
    nc_train = sum(1 for r in kept if r['category'] not in CC and r['oversampling'] in ('1', '2'))
    print(f"wrote {OUT}")
    print(f"  total rows {len(kept)} | cryptarithm {len(crypt)} (provider 800 + selected aug {len(sel)})")
    print(f"  cryptarithm TRAINABLE: {len(tr)}  by subtype: {dict(Counter(r['new label'] for r in tr))}")
    print(f"  cryptarithm share of training: {100*len(tr)/(len(tr)+nc_train):.0f}%  (was 55%)")
    # coverage check
    full = set().union(*[toks(r['prompt']) for r in (prov_train + harv)])
    ds = set().union(*[r['_t'] if '_t' in r else toks(r['prompt']) for r in (prov_train + sel)])
    a2 = {t for t in ACH if len(t) == 2}; a3 = {t for t in ACH if len(t) == 3}
    print(f"  coverage vs full 10k: 2-sym {len(ds&a2)}/{len(full&a2)} kept | 3-sym {len(ds&a3)}/{len(full&a3)} kept | lost pairs {len((full&a2)-ds)} 3-sym {len((full&a3)-ds)}")
    diff = sum(1 for r in base if r['category'] not in CC and r != next((x for x in kept if x['id'] == r['id'] and x['category'] == r['category']), None))
    print(f"  numeric-eq aug preserved: {sum(1 for r in kept if r['source']=='260607_NE_aug')}")
    print(f"  max token length among 1x/2x rows: {max(int(r['token length']) for r in kept if r['oversampling'] in ('1','2'))}")

if __name__ == '__main__':
    main()
