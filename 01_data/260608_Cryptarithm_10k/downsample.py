#!/usr/bin/env python3
"""Coverage-aware downsample of the 10k aug: pick a SMALLER subset that STILL covers every achievable
BPE token (pairs + 3-sym), every subtype, both readings, and every operator family — by rare-first
set-cover, then stratified balance-fill. Keeps the 800 provider rows (their coverage seeds the cover).
Verifies the downsampled (provider + selected aug) matches the FULL 10k's coverage."""
import os, sys, csv, re, json, random
from collections import defaultdict, Counter
csv.field_size_limit(10 ** 8)
HERE = os.path.dirname(os.path.abspath(__file__))
CRYPT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260525_Cryptarithm'
sys.path.insert(0, CRYPT)
from tokenizers import Tokenizer
TOK = Tokenizer.from_file('/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/02_train/260512_huikang_085/repo/tokenizer.json')
POOL = set('''!"#$%&'()/:<>?@[\\]^`{|}+-*'''); OPS = set('+-*')
TABLE = json.load(open(os.path.join(CRYPT, 'full_merge_table.json')))['map']
ALL = set(TABLE); ACH = {t for t in ALL if sum(c in OPS for c in t) <= 1}
FULL = os.path.join(HERE, '260609_Crypt10k_eqHardened_FULL.csv')
SUBS = ['arithmetic', 'little_endian', 'pure_concat', 'mixed_concat', 'mixed_concat_little_endian', 'query_unseen_concat']

def toks(prompt):
    out = set()
    for line in prompt.strip().splitlines():
        if line.startswith('In'): continue
        for run in line.replace('Now, determine the result for:', '').replace(' = ', ' ').split():
            for a, b in TOK.encode(run, add_special_tokens=False).offsets:
                seg = run[a:b]
                if len(seg) >= 2 and all(c in POOL for c in seg): out.add(seg)
    return out & ALL

def reading(cot): return 'leftward' if 'reading order = leftward' in cot else 'standard'
def fams(cot):
    f = set()
    for m in re.finditer(r'lock [fgh] = (\S+)', cot):
        s = m.group(1)
        if '∥' in s: f.add('concat')
        elif '×' in s: f.add('mul')
        elif '+' in s: f.add('add')
        elif '-' in s or '|' in s: f.add('sub')
    return f

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--combined', type=int, default=3000); a = ap.parse_args()
    rng = random.Random(20260609)
    rows = list(csv.DictReader(open(FULL)))
    prov = [r for r in rows if r['source'] == '260601_new_solver' and r['oversampling'] == '1']
    aug = [r for r in rows if r['source'] == '260608_crypt10k']
    for r in prov + aug: r['_t'] = toks(r['prompt'])

    # balanced combined target 30/30/10/10/10/10
    ratio = {'arithmetic': .30, 'little_endian': .30, 'pure_concat': .10, 'mixed_concat': .10,
             'mixed_concat_little_endian': .10, 'query_unseen_concat': .10}
    prov_n = Counter(r['new label'] for r in prov)
    aug_target = {s: max(0, round(a.combined * ratio[s]) - prov_n.get(s, 0)) for s in SUBS}

    pool = {s: [r for r in aug if r['new label'] == s] for s in SUBS}
    for s in SUBS: rng.shuffle(pool[s])
    chosen, chosen_ids = {s: [] for s in SUBS}, set()
    covered = set().union(*[r['_t'] for r in prov]) if prov else set()

    # Phase A: rare-token-first set cover
    tok2p = defaultdict(list)
    for s in SUBS:
        for r in pool[s]:
            for t in (r['_t'] & ACH): tok2p[t].append((s, r))
    for t in sorted(ACH, key=lambda t: len(tok2p[t])):
        if t in covered or not tok2p[t]: continue
        cand = [(s, r) for (s, r) in tok2p[t] if r['id'] not in chosen_ids and len(chosen[s]) < aug_target[s]]
        if not cand: continue
        s, r = max(cand, key=lambda sr: len((sr[1]['_t'] & ACH) - covered))
        chosen[s].append(r); chosen_ids.add(r['id']); covered |= r['_t']
    # Phase B: stratified balance-fill to target (stratify by reading)
    for s in SUBS:
        rem = aug_target[s] - len(chosen[s])
        cand = [r for r in pool[s] if r['id'] not in chosen_ids]
        strata = defaultdict(list)
        for r in cand: strata[reading(r['solver_cot'])].append(r)
        order = sorted(strata, key=lambda k: -len(strata[k]))
        while rem > 0 and any(strata[k] for k in order):
            for k in order:
                if rem <= 0: break
                if strata[k]: chosen[s].append(strata[k].pop()); rem -= 1
    sel = [r for s in SUBS for r in chosen[s]]
    combined = prov + sel

    def cov(rs):
        tk = set().union(*[r['_t'] for r in rs]); rd = set(); fm = set(); sub = Counter()
        for r in rs: rd.add(reading(r['solver_cot'])); fm |= fams(r['solver_cot']); sub[r['new label']] += 1
        return tk, rd, fm, sub
    ftk, frd, ffm, _ = cov(prov + aug)       # FULL coverage (provider + all 10k)
    dtk, drd, dfm, dsub = cov(combined)      # downsampled coverage
    ach2 = {t for t in ACH if len(t) == 2}; ach3 = {t for t in ACH if len(t) == 3}
    print(f"DOWNSAMPLE: provider {len(prov)} + selected aug {len(sel)} = {len(combined)} cryptarithm (target ~{a.combined})")
    print(f"            (was provider 307 + 10000 aug = 10307)")
    print(f"  subtype counts: {dict(dsub)}")
    print()
    print(f"COVERAGE — downsampled vs FULL 10k:")
    print(f"  2-sym pairs:   {len(dtk&ach2)}/{len(ach2)} achievable  | full had {len(ftk&ach2)}  -> kept {len(dtk&ftk&ach2)}/{len(ftk&ach2)}")
    print(f"  3-sym tokens:  {len(dtk&ach3)}/{len(ach3)} achievable  | full had {len(ftk&ach3)}  -> kept {len(dtk&ftk&ach3)}/{len(ftk&ach3)}")
    print(f"  readings:      {sorted(drd)}  (full {sorted(frd)})")
    print(f"  operator fams: {sorted(dfm)}  (full {sorted(ffm)})")
    lostpairs = (ftk & ach2) - dtk; lost3 = (ftk & ach3) - dtk
    print(f"  pairs in full but LOST in downsample: {len(lostpairs)} {sorted(lostpairs)[:20]}")
    print(f"  3-sym in full but LOST in downsample:  {len(lost3)}")

if __name__ == '__main__':
    main()
