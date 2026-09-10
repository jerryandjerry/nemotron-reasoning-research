#!/usr/bin/env python3
"""Build the 5000-effective equation_numeric training set, embedded in the latest FULL multi-category
training CSV (base: 260609_CryptAug_eqHardened_FULL.csv — its non-eq rows pass through UNCHANGED).

⚠️ BUILD ORDER (the current deliverable is this script THEN the scenario-coverage fold — running this
   script alone reproduces the PRE-fold deliverable):
     1. python3 _build_5000.py        # this file -> 260612_NumericEq5k_FULL.csv (5000 eff)
     2. python3 _gen_missing.py 60 25  # single-op ambiguous + unseen_leading_zero (GT-true)
     3. python3 _fold_missing.py       # swaps 30+15 of those into the buckets (keeps 5000 + proportions)
   The folded NE5k_003xxx ids are in _gtmatch_true_ids.txt + _manifest.csv. (2026-06-13)

Target eq distribution (effective = sum of oversampling): 5000 @ the full nominal 55/10/10/10/15
  deducible 2750 | exotic 500 | unseen 500 | ambiguous 500 | concat 750

Composition:
  - ALL 621 original eq GT-True rows at 1x. Ambiguous originals DROP from 2x to 1x — the 2x was a
    stopgap for "can't synthesize ambiguous"; this round synthesizes it (260612 pool, 560 GT-true).
  - 111 original eq GT-False rows kept at 0x (carried, never trained).
  - Aug fills the rest at 1x from the COMBINED pools (260607: 1875 GT-true; 260612: 2887 GT-true),
    subsampled to exact per-bucket targets, STRATIFIED by reading (~50/50 within every bucket).
Output: 260612_TrainData/260612_NumericEq5k_FULL.csv (+ aug-only record).
"""
import csv, os, collections, random
from tokenizers import Tokenizer
csv.field_size_limit(10 ** 7)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron'
BASE = f'{ROOT}/01_data/260607_Cryptarithm_Aug/260607_CryptAug_TrainData/260609_CryptAug_eqHardened_FULL.csv'
POOLS = [(f'{ROOT}/01_data/260607_NumericEq_Aug', '260607_NE_aug'),
         (HERE, '260612_NE5k')]
TOKENIZER = f'{ROOT}/02_train/260512_huikang_085/repo/tokenizer.json'
OUTDIR = os.path.join(HERE, '260612_TrainData'); os.makedirs(OUTDIR, exist_ok=True)
OUT_FULL = os.path.join(OUTDIR, '260612_NumericEq5k_FULL.csv')
OUT_AUG = os.path.join(OUTDIR, '260612_NE5k_aug.csv')
COLS = ['id', 'prompt', 'answer', 'category', 'solver_cot', 'source', 'in_7830',
        'oversampling', 'token length', 'new label', 'GT-match']

BUCKET = {'arithmetic_left_to_right': 'deducible', 'arithmetic_right_to_left': 'deducible', 'no_rule_fits': 'deducible',
          'ambiguous_arithmetic': 'ambiguous', 'unseen_operator': 'unseen', 'unseen_leading_zero': 'unseen',
          'exotic_operation': 'exotic', 'concatenation': 'concat'}
# combined target 5000 = orig@1x (437/18/58/36/72) + AUG_TARGET
AUG_TARGET = {'deducible': 2313, 'exotic': 482, 'unseen': 442, 'ambiguous': 464, 'concat': 678}


def main():
    rng = random.Random(20260612)
    tok = Tokenizer.from_file(TOKENIZER)
    base = list(csv.DictReader(open(BASE, newline='')))
    eq = [r for r in base if r['category'].startswith('equation_numeric')]
    non_eq = [r for r in base if not r['category'].startswith('equation_numeric')]
    orig = [r for r in eq if not r['id'].startswith('NE_aug')]          # the 732; OLD aug rows are rebuilt from pools
    eq_true = [r for r in orig if r['GT-match'] == 'True']
    eq_false = [r for r in orig if r['GT-match'] != 'True']
    for r in eq_true: r['oversampling'] = '1'                           # ambiguous 2x stopgap -> 1x (now synthesized)
    for r in eq_false: r['oversampling'] = '0'
    orig_eff = collections.Counter()
    for r in eq_true: orig_eff[BUCKET[r['new label']]] += 1
    amb_cat = collections.Counter(r['category'] for r in eq_true if r['new label'] == 'ambiguous_arithmetic')

    # ---- combined GT-true pools, bucketed by (type, reading) ----
    by_cat = collections.defaultdict(lambda: collections.defaultdict(list))   # type -> reading -> [(dir, pid, source)]
    for pool_dir, source in POOLS:
        man = {r['id']: r for r in csv.DictReader(open(os.path.join(pool_dir, '_manifest.csv'), newline=''))}
        for pid in (l.strip() for l in open(os.path.join(pool_dir, '_gtmatch_true_ids.txt'))):
            if not pid: continue
            m = man[pid]
            by_cat[m['type']][m['reading']].append((pool_dir, pid, source))

    chosen = []
    for cat, want in AUG_TARGET.items():
        left = by_cat[cat]['leftward'][:]; right = by_cat[cat]['rightward'][:]
        rng.shuffle(left); rng.shuffle(right)
        nl = min(len(left), want // 2); nr = min(len(right), want - nl)
        nl = want - nr
        pick = left[:nl] + right[:nr]
        if len(pick) < want:
            extra = left[nl:] + right[nr:]; rng.shuffle(extra); pick += extra[:want - len(pick)]
        assert len(pick) == want, f"{cat}: got {len(pick)} want {want} (pool L{len(left)}/R{len(right)})"
        rng.shuffle(pick); chosen += [(cat, p) for p in pick]

    aug_rows = []
    for cat, (pool_dir, pid, source) in chosen:
        d = os.path.join(pool_dir, pid)
        cot = open(d + '/track/tree_cot.txt').read()
        aug_rows.append({
            'id': pid, 'prompt': open(d + '/question.txt').read(), 'answer': open(d + '/answer.txt').read(),
            'category': 'equation_numeric_guess' if cat == 'unseen' else 'equation_numeric_deduce',
            'solver_cot': cot, 'source': source, 'in_7830': 'no', 'oversampling': '1',
            'token length': str(len(tok.encode(cot, add_special_tokens=False).ids)),
            'new label': open(d + '/label.txt').read().strip(), 'GT-match': 'True'})

    full = non_eq + eq_true + eq_false + aug_rows
    with open(OUT_FULL, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader()
        for r in full: w.writerow({k: r.get(k, '') for k in COLS})
    with open(OUT_AUG, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader(); w.writerows(aug_rows)

    # ---- report ----
    aug_eff = collections.Counter(c for c, _ in chosen)
    aug_src = collections.Counter(s for _, (_, _, s) in chosen)
    print(f"orig ambiguous category convention: {dict(amb_cat)}")
    print(f"FULL rows: {len(full)}  (non-eq {len(non_eq)} | eq-true {len(eq_true)} | eq-false {len(eq_false)} | aug {len(aug_rows)})")
    print(f"aug by source: {dict(aug_src)}\n")
    tots = {b: orig_eff[b] + aug_eff.get(b, 0) for b in ['deducible', 'exotic', 'unseen', 'ambiguous', 'concat']}
    grand = sum(tots.values())
    print(f"{'bucket':11s} {'orig(1x)':>9s} {'aug(1x)':>8s} {'TOTAL':>7s} {'%':>5s}")
    for b in ['deducible', 'exotic', 'unseen', 'ambiguous', 'concat']:
        print(f"{b:11s} {orig_eff[b]:>9d} {aug_eff.get(b,0):>8d} {tots[b]:>7d} {tots[b]/grand:>4.0%}")
    print(f"{'EQ TOTAL':11s} {'':>9s} {'':>8s} {grand:>7d}")
    non_eq_eff = sum(int(r['oversampling']) for r in non_eq)
    print(f"\nnon-eq effective (unchanged): {non_eq_eff}  | GRAND effective: {grand + non_eq_eff}")
    print(f"\nwrote:\n  {OUT_FULL}\n  {OUT_AUG}")

if __name__ == '__main__':
    main()
