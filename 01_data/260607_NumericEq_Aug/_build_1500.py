#!/usr/bin/env python3
"""Build the 1500-effective equation_numeric training set, embedded in the FULL multi-category training CSV.

Target eq distribution (effective = sum of oversampling): 1500 @ 55/10/10/10/15
  deducible 825 | exotic 150 | unseen 150 | ambiguous 150 | concat 225

Composition:
  - ALL 621 original eq GT-True rows are USED at 1x, EXCEPT the 36 ambiguous rows, which are OVERSAMPLED to
    150 effective (ambiguous can't be synthesized, so the only way to hit 10% is to weight the originals).
  - 111 original eq GT-False rows kept at 0x (carried, never trained).
  - Fresh decorrelated aug (GT-match=True) fills the rest, all at 1x, SUBSAMPLED to exact per-category targets,
    STRATIFIED by reading so each bucket stays ~50/50 (no reading<->type correlation introduced by the subsample).
  - Non-eq rows are passed through UNCHANGED (their own categories/oversampling untouched).

Output: 260607_NE_aug_TrainData/260607_NumericEq_1500_FULL.csv (full set) + 260607_NE_aug_1500.csv (aug-only record).
"""
import csv, os, re, math, collections, random
from tokenizers import Tokenizer
csv.field_size_limit(10 ** 7)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron'
BASE = f'{ROOT}/01_data/260606_Numeric_Equation/260606_TrainData/260606_NumericEq_gtTrue.csv'   # full 7077-row multi-category CSV
TOKENIZER = f'{ROOT}/02_train/260512_huikang_085/repo/tokenizer.json'
SOURCE = '260607_NE_aug'
OUTDIR = os.path.join(HERE, '260607_NE_aug_TrainData'); os.makedirs(OUTDIR, exist_ok=True)
OUT_FULL = os.path.join(OUTDIR, '260607_NumericEq_1500_FULL.csv')
OUT_AUG = os.path.join(OUTDIR, '260607_NE_aug_1500.csv')
COLS = ['id', 'prompt', 'answer', 'category', 'solver_cot', 'source', 'in_7830',
        'oversampling', 'token length', 'new label', 'GT-match']

# new-label -> 5-bucket map (originals)
BUCKET = {'arithmetic_left_to_right': 'deducible', 'arithmetic_right_to_left': 'deducible', 'no_rule_fits': 'deducible',
          'ambiguous_arithmetic': 'ambiguous', 'unseen_operator': 'unseen', 'unseen_leading_zero': 'unseen',
          'exotic_operation': 'exotic', 'concatenation': 'concat'}
AMB_OS = 2                                                    # ambiguous oversampled flat 2x (can't synthesize; light weight avoids memorizing 36 specific puzzles)
# Expanded build (2nd 1000-puzzle batch): synthesizable 4 categories kept at EXACTLY 55:10:10:15, grown to the
# limit of the deducible aug pool (binding). Combined: deducible 1320 / exotic 240 / unseen 240 / concat 360 (+ amb 72 = ~2232 eff).
AUG_TARGET = {'deducible': 883, 'exotic': 222, 'unseen': 182, 'concat': 288}   # = combined target - 1x originals (437/18/58/72)


def main():
    rng = random.Random(20260607)
    tok = Tokenizer.from_file(TOKENIZER)
    base = list(csv.DictReader(open(BASE, newline='')))

    eq = [r for r in base if r['category'].startswith('equation_numeric')]
    non_eq = [r for r in base if not r['category'].startswith('equation_numeric')]
    eq_true = [r for r in eq if r['GT-match'] == 'True']
    eq_false = [r for r in eq if r['GT-match'] != 'True']

    # ---- originals: 1x everywhere, ambiguous oversampled to 150 ----
    amb = [r for r in eq_true if r['new label'] == 'ambiguous_arithmetic']
    non_amb = [r for r in eq_true if r['new label'] != 'ambiguous_arithmetic']
    for r in non_amb: r['oversampling'] = '1'
    for r in amb: r['oversampling'] = str(AMB_OS)             # flat 2x
    for r in eq_false: r['oversampling'] = '0'

    orig_eff = collections.Counter()
    for r in eq_true: orig_eff[BUCKET[r['new label']]] += int(r['oversampling'])

    # ---- aug: subsample GT-match=True per category to AUG_TARGET, stratified by reading ----
    man = {r['id']: r for r in csv.DictReader(open(os.path.join(HERE, '_manifest.csv'), newline=''))}
    keep_ids = [l.strip() for l in open(os.path.join(HERE, '_gtmatch_true_ids.txt')) if l.strip()]
    by_cat = collections.defaultdict(lambda: collections.defaultdict(list))   # cat -> reading -> [ids]
    for pid in keep_ids:
        m = man[pid]; by_cat[m['type']][m['reading']].append(pid)

    chosen = []
    for cat, want in AUG_TARGET.items():
        left = by_cat[cat]['leftward'][:]; right = by_cat[cat]['rightward'][:]
        rng.shuffle(left); rng.shuffle(right)
        nl = min(len(left), want // 2); nr = min(len(right), want - nl)
        nl = want - nr                                              # backfill if one side short
        pick = left[:nl] + right[:nr]
        if len(pick) < want:                                       # last-resort fill from leftovers
            extra = [p for p in (left[nl:] + right[nr:])]; rng.shuffle(extra); pick += extra[:want - len(pick)]
        rng.shuffle(pick); chosen += [(cat, p) for p in pick]
        assert len(pick) == want, f"{cat}: got {len(pick)} want {want}"

    aug_rows = []
    for cat, pid in chosen:
        d = os.path.join(HERE, pid)
        cot = open(d + '/track/tree_cot.txt').read()
        aug_rows.append({
            'id': pid, 'prompt': open(d + '/question.txt').read(), 'answer': open(d + '/answer.txt').read(),
            'category': 'equation_numeric_guess' if cat == 'unseen' else 'equation_numeric_deduce',
            'solver_cot': cot, 'source': SOURCE, 'in_7830': 'no', 'oversampling': '1',
            'token length': str(len(tok.encode(cot, add_special_tokens=False).ids)),
            'new label': open(d + '/label.txt').read().strip(), 'GT-match': 'True'})

    # ---- assemble & write ----
    full = non_eq + eq_true + eq_false + aug_rows
    with open(OUT_FULL, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader()
        for r in full: w.writerow({k: r.get(k, '') for k in COLS})
    with open(OUT_AUG, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader(); w.writerows(aug_rows)

    # ---- report ----
    aug_eff = collections.Counter(c for c, _ in chosen)
    print(f"FULL rows: {len(full)}  (non-eq {len(non_eq)} | eq-true {len(eq_true)} | eq-false {len(eq_false)} | aug {len(aug_rows)})")
    print(f"AUG-only rows: {len(aug_rows)}\n")
    tots = {b: orig_eff[b] + aug_eff.get(b, 0) for b in ['deducible', 'exotic', 'unseen', 'ambiguous', 'concat']}
    grand = sum(tots.values())
    print(f"{'bucket':11s} {'orig(eff)':>10s} {'aug(1x)':>8s} {'TOTAL':>7s} {'%':>5s}")
    for b in ['deducible', 'exotic', 'unseen', 'ambiguous', 'concat']:
        print(f"{b:11s} {orig_eff[b]:>10d} {aug_eff.get(b,0):>8d} {tots[b]:>7d} {tots[b]/grand:>4.0%}")
    print(f"{'EQ TOTAL':11s} {'':>10s} {'':>8s} {grand:>7d} {'100%':>5s}")
    non_eq_eff = sum(int(r['oversampling']) for r in non_eq)
    print(f"\nnon-eq effective (unchanged): {non_eq_eff}  | GRAND effective: {grand + non_eq_eff}")
    print(f"\nwrote:\n  {OUT_FULL}\n  {OUT_AUG}")

if __name__ == '__main__':
    main()
