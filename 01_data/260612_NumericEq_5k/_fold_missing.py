#!/usr/bin/env python3
"""Fold the missing-case puzzles (single-op ambiguous + unseen_leading_zero) into the deliverable:
swap N ambiguous-aug rows -> N single-op ambiguous, and M unseen-aug rows -> M unseen_leading_zero,
keeping each bucket's total (so eq stays 5000 @ 55/10/10/10/15). Deterministic (seed 20260613)."""
import csv, os, random, re
from tokenizers import Tokenizer
csv.field_size_limit(10 ** 7)
ROOT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron'
HERE = os.path.dirname(os.path.abspath(__file__))
FULL = f'{HERE}/260612_TrainData/260612_NumericEq5k_FULL.csv'
tok = Tokenizer.from_file(f'{ROOT}/02_train/260512_huikang_085/repo/tokenizer.json')
N_AMB, N_LZ = 30, 15                                   # ~6% nops=1 ambiguous, ~3% unseen_leadzero
rng = random.Random(20260613)

rows = list(csv.DictReader(open(FULL, newline=''))); cols = list(rows[0].keys())
miss = list(csv.DictReader(open(f'{HERE}/_manifest_missing.csv', newline='')))
def newrow(m):
    pid = m['id']; cot = open(f'{HERE}/{pid}/track/tree_cot.txt').read()
    typ = 'unseen' if m['case'] == 'unseen_leading_zero' else 'ambiguous'
    return {'id': pid, 'prompt': open(f'{HERE}/{pid}/question.txt').read(), 'answer': open(f'{HERE}/{pid}/answer.txt').read(),
            'category': 'equation_numeric_guess' if typ == 'unseen' else 'equation_numeric_deduce',
            'solver_cot': cot, 'source': '260612_NE5k', 'in_7830': 'no', 'oversampling': '1',
            'token length': str(len(tok.encode(cot, add_special_tokens=False).ids)),
            'new label': m['label'], 'GT-match': 'True'}
amb_new = [newrow(m) for m in miss if m['case'] == 'single_op_ambiguous'][:N_AMB]
lz_all = [newrow(m) for m in miss if m['case'] == 'unseen_leading_zero']
has_arith = lambda r: any(s in r['prompt'].split('examples:')[-1] for s in '+-*')
lz_all.sort(key=lambda r: not has_arith(r))            # arith-symbol unseen_leadzero first (cover the arith=True combo)
lz_new = lz_all[:N_LZ]

# swappable = TRAINED aug rows of each bucket (exclude originals)
def is_aug(r): return r['source'] in ('260607_NE_aug', '260612_NE5k') and int(r['oversampling']) > 0
amb_aug = [r for r in rows if is_aug(r) and 'ambiguous' in r['new label']]
uns_aug = [r for r in rows if is_aug(r) and ('unseen' in r['new label']) and 'leading' not in r['new label']]
rng.shuffle(amb_aug); rng.shuffle(uns_aug)
drop = set(id(r) for r in amb_aug[:N_AMB] + uns_aug[:N_LZ])
kept = [r for r in rows if id(r) not in drop]
out = kept + amb_new + lz_new
with open(FULL, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(out)
eq = [r for r in out if r['category'].startswith('equation_numeric')]
import collections
def bucket(nl):
    for k in ('ambiguous','unseen','exotic','concat'):
        if k in nl: return k
    return 'deducible'
eff = collections.Counter()
for r in eq:
    if int(r['oversampling']) > 0: eff[bucket(r['new label'])] += int(r['oversampling'])
print(f"folded in: {len(amb_new)} single-op ambiguous + {len(lz_new)} unseen_leading_zero; dropped {len(drop)} aug rows")
print(f"deliverable rows {len(out)} | eq eff {sum(eff.values())} | buckets {dict(eff)}")
