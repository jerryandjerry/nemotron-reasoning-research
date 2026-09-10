#!/usr/bin/env python3
"""Show the COMBINED trainable distribution: original (corpus) trainable cryptarithm + the 542 aug trainable.
Per-category counts AND per-category incidental-feature rates (§4b), computed from each puzzle's prompt so
original and aug are measured identically. Lets us see (a) the complement, (b) whether any feature predicts
the category."""
import os, sys, re, csv, statistics
from collections import defaultdict
csv.field_size_limit(10 ** 7)
HERE = os.path.dirname(os.path.abspath(__file__))
CRYPT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260525_Cryptarithm'
sys.path.insert(0, CRYPT)
import cot_generator as cg
BASE = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260525_Cryptarithm/260607_TrainData/260607_Cryptarithm_gtTrue.csv'

def feats_from_prompt(prompt, subtype, reading_override=None):
    """Derive incidental features from the puzzle text (same method for original + aug)."""
    examples, query = [], None
    for line in prompt.strip().splitlines():
        line = line.strip()
        if line.startswith('Now'):
            query = line.split(': ', 1)[1].strip()
        elif ' = ' in line and not line.startswith('In'):
            lhs, rhs = line.split(' = ', 1); lhs = lhs.strip(); rhs = rhs.strip()
            if len(lhs) == 5: examples.append((lhs, rhs))
    op_syms = []
    for lhs, rhs in examples:
        if lhs[2] not in op_syms: op_syms.append(lhs[2])
    if query and query[2] not in op_syms: op_syms.append(query[2])
    op_set = set(op_syms)
    digs = set()
    for lhs, rhs in examples:
        for ch in (lhs[0], lhs[1], lhs[3], lhs[4]):
            if ch not in op_set: digs.add(ch)
        for ch in rhs:
            if ch not in op_set: digs.add(ch)
    if query:
        for ch in (query[0], query[1], query[3], query[4]):
            if ch not in op_set: digs.add(ch)
    signed = any(rhs and (rhs[0] in op_set or rhs[-1] in op_set) for _l, rhs in examples)
    # concat present among example ops?
    has_concat = False
    for o in op_syms:
        exs = [type('E', (), {'d1': l[0], 'd2': l[1], 'op': l[2], 'd3': l[3], 'd4': l[4], 'rhs': r})()
               for l, r in examples if l[2] == o]
        if exs and cg.check_concat(o, exs): has_concat = True
    reading = reading_override or ('leftward' if subtype in ('little_endian', 'mixed_concat_little_endian') else 'standard')
    return {'signed': signed, 'n_ops': len(op_syms), 'n_ex': len(examples),
            'arith_syms': sum(1 for s in op_syms if s in '+-*'), 'has_concat': has_concat,
            'n_digits': len(digs), 'reading': reading}

# ---- gather original trainable cryptarithm ----
rows = []
with open(BASE, newline='') as fh:
    for r in csv.DictReader(fh):
        if r['category'] in ('cryptarithm_deduce', 'cryptarithm_guess') and r['GT-match'] == 'True' and int(r['token length']) < 7680:
            f = feats_from_prompt(r['prompt'], r['new label'])
            rows.append({'src': 'orig', 'subtype': r['new label'], 'ntok': int(r['token length']), **f})
# ---- gather aug trainable ----
aug_ids = set(open(os.path.join(HERE, '_trainable_ids.txt')).read().split())
man = {row['id']: row for row in csv.DictReader(open(os.path.join(HERE, 'puzzles', '_manifest.csv')))}
gt = {row['id']: row for row in csv.DictReader(open(os.path.join(HERE, '_gtmatch.csv')))}
for pid in aug_ids:
    m = man[pid]; g = gt[pid]
    f = feats_from_prompt(open(os.path.join(HERE, 'puzzles', pid, 'question.txt')).read(), m['subtype'], m['reading'])
    rows.append({'src': 'aug', 'subtype': m['subtype'], 'ntok': int(g['ntok']), **f})

ORDER = ['arithmetic', 'little_endian', 'pure_concat', 'mixed_concat', 'mixed_concat_little_endian', 'query_unseen_concat']
PATH = {'arithmetic': 'deduce', 'little_endian': 'deduce', 'mixed_concat': 'deduce',
        'mixed_concat_little_endian': 'deduce', 'pure_concat': 'concat-short', 'query_unseen_concat': 'guess'}

# ---- 1. counts ----
print("=" * 78)
print("COMBINED TRAINABLE DISTRIBUTION (original corpus + aug)")
print("=" * 78)
print(f"{'subtype':<30}{'orig':>6}{'aug':>6}{'combined':>10}{'medTok':>8}")
co = defaultdict(int); ca = defaultdict(int); tk = defaultdict(list)
for r in rows:
    (ca if r['src'] == 'aug' else co)[r['subtype']] += 1
    tk[r['subtype']].append(r['ntok'])
for sub in ORDER:
    med = int(statistics.median(tk[sub])) if tk[sub] else 0
    print(f"{sub:<30}{co[sub]:>6}{ca[sub]:>6}{co[sub] + ca[sub]:>10}{med:>8}")
print('-' * 60)
print(f"{'TOTAL':<30}{sum(co.values()):>6}{sum(ca.values()):>6}{sum(co.values()) + sum(ca.values()):>10}")

# ---- 2. per-subtype feature rates on COMBINED ----
def block(title, keyfn, keys):
    print("\n" + "=" * 78); print(title); print("=" * 78)
    print(f"{'group':<30}{'n':>5}{'signed%':>8}{'left%':>7}{'concat%':>8}{'arithSym%':>10}{'avgOps':>7}{'avgEx':>6}{'avgDig':>7}")
    for k in keys:
        sr = [r for r in rows if keyfn(r) == k]
        if not sr: continue
        n = len(sr)
        print(f"{k:<30}{n:>5}{100*sum(r['signed'] for r in sr)/n:>7.0f}%{100*sum(r['reading']=='leftward' for r in sr)/n:>6.0f}%"
              f"{100*sum(r['has_concat'] for r in sr)/n:>7.0f}%{100*sum(r['arith_syms']>0 for r in sr)/n:>9.0f}%"
              f"{statistics.mean(r['n_ops'] for r in sr):>7.2f}{statistics.mean(r['n_ex'] for r in sr):>6.2f}{statistics.mean(r['n_digits'] for r in sr):>7.2f}")

block("PER-SUBTYPE feature rates (COMBINED) — note reading is DEFINITIONAL for the *_little_endian labels",
      lambda r: r['subtype'], ORDER)
block("PER-REASONING-PATH feature rates (COMBINED) — the axis the model must NOT shortcut",
      lambda r: PATH[r['subtype']], ['deduce', 'concat-short', 'guess'])
