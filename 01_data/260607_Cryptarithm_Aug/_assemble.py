#!/usr/bin/env python3
"""Assemble the final BALANCED 1000-row trainable set: keep all 307 original trainable, then select aug
trainable to hit the target per category (arithmetic 300, little_endian 300, each concat type 100 — incl the
originals). Seeded subsample. Writes _final_selection.txt (chosen aug ids) and prints the final distribution
+ per-reasoning-path feature table (§4b proof)."""
import os, sys, re, csv, json, math, glob, random, statistics
from collections import defaultdict
csv.field_size_limit(10 ** 7)
HERE = os.path.dirname(os.path.abspath(__file__))
CRYPT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260525_Cryptarithm'
sys.path.insert(0, CRYPT)
import cot_generator as cg
BASE = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260525_Cryptarithm/260607_TrainData/260607_Cryptarithm_gtTrue.csv'

TARGET = {'arithmetic': 300, 'little_endian': 300, 'pure_concat': 100,
          'mixed_concat': 100, 'mixed_concat_little_endian': 100, 'query_unseen_concat': 100}
ORDER = list(TARGET)
PATH = {'arithmetic': 'deduce', 'little_endian': 'deduce', 'mixed_concat': 'deduce',
        'mixed_concat_little_endian': 'deduce', 'pure_concat': 'concat-short', 'query_unseen_concat': 'guess'}

def verify(stored, pred):
    stored = (stored or '').strip(); pred = (pred or '').strip()
    if re.fullmatch(r'[01]+', stored): return pred.lower() == stored.lower()
    try: return math.isclose(float(stored), float(pred), rel_tol=1e-2, abs_tol=1e-5)
    except Exception: return pred.lower() == stored.lower()

def feats(prompt, subtype, reading_override=None):
    examples, query = [], None
    for line in prompt.strip().splitlines():
        line = line.strip()
        if line.startswith('Now'): query = line.split(': ', 1)[1].strip()
        elif ' = ' in line and not line.startswith('In'):
            lhs, rhs = line.split(' = ', 1); lhs = lhs.strip(); rhs = rhs.strip()
            if len(lhs) == 5: examples.append((lhs, rhs))
    ops = []
    for lhs, rhs in examples:
        if lhs[2] not in ops: ops.append(lhs[2])
    if query and query[2] not in ops: ops.append(query[2])
    opset = set(ops); digs = set()
    for lhs, rhs in examples:
        for ch in (lhs[0], lhs[1], lhs[3], lhs[4]):
            if ch not in opset: digs.add(ch)
        for ch in rhs:
            if ch not in opset: digs.add(ch)
    if query:
        for ch in (query[0], query[1], query[3], query[4]):
            if ch not in opset: digs.add(ch)
    signed = any(rhs and (rhs[0] in opset or rhs[-1] in opset) for _l, rhs in examples)
    has_concat = False
    for o in ops:
        exs = [type('E', (), {'d1': l[0], 'd2': l[1], 'op': l[2], 'd3': l[3], 'd4': l[4], 'rhs': r})()
               for l, r in examples if l[2] == o]
        if exs and cg.check_concat(o, exs): has_concat = True
    reading = reading_override or ('leftward' if subtype in ('little_endian', 'mixed_concat_little_endian') else 'standard')
    return {'signed': signed, 'n_ops': len(ops), 'n_ex': len(examples),
            'arith_syms': sum(1 for s in ops if s in '+-*'), 'has_concat': has_concat,
            'n_digits': len(digs), 'reading': reading}

def main():
    rng = random.Random(20260608)
    # ---- original trainable (keep ALL) ----
    orig = defaultdict(list)
    with open(BASE, newline='') as fh:
        for r in csv.DictReader(fh):
            if r['category'] in ('cryptarithm_deduce', 'cryptarithm_guess') and r['GT-match'] == 'True' and int(r['token length']) < 7680:
                f = feats(r['prompt'], r['new label'])
                orig[r['new label']].append({'src': 'orig', 'id': r['id'], 'subtype': r['new label'], 'ntok': int(r['token length']), **f})

    # ---- aug trainable (both batches) ----
    blind = {}
    for f in glob.glob(os.path.join(HERE, '_blind_out', 'shard_*.jsonl')) + glob.glob(os.path.join(HERE, '_blind_out2', 'shard_*.jsonl')):
        for line in open(f):
            line = line.strip()
            if line: o = json.loads(line); blind[o['id']] = (o['box'], o['ntok'])
    man = {}
    for mf in [os.path.join(HERE, 'puzzles', '_manifest.csv'), os.path.join(HERE, 'puzzles2', '_manifest.csv')]:
        if os.path.exists(mf):
            for row in csv.DictReader(open(mf)): man[row['id']] = row
    aug = defaultdict(list)
    for pid, row in man.items():
        box, ntok = blind.get(pid, ('MISSING', -1))
        if box in ('MISSING', 'TIMEOUT') or str(box).startswith('ERR'): continue
        if not (verify(row['gold'], box) and 0 <= ntok < 7680): continue
        pdir = 'puzzles2' if os.path.exists(os.path.join(HERE, 'puzzles2', pid)) else 'puzzles'
        f = feats(open(os.path.join(HERE, pdir, pid, 'question.txt')).read(), row['subtype'], row['reading'])
        aug[row['subtype']].append({'src': 'aug', 'id': pid, 'subtype': row['subtype'], 'ntok': ntok, **f})

    # ---- select aug to hit target (stratify by reading for free-reading types) ----
    selected = []; shortfall = {}
    for sub in ORDER:
        need = TARGET[sub] - len(orig[sub])
        pool = aug[sub][:]
        if need <= 0:
            shortfall[sub] = ('orig already >= target', 0); continue
        if len(pool) < need:
            shortfall[sub] = ('SHORT', need - len(pool)); chosen = pool
        else:
            # stratify by reading so the kept set keeps both readings where the type allows it
            byread = defaultdict(list)
            for r in pool: byread[r['reading']].append(r)
            for v in byread.values(): rng.shuffle(v)
            chosen = []
            # proportional allocation across reading buckets
            tot = len(pool)
            for rd, items in byread.items():
                k = round(need * len(items) / tot)
                chosen += items[:k]
            # fix rounding
            rng.shuffle(chosen)
            if len(chosen) > need: chosen = chosen[:need]
            while len(chosen) < need:
                extra = [r for r in pool if r not in chosen]
                if not extra: break
                chosen.append(rng.choice(extra))
        selected += chosen
    open(os.path.join(HERE, '_final_selection.txt'), 'w').write('\n'.join(sorted(r['id'] for r in selected)) + '\n')

    combined = [r for sub in ORDER for r in orig[sub]] + selected

    # ---- report ----
    print("=" * 70); print("FINAL BALANCED TRAINABLE DISTRIBUTION (target 1000, incl 307 orig)"); print("=" * 70)
    print(f"{'subtype':<30}{'orig':>6}{'aug':>6}{'final':>7}{'target':>8}{'avail_aug':>11}")
    selcnt = defaultdict(int)
    for r in selected: selcnt[r['subtype']] += 1
    augcnt = {s: len(aug[s]) for s in ORDER}
    for sub in ORDER:
        o = len(orig[sub])
        print(f"{sub:<30}{o:>6}{selcnt[sub]:>6}{o + selcnt[sub]:>7}{TARGET[sub]:>8}{augcnt.get(sub, 0):>11}")
    print('-' * 67)
    print(f"{'TOTAL':<30}{sum(len(orig[s]) for s in ORDER):>6}{len(selected):>6}{len(combined):>7}{sum(TARGET.values()):>8}")
    if shortfall: print("\nSHORTFALLS:", {k: v for k, v in shortfall.items() if v[0] == 'SHORT'})

    # per-path feature table on the FINAL set
    print("\n" + "=" * 78); print("§4b — per-reasoning-path feature rates (FINAL combined set)"); print("=" * 78)
    print(f"{'path':<14}{'n':>5}{'signed%':>8}{'left%':>7}{'concat%':>8}{'arithSym%':>10}{'avgOps':>7}{'avgEx':>6}{'avgDig':>7}")
    for pth in ['deduce', 'concat-short', 'guess']:
        sr = [r for r in combined if PATH[r['subtype']] == pth]
        if not sr: continue
        n = len(sr)
        print(f"{pth:<14}{n:>5}{100*sum(r['signed'] for r in sr)/n:>7.0f}%{100*sum(r['reading']=='leftward' for r in sr)/n:>6.0f}%"
              f"{100*sum(r['has_concat'] for r in sr)/n:>7.0f}%{100*sum(r['arith_syms']>0 for r in sr)/n:>9.0f}%"
              f"{statistics.mean(r['n_ops'] for r in sr):>7.2f}{statistics.mean(r['n_ex'] for r in sr):>6.2f}{statistics.mean(r['n_digits'] for r in sr):>7.2f}")
    print(f"\nselected aug ids -> _final_selection.txt ({len(selected)})")

if __name__ == '__main__':
    main()
