#!/usr/bin/env python3
"""Measure merged-token + symbol coverage and per-category availability of the TRAINABLE pool
(original 307 + all aug trainable, both batches). Tells us the ceiling for coverage-aware selection."""
import os, sys, re, csv, json, math, glob
from collections import Counter, defaultdict
csv.field_size_limit(10 ** 7)
HERE = os.path.dirname(os.path.abspath(__file__))
CRYPT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260525_Cryptarithm'
sys.path.insert(0, CRYPT)
from tokenizers import Tokenizer
TOK = Tokenizer.from_file('/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/02_train/260512_huikang_085/repo/tokenizer.json')
POOL = set('''!"#$%&'()/:<>?@[\\]^`{|}+-*''')
OPS = set('+-*')
TABLE = json.load(open(os.path.join(CRYPT, 'full_merge_table.json')))['map']
ALL = set(TABLE); ACH = set(t for t in ALL if sum(c in OPS for c in t) <= 1)
BASE = os.path.join(CRYPT, '260607_TrainData/260607_Cryptarithm_gtTrue.csv')

def verify(s, p):
    s = (s or '').strip(); p = (p or '').strip()
    if re.fullmatch(r'[01]+', s): return p.lower() == s.lower()
    try: return math.isclose(float(s), float(p), rel_tol=1e-2, abs_tol=1e-5)
    except Exception: return p.lower() == s.lower()

def fused(text):
    out = set()
    for line in text.strip().splitlines():
        for run in line.replace(' = ', ' ').split():
            for a, b in TOK.encode(run, add_special_tokens=False).offsets:
                seg = run[a:b]
                if len(seg) >= 2 and all(c in POOL for c in seg): out.add(seg)
    return out & ALL

def report(name, texts):
    cov = set(); sym = Counter()
    for t in texts:
        cov |= fused(t)
        for ch in t:
            if ch in POOL: sym[ch] += 1
    operand_freq = {s: sym[s] for s in sym if s not in OPS}
    vals = sorted(operand_freq.values())
    print(f"{name}: {len(texts)} puzzles | token cov {len(cov & ACH)}/{len(ACH)} achievable ({100*len(cov&ACH)/len(ACH):.1f}%) | "
          f"symbols {len(sym)}/26 | operand-freq min/med/max {vals[0]}/{vals[len(vals)//2]}/{vals[-1]}")
    return cov

# ---- original trainable ----
orig_text = defaultdict(list)
with open(BASE, newline='') as fh:
    for r in csv.DictReader(fh):
        if r['category'] in ('cryptarithm_deduce', 'cryptarithm_guess') and r['GT-match'] == 'True' and int(r['token length']) < 7680:
            orig_text[r['new label']].append(r['prompt'])
orig_all = [t for v in orig_text.values() for t in v]

# ---- aug trainable pool ----
blind = {}
for f in glob.glob(os.path.join(HERE, '_blind_out', 'shard_*.jsonl')) + glob.glob(os.path.join(HERE, '_blind_out2', 'shard_*.jsonl')):
    for line in open(f):
        line = line.strip()
        if line: o = json.loads(line); blind[o['id']] = (o['box'], o['ntok'])
aug_text = defaultdict(list)
for mf, pdir in [(os.path.join(HERE, 'puzzles', '_manifest.csv'), 'puzzles'),
                 (os.path.join(HERE, 'puzzles2', '_manifest.csv'), 'puzzles2')]:
    if not os.path.exists(mf): continue
    for row in csv.DictReader(open(mf)):
        box, ntok = blind.get(row['id'], ('M', -1))
        if box in ('M', 'TIMEOUT') or str(box).startswith('ERR'): continue
        if verify(row['gold'], box) and 0 <= ntok < 7680:
            aug_text[row['subtype']].append(open(os.path.join(HERE, pdir, row['id'], 'question.txt')).read())
aug_all = [t for v in aug_text.values() for t in v]

print("=== per-category TRAINABLE availability ===")
ORDER = ['arithmetic', 'little_endian', 'pure_concat', 'mixed_concat', 'mixed_concat_little_endian', 'query_unseen_concat']
TARGET = {'arithmetic': 300, 'little_endian': 300, 'pure_concat': 100, 'mixed_concat': 100, 'mixed_concat_little_endian': 100, 'query_unseen_concat': 100}
for s in ORDER:
    need = TARGET[s] - len(orig_text[s])
    print(f"  {s:<30} orig {len(orig_text[s]):>4} | aug avail {len(aug_text[s]):>4} | need {need:>4} aug | {'OK' if len(aug_text[s])>=need else 'SHORT '+str(need-len(aug_text[s]))}")
print()
co = report("ORIG trainable (307, fixed)", orig_all)
ca = report("AUG trainable pool", aug_all)
cc = report("COMBINED pool (orig + all aug)", orig_all + aug_all)
miss = sorted(ACH - cc)
print(f"\nachievable tokens NOT in the combined trainable pool ({len(miss)}): {miss}")
