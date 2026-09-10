#!/usr/bin/env python3
"""Build the FULL training CSV with cryptarithm augmentation, based on 260607_NumericEq_1500_FULL.csv.
  - every non-cryptarithm row: byte-identical to the reference (numeric eq + its aug + all other categories).
  - the 800 cryptarithm rows: solver_cot SWAPPED to OUR 260525 solver CoT (recompute token length + GT-match).
  - APPEND the 693 selected cryptarithm aug rows (our balanced/coverage-selected complement).
  - oversampling: ALL GT-True cryptarithm (orig + aug) that are trainable (under 7680) -> 1x ; else 0x.
Output: 260607_CryptAug_TrainData/260607_Cryptarithm_1000_FULL.csv"""
import os, sys, re, csv, glob, math
csv.field_size_limit(10 ** 7)
from tokenizers import Tokenizer
ROOT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron'
HERE = os.path.dirname(os.path.abspath(__file__))
REF = f'{ROOT}/01_data/260607_NumericEq_Aug/260607_NE_aug_TrainData/260607_NumericEq_1500_FULL.csv'
CRYPT = f'{ROOT}/01_data/260525_Cryptarithm'
TOKZ = f'{ROOT}/02_train/260512_huikang_085/repo/tokenizer.json'
OUTDIR = f'{HERE}/260607_CryptAug_TrainData'; os.makedirs(OUTDIR, exist_ok=True)
OUT = f'{OUTDIR}/260607_Cryptarithm_1000_FULL.csv'
CAP = 7680
CRYPT_CATS = ('cryptarithm_deduce', 'cryptarithm_guess')

def extract(t):
    bs = list(re.finditer(r'\\boxed\{', t)); ms = []
    for i, m in enumerate(bs):
        end = bs[i + 1].start() if i + 1 < len(bs) else len(t)
        seg = t[m.end():end]; lb = seg.rfind('}'); ms.append(seg[:lb] if lb != -1 else seg)
    ne = [m.strip() for m in ms if m.strip()]
    return ne[-1] if ne else (ms[-1].strip() if ms else 'NOT_FOUND')

def verify(stored, pred):
    stored = stored.strip(); pred = pred.strip()
    if re.fullmatch(r'[01]+', stored): return pred.lower() == stored.lower()
    try: return math.isclose(float(stored), float(pred), rel_tol=1e-2, abs_tol=1e-5)
    except Exception: return pred.lower() == stored.lower()

def main():
    tok = Tokenizer.from_file(TOKZ)
    tlen = lambda s: len(tok.encode(s, add_special_tokens=False).ids)

    # our 260525 CoTs keyed by 8-hex id
    ours = {}
    for d in glob.glob(f'{CRYPT}/*_*'):
        if not os.path.isdir(d): continue
        q = os.path.join(d, 'question.txt'); c = os.path.join(d, 'track/tree_cot.txt')
        if not (os.path.exists(q) and os.path.exists(c)): continue
        pid = os.path.basename(d).rsplit('_', 1)[1]
        ours[pid] = {'cot': open(c).read(), 'label': open(os.path.join(d, 'label.txt')).read().strip()
                     if os.path.exists(os.path.join(d, 'label.txt')) else os.path.basename(d).rsplit('_', 1)[0]}

    with open(REF, newline='') as fh:
        r = csv.DictReader(fh); cols = r.fieldnames; rows = list(r)

    swapped = miss = 0
    for row in rows:
        if row['category'] not in CRYPT_CATS: continue          # non-crypt: untouched
        pid = row['id']
        if pid not in ours: miss += 1; continue
        o = ours[pid]
        row['solver_cot'] = o['cot']; row['source'] = '260601_new_solver'; row['new label'] = o['label']
        n = tlen(o['cot']); row['token length'] = str(n)
        gt = verify(row['answer'], extract(o['cot'])); row['GT-match'] = str(gt)
        row['oversampling'] = '1' if (gt and n < CAP) else '0'
        swapped += 1

    # ---- append the 693 selected aug rows ----
    sel = [l.strip() for l in open(f'{HERE}/_final_selection.txt') if l.strip()]
    man = {}
    for mf in ['puzzles/_manifest.csv', 'puzzles2/_manifest.csv', 'puzzles3/_manifest.csv', 'puzzles4/_manifest.csv']:
        p = os.path.join(HERE, mf)
        if os.path.exists(p):
            for m in csv.DictReader(open(p)): man[m['id']] = m
    def adir(pid):
        n = int(pid.rsplit('_', 1)[1])
        return 'puzzles' if n < 100000 else 'puzzles2' if n < 200000 else 'puzzles3' if n < 300000 else 'puzzles4'
    added = 0
    for pid in sel:
        m = man[pid]; d = os.path.join(HERE, adir(pid), pid)
        cot = open(os.path.join(d, 'track/tree_cot.txt')).read()
        prompt = open(os.path.join(d, 'question.txt')).read()
        n = tlen(cot); gt = verify(m['gold'], extract(cot))
        assert gt and n < CAP, f'selected row not trainable?? {pid}'
        rows.append({'id': pid, 'prompt': prompt, 'answer': m['gold'], 'category': m['category'],
                     'solver_cot': cot, 'source': '260607_crypt_aug', 'in_7830': 'no',
                     'oversampling': '1', 'token length': str(n), 'new label': m['subtype'], 'GT-match': 'True'})
        added += 1

    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

    # ---- report ----
    from collections import Counter
    crypt = [r for r in rows if r['category'] in CRYPT_CATS]
    train = [r for r in crypt if r['oversampling'] == '1']
    print(f"wrote {OUT}")
    print(f"  total rows: {len(rows)}  (was {len(rows)-added} ref + {added} aug)")
    print(f"  cryptarithm rows: {len(crypt)}  | swapped originals: {swapped}  missing: {miss}  | aug appended: {added}")
    print(f"  cryptarithm oversampling: {dict(Counter(r['oversampling'] for r in crypt))}")
    print(f"  cryptarithm TRAINABLE (1x): {len(train)}   <-- target 1000")
    print(f"  trainable by subtype: {dict(Counter(r['new label'] for r in train))}")
    print(f"  trainable by source: {dict(Counter(r['source'] for r in train))}")
    print(f"  max token length among 1x rows (all categories): {max(int(r['token length']) for r in rows if r['oversampling']=='1')}")
    # integrity: non-crypt rows unchanged count
    nonc = [r for r in rows if r['category'] not in CRYPT_CATS]
    print(f"  non-cryptarithm rows (unchanged): {len(nonc)}")

if __name__ == '__main__':
    main()
