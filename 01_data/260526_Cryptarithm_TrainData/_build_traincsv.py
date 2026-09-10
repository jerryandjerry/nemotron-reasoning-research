#!/usr/bin/env python3
"""Build the cryptarithm training datasets with the new-solver CoTs (260525_Cryptarithm on-disk output).

Produces TWO datasets, both with the 800 cryptarithm CoTs from 260525_Cryptarithm/<cat>_<hash>/track/tree_cot.txt:
  1. newsolver.csv  — REF golden base; cryptarithm swapped, every other row recomputed (numeric stays ORIGINAL).
  2. NumericEq.csv  — combined set built from NUMEQ_BASE (260527_..._allEq_gtTrue.csv, the latest numeric
                       solver set); swap ONLY the cryptarithm rows, leaving the numeric-solver CoTs (and every
                       other row, with their own oversampling) byte-for-byte untouched.
Columns: id, prompt, answer, category, solver_cot, source, in_7830, oversampling, token length, new label, GT-match.
Cryptarithm rows matched by id = puzzle hash = folder suffix. `new label` reused from the old ours CSV
(answer-path bucket is unchanged — same search/answer logic). token length + GT-match recomputed on the new CoT.
Cryptarithm rows <CAP tokens get oversampling=2; >= CAP keep the CoT but get oversampling=0 (excluded).
Final gate (ALL rows, both files): oversampling=0 if token length >= CAP OR GT-match != True (over-cap
and GT-mismatch rows are kept in the file but excluded from training)."""
import csv, os, re, math, glob
from tokenizers import Tokenizer
csv.field_size_limit(10 ** 7)

def extract_final_answer(text):
    if text is None: return 'NOT_FOUND'
    boxed_starts = list(re.finditer(r'\\boxed\{', text)); matches = []
    for i, m in enumerate(boxed_starts):
        start = m.end(); end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        seg = text[start:end]; lb = seg.rfind('}'); matches.append(seg[:lb] if lb != -1 else seg)
    if matches:
        ne = [m.strip() for m in matches if m.strip()]; return ne[-1] if ne else matches[-1].strip()
    for pat in (r'The final answer is:\s*([^\n]+)', r'Final answer is:\s*([^\n]+)',
                r'Final answer\s*[:：]\s*([^\n]+)', r'final answer\s*[:：]\s*([^\n]+)'):
        mm = re.findall(pat, text, re.IGNORECASE)
        if mm: return mm[-1].strip()
    mm = re.findall(r'-?\d+(?:\.\d+)?', text)
    if mm: return mm[-1]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else 'NOT_FOUND'

def verify(stored, pred):
    stored = stored.strip(); pred = pred.strip()
    if re.fullmatch(r'[01]+', stored): return pred.lower() == stored.lower()
    try: return math.isclose(float(stored), float(pred), rel_tol=1e-2, abs_tol=1e-5)
    except Exception: return pred.lower() == stored.lower()

CAP = 7680
ROOT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron'
REF  = f'{ROOT}/01_data/260514_lkevincc_golden/260515_huikang_lkall_stripped.csv'
NEW  = f'{ROOT}/01_data/260525_Cryptarithm'                                   # new CoTs live here
OURS = f'{ROOT}/01_data/260516_Cryptarithm/golden_cryptarithm_cot_ours.csv'   # for the `new label` bucket
TOK  = f'{ROOT}/02_train/260512_huikang_085/repo/tokenizer.json'
TD   = f'{ROOT}/01_data/260526_Cryptarithm_TrainData'
NUMEQ_BASE = f'{ROOT}/01_data/260526_Numeric_Equation/260527_NumericEq_TrainData/260527_huikang_NumericEq_allEq_gtTrue.csv'  # latest numeric-solver set; combined base
OUT_NEWSOLVER = f'{TD}/260527_huikang_newsolver.csv'   # base + new cryptarithm (numeric stays ORIGINAL golden)
OUT_NUMEQ     = f'{TD}/260527_huikang_NumericEq.csv'   # combined: new cryptarithm swapped into NUMEQ_BASE (numeric untouched)

tok = Tokenizer.from_file(TOK)
newcot = {d.split('_')[-1]: open(os.path.join(d, 'track', 'tree_cot.txt'), encoding='utf-8').read()
          for d in glob.glob(f'{NEW}/*_*') if os.path.isdir(d) and os.path.exists(os.path.join(d, 'track', 'tree_cot.txt'))}
label = {r['id']: r['new label'] for r in csv.DictReader(open(OURS, newline=''))}
out_cols = ['id', 'prompt', 'answer', 'category', 'solver_cot', 'source',
            'in_7830', 'oversampling', 'token length', 'new label', 'GT-match']

def swap_crypt(o):
    """If this row is a cryptarithm puzzle (id matches a new-CoT folder), swap in today's CoT and recompute
    its derived columns. Returns True if swapped (so callers can count)."""
    if o['id'] not in newcot: return False
    cot = newcot[o['id']]
    o['solver_cot'] = cot
    o['source'] = '260526_new solver'
    o['new label'] = label.get(o['id'], o.get('new label', ''))
    o['token length'] = str(len(tok.encode(cot, add_special_tokens=False).ids))
    o['GT-match'] = str(verify(o['answer'], extract_final_answer(cot)))
    o['oversampling'] = '0' if int(o['token length']) >= CAP else '2'   # <7680 -> 2x, else excluded
    return True

def gate(o):
    """Final training filter: a row is trained only if it is under the token cap AND its boxed answer
    matches answer.txt (GT-True). Otherwise oversampling=0 (kept in the file, excluded from training)."""
    if int(o['token length']) >= CAP or o['GT-match'] != 'True':
        o['oversampling'] = '0'

def write(path, rows):
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=out_cols); w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)

# ---- Build 1: newsolver.csv — REF golden base; cryptarithm swapped, every other row recomputed (original recipe) ----
rows = []; nrepl = nexcl = ngt = 0
for r in csv.DictReader(open(REF, newline='')):
    o = dict(r)
    if swap_crypt(o):
        nrepl += 1
    else:
        o['new label'] = ''
        o['token length'] = str(len(tok.encode(r['solver_cot'], add_special_tokens=False).ids))
        o['GT-match'] = str(verify(o['answer'], extract_final_answer(r['solver_cot'])))
    gate(o)                                          # token>=CAP OR GT-mismatch -> oversampling=0
    if o.get('oversampling') == '0': nexcl += 1
    ngt += (o['GT-match'] == 'True')
    rows.append({k: o.get(k, '') for k in out_cols})
write(OUT_NEWSOLVER, rows)
print(f"wrote {OUT_NEWSOLVER}\n  rows {len(rows)} | cryptarithm replaced {nrepl} | "
      f"oversampling=0 (>= {CAP}) {nexcl} | GT-match {ngt}/{len(rows)} | {os.path.getsize(OUT_NEWSOLVER):,} bytes")

# ---- Build 2: NumericEq.csv (combined) — swap ONLY cryptarithm rows into NUMEQ_BASE; numeric + all else untouched ----
base = list(csv.DictReader(open(NUMEQ_BASE, newline='')))
rows = []; nrepl = nexcl = ngt = 0
for r in base:
    o = dict(r)
    if swap_crypt(o): nrepl += 1
    gate(o)                                          # token>=CAP OR GT-mismatch -> oversampling=0
    if o.get('oversampling') == '0': nexcl += 1
    ngt += (o.get('GT-match', '') == 'True')
    rows.append({k: o.get(k, '') for k in out_cols})
write(OUT_NUMEQ, rows)
print(f"wrote {OUT_NUMEQ}\n  rows {len(rows)} | cryptarithm replaced {nrepl} | "
      f"oversampling=0 (>= {CAP}) {nexcl} | GT-match {ngt}/{len(rows)} | {os.path.getsize(OUT_NUMEQ):,} bytes")
