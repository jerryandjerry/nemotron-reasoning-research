#!/usr/bin/env python3
"""Build golden_equation_numeric_cot_ours.csv from the current track/tree_cot.txt files.
Columns (mirrors 260516_Cryptarithm/golden_cryptarithm_cot_ours.csv):
  id, prompt (question.txt), answer (answer.txt), solver_cot (OURS, track/tree_cot.txt),
  ikev label  = the golden 'category' for this id (from 260514_huikang_golden_stripped.csv; blank if not in that set),
  new label   = our label.txt (from classify()),
  solver answer = boxed answer via the live-metric extractor,
  GT-match    = our verify() (same metric the generator reports, so the True-count == OUR GT-match),
  cot token length = tokens under the real Nemotron tokenizer."""
import csv, glob, os, re, math
from tokenizers import Tokenizer
csv.field_size_limit(10 ** 7)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron'
GOLD = f'{ROOT}/01_data/260514_lkevincc_golden/260514_huikang_golden_stripped.csv'
TOKENIZER = f'{ROOT}/02_train/260512_huikang_085/repo/tokenizer.json'

def extract_final_answer(text):                    # identical to _gen_eq.py / the live Kaggle metric
    bs = list(re.finditer(r'\\boxed\{', text)); ms = []
    for i, m in enumerate(bs):
        end = bs[i + 1].start() if i + 1 < len(bs) else len(text)
        seg = text[m.end():end]; lb = seg.rfind('}')
        ms.append(seg[:lb] if lb != -1 else seg)
    ne = [m.strip() for m in ms if m.strip()]
    return ne[-1] if ne else (ms[-1].strip() if ms else 'NOT_FOUND')

def verify(stored, pred):                          # identical to _gen_eq.py verify()
    stored = stored.strip(); pred = pred.strip()
    if re.fullmatch(r'[01]+', stored): return pred.lower() == stored.lower()
    try: return math.isclose(float(stored), float(pred), rel_tol=1e-2, abs_tol=1e-5)
    except Exception: return pred.lower() == stored.lower()

def main():
    tok = Tokenizer.from_file(TOKENIZER)
    gold = {}                                       # id -> golden category (the 'ikev label')
    with open(GOLD, newline='') as fh:
        for row in csv.DictReader(fh):
            gold[row['id']] = row['category']
    cols = ['id', 'prompt', 'answer', 'solver_cot', 'ikev label', 'new label',
            'solver answer', 'GT-match', 'cot token length']
    rows = []; n = match = have_ikev = 0
    for d in sorted(glob.glob(os.path.join(HERE, 'equation_numeric_*'))):
        if not os.path.isdir(d): continue
        pid = os.path.basename(d).replace('equation_numeric_', '')
        prompt = open(os.path.join(d, 'question.txt')).read()
        answer = open(os.path.join(d, 'answer.txt')).read()
        cot = open(os.path.join(d, 'track/tree_cot.txt')).read()
        new_label = open(os.path.join(d, 'label.txt')).read().strip()
        sa = extract_final_answer(cot)
        gtm = verify(answer, sa)
        ntok = len(tok.encode(cot, add_special_tokens=False).ids)
        ikev = gold.get(pid, '')
        n += 1; match += gtm; have_ikev += bool(ikev)
        rows.append({'id': pid, 'prompt': prompt, 'answer': answer, 'solver_cot': cot,
                     'ikev label': ikev, 'new label': new_label, 'solver answer': sa,
                     'GT-match': str(gtm), 'cot token length': str(ntok)})
    out = os.path.join(HERE, 'golden_equation_numeric_cot_ours.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}: {n} rows, GT-match {match}/{n}, ikev label present {have_ikev}/{n}")

if __name__ == '__main__':
    main()
