#!/usr/bin/env python3
"""Build the augmentation training CSV: the GT-match=True NE_aug puzzles in the SAME column format as
260606_Numeric_Equation/260601_NumericEq_TrainData/260601_NumericEq_gtTrue.csv, so it appends cleanly.
Columns: id, prompt, answer, category, solver_cot, source, in_7830, oversampling, token length, new label, GT-match.
source=260607_NE_aug ; category = equation_numeric_guess (unseen) | equation_numeric_deduce (else) ;
oversampling = 1x for ALL NE_aug rows (the complement is carried by the type distribution, not by weighting)."""
import csv, os, re, math, collections
from tokenizers import Tokenizer
csv.field_size_limit(10 ** 7)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron'
TOKENIZER = f'{ROOT}/02_train/260512_huikang_085/repo/tokenizer.json'
SOURCE = '260607_NE_aug'
OUTDIR = os.path.join(HERE, '260607_NE_aug_TrainData'); os.makedirs(OUTDIR, exist_ok=True)
OUT = os.path.join(OUTDIR, '260607_NE_aug_gtTrue.csv')

def extract_final_answer(text):                     # the live Kaggle metric (last non-empty box, rfind '}')
    bs = list(re.finditer(r'\\boxed\{', text)); ms = []
    for i, m in enumerate(bs):
        end = bs[i + 1].start() if i + 1 < len(bs) else len(text)
        seg = text[m.end():end]; lb = seg.rfind('}'); ms.append(seg[:lb] if lb != -1 else seg)
    ne = [m.strip() for m in ms if m.strip()]
    return ne[-1] if ne else (ms[-1].strip() if ms else 'NOT_FOUND')

def verify(stored, pred):
    stored = stored.strip(); pred = pred.strip()
    if re.fullmatch(r'[01]+', stored): return pred.lower() == stored.lower()
    try: return math.isclose(float(stored), float(pred), rel_tol=1e-2, abs_tol=1e-5)
    except Exception: return pred.lower() == stored.lower()

def main():
    tok = Tokenizer.from_file(TOKENIZER)
    man = {r['id']: r for r in csv.DictReader(open(os.path.join(HERE, '_manifest.csv'), newline=''))}
    keep = [l.strip() for l in open(os.path.join(HERE, '_gtmatch_true_ids.txt')) if l.strip()]
    cols = ['id', 'prompt', 'answer', 'category', 'solver_cot', 'source', 'in_7830',
            'oversampling', 'token length', 'new label', 'GT-match']
    rows = []; mism = 0
    for pid in keep:
        m = man[pid]; d = os.path.join(HERE, pid)
        prompt = open(d + '/question.txt').read()
        answer = open(d + '/answer.txt').read()
        cot = open(d + '/track/tree_cot.txt').read()
        label = open(d + '/label.txt').read().strip()                    # NE_aug_<type>
        sa = extract_final_answer(cot); gtm = verify(answer, sa)
        if not gtm: mism += 1
        category = 'equation_numeric_guess' if m['type'] == 'unseen' else 'equation_numeric_deduce'
        osv = '1'                                                        # ALL NE_aug samples 1x (complement comes from the type distribution, not oversampling)
        rows.append({'id': pid, 'prompt': prompt, 'answer': answer, 'category': category, 'solver_cot': cot,
                     'source': SOURCE, 'in_7830': 'no', 'oversampling': osv,
                     'token length': str(len(tok.encode(cot, add_special_tokens=False).ids)),
                     'new label': label, 'GT-match': str(gtm)})
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT}: {len(rows)} rows  (GT-match!=True: {mism})")
    print("category:", dict(collections.Counter(r['category'] for r in rows)))
    print("new label:", dict(collections.Counter(r['new label'] for r in rows)))
    print("oversampling:", dict(collections.Counter(r['oversampling'] for r in rows)),
          "| effective (sum):", sum(int(r['oversampling']) for r in rows))

if __name__ == '__main__':
    main()
