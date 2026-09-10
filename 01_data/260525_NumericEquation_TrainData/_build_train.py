#!/usr/bin/env python3
"""Build the merged training CSV in 260525_NumericEquation_TrainData/.

Start from the golden base (260514_huikang_golden_stripped.csv, 6906 rows, 9 categories) and:
  - for every equation_numeric row (561, all in our 732): replace solver_cot with OURS, set
    source=260525_new_solver, KEEP its base in_7830 / oversampling / category / prompt / answer;
  - append our 171 equation_numeric puzzles that are NOT in the base (from the unused pool):
    prompt/answer/category from huikang_unused.csv, solver_cot OURS, source=260525_new_solver,
    in_7830=no, oversampling=0;
  - leave all non-equation_numeric base rows untouched;
  - add 3 columns to EVERY row: token length (real Nemotron tokenizer over solver_cot),
    new label (our label.txt for equation_numeric rows, blank otherwise), GT-match (verify()).
Total = 6906 + 171 = 7077 rows."""
import csv, glob, os, re, math
from tokenizers import Tokenizer
csv.field_size_limit(10 ** 7)
ROOT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron'
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '260514_huikang_golden_stripped.csv')
UNUSED = f'{ROOT}/01_data/260514_huikang_update/huikang_unused.csv'
OURDIR = f'{ROOT}/01_data/260523_Numeric_Equation'
TOKENIZER = f'{ROOT}/02_train/260512_huikang_085/repo/tokenizer.json'
OUT = os.path.join(HERE, '260525_NumericEquation_train.csv')

def extract_final_answer(text):                     # the live Kaggle metric (last non-empty box, rfind '}')
    bs = list(re.finditer(r'\\boxed\{', text)); ms = []
    for i, m in enumerate(bs):
        end = bs[i + 1].start() if i + 1 < len(bs) else len(text)
        seg = text[m.end():end]; lb = seg.rfind('}')
        ms.append(seg[:lb] if lb != -1 else seg)
    ne = [m.strip() for m in ms if m.strip()]
    return ne[-1] if ne else (ms[-1].strip() if ms else 'NOT_FOUND')

def verify(stored, pred):
    stored = stored.strip(); pred = pred.strip()
    if re.fullmatch(r'[01]+', stored): return pred.lower() == stored.lower()
    try: return math.isclose(float(stored), float(pred), rel_tol=1e-2, abs_tol=1e-5)
    except Exception: return pred.lower() == stored.lower()

def main():
    tok = Tokenizer.from_file(TOKENIZER)
    # our equation_numeric data: id -> {cot, label, prompt, answer}
    ours = {}
    for d in glob.glob(os.path.join(OURDIR, 'equation_numeric_*')):
        if not os.path.isdir(d): continue
        pid = os.path.basename(d).replace('equation_numeric_', '')
        ours[pid] = {'cot': open(os.path.join(d, 'track/tree_cot.txt')).read(),
                     'label': open(os.path.join(d, 'label.txt')).read().strip(),
                     'prompt': open(os.path.join(d, 'question.txt')).read(),
                     'answer': open(os.path.join(d, 'answer.txt')).read()}
    base = list(csv.DictReader(open(BASE, newline='')))
    base_ids = {r['id'] for r in base}
    unused = {r['id']: r for r in csv.DictReader(open(UNUSED, newline=''))}

    out_cols = ['id', 'prompt', 'answer', 'category', 'solver_cot', 'source', 'in_7830',
                'oversampling', 'token length', 'new label', 'GT-match']
    out = []
    prompt_mismatch = answer_mismatch = 0

    def new_cols(row):                              # token length / new label / GT-match for any finished row
        row['token length'] = str(len(tok.encode(row['solver_cot'], add_special_tokens=False).ids))
        row['new label'] = ours[row['id']]['label'] if row['category'].startswith('equation_numeric') and row['id'] in ours else ''
        row['GT-match'] = str(verify(row['answer'], extract_final_answer(row['solver_cot'])))
        return {c: row.get(c, '') for c in out_cols}

    # 1) all base rows in original order; swap our solver_cot into the equation_numeric ones
    for r in base:
        row = dict(r)
        if r['category'].startswith('equation_numeric'):
            o = ours[r['id']]                       # every base equation_numeric id is one of our 732 (verified)
            if r['prompt'] != o['prompt']: prompt_mismatch += 1
            if r['answer'].strip() != o['answer'].strip(): answer_mismatch += 1
            row['solver_cot'] = o['cot']
            row['source'] = '260525_new_solver'
        out.append(new_cols(row))

    # 2) append our 171 puzzles not in the base (prompt/answer/category from the unused pool)
    new_ids = sorted(pid for pid in ours if pid not in base_ids)
    for pid in new_ids:
        u = unused[pid]
        row = {'id': pid, 'prompt': u['prompt'], 'answer': u['answer'], 'category': u['category'],
               'solver_cot': ours[pid]['cot'], 'source': '260525_new_solver',
               'in_7830': 'no', 'oversampling': '0'}
        out.append(new_cols(row))

    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=out_cols); w.writeheader(); w.writerows(out)

    eq = [r for r in out if r['category'].startswith('equation_numeric')]
    eqm = sum(r['GT-match'] == 'True' for r in eq)
    print(f"wrote {OUT}: {len(out)} rows (base {len(base)} + new {len(new_ids)})")
    print(f"  equation_numeric rows: {len(eq)}  | our GT-match: {eqm}/{len(eq)}")
    print(f"  source=260525_new_solver rows: {sum(r['source']=='260525_new_solver' for r in out)}")
    print(f"  prompt mismatches (base vs ours): {prompt_mismatch}  answer mismatches: {answer_mismatch}")

if __name__ == '__main__':
    main()
