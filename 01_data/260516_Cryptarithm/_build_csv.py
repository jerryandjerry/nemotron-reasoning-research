#!/usr/bin/env python3
"""Build golden_cryptarithm_cot_ours.csv from the current track/tree_cot.txt files.
Columns: id, prompt, answer (preserved from golden_cryptarithm_cot_raw.csv),
solver_cot (OURS), ikev label (label.txt), new label (answer-path bucket),
solver answer (boxed, via the live-metric extractor), GT-match (true/false)."""
import csv, glob, os, re
from tokenizers import Tokenizer
csv.field_size_limit(10**7)
HERE = os.path.dirname(os.path.abspath(__file__))
TOKENIZER = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/02_train/260512_huikang_085/repo/tokenizer.json'

def extract_boxed(text):                       # mirrors the live Kaggle metric (last non-empty box, rfind '}')
    starts = list(re.finditer(r'\\boxed\{', text))
    if not starts:
        return ''
    ms = []
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        seg = text[m.end():end]; lb = seg.rfind('}')
        ms.append(seg[:lb] if lb != -1 else seg)
    ne = [x.strip() for x in ms if x.strip()]
    return ne[-1] if ne else ms[-1].strip()

def bucket(cot):                               # answer-path category (exam-legitimate: derived from the CoT only)
    if "aren't arithmetically solvable" in cot: return 'blind_fallback'
    og = 'arithmetic operator too' in cot
    mg = 'first one that is left' in cot
    if og and mg: return 'guess_operator_and_symbol'
    if og: return 'guess_operator'
    if mg: return 'guess_symbol'
    if re.search(r'is concat_\w+ \(from the examples\)', cot): return 'derived_concat'
    return 'derived_arithmetic'

def main():
    tok = Tokenizer.from_file(TOKENIZER)                       # the actual Nemotron training tokenizer
    idmap = {d.rsplit('_', 1)[1]: d for d in glob.glob(os.path.join(HERE, '*_*')) if os.path.isdir(d)}
    cols = ['id', 'prompt', 'answer', 'solver_cot', 'ikev label', 'new label', 'solver answer', 'GT-match', 'cot token length']
    rows = []; n = match = 0
    with open(os.path.join(HERE, 'golden_cryptarithm_cot_raw.csv'), newline='') as fh:
        for row in csv.DictReader(fh):
            d = idmap.get(row['id'])
            if not d:
                continue
            cot = open(os.path.join(d, 'track/tree_cot.txt')).read()
            sa = extract_boxed(cot)
            gtm = sa.strip().lower() == row['answer'].strip().lower()
            ntok = len(tok.encode(cot, add_special_tokens=False).ids)
            n += 1; match += gtm
            rows.append({'id': row['id'], 'prompt': row['prompt'], 'answer': row['answer'],
                         'solver_cot': cot, 'ikev label': open(os.path.join(d, 'label.txt')).read().strip(),
                         'new label': bucket(cot), 'solver answer': sa, 'GT-match': str(gtm),
                         'cot token length': str(ntok)})
    out = os.path.join(HERE, 'golden_cryptarithm_cot_ours.csv')
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}: {n} rows, GT-match {match}/{n}")

if __name__ == '__main__':
    main()
