#!/usr/bin/env python3
"""Build the new-solver training dataset.

Start from the current multi-category reference dataset and, for the 800 cryptarithm rows
(matched by `id`), swap in OUR solver CoT and mark them `source="new solver"`. Leave every other
row (the 6106 non-cryptarithm examples) untouched. Then add three columns to ALL rows:
  - `token length` : tokens of `solver_cot` under the actual Nemotron training tokenizer
  - `new label`    : our answer-path bucket for cryptarithm rows; blank for the others
  - `GT-match`     : True/False via the official Kaggle metric (extract + verify) on the CoT
Cryptarithm CoTs of >= CAP tokens are KEPT in the CSV but get `oversampling = 0` (excluded from
training); `in_7830` and every other row's `oversampling` pass through unchanged."""
import csv, os, re, math
from tokenizers import Tokenizer
csv.field_size_limit(10 ** 7)


# ── verbatim Kaggle metric (metric/nvidia-nemotron-metric) ──────────────────────
def extract_final_answer(text):
    if text is None:
        return 'NOT_FOUND'
    boxed_starts = list(re.finditer(r'\\boxed\{', text))
    matches = []
    for i, m in enumerate(boxed_starts):
        start = m.end()
        end = boxed_starts[i + 1].start() if i + 1 < len(boxed_starts) else len(text)
        segment = text[start:end]
        last_brace = segment.rfind('}')
        matches.append(segment[:last_brace] if last_brace != -1 else segment)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        return non_empty[-1] if non_empty else matches[-1].strip()
    for pattern in (r'The final answer is:\s*([^\n]+)', r'Final answer is:\s*([^\n]+)',
                    r'Final answer\s*[:：]\s*([^\n]+)', r'final answer\s*[:：]\s*([^\n]+)'):
        mm = re.findall(pattern, text, re.IGNORECASE)
        if mm:
            return mm[-1].strip()
    mm = re.findall(r'-?\d+(?:\.\d+)?', text)
    if mm:
        return mm[-1]
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else 'NOT_FOUND'


def verify(stored_answer, predicted):
    stored_answer = stored_answer.strip(); predicted = predicted.strip()
    if re.fullmatch(r'[01]+', stored_answer):                 # binary string -> strict
        return predicted.lower() == stored_answer.lower()
    try:                                                      # numeric -> 1% tolerance
        return math.isclose(float(stored_answer), float(predicted), rel_tol=1e-2, abs_tol=1e-5)
    except Exception:                                         # else -> case-insensitive string
        return predicted.lower() == stored_answer.lower()
# ────────────────────────────────────────────────────────────────────────────────

CAP = 7680   # rows whose CoT is >= CAP tokens get oversampling=0 (CoT kept in the CSV, excluded from training)
ROOT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron'
REF = f'{ROOT}/01_data/260514_lkevincc_golden/260515_huikang_lkall_stripped.csv'
OURS = f'{ROOT}/01_data/260516_Cryptarithm/golden_cryptarithm_cot_ours.csv'
TOK = f'{ROOT}/02_train/260512_huikang_085/repo/tokenizer.json'
OUT = f'{ROOT}/01_data/260523_new_data/260523_huikang_newsolver.csv'

tok = Tokenizer.from_file(TOK)

# our cryptarithm rows, keyed by id: our CoT, our bucket, and our (same-tokenizer) token length
crypto = {r['id']: {'cot': r['solver_cot'], 'label': r['new label'], 'tok': r['cot token length']}
          for r in csv.DictReader(open(OURS, newline=''))}

ref = list(csv.DictReader(open(REF, newline='')))
out_cols = ['id', 'prompt', 'answer', 'category', 'solver_cot', 'source',
            'in_7830', 'oversampling', 'token length', 'new label', 'GT-match']
rows = []; nrepl = nexcl = ngt = 0
for r in ref:
    o = dict(r)
    if r['id'] in crypto:                          # cryptarithm: swap in our CoT
        c = crypto[r['id']]
        o['solver_cot'] = c['cot']
        o['source'] = 'new solver'
        o['new label'] = c['label']
        o['token length'] = c['tok']               # reuse: same Nemotron tokenizer, cot-only
        nrepl += 1
    else:                                          # everything else: untouched, just measured/labelled
        o['new label'] = ''
        o['token length'] = str(len(tok.encode(r['solver_cot'], add_special_tokens=False).ids))
    # GT-match via the official metric, on the actual CoT
    gt = verify(o['answer'], extract_final_answer(o['solver_cot']))
    o['GT-match'] = str(gt); ngt += gt
    if int(o['token length']) >= CAP:              # over-long: keep the CoT but zero its oversampling
        o['oversampling'] = '0'                     # (kept in the CSV for the record, excluded from training)
        nexcl += 1
    rows.append({k: o.get(k, '') for k in out_cols})

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=out_cols); w.writeheader(); w.writerows(rows)
print(f"wrote {OUT}\n  rows {len(rows)} | cryptarithm CoTs replaced {nrepl} | "
      f"over-long rows oversampling=0 (>= {CAP} tok, CoT kept) {nexcl} | GT-match {ngt}/{len(rows)} | {os.path.getsize(OUT):,} bytes")
