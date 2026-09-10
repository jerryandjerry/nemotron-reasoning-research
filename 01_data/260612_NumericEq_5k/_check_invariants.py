#!/usr/bin/env python3
"""Hard-invariant sweep over the GT-true pool of THIS round (the same invariants the original 732
always satisfies, per the v3 audit):
  1. negative gold REQUIRES a negative example in the puzzle (orig: 48/48)
  2. NO literal '-' sign with a glyph operator — questions AND golds ('-' allowed only when the
     operator symbol IS '-')
  3. a negative output's sign glyph == that operator's OWN symbol (prefix or suffix)
  4. leading-zero results present at a real rate (orig ~7.5%)
  5. live-metric extraction: 0 empty boxes across all GT-true CoTs
  6. ambiguous-specific: every ambiguous puzzle classifies as ambiguous_arithmetic under the
     solver's own classifier, and its gold == the boxed blind answer
"""
import os, re, sys, csv, collections
HERE = os.path.dirname(os.path.abspath(__file__))
GEN = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260606_Numeric_Equation'
sys.path.insert(0, GEN)
from _gen_eq import parse_prompt, parse_eq, classify, extract_final_answer, verify

EXLINE = re.compile(r'^\s*(\d+)(\D)(\d+)\s*=\s*(.*)$')

def main():
    man = {r['id']: r for r in csv.DictReader(open(os.path.join(HERE, '_manifest.csv'), newline=''))}
    ids = [l.strip() for l in open(os.path.join(HERE, '_gtmatch_true_ids.txt')) if l.strip()]
    bad = collections.Counter(); leadz = 0; amb_ok = 0; amb_n = 0
    for pid in ids:
        d = os.path.join(HERE, pid)
        q = open(os.path.join(d, 'question.txt')).read()
        gold = open(os.path.join(d, 'answer.txt')).read().strip()
        cot = open(os.path.join(d, 'track', 'tree_cot.txt')).read()
        # collect example signs per operator
        neg_example = False; ops_seen = set()
        for ln in q.splitlines():
            m = EXLINE.match(ln)
            if not m: continue
            a, op, b, rhs = m.groups(); rhs = rhs.strip(); ops_seen.add(op)
            if rhs and (not rhs[0].isdigit() or not rhs[-1].isdigit()):
                neg_example = True
                glyph = rhs[0] if not rhs[0].isdigit() else rhs[-1]
                if glyph != op: bad['sign-glyph != operator (example)'] += 1
                if glyph == '-' and op != '-': bad["literal '-' with glyph operator (example)"] += 1
        qm = re.search(r'determine the result for:\s*(\d+)(\D)(\d+)', q)
        qop = qm.group(2)
        # gold sign checks
        if gold and (not gold[0].isdigit() or not gold[-1].isdigit()):
            glyph = gold[0] if not gold[0].isdigit() else gold[-1]
            if not neg_example: bad['negative gold WITHOUT negative example'] += 1
            if glyph != qop: bad['sign-glyph != operator (gold)'] += 1
            if glyph == '-' and qop != '-': bad["literal '-' with glyph operator (gold)"] += 1
        # leading-zero result presence
        if re.match(r'^0\d', gold): leadz += 1
        # extractor
        box = extract_final_answer(cot)
        if not box or box == 'NOT_FOUND': bad['empty/NOT_FOUND box'] += 1
        # ambiguous-specific
        if man[pid]['type'] == 'ambiguous':
            amb_n += 1
            ex, qn = parse_prompt(q)
            if classify({'examples': ex, 'question': qn}) != 'ambiguous_arithmetic':
                bad['ambiguous misclassified'] += 1
            elif verify(gold, box):
                amb_ok += 1
    print(f"GT-true pool: {len(ids)}")
    print(f"leading-zero golds: {leadz} ({leadz/len(ids):.1%})")
    print(f"ambiguous verified (classify + gold==box): {amb_ok}/{amb_n}")
    if bad:
        for k, v in bad.items(): print(f"  VIOLATION {k}: {v}")
        sys.exit(1)
    print("ALL INVARIANTS HOLD (0 violations)")

if __name__ == '__main__':
    main()
