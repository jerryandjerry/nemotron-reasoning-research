#!/usr/bin/env python3
"""Prove decorrelation: per-category feature distributions on the COMBINED 1500-effective eq set,
weighted by oversampling. If no surface feature predicts the category, every column is ~flat across rows.
Features (all parsed from question.txt — the surface the model could shortcut on):
  reading (from CoT 'Confirm reading order =' / aug manifest), signed (non-digit in any RHS),
  n_distinct_ops (2 vs 3), arith_symbol (+-* used as an operator), leadzero (a leading-0 operand)."""
import csv, re, collections, os
csv.field_size_limit(10 ** 7)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron'
FULL = os.path.join(HERE, '260612_TrainData', '260612_NumericEq5k_FULL.csv')
BUCKET = {'arithmetic_left_to_right': 'deducible', 'arithmetic_right_to_left': 'deducible', 'no_rule_fits': 'deducible',
          'ambiguous_arithmetic': 'ambiguous', 'unseen_operator': 'unseen', 'unseen_leading_zero': 'unseen',
          'exotic_operation': 'exotic', 'concatenation': 'concat'}
AUGB = {'NE_aug_arithmetic_left_to_right': 'deducible', 'NE_aug_arithmetic_right_to_left': 'deducible',
        'NE_aug_exotic_operation': 'exotic', 'NE_aug_unseen_operator': 'unseen', 'NE_aug_concatenation': 'concat',
        'NE_aug_ambiguous_arithmetic': 'ambiguous'}
EXLINE = re.compile(r'^\s*(\d+)\s*(\D)\s*(\d+)\s*=\s*(.*)$')
QLINE = re.compile(r'determine the result for:\s*(\d+)\s*(\D)\s*(\d+)')

def reading_from_cot(cot):
    ms = re.findall(r'(?:reading order =|actual reading is)\s*(rightward|leftward)', cot)
    return ms[-1] if ms else '?'

def features(prompt):
    ops = set(); operands = []; signed = False
    for ln in prompt.splitlines():
        m = EXLINE.match(ln)
        if m:
            a, op, b, rhs = m.groups(); ops.add(op); operands += [a, b]
            if re.search(r'\D', rhs.strip()): signed = True            # any non-digit in RHS = a sign glyph
    q = QLINE.search(prompt)
    if q: ops.add(q.group(2)); operands += [q.group(1), q.group(3)]
    arith = any(o in '+-*' for o in ops)
    leadz = any(len(x) > 1 and x[0] == '0' for x in operands)
    return {'signed': signed, 'nops': len(ops), 'arith': arith, 'leadz': leadz}

def main():
    rows = [r for r in csv.DictReader(open(FULL, newline='')) if r['category'].startswith('equation_numeric')]
    man = {r['id']: r for r in csv.DictReader(open(os.path.join(ROOT, '01_data/260607_NumericEq_Aug/_manifest.csv'), newline=''))}
    man.update({r['id']: r for r in csv.DictReader(open(os.path.join(HERE, '_manifest.csv'), newline=''))})
    agg = collections.defaultdict(lambda: collections.Counter())   # bucket -> feature counters (weighted)
    eff = collections.Counter()
    for r in rows:
        w = int(r['oversampling'])
        if w == 0: continue
        if r['source'] in ('260607_NE_aug', '260612_NE5k'):
            b = man[r['id']]['type'] if r['id'] in man else AUGB.get(r['new label'], '?')
            reading = man[r['id']]['reading'] if r['id'] in man else reading_from_cot(r['solver_cot'])
        else:
            b = BUCKET.get(r['new label'], '?')
            reading = ('leftward' if r['new label'] == 'arithmetic_right_to_left'
                       else 'rightward' if r['new label'] == 'arithmetic_left_to_right'
                       else reading_from_cot(r['solver_cot']))
        f = features(r['prompt'])
        eff[b] += w
        c = agg[b]
        c['rightward'] += w * (reading == 'rightward'); c['leftward'] += w * (reading == 'leftward')
        c['read?'] += w * (reading not in ('rightward', 'leftward'))
        c['signed'] += w * f['signed']
        c['3ops'] += w * (f['nops'] == 3); c['2ops'] += w * (f['nops'] == 2); c['other_ops'] += w * (f['nops'] not in (2, 3))
        c['arith'] += w * f['arith']
        c['leadz'] += w * f['leadz']

    order = ['deducible', 'exotic', 'unseen', 'ambiguous', 'concat']
    hdr = f"{'bucket':10s} {'eff':>5s} | {'right%':>6s} {'left%':>6s} | {'signed%':>7s} | {'2ops%':>6s} {'3ops%':>6s} | {'arith%':>6s} | {'leadz%':>6s}"
    print(hdr); print('-' * len(hdr))
    for b in order:
        n = eff[b]; c = agg[b]
        def p(k): return f"{c[k]/n:.0%}"
        print(f"{b:10s} {n:>5d} | {p('rightward'):>6s} {p('leftward'):>6s} | {p('signed'):>7s} | {p('2ops'):>6s} {p('3ops'):>6s} | {p('arith'):>6s} | {p('leadz'):>6s}")
    # spread = max-min across categories per feature (the shortcut metric: small = decorrelated)
    print('\nspread across categories (max-min; smaller = less shortcut signal):')
    for k in ['rightward', 'signed', '3ops', 'arith', 'leadz']:
        vals = [agg[b][k] / eff[b] for b in order]
        print(f"  {k:10s} {max(vals)-min(vals):.0%}   (per-cat: {[f'{v:.0%}' for v in vals]})")

if __name__ == '__main__':
    main()
