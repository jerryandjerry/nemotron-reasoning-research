#!/usr/bin/env python3
"""Axis distribution of the WHOLE equation_numeric training set (original + aug1 + aug2), independent
of any curriculum staging. All axes derived uniformly from the row's prompt / CoT / label, so original
and aug are measured the same way. Writes AXIS_DISTRIBUTION_eq_full.md + axis_distribution_eq_full.csv
next to this script."""
import csv, collections, re, sys, os, statistics
csv.field_size_limit(10 ** 7)
ROOT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron'
sys.path.insert(0, f'{ROOT}/01_data/260606_Numeric_Equation')
from _gen_eq import parse_prompt, parse_eq, consistent_ops, choose_mode, fam_of
HERE = os.path.dirname(os.path.abspath(__file__))
FULL = f'{HERE}/260612_TrainData/260612_NumericEq5k_FULL.csv'

BUCKET = {'arithmetic_left_to_right': 'deducible', 'arithmetic_right_to_left': 'deducible',
          'no_rule_fits': 'deducible', 'ambiguous_arithmetic': 'ambiguous', 'unseen_operator': 'unseen',
          'unseen_leading_zero': 'unseen', 'exotic_operation': 'exotic', 'concatenation': 'concat'}
def bucket(nl):
    nl = nl.replace('NE_aug_', '')
    return BUCKET.get(nl, 'deducible')
def source(r):
    return {'260606_new_solver': 'original', '260607_NE_aug': 'aug1', '260612_NE5k': 'aug2'}.get(r['source'], r['source'])

def axes(r):
    q = r['prompt']; cot = r['solver_cot']
    ex_lines = [l for l in q.splitlines() if re.match(r'^\s*\d+\D\d+\s*=', l)]
    ops = set(); operands = []; rhs_list = []
    for l in ex_lines:
        m = re.match(r'^\s*(\d+)(\D)(\d+)\s*=\s*(.*)$', l)
        if m: ops.add(m.group(2)); operands += [m.group(1), m.group(3)]; rhs_list.append(m.group(4).strip())
    qm = re.search(r'result for:\s*(\d+)(\D)(\d+)', q)
    if qm: ops.add(qm.group(2)); operands += [qm.group(1), qm.group(3)]
    # signed: where the sign glyph sits in any example RHS
    signed = 'none'
    for o in rhs_list:
        if o and not o[0].isdigit(): signed = 'prefix'; break
        if o and not o[-1].isdigit(): signed = 'suffix'
    leadz = any(len(x) > 1 and x[0] == '0' for x in operands)
    rm = re.search(r'Confirm reading order = (rightward|leftward)', cot)
    reading = rm.group(1) if rm else '?'
    typ = bucket(r['new label'])
    sub = '-'
    if typ in ('ambiguous', 'unseen'):
        ex, qn = parse_prompt(q); by = collections.defaultdict(list)
        for e in ex:
            pp = parse_eq(e['input_value'])
            if pp: by[pp[1]].append((pp[0], pp[2], e['output_value'].strip()))
        p = parse_eq(qn)
        if p:
            qop = p[1]; ro, rr = choose_mode(by)
            if typ == 'ambiguous':
                fit = consistent_ops(by.get(qop, []), ro, rr); sub = 'amb:' + (fit[0] if fit else '?')
            else:
                sub = 'uns:arith' if any(consistent_ops(v, ro, rr) and fam_of(consistent_ops(v, ro, rr)[0])
                                         in ('noisy_add', 'noisy_subtraction', 'noisy_mul') for v in by.values()) else 'uns:concat'
    return {'type': typ, 'reading': reading, 'signed': signed, 'leadzero': leadz,
            'nops': len(ops), 'arith': any(s in '+-*' for s in ops), 'source': source(r), 'subcase': sub}

rows = [r for r in csv.DictReader(open(FULL, newline=''))
        if r['category'].startswith('equation_numeric') and int(r['oversampling']) > 0]
recs = [(axes(r), int(r['token length'])) for r in rows]
N = len(recs)
AX = ['type', 'reading', 'signed', 'leadzero', 'nops', 'arith', 'source', 'subcase']
out = [f'# Axis distribution — whole equation_numeric training set (original + aug1 + aug2)', '',
       f'{N} trained rows (oversampling > 0). All axes derived uniformly from each row\'s prompt/CoT/label.', '']
for ax in AX:
    c = collections.Counter(str(a[ax]) for a, _ in recs)
    out.append(f'## {ax}'); out.append('| value | count | % |'); out.append('|---|---|---|')
    for v in sorted(c, key=lambda k: (-c[k], k)):
        out.append(f'| {v} | {c[v]} | {c[v]/N*100:.1f}% |')
    out.append('')
# operator-symbol axis: frequency of each glyph + coverage of the full 26-symbol set
PUNCT = list('/"`%!<[$^\'{?\\#(}):>@&|]'); ALLSYM = ['+', '-', '*'] + PUNCT
sym = collections.Counter()
for r in rows:
    for ln in r['prompt'].splitlines():
        m = re.match(r'^\s*\d+(\D)\d+\s*=', ln)
        if m: sym[m.group(1)] += 1
    qm = re.search(r'result for:\s*\d+(\D)\d+', r['prompt'])
    if qm: sym[qm.group(1)] += 1
miss = [s for s in ALLSYM if s not in sym]
out.append('## operator symbol')
out.append(f'coverage: **{len([s for s in ALLSYM if s in sym])} of {len(ALLSYM)}** possible symbols present'
           + (f'; MISSING {miss}' if miss else ' — full coverage')); out.append('')
out.append('| symbol | count | % of slots |'); out.append('|---|---|---|')
tot = sum(sym.values())
for s, c in sym.most_common(): out.append(f'| `{s}` | {c} | {c/tot*100:.1f}% |')
out.append('')

# length bands (terciles of the whole set)
L = sorted(tl for _, tl in recs); t1, t2 = L[N//3], L[2*N//3]
def band(tl): return 'short' if tl <= t1 else ('medium' if tl <= t2 else 'long')
cb = collections.Counter(band(tl) for _, tl in recs)
out.append('## length'); out.append(f'(terciles: short ≤{t1} · medium ≤{t2} · long >{t2})')
out.append('| band | count | % |'); out.append('|---|---|---|')
for b in ('short', 'medium', 'long'): out.append(f'| {b} | {cb[b]} | {cb[b]/N*100:.1f}% |')
out.append(f'\nmedian {int(statistics.median(L))} · mean {int(statistics.mean(L))} · min {min(L)} · max {max(L)}')
open(f'{HERE}/AXIS_DISTRIBUTION_eq_full.md', 'w').write('\n'.join(out) + '\n')
with open(f'{HERE}/axis_distribution_eq_full.csv', 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['axis', 'value', 'count', 'pct'])
    for ax in AX:
        c = collections.Counter(str(a[ax]) for a, _ in recs)
        for v in sorted(c): w.writerow([ax, v, c[v], round(c[v]/N*100, 1)])
    for s, c in sym.most_common(): w.writerow(['symbol', s, c, round(c/sum(sym.values())*100, 1)])
    for b in ('short', 'medium', 'long'): w.writerow(['length_band', b, cb[b], round(cb[b]/N*100, 1)])
print(f'wrote AXIS_DISTRIBUTION_eq_full.md + axis_distribution_eq_full.csv ({N} rows)')
