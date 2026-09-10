#!/usr/bin/env python3
"""Regenerate ALL 1,602 puzzle CoTs with THE solver. Verify everything."""
import os, re, sys, glob, csv as _csv
sys.path.insert(0, '.')
import _gen_bitm_final as F
import _gen_bitm_v3 as V
from tokenizers import Tokenizer
_csv.field_size_limit(10**8)
TOK = Tokenizer.from_file('/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/study/2606090112_grpo/cryptarithm_solver/tokenizer.json')
stored = {}
for r in _csv.DictReader(open('260514_huikang_golden_stripped.csv')):
    if r['category'] == 'bit_manipulation': stored[r['id']] = r['solver_cot']

def eval_lab(lab, ib):
    bits = ''.join(str(b) for b in ib)
    m = re.match(r'I(\d)$', lab)
    if m: return int(bits[int(m.group(1))])
    m = re.match(r'NOT(\d)$', lab)
    if m: return 1 - int(bits[int(m.group(1))])
    m = re.match(r'C(\d)$', lab)
    if m: return int(m.group(1))
    m = re.match(r'(AND|OR|XOR)(-NOT)?(\d)(\d)$', lab)
    if m:
        o, nt, a, b = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        x, y = ib[a], (1 - ib[b]) if nt else ib[b]
        return {'AND': x & y, 'OR': x | y, 'XOR': x ^ y}[o]
    m = re.match(r'Maj(\d)(\d)(\d)$', lab)
    if m:
        j, k, l = (int(m.group(i)) for i in (1, 2, 3)); return 1 if ib[j] + ib[k] + ib[l] >= 2 else 0
    if lab == 'default 1': return 1
    return int(V._eval_ext(lab, bits))

rows = []
stats = {}
lens_gt = []
for d in sorted(glob.glob('bitm_*')):
    pid = d[5:]
    cot = F.gen_cot(d)
    os.makedirs(os.path.join(d, 'track'), exist_ok=True)
    open(os.path.join(d, 'track', 'tree_cot.txt'), 'w').write(cot)
    q = open(os.path.join(d, 'question.txt')).read()
    gold = open(os.path.join(d, 'answer.txt')).read().strip()
    exs = re.findall(r'([01]{8}) -> ([01]{8})', q)
    qq = re.search(r'determine the output for: ([01]{8})', q).group(1)
    box = re.findall(r'\\boxed\{([^}]*)\}', cot)[-1]
    # soundness: every Selected rule fits all examples; boxed = rules applied to query
    m = re.search(r'\nSelected\n((?:\d .+\n){8})', cot)
    sel = {int(l.split(' ', 1)[0]): l.split(' ', 1)[1] for l in m.group(1).strip().split('\n')}
    sound = all(all(eval_lab(sel[b], [int(c) for c in i]) == int(o[b]) for i, o in exs)
                for b in range(8) if sel[b] != 'default 1')
    applied = ''.join(str(eval_lab(sel[b], [int(c) for c in qq])) for b in range(8))
    sound = sound and applied == box
    guessed = any(v == 'default 1' for v in sel.values())
    pop = 'train' if pid in stored else 'unused'
    ident = pop == 'train' and cot.strip() == stored[pid].strip()
    st = ('GT' if not guessed else 'GT_GUESS') if box == gold else 'WRONG'
    ntok = len(TOK.encode(q + cot, add_special_tokens=False).ids)
    key = f'{pop}:{st}' + (':identical' if ident else '')
    stats[key] = stats.get(key, 0) + 1
    if not sound: stats['UNSOUND'] = stats.get('UNSOUND', 0) + 1
    if st == 'GT': lens_gt.append(ntok)
    rows.append((d, pop, st, 'identical' if ident else 'regen', 'sound' if sound else 'UNSOUND', ntok))
with open('_regen_summary.csv', 'w', newline='') as fh:
    w = _csv.writer(fh); w.writerow(['folder', 'population', 'status', 'bytes', 'soundness', 'ntok']); w.writerows(rows)
print('REGEN ALL 1,602 — THE solver:')
for k in sorted(stats): print(f'  {k:<28} {stats[k]}')
v = sorted(lens_gt)
print(f'GT lengths: p50 {v[len(v)//2]} p90 {v[int(.9*len(v))]} max {v[-1]} | >=7680: {sum(1 for x in v if x>=7680)} | >=8192: {sum(1 for x in v if x>=8192)}')
gt = sum(n for k, n in stats.items() if ':GT' in k and 'GUESS' not in k)
gg = sum(n for k, n in stats.items() if 'GT_GUESS' in k)
print(f'TOTAL: honest-GT {gt} | default-assisted GT {gg} | trainable candidates {gt+gg} / 1602')
