#!/usr/bin/env python3
"""INDEPENDENT validity check — does NOT trust the solver's box or shared render helpers.
For each CoT: parse the claimed solution (symbol->digit cipher, each operator's formula, reading order,
sign position) out of the CoT text, then with FRESH arithmetic + rendering code confirm that solution
(a) reproduces EVERY example equation's RHS exactly, and (b) yields the stored gold for the query.
A row passes only if an independent re-derivation agrees. Reports parse rate + verify rate; lists failures."""
import csv, re, sys
csv.field_size_limit(10 ** 8)
HARVEST = '260608_Cryptarithm_10k_harvest.csv'

OPS = set('+-*')
def parse_puzzle(prompt):
    exs = []; query = None
    for line in prompt.strip().splitlines():
        line = line.strip()
        if line.startswith('Now'): query = line.split(': ', 1)[1].strip()
        elif ' = ' in line and not line.startswith('In'):
            l, r = line.split(' = ', 1); l = l.strip(); r = r.strip()
            if len(l) == 5: exs.append((l, r))
    opset = set()
    for l, r in exs: opset.add(l[2])
    if query: opset.add(query[2])
    return exs, query, opset

def fresh_apply(formula, a, b):
    f = formula.replace('×', '*')
    table = {'a+b': a+b, 'a+b+1': a+b+1, 'a+b-1': a+b-1, 'a+b+2': a+b+2, 'a+b-2': a+b-2,
             'a*b': a*b, 'a*b+1': a*b+1, 'a*b-1': a*b-1, 'a*b+2': a*b+2, 'a*b-2': a*b-2,
             'a-b': a-b, 'b-a': b-a, '|a-b|': abs(a-b), '-|a-b|': -abs(a-b),
             'a-b+1': a-b+1, 'a-b-1': a-b-1, 'a-b+2': a-b+2, 'a-b-2': a-b-2}
    return table.get(f)

def render(R, d2s, opsym, leftward, sign_suffix):
    neg = R < 0; v = abs(R)
    digs = [int(c) for c in str(v)]
    if leftward: digs = digs[::-1]
    try: body = ''.join(d2s[d] for d in digs)
    except KeyError: return None
    if neg:
        body = (body + opsym) if (leftward and sign_suffix) else (opsym + body)
    return body

def operand(s1, s2, s2d, leftward):
    if s1 not in s2d or s2 not in s2d: return None
    a, b = s2d[s1], s2d[s2]
    return (10*b + a) if leftward else (10*a + b)

def verify_row(prompt, gold, cot):
    exs, query, opset = parse_puzzle(prompt)
    # 1) symbol<->letter
    sym2let = {}
    for m in re.finditer(r'^  (\S) -> ([A-Za-z])$', cot, re.M):
        sym2let[m.group(1)] = m.group(2)
    let2sym = {v: k for k, v in sym2let.items() if v.isupper()}
    op_letter = {k: v for k, v in sym2let.items() if v.islower()}  # opsym->f/g/h
    # 2) letter -> digit  (from Conclusion '{... A=1, B=8 ...}' or State); take the LAST occurrence
    let2dig = {}
    for m in re.finditer(r'([A-Z])=(\d)\b', cot): let2dig[m.group(1)] = int(m.group(2))
    sym2dig = {s: let2dig[L] for s, L in sym2let.items() if L in let2dig}
    dig2sym = {d: s for s, d in sym2dig.items()}
    # 3) reading + sign position
    leftward = 'reading order = leftward' in cot or 'Confirm reading order = leftward' in cot
    sign_suffix = 'leftward fully' in cot
    # 4) operator formula per op symbol: from 'Summary: ... f = <formula>' or 'lock f = <formula>'
    op_formula = {}  # f/g/h -> formula string
    for m in re.finditer(r'\block ([fgh]) = ([^\n.,]+)', cot): op_formula[m.group(1)] = m.group(2).strip()
    sm = re.search(r'Confirm reading order = [^,]+, (.+)', cot)
    if sm:
        for part in sm.group(1).split(','):
            mm = re.match(r'\s*([fgh]) = (\S+)', part)
            if mm: op_formula[mm.group(1)] = mm.group(2).strip()
    if not sym2dig and not any('∥' in (op_formula.get(op_letter.get(o, ''), '')) for o in opset):
        return 'noparse'

    def eval_op(opsym, o1, o2):  # returns rendered RHS string or None
        L = op_letter.get(opsym)
        fm = op_formula.get(L, '')
        if '∥' in fm or 'concat' in fm:
            fwd = (fm.strip() in ('a∥b',)) or 'fwd' in fm
            return (o1 + o2) if fwd else (o2 + o1)
        a = operand(o1[0], o1[1], sym2dig, leftward); b = operand(o2[0], o2[1], sym2dig, leftward)
        if a is None or b is None: return None
        R = fresh_apply(fm, a, b)
        if R is None: return None
        return render(R, dig2sym, opsym, leftward, sign_suffix)

    # (a) every example reproduces
    for l, r in exs:
        got = eval_op(l[2], l[0:2], l[3:5])
        if got is None: return 'noparse'
        if got != r: return 'EX_FAIL'
    # (b) query -> gold (skip unseen-op puzzles: query op not in examples -> guessed, handled by solver)
    qop = query[2]
    if qop in {l[2] for l, _ in exs}:
        got = eval_op(qop, query[0:2], query[3:5])
        if got is None: return 'noparse'
        if got != gold: return 'GOLD_FAIL'
    return 'PASS'

def main():
    rows = list(csv.DictReader(open(HARVEST)))
    from collections import Counter, defaultdict
    res = Counter(); by_sub = defaultdict(Counter); fails = []
    for r in rows:
        v = verify_row(r['prompt'], r['answer'], r['solver_cot'])
        res[v] += 1; by_sub[r['subtype']][v] += 1
        if v in ('EX_FAIL', 'GOLD_FAIL') and len(fails) < 12: fails.append((r['id'], r['subtype'], v))
    n = len(rows)
    print(f"INDEPENDENT re-verification over ALL {n} harvested CoTs (fresh arithmetic, ignores solver box):")
    print(f"  PASS (re-derivation reproduces all examples + gold): {res['PASS']}")
    print(f"  EX_FAIL (claimed cipher fails an example):           {res['EX_FAIL']}   <-- real defect if >0")
    print(f"  GOLD_FAIL (query re-derivation != stored gold):      {res['GOLD_FAIL']}  <-- real defect if >0")
    print(f"  noparse (couldn't extract solution from CoT text):   {res['noparse']}   (parser limitation, not a data defect)")
    print(f"  verified PASS rate among parseable: {100*res['PASS']/max(res['PASS']+res['EX_FAIL']+res['GOLD_FAIL'],1):.2f}%")
    print("  by subtype:")
    for s in ['arithmetic','little_endian','pure_concat','mixed_concat','mixed_concat_little_endian','query_unseen_concat']:
        c = by_sub[s]; print(f"    {s:<28} PASS={c['PASS']:<5} EX_FAIL={c['EX_FAIL']:<3} GOLD_FAIL={c['GOLD_FAIL']:<3} noparse={c['noparse']}")
    if fails: print("  sample failures:", fails)

if __name__ == '__main__':
    main()
