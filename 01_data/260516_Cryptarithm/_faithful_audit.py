#!/usr/bin/env python3
"""Mechanically check that the LATEST CoT faithfully logs its own search.

Two checks:
A) ARITHMETIC ASSERTIONS (cheap, run on the on-disk CoT for ALL puzzles):
   Every "<value> = <arithmetic expression>" the CoT writes (lock lines, variant-try lines,
   the final QUERY line) must be literally true. Evaluated independently with eval().
B) INTERVAL FAITHFULNESS (deep, sampled, re-runs the solver under a timeout):
   Re-run the solver with its res_iv / rhs_iv / narrow monkeypatched; recompute every interval
   and every narrowing BY BRUTE FORCE over the operand/digit domains and confirm the solver's
   emitted numbers are correct (exact for add/mul/locked; sound containment for subtraction).
   Also assert the on-disk CoT == freshly regenerated CoT, and the final solution verifies all examples.
"""
import os, re, sys, glob, json, signal, random
import cot_generator as cg
import _honest_solver as hs
from cot_generator import VARIANT_FN, CORES
CORE = hs.CORE

# ---------- check A: arithmetic-assertion evaluator over the on-disk CoT text ----------
def eval_arith(s):
    s = s.strip().replace('×', '*').replace('−', '-')
    s = re.sub(r'\|\s*(-?\d+)\s*-\s*(-?\d+)\s*\|', lambda m: f'abs({m.group(1)}-({m.group(2)}))', s)
    if not re.fullmatch(r'[-+*()\s0-9abs]+', s):
        return None
    try:
        return eval(s, {'abs': abs, '__builtins__': {}})
    except Exception:
        return None

def audit_arith_text(path):
    """Return (n_checked, [bad assertions])."""
    t = open(path, encoding='utf-8').read()
    checks = []
    # lock:  "143 = 11×13, so lock f = mul."
    for m in re.finditer(r'(-?\d+) = ([^,\n]+?), so lock ', t):
        checks.append((m.group(2), int(m.group(1)), m.group(0)))
    # variant-try: "f = mul (25×40 = 1001):"
    for m in re.finditer(r'\(([^()=\n]+?) = (-?\d+)\):', t):
        checks.append((m.group(1), int(m.group(2)), m.group(0)))
    # final query: "QUERY is 62×32+1 = 1985"
    for m in re.finditer(r'QUERY is (.+?) = (-?\d+)', t):
        checks.append((m.group(1), int(m.group(2)), m.group(0)))
    bad = []
    n = 0
    for expr, val, raw in checks:
        got = eval_arith(expr)
        if got is None:
            continue
        n += 1
        if got != val:
            bad.append((raw, f'eval={got}'))
    return n, bad

# ---------- check B: interval faithfulness via brute-force recompute ----------
def op_values(d1, d2, dom, mode):
    a, b = dom[d1], dom[d2]
    return {10 * x + y for x in a for y in b} if mode == 'standard' else {10 * y + x for x in a for y in b}

def indep_res(e, dom, locked, fam, mode):
    o1 = op_values(e.d1, e.d2, dom, mode); o2 = op_values(e.d3, e.d4, dom, mode)
    vs = [locked[e.op]] if e.op in locked else CORES[CORE[fam[e.op]]]
    vals = {VARIANT_FN[v](x, y) for v in vs for x in o1 for y in o2}
    tight = e.op in locked or all(v.startswith(('mul', 'add')) for v in vs)
    return (min(vals), max(vals)), tight

def indep_rhs(rhs, dom, op_set, mode):
    ch = list(rhs); sign = 1
    if ch and ch[0] in op_set: sign = -1; ch = ch[1:]
    elif ch and ch[-1] in op_set: sign = -1; ch = ch[:-1]
    nn = len(ch); co = [10 ** (nn - 1 - i) for i in range(nn)] if mode == 'standard' else [10 ** i for i in range(nn)]
    lo = sum(c * min(dom[s]) for c, s in zip(co, ch)); hi = sum(c * max(dom[s]) for c, s in zip(co, ch))
    return (-hi, -lo) if sign < 0 else (lo, hi)

class TO(Exception): pass
def _alarm(sig, frm): raise TO()

def audit_intervals(name, timeout=25):
    """Re-run solver with monkeypatched interval methods; brute-force-verify each. Returns dict."""
    stats = {'res': 0, 'res_unsound': 0, 'res_inexact': 0, 'rhs': 0, 'rhs_bad': 0,
             'narrow': 0, 'narrow_bad': 0, 'disk_eq_regen': None, 'solution_verifies': None, 'timeout': False}
    _res, _rhs, _narrow = hs.Engine.res_iv, hs.Engine.rhs_iv, hs.Engine.narrow

    def res_a(self, e, dom, locked):
        out = _res(self, e, dom, locked)
        truth, tight = indep_res(e, dom, locked, self.fam, self.mode)
        stats['res'] += 1
        if not (out[0] <= truth[0] and truth[1] <= out[1]): stats['res_unsound'] += 1
        if tight and out != truth: stats['res_inexact'] += 1
        return out
    def rhs_a(self, rhs, dom):
        out = _rhs(self, rhs, dom)
        stats['rhs'] += 1
        if out != indep_rhs(rhs, dom, self.op_set, self.mode): stats['rhs_bad'] += 1
        return out
    def narrow_a(self, sym, e, dom, locked):
        out = _narrow(self, sym, e, dom, locked)
        surv = set()
        for v in dom[sym]:
            d2 = {**dom, sym: {v}}
            r, _ = indep_res(e, d2, locked, self.fam, self.mode); rh = indep_rhs(e.rhs, d2, self.op_set, self.mode)
            if not (r[1] < rh[0] or rh[1] < r[0]): surv.add(v)
        stats['narrow'] += 1
        if out != surv: stats['narrow_bad'] += 1
        return out

    hs.Engine.res_iv, hs.Engine.rhs_iv, hs.Engine.narrow = res_a, rhs_a, narrow_a
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(timeout)
    try:
        cot, A, variants, mode = hs.gen(name)
        disk = open(os.path.join(name, 'track', 'tree_cot.txt'), encoding='utf-8').read()
        stats['disk_eq_regen'] = (disk.strip() == cot.strip())
        if A and variants:
            p = cg.parse(f'{name}/question.txt'); op_set = set(p.op_syms)
            ok = True
            for e in p.examples:
                if e.op not in variants: continue
                a = 10 * A[e.d1] + A[e.d2] if mode == 'standard' else 10 * A[e.d2] + A[e.d1]
                b = 10 * A[e.d3] + A[e.d4] if mode == 'standard' else 10 * A[e.d4] + A[e.d3]
                if VARIANT_FN[variants[e.op]](a, b) != cg.rhs_int(e.rhs, A, mode, op_set):
                    ok = False; break
            stats['solution_verifies'] = ok
    except TO:
        stats['timeout'] = True
    finally:
        signal.alarm(0)
        hs.Engine.res_iv, hs.Engine.rhs_iv, hs.Engine.narrow = _res, _rhs, _narrow
    return stats

def cats():
    out = {}
    for c in ['arithmetic', 'little_endian', 'mixed_concat_little_endian', 'mixed_concat', 'pure_concat', 'query_unseen_concat']:
        ds = []
        for d in sorted(glob.glob(c + '_*')):
            if os.path.isdir(d) and all(ch in '0123456789abcdef' for ch in d[len(c) + 1:]) and d[len(c) + 1:]:
                ds.append(d)
        out[c] = ds
    return out

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'A'
    allcats = cats()
    if mode == 'A':   # cheap arithmetic-assertion audit over ALL puzzles, on-disk text
        tot_checks = tot_bad = tot_puz = 0; bad_puz = []
        for c, ds in allcats.items():
            for d in ds:
                p = os.path.join(d, 'track', 'tree_cot.txt')
                if not os.path.exists(p): continue
                tot_puz += 1
                n, bad = audit_arith_text(p)
                tot_checks += n
                if bad:
                    tot_bad += len(bad); bad_puz.append((d, bad[:3]))
        print(f'[A] arithmetic-assertion audit over {tot_puz} on-disk CoTs')
        print(f'    total assertions checked: {tot_checks}')
        print(f'    FALSE assertions: {tot_bad}  (puzzles with any: {len(bad_puz)})')
        for d, bad in bad_puz[:20]:
            print('     ', d, bad)
    elif mode == 'B':  # deep interval audit on a stratified sample
        random.seed(7); per = int(sys.argv[2]) if len(sys.argv) > 2 else 25
        agg = {}
        for c, ds in allcats.items():
            sample = ds if len(ds) <= per else random.sample(ds, per)
            a = {'puz': 0, 'timeout': 0, 'res': 0, 'res_unsound': 0, 'res_inexact': 0,
                 'rhs': 0, 'rhs_bad': 0, 'narrow': 0, 'narrow_bad': 0, 'disk_ne_regen': 0, 'sol_fail': 0, 'sol_checked': 0}
            for d in sample:
                s = audit_intervals(d)
                a['puz'] += 1
                if s['timeout']: a['timeout'] += 1; continue
                for k in ['res', 'res_unsound', 'res_inexact', 'rhs', 'rhs_bad', 'narrow', 'narrow_bad']:
                    a[k] += s[k]
                if s['disk_eq_regen'] is False: a['disk_ne_regen'] += 1
                if s['solution_verifies'] is not None:
                    a['sol_checked'] += 1
                    if not s['solution_verifies']: a['sol_fail'] += 1
            agg[c] = a
            print(f'[B] {c:30s} n={a["puz"]:3d} timeout={a["timeout"]} | '
                  f'res {a["res"]} (unsound {a["res_unsound"]}, inexact {a["res_inexact"]}) | '
                  f'rhs {a["rhs"]} (bad {a["rhs_bad"]}) | narrow {a["narrow"]} (bad {a["narrow_bad"]}) | '
                  f'disk!=regen {a["disk_ne_regen"]} | sol {a["sol_checked"]-a["sol_fail"]}/{a["sol_checked"]} verify')
        tot = {k: sum(agg[c][k] for c in agg) for k in next(iter(agg.values()))}
        print('\n[B TOTAL]', json.dumps(tot))
