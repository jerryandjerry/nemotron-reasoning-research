#!/usr/bin/env python3
"""Measure how often each carry-column rule produces a NEW pruning, offered AFTER the existing
propagation fixpoint settles. Only NEW cuts count (a cut the current interval+naked/hidden-single
propagation did not already make). Mirrors _ruleb_instrument.py's search scaffold so the node
distribution matches the real generator's DFS.

Rules measured (as offered post-settle, per search node):
  R1  units congruence mod10 (add/mul)           -> Engine._units_narrow (existing, but normally OFF)
  R2  add 3-digit MSB=1 (guarded)
  R3  mul 4-digit MSB!=9 (4-distinct-digit guard)
  R5  sub units band (sign+reversal, +-2)

For each rule we also distinguish: did it produce a SINGLETON-or-empty cut (pin-only safe) vs a
partial removal (only safe under the pin-only wrapper)?
"""
import copy, os, glob, json, sys
import cot_generator as cg
import _gen_crypt as gc
from _gen_crypt import Engine
from _probe_heavy import get_engines_for, PREFIXES


# ---------- rule cut functions: return [(sym, kept_domain)] for changed syms ----------
def r1_units(eng, e, dom):
    return eng._units_narrow(e, dom)   # existing sound units narrow (add/mul)

def r2_add_msb(eng, e, dom):
    if eng.fam.get(e.op) != 'noisy_add': return []
    rch = [c for c in e.rhs if c not in eng.op_set]
    if len(rch) != 3: return []
    if e.d1 == e.d2 and e.d3 == e.d4: return []      # guard
    ms = rch[0] if eng.mode == 'standard' else rch[-1]
    if len(dom[ms]) == 1: return []
    if 1 in dom[ms]:
        if dom[ms] != {1}: return [(ms, {1})]
        return []
    return [(ms, set())]   # contradiction (true puzzles won't hit this on the gold branch)

def r3_mul_msb(eng, e, dom):
    if eng.fam.get(e.op) != 'noisy_mul': return []
    rch = [c for c in e.rhs if c not in eng.op_set]
    if len(rch) != 4: return []
    if len({e.d1, e.d2, e.d3, e.d4}) != 4: return []  # 4-distinct-symbol guard
    ms = rch[0] if eng.mode == 'standard' else rch[-1]
    if 9 in dom[ms]:
        nd = dom[ms] - {9}
        if nd != dom[ms]: return [(ms, nd)]
    return []

def r5_sub_units(eng, e, dom):
    if eng.fam.get(e.op) != 'noisy_subtraction': return []
    us = eng._units_syms(e)
    if not us: return []
    u1, u2, ur = us
    distinct = list(dict.fromkeys([u1, u2, ur]))
    ok = {s: set() for s in distinct}
    from itertools import product as iproduct
    for combo in iproduct(*[sorted(dom[s]) for s in distinct]):
        if len(set(combo)) != len(distinct): continue
        val = dict(zip(distinct, combo))
        band = set()
        for base in ((val[u1] - val[u2]) % 10, (val[u2] - val[u1]) % 10):
            for k in (-2, -1, 0, 1, 2):
                band.add((base + k) % 10)
        if val[ur] % 10 in band:
            for s in distinct: ok[s].add(val[s])
    return [(s, ok[s]) for s in distinct if ok[s] != dom[s]]


RULES = {'R1_units': r1_units, 'R2_add_msb': r2_add_msb, 'R3_mul_msb': r3_mul_msb, 'R5_sub_units': r5_sub_units}


def settle(eng, dom, locked):
    return eng.propagate(dom, locked, lambda m: None)


def offer_rules(eng, dom, locked, stats):
    """After the existing fixpoint settles, offer each rule ONCE per example and record NEW cuts.
    A cut counts only if it removes a value the settled domain still had. We DO apply the cut and
    re-settle (so we measure cascade-free, post-settle new prunings honestly), but we tally per rule.
    To avoid rules stealing each other's cuts, we evaluate each rule on the SAME settled snapshot."""
    snap = {s: set(d) for s, d in dom.items()}
    for rname, fn in RULES.items():
        for e in eng.p.examples:
            if e.op not in eng.fam: continue
            ch = fn(eng, e, snap)
            for s, nd in ch:
                removed = snap[s] - nd
                if removed:
                    stats[rname]['fires'] += 1
                    stats[rname]['vals_removed'] += len(removed)
                    if len(nd) <= 1:
                        stats[rname]['singleton_or_empty'] += 1
                    else:
                        stats[rname]['partial'] += 1


def search_instr(eng, dom, locked, stats, budget):
    budget[1] += 1
    if budget[1] > budget[0]:
        return 'BUDGET'
    d = copy.deepcopy(dom); lk = dict(locked)
    if not settle(eng, d, lk):
        return None
    offer_rules(eng, d, lk, stats)     # measure NEW cuts at this settled node
    for ei, e in enumerate(eng.p.examples, 1):
        if e.op in eng.fam and e.op not in lk and all(len(d[s]) == 1 for s in (e.d1, e.d2, e.d3, e.d4)):
            a, b, rstr, opc, zl, keep, excluded, used = eng._op_cands(e, d, lk)
            for v, rv in keep:
                r = search_instr(eng, d, {**lk, e.op: v}, stats, budget)
                if r == 'BUDGET': return 'BUDGET'
                if r: return r
            return None
    un = [s for s in eng.syms if len(d[s]) > 1]
    if not un:
        A = {s: next(iter(d[s])) for s in eng.syms}
        if all(op in lk for op in eng.fam) and eng.verify(A, lk): return (A, dict(lk))
        return None
    s = eng.pick_branch(d, un)
    for val in sorted(d[s]):
        nd = {t: ({val} if t == s else (dt - {val} if len(dt) > 1 else dt)) for t, dt in d.items()}
        r = search_instr(eng, nd, lk, stats, budget)
        if r == 'BUDGET': return 'BUDGET'
        if r: return r
    return None


def run_one(name, stats, budget=300_000):
    res = get_engines_for(name)
    bud = [budget, 0]
    for item in res:
        if len(item) == 4 and item[3] == 'concat':
            return 'CONCAT'
        mode, p, concat_results, sign_ops, cand, arith_ops, combos = item
        for combo in combos:
            fam = dict(zip(arith_ops, combo))
            eng = Engine(p, mode, set(p.op_syms), fam, concat_results, sign_ops)
            dom = {s: (set(range(1, 10)) if s in eng.lead else set(range(10))) for s in p.digit_syms}
            r = search_instr(eng, dom, {}, stats, bud)
            if r == 'BUDGET':
                return 'BUDGET'
            if r:
                return r
    return None


def all_dirs():
    out = []
    for d in sorted(glob.glob('*/')):
        name = d.rstrip('/').split('/')[-1]
        if os.path.exists(os.path.join(d, 'question.txt')) and any(name.startswith(p) for p in PREFIXES):
            out.append(name)
    return out


def main():
    names = all_dirs()
    if len(sys.argv) > 1:
        names = names[:int(sys.argv[1])]
    stats = {r: dict(fires=0, vals_removed=0, singleton_or_empty=0, partial=0, puzzles_fired=0) for r in RULES}
    budgeted = []
    for name in names:
        per = {r: dict(fires=0, vals_removed=0, singleton_or_empty=0, partial=0) for r in RULES}
        if run_one(name, per) == 'BUDGET':
            budgeted.append(name)
        for r in RULES:
            for k in ('fires', 'vals_removed', 'singleton_or_empty', 'partial'):
                stats[r][k] += per[r][k]
            if per[r]['fires'] > 0:
                stats[r]['puzzles_fired'] += 1
    print(json.dumps({'n_puzzles': len(names), 'budgeted': len(budgeted),
                      'budgeted_names': budgeted[:20], 'rules': stats}, indent=2))


if __name__ == '__main__':
    main()
