#!/usr/bin/env python3
"""ROOT-level (pre-branch, deterministic) firing of each carry-column rule.

For each puzzle's WINNING reading+combo (the first that solves, mirroring the generator), settle the
existing propagation at the ROOT, then offer each rule and tally NEW cuts ONCE. This is the
deterministic §N.4 phase where a narrated cut actually shortens the CoT (and can prune branches).
We also report how many puzzles get a NEW *singleton/contradiction* (pin-only-safe) cut at root.
"""
import copy, os, glob, json, sys
import cot_generator as cg
from _gen_crypt import Engine
from _probe_heavy import get_engines_for, PREFIXES
from _fire_carrycol import RULES, settle


def root_fire_for_engine(eng):
    dom = {s: (set(range(1, 10)) if s in eng.lead else set(range(10))) for s in eng.p.digit_syms}
    if not settle(eng, dom, {}):
        return None   # this combo dies immediately at root
    res = {}
    snap = {s: set(d) for s, d in dom.items()}
    for rname, fn in RULES.items():
        fires = 0; singleton = 0; partial = 0; vals = 0
        for e in eng.p.examples:
            if e.op not in eng.fam: continue
            for s, nd in fn(eng, e, snap):
                removed = snap[s] - nd
                if removed:
                    fires += 1; vals += len(removed)
                    if len(nd) <= 1: singleton += 1
                    else: partial += 1
        res[rname] = dict(fires=fires, singleton=singleton, partial=partial, vals=vals)
    return res


def winning_engine(name):
    """Reproduce the generator's reading+combo selection enough to find the engine that SOLVES,
    and return it. Falls back to the first engine that settles at root."""
    res = get_engines_for(name)
    import copy as _c
    for item in res:
        if len(item) == 4 and item[3] == 'concat':
            return None
        mode, p, concat_results, sign_ops, cand, arith_ops, combos = item
        for combo in combos:
            fam = dict(zip(arith_ops, combo))
            eng = Engine(p, mode, set(p.op_syms), fam, concat_results, sign_ops)
            # try to solve to confirm this is a real winning combo
            dom = {s: (set(range(1, 10)) if s in eng.lead else set(range(10))) for s in p.digit_syms}
            if _solves(eng, dom, {}, [300_000, 0]):
                return eng
    # fallback: first engine that at least settles at root
    for item in res:
        if len(item) == 4 and item[3] == 'concat':
            return None
        mode, p, concat_results, sign_ops, cand, arith_ops, combos = item
        for combo in combos:
            fam = dict(zip(arith_ops, combo))
            eng = Engine(p, mode, set(p.op_syms), fam, concat_results, sign_ops)
            dom = {s: (set(range(1, 10)) if s in eng.lead else set(range(10))) for s in p.digit_syms}
            if settle(eng, dom, {}):
                return eng
    return None


def _solves(eng, dom, locked, budget):
    budget[1] += 1
    if budget[1] > budget[0]: return False
    d = copy.deepcopy(dom); lk = dict(locked)
    if not settle(eng, d, lk): return False
    for ei, e in enumerate(eng.p.examples, 1):
        if e.op in eng.fam and e.op not in lk and all(len(d[s]) == 1 for s in (e.d1, e.d2, e.d3, e.d4)):
            a, b, rstr, opc, zl, keep, excluded, used = eng._op_cands(e, d, lk)
            for v, rv in keep:
                if _solves(eng, d, {**lk, e.op: v}, budget): return True
            return False
    un = [s for s in eng.syms if len(d[s]) > 1]
    if not un:
        A = {s: next(iter(d[s])) for s in eng.syms}
        return all(op in lk for op in eng.fam) and eng.verify(A, lk)
    s = eng.pick_branch(d, un)
    for val in sorted(d[s]):
        nd = {t: ({val} if t == s else (dt - {val} if len(dt) > 1 else dt)) for t, dt in d.items()}
        if _solves(eng, nd, lk, budget): return True
    return False


def all_dirs():
    out = []
    for d in sorted(glob.glob('*/')):
        name = d.rstrip('/').split('/')[-1]
        if os.path.exists(os.path.join(d, 'question.txt')) and any(name.startswith(p) for p in PREFIXES):
            out.append(name)
    return out


def main():
    names = all_dirs()
    if len(sys.argv) > 1: names = names[:int(sys.argv[1])]
    agg = {r: dict(puzzles_any=0, puzzles_singleton=0, total_fires=0, total_singleton=0, total_partial=0, total_vals=0) for r in RULES}
    n_with_engine = 0
    for name in names:
        eng = winning_engine(name)
        if eng is None: continue
        rf = root_fire_for_engine(eng)
        if rf is None: continue
        n_with_engine += 1
        for r in RULES:
            f = rf[r]
            if f['fires'] > 0: agg[r]['puzzles_any'] += 1
            if f['singleton'] > 0: agg[r]['puzzles_singleton'] += 1
            agg[r]['total_fires'] += f['fires']
            agg[r]['total_singleton'] += f['singleton']
            agg[r]['total_partial'] += f['partial']
            agg[r]['total_vals'] += f['vals']
    print(json.dumps({'n_puzzles': len(names), 'n_with_engine': n_with_engine, 'rules': agg}, indent=2))


if __name__ == '__main__':
    main()
