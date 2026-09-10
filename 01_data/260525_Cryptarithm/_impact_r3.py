#!/usr/bin/env python3
"""Measure R3 (mul 4-digit MSB!=9, partial removal) IMPACT: node-count reduction (proxy for tokens)
and answer preservation, on borderline over-cap puzzles. Inject R3 into the propagation fixpoint.

We compare:
  baseline : current propagate
  +R3      : current propagate, then apply R3 partial removals, re-settle, loop to fixpoint
node count = search() node visits (the codebase's CoT-length proxy, see _probe_heavy.py).
"""
import copy, json, sys, os, glob
import cot_generator as cg
from _gen_crypt import Engine
from _probe_heavy import get_engines_for, PREFIXES
from _fire_carrycol import r3_mul_msb, settle


def settle_r3(eng, dom, locked):
    if not settle(eng, dom, locked):
        return False
    while True:
        cut = False
        for e in eng.p.examples:
            if e.op not in eng.fam: continue
            for s, nd in r3_mul_msb(eng, e, dom):
                if nd != dom[s]:
                    dom[s] = nd; cut = True
                    if not nd: return False
        if cut:
            if not settle(eng, dom, locked): return False
        else:
            return True


def count_search(eng, dom, locked, settler, budget):
    cnt = [0]
    def rec(dom, locked):
        cnt[0] += 1
        if cnt[0] > budget: return 'BUDGET'
        d = copy.deepcopy(dom); lk = dict(locked)
        if not settler(eng, d, lk): return None
        for ei, e in enumerate(eng.p.examples, 1):
            if e.op in eng.fam and e.op not in lk and all(len(d[s]) == 1 for s in (e.d1, e.d2, e.d3, e.d4)):
                a, b, rstr, opc, zl, keep, excluded, used = eng._op_cands(e, d, lk)
                for vv, rv in keep:
                    r = rec(d, {**lk, e.op: vv})
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
            r = rec(nd, lk)
            if r == 'BUDGET': return 'BUDGET'
            if r: return r
        return None
    res = rec(dom, locked)
    return res, cnt[0]


def solve_count(name, settler, budget=2_000_000):
    res = get_engines_for(name)
    total = 0
    for item in res:
        if len(item) == 4 and item[3] == 'concat':
            return ('CONCAT', 0)
        mode, p, concat_results, sign_ops, cand, arith_ops, combos = item
        for combo in combos:
            fam = dict(zip(arith_ops, combo))
            eng = Engine(p, mode, set(p.op_syms), fam, concat_results, sign_ops)
            dom = {s: (set(range(1, 10)) if s in eng.lead else set(range(10))) for s in p.digit_syms}
            r, c = count_search(eng, dom, {}, settler, budget - total)
            total += c
            if r == 'BUDGET': return ('BUDGET', total)
            if r: return (r[0], total)
    return (None, total)


def main():
    base = json.load(open('_baseline_fresh.json'))
    # borderline over-cap puzzles with an R3-fireable example, tok in (8192, 60000]
    cands = []
    for name, v in base.items():
        t = v.get('tok', 0)
        if not (8192 < t <= 60000): continue
        p = cg.parse(name + '/question.txt'); op_set = set(p.op_syms)
        fire = any(len([c for c in e.rhs if c not in op_set]) == 4 and len({e.d1,e.d2,e.d3,e.d4}) == 4
                   for e in p.examples)
        if fire: cands.append((name, t))
    cands.sort(key=lambda x: x[1])
    if len(sys.argv) > 1: cands = cands[:int(sys.argv[1])]
    print(f"borderline R3-fireable over-cap puzzles (8192<tok<=60000): {len(cands)}")
    same_ans = 0; diff_ans = 0; budget_hit = 0
    reductions = []
    rows = []
    for name, tok in cands:
        ba, bn = solve_count(name, settle)
        ra, rn = solve_count(name, settle_r3)
        if ba == 'BUDGET' or ra == 'BUDGET':
            budget_hit += 1; rows.append((name, tok, bn, rn, 'BUDGET')); continue
        ansok = (ba == ra)
        if ansok: same_ans += 1
        else: diff_ans += 1
        red = (bn - rn) / bn if bn else 0
        reductions.append(red)
        rows.append((name, tok, bn, rn, 'same' if ansok else 'DIFF'))
    print(f"answer same={same_ans} diff={diff_ans} budget_hit={budget_hit}")
    if reductions:
        import statistics
        print(f"node-count reduction: mean={statistics.mean(reductions)*100:.1f}% median={statistics.median(reductions)*100:.1f}% max={max(reductions)*100:.1f}%")
    # estimate token cap crossings: assume tokens scale ~linearly with nodes (rough). New_tok ~ tok*(rn/bn).
    crossed = []
    for name, tok, bn, rn, st in rows:
        if st in ('BUDGET',) or bn == 0: continue
        new_tok = tok * (rn / bn)
        if tok > 8192 and new_tok <= 8192:
            crossed.append((name, tok, round(new_tok), st))
    print(f"estimated cap crossings (tok*nodes-ratio <= 8192): {len(crossed)}")
    for r in crossed[:30]: print("   ", r)
    # also show the diff-answer ones
    diffs = [r for r in rows if r[4] == 'DIFF']
    print(f"answer-changing puzzles: {len(diffs)}")
    for r in diffs[:20]: print("   DIFF", r)


if __name__ == '__main__':
    main()
