#!/usr/bin/env python3
"""For each puzzle: in the WINNING reading+combo, count how many complete solutions exist (cap at 2).
A puzzle with a UNIQUE solution is GT-safe under ANY variable/value reordering. Multi-solution puzzles
are pinned to the current order. We replicate the search but enumerate ALL solutions (no early return)
up to 2, under the FIRST family-combo that yields a solution (the one the current solver locks)."""
import copy, glob, os, sys, json
import cot_generator as cg
import _gen_crypt as gc
from _gen_crypt import Engine
from _probe_heavy import get_engines_for, PREFIXES
from itertools import product as iproduct

def all_dirs():
    out=[]
    for d in sorted(glob.glob('*/')):
        name=d.rstrip('/').split('/')[-1]
        if os.path.exists(os.path.join(d,'question.txt')) and any(name.startswith(p) for p in PREFIXES):
            out.append(name)
    return out

def count_solutions_engine(eng, dom, locked, cap=2, budget=3_000_000):
    found=[0]; cnt=[0]
    def rec(dom, locked):
        if found[0]>=cap: return
        cnt[0]+=1
        if cnt[0]>budget: return
        d=copy.deepcopy(dom); lk=dict(locked)
        if not eng.propagate(d, lk, lambda m:None): return
        for ei,e in enumerate(eng.p.examples,1):
            if e.op in eng.fam and e.op not in lk and all(len(d[s])==1 for s in (e.d1,e.d2,e.d3,e.d4)):
                a,b,rstr,opc,zl,keep,excluded,used=eng._op_cands(e,d,lk)
                for v,rv in keep:
                    rec(d, {**lk, e.op:v})
                    if found[0]>=cap: return
                return
        un=[s for s in eng.syms if len(d[s])>1]
        if not un:
            A={s:next(iter(d[s])) for s in eng.syms}
            if all(op in lk for op in eng.fam) and eng.verify(A,lk): found[0]+=1
            return
        s=eng.pick_branch(d,un)
        for val in sorted(d[s]):
            nd={t:({val} if t==s else (dt-{val} if len(dt)>1 else dt)) for t,dt in d.items()}
            rec(nd, lk)
            if found[0]>=cap: return
    rec(dom, locked)
    return found[0]

def solutions_for(name):
    res=get_engines_for(name)
    for item in res:
        if len(item)==4 and item[3]=='concat': return ('concat',1)
        mode,p,concat_results,sign_ops,cand,arith_ops,combos=item
        for combo in combos:
            fam=dict(zip(arith_ops,combo))
            eng=Engine(p,mode,set(p.op_syms),fam,concat_results,sign_ops)
            dom={s:(set(range(1,10)) if s in eng.lead else set(range(10))) for s in p.digit_syms}
            n=count_solutions_engine(eng,dom,{})
            if n>0: return (mode, n)
    return (None,0)

def main():
    names=all_dirs()
    uniq=0; multi=0; concat=0; none=0
    multi_names=[]
    for name in names:
        mode,n=solutions_for(name)
        if mode=='concat': concat+=1
        elif n==0: none+=1
        elif n==1: uniq+=1
        else: multi+=1; multi_names.append(name)
    print(f"total={len(names)}")
    print(f"  unique-solution (GT-safe under ANY reorder): {uniq}")
    print(f"  multi-solution  (must keep current order):   {multi}")
    print(f"  pure-concat short-circuit (no search):       {concat}")
    print(f"  no-solution-in-first-combo (odd):            {none}")
    json.dump(multi_names, open('_multi_solution_names.json','w'))
    print(f"  multi examples: {multi_names[:15]}")

if __name__=='__main__':
    main()
