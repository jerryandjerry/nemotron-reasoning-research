#!/usr/bin/env python3
"""For every matching-INFEASIBLE branch snapshot, find the SMALLEST Hall-violator set:
a set S of symbols with |union(domains)| < |S| (deficient set). By Hall's theorem such a
set EXISTS iff no perfect matching exists. We want to confirm it's always SMALL (narratable)
and report its size distribution. Also check: is the deficient set found by examining
symbols with the smallest domains (so the narration is short)?"""
import copy, sys, json
from itertools import combinations
import cot_generator as cg
import _gen_crypt as gc
from _probe_fixpoint import all_dirs, matching_filter
from _probe_thread import collect

BASE=json.load(open('_baseline_rows.json'))
OVER={k for k,v in BASE.items() if v.get('tok',0)>8192}

def smallest_deficient(eng,dom,maxk=6):
    """Find smallest S (|union|<|S|). Search increasing k over symbols sorted by domain size."""
    syms=sorted(eng.syms,key=lambda s:len(dom[s]))
    for k in range(2,maxk+1):
        # restrict to symbols whose domain <= k (a deficient set of size k needs union<k, so each domain<k)
        pool=[s for s in syms if len(dom[s])<k]
        for combo in combinations(pool,k):
            union=set().union(*(dom[s] for s in combo))
            if len(union)<k:
                return k, combo, union
    return None

if __name__=='__main__':
    names=sorted(OVER)
    sizes={}
    infeasible_total=0; witness_found=0; big=[]
    for name in names:
        for (eng,dom,lk) in collect(name,cap=200):
            if matching_filter(eng,dom): continue
            infeasible_total+=1
            w=smallest_deficient(eng,dom,maxk=8)
            if w:
                witness_found+=1; sizes[w[0]]=sizes.get(w[0],0)+1
            else:
                big.append(name)
    print(f"infeasible nodes={infeasible_total}  with small Hall witness (k<=8)={witness_found}")
    print("witness size distribution:", dict(sorted(sizes.items())))
    if big: print("no small witness:", big[:10])
