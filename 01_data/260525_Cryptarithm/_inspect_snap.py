#!/usr/bin/env python3
"""Inspect the FIRST matching-infeasible / hall3 snapshot of a puzzle: print domains + the rule's finding,
so we can hand-verify soundness and see what English narration it would produce."""
import sys, copy
import cot_generator as cg
import _gen_crypt as gc
from _probe_fixpoint import alldiff_gac_hall, hidden_subset, matching_filter
from _probe_fixpoint2 import run_probe_capped

def show(name, want='matching', cap=200):
    snaps = []
    run_probe_capped(name, snaps, cap)
    for i,(eng, dom, lk) in enumerate(snaps):
        feas = matching_filter(eng, dom)
        hall = alldiff_gac_hall(eng, dom)
        hsub = hidden_subset(eng, dom)
        flag = (want=='matching' and not feas) or (want=='hall' and hall) or (want=='hidden' and hsub)
        if flag:
            print(f"=== {name} snapshot #{i} (of {len(snaps)}) want={want} ===")
            avail = sorted({d for s in eng.syms for d in dom[s]})
            print(f"  symbols N={len(eng.syms)}, available digits ({len(avail)}): {avail}")
            for s in eng.syms:
                print(f"    {eng.lm[s]} = {sorted(dom[s])}")
            print(f"  matching feasible: {feas}")
            if hall: print(f"  HALL k={hall[0]}: digits {sorted(hall[2])} confined to {[eng.lm[s] for s in hall[1]]}; remove from {[eng.lm[s] for s in hall[3]]}")
            if hsub: print(f"  HIDDEN k={hsub[0]}: digits {sorted(hsub[1])} only fit {[eng.lm[s] for s in hsub[2]]}")
            return
    print(f"{name}: no {want} snapshot found in first {len(snaps)} branch states")

if __name__ == '__main__':
    want = sys.argv[1]
    for name in sys.argv[2:]:
        show(name, want)
