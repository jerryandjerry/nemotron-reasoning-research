#!/usr/bin/env python3
"""SOUNDNESS verification: at every collected branch snapshot, take the puzzle's TRUE
final assignment (from answer.txt / the solver's own solution) and confirm that none of
the candidate rules (naked-subset k>=3, hidden-subset, matching-infeasible, leadcount)
would EVER remove the true digit from a symbol's domain OR flag a true-consistent node
as infeasible. If a rule fired on a node that lies on the TRUE solution path and removed
a true value, that rule is UNSOUND. We reconstruct the true path by checking that the
snapshot's domains all still contain the true assignment."""
import copy, glob, os, sys, json
from itertools import combinations
import cot_generator as cg
import _gen_crypt as gc
from _probe_fixpoint import all_dirs, matching_filter
from _probe_thread import collect, naked_subset, hidden_subset, leadcount

def true_assign(name):
    """Recover the solver's solution: parse the boxed answer is not enough (it's symbols);
    instead re-run gen_cot via the Engine and capture the winning A. Simplest: parse the
    'Conclusion' / 'Summary' digit map from the CoT? Use the solver directly."""
    # Run the actual search once and grab the solution A (sym->digit) + mode.
    p = cg.parse(f'{name}/question.txt')
    # use cot_generator.build_tree which returns the solution mapping letter->digit
    from cot_generator import make_prefix, build_tree
    prefix, mode, concat_results, csub = make_prefix(p)
    tree, sol, opn = build_tree(p, mode, concat_results, csub)
    if sol is None and mode=='standard':
        tree2, sol, opn = build_tree(p,'little_endian',concat_results,csub); mode='little_endian'
    if sol is None: return None, None
    return {s:v for s,v in sol.items()}, mode  # sym->digit

def on_true_path(dom, true):
    return all((true[s] in dom[s]) for s in true if s in dom)

def check(name):
    true, tmode = true_assign(name)
    if true is None: return None
    snaps = collect(name, cap=200)
    viol = []
    for (eng,dom,lk) in snaps:
        # only snapshots whose mode matches AND that are on the true path can test soundness
        if eng.mode != tmode: continue
        if not on_true_path(dom, true): continue
        # 1) matching-infeasible must NOT fire on a true-path node
        if not matching_filter(eng,dom):
            viol.append(('C1_infeasible_on_true', name)); continue
        # 2) naked-subset must not remove a true value
        r=naked_subset(eng,dom)
        if r:
            k,act,combo,union=r
            victims=[t for t in eng.syms if t not in combo and (dom[t]&union)]
            for t in victims:
                if true.get(t) in union and (true.get(t) not in (dom[t]-union)):
                    viol.append(('C2_removed_true',name,eng.lm[t],true[t])); 
        # 3) hidden-subset
        h=hidden_subset(eng,dom)
        if h:
            k,act,combo,union=h
            # symbols in union confined to set(combo); if a true value of a union-symbol is NOT in combo -> unsound
            for s in union:
                if true.get(s) is not None and true[s] not in set(combo):
                    viol.append(('C3_removed_true',name,eng.lm[s],true[s]))
        # 4) leadcount
        l=leadcount(eng,dom)
        if l:
            k,tag,combo,unionv,victims=l
            for t in victims:
                if true.get(t) in unionv and true.get(t) not in (dom[t]-unionv):
                    viol.append(('C4_removed_true',name,eng.lm[t],true[t]))
    return viol

if __name__=='__main__':
    names=sys.argv[1:] or all_dirs()
    allv=[]
    for name in names:
        v=check(name)
        if v: allv+=v
    if not allv:
        print(f"SOUND: no rule removed a true value or flagged a true node infeasible across {len(names)} puzzles")
    else:
        print(f"VIOLATIONS: {len(allv)}")
        for x in allv[:40]: print(x)
