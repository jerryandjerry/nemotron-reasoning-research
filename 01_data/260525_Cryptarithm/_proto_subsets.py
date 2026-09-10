#!/usr/bin/env python3
"""Prototype v2: generalized naked subsets (k=3,4,5) + hidden subsets (k=3,4, sound when available==N),
one action per pass (mirroring the existing PAIRS discipline), inside the propagate fixpoint loop.
Drops the matching check (redundant — every infeasible node is a naked-subset wipeout)."""
import copy, sys, json
from itertools import combinations
import cot_generator as cg
import _gen_crypt as gc

ORIG = gc.Engine.propagate

def patched_propagate(self, dom, locked, emit):
    changed=True
    while changed:
        changed=False
        ok=ORIG(self, dom, locked, emit)
        if ok is False: return False
        did=False
        # ---- generalized NAKED subset k=3,4,5 ----
        small=[s for s in self.syms if 2<=len(dom[s])<=5]
        for k in (3,4,5):
            done=False
            for combo in combinations(small,k):
                u=set().union(*(dom[s] for s in combo))
                if len(u)<k:
                    emit(f"All-different: {', '.join(self.lm[s] for s in combo)} together can only use {{{','.join(map(str,sorted(u)))}}} — {k} symbols, {len(u)} digits, impossible — no."); return False
                if len(u)==k:
                    vic=[t for t in self.syms if t not in combo and (dom[t]&u)]
                    if vic:
                        for t in vic:
                            dom[t]=dom[t]-u
                            if not dom[t]:
                                emit(f"All-different: {self.lm[t]} loses every value to the {{{','.join(map(str,sorted(u)))}}} block — no."); return False
                        emit(f"Naked {k}-set: {', '.join(self.lm[s] for s in combo)} use up {{{','.join(map(str,sorted(u)))}}}, remove these from {', '.join(self.lm[t] for t in vic)}.")
                        changed=True; did=True; done=True; break
            if done: break
        # ---- HIDDEN subset k=3,4 (sound when available==N) ----
        if not did:
            avail=[d for d in range(10) if any(d in dom[t] for t in self.syms)]
            if len(avail)==len(self.syms):
                holders={d:tuple(t for t in self.syms if d in dom[t]) for d in avail}
                done=False
                for k in (3,4):
                    for combo in combinations(avail,k):
                        un=set()
                        for d in combo: un|=set(holders[d])
                        if len(un)==k and any(dom[s]-set(combo) for s in un):
                            for s in un: dom[s]=dom[s]&set(combo)
                            emit(f"Hidden {k}-set: digits {', '.join(map(str,combo))} can only go to {', '.join(self.lm[s] for s in un)}, so those symbols are {{{','.join(map(str,combo))}}}.")
                            changed=True; did=True; done=True; break
                    if done: break
        if not changed: break
    return True

def boxed(cot):
    i=cot.rfind('\\boxed{'); j=cot.find('}',i) if i>=0 else -1
    return cot[i+7:j] if i>=0 else None

if __name__=='__main__':
    from tokenizers import Tokenizer
    TOK=Tokenizer.from_file('/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/02_train/260512_huikang_085/repo/tokenizer.json')
    BASE=json.load(open('_baseline_rows.json'))
    names=sys.argv[1:] or open('_affected_over_runnable.txt').read().split()
    gc.Engine.propagate=patched_propagate
    mism=[]; rows={}
    for name in names:
        try: cot=gc.gen_cot(name)
        except Exception as e: rows[name]={'err':str(e)[:80]}; continue
        ans=boxed(cot); tok=len(TOK.encode(cot).ids); base=BASE.get(name,{})
        rows[name]={'tok':tok,'base_tok':base.get('tok'),'ans':ans,'base_ans':base.get('ans')}
        if base.get('ans') is not None and ans!=base.get('ans'): mism.append(name)
    gc.Engine.propagate=ORIG
    deltas=[(n,r['base_tok']-r['tok']) for n,r in rows.items() if 'tok' in r and r['base_tok']]
    saved=sum(d for _,d in deltas if d>0)
    print(f"tested:{len(names)} mismatches:{len(mism)} {mism[:10]}")
    print(f"improved:{sum(1 for _,d in deltas if d>0)} tokens_saved:{saved} newly_under:{sum(1 for n,r in rows.items() if 'tok' in r and r['base_tok']>8192 and r['tok']<=8192)}")
    for n,d in sorted(deltas,key=lambda x:-x[1])[:12]: print(f"  {n}: -{d} ({rows[n]['base_tok']}->{rows[n]['tok']})")
