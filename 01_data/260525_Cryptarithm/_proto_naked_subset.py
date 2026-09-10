#!/usr/bin/env python3
"""Prototype: monkeypatch Engine.propagate to ADD generalized naked-subset (Hall-set) elimination
k=3,4 + a matching-feasibility contradiction check, inside the SAME fixpoint loop, right after the
existing PAIRS block. Then measure token deltas + answer-equality vs baseline on a set of puzzles.

SOUNDNESS: pure all-different consequences. If k symbols have a combined domain of exactly k digits,
those k digits are used up by those k symbols (a sub-bijection forced by all-different + pigeonhole),
so no OTHER symbol can take any of them. Independent of the ±2 operator noise (noise only shaped the
domains; the elimination uses only the all-different constraint over whatever domains exist)."""
import copy, sys, json
from itertools import combinations
import cot_generator as cg
import _gen_crypt as gc

def matching_feasible(syms, dom):
    matchR={}
    def kuhn(s,seen,adj):
        for v in adj[s]:
            if v in seen: continue
            seen.add(v)
            if v not in matchR or kuhn(matchR[v],seen,adj):
                matchR[v]=s; return True
        return False
    adj={s:sorted(dom[s]) for s in syms}
    m=0
    for s in syms:
        if kuhn(s,set(),adj): m+=1
    return m==len(syms)

ORIG = gc.Engine.propagate

def patched_propagate(self, dom, locked, emit):
    changed = True
    while changed:
        changed = False
        # ---- existing fixpoint body (call original ONE pass via a trick) ----
        # We re-implement by calling ORIG but ORIG runs its OWN while-loop to fixpoint;
        # to interleave we instead run ORIG to fixpoint, THEN apply our extra rules, THEN
        # loop again if our rules changed something. This is correct because ORIG is monotone.
        ok = ORIG(self, dom, locked, emit)
        if ok is False: return False
        # ---- our extra all-different rules ----
        # generalized naked subset k=3,4 (k=2 already handled by PAIRS)
        did = False
        small = [s for s in self.syms if 2 <= len(dom[s]) <= 4]
        for k in (3,4):
            done=False
            for combo in combinations(small, k):
                union = set().union(*(dom[s] for s in combo))
                if len(union) < k:
                    emit(f"All-different: {', '.join(self.lm[s] for s in combo)} together can only use {{{','.join(map(str,sorted(union)))}}} — {k} symbols but {len(union)} digits, impossible — no.")
                    return False
                if len(union) == k:
                    victims = [t for t in self.syms if t not in combo and (dom[t] & union)]
                    if victims:
                        for t in victims:
                            dom[t] = dom[t] - union
                            if not dom[t]:
                                emit(f"All-different: {self.lm[t]} loses every value to the {{{','.join(map(str,sorted(union)))}}} block — no.")
                                return False
                        emit(f"Naked {k}-set: {', '.join(self.lm[s] for s in combo)} use up {{{','.join(map(str,sorted(union)))}}}, remove these from {', '.join(self.lm[t] for t in victims)}.")
                        changed=True; did=True; done=True; break
            if done: break
        # matching-feasibility contradiction (Régin GAC, value-graph perfect matching)
        if not did:
            if not matching_feasible(self.syms, dom):
                emit("All-different: no way to give every symbol a distinct digit from its remaining candidates — no.")
                return False
        if not changed:
            break
    return True

def boxed(cot):
    i=cot.rfind('\\boxed{');  j=cot.find('}',i) if i>=0 else -1
    return cot[i+7:j] if i>=0 else None

if __name__=='__main__':
    from _probe_fixpoint import all_dirs
    from tokenizers import Tokenizer
    TOK=Tokenizer.from_file('/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/02_train/260512_huikang_085/repo/tokenizer.json')
    BASE=json.load(open('_baseline_rows.json'))
    names=sys.argv[1:] or sorted([k for k,v in BASE.items() if v.get('tok',0)>8192])
    gc.Engine.propagate = patched_propagate
    mism=[]; rows={}
    for name in names:
        try:
            cot=gc.gen_cot(name)
        except Exception as e:
            rows[name]={'err':str(e)[:80]}; continue
        ans=boxed(cot); tok=len(TOK.encode(cot).ids)
        base=BASE.get(name,{})
        rows[name]={'tok':tok,'base_tok':base.get('tok'),'ans':ans,'base_ans':base.get('ans')}
        if base.get('ans') is not None and ans!=base.get('ans'):
            mism.append(name)
    gc.Engine.propagate = ORIG
    deltas=[(name,r['base_tok']-r['tok']) for name,r in rows.items() if 'tok' in r and r['base_tok']]
    saved=[d for _,d in deltas if d>0]
    improved=sum(1 for _,d in deltas if d>0)
    now_under=sum(1 for name,r in rows.items() if 'tok' in r and r['base_tok']>8192 and r['tok']<=8192)
    print(f"puzzles tested: {len(names)}")
    print(f"answer mismatches: {len(mism)} -> {mism[:20]}")
    print(f"improved (fewer tokens): {improved}")
    print(f"total tokens saved: {sum(saved)}")
    print(f"newly under 8192: {now_under}")
    top=sorted(deltas,key=lambda x:-x[1])[:15]
    print("top savings:")
    for name,d in top: print(f"  {name}: -{d}  ({rows[name]['base_tok']} -> {rows[name]['tok']})")
