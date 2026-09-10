#!/usr/bin/env python3
"""Fire-counting: instrument the generator's propagate to count NEW prunings each proposed rule
would make AFTER the current propagation settles (only new prunings count).

We monkeypatch Engine.propagate to call the original, then at its fixpoint compute the proposed
rule's keep-sets and tally any value that is still in the settled domain but would be removed.

Three rules:
  R1 combined_linear_bound_add  : ~add and locked signed-sub (sub_signed/rsub_signed) net linear bound
  R2 shared_symbol_doubled_coeff: subset of R1 firing ONLY where a symbol's net coeff came from >=2
                                   occurrences being summed (doubled-operand or op+rhs cancellation).
                                   We tag a fire as "R2-relevant" if the firing symbol has >1 raw
                                   occurrence in the example (i.e. the per-symbol narrow could not see
                                   the collapsed coefficient).
  R3 equal_operands_sub_near_zero: ~sub example with identical operand pair + 1-digit result -> {0,1,2}.
"""
import sys, glob, copy, collections
from pathlib import Path
import cot_generator as cg
import _gen_crypt as G

DATA = Path('/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260525_Cryptarithm')

ADD_FAM = 'noisy_add'
SIGNED_SUB_VARS = {'sub_signed','sub_signed_p1','sub_signed_m1','sub_signed_p2','sub_signed_m2','rsub_signed'}

# ---- net coefficient builder (matches generator weight conventions) ----
def net_coeffs_add(e, mode, op_set):
    c = collections.defaultdict(int); occ = collections.Counter()
    for (d1,d2) in [(e.d1,e.d2),(e.d3,e.d4)]:
        t,u = (d1,d2) if mode=='standard' else (d2,d1)
        c[t]+=10; c[u]+=1; occ[t]+=1; occ[u]+=1
    ch=list(e.rhs); sign=1
    if ch and ch[0] in op_set: sign=-1; ch=ch[1:]
    elif ch and ch[-1] in op_set: sign=-1; ch=ch[:-1]
    n=len(ch); co=[10**(n-1-i) for i in range(n)] if mode=='standard' else [10**i for i in range(n)]
    for k,s in zip(co,ch):
        c[s]-=sign*k; occ[s]+=1
    return c, occ  # the net identity Sum c v = noise in [-2,2]

def net_coeffs_sub(e, mode, op_set, reverse):
    c=collections.defaultdict(int); occ=collections.Counter()
    op1,op2=(e.d1,e.d2),(e.d3,e.d4)
    if reverse: op1,op2=op2,op1
    for sgn,(d1,d2) in [(1,op1),(-1,op2)]:
        t,u=(d1,d2) if mode=='standard' else (d2,d1)
        c[t]+=sgn*10; c[u]+=sgn*1; occ[t]+=1; occ[u]+=1
    ch=list(e.rhs); rsign=1
    if ch and ch[0] in op_set: rsign=-1; ch=ch[1:]
    elif ch and ch[-1] in op_set: rsign=-1; ch=ch[:-1]
    n=len(ch); co=[10**(n-1-i) for i in range(n)] if mode=='standard' else [10**i for i in range(n)]
    for k,s in zip(co,ch):
        c[s]-=rsign*k; occ[s]+=1
    return c, occ

def linear_keep(coeffs, dom, s, K=2):
    """Keep {v in dom[s]: -K <= c_s*v + sum_{t!=s} c_t*[interval] window holds}.
       i.e. c_s*v in [-K - hi_other, K - lo_other] where other contributes
       [sum c_t min, sum c_t max] (interval relaxation)."""
    cs = coeffs[s]
    if cs == 0: return set(dom[s])  # symbol cancels out -> no info
    lo_other = hi_other = 0
    for t,ct in coeffs.items():
        if t==s: continue
        vals = dom[t]
        mn,mx = min(vals),max(vals)
        if ct>=0: lo_other += ct*mn; hi_other += ct*mx
        else:     lo_other += ct*mx; hi_other += ct*mn
    # Sum c v in [-K,K]  =>  c_s v in [-K - hi_other, K - lo_other]
    lo = -K - hi_other; hi = K - lo_other
    keep=set()
    for v in dom[s]:
        if lo <= cs*v <= hi: keep.add(v)
    return keep

# globals for tallying
def newtally():
    return {'R1_fires':0,'R1_vals':0,'R1_puzzles':set(),
            'R2_fires':0,'R2_vals':0,'R2_puzzles':set(),
            'R3_fires':0,'R3_vals':0,'R3_puzzles':set(),
            'R1_singleton_collapse':0,'R3_singleton_collapse':0}
TALLY = newtally()        # all propagate calls (every search node)
TALLY_PRE = newtally()    # ONLY the first propagate call per Engine.search invocation at top reading level (pre-branch)
CURNAME=[None]
# track how many propagate calls have happened for the current Engine instance during the current
# search recursion; the rule's pre-branch value is the FIRST settle before any branching at depth 0.
PRE_FLAG=[True]   # set True at the start of each reading; first propagate after that is "pre-branch"

orig_propagate = G.Engine.propagate

def tally_into(T, self, dom, locked):
    name=CURNAME[0]; op_set=self.op_set; mode=self.mode
    # R1 / R2: linear bound on ~add and locked signed-sub
    for e in self.p.examples:
        if e.op not in self.fam: continue
        fam=self.fam.get(e.op)
        coeffs=occ=None
        if fam==ADD_FAM:
            coeffs,occ=net_coeffs_add(e,mode,op_set)
        elif e.op in locked and locked[e.op] in SIGNED_SUB_VARS:
            rev = locked[e.op]=='rsub_signed'
            coeffs,occ=net_coeffs_sub(e,mode,op_set,rev)
        if coeffs is None: continue
        esyms=[s for s in coeffs if len(dom[s])>1]
        for s in esyms:
            keep=linear_keep(coeffs,dom,s)
            removed=set(dom[s])-keep
            if removed:
                T['R1_fires']+=1; T['R1_vals']+=len(removed); T['R1_puzzles'].add(name)
                if len(keep)==1: T['R1_singleton_collapse']+=1
                if occ[s]>1:
                    T['R2_fires']+=1; T['R2_vals']+=len(removed); T['R2_puzzles'].add(name)
    # R3: equal-operand 1-digit ~sub
    for e in self.p.examples:
        if self.fam.get(e.op)!='noisy_subtraction': continue
        if (e.d1,e.d2)!=(e.d3,e.d4): continue
        rch=[c for c in e.rhs if c not in op_set]
        if len(rch)!=1: continue
        r=rch[0]
        if len(dom[r])<=1: continue
        allowed={0,1,2}
        if r in self.lead: allowed={1,2}
        keep=set(dom[r])&allowed
        removed=set(dom[r])-keep
        if removed:
            T['R3_fires']+=1; T['R3_vals']+=len(removed); T['R3_puzzles'].add(name)
            if len(keep)==1: T['R3_singleton_collapse']+=1

def patched_propagate(self, dom, locked, emit):
    ok = orig_propagate(self, dom, locked, emit)
    if not ok: return ok
    tally_into(TALLY, self, dom, locked)
    if PRE_FLAG[0]:
        tally_into(TALLY_PRE, self, dom, locked)
        PRE_FLAG[0]=False     # only the first settled propagate of a reading counts as pre-branch
    return ok

G.Engine.propagate = patched_propagate

# PRE = the first propagate of each depth-0 search call (the deterministic §N.4 narrowing the
# reader sees before any branching, for each operator-family combo).
orig_search = G.Engine.search
def patched_search(self, dom, locked, emit, depth=0):
    if depth==0: PRE_FLAG[0]=True
    return orig_search(self, dom, locked, emit, depth)
G.Engine.search = patched_search

def main():
    dirs = sorted([Path(p).parent.name for p in glob.glob(str(DATA/'*/question.txt'))])
    done=0; errs=0
    for name in dirs:
        CURNAME[0]=name
        try:
            G.gen_cot(name)
            done+=1
        except Exception as ex:
            errs+=1
    print(f"puzzles run: {done}  errors: {errs}")
    print("=== ALL propagate calls (every search node) ===")
    for r in ['R1','R2','R3']:
        print(f"  {r}: fires(symbol-events)={TALLY[r+'_fires']}  vals_removed={TALLY[r+'_vals']}  "
              f"puzzles_touched={len(TALLY[r+'_puzzles'])}")
    print(f"  R1 singleton-collapse fires: {TALLY['R1_singleton_collapse']}")
    print(f"  R3 singleton-collapse fires: {TALLY['R3_singleton_collapse']}")
    print("=== PRE-BRANCH only (first settle of each depth-0 search; deterministic §N.4) ===")
    for r in ['R1','R2','R3']:
        print(f"  {r}: fires={TALLY_PRE[r+'_fires']}  vals_removed={TALLY_PRE[r+'_vals']}  "
              f"puzzles_touched={len(TALLY_PRE[r+'_puzzles'])}")
    print(f"  R1 singleton-collapse fires (pre): {TALLY_PRE['R1_singleton_collapse']}")
    print(f"  R3 singleton-collapse fires (pre): {TALLY_PRE['R3_singleton_collapse']}")

if __name__=='__main__':
    main()
