#!/usr/bin/env python3
"""Implement a GUARDED, provably answer-preserving query early-stop by PATCHING _gen_crypt at runtime,
then A/B vs baseline for (1) answer changes, (2) token savings, (3) firing rate.

GUARD (all must hold to stop early in the CURRENT reading at the current partial assignment d, lk):
  G1 reading-confirmed: the current reading is the one the pipeline will accept REGARDLESS of the other
     reading. Sound cases:
       * trailing-sign present  -> standard is illegal, so we are in §2 and §2 is the ONLY legal reading. OK.
       * we are in §1 (standard) AND no trailing sign: we can stop early ONLY IF §1 would be ACCEPTED,
         which the pipeline decides by completeness. To keep this sound we DO complete §1 (no skip) UNLESS
         we additionally KNOW §1 wins. We DON'T, mid-search. So for §1 we additionally require the answer
         to be PROVABLY reading-invariant... which it is NOT in general.  => For G1 we ONLY allow stopping
         in the FORCED reading (trailing sign => §2), which is always accepted.
  G2 query answerable (existing _answer_ready): query operands pinned, query-op variant locked & verified
     against ALL its (now-pinned) examples, result-digit symbols pinned.
  G3 variant-unambiguous: among the query-op family's variants, EXACTLY ONE matches the (pinned) query-op
     examples; otherwise a later example could force a different variant -> different answer. (When the
     query op appears in >=1 fully-pinned example and _answer_ready verified it, ambiguity can only come
     from sibling variants that ALSO match every pinned example; we check none does.)
  G4 completable: a full all-different completion of the remaining unpinned symbols EXISTS such that every
     remaining example verifies. We check this by a bounded completion search (the same fill_remaining the
     full engine would do). If a completion exists, the reading SUCCEEDS with the SAME query-relevant
     content -> answer-preserving. If NOT, we must NOT stop (the branch is actually a dead-end).

We implement this as a new flag SOUND_QSTOP and route _answer_ready through the guard.
"""
import glob, os, json, copy
from itertools import permutations
import cot_generator as cg
import _gen_crypt as gc
from tokenizers import Tokenizer

TOK = Tokenizer.from_file('/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/02_train/260512_huikang_085/repo/tokenizer.json')
PREFIXES = ('arithmetic','little_endian','mixed_concat','pure_concat','query_unseen_concat','mixed_concat_little_endian')

def all_dirs():
    out = []
    for d in sorted(glob.glob('*/')):
        name = d.rstrip('/').split('/')[-1]
        if not os.path.exists(os.path.join(d,'question.txt')): continue
        if any(name.startswith(p) for p in PREFIXES): out.append(name)
    return out

def boxed(cot):
    i = cot.rfind('\\boxed{'); j = cot.find('}', i); return cot[i+7:j] if i>=0 else None

# ---- patch: a guarded _answer_ready ----
_orig_answer_ready = gc.Engine._answer_ready

def _has_trailing_sign(p, op_set):
    for e in p.examples:
        if e.rhs and e.rhs[-1] in op_set and e.rhs[0] not in op_set: return True
    return False

def _variant_unambiguous(self, d, lk):
    """G3: the query-op variant is the UNIQUE family variant matching every pinned query-op example."""
    qop = self.p.query[2]
    v = lk[qop]
    fam = self.fam.get(qop)
    if fam is None: return False
    core = gc.CORE[fam]
    A = {s: next(iter(d[s])) for s in self.syms if len(d[s]) == 1}
    matching = set()
    for e in self.p.examples:
        if e.op != qop: continue
        esyms = (e.d1,e.d2,e.d3,e.d4,*[c for c in e.rhs if c not in self.op_set])
        if not all(s in A for s in esyms): return False   # an example of the query op is not pinned -> can't be sure
        a,b = self.opval(e.d1,e.d2,A), self.opval(e.d3,e.d4,A)
        ev = cg.rhs_int(e.rhs, A, self.mode, self.op_set)
        ms = {var for var in gc.CORES[core] if cg.apply_op(var,a,b)==ev}
        matching = ms if not matching else (matching & ms)
    return matching == {v} or (len(matching)>=1 and all(cg.apply_op(m, *_qab(self,A))==cg.apply_op(v,*_qab(self,A)) for m in matching))

def _qab(self, A):
    q=self.p.query
    a = 10*A[q[0]]+A[q[1]] if self.mode=='standard' else 10*A[q[1]]+A[q[0]]
    b = 10*A[q[3]]+A[q[4]] if self.mode=='standard' else 10*A[q[4]]+A[q[3]]
    return a,b

def _completable(self, d, lk):
    """G4: does a full all-different completion exist that verifies every remaining example?"""
    syms = self.syms
    pinned = {s: next(iter(d[s])) for s in syms if len(d[s])==1}
    free = [s for s in syms if len(d[s])>1]
    used = set(pinned.values())
    avail = [x for x in range(10) if x not in used]
    # need an operator variant for any unlocked op that appears in examples; the full engine would lock them.
    # We accept completion if SOME assignment of free symbols (all-different) and SOME variant per unlocked
    # example-op makes every example hold.
    from cot_generator import CORES, CORE_FAMILY, apply_op, rhs_int
    if len(free) > 7:  # cap the brute force; if too many free, fall back to "not completable-known" => don't stop
        return None  # unknown
    for perm in permutations(avail, len(free)):
        A = dict(pinned); A.update(zip(free, perm))
        if any(A[s]==0 for s in self.lead): continue
        ok=True
        for e in self.p.examples:
            if e.op not in self.fam: continue
            a,b=self.opval(e.d1,e.d2,A),self.opval(e.d3,e.d4,A)
            ev=rhs_int(e.rhs,A,self.mode,self.op_set)
            if e.op in lk:
                if apply_op(lk[e.op],a,b)!=ev: ok=False;break
            else:
                core=gc.CORE[self.fam[e.op]]
                if not any(apply_op(v,a,b)==ev for v in CORES[core]): ok=False;break
        if ok: return True
    return False

def guarded_answer_ready(self, d, lk):
    base = _orig_answer_ready(self, d, lk)         # G2
    if base is None: return None
    op_set = self.op_set; p = self.p
    # G1: only fire in a FORCED reading. trailing sign => standard illegal => §2 is the only legal reading.
    if not (self.mode != 'standard' and _has_trailing_sign(p, op_set)):
        return None
    # G3
    if not _variant_unambiguous(self, d, lk): return None
    # G4
    comp = _completable(self, d, lk)
    if comp is not True: return None
    return base

def run(names, patched):
    if patched:
        gc.Engine._answer_ready = guarded_answer_ready
        gc.QUERY_MIN = True
    else:
        gc.Engine._answer_ready = _orig_answer_ready
        gc.QUERY_MIN = False
    rows={}
    for name in names:
        try:
            cot=gc.gen_cot(name); rows[name]={'tok':len(TOK.encode(cot).ids),'ans':boxed(cot)}
        except Exception as ex:
            rows[name]={'err':repr(ex)}
    return rows

if __name__=='__main__':
    names=all_dirs()
    base=run(names, False)
    var=run(names, True)
    diff=[(n,base[n].get('ans'),var[n].get('ans')) for n in names if base[n].get('ans')!=var[n].get('ans')]
    errs=[(n,base[n].get('err'),var[n].get('err')) for n in names if 'err' in base[n] or 'err' in var[n]]
    fired=[n for n in names if 'tok' in base[n] and 'tok' in var[n] and var[n]['tok']<base[n]['tok']]
    ob=sum(base[n]['tok']>8192 for n in names if 'tok' in base[n])
    ov=sum(var[n]['tok']>8192 for n in names if 'tok' in var[n])
    print(f"answers changed: {len(diff)} {diff[:10]}")
    print(f"errors: {len(errs)} {errs[:5]}")
    print(f"fired (shorter): {len(fired)}  {fired[:10]}")
    print(f"over8192: base={ob} var={ov} delta={ov-ob}")
    print(f"tok sum base={sum(base[n]['tok'] for n in names if 'tok' in base[n])} var={sum(var[n]['tok'] for n in names if 'tok' in var[n])}")
