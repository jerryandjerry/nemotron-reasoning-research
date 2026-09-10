#!/usr/bin/env python3
"""Empirical verification of the 'singleton-cascade' thread rules.

Two new all-different rules, inserted INSIDE Engine.propagate's `while changed` loop and
cascading (changed=True so a pin re-triggers interval + naked/hidden single):
  R1 naked_pair_alldiff : two symbols with identical 2-elt domain {a,b} confine a,b -> drop from others;
                          >2 holders of the same 2-set => contradiction (Hall k>2).
  R2 hall_violation     : any k>=3 symbols (each |dom|<=k) whose union has < k digits => dead node.

We measure NEW fires only: the new block runs only when the EXISTING rules have settled in a pass
(no change made by interval/naked/hidden/leadpair in that pass). Any cut it then makes is genuinely
new (not subsumed by what propagate already does).

Outputs:
  - per-rule NEW fire counts, branch-point avoidances (a fire that pins a symbol), contradictions
  - full-CoT regression: boxed answer + Summary line equality vs ORIG, token deltas, newly-under-8192
"""
import copy, sys, json, glob, os
from itertools import combinations
import cot_generator as cg
import _gen_crypt as gc
from _probe_fixpoint import all_dirs

ORIG = gc.Engine.propagate

# ---- global counters (reset per run) ----
CNT = {}
def reset():
    CNT.clear()
    CNT.update(dict(
        r1_fire=0, r1_pin=0, r1_contra=0,        # naked-pair: value-removal fires, fires that pin a symbol, >2-holder contradictions
        r1_branchpts=0,                          # branch points at which r1 fired at least once
        r2_fire=0,                               # hall k>=3 contradiction fires
        r2_branchpts=0,
        any_branchpts=0,                         # branch points where EITHER rule fired
        passes=0,
    ))

# helper: does a fresh interval/naked/hidden/leadpair pass change anything? (used to know "settled")
def _existing_settles_check(self, dom, locked):
    """Return True if a single pass of the EXISTING rules would change dom (i.e. NOT settled)."""
    # naked-single removal
    singles = {next(iter(dom[s])): s for s in self.syms if len(dom[s]) == 1}
    for s in self.syms:
        if len(dom[s]) > 1 and (set(dom[s]) & set(singles)):
            return True
    # hidden single (available==N)
    avail = [d for d in range(10) if any(d in dom[t] for t in self.syms)]
    N = len(self.syms)
    if len(avail) < N:
        return True
    if len(avail) == N:
        for d in avail:
            holders = [t for t in self.syms if d in dom[t]]
            if len(holders) == 1 and len(dom[holders[0]]) > 1:
                return True
    # interval narrowing
    for e in self.p.examples:
        if e.op not in self.fam: continue
        esyms = list(dict.fromkeys((e.d1, e.d2, e.d3, e.d4, *[c for c in e.rhs if c not in self.op_set])))
        for s in esyms:
            if len(dom[s]) > 1 and self.narrow(s, e, dom, locked) != dom[s]:
                return True
        # leadpair D
        if gc.LEADPAIR_D:
            for s, nd in self._leadpair_D(e, dom):
                if nd != dom[s]:
                    return True
    return False


def patched_propagate(self, dom, locked, emit):
    changed = True
    r1_fired_here = r2_fired_here = False
    while changed:
        changed = False
        CNT['passes'] += 1
        # ====== run ONE pass of the existing propagate-to-fixpoint ======
        # Call ORIG which runs the existing rules to their OWN fixpoint. ORIG is monotone, so this
        # is equivalent to interleaving for the EXISTING rules; we then add our extra rules and loop.
        ok = ORIG(self, dom, locked, emit)
        if ok is False:
            if r1_fired_here: CNT['r1_branchpts'] += 1
            if r2_fired_here: CNT['r2_branchpts'] += 1
            if r1_fired_here or r2_fired_here: CNT['any_branchpts'] += 1
            return False
        # ORIG has settled. Now NEW prunings are guaranteed not subsumed by existing rules.
        # ---- R1 naked-pair (k=2 Hall set) ----
        pairs = {}
        for s in self.syms:
            if len(dom[s]) == 2:
                pairs.setdefault(frozenset(dom[s]), []).append(s)
        did = False
        for ab, holders in pairs.items():
            if len(holders) > 2:                       # >2 symbols sharing a 2-set: pigeonhole contradiction
                CNT['r1_contra'] += 1; r1_fired_here = True
                emit(f"All-different: {', '.join(self.lm[s] for s in holders)} are all confined to {{{','.join(map(str,sorted(ab)))}}} — impossible, no.")
                if r1_fired_here: CNT['r1_branchpts'] += 1
                if r2_fired_here: CNT['r2_branchpts'] += 1
                CNT['any_branchpts'] += 1
                return False
            if len(holders) == 2:
                a, b = sorted(ab)
                victims = [t for t in self.syms if t not in holders and (dom[t] & ab)]
                if victims:
                    pin = False
                    for t in victims:
                        dom[t] = dom[t] - ab
                        if not dom[t]:
                            CNT['r1_fire'] += 1; r1_fired_here = True
                            emit(f"All-different: {self.lm[t]} loses every value to the {{{a},{b}}} pair — no.")
                            if r1_fired_here: CNT['r1_branchpts'] += 1
                            if r2_fired_here: CNT['r2_branchpts'] += 1
                            CNT['any_branchpts'] += 1
                            return False
                        if len(dom[t]) == 1: pin = True
                    CNT['r1_fire'] += 1; r1_fired_here = True
                    if pin: CNT['r1_pin'] += 1
                    emit(f"Naked pair: {self.lm[holders[0]]} and {self.lm[holders[1]]} can only be {{{a},{b}}}, remove from "
                         + ", ".join(self.lm[t] for t in victims) + ".")
                    changed = True; did = True
                    break
        if did:
            continue   # re-cascade existing rules with the new pins
        # ---- R2 Hall violation k>=3 (contradiction only; pure all-different) ----
        small = [s for s in self.syms if 2 <= len(dom[s]) <= 5]
        hall_hit = None
        for k in (3, 4, 5):
            for combo in combinations(small, k):
                u = set().union(*(dom[s] for s in combo))
                if len(u) < k:
                    hall_hit = (k, combo, u); break
            if hall_hit: break
        if hall_hit:
            k, combo, u = hall_hit
            CNT['r2_fire'] += 1; r2_fired_here = True
            emit(f"All-different: {', '.join(self.lm[s] for s in combo)} ({k} symbols) share only {{{','.join(map(str,sorted(u)))}}} — impossible, no.")
            if r1_fired_here: CNT['r1_branchpts'] += 1
            if r2_fired_here: CNT['r2_branchpts'] += 1
            CNT['any_branchpts'] += 1
            return False
        # nothing new fired this pass -> loop ends (changed stayed False)
    if r1_fired_here: CNT['r1_branchpts'] += 1
    if r2_fired_here: CNT['r2_branchpts'] += 1
    if r1_fired_here or r2_fired_here: CNT['any_branchpts'] += 1
    return True


def boxed(cot):
    i = cot.rfind('\\boxed{'); j = cot.find('}', i) if i >= 0 else -1
    return cot[i+7:j] if i >= 0 else None

def summary_line(cot):
    for ln in cot.splitlines():
        if ln.startswith('Summary:'): return ln
    return None

if __name__ == '__main__':
    from tokenizers import Tokenizer
    TOK = Tokenizer.from_file('/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/02_train/260512_huikang_085/repo/tokenizer.json')
    names = sys.argv[1:] or all_dirs()

    # ---- baseline (ORIG) ----
    base = {}
    for name in names:
        try:
            cot = ORIG_COT = gc.gen_cot(name)
            base[name] = dict(tok=len(TOK.encode(cot).ids), ans=boxed(cot), summ=summary_line(cot))
        except Exception as e:
            base[name] = dict(err=str(e)[:120])

    # ---- patched (ORIG + R1 + R2) ----
    reset()
    gc.Engine.propagate = patched_propagate
    rows = {}; ans_mismatch = []; summ_mismatch = []; errs = []
    for name in names:
        try:
            cot = gc.gen_cot(name)
        except Exception as e:
            rows[name] = dict(err=str(e)[:120]); errs.append(name); continue
        rows[name] = dict(tok=len(TOK.encode(cot).ids), ans=boxed(cot), summ=summary_line(cot))
        b = base.get(name, {})
        if 'err' in b: continue
        if rows[name]['ans'] != b['ans']: ans_mismatch.append((name, b['ans'], rows[name]['ans']))
        if rows[name]['summ'] != b['summ']: summ_mismatch.append(name)
    gc.Engine.propagate = ORIG

    # ---- token deltas ----
    deltas = []
    for name in names:
        b = base.get(name, {}); r = rows.get(name, {})
        if 'tok' in b and 'tok' in r:
            deltas.append((name, b['tok'] - r['tok'], b['tok'], r['tok']))
    saved = sum(d for _, d, _, _ in deltas if d > 0)
    grew = sum(1 for _, d, _, _ in deltas if d < 0)
    improved = sum(1 for _, d, _, _ in deltas if d > 0)
    newly_under = [(n, bt, rt) for n, d, bt, rt in deltas if bt > 8192 and rt <= 8192]
    pushed_over = [(n, bt, rt) for n, d, bt, rt in deltas if bt <= 8192 and rt > 8192]

    out = dict(
        puzzles=len(names),
        base_errs=sum(1 for v in base.values() if 'err' in v),
        patched_errs=len(errs),
        counters=dict(CNT),
        ans_mismatches=ans_mismatch,
        summ_mismatch_count=len(summ_mismatch),
        summ_mismatch_names=summ_mismatch[:30],
        improved=improved, grew=grew, tokens_saved=saved,
        newly_under_8192=newly_under,
        pushed_over_8192=pushed_over,
        base_over_8192=sum(1 for v in base.values() if v.get('tok',0) > 8192),
    )
    print(json.dumps(out, indent=2, default=str))
    print("\nTOP TOKEN SAVINGS:")
    for n, d, bt, rt in sorted(deltas, key=lambda x: -x[1])[:15]:
        if d > 0: print(f"  {n}: -{d} ({bt}->{rt})")
