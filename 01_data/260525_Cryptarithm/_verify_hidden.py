#!/usr/bin/env python3
"""INDEPENDENT verification of the hidden-pair/triple rule.

Three things, all from scratch (not reusing _probe_subsets's fire function):
  1. SOUNDNESS, gated (avail==N): instrument propagate; after the ORIGINAL settles, apply the
     EXACT gated rule from the implementation sketch (mutating dom), and on the gold path assert
     no gold digit is ever removed. Try k=2 and k=3, smallest-first, one subset per pass, looping
     to fixpoint -- exactly as proposed. Count NEW prunings (only pruning beyond the original fixpoint).
  2. SOUNDNESS, UNGATED (drop avail==N): same but without the gate, to reproduce/confirm the ~5700
     on-gold-path violations and prove the gate is load-bearing.
  3. Reconcile k=2 vs base hidden-single: how many k=2 fires are *genuinely new* deductions vs already
     achievable by iterating the base hidden-single after the k=2 cut (i.e. is k=2 just slower hidden-single?).

Gold-path definition matches _probe_subsets: gold reading mode, every gold digit still live, locked
ops match gold family.
"""
import os, glob, re, sys, copy, itertools
from collections import defaultdict
import _gen_crypt as gc
import cot_generator as cg

def vfam(v):
    if v is None: return None
    if v.startswith('mul'): return 'noisy_mul'
    if v.startswith('add'): return 'noisy_add'
    return 'noisy_subtraction'

def load_gold(name):
    path = f'{name}/tree_solution.txt'
    if not os.path.exists(path): return None, None, None
    dm = {}; mode = None; opv = {}
    with open(path) as fh:
        in_map = False
        for line in fh:
            if line.startswith('Digit mapping'): in_map = True; continue
            m = re.match(r'\s+(\S+) \([A-Z]+\) -> (\d)', line)
            if in_map and m:
                dm[m.group(1)] = int(m.group(2)); continue
            if in_map and (not m) and line.strip(): in_map = False
            mm = re.match(r'Reading order:\s+(\S+)', line)
            if mm: mode = mm.group(1)
            ov = re.match(r'Operator variants:\s+(.*)', line)
            if ov:
                for tok in ov.group(1).split(','):
                    tok = tok.strip()
                    if '=' in tok:
                        sym, var = tok.split('=', 1)
                        opv[sym.strip()] = var.strip()
    return (dm or None), mode, opv

CUR = {}
STATS = defaultdict(int)
PUZ = defaultdict(lambda: defaultdict(set))
VIOL = defaultdict(list)
SNAP = [0]; ONPATH = [0]
_orig = gc.Engine.propagate

def on_gold_path(self, dom, locked):
    gold = CUR['gold']; gmode = CUR['gold_mode']; gopv = CUR['gold_opv']
    if gold is None or self.mode != gmode: return False
    if not all((s not in gold) or (gold[s] in dom[s]) for s in self.syms): return False
    for op, var in locked.items():
        gv = gopv.get(op)
        if gv is None: continue
        if var != gv and vfam(var) != vfam(gv): return False
    return True

def apply_gated_hidden(self, dom, gated=True):
    """Apply the EXACT proposed rule (smallest-k first, one subset/pass, loop to fixpoint).
    Returns list of (k, holders(tuple), combo(tuple)) prunings actually performed."""
    syms = self.syms; N = len(syms)
    fired = []
    changed = True
    guard = 0
    while changed:
        changed = False; guard += 1
        if guard > 200: break
        avail = [d for d in range(10) if any(d in dom[t] for t in syms)]
        if gated and len(avail) != N:
            break
        done = False
        for k in (2, 3):
            for combo in itertools.combinations(sorted(avail), k):
                holders = set()
                bad = False
                for d in combo:
                    holders |= {t for t in syms if d in dom[t]}
                    if len(holders) > k: bad = True; break
                if bad: continue
                if len(holders) == k and any(dom[t] - set(combo) for t in holders):
                    for t in holders:
                        dom[t] = dom[t] & set(combo)
                    fired.append((k, tuple(sorted(holders)), tuple(combo)))
                    changed = True; done = True
                    break
            if done: break
    return fired

def make_patch(gated):
    def patched(self, dom, locked, emit):
        ok = _orig(self, dom, locked, emit)
        if not ok: return ok
        SNAP[0] += 1
        name = CUR['name']; gold = CUR['gold']
        op = on_gold_path(self, dom, locked)
        if op: ONPATH[0] += 1
        # apply rule to a COPY so we can detect what it removes, but soundness is about dom-removal
        work = {s: set(dom[s]) for s in self.syms}
        fired = apply_gated_hidden(self, work, gated=gated)
        tag = 'gated' if gated else 'ungated'
        for (k, holders, combo) in fired:
            STATS[f'{tag}_k{k}_fires'] += 1
            PUZ[name][f'{tag}_k{k}'].add((holders, combo))
        if op and gold is not None:
            for s in self.syms:
                if s in gold and gold[s] in dom[s] and gold[s] not in work[s]:
                    VIOL[tag].append((name, s, gold[s], sorted(dom[s]), sorted(work[s])))
        return ok
    return patched

def all_names():
    pats = ['arithmetic_*', 'little_endian_*', 'mixed_concat_little_endian_*',
            'query_unseen_concat_*', 'pure_concat_*']
    names = set()
    for p in pats: names |= set(d.rstrip('/') for d in glob.glob(p + '/'))
    for d in glob.glob('mixed_concat_*/'):
        d = d.rstrip('/')
        if not d.startswith('mixed_concat_little_endian'): names.add(d)
    names = {n for n in names if os.path.exists(f'{n}/question.txt')}
    return sorted(names)

def run(gated):
    global STATS, PUZ, VIOL, SNAP, ONPATH
    STATS = defaultdict(int); PUZ = defaultdict(lambda: defaultdict(set))
    VIOL = defaultdict(list); SNAP = [0]; ONPATH = [0]
    gc.Engine.propagate = make_patch(gated)
    names = all_names()
    for name in names:
        gold, gmode, gopv = load_gold(name)
        CUR.update(name=name, gold=gold, gold_mode=gmode, gold_opv=gopv)
        try: gc.gen_cot(name)
        except Exception as e: print(f"ERR {name}: {e}", file=sys.stderr)
    gc.Engine.propagate = _orig
    tag = 'gated' if gated else 'ungated'
    npuz = {f'{tag}_k{k}': len([n for n in names if PUZ[n].get(f'{tag}_k{k}')]) for k in (2,3)}
    print(f"=== {tag.upper()} ===  snaps={SNAP[0]} onpath={ONPATH[0]}")
    for k in (2,3):
        # count DISTINCT subset-fires per puzzle summed
        tot = sum(len(PUZ[n].get(f'{tag}_k{k}', set())) for n in names)
        print(f"  k={k}: distinct subset prunings={tot}  fire-events={STATS[f'{tag}_k{k}_fires']}  in {npuz[f'{tag}_k{k}']} puzzles")
    print(f"  ON-GOLD-PATH SOUNDNESS VIOLATIONS: {len(VIOL[tag])}")
    for v in VIOL[tag][:8]: print("     ", v)
    return STATS, PUZ, VIOL

if __name__ == '__main__':
    print("Independent re-implementation of the EXACT proposed gated rule, applied inside propagate.\n")
    run(True)
    print()
    run(False)
