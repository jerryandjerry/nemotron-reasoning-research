#!/usr/bin/env python3
"""Full-corpus check of the honest solver: every solved puzzle must yield a VALID solution
(distinct digits, leading-nonzero respected by construction, every example satisfied by the
chosen fixed variant). Reports invalids, unsolvable counts, and CoT length distribution."""
import glob, time, sys
import _honest_solver as hs, cot_generator as cg

def valid(p, A, variants, mode):
    op_set = set(p.op_syms)
    if len(set(A.values())) != len(A): return "dup-digit"
    for op in p.by_op:
        if op not in variants: return f"no-variant-for-{op}"
        for e in p.by_op[op]:
            a = 10*A[e.d1]+A[e.d2] if mode == 'standard' else 10*A[e.d2]+A[e.d1]
            b = 10*A[e.d3]+A[e.d4] if mode == 'standard' else 10*A[e.d4]+A[e.d3]
            if cg.apply_op(variants[op], a, b) != cg.rhs_int(e.rhs, A, mode, op_set):
                return f"ex-fail-{op}"
    return None

names = sorted(d.split('/')[0] for d in glob.glob('arithmetic_*/question.txt') + glob.glob('little_endian_*/question.txt'))
invalid = []; nosol = []; lens = []; t0 = time.time()
for i, name in enumerate(names):
    try:
        cot, A, variants, mode = hs.gen(name)
    except Exception as e:
        invalid.append((name, f"ERR {e}")); continue
    lens.append(len(cot))
    if A is None:
        nosol.append(name); continue
    r = valid(cg.parse(name+'/question.txt'), A, variants, mode)
    if r: invalid.append((name, r))
    if (i+1) % 50 == 0: print(f"  ...{i+1}/{len(names)}  elapsed {time.time()-t0:.0f}s", flush=True)
lens.sort()
def pct(q): return lens[min(len(lens)-1, q*len(lens)//100)] if lens else 0
print(f"\nchecked {len(names)} | solved {len(names)-len(nosol)-len([x for x in invalid if x[1].startswith('ERR')])} | "
      f"unsolvable(no arith soln) {len(nosol)} | INVALID {len(invalid)}")
print("INVALID list:", invalid[:30])
print("CoT chars:", {f"p{q}": pct(q) for q in (50,75,90,95,99)}, "max", lens[-1] if lens else 0)
print(f"total elapsed {time.time()-t0:.0f}s")
