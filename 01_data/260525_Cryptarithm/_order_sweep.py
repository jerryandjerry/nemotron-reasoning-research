#!/usr/bin/env python3
"""Sweep all 6 fixed arithmetic-family orders (no per-symbol prior) and measure GT-match + CoT length.
concat is §N.2-detected (not searched), so the search order is a permutation of [add, mul, sub]."""
import glob, os, re, math, signal, itertools
import cot_generator as cg, _gen_crypt as gx

dirs = [d for d in sorted(glob.glob('*_*')) if os.path.isdir(d) and os.path.exists(d+'/question.txt')]
ans = {d: open(d+'/answer.txt').read().strip() for d in dirs if os.path.exists(d+'/answer.txt')}

def ext(t):
    bs = list(re.finditer(r'\\boxed\{', t)); ms = []
    for i, m in enumerate(bs):
        s = m.end(); e = bs[i+1].start() if i+1 < len(bs) else len(t); seg = t[s:e]; lb = seg.rfind('}')
        ms.append(seg[:lb] if lb != -1 else seg)
    ne = [m.strip() for m in ms if m.strip()]; return ne[-1] if ne else (ms[-1].strip() if ms else '')
def ver(a, p):
    a, p = a.strip(), p.strip()
    if re.fullmatch(r'[01]+', a): return p.lower() == a.lower()
    try: return math.isclose(float(a), float(p), rel_tol=1e-2, abs_tol=1e-5)
    except Exception: return p.lower() == a.lower()

class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(TO()))
FAM = {'noisy_add':'add', 'noisy_mul':'mul', 'noisy_subtraction':'sub'}

def patch(perm):
    def fr(counts, prior=None):
        cand = set.intersection(*(cg.possible_families(d) for d in counts)) if counts else set(perm)
        if not cand: cand = set(perm)
        return [f for f in perm if f in cand]
    cg.family_ranking = fr
    cg.op_top_family = lambda s: perm[0]
    cg.op_seen_families = lambda s: list(perm) + ['concat']

print("order            GT/800   completed(timeout)   median_chars  <=16k_chars", flush=True)
best=[]
for perm in itertools.permutations(['noisy_add','noisy_mul','noisy_subtraction']):
    patch(perm); gt=comp=to=0; clen=[]
    for d in dirs:
        signal.alarm(5)
        try:
            c = gx.gen_cot(d); comp += 1; clen.append(len(c))
            if d in ans and ans[d]: gt += ver(ans[d], ext(c))
        except TO: to += 1
        finally: signal.alarm(0)
    clen.sort(); med = clen[len(clen)//2] if clen else 0
    name = '>'.join(FAM[f] for f in perm)
    u16 = sum(1 for x in clen if x <= 16000)
    best.append((gt, med, name))
    print(f"{name:14s}   {gt:4d}     {comp:4d} ({to:3d})           {med:7d}      {u16}", flush=True)
print(f"\nbest GT: {max(best)[2]}  |  shortest median: {min(best, key=lambda r:r[1])[2]}", flush=True)
