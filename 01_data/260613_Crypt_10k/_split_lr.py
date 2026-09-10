import sys, re, os, glob, statistics
sys.path.insert(0,'_cryptarithm_solver'); import _gen_crypt as gc
tok=lambda t: len(gc._TOK.encode(t, add_special_tokens=False).ids)
dirs=[d.rstrip('/') for d in sorted(glob.glob('*/')) if os.path.exists(os.path.join(d,'track','tree_cot.txt'))]
rows=[]
for d in dirs:
    cot=open(os.path.join(d,'track','tree_cot.txt')).read()
    m=re.search(r'\n§2 reading leftward', cot)
    if not m: continue                       # only puzzles that actually went to leftward
    pre, post = cot[:m.start()], cot[m.start():]
    rows.append((os.path.basename(d), tok(cot), tok(pre), tok(post), tok(cot)<7680))
print(f"=== {len(rows)} puzzles that fell through to §2 leftward ===")
def stats(name, xs): xs=sorted(xs); return f"{name}: median={int(statistics.median(xs))} min={min(xs)} max={max(xs)}"
s1=[r[2] for r in rows]; s2=[r[3] for r in rows]; tot=[r[1] for r in rows]
print(stats("total CoT", tot)); print(stats("§1 failed-rightward part", s1)); print(stats("§2 leftward part", s2))
frac=[100*r[2]/r[1] for r in rows]
print(f"\n§1 (failed rightward) is a median of {int(statistics.median(frac))}% of the whole leftward CoT (range {int(min(frac))}–{int(max(frac))}%)")
uc=[r for r in rows if r[4]]
print(f"\namong the {len(uc)} UNDER-CAP leftward puzzles:")
print("  " + stats("  §1 part", [r[2] for r in uc])); print("  " + stats("  §2 part", [r[3] for r in uc]))
print(f"  §1 median share: {int(statistics.median([100*r[2]/r[1] for r in uc]))}%")
