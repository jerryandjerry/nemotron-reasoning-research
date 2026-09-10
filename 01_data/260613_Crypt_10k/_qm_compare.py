import sys, re, os, glob, signal, time
sys.path.insert(0,'_cryptarithm_solver'); import _gen_crypt as gc
class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s,f:(_ for _ in ()).throw(TO()))
def box(t):
    bs=list(re.finditer(r'\\boxed\{',t))
    if not bs: return None
    m=bs[-1]; seg=t[m.end():]; lb=seg.rfind('}'); return (seg[:lb] if lb!=-1 else seg).strip()
dirs=[d.rstrip('/') for d in sorted(glob.glob('*/')) if os.path.exists(os.path.join(d,'track','tree_cot.txt'))]
chg=[]; gtT=0; gtF=0; to=[]; n=0; early=0; t0=time.time()
gc.QUERY_MIN=False
for d in dirs:
    cur=open(os.path.join(d,'track','tree_cot.txt')).read()   # QUERY_MIN=True (current)
    bT=box(cur); gold=open(os.path.join(d,'answer.txt')).read().strip()
    has_es='enough to answer' in cur
    if has_es: early+=1
    try:
        signal.alarm(90); new=gc.gen_cot(d); signal.alarm(0)   # QUERY_MIN=False (sound)
    except TO: signal.alarm(0); to.append(os.path.basename(d)); continue
    except Exception as e: signal.alarm(0); print("ERR",os.path.basename(d),str(e)[:40],flush=True); continue
    n+=1; bF=box(new)
    gtT+=(bT==gold); gtF+=(bF==gold)
    if bT!=bF: chg.append((os.path.basename(d), bT, bF, 'T' if bT==gold else 'F', 'T' if bF==gold else 'F', has_es))
    if n%200==0: print(f"  ...{n} ({time.time()-t0:.0f}s)",flush=True)
print(f"\n=== QUERY_MIN True (current) vs False (sound full search) ===")
print(f"compared {n}/{len(dirs)} (timeouts {len(to)}: {to[:6]})   puzzles with early-stop: {early}")
print(f"boxed-answer CHANGED: {len(chg)}")
print(f"GT-match: QUERY_MIN=True {gtT}/{n}   QUERY_MIN=False {gtF}/{n}   (delta {gtF-gtT:+d})")
print(f"\nof the changed, how GT moved (True_gt -> False_gt):")
from collections import Counter
print("  ", dict(Counter((c[3],c[4]) for c in chg)))
print("changed detail (id, True_box, False_box, gtT, gtF, had_earlystop):")
for c in chg[:25]: print("   ", c)
