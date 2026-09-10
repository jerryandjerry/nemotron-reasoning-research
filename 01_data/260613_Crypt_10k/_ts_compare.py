import sys, re, os, glob, signal, time
sys.path.insert(0,'_cryptarithm_solver'); import _gen_crypt as gc
class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s,f:(_ for _ in ()).throw(TO()))
def box(t):
    bs=list(re.finditer(r'\\boxed\{',t))
    if not bs: return None
    m=bs[-1]; seg=t[m.end():]; lb=seg.rfind('}'); return (seg[:lb] if lb!=-1 else seg).strip()
tok=lambda t: len(gc._TOK.encode(t, add_special_tokens=False).ids)
TWO=re.compile(r'lhs \[\-?\d+,\d+\], rhs \[\-?\d+,\d+\] — no overlap')
dirs=[d.rstrip('/') for d in sorted(glob.glob('*/')) if os.path.exists(os.path.join(d,'track','tree_cot.txt'))]
chg=[]; gtO=0; gtN=0; to=[]; n=0; oldL=0; newL=0; removed=0; capO=0; capN=0; t0=time.time()
for d in dirs:
    cur=open(os.path.join(d,'track','tree_cot.txt')).read()   # old: two-sided narrowing ON
    bO=box(cur); gold=open(os.path.join(d,'answer.txt')).read().strip()
    removed += len(TWO.findall(cur))
    try:
        signal.alarm(150); new=gc.gen_cot(d); signal.alarm(0)    # new: SKIP_TWO_SIDED
    except TO: signal.alarm(0); to.append(os.path.basename(d)); continue
    except Exception as e: signal.alarm(0); print("ERR",os.path.basename(d),str(e)[:50],flush=True); continue
    n+=1; bN=box(new); lo,ln=tok(cur),tok(new)
    oldL+=lo; newL+=ln; gtO+=(bO==gold); gtN+=(bN==gold); capO+=(lo<7680); capN+=(ln<7680)
    if bO!=bN: chg.append((os.path.basename(d),bO,bN,'T' if bO==gold else 'F','T' if bN==gold else 'F'))
    if n%200==0: print(f"  ...{n} ({time.time()-t0:.0f}s)",flush=True)
print(f"\n=== SKIP_TWO_SIDED effect (old two-sided-narrowing vs new skip) ===")
print(f"compared {n}/{len(dirs)}  (timeouts {len(to)}: {to[:8]})")
print(f"two-sided 'lhs[..],rhs[..] — no overlap' witnesses in OLD corpus: {removed}  (these disappear)")
print(f"boxed-answer CHANGED: {len(chg)}")
print(f"GT-match: OLD {gtO}/{n}  ->  NEW {gtN}/{n}  ({gtN-gtO:+d})")
print(f"total tokens: OLD {oldL:,}  ->  NEW {newL:,}  ({100*(newL-oldL)/oldL:+.1f}%)")
print(f"under 7680 cap: OLD {capO}/{n}  ->  NEW {capN}/{n}  ({capN-capO:+d})")
from collections import Counter
print("changed GT moves:", dict(Counter((c[3],c[4]) for c in chg)))
for c in chg[:12]: print("   ",c)
