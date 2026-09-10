import sys, re, glob, os, signal, time, statistics
sys.path.insert(0,'_cryptarithm_solver'); import _gen_crypt as gc
gc.QUERY_MIN=False
class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s,f:(_ for _ in ()).throw(TO()))
def box(t):
    if not t: return None
    bs=list(re.finditer(r'\\boxed\{',t))
    if not bs: return None
    m=bs[-1]; seg=t[m.end():]; lb=seg.rfind('}'); return (seg[:lb] if lb!=-1 else seg).strip()
tok=lambda t: len(gc._TOK.encode(t,add_special_tokens=False).ids)
dirs=[d.rstrip('/') for d in sorted(glob.glob('*/')) if os.path.exists(os.path.join(d.rstrip('/'),'track','tree_cot.txt'))]
gt=uc=uc_gt=0; to=[]; boxchg=0; allL=[]; trainL=[]; n=0; t0=time.time()
for d in dirs:
    cur=open(os.path.join(d,'track','tree_cot.txt')).read(); bC=box(cur); gold=open(os.path.join(d,'answer.txt')).read().strip()
    try: signal.alarm(180); new=gc.gen_cot(d); signal.alarm(0)
    except TO: signal.alarm(0); to.append(os.path.basename(d)); continue
    except Exception as e: signal.alarm(0); print("ERR",os.path.basename(d),str(e)[:40],flush=True); continue
    n+=1; bN=box(new); L=tok(new); ok=(bN==gold)
    gt+=ok; uc+=(L<7680); uc_gt+=(L<7680 and ok); allL.append(L); boxchg+=(bC!=bN)
    if L<7680 and ok: trainL.append(L)
    if n%150==0: print(f"  ...{n} ({time.time()-t0:.0f}s)",flush=True)
def s(xs): xs=sorted(xs); return f"median={int(statistics.median(xs))} mean={int(statistics.mean(xs))} max={max(xs)} p90={xs[min(len(xs)-1,int(.9*len(xs)))]}"
print(f"\n=== QUERY_MIN=False, {n} regenerated (timeouts {len(to)}: {to[:8]}) ===")
print(f"GT-match: {gt}  (was 286)")
print(f"under-cap: {uc}  (was 447)")
print(f"TRAINABLE (GT-True & under-cap): {uc_gt}  (was 145)")
print(f"box changed vs current: {boxchg}")
print(f"ALL length:       {s(allL)}")
print(f"TRAINABLE length: {s(trainL)}  (was median 4044)")
