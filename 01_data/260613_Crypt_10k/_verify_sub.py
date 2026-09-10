import sys, re, os, glob, signal, time
sys.path.insert(0,'_cryptarithm_solver'); import _gen_crypt_v2 as gc
class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s,f:(_ for _ in ()).throw(TO()))
def box(t):
    bs=list(re.finditer(r'\\boxed\{',t))
    if not bs: return None
    m=bs[-1]; seg=t[m.end():]; lb=seg.rfind('}'); return (seg[:lb] if lb!=-1 else seg).strip()
def ev(expr):
    e=expr.replace('−','-').replace('×','*')
    e=re.sub(r'\|([^|]*)\|', r'abs(\1)', e)
    return eval(e, {'abs':abs})
dirs=[d.rstrip('/') for d in sorted(glob.glob('*/')) if os.path.exists(os.path.join(d,'track','tree_cot.txt'))]
gtO=gtN=boxchg=0; to=[]; n=0; subN=0; bad_eval=[]; bad_off=[]; t0=time.time()
OFFP=re.compile(r'^-?\d+−\d+([+−]\d+)$')          # X−Y±k  -> capture the offset k (artifact if |k|>2)
for d in dirs:
    cur=open(os.path.join(d,'track','tree_cot.txt')).read(); bO=box(cur); gold=open(os.path.join(d,'answer.txt')).read().strip()
    try: signal.alarm(180); new=gc.gen_cot(d); signal.alarm(0)
    except TO: signal.alarm(0); to.append(os.path.basename(d)); continue
    except Exception as e: signal.alarm(0); print("ERR",os.path.basename(d),str(e)[:60],flush=True); continue
    n+=1; bN=box(new); gtO+=(bO==gold); gtN+=(bN==gold); boxchg+=(bO!=bN)
    for l in new.splitlines():
        if 'lhs ~sub' not in l: continue
        subN+=1
        m=re.search(r'lhs ~sub [≥≤] (.+?)\. ', l) or re.search(r'lhs ~sub [≥≤] (.+?)$', l)
        if not m: continue
        segs=[s.strip() for s in m.group(1).split(' = ')]
        try:
            expr, val = segs[-2], int(segs[-1])
            if ev(expr) != val: bad_eval.append((os.path.basename(d), m.group(1)))
            om=OFFP.match(expr)
            if om and abs(int(om.group(1).replace('−','-')))>2: bad_off.append((os.path.basename(d), m.group(1)))
        except Exception: pass
    if n%150==0: print(f"  ...{n} ({time.time()-t0:.0f}s)",flush=True)
print(f"\n=== regenerated {n} (timeouts {len(to)}) ===")
print(f"box changed {boxchg} | GT {gtO}->{gtN}")
print(f"~sub witness lines checked: {subN}")
print(f"self-inconsistent arithmetic: {len(bad_eval)}  {bad_eval[:4]}")
print(f"offset > ±2 (the −34 artifact): {len(bad_off)}  {bad_off[:6]}")
