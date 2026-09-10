import sys, re, os, glob, signal, time, statistics
sys.path.insert(0,'_cryptarithm_solver')
import _gen_crypt as gc
class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s,f:(_ for _ in ()).throw(TO()))
def extract(t):
    bs=list(re.finditer(r'\\boxed\{',t))
    if not bs: return None
    m=bs[-1]; seg=t[m.end():]; lb=seg.rfind('}'); return (seg[:lb] if lb!=-1 else seg).strip()
tok=lambda t: len(gc._TOK.encode(t, add_special_tokens=False).ids)
CAP=7680
dirs=[d.rstrip('/') for d in sorted(glob.glob('*/')) if os.path.exists(os.path.join(d,'question.txt')) and os.path.exists(os.path.join(d,'track','tree_cot.txt'))]
oldL=[]; newL=[]; boxchg=[]; gtO=0; gtN=0; capO=0; capN=0; to=[]; n=0
iso=0; wbad=0; n5state=0; n5tot=0; unsolv=0; t0=time.time()
for d in dirs:
    cp=os.path.join(d,'track','tree_cot.txt')
    old=open(cp).read(); gold=open(os.path.join(d,'answer.txt')).read().strip()
    try:
        signal.alarm(150); new=gc.gen_cot(d); signal.alarm(0)
    except TO: signal.alarm(0); to.append(os.path.basename(d)); continue
    except Exception as e: signal.alarm(0); print("ERR",os.path.basename(d),str(e)[:60],flush=True); continue
    n+=1
    lo,ln=tok(old),tok(new); bo,bn=extract(old),extract(new)
    oldL.append(lo); newL.append(ln)
    if bo!=bn: boxchg.append((os.path.basename(d),bo,bn))
    gtO+=(bo==gold); gtN+=(bn==gold); capO+=(lo<CAP); capN+=(ln<CAP)
    iso += len(re.findall(r" in \[-?\d+,\d+\] -> [A-J]=", new))      # opaque isolates remaining
    if ('= unknown' in new) or ('exotic' in new) or (re.search(r'\bguess\b', new)): wbad+=1
    if '.5 search the remaining' in new:
        n5tot+=1; i=new.index('.5 search the remaining'); n5state += ('Current State:' in new[i:i+300])
    unsolv += new.count('Equations are arithmetically unsolvable')
    open(cp,'w').write(new)
    if n%200==0: print(f"  ...{n} done ({time.time()-t0:.0f}s)",flush=True)
oldL.sort(); newL.sort(); md=lambda x:x[len(x)//2]
print("\n================ REGEN 800 — FOUR-FIX EFFECT ================")
print(f"regenerated {n} (timeouts {len(to)}: {to})")
print(f"CoT length (tokens): OLD median {md(oldL)} total {sum(oldL):,}  ->  NEW median {md(newL)} total {sum(newL):,}  ({sum(newL)-sum(oldL):+,})")
print(f"under 7680 cap (trainable): OLD {capO}/{n}  ->  NEW {capN}/{n}  ({capN-capO:+d})")
print(f"GT-match: OLD {gtO}/{n} ({100*gtO/n:.1f}%)  ->  NEW {gtN}/{n} ({100*gtN/n:.1f}%)")
print(f"boxed-answer changed: {len(boxchg)}")
for nm,a,b in boxchg[:10]: print(f"   {nm}: {a!r} -> {b!r}")
print(f"\nopaque 'in [..] -> X=' isolate lines REMAINING: {iso}  (target 0)")
print(f"forbidden wording (guess/unknown/exotic): {wbad}  (target 0)")
print(f"§N.5 blocks carrying a Current State: {n5state}/{n5tot}")
print(f"CoTs containing 'Equations are arithmetically unsolvable': {unsolv}")
print("============================================================")
