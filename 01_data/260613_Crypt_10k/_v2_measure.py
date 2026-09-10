import sys, re, os, glob, signal, time
sys.path.insert(0,'_cryptarithm_solver'); import _gen_crypt_v2 as gc
class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s,f:(_ for _ in ()).throw(TO()))
def box(t):
    bs=list(re.finditer(r'\\boxed\{',t)); 
    if not bs: return None
    m=bs[-1]; seg=t[m.end():]; lb=seg.rfind('}'); return (seg[:lb] if lb!=-1 else seg).strip()
tok=lambda t: len(gc._TOK.encode(t, add_special_tokens=False).ids)
dirs=[d.rstrip('/') for d in sorted(glob.glob('*/')) if os.path.exists(os.path.join(d,'track','tree_cot.txt'))]
boxchg=[]; gtO=gtN=capO=capN=0; oldL=newL=0; to=[]; n=0; degen=0; t0=time.time()
DG=re.compile(r' (-?\d+) ([<>]) (-?\d+)[.,]')
for d in dirs:
    cur=open(os.path.join(d,'track','tree_cot.txt')).read(); bO=box(cur); gold=open(os.path.join(d,'answer.txt')).read().strip()
    try: signal.alarm(180); new=gc.gen_cot(d); signal.alarm(0)
    except TO: signal.alarm(0); to.append(os.path.basename(d)); continue
    except Exception as e: signal.alarm(0); print("ERR",os.path.basename(d),str(e)[:50],flush=True); continue
    n+=1; bN=box(new); lo,ln=tok(cur),tok(new); oldL+=lo; newL+=ln
    gtO+=(bO==gold); gtN+=(bN==gold); capO+=(lo<7680); capN+=(ln<7680)
    if bO!=bN: boxchg.append((os.path.basename(d),bO,bN))
    for l in new.splitlines():
        m=DG.search(l)
        if m and ((m.group(2)=='>' and int(m.group(1))<=int(m.group(3))) or (m.group(2)=='<' and int(m.group(1))>=int(m.group(3)))): degen+=1
    if n%150==0: print(f"  ...{n} ({time.time()-t0:.0f}s)",flush=True)
print(f"\n=== V2 verbose-witness effect (measure only, track NOT overwritten) ===")
print(f"regenerated {n} (timeouts {len(to)}: {to[:6]})")
print(f"BOXED-ANSWER CHANGED: {len(boxchg)}  (witness is display-only -> expect ~0)  {boxchg[:6]}")
print(f"GT-match: {gtO}/{n} -> {gtN}/{n}")
print(f"total tokens: {oldL:,} -> {newL:,}  ({100*(newL-oldL)/oldL:+.0f}%)")
print(f"under 7680 cap: {capO}/{n} -> {capN}/{n}  ({capN-capO:+d})")
print(f"degenerate comparisons (X>X / X<X): {degen}  (expect 0)")
