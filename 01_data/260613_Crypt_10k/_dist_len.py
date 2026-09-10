import sys, re, os, glob, signal, time, statistics
sys.path.insert(0,'_cryptarithm_solver'); import _gen_crypt_v2 as gc
class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s,f:(_ for _ in ()).throw(TO()))
def box(t):
    bs=list(re.finditer(r'\\boxed\{',t))
    if not bs: return None
    m=bs[-1]; seg=t[m.end():]; lb=seg.rfind('}'); return (seg[:lb] if lb!=-1 else seg).strip()
tok=lambda t: len(gc._TOK.encode(t, add_special_tokens=False).ids)
DG=re.compile(r' (-?\d+) ([<>]) (-?\d+)[.,]')
def reading(cot):
    m=re.search(r'reading order = (rightward|leftward)', cot)
    if m: return m.group(1)
    if 'not arithmetically solvable' in cot or 'neither reading' in cot: return 'unsolved'
    return 'short-circuit'
dirs=[d.rstrip('/') for d in sorted(glob.glob('*/')) if os.path.exists(os.path.join(d,'track','tree_cot.txt'))]
rows=[]; gtO=gtN=0; boxchg=0; degen=0; to=[]; t0=time.time(); n=0
for d in dirs:
    cur=open(os.path.join(d,'track','tree_cot.txt')).read(); bO=box(cur); gold=open(os.path.join(d,'answer.txt')).read().strip()
    try: signal.alarm(180); new=gc.gen_cot(d); signal.alarm(0)
    except TO: signal.alarm(0); to.append(os.path.basename(d)); continue
    except Exception as e: signal.alarm(0); print("ERR",os.path.basename(d),str(e)[:50],flush=True); continue
    n+=1; bN=box(new); L=tok(new); gtO+=(bO==gold); gtN+=(bN==gold); boxchg+=(bO!=bN)
    for l in new.splitlines():
        mm=DG.search(l)
        if mm and ((mm.group(2)=='>' and int(mm.group(1))<=int(mm.group(3))) or (mm.group(2)=='<' and int(mm.group(1))>=int(mm.group(3)))): degen+=1
    rows.append((os.path.basename(d), L, L<7680, reading(new)))
    if n%150==0: print(f"  ...{n} ({time.time()-t0:.0f}s)",flush=True)
print(f"\n=== regenerated {n} (timeouts {len(to)}) ===")
print(f"box changed {boxchg} | GT {gtO}->{gtN} | degenerate {degen}")
under=[r for r in rows if r[2]]
print(f"under 7680 cap: {len(under)}/{n}")
def pct(xs,p): xs=sorted(xs); return xs[min(len(xs)-1,int(p*len(xs)))]
print("\n## length (Nemotron tokens) by reading — ALL puzzles")
for rd in ['rightward','leftward','short-circuit','unsolved']:
    xs=[r[1] for r in rows if r[3]==rd]
    if xs: print(f"  {rd:<14} n={len(xs):<4} median={int(statistics.median(xs)):<6} p90={pct(xs,.9):<7} max={max(xs):<8} under_cap={sum(x<7680 for x in xs)}")
print("\n## within the UNDER-CAP set — how close to 7680 by reading")
for rd in ['rightward','leftward','short-circuit','unsolved']:
    xs=[r[1] for r in under if r[3]==rd]
    if xs: print(f"  {rd:<14} n={len(xs):<4} median={int(statistics.median(xs)):<6} p90={pct(xs,.9):<7} max={max(xs):<7} mean={int(statistics.mean(xs))}")
# how many of the under-cap puzzles in the top length-decile are leftward
under_sorted=sorted(under, key=lambda r:-r[1]); top=under_sorted[:max(1,len(under)//10)]
from collections import Counter
print(f"\n## top length-decile of the under-cap set (n={len(top)}, longest survivors): reading mix = {dict(Counter(r[3] for r in top))}")
