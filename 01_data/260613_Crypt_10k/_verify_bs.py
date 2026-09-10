import sys, re, os, glob, signal, time
sys.path.insert(0,'_cryptarithm_solver'); import _gen_crypt as gc
class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s,f:(_ for _ in ()).throw(TO()))
def box(t):
    bs=list(re.finditer(r'\\boxed\{',t))
    if not bs: return None
    m=bs[-1]; seg=t[m.end():]; lb=seg.rfind('}'); return (seg[:lb] if lb!=-1 else seg).strip()
tok=lambda t: len(gc._TOK.encode(t, add_special_tokens=False).ids)
DG=re.compile(r' (-?\d+) ([<>]) (-?\d+) ')
dirs=[d.rstrip('/') for d in sorted(glob.glob('*/')) if os.path.exists(os.path.join(d,'track','tree_cot.txt'))]
gtO=gtN=boxchg=0; to=[]; n=0; bsN=0; degen=0; newL=0; t0=time.time()
for d in dirs:
    cur=open(os.path.join(d,'track','tree_cot.txt')).read(); bO=box(cur); gold=open(os.path.join(d,'answer.txt')).read().strip()
    try: signal.alarm(180); new=gc.gen_cot(d); signal.alarm(0)
    except TO: signal.alarm(0); to.append(os.path.basename(d)); continue
    except Exception as e: signal.alarm(0); print("ERR",os.path.basename(d),str(e)[:70],flush=True); continue
    n+=1; bN=box(new); gtO+=(bO==gold); gtN+=(bN==gold); boxchg+=(bO!=bN); newL+=tok(new)
    if 'search the largest' in new or 'search the smallest' in new: bsN+=1
    for l in new.splitlines():
        if '— yes' in l or '— no' in l or 'Impossible' in l:
            m=DG.search(l)
            if m and ((m.group(2)=='>' and int(m.group(1))<=int(m.group(3))) or (m.group(2)=='<' and int(m.group(1))>=int(m.group(3)))):
                # a '— yes' with X>Y or '— no' with X<=Y would be a wrong verdict; flag any inconsistent compare
                degen+=1
    if n%150==0: print(f"  ...{n} ({time.time()-t0:.0f}s)",flush=True)
print(f"\n=== regenerated {n} (timeouts {len(to)}: {to[:5]}) ===")
print(f"box changed {boxchg} | GT {gtO}->{gtN}")
print(f"puzzles using binary-search witness: {bsN}")
print(f"inconsistent compares: {degen}")
print(f"total tokens NEW: {newL:,}")
