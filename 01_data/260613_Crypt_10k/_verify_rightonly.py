import sys, re, os, glob, signal, time, collections
sys.path.insert(0,'_cryptarithm_solver'); import _gen_crypt as gc
class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s,f:(_ for _ in ()).throw(TO()))
def box(t):
    if not t: return None
    bs=list(re.finditer(r'\\boxed\{',t))
    if not bs: return None
    m=bs[-1]; seg=t[m.end():]; lb=seg.rfind('}'); return (seg[:lb] if lb!=-1 else seg).strip()
BAD=['reading leftward','§1.1','§1.2','§2.1','Conclusion of step','Need to try reading leftward','neither reading','§1 reading rightward','reading order = [rightward, leftward]']
dirs=[x.rstrip('/') for x in sorted(glob.glob('*/')) if os.path.exists(os.path.join(x,'track','tree_cot.txt'))]
gt=0; n=0; err=[]; leftover=collections.Counter(); concat_dflt=0; t0=time.time()
for d in dirs:
    gold=open(os.path.join(d,'answer.txt')).read().strip()
    try: signal.alarm(150); c=gc.gen_cot(d); signal.alarm(0)
    except Exception as e: signal.alarm(0); err.append((os.path.basename(d),str(e)[:50])); continue
    n+=1; gt+=(box(c)==gold)
    for b in BAD:
        if b in c: leftover[b]+=1
    if 'default to ~concat' in c: concat_dflt+=1
    if n%150==0: print(f"  ...{n} ({time.time()-t0:.0f}s)",flush=True)
print(f"\n=== rightward-only flattened: {n} puzzles ===")
print(f"GT-match: {gt}/{n}   (expect ~286)")
print(f"errors: {len(err)}  {err[:4]}")
print(f"leftover leftward/nested strings: {dict(leftover) or 'NONE'}")
print(f"puzzles with 'default to ~concat': {concat_dflt}  (expect ~285)")
