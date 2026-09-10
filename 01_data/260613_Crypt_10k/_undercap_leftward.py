import sys, re, os, glob, signal, time
sys.path.insert(0,'_cryptarithm_solver'); import _gen_crypt as gc
class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s,f:(_ for _ in ()).throw(TO()))
def box(t):
    if not t: return None
    bs=list(re.finditer(r'\\boxed\{',t))
    if not bs: return None
    m=bs[-1]; seg=t[m.end():]; lb=seg.rfind('}'); return (seg[:lb] if lb!=-1 else seg).strip()
tok=lambda t: len(gc._TOK.encode(t, add_special_tokens=False).ids)
def gen(d):
    try: signal.alarm(150); r=gc.gen_cot(d); signal.alarm(0); return r
    except Exception: signal.alarm(0); return None
CAP=7680
dirs=[x.rstrip('/') for x in sorted(glob.glob('*/')) if os.path.exists(os.path.join(x,'track','tree_cot.txt'))]
rows=[]; t0=time.time()
for i,d in enumerate(dirs,1):
    full=open(os.path.join(d,'track','tree_cot.txt')).read()        # current deliverable (with leftward)
    gold=open(os.path.join(d,'answer.txt')).read().strip()
    fl=tok(full); fb=box(full); has2='§2 reading leftward' in full
    gc.LEFTWARD=False; rc=gen(d); rl=tok(rc) if rc else 10**9; rb=box(rc)
    rows.append((os.path.basename(d), gold, fb, rb, fl, rl, int(has2)))
    if i%150==0: print(f"  ...{i} ({time.time()-t0:.0f}s)",flush=True)
n=len(rows)
def C(p): return sum(1 for r in rows if p(r))
print(f"\n=== {n} puzzles: TRAINABLE (under {CAP}) accounting ===")
# current (with leftward)
cur_uc   = C(lambda r: r[4]<CAP)
cur_uc_gt= C(lambda r: r[4]<CAP and r[1]==r[2])
print(f"WITH leftward (current):     under-cap {cur_uc} | of those GT-correct {cur_uc_gt}")
# rightward-only
ro_uc    = C(lambda r: r[5]<CAP)
ro_uc_gt = C(lambda r: r[5]<CAP and r[1]==r[3])
print(f"rightward-only (concat fb):  under-cap {ro_uc} | of those GT-correct {ro_uc_gt}")
print(f"\n--- the leftward wins, by cap ---")
w=[r for r in rows if r[6]==1 and r[1]==r[2]]                 # §2 puzzles that are GT-correct today (the 173)
print(f"leftward GT-wins total: {len(w)}")
print(f"  of those UNDER cap today (trainable wins we'd lose): {sum(1 for r in w if r[4]<CAP)}")
print(f"  of those OVER cap today (not trainable anyway):       {sum(1 for r in w if r[4]>=CAP)}")
print(f"\n--- what removing leftward does to the §2 puzzles' length ---")
s2=[r for r in rows if r[6]==1]
print(f"§2 puzzles: {len(s2)}")
print(f"  under cap WITH leftward:        {sum(1 for r in s2 if r[4]<CAP)}")
print(f"  under cap as rightward-only:    {sum(1 for r in s2 if r[5]<CAP)}   (shorter: drop §2, emit concat)")
print(f"  newly-under-cap by removing §2: {sum(1 for r in s2 if r[4]>=CAP and r[5]<CAP)}")
