import sys, re, os, glob, signal, time, csv
sys.path.insert(0,'_cryptarithm_solver'); import _gen_crypt as gc; import cot_generator as cg
class TO(Exception): pass
signal.signal(signal.SIGALRM, lambda s,f:(_ for _ in ()).throw(TO()))
def box(t):
    if t is None: return None
    bs=list(re.finditer(r'\\boxed\{',t))
    if not bs: return None
    m=bs[-1]; seg=t[m.end():]; lb=seg.rfind('}'); return (seg[:lb] if lb!=-1 else seg).strip()
def gen(d):
    try: signal.alarm(150); r=gc.gen_cot(d); signal.alarm(0); return r
    except Exception: signal.alarm(0); return None
dirs=[x.rstrip('/') for x in sorted(glob.glob('*/')) if os.path.exists(os.path.join(x,'track','tree_cot.txt'))]
rows=[]; t0=time.time()
for i,d in enumerate(dirs,1):
    gold=open(os.path.join(d,'answer.txt')).read().strip()
    p=cg.parse(f'{d}/question.txt'); q=p.query
    o1=q[0]+q[1]; o2=q[3]+q[4]                       # query operand symbol strings
    gc.LEFTWARD=True;  cf=gen(d); bf=box(cf); has2=('§2 reading leftward' in cf) if cf else False
    gc.LEFTWARD=False; cr=gen(d); br=box(cr)
    rows.append((os.path.basename(d), gold, bf, br, int(has2), o1, o2))
    if i%150==0: print(f"  ...{i} ({time.time()-t0:.0f}s)",flush=True)
with open('_rightonly.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['name','gold','full_box','right_box','has_leftward','q_op1','q_op2']); w.writerows(rows)
n=len(rows)
gtF=sum(r[1]==r[2] for r in rows); gtR=sum(r[1]==r[3] for r in rows)
print(f"\n=== {n} puzzles ===")
print(f"GT  full(with leftward): {gtF}/{n}")
print(f"GT  rightward-only     : {gtR}/{n}   (delta {gtR-gtF})")
aff=[r for r in rows if r[4]==1]            # puzzles that used leftward (§2 present)
print(f"\npuzzles that reach §2 leftward today: {len(aff)}")
print(f"  of those, GT-correct today (full):        {sum(r[1]==r[2] for r in aff)}")
print(f"  of those, GT-correct rightward-only:      {sum(r[1]==r[3] for r in aff)}")
print(f"  answer changed when leftward removed:     {sum(r[2]!=r[3] for r in aff)}")
print("wrote _rightonly.csv")
