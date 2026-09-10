import sys, re, glob, os, collections
sys.path.insert(0,'_cryptarithm_solver'); import _gen_crypt as gc
def box(t):
    bs=list(re.finditer(r'\\boxed\{',t))
    if not bs: return None
    m=bs[-1]; seg=t[m.end():]; lb=seg.rfind('}'); return (seg[:lb] if lb!=-1 else seg).strip()
tok=lambda t: len(gc._TOK.encode(t,add_special_tokens=False).ids)
def scen(c):
    if 'it is a QUERY-only operator that we resolve by default' in c or 'appears only in the QUERY, so no example fixes it' in c: return 'query-only'
    if 'Summary: Equations are arithmetically unsolvable' in c: return 'not-arith-solvable'
    tail=c.split('Now solve the QUERY')[-1] if 'Now solve the QUERY' in c else c[-400:]
    return 'concat' if ('a∥b' in tail or 'b∥a' in tail) else 'deduce'
dirs=[d.rstrip('/') for d in sorted(glob.glob('*/')) if os.path.exists(os.path.join(d.rstrip('/'),'track','tree_cot.txt'))]
BAN=['reading leftward','§1.1','§1.2','§2.1','Conclusion of step','Need to try reading leftward','neither reading',
     ' guess','= unknown','exotic',' fits ','reading order = [rightward, leftward]','§1 reading rightward']
empty_box=[]; forbidden=collections.Counter(); n=0; under=[]
for d in dirs:
    t=open(os.path.join(d,'track','tree_cot.txt')).read(); n+=1
    if box(t) in (None,''): empty_box.append(d)
    for b in BAN:
        if b in t: forbidden[b]+=1
    if tok(t)<7680: under.append((d, scen(t)))
print(f"=== deterministic sweep over {n} ===")
print(f"boxes extracting empty: {len(empty_box)} {empty_box[:3]}")
print(f"forbidden/leftover tokens: {dict(forbidden) or 'NONE'}")
print(f"under-cap (readable, auditable): {len(under)}")
print("  scenario mix:", dict(collections.Counter(s for _,s in under)))
under.sort(key=lambda x:(x[1],x[0]))            # group by scenario then round-robin => balanced slices
K=10; slices=[[] for _ in range(K)]
for i,(d,s) in enumerate(under): slices[i%K].append(d)
for i,sl in enumerate(slices): open(f"/tmp/audit_slice_{i}.txt","w").write("\n".join(sl)+"\n")
print(f"wrote {K} balanced slices, sizes {[len(s) for s in slices]}")
