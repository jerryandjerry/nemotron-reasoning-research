import sys, re, os, glob, statistics, collections
sys.path.insert(0,'_cryptarithm_solver'); import _gen_crypt as gc
tok=lambda t: len(gc._TOK.encode(t, add_special_tokens=False).ids)
def classify(cot):
    if 'it is a QUERY-only operator that we resolve by default' in cot or 'appears only in the QUERY, so no example fixes it' in cot: return 'query-only'
    if 'not arithmetically solvable' in cot or 'neither reading can solve' in cot: return 'not-arith-solvable'
    tail = cot.split('Now solve the QUERY')[-1] if 'Now solve the QUERY' in cot else cot[-400:]
    return 'concat' if ('a∥b' in tail or 'b∥a' in tail) else 'deduce'
def reading(cot):
    m=re.search(r'reading order = (rightward|leftward)', cot)
    if m: return m.group(1)
    if 'not arithmetically solvable' in cot or 'neither reading' in cot: return 'unsolved'
    return 'short-circuit'
def box(t):
    bs=list(re.finditer(r'\\boxed\{',t)); 
    if not bs: return None
    m=bs[-1]; seg=t[m.end():]; lb=seg.rfind('}'); return (seg[:lb] if lb!=-1 else seg).strip()
dirs=[d.rstrip('/') for d in sorted(glob.glob('*/')) if os.path.exists(os.path.join(d,'track','tree_cot.txt'))]
rows=[]; gt=0
for d in dirs:
    cot=open(os.path.join(d,'track','tree_cot.txt')).read()
    gold=open(os.path.join(d,'answer.txt')).read().strip()
    L=tok(cot); gt+=(box(cot)==gold)
    rows.append((os.path.basename(d), L, L<7680, classify(cot), reading(cot)))
n=len(rows); under=sum(r[2] for r in rows)
print(f"=== distribution over {n} regenerated track files ===")
print(f"GT-match: {gt}/{n}   |   under 7680 cap: {under}/{n} ({100*under/n:.0f}%)")
def table(title, idx):
    print(f"\n## {title}")
    full=collections.Counter(r[idx] for r in rows); u=collections.Counter(r[idx] for r in rows if r[2])
    print(f"{'value':<20}{'total':>7}{'under_cap':>11}{'keep%':>8}")
    for k in sorted(full, key=lambda k:-full[k]):
        print(f"{k:<20}{full[k]:>7}{u[k]:>11}{100*u[k]/full[k]:>7.0f}%")
table("A1 scenario type", 3)
table("A2 reading", 4)
def pct(xs,p): xs=sorted(xs); return xs[min(len(xs)-1,int(p*len(xs)))]
print("\n## CoT length (Nemotron tokens) by reading")
for rd in ['rightward','leftward','short-circuit','unsolved']:
    xs=[r[1] for r in rows if r[4]==rd]
    if xs: print(f"  {rd:<14} n={len(xs):<4} median={int(statistics.median(xs)):<6} p90={pct(xs,.9):<7} max={max(xs):<8} under_cap={sum(x<7680 for x in xs)}")
allL=[r[1] for r in rows]
print(f"\n## overall length: median={int(statistics.median(allL))} mean={int(statistics.mean(allL))} min={min(allL)} max={max(allL)}")
