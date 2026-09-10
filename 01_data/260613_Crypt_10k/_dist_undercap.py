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
dirs=[d.rstrip('/') for d in sorted(glob.glob('*/')) if os.path.exists(os.path.join(d,'track','tree_cot.txt'))]
rows=[]
for d in dirs:
    cot=open(os.path.join(d,'track','tree_cot.txt')).read(); L=tok(cot)
    rows.append((os.path.basename(d), L, L<7680, classify(cot), reading(cot)))
uc=[r for r in rows if r[2]]; N=len(uc)
print(f"=== distribution AMONG the {N} under-cap (trainable) puzzles ===")
def share(title, idx):
    print(f"\n## {title} (share of the {N})")
    c=collections.Counter(r[idx] for r in uc)
    for k in sorted(c, key=lambda k:-c[k]): print(f"  {k:<18} {c[k]:>4}  ({100*c[k]/N:.0f}%)")
share("scenario", 3); share("reading", 4)
def pct(xs,p): xs=sorted(xs); return xs[min(len(xs)-1,int(p*len(xs)))]
print(f"\n## length (tokens) of UNDER-CAP puzzles by reading (all < 7680):")
for rd in ['rightward','leftward','short-circuit','unsolved']:
    xs=[r[1] for r in uc if r[4]==rd]
    if xs: print(f"  {rd:<14} n={len(xs):<4} median={int(statistics.median(xs)):<6} p25={pct(xs,.25):<6} p75={pct(xs,.75):<6} p90={pct(xs,.9):<6} min={min(xs):<5} max={max(xs)}")
xs=[r[1] for r in uc if r[4]=='leftward']
print(f"\n## the {len(xs)} under-cap LEFTWARD lengths, sorted:")
print("  " + ", ".join(str(v) for v in sorted(xs)))
