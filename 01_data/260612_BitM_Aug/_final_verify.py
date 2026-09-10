import csv, sys, re, collections, statistics
sys.path.insert(0,'.'); sys.path.insert(0,'../260612_BitM')
csv.field_size_limit(sys.maxsize)
import _pattern_check as PC
DEL='/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260612_BitM_5k'
man={r['folder']:r for r in csv.DictReader(open(f'{DEL}/axis_manifest.csv'))}
csvrows=list(csv.DictReader(open(f'{DEL}/260614_BitM5k_FULL.csv')))
tl={r['id']:int(r['token length']) for r in csvrows}
boxfail=soundfail=0
for r in man.values():
    f=r['folder']; cot=open(f'{DEL}/{f}/track/tree_cot.txt').read()
    q=open(f'{DEL}/{f}/question.txt').read(); exs=re.findall(r'([01]{8}) -> ([01]{8})', q)
    sel=re.search(r'\nSelected\n((?:\d .+\n?){8})', cot).group(1)
    picks={int(l.split(' ',1)[0]):l.split(' ',1)[1].strip() for l in sel.strip().split('\n')}
    box=re.findall(r'\\boxed\{([^}]*)\}', cot)[-1]
    if box!=r['gold']: boxfail+=1
    if not all(all(str(PC._eval_label(picks[p],[int(c) for c in x]))==y[p] for p in range(8)) for x,y in exs): soundfail+=1
print("=== FINAL (all 5000) ===")
print(f"over-cap (>=7680): {sum(1 for v in tl.values() if v>=7680)} | box==gold fails: {boxfail} | soundness fails: {soundfail}")
print("buckets:", {b:sum(1 for r in man.values() if r['bucket']==b) for b in ['direct','maj','ch','notwrap','composite','fallback']})
# per-bucket: rotate, n_examples flat
print(f"\n{'bucket':10s} {'n':>5s} {'rot%':>5s} {'n7':>4s} {'n8':>4s} {'n9':>4s} {'n10':>4s} {'medTok':>7s}")
for b in ['direct','maj','ch','notwrap','composite','fallback']:
    rs=[r for r in man.values() if r['bucket']==b]; n=len(rs)
    rot=sum(int(r['rotate']) for r in rs); sh=sum(int(r['shift']) for r in rs)
    c=collections.Counter(r['n_examples'] for r in rs)
    rp=f"{rot/(rot+sh)*100:.0f}" if rot+sh else "-"
    mt=int(statistics.median([tl[r['folder']] for r in rs]))
    print(f"{b:10s} {n:5d} {rp:>5s} {c['7']/n*100:3.0f}% {c['8']/n*100:3.0f}% {c['9']/n*100:3.0f}% {c['10']/n*100:3.0f}% {mt:>7d}")
# token distribution
tls=list(tl.values())
print(f"\nTOKEN LENGTH (all under 7680): min {min(tls)} | p25 {int(statistics.quantiles(tls,n=4)[0])} | median {int(statistics.median(tls))} | p75 {int(statistics.quantiles(tls,n=4)[2])} | max {max(tls)}")
for lo,hi in [(0,6000),(6000,6500),(6500,7000),(7000,7500),(7500,7680)]:
    print(f"  {lo}-{hi}: {sum(1 for t in tls if lo<=t<hi)}")
