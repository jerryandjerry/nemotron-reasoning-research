import csv, collections
rows=list(csv.DictReader(open('_rightonly.csv')))
aff=[r for r in rows if r['has_leftward']=='1']; N=len(aff)
golds=[r['gold'] for r in aff]
# statistical fallbacks
mode_gold=collections.Counter(golds).most_common(1)[0]
by_len={}
for r in aff: by_len.setdefault(len(r['gold']),collections.Counter())[r['gold']]+=1
print(f"=== extended fallback sweep, {N} affected puzzles ===")
def hit(fn): return sum(1 for r in aff if fn(r)==r['gold'])
strats={
 'concat fwd a∥b': lambda r: r['q_op1']+r['q_op2'],
 'concat rev b∥a': lambda r: r['q_op2']+r['q_op1'],
 'always most-common-gold': lambda r: mode_gold[0],
 'most-common-gold of its length (oracle-len)': lambda r: by_len[len(r['gold'])].most_common(1)[0][0],
}
for name,fn in strats.items():
    print(f"  {name:<44} {hit(fn):>3}/{N}")
print(f"\n  (most common single gold = {mode_gold[0]!r}, appears {mode_gold[1]}x)")
print("\n=== 8 examples: query operands vs gold (note gold is an ARITHMETIC result, not the operands) ===")
for r in aff[:8]:
    print(f"  {r['name']}: operands {r['q_op1']!r} , {r['q_op2']!r}  ->  gold {r['gold']!r}   (rightward-only emitted {r['right_box']!r})")
