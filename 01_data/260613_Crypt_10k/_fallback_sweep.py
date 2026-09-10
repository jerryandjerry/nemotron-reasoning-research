import csv, collections
rows=list(csv.DictReader(open('_rightonly.csv')))
aff=[r for r in rows if r['has_leftward']=='1']            # the puzzles that go to §2 today (would become fallback under rightward-only)
N=len(aff)
print(f"=== fallback sweep over the {N} puzzles that reach §2 leftward today ===")
# symbol-level fallback candidates (cipher is unresolved, so only symbol concatenations are computable)
def cands(r):
    a,b=r['q_op1'],r['q_op2']
    return {
        'concat fwd  a∥b': a+b,
        'concat rev  b∥a': b+a,
        'op1 only    a'   : a,
        'op2 only    b'   : b,
        'current rightward-only': r['right_box'],
    }
hits=collections.Counter(); 
for r in aff:
    g=r['gold']
    for name,val in cands(r).items():
        if val==g: hits[name]+=1
print(f"\n{'strategy':<24}{'GT-match':>9}{'of N':>8}")
for name,_ in cands(aff[0]).items():
    print(f"  {name:<24}{hits[name]:>7}  ({100*hits[name]/N:.0f}%)")
# oracle: best achievable if we could pick the right symbol-level candidate per puzzle
ora=sum(1 for r in aff if r['gold'] in set(cands(r).values()))
print(f"\n  ORACLE (best symbol-level per puzzle): {ora} ({100*ora/N:.0f}%)")
# how many affected golds are even length-compatible with a 4-symbol concat / 2-symbol operand
glen=collections.Counter(len(r['gold']) for r in aff)
print(f"\n  affected-gold length histogram: {dict(sorted(glen.items()))}")
print(f"  (op1/op2 are 2 symbols, concat is 4 symbols)")
