import re, sys, glob; sys.path.insert(0,'.')
import _gen_bitm as W
from collections import Counter
puzzles=[]
for d in sorted(glob.glob('bitm_*')):
    q=open(f'{d}/question.txt').read()
    exs=re.findall(r'([01]{8}) -> ([01]{8})', q)
    puzzles.append((d[5:], exs))
N=len(puzzles)

# ===== PRIMARY AXIS: reasoning path (CoT shape) = resolution tier x ambiguity =====
def analyze(exs):
    h=W.word_solve_full(exs)
    if h is None: return ('NOFIT', None, None, False)
    op=h[0]
    if op=='T3': tier='3term'
    else:
        pref=W.full_prior(exs)
        if any(p==W.PREF_NONFAM for p in pref): return ('NOFIT', op, None, False)
        tiers=set()
        for lab in pref:
            if lab.startswith('Maj'): tiers.add('maj')
            elif lab.startswith('Ch'): tiers.add('ch')
            elif lab.startswith('NOT-'): tiers.add('notwrap')
        tier = ('notwrap' if 'notwrap' in tiers else 'ch' if 'ch' in tiers else 'maj' if 'maj' in tiers else 'direct')
    # ambiguity: does the global rule differ from a naive per-bit first-fit? (proxy: >1 rule fits some col)
    # measure: any output column has >=2 distinct fitting per-bit rules disagreeing -> ambiguous-prone
    return (tier, op, h, None)

tier_c=Counter(); op_c=Counter(); joint=Counter()
trans_c=Counter(); dir_c=Counter(); notwrap_c=Counter(); nex_c=Counter(); fill_c=Counter()
for pid, exs in puzzles:
    tier, op, h, _ = analyze(exs)
    tier_c[tier]+=1
    if op: op_c[op]+=1; joint[(tier,op)]+=1
    nex_c[len(exs)]+=1
    if h:
        for f in h[1:]:
            if not isinstance(f,str): continue
            neg = f.startswith('not_'); base=f[4:] if neg else f
            if neg: notwrap_c['NOT-wrapped']+=1
            if base=='id': trans_c['identity']+=1
            elif base.startswith('rot'): trans_c['rotate']+=1; dir_c['L' if 'rotl' in base else 'R']+=1
            elif base.startswith('sh'):
                trans_c['shift-fill'+base[-1]]+=1; dir_c['L' if base.startswith('shl') else 'R']+=1
                fill_c['fill'+base[-1]]+=1
print("===== PRIMARY AXIS — reasoning path (CoT shape), the thing to BALANCE =====")
for k,v in tier_c.most_common(): print(f"  {k:<10} {v:>5}  ({100*v/N:.0f}%)")
print("\n===== axis 2 — operation family =====")
for k,v in op_c.most_common(): print(f"  {k:<6} {v:>5}  ({100*v/sum(op_c.values()):.0f}%)")
print("\n===== JOINT (path x operation) — shows they're INDEPENDENT axes =====")
ops=['XOR','AND','OR','MAJ','CH','U','T3']
print(f"  {'path':<9}"+"".join(f"{o:>6}" for o in ops))
for t in ['direct','maj','ch','notwrap','3term','NOFIT']:
    print(f"  {t:<9}"+"".join(f"{joint.get((t,o),0):>6}" for o in ops))
print("\n===== FEATURE AXES (decorrelate) =====")
print(f"  operand transform: {dict(trans_c.most_common())}")
print(f"  shift/rotate direction: {dict(dir_c)}")
print(f"  shift fill bit: {dict(fill_c)}")
print(f"  NOT-wrapped operands: {notwrap_c['NOT-wrapped']} occurrences")
print(f"  #examples: {dict(sorted(nex_c.items()))}")
