#!/usr/bin/env python3
import os, re, sys, glob, csv as _csv; _csv.field_size_limit(10**8)
sys.path.insert(0,'.')
import _gen_bitm_faithful as F
import _gen_bitm_v3 as V
import _pattern_check as PC
from tokenizers import Tokenizer
TOK=Tokenizer.from_file('/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/study/2606090112_grpo/cryptarithm_solver/tokenizer.json')
stored={r['id'] for r in _csv.DictReader(open('260514_huikang_golden_stripped.csv')) if r['category']=='bit_manipulation'}

def clen(e):
    m=re.search(r': (\d+)\s*$', e.strip()); return int(m.group(1)) if m else 0
def block(cot,h):
    i=cot.index(f'\n{h}\n')+len(h)+2; o=[]
    for ln in cot[i:].splitlines():
        if not ln.strip(): break
        o.append(ln)
    return o
def contradicts(cot):
    for s in ('Left','Right'):
        try:
            l=block(cot,f'{s}s'); d=int(re.search(rf'{s} longest: (\d+)',cot).group(1))
            if l and max(clen(x) for x in l)>d: return 'longest'
        except: pass
    mm=re.search(r'\nMatching\n((?:.+\n)+?)\nPerfect match',cot); m1=re.search(r'\nMatched\n((?:\d .+\n){8})',cot)
    if mm and m1:
        c=set()
        for ln in mm.group(1).splitlines():
            mp=re.match(r'(\d) \?\S+ - (.+)',ln)
            if mp and any(not v.strip().endswith(' absent') for v in mp.group(2).split(',')): c.add(int(mp.group(1)))
        md={int(x.split()[0]):x.split(' ',1)[1] for x in m1.group(1).splitlines()}
        if any(md.get(p)=='none' for p in c): return 'matching_none'
    return None
def sound(cot, exs, query):
    m=re.search(r'\nSelected\n((?:\d .+\n){8})', cot)
    sel={int(l.split(' ',1)[0]):l.split(' ',1)[1] for l in m.group(1).strip().split('\n')}
    for b in range(8):
        if sel[b]=='default 1': continue
        if not all(PC._eval_label(sel[b],[int(c) for c in i])==int(o[b]) for i,o in exs): return False
    box=re.findall(r'\\boxed\{([^}]*)\}',cot)[-1]
    applied="".join(("1" if sel[b]=="default 1" else str(PC._eval_label(sel[b],[int(c) for c in query]))) for b in range(8))
    return applied==box

stat={}; rows=[]; lens=[]; contra=0; unsound=0; withblock=0
for d in sorted(glob.glob('bitm_*')):
    cot=F.gen_cot(d)
    open(os.path.join(d,'track','tree_cot.txt'),'w').write(cot)
    q=open(os.path.join(d,'question.txt')).read(); gold=open(os.path.join(d,'answer.txt')).read().strip()
    exs=re.findall(r'([01]{8}) -> ([01]{8})', q); query=re.search(r'determine the output for: ([01]{8})', q).group(1)
    box=re.findall(r'\\boxed\{([^}]*)\}',cot)[-1]
    guessed=bool(re.search(r'(?m)^\d default 1$', cot))
    st=('GT' if not guessed else 'GT_GUESS') if box==gold else 'WRONG'
    pop='train' if d[5:] in stored else 'unused'
    if 'Pattern consistency check' in cot: withblock+=1
    if contradicts(cot): contra+=1
    if not sound(cot, exs, query): unsound+=1
    ntok=len(TOK.encode(q+cot, add_special_tokens=False).ids)
    stat[f'{pop}:{st}']=stat.get(f'{pop}:{st}',0)+1
    if st in ('GT','GT_GUESS'): lens.append(ntok)
    rows.append((d,pop,st,ntok))
with open('_regen_summary.csv','w',newline='') as fh:
    w=_csv.writer(fh); w.writerow(['folder','population','status','ntok']); w.writerows(rows)
print("FAITHFUL REGEN — all 1,602:")
for k in sorted(stat): print(f"  {k:<22} {stat[k]}")
gt=sum(v for k,v in stat.items() if ':GT' in k and 'GUESS' not in k); gg=sum(v for k,v in stat.items() if 'GT_GUESS' in k)
print(f"  pattern-check block emitted in: {withblock} rows")
print(f"  *** internal contradictions: {contra}  (target 0) ***")
print(f"  *** unsound traces: {unsound}  (target 0) ***")
v=sorted(lens); u=sum(1 for x in v if x<7680)
print(f"  trainable GT: {gt+gg} | UNDER 7680: {u} | lengths p50 {v[len(v)//2]} p90 {v[int(.9*len(v))]} max {v[-1]}")

# extra check: every "(column X) matches output column p" claim is true
