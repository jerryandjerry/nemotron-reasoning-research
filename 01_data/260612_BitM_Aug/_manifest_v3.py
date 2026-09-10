import os, re, sys, json, csv
sys.path.insert(0,'.'); sys.path.insert(0,'../260612_BitM')
import _gen_bitm as W, _gen_aug_bitm as G
STAGE='/tmp/bitm5k_v3'
def classify(exs):
    H=W.word_solve_full(exs)
    if H is None: return 'fallback', H
    if H[0]=='T3': return 'composite', H
    pref=W.full_prior(exs)
    if any(p==W.PREF_NONFAM for p in pref): return 'composite', H
    if any(l.startswith('NOT-') for l in pref): return 'notwrap', H
    if any(l.startswith('Ch') for l in pref): return 'ch', H
    if any(l.startswith('Maj') for l in pref): return 'maj', H
    return 'direct', H
rows=[]
folders=[d for d in os.listdir(STAGE) if os.path.isdir(os.path.join(STAGE,d))]
for i,f in enumerate(folders):
    q=open(f'{STAGE}/{f}/question.txt').read(); exs=re.findall(r'([01]{8}) -> ([01]{8})', q)
    cot=open(f'{STAGE}/{f}/track/tree_cot.txt').read()
    gold=open(f'{STAGE}/{f}/answer.txt').read().strip()
    b,H=classify(exs)
    feats=G.features_of(H,len(exs)) if (H and b!='fallback') else {'rotate':0,'shift':0,'notwrap':0,'dir_L':0,'dir_R':0}
    rows.append({'folder':f,'source':'orig' if f.startswith('bitm_') else 'aug','bucket':b,
                 'pcc':int('Pattern consistency check' in cot),
                 'fallback':int('No single rule reproduces all examples' in cot),
                 'n_examples':len(exs),'rotate':feats['rotate'],'shift':feats['shift'],
                 'notwrap':feats['notwrap'],'dir_L':feats['dir_L'],'dir_R':feats['dir_R'],'gold':gold})
    if i%500==0: print(i,flush=True)
with open(f'{STAGE}/axis_manifest.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("DONE manifest", len(rows))
