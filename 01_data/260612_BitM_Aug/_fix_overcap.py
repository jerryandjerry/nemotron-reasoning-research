import csv, sys, os, re, random, hashlib, json, collections
sys.path.insert(0,'.'); sys.path.insert(0,'../260612_BitM')
csv.field_size_limit(sys.maxsize)
import _build_aug6s as B
import _gen_aug_bitm as G, _gen_bitm as W
from tokenizers import Tokenizer
ROOT='/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron'
tok=Tokenizer.from_file(f'{ROOT}/02_train/260512_huikang_085/repo/tokenizer.json')
DEL=f'{ROOT}/01_data/260612_BitM_5k'
rows=list(csv.DictReader(open(f'{DEL}/260614_BitM5k_FULL.csv')))
man={r['folder']:r for r in csv.DictReader(open(f'{DEL}/axis_manifest.csv'))}
tl={r['id']:int(r['token length']) for r in rows}
over=[r['id'] for r in rows if tl[r['id']]>=7680]   # all aug
need=collections.Counter((man[i]['bucket'], int(man[i]['n_examples'])) for i in over)
print("over-cap to replace by (bucket,nex):", dict(need))
existing_ids=set(man)
def gen_one_nex(bucket, nex, rng):
    feat=B.FEAT_NW if bucket=='notwrap' else B.FEAT
    if bucket=='fallback':
        text=G.construct_fallback(rng,nex)
        if text is None: return None
    else:
        H=G.rand_H(rng.choice(B.OPS[bucket]),rng,feat); c=G.construct(H,nex,rng)
        if not c: return None
        text=c[0]
    exs=re.findall(r'([01]{8}) -> ([01]{8})', text)
    if B.classify(exs)!=bucket: return None
    cot=B.solve(text); bx=B.box(cot)
    if bx is None or 'default 1' in cot: return None
    if bucket=='composite' and 'follow a single rule' not in cot: return None
    if len(tok.encode(cot).ids)>=7680: return None       # THE cap
    H2=W.word_solve_full(exs)
    feats=G.features_of(H2,nex) if (H2 and bucket!='fallback') else {'rotate':0,'shift':0,'notwrap':0,'dir_L':0,'dir_R':0}
    pid='aug_aug'+hashlib.sha1(text.encode()).hexdigest()[:8]
    if pid in existing_ids: return None
    return {'id':pid,'bucket':bucket,'n_ex':nex,'gold':bx,'question':text,'cot':cot,
            **{k:feats[k] for k in ('rotate','shift','notwrap','dir_L','dir_R')}}
rng=random.Random(20260614); newrecs=[]
for (bucket,nex),k in need.items():
    got=0
    for _ in range(k*400):
        if got>=k: break
        r=gen_one_nex(bucket,nex,rng)
        if r: existing_ids.add(r['id']); newrecs.append(r); got+=1
    print(f"  {bucket} nex{nex}: {got}/{k}")
json.dump(newrecs, open('/tmp/_overcap_repl.json','w'))
print("TOTAL replacements:", len(newrecs), "/ 162")
