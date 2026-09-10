import csv, sys, os, re, json, random, collections, shutil
sys.path.insert(0,'.'); sys.path.insert(0,'../260612_BitM')
csv.field_size_limit(sys.maxsize)
import _build_aug6s as B
from tokenizers import Tokenizer
ROOT='/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron'
tok=B.TOK
DEL=f'{ROOT}/01_data/260612_BitM_5k'
# 1. build a fresh composite pool with the FIXED gen_cot (Hq==box gate) -> subsample to 613
rng=random.Random(20260615); need=613; pool=[]
existing=set(r['folder'] for r in csv.DictReader(open(f'{DEL}/axis_manifest.csv')))
for _ in range(need*300):
    if len(pool)>=int(need*1.7): break
    r=B.gen_one('composite', rng)
    if r and ('aug_'+r['id']) not in existing: pool.append(r)
seen=set(); pool=[r for r in pool if not (r['id'] in seen or seen.add(r['id']))]
sel=B.subsample(pool, need)
print(f"composite rebuilt: pool {len(pool)} -> {len(sel)}")
# 2. swap: remove old composite-aug folders, write new
man={r['folder']:r for r in csv.DictReader(open(f'{DEL}/axis_manifest.csv'))}
oldcomp=[f for f,r in man.items() if r['source']=='aug' and r['bucket']=='composite']
for f in oldcomp: shutil.rmtree(f'{DEL}/{f}')
print("removed old composite aug:", len(oldcomp))
newman=[r for f,r in man.items() if f not in set(oldcomp)]
cols=list(next(iter(man.values())).keys())
for r in sel:
    fid='aug_'+r['id']; d=f'{DEL}/{fid}'; os.makedirs(d+'/track',exist_ok=True)
    open(d+'/question.txt','w').write(r['question']); open(d+'/answer.txt','w').write(r['gold'])
    open(d+'/track/tree_cot.txt','w').write(r['cot'])
    cot=r['cot']
    newman.append({'folder':fid,'source':'aug','bucket':'composite','pcc':str(int('Pattern consistency check' in cot)),
        'fallback':'0','n_examples':str(r['n_ex']),'rotate':str(r['rotate']),'shift':str(r['shift']),
        'notwrap':str(r['notwrap']),'dir_L':str(r['dir_L']),'dir_R':str(r['dir_R']),'gold':r['gold']})
with open(f'{DEL}/axis_manifest.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=cols); w.writeheader(); w.writerows(newman)
print("manifest rows:", len(newman))
json.dump([r['id'] for r in sel], open('/tmp/_newcomp.json','w'))
