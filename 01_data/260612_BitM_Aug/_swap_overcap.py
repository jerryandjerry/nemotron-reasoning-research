import csv, sys, os, re, json, shutil, collections
sys.path.insert(0,'.'); sys.path.insert(0,'../260612_BitM')
csv.field_size_limit(sys.maxsize)
from tokenizers import Tokenizer
ROOT='/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron'
tok=Tokenizer.from_file(f'{ROOT}/02_train/260512_huikang_085/repo/tokenizer.json')
DEL=f'{ROOT}/01_data/260612_BitM_5k'
csvrows=list(csv.DictReader(open(f'{DEL}/260614_BitM5k_FULL.csv')))
tl={r['id']:int(r['token length']) for r in csvrows}
over=[r['id'] for r in csvrows if tl[r['id']]>=7680]
repl=json.load(open('/tmp/_overcap_repl.json'))
assert len(over)==len(repl)==162, (len(over),len(repl))
# 1. remove over-cap folders + 2. write replacements
for fid in over: shutil.rmtree(f'{DEL}/{fid}')
for r in repl:
    d=f"{DEL}/{r['id']}"; os.makedirs(d+'/track',exist_ok=True)
    open(d+'/question.txt','w').write(r['question'])
    open(d+'/answer.txt','w').write(r['gold'])
    open(d+'/track/tree_cot.txt','w').write(r['cot'])
# 3. update manifest
man=[r for r in csv.DictReader(open(f'{DEL}/axis_manifest.csv')) if r['folder'] not in set(over)]
cols=list(man[0].keys())
for r in repl:
    cot=r['cot']
    man.append({'folder':r['id'],'source':'aug','bucket':r['bucket'],
                'pcc':str(int('Pattern consistency check' in cot)),
                'fallback':str(int('No single rule reproduces all examples' in cot)),
                'n_examples':str(r['n_ex']),'rotate':str(r['rotate']),'shift':str(r['shift']),
                'notwrap':str(r['notwrap']),'dir_L':str(r['dir_L']),'dir_R':str(r['dir_R']),'gold':r['gold']})
with open(f'{DEL}/axis_manifest.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=cols); w.writeheader(); w.writerows(man)
print("swapped 162 | manifest rows:", len(man))
# 4. rebuild CSV (recompute token length for all; carry source/oversampling/in_7830 conventions)
oldp=f'{ROOT}/01_data/260612_NumericEq_5k/260612_Curriculum/260612_STAGE3_FULL.csv'
old={r['id']:(r['oversampling'],r['in_7830']) for r in csv.DictReader(open(oldp)) if r['category']=='bit_manipulation'}
ccols=['id','prompt','answer','category','solver_cot','source','in_7830','oversampling','token length','new label','GT-match']
out=[]; folders=sorted(r['folder'] for r in man)
for f in folders:
    q=open(f'{DEL}/{f}/question.txt').read(); a=open(f'{DEL}/{f}/answer.txt').read().strip()
    cot=open(f'{DEL}/{f}/track/tree_cot.txt').read(); t=len(tok.encode(cot).ids)
    key=f.replace('bitm_','') if f.startswith('bitm_') else None
    ov,in78=old[key] if (key in old) else ('1','no')
    out.append({'id':f,'prompt':q,'answer':a,'category':'bit_manipulation','solver_cot':cot,
                'source':'260614_new_solver','in_7830':in78,'oversampling':ov,
                'token length':str(t),'new label':'','GT-match':'True'})
with open(f'{DEL}/260614_BitM5k_FULL.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=ccols); w.writeheader(); w.writerows(out)
overnow=sum(1 for r in out if int(r['token length'])>=7680)
print(f"CSV rebuilt: {len(out)} rows | over-cap NOW: {overnow}")
