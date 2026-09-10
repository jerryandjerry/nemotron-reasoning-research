import csv, sys, os, re, json, shutil, collections
sys.path.insert(0,'.'); sys.path.insert(0,'../260612_BitM')
csv.field_size_limit(sys.maxsize)
from tokenizers import Tokenizer
ROOT='/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron'
tok=Tokenizer.from_file(f'{ROOT}/02_train/260512_huikang_085/repo/tokenizer.json')
DEL=f'{ROOT}/01_data/260612_BitM_5k'
aug=[json.loads(l) for l in open('/tmp/_aug6s_rows.jsonl')]
# keep originals' manifest rows; drop old aug
oldman=[r for r in csv.DictReader(open(f'{DEL}/axis_manifest.csv')) if r['source']=='orig']
cols=list(oldman[0].keys())
print("originals kept:", len(oldman))
# remove old aug folders
import glob
for d in glob.glob(f'{DEL}/aug_*'):
    if os.path.isdir(d): shutil.rmtree(d)
print("old aug removed; folders now:", len([d for d in os.listdir(DEL) if os.path.isdir(os.path.join(DEL,d))]))
# write new aug folders + manifest rows
newman=list(oldman)
seen=set()
for r in aug:
    fid='aug_'+r['id']
    if fid in seen: continue
    seen.add(fid)
    d=f'{DEL}/{fid}'; os.makedirs(d+'/track', exist_ok=True)
    open(d+'/question.txt','w').write(r['question']); open(d+'/answer.txt','w').write(r['gold'])
    open(d+'/track/tree_cot.txt','w').write(r['cot'])
    cot=r['cot']
    newman.append({'folder':fid,'source':'aug','bucket':r['bucket'],
        'pcc':str(int('Pattern consistency check' in cot)),'fallback':str(int('No single rule reproduces all examples' in cot)),
        'n_examples':str(r['n_ex']),'rotate':str(r['rotate']),'shift':str(r['shift']),'notwrap':str(r['notwrap']),
        'dir_L':str(r['dir_L']),'dir_R':str(r['dir_R']),'gold':r['gold']})
with open(f'{DEL}/axis_manifest.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=cols); w.writeheader(); w.writerows(newman)
tot=len([d for d in os.listdir(DEL) if os.path.isdir(os.path.join(DEL,d))])
print(f"aug written: {len(seen)} | total folders: {tot} | manifest rows: {len(newman)}")
# rebuild CSV
oldp=f'{ROOT}/01_data/260612_NumericEq_5k/260612_Curriculum/260612_STAGE3_FULL.csv'
old={r['id']:(r['oversampling'],r['in_7830']) for r in csv.DictReader(open(oldp)) if r['category']=='bit_manipulation'}
ccols=['id','prompt','answer','category','solver_cot','source','in_7830','oversampling','token length','new label','GT-match']
out=[]
for r in newman:
    f=r['folder']; q=open(f'{DEL}/{f}/question.txt').read(); a=open(f'{DEL}/{f}/answer.txt').read().strip()
    cot=open(f'{DEL}/{f}/track/tree_cot.txt').read(); t=len(tok.encode(cot).ids)
    key=f.replace('bitm_','') if f.startswith('bitm_') else None
    ov,in78=old[key] if (key in old) else ('1','no')
    out.append({'id':f,'prompt':q,'answer':a,'category':'bit_manipulation','solver_cot':cot,'source':'260614_new_solver',
                'in_7830':in78,'oversampling':ov,'token length':str(t),'new label':'','GT-match':'True'})
with open(f'{DEL}/260614_BitM5k_FULL.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=ccols); w.writeheader(); w.writerows(out)
print(f"CSV rebuilt: {len(out)} rows | over-cap: {sum(1 for r in out if int(r['token length'])>=7680)}")
