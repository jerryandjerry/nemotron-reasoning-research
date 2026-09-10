import sys, re, csv, json
sys.path.insert(0,'.')
import _gen_bitm_faithful as F
ROOT='/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260612_BitM_5k'
rows=[r for r in csv.DictReader(open(ROOT+'/axis_manifest.csv')) if r['source']=='orig']
def box(t):
    b=re.findall(r'\\boxed\{([^}]*)\}', t); return b[-1] if b else None
drift=[]; n=0
for r in rows:
    f=r['folder']; stored=open(f'{ROOT}/{f}/track/tree_cot.txt').read()
    new=F.gen_cot(f'{ROOT}/{f}')
    n+=1
    if box(new)!=box(stored): drift.append(f)
    if n%300==0: print(n,'checked,',len(drift),'drift',flush=True)
print("DONE: originals", n, "| solver-box != stored-box (STALE):", len(drift))
json.dump(drift, open('/tmp/_drift.json','w'))
print("sample:", drift[:10])
