import sys, re, csv, json
sys.path.insert(0,'.')
import _gen_bitm_faithful as F
ROOT='/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260612_BitM_5k'
rows=list(csv.DictReader(open(ROOT+'/axis_manifest.csv')))
box_mismatch=[]; text_mismatch=[]; n=0
def box(t):
    b=re.findall(r'\\boxed\{([^}]*)\}', t); return b[-1] if b else None
for r in rows:
    f=r['folder']; stored=open(f'{ROOT}/{f}/track/tree_cot.txt').read()
    new=F.gen_cot(f'{ROOT}/{f}')
    n+=1
    if box(new)!=box(stored): box_mismatch.append(f)
    if new.rstrip('\n')!=stored.rstrip('\n'): text_mismatch.append((f, r['source'], r['bucket']))
    if n%800==0: print(n,'checked | box-mismatch',len(box_mismatch),'| text-mismatch',len(text_mismatch),flush=True)
print(f"DONE all {n}: box-mismatch {len(box_mismatch)} | text-mismatch {len(text_mismatch)}")
from collections import Counter
if text_mismatch:
    print("text-mismatch by source/bucket:", dict(Counter((s,b) for _,s,b in text_mismatch)))
    print("sample:", [t[0] for t in text_mismatch[:10]])
json.dump([t[0] for t in text_mismatch], open('/tmp/_text_mismatch.json','w'))
