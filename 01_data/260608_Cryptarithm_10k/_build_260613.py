#!/usr/bin/env python3
"""Build 01_data/260613_Cryt_10k: ONE FOLDER PER PUZZLE for original 800 + reused 10k harvest, re-solved
BLIND with the new-wording solver. GT-match uses the LIVE metric (last non-empty \boxed via rfind)."""
import csv, os, sys, re, glob, signal, shutil
csv.field_size_limit(10**8)
from concurrent.futures import ProcessPoolExecutor
ROOT='/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron'
CRYPT=f'{ROOT}/01_data/260525_Cryptarithm'; sys.path.insert(0, CRYPT)
OUT=f'{ROOT}/01_data/260613_Cryt_10k'; HARVEST=f'{ROOT}/01_data/260608_Cryptarithm_10k/260608_Cryptarithm_10k_harvest.csv'
def extract(t):
    bs=list(re.finditer(r'\\boxed\{',t)); ms=[]
    for i,m in enumerate(bs):
        s=m.end(); e=bs[i+1].start() if i+1<len(bs) else len(t); seg=t[s:e]; lb=seg.rfind('}'); ms.append(seg[:lb] if lb!=-1 else seg)
    ne=[x.strip() for x in ms if x.strip()]; return ne[-1] if ne else None
def work(args):
    folder, subtype, prompt, gold, src = args
    import _gen_crypt as gc
    d=os.path.join(OUT, folder); os.makedirs(os.path.join(d,'track'), exist_ok=True)
    open(os.path.join(d,'question.txt'),'w').write(prompt if prompt.endswith('\n') else prompt+'\n')
    open(os.path.join(d,'answer.txt'),'w').write(gold)
    def to(s,f): raise TimeoutError()
    signal.signal(signal.SIGALRM,to); signal.alarm(120)
    try: cot=gc.gen_cot(d); signal.alarm(0)
    except Exception as e: signal.alarm(0); cot=f"__ERROR__:{type(e).__name__}"
    open(os.path.join(d,'track','tree_cot.txt'),'w').write(cot)
    pred=extract(cot); m=(pred is not None and pred.strip()==gold.strip())
    open(os.path.join(d,'label.txt'),'w').write(f"{subtype}\tsrc={src}\tGT={'True' if m else 'False'}\n")
    return (src, m)
def gather():
    items=[]
    for q in glob.glob(f'{CRYPT}/*/question.txt'):
        dd=os.path.dirname(q); name=os.path.basename(dd)
        if not os.path.exists(os.path.join(dd,'answer.txt')): continue
        sub=open(os.path.join(dd,'label.txt')).read().strip() if os.path.exists(os.path.join(dd,'label.txt')) else name.rsplit('_',1)[0]
        items.append((name, sub, open(q).read(), open(os.path.join(dd,'answer.txt')).read().strip(), 'orig800'))  # folder = original name
    for r in csv.DictReader(open(HARVEST)):
        items.append((f"{r['subtype']}_{r['id']}", r['subtype'], r['prompt'], r['answer'].strip(), 'aug10k'))
    return items
def main():
    if os.path.isdir(OUT): shutil.rmtree(OUT)
    os.makedirs(OUT)
    items=gather(); print(f"building {len(items)} folders ...")
    from collections import Counter; c=Counter()
    with ProcessPoolExecutor(max_workers=8) as ex:
        for src,m in ex.map(work, items): c[(src,'GT' if m else 'noGT')]+=1
    print("GT-match:", dict(c))
if __name__=='__main__': main()
