#!/usr/bin/env python3
"""Create one folder per puzzle for the 248 golden-uncovered bit_manipulation puzzles.
bitm_<id>/question.txt  — THE ONLY INPUT the solver may read
bitm_<id>/answer.txt    — gold, used ONLY by the harvest GT-filter (never by the solver)"""
import csv, os, json, glob; csv.field_size_limit(10**8)
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT='/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron'
un=[r for r in csv.DictReader(open(f'{ROOT}/01_data/260514_huikang_update/huikang_unused.csv')) if r['category']=='bit_manipulation']
batch=set()
for f in glob.glob(f'{ROOT}/01_data/260516_bit_manipulation_update/_batch_*.json'):
    for p in json.load(open(f)): batch.add(p['id'])
assert {r['id'] for r in un}==batch and len(un)==248
n=0
for r in sorted(un, key=lambda x:x['id']):
    d=os.path.join(HERE, f"bitm_{r['id']}")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d,'question.txt'),'w').write(r['prompt'])
    open(os.path.join(d,'answer.txt'),'w').write(r['answer'])
    n+=1
print(f"created {n} puzzle folders under {HERE}")
