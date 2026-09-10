import sys, json, random
sys.path.insert(0,'.'); sys.path.insert(0,'../260612_BitM')
import _build_aug6s as B
rng=random.Random(7); need=491; pool=[]
for _ in range(need*40):
    if len(pool)>=int(need*1.8): break
    r=B.gen_one('fallback', rng)
    if r: pool.append(r)
seen=set(); pool=[r for r in pool if not (r['id'] in seen or seen.add(r['id']))]
sel=B.subsample(pool, need)   # n_ex stratified (transform greedy is a no-op since fallback has no forms)
import collections
print("fallback re-stratified:", dict(sorted(collections.Counter(r['n_ex'] for r in sel).items())))
# replace fallback rows in the jsonl
rows=[json.loads(l) for l in open('/tmp/_aug6s_rows.jsonl')]
rows=[r for r in rows if r['bucket']!='fallback'] + sel
open('/tmp/_aug6s_rows.jsonl','w').write('\n'.join(json.dumps(r) for r in rows)+'\n')
print("total rows now:", len(rows))
