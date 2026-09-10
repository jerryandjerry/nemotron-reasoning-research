import sys, json, random, collections
sys.path.insert(0,'.'); sys.path.insert(0,'../260612_BitM')
import _build_aug6s as B
B.TARGET={'direct':1026,'maj':440,'ch':449,'notwrap':484}
rng=random.Random(20260614); out=open('/tmp/_aug6s_4.jsonl','w')
for bucket,need in B.TARGET.items():
    pool=[]; pt=int(need*1.7)
    for _ in range(need*250):
        if len(pool)>=pt: break
        r=B.gen_one(bucket,rng)
        if r: pool.append(r)
    seen=set(); pool=[r for r in pool if not (r['id'] in seen or seen.add(r['id']))]
    sel=B.subsample(pool,need)
    for r in sel: out.write(json.dumps(r)+'\n')
    out.flush()
    rot=sum(r['rotate'] for r in sel); sh=sum(r['shift'] for r in sel)
    nex=dict(sorted(collections.Counter(r['n_ex'] for r in sel).items()))
    rp = rot/(rot+sh)*100 if rot+sh else 0
    print('  %s: pool %d -> %d | rot %.0f%% | nex %s' % (bucket, len(pool), len(sel), rp, nex), flush=True)
print('DONE')
