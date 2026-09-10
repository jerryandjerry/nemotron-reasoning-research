"""Composite (T3) bucket build — appends to /tmp/_aug6_rows.jsonl. Run from this dir."""
import sys, json, random, re, hashlib
sys.path.insert(0, '.'); sys.path.insert(0, '../260612_BitM')
import _build_aug6 as B
import _gen_aug_bitm as G, _gen_bitm as W
rng = random.Random(99); need = 613; kept = 0
out = open('/tmp/_aug6_rows.jsonl', 'a')
for _ in range(need * 120):
    if kept >= need: break
    n_ex = rng.choice([7, 8, 9, 10])
    H = G.rand_H('T3', rng, B.FEAT)
    c = G.construct(H, n_ex, rng)
    if not c: continue
    text = c[0]; exs = re.findall(r'([01]{8}) -> ([01]{8})', text)
    b, _ = B.classify(exs)
    if b != 'composite': continue
    cot = B.solve(text); bx = B.box(cot)
    if bx is None or 'default 1' in cot or 'follow a single rule' not in cot: continue
    feats = G.features_of(W.word_solve_full(exs), n_ex)
    pid = 'aug' + hashlib.sha1(text.encode()).hexdigest()[:8]
    rec = {'id': pid, 'bucket': 'composite', 'pcc': int('Pattern consistency check' in cot),
           'fallback': 0, 'n_ex': n_ex, 'gold': bx, 'question': text, 'cot': cot,
           **{k: feats[k] for k in ('rotate', 'shift', 'notwrap', 'dir_L', 'dir_R')}}
    out.write(json.dumps(rec) + '\n'); out.flush(); kept += 1
    if kept % 100 == 0: print('composite', kept, '/', need, flush=True)
print('DONE composite', kept)
