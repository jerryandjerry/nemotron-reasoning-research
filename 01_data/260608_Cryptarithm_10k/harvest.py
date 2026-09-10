#!/usr/bin/env python3
"""Harvest 10,000 TRAINABLE cryptarithm rows, balanced 30/30/10/10/10/10.
Generate-and-test: build a gold-bearing puzzle -> blind-solve with the deterministic solver (gen_cot, which
reads question.txt only, never the planted gold) -> keep iff GT-match AND under the 7680 cap. Only keepers
are written (straight to a CSV), so ~36k rejects never touch the filesystem. Multiprocessing, per-puzzle
timeout. Resumable: re-running continues from the partial CSV (per-subtype counts read back)."""
import os, sys, re, csv, random, signal, time, argparse
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron'
CRYPT = f'{ROOT}/01_data/260525_Cryptarithm'
AUG = f'{ROOT}/01_data/260607_Cryptarithm_Aug'
sys.path.insert(0, AUG); sys.path.insert(0, CRYPT)
import _gen_crypt_aug as G

CAP = 7680
PLAN = [('arithmetic', 3000), ('little_endian', 3000), ('pure_concat', 1000),
        ('mixed_concat', 1000), ('mixed_concat_little_endian', 1000), ('query_unseen_concat', 1000)]
YIELD = {'arithmetic': .54, 'little_endian': .35, 'pure_concat': 1.0,
         'mixed_concat': .17, 'mixed_concat_little_endian': .08, 'query_unseen_concat': .36}
COARSE = {'arithmetic': 'cryptarithm_deduce', 'little_endian': 'cryptarithm_deduce',
          'pure_concat': 'cryptarithm_deduce', 'mixed_concat': 'cryptarithm_deduce',
          'mixed_concat_little_endian': 'cryptarithm_deduce', 'query_unseen_concat': 'cryptarithm_guess'}
OUT = f'{HERE}/260608_Cryptarithm_10k_harvest.csv'
COLS = ['id', 'subtype', 'category', 'prompt', 'answer', 'solver_cot', 'token length']

# ---- worker ----
class _TO(Exception): pass
def _alarm(s, f): raise _TO()
def _init():
    global _TMP, _gc
    signal.signal(signal.SIGALRM, _alarm)
    import _gen_crypt as _gc_mod
    globals()['_gc'] = _gc_mod
    _TMP = os.path.join('/tmp', f'harv_{os.getpid()}')
    os.makedirs(os.path.join(_TMP, 'track'), exist_ok=True)

def _extract(t):
    bs = list(re.finditer(r'\\boxed\{', t))
    if not bs: return None
    m = bs[-1]; seg = t[m.end():]; lb = seg.rfind('}')
    return (seg[:lb] if lb != -1 else seg).strip()

def solve(args):
    text, gold, subtype = args
    open(os.path.join(_TMP, 'question.txt'), 'w').write(text)
    try:
        signal.alarm(30)
        cot = _gc.gen_cot(_TMP)
        signal.alarm(0)
    except Exception:
        signal.alarm(0); return None
    box = _extract(cot)
    if box != gold: return None
    ntok = len(_gc._TOK.encode(cot, add_special_tokens=False).ids)
    if ntok >= CAP: return None
    return (text, gold, cot, ntok, subtype)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument('--seed', type=int, default=20260608)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    # resume: count what's already harvested per subtype
    have = {s: 0 for s, _ in PLAN}; idx = 0
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            have[r['subtype']] = have.get(r['subtype'], 0) + 1; idx = max(idx, int(r['id'].rsplit('_', 1)[1]))
        print(f"resume: already have {dict(have)} (idx={idx})", flush=True)
    fh = open(OUT, 'a', newline=''); w = csv.DictWriter(fh, fieldnames=COLS)
    if idx == 0: w.writeheader()

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init) as ex:
        for subtype, target in PLAN:
            kept = have[subtype]; gen = 0
            while kept < target:
                need = target - kept
                bn = max(400, int(need / YIELD[subtype] * 1.25))
                bn = min(bn, 8000)
                batch = []
                while len(batch) < bn:
                    try: t, g, _m = G.build_puzzle(subtype, rng); batch.append((t, g, subtype))
                    except G.Reject: pass
                gen += len(batch)
                for r in ex.map(solve, batch, chunksize=8):
                    if r and kept < target:
                        idx += 1; kept += 1
                        w.writerow({'id': f'crypt10k_{idx:06d}', 'subtype': subtype, 'category': COARSE[subtype],
                                    'prompt': r[0], 'answer': r[1], 'solver_cot': r[2], 'token length': r[3]})
                fh.flush()
                el = time.time() - t0
                print(f"[{el:6.0f}s] {subtype:<28} kept {kept}/{target}  (generated {gen}, yield {100*kept/max(gen,1):.0f}%)", flush=True)
            print(f"  DONE {subtype}: {kept} kept from {gen} generated", flush=True)
    fh.close()
    tot = sum(min(t, have.get(s, 0)) for s, t in PLAN)
    print(f"HARVEST COMPLETE -> {OUT}  ({time.time()-t0:.0f}s)", flush=True)

if __name__ == '__main__':
    main()
