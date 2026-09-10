#!/usr/bin/env python3
"""Measure the TRUE GT-true rate and trainable rate per subtype on a clean uncapped sample
(the harvest log understates yields because it stops at target)."""
import os, sys, re, random, signal
from concurrent.futures import ProcessPoolExecutor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import harvest as H
CAP = 7680

class _TO(Exception): pass
def _al(s, f): raise _TO()
def _init():
    signal.signal(signal.SIGALRM, _al)
    import _gen_crypt as gc; globals()['_gc'] = gc
    globals()['_TMP'] = os.path.join('/tmp', f'meas_{os.getpid()}')
    os.makedirs(os.path.join(_TMP, 'track'), exist_ok=True)

def measure(args):
    text, gold = args
    open(os.path.join(_TMP, 'question.txt'), 'w').write(text)
    try:
        signal.alarm(30); cot = _gc.gen_cot(_TMP); signal.alarm(0)
    except Exception:
        signal.alarm(0); return (False, False)
    bs = list(re.finditer(r'\\boxed\{', cot)); box = None
    if bs:
        m = bs[-1]; seg = cot[m.end():]; lb = seg.rfind('}'); box = (seg[:lb] if lb != -1 else seg).strip()
    gt = (box == gold)
    ntok = len(_gc._TOK.encode(cot, add_special_tokens=False).ids)
    return (gt, gt and ntok < CAP)

def main():
    rng = random.Random(99); N = 250
    plan = ['arithmetic', 'little_endian', 'pure_concat', 'mixed_concat', 'mixed_concat_little_endian', 'query_unseen_concat']
    print(f'{"subtype":<28}{"n":>5}{"GT-true%":>10}{"trainable%":>12}{"GTtrue,over-cap%":>18}')
    with ProcessPoolExecutor(max_workers=8, initializer=_init) as ex:
        for sub in plan:
            batch = []
            while len(batch) < N:
                try: t, g, m = H.G.build_puzzle(sub, rng); batch.append((t, g))
                except H.G.Reject: pass
            res = list(ex.map(measure, batch, chunksize=8))
            gt = sum(r[0] for r in res); tr = sum(r[1] for r in res)
            print(f'{sub:<28}{N:>5}{100*gt/N:>9.0f}%{100*tr/N:>11.0f}%{100*(gt-tr)/N:>16.0f}%')

if __name__ == '__main__':
    main()
