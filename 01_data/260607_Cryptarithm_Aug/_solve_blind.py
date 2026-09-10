#!/usr/bin/env python3
"""BLIND solver driver. Reads ONLY <dir>/question.txt, runs gen_cot (forward-only; never uses the answer),
writes <dir>/track/tree_cot.txt, records the boxed answer + token length. NEVER opens answer.txt / _manifest.

Usage:  python _solve_blind.py --shard K --of N      # solves puzzles where index % N == K
Output: _blind_out/shard_K.jsonl  (one {id, box, ntok} per line)
read_any_gold = FALSE (this driver opens only question.txt).
"""
import os, sys, re, json, argparse, glob, signal
HERE = os.path.dirname(os.path.abspath(__file__))
CRYPT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260525_Cryptarithm'
sys.path.insert(0, CRYPT)
import _gen_crypt as gc

class _Timeout(Exception): pass
def _alarm(signum, frame): raise _Timeout()
signal.signal(signal.SIGALRM, _alarm)

def extract_final_answer(text):
    bs = list(re.finditer(r'\\boxed\{', text)); ms = []
    for i, m in enumerate(bs):
        end = bs[i + 1].start() if i + 1 < len(bs) else len(text)
        seg = text[m.end():end]; lb = seg.rfind('}')
        ms.append(seg[:lb] if lb != -1 else seg)
    ne = [m.strip() for m in ms if m.strip()]
    return ne[-1] if ne else (ms[-1].strip() if ms else 'NOT_FOUND')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, required=True)
    ap.add_argument('--of', type=int, required=True)
    ap.add_argument('--pdir', default=os.path.join(HERE, 'puzzles'))
    ap.add_argument('--outdir', default=os.path.join(HERE, '_blind_out'))
    a = ap.parse_args()
    dirs = sorted(d for d in glob.glob(os.path.join(a.pdir, 'crypt_aug_*')) if os.path.isdir(d))
    mine = [d for i, d in enumerate(dirs) if i % a.of == a.shard]
    outdir = a.outdir; os.makedirs(outdir, exist_ok=True)
    outf = os.path.join(outdir, f'shard_{a.shard}.jsonl')
    n = 0
    with open(outf, 'w') as fh:
        for d in mine:
            try:
                signal.alarm(90)                                # per-puzzle wall cap (pathological searches)
                cot = gc.gen_cot(d)                              # reads d/question.txt ONLY
                signal.alarm(0)
                box = extract_final_answer(cot)
                ntok = len(gc._TOK.encode(cot, add_special_tokens=False).ids)
                open(os.path.join(d, 'track', 'tree_cot.txt'), 'w').write(cot)
            except _Timeout:
                box, ntok = 'TIMEOUT', -1
            except Exception as e:
                signal.alarm(0)
                box, ntok = f'ERR:{type(e).__name__}', -1
            fh.write(json.dumps({'id': os.path.basename(d), 'box': box, 'ntok': ntok}) + '\n')
            n += 1
    print(f"shard {a.shard}/{a.of}: solved {n} puzzles -> {outf}  (read_any_gold=False)")

if __name__ == '__main__':
    main()
