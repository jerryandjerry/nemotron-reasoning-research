#!/usr/bin/env python3
"""Pilot: generate N puzzles/category, run the BLIND solver (gen_cot) on each, measure GT-match + token
length. The gen_cot round-trip IS the GT-match filter (solver sees question.txt only). Tells us feasibility
+ trainable-rate per reasoning path BEFORE scaling."""
import os, sys, re, math, json, argparse, random, shutil
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FTimeout
HERE = os.path.dirname(os.path.abspath(__file__))
CRYPT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260525_Cryptarithm'
sys.path.insert(0, CRYPT); sys.path.insert(0, HERE)
import _gen_crypt_aug as G

PDIR = os.path.join(HERE, '_pilot')

def extract_final_answer(text):
    bs = list(re.finditer(r'\\boxed\{', text)); ms = []
    for i, m in enumerate(bs):
        end = bs[i + 1].start() if i + 1 < len(bs) else len(text)
        seg = text[m.end():end]; lb = seg.rfind('}')
        ms.append(seg[:lb] if lb != -1 else seg)
    ne = [m.strip() for m in ms if m.strip()]
    return ne[-1] if ne else (ms[-1].strip() if ms else 'NOT_FOUND')

def verify(stored, pred):
    stored = stored.strip(); pred = pred.strip()
    if re.fullmatch(r'[01]+', stored): return pred.lower() == stored.lower()
    try: return math.isclose(float(stored), float(pred), rel_tol=1e-2, abs_tol=1e-5)
    except Exception: return pred.lower() == stored.lower()

def solve_one(d):
    """Run blind solver on dir d/question.txt; return (cot_len_tokens, boxed)."""
    import _gen_crypt as gc
    cot = gc.gen_cot(d)
    box = extract_final_answer(cot)
    ntok = len(gc._TOK.encode(cot, add_special_tokens=False).ids)
    return ntok, box

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=40)
    ap.add_argument('--seed', type=int, default=20260607)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--timeout', type=int, default=120)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    if os.path.exists(PDIR): shutil.rmtree(PDIR)
    os.makedirs(PDIR)

    # generate
    items = []   # (dir, category, gold, meta)
    for cat in G.CATEGORIES:
        for i in range(a.n):
            r = None
            while r is None:
                try: r = G.build_puzzle(cat, rng)
                except G.Reject: r = None
            text, gold, meta = r
            d = os.path.join(PDIR, f"{cat}_{i:03d}")
            os.makedirs(os.path.join(d, 'track'), exist_ok=True)
            open(os.path.join(d, 'question.txt'), 'w').write(text)
            open(os.path.join(d, 'answer.txt'), 'w').write(gold)
            items.append((d, cat, gold, meta))
    print(f"generated {len(items)} puzzles -> {PDIR}")

    # solve (parallel, per-task timeout)
    results = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(solve_one, d): (d, cat, gold) for (d, cat, gold, _m) in items}
        for fut in list(futs):
            d, cat, gold = futs[fut]
            try:
                ntok, box = fut.result(timeout=a.timeout)
                results[d] = (ntok, box, verify(gold, box))
            except FTimeout:
                results[d] = (999999, 'TIMEOUT', False)
            except Exception as e:
                results[d] = (-1, f'ERR:{type(e).__name__}:{e}', False)

    # aggregate
    from collections import defaultdict
    agg = defaultdict(lambda: {'n': 0, 'gt': 0, 'train': 0, 'toks': [], 'err': 0, 'to': 0})
    rows_out = []
    for (d, cat, gold, meta) in items:
        ntok, box, gtm = results[d]
        s = agg[cat]; s['n'] += 1
        if box == 'TIMEOUT': s['to'] += 1
        if isinstance(box, str) and box.startswith('ERR'): s['err'] += 1
        if gtm:
            s['gt'] += 1
            if ntok < 7680: s['train'] += 1; s['toks'].append(ntok)
        rows_out.append({'dir': os.path.basename(d), 'cat': cat, 'gold': gold, 'box': box,
                         'gt': gtm, 'ntok': ntok, **meta})
    import statistics
    print(f"\n{'category':<16}{'n':>4}{'GT':>5}{'GT%':>6}{'train':>7}{'tr%':>6}{'medTok':>8}{'TO':>4}{'ERR':>5}")
    for cat in G.CATEGORIES:
        s = agg[cat]
        med = int(statistics.median(s['toks'])) if s['toks'] else 0
        print(f"{cat:<16}{s['n']:>4}{s['gt']:>5}{100*s['gt']/s['n']:>5.0f}%{s['train']:>7}"
              f"{100*s['train']/s['n']:>5.0f}%{med:>8}{s['to']:>4}{s['err']:>5}")
    json.dump(rows_out, open(os.path.join(HERE, '_pilot_results.json'), 'w'), indent=1)
    # show a few failures for diagnosis
    print("\n--- sample GT=False (non-timeout) ---")
    shown = 0
    for r in rows_out:
        if not r['gt'] and r['box'] not in ('TIMEOUT',) and not str(r['box']).startswith('ERR') and shown < 8:
            print(f"  {r['dir']:<22} gold={r['gold']!r:<10} box={r['box']!r}")
            shown += 1
    # show errors
    errs = [r for r in rows_out if str(r['box']).startswith('ERR')]
    if errs:
        print("\n--- sample ERRORS ---")
        for r in errs[:5]: print(f"  {r['dir']:<22} {r['box']}")

if __name__ == '__main__':
    main()
