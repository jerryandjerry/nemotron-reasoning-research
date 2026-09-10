#!/usr/bin/env python3
"""Merge the 10 blind-solve out_*.json, extract each blind boxed answer with the LIVE Kaggle metric, and
GT-match against the planted gold (from _manifest.csv). Writes _gtmatch_true_ids.txt (the usable aug) and
prints per-(reading,type) yield. The gold is used ONLY here as a filter — the solver that produced the CoT
never saw it."""
import os, re, json, csv, math, collections, glob
HERE = os.path.dirname(os.path.abspath(__file__))

def extract_final_answer(text):                     # the live Kaggle metric (last non-empty box, rfind '}')
    bs = list(re.finditer(r'\\boxed\{', text)); ms = []
    for i, m in enumerate(bs):
        end = bs[i + 1].start() if i + 1 < len(bs) else len(text)
        seg = text[m.end():end]; lb = seg.rfind('}'); ms.append(seg[:lb] if lb != -1 else seg)
    ne = [m.strip() for m in ms if m.strip()]
    return ne[-1] if ne else (ms[-1].strip() if ms else 'NOT_FOUND')

def verify(stored, pred):
    stored = stored.strip(); pred = pred.strip()
    if re.fullmatch(r'[01]+', stored): return pred.lower() == stored.lower()
    try: return math.isclose(float(stored), float(pred), rel_tol=1e-2, abs_tol=1e-5)
    except Exception: return pred.lower() == stored.lower()

def main():
    man = {r['id']: r for r in csv.DictReader(open(os.path.join(HERE, '_manifest.csv'), newline=''))}
    ans = {}
    for jf in sorted(glob.glob('/tmp/aug_solve_v3/out_*.json')):
        ans.update(json.load(open(jf)))
    print(f"answers {len(ans)} | manifest {len(man)}")
    keep = []; by = collections.Counter(); tot = collections.Counter(); crash = []
    for pid, m in man.items():
        cot_path = os.path.join(HERE, pid, 'track', 'tree_cot.txt')
        if not os.path.exists(cot_path): crash.append(pid); continue
        cot = open(cot_path).read()
        pred = extract_final_answer(cot)
        gtm = verify(m['gold'], pred)
        key = (m['reading'], m['type'])
        tot[key] += 1
        if gtm: by[key] += 1; keep.append(pid)
    print("\nGT-match by reading/type:")
    for key in sorted(tot):
        r, t = key
        print(f"  {r:10s} {t:14s} {by[key]:3d}/{tot[key]:3d} = {by[key]/tot[key]:.0%}")
    print(f"OVERALL {len(keep)}/{sum(tot.values())} = {len(keep)/sum(tot.values()):.1%}")
    if crash: print(f"\nNO COT (crash/unsolved) {len(crash)}: {crash[:20]}{'...' if len(crash)>20 else ''}")
    # per-type usable totals
    usable = collections.Counter(man[p]['type'] for p in keep)
    print("\nusable by type:", dict(usable))
    open(os.path.join(HERE, '_gtmatch_true_ids.txt'), 'w').write('\n'.join(keep) + '\n')
    print(f"\n_gtmatch_true_ids.txt: {len(keep)} usable")

if __name__ == '__main__':
    main()
