#!/usr/bin/env python3
"""Generate the full aug batch (faithful, gold-bearing). Writes per puzzle: question.txt, answer.txt (gold,
NEVER shown to the blind solver), label.txt (subtype + coarse category), and a master _manifest.csv.
Gold is held ONLY here + in answer.txt; the blind driver never opens answer.txt."""
import os, sys, csv, random, argparse, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _gen_crypt_aug as G

# coarse Kaggle category per subtype (matches the corpus: deduce vs guess)
COARSE = {'arithmetic': 'cryptarithm_deduce', 'little_endian': 'cryptarithm_deduce',
          'pure_concat': 'cryptarithm_deduce', 'mixed_concat': 'cryptarithm_deduce',
          'mixed_concat_little_endian': 'cryptarithm_deduce',
          'query_unseen_concat': 'cryptarithm_guess'}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arithmetic', type=int, default=0)
    ap.add_argument('--little_endian', type=int, default=0)
    ap.add_argument('--pure_concat', type=int, default=0)
    ap.add_argument('--mixed', type=int, default=0)
    ap.add_argument('--mixed_le', type=int, default=0)
    ap.add_argument('--unseen', type=int, default=0)
    ap.add_argument('--seed', type=int, default=20260607)
    ap.add_argument('--idoff', type=int, default=0)            # id offset (avoid collisions across batches)
    ap.add_argument('--out', default=os.path.join(HERE, 'puzzles'))
    a = ap.parse_args()
    plan = [('arithmetic', a.arithmetic), ('little_endian', a.little_endian), ('pure_concat', a.pure_concat),
            ('mixed_concat', a.mixed), ('mixed_concat_little_endian', a.mixed_le),
            ('query_unseen_concat', a.unseen)]
    plan = [(c, n) for c, n in plan if n > 0]
    rng = random.Random(a.seed)
    if os.path.exists(a.out): shutil.rmtree(a.out)
    os.makedirs(a.out)

    manifest = []; idx = 0
    for cat, n in plan:
        made = tries = 0
        while made < n:
            tries += 1
            if tries > n * 2000:
                print(f"WARN {cat}: only {made}/{n}"); break
            try:
                text, gold, meta = G.build_puzzle(cat, rng)
            except G.Reject:
                continue
            idx += 1; pid = f"crypt_aug_{a.idoff + idx:06d}"
            d = os.path.join(a.out, pid); os.makedirs(os.path.join(d, 'track'), exist_ok=True)
            open(os.path.join(d, 'question.txt'), 'w').write(text)
            open(os.path.join(d, 'answer.txt'), 'w').write(gold)
            open(os.path.join(d, 'label.txt'), 'w').write(cat)
            manifest.append({'id': pid, 'subtype': cat, 'category': COARSE[cat], 'gold': gold,
                             'reading': meta['reading'], 'signed': meta['signed'], 'n_ops': meta['n_ops'],
                             'n_ex': meta['n_ex'], 'arith_syms': meta['arith_syms'], 'has_concat': meta['has_concat']})
            made += 1
        print(f"{cat}: {made}")
    with open(os.path.join(a.out, '_manifest.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys())); w.writeheader(); w.writerows(manifest)
    print(f"TOTAL {len(manifest)} -> {a.out}")

if __name__ == '__main__':
    main()
