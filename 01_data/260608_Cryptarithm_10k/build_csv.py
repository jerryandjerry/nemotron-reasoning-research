#!/usr/bin/env python3
"""Build the full training CSV: base = 260609_CryptAug_eqHardened_FULL.csv (latest numeric-eq aug + others).
Replace the ENTIRE cryptarithm portion (the old 800 provider + 693 aug = 1493 rows) with the balanced,
full-coverage 10k harvest, all at 1x. Every non-cryptarithm row is byte-identical to the base."""
import os, csv, math, re
csv.field_size_limit(10 ** 8)
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260607_Cryptarithm_Aug/260607_CryptAug_TrainData/260609_CryptAug_eqHardened_FULL.csv'
HARVEST = os.path.join(HERE, '260608_Cryptarithm_10k_harvest.csv')
OUT = os.path.join(HERE, '260609_Crypt10k_eqHardened_FULL.csv')
CC = ('cryptarithm_deduce', 'cryptarithm_guess')

def main():
    with open(BASE, newline='') as fh:
        r = csv.DictReader(fh); cols = r.fieldnames; base = list(r)
    # keep all non-crypt rows AND the 800 ORIGINAL provider cryptarithm puzzles (real distribution);
    # drop ONLY the old 693 aug (source 260607_crypt_aug), which the 10k replaces.
    kept = [row for row in base if row['category'] not in CC or row['source'] == '260601_new_solver']
    removed = len(base) - len(kept)
    prov = sum(1 for row in kept if row['category'] in CC)

    added = 0
    over = 0
    for h in csv.DictReader(open(HARVEST)):
        n = int(h['token length'])
        if n >= 7680: over += 1; continue                          # safety: only trainable rows
        kept.append({'id': h['id'], 'prompt': h['prompt'], 'answer': h['answer'], 'category': h['category'],
                     'solver_cot': h['solver_cot'], 'source': '260608_crypt10k', 'in_7830': 'no',
                     'oversampling': '1', 'token length': h['token length'], 'new label': h['subtype'],
                     'GT-match': 'True'})
        added += 1

    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(kept)

    from collections import Counter
    crypt = [row for row in kept if row['category'] in CC]
    print(f"wrote {OUT}")
    print(f"  base rows {len(base)} | kept original provider crypt {prov} | removed old 693 aug {removed} | added 10k {added} (skipped over-cap {over})")
    print(f"  total rows: {len(kept)}")
    print(f"  cryptarithm rows: {len(crypt)}  (oversampling: {dict(Counter(row['oversampling'] for row in crypt))})")
    print(f"  cryptarithm by source: {dict(Counter(row['source'] for row in crypt))}")
    print(f"  cryptarithm TRAINABLE (>=1x): {sum(1 for row in crypt if row['oversampling'] in ('1','2'))}")
    print(f"  cryptarithm by subtype: {dict(Counter(row['new label'] for row in crypt))}")
    print(f"  max token length among 1x rows (all cats): {max(int(row['token length']) for row in kept if row['oversampling'] in ('1','2'))}")
    # integrity: non-crypt byte-identical
    base_nc = {row['id']: row for row in base if row['category'] not in CC}
    out_nc = {row['id']: row for row in kept if row['category'] not in CC}
    diff = sum(1 for k in base_nc if base_nc[k] != out_nc.get(k))
    print(f"  non-cryptarithm rows: {len(out_nc)} (same ids as base: {set(base_nc)==set(out_nc)}, differ: {diff})")
    print(f"  numeric-eq aug preserved: {sum(1 for row in kept if row['source']=='260607_NE_aug')}")

if __name__ == '__main__':
    main()
