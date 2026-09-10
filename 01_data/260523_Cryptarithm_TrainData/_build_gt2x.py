#!/usr/bin/env python3
"""Variant of the new-solver dataset: oversample the CORRECT cryptarithm CoTs 2x.

For every cryptarithm row (source == 'new solver') that is GT-match True AND still has a CoT
(oversampling != '0', i.e. not over-length), set oversampling = '2'. Everything else is
copied unchanged from the base dataset (over-length rows stay 0; GT-match-False rows keep their
factor; non-cryptarithm rows untouched)."""
import csv, os
csv.field_size_limit(10 ** 7)
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '260523_huikang_newsolver.csv')
OUT = os.path.join(HERE, '260523_huikang_newsolver_gt2x.csv')

rows = list(csv.DictReader(open(SRC, newline='')))
cols = list(rows[0].keys())
bumped = 0
for r in rows:
    if r['source'] == 'new solver' and r['GT-match'] == 'True' and r['oversampling'] != '0':
        r['oversampling'] = '2'
        bumped += 1
with open(OUT, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
print(f"wrote {OUT}\n  rows {len(rows)} | cryptarithm GT-True rows bumped to oversampling=2: {bumped} | {os.path.getsize(OUT):,} bytes")
