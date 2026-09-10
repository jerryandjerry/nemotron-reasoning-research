#!/usr/bin/env python3
"""Gold-only variant (built from the base dataset).

Train only on cryptarithm CoTs that reach the correct answer, oversampled 3x:
  - our cryptarithm rows (source == 'new solver') that are kept (oversampling != '0') and
    GT-match True  -> oversampling = '3'
  - kept cryptarithm rows that are GT-match False (honest-but-wrong) -> oversampling = '0'
Over-long cryptarithm rows (oversampling 0) stay at 0, and all non-cryptarithm rows are untouched (they
are already 100% GT-match True)."""
import csv, os
csv.field_size_limit(10 ** 7)
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '260523_huikang_newsolver.csv')
OUT = os.path.join(HERE, '260523_huikang_newsolver_goldonly.csv')

rows = list(csv.DictReader(open(SRC, newline='')))
cols = list(rows[0].keys())
n3 = nzero = 0
for r in rows:
    if r['source'] == 'new solver' and r['oversampling'] != '0':   # kept cryptarithm CoT
        if r['GT-match'] == 'True':
            r['oversampling'] = '3'; n3 += 1                       # correct -> oversample 3x
        else:
            r['oversampling'] = '0'; nzero += 1                    # honest-but-wrong -> drop
with open(OUT, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
print(f"wrote {OUT}\n  rows {len(rows)} | correct cryptarithm CoTs set to oversampling=3: {n3} | "
      f"honest-wrong dropped (->0): {nzero} | {os.path.getsize(OUT):,} bytes")
