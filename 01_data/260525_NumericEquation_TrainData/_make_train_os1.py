#!/usr/bin/env python3
"""Variant of 260525_NumericEquation_train.csv with the 171 NEW equation_numeric rows
(the only oversampling==0 rows) bumped to oversampling=1, so they are actually sampled.
Everything else is byte-identical."""
import csv, os
csv.field_size_limit(10 ** 7)
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '260525_NumericEquation_train.csv')
OUT = os.path.join(HERE, '260525_NumericEquation_train_os1.csv')

def main():
    with open(SRC, newline='') as fh:
        r = csv.DictReader(fh); cols = r.fieldnames; rows = list(r)
    flipped = 0
    for row in rows:
        if row['oversampling'] == '0':                 # exactly the 171 new equation_numeric rows
            row['oversampling'] = '1'; flipped += 1
    with open(OUT, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    eff = sum(int(x['oversampling']) for x in rows)
    print(f"wrote {OUT}: {len(rows)} rows, flipped {flipped} rows 0->1, effective (sum oversampling) = {eff}")

if __name__ == '__main__':
    main()
