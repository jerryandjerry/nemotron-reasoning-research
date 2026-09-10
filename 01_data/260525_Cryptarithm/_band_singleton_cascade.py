#!/usr/bin/env python3
"""Per-puzzle token deltas focused on the rescuable band (base tok in (8192, 20000]).
Reuses the patched_propagate from _verify_singleton_cascade."""
import json, sys
import _gen_crypt as gc
from _verify_singleton_cascade import patched_propagate, boxed, reset, ORIG
from _probe_fixpoint import all_dirs
from tokenizers import Tokenizer

TOK = Tokenizer.from_file('/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/02_train/260512_huikang_085/repo/tokenizer.json')
base = json.load(open('_baseline_fresh.json'))

# rescuable band: base tok in (8192, 20000]
band = [n for n, v in base.items() if 'tok' in v and 8192 < v['tok'] <= 20000]
print(f"band size (8192,20000]: {len(band)}")

reset()
gc.Engine.propagate = patched_propagate
rows = []
for n in band:
    try:
        cot = gc.gen_cot(n)
        t = len(TOK.encode(cot).ids)
        rows.append((n, base[n]['tok'], t, base[n]['tok'] - t, boxed(cot) == base[n]['ans']))
    except Exception as e:
        rows.append((n, base[n]['tok'], None, None, f"ERR {e}"))
gc.Engine.propagate = ORIG

newly = [r for r in rows if r[2] and r[1] > 8192 and r[2] <= 8192]
touched = [r for r in rows if r[3] not in (None, 0)]
grew = [r for r in rows if r[3] and r[3] < 0]
mism = [r for r in rows if r[4] is not True]
print(f"touched (delta!=0): {len(touched)}")
print(f"newly under 8192 in band: {len(newly)} -> {[(r[0],r[1],r[2]) for r in newly]}")
print(f"grew (delta<0) in band: {len(grew)} -> {[(r[0],r[1],r[2]) for r in grew][:10]}")
print(f"ans mismatches in band: {len(mism)} -> {[(r[0],r[4]) for r in mism]}")
print("\nALL touched rows (base, patched, delta):")
for r in sorted(touched, key=lambda x: x[1]):
    print(f"  {r[0]}: {r[1]} -> {r[2]} ({'+' if r[3]<0 else '-'}{abs(r[3])})")
