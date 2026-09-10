#!/usr/bin/env python3
"""Stage the bit_manipulation rows in the 3 curriculum CSVs (260612_Curriculum/260612_STAGE{1,2,3}_FULL.csv).
Replaces the old placeholder bitm rows (1354) with my 5000 (260614_BitM5k_FULL.csv); touches ONLY bitm rows —
every other category (eq, cryptarithm, cipher, gravity, numeral, unit_conversion) passes through unchanged.
Only the `oversampling` column differs across stages. Stage 1 = originals-at-factor, aug 0. Stage 2 = originals
+ ~50% cover-all length-biased aug. Stage 3 = originals + all aug. Deterministic (seed 20260614)."""
import csv, sys, os, re, random, collections, statistics
sys.path.insert(0, '.'); sys.path.insert(0, '../260612_BitM')
csv.field_size_limit(sys.maxsize)
import _gen_bitm as W

ROOT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron'
DEL = f'{ROOT}/01_data/260612_BitM_5k'
CUR = f'{ROOT}/01_data/260612_NumericEq_5k/260612_Curriculum'
SEED = 20260614
BAND_RATE = {'short': 0.80, 'med': 0.55, 'long': 0.15}    # per-bucket length-tercile sampling -> ~50%/bucket

# my 5000 bitm rows + per-folder bucket/features
mine = {r['id']: r for r in csv.DictReader(open(f'{DEL}/260614_BitM5k_FULL.csv'))}
man = {r['folder']: r for r in csv.DictReader(open(f'{DEL}/axis_manifest.csv'))}
cols = list(next(iter(mine.values())).keys())

# ---- per-AUG axis values (operation derived; everything else from manifest) ----
def axes(fid):
    m = man[fid]
    q = open(f'{DEL}/{fid}/question.txt').read(); exs = re.findall(r'([01]{8}) -> ([01]{8})', q)
    H = W.word_solve_full(exs); op = H[0] if H else 'none'
    return {'bucket': m['bucket'], 'operation': op, 'n_examples': m['n_examples'],
            'notwrap_n': m['notwrap'], 'rotate': m['rotate'], 'shift_mag': m['shift'],
            'dir_L': m['dir_L'], 'dir_R': m['dir_R'], 'pcc': m['pcc'], 'fallback': m['fallback']}

aug_ids = [i for i in mine if i.startswith('aug_')]
tl = {i: int(mine[i]['token length']) for i in mine}
A = {i: axes(i) for i in aug_ids}
by_bucket = collections.defaultdict(list)           # bucket -> [(id, token_len)]
for i in aug_ids:
    by_bucket[A[i]['bucket']].append((i, tl[i]))

# every scenario cell that must keep >=1 member in Stage 2: the 4-tuple (with NOT-wrap COUNT,
# not just the flag) + every (bucket x secondary-axis) cell + every single-axis value.
SEC = ['operation', 'n_examples', 'notwrap_n', 'rotate', 'shift_mag', 'dir_L', 'dir_R', 'pcc', 'fallback']
cells = collections.defaultdict(list)               # cell-key -> [(id, token_len)]
for i in aug_ids:
    a = A[i]
    cells[('4tup', a['bucket'], a['operation'], a['n_examples'], a['notwrap_n'])].append((i, tl[i]))
    for s in SEC:
        cells[('bxs', s, a['bucket'], a[s])].append((i, tl[i]))   # bucket x secondary
        cells[('uni', s, a[s])].append((i, tl[i]))                # single-axis value

# ---- Stage-2 selection: per-bucket 3-band short-bias, then exhaustive cover-all floor ----
rng = random.Random(SEED); sel = set()
for bucket, mem in by_bucket.items():
    mem = sorted(mem, key=lambda x: x[1]); n = len(mem)
    for idx, (pid, t) in enumerate(mem):
        band = 'short' if idx < n / 3 else ('med' if idx < 2 * n / 3 else 'long')
        if rng.random() < BAND_RATE[band]: sel.add(pid)
for c, mem in cells.items():                          # every populated cell keeps >=1 (its shortest)
    if not any(p in sel for p, _ in mem): sel.add(min(mem, key=lambda x: x[1])[0])

# ---- oversampling per stage (bitm rows only) ----
def os_of(stage, fid):
    if fid.startswith('bitm_'): return mine[fid]['oversampling']     # originals: carried 2x/1x factor
    if stage == 1: return '0'
    if stage == 2: return '1' if fid in sel else '0'
    return '1'                                                       # stage 3: all aug

def write(stage):
    src = f'{CUR}/260612_STAGE{stage}_FULL.csv'
    rows = []
    for r in csv.DictReader(open(src)):
        if r['category'] != 'bit_manipulation':
            rows.append(r)                                            # untouched passthrough (eq staged, others as-is)
    for fid, r in mine.items():                                       # my 5000 bitm rows, staged
        rows.append({**r, 'oversampling': os_of(stage, fid)})
    with open(src, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    bm = [r for r in rows if r['category'] == 'bit_manipulation']
    trained_aug = sum(1 for r in bm if r['id'].startswith('aug_') and int(r['oversampling']) > 0)
    trained_orig = sum(1 for r in bm if r['id'].startswith('bitm_') and int(r['oversampling']) > 0)
    return len(rows), len(bm), trained_orig, trained_aug

for s in (1, 2, 3):
    tot, nbm, to, ta = write(s)
    print(f"STAGE{s}: {tot} rows | bitm {nbm} | trained orig {to} aug {ta}", flush=True)

# ---- report Stage-2 coverage + length skew ----
print(f"\nStage 2 aug selected: {len(sel)} / {len(aug_ids)} ({len(sel)/len(aug_ids)*100:.0f}%)")
gaps = [c for c, mem in cells.items() if not any(p in sel for p, _ in mem)]
print(f"scenario cells covered: {len(gaps) == 0} ({len(cells)} cells, gaps {gaps[:8]})")
btot = {b: len(v) for b, v in by_bucket.items()}
bsel = collections.Counter(man[i]['bucket'] for i in sel)
print("per-bucket selected %:", {b: f'{bsel[b]/btot[b]*100:.0f}%' for b in btot})
selL = [tl[i] for i in sel]; fullL = [tl[i] for i in aug_ids]
print(f"length: pool median {int(statistics.median(fullL))} -> selected median {int(statistics.median(selL))}; "
      f"longest kept {max(selL)} (pool {max(fullL)})")
open(f'{DEL}/_stage2_aug_ids.txt', 'w').write('\n'.join(sorted(sel)) + '\n')
print("DONE")
