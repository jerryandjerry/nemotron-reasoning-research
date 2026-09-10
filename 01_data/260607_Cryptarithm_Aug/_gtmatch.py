#!/usr/bin/env python3
"""GT-match + trainable filter. Joins the blind shard outputs (boxed answers + token length) with the
planted gold from _manifest.csv, applies the LIVE Kaggle metric, and marks each puzzle:
  GT-match  = verify(gold, blind_box)            (binary->exact, float->±1%, else->string)
  trainable = GT-match AND 0 <= ntok < 7680
Writes _gtmatch.csv (all rows) + _trainable_ids.txt (the keepers). Prints per-subtype yield."""
import os, re, csv, json, math, glob
HERE = os.path.dirname(os.path.abspath(__file__))

def extract(text):  # already-extracted boxes come straight from the driver; identity guard
    return text

def verify(stored, pred):
    stored = (stored or '').strip(); pred = (pred or '').strip()
    if re.fullmatch(r'[01]+', stored): return pred.lower() == stored.lower()
    try: return math.isclose(float(stored), float(pred), rel_tol=1e-2, abs_tol=1e-5)
    except Exception: return pred.lower() == stored.lower()

def main():
    # blind outputs
    blind = {}
    for f in glob.glob(os.path.join(HERE, '_blind_out', 'shard_*.jsonl')):
        for line in open(f):
            line = line.strip()
            if not line: continue
            o = json.loads(line); blind[o['id']] = (o['box'], o['ntok'])
    # manifest (gold + features)
    man = {}
    with open(os.path.join(HERE, 'puzzles', '_manifest.csv'), newline='') as fh:
        for row in csv.DictReader(fh): man[row['id']] = row

    from collections import defaultdict
    agg = defaultdict(lambda: {'n': 0, 'solved': 0, 'gt': 0, 'train': 0, 'toks': [], 'to': 0, 'err': 0})
    out_rows = []; trainable_ids = []
    for pid, row in man.items():
        box, ntok = blind.get(pid, ('MISSING', -1))
        s = agg[row['subtype']]; s['n'] += 1
        if box == 'TIMEOUT': s['to'] += 1
        elif isinstance(box, str) and box.startswith('ERR'): s['err'] += 1
        elif box not in ('MISSING',): s['solved'] += 1
        gt = verify(row['gold'], box) if box not in ('MISSING', 'TIMEOUT') and not str(box).startswith('ERR') else False
        trainable = bool(gt and 0 <= ntok < 7680)
        if gt: s['gt'] += 1
        if trainable: s['train'] += 1; s['toks'].append(ntok); trainable_ids.append(pid)
        out_rows.append({'id': pid, 'subtype': row['subtype'], 'category': row['category'], 'gold': row['gold'],
                         'box': box, 'ntok': ntok, 'gt': gt, 'trainable': trainable,
                         'reading': row['reading'], 'signed': row['signed'], 'n_ops': row['n_ops'],
                         'n_ex': row['n_ex'], 'arith_syms': row['arith_syms'], 'has_concat': row['has_concat']})

    with open(os.path.join(HERE, '_gtmatch.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys())); w.writeheader(); w.writerows(out_rows)
    open(os.path.join(HERE, '_trainable_ids.txt'), 'w').write('\n'.join(sorted(trainable_ids)) + '\n')

    import statistics
    print(f"{'subtype':<30}{'n':>6}{'solv':>6}{'GT':>6}{'GT%':>6}{'train':>7}{'tr%':>6}{'medTok':>8}{'TO':>4}{'ERR':>5}")
    tot = defaultdict(int)
    for sub in sorted(agg):
        s = agg[sub]; med = int(statistics.median(s['toks'])) if s['toks'] else 0
        print(f"{sub:<30}{s['n']:>6}{s['solved']:>6}{s['gt']:>6}{100*s['gt']/s['n']:>5.0f}%{s['train']:>7}"
              f"{100*s['train']/s['n']:>5.0f}%{med:>8}{s['to']:>4}{s['err']:>5}")
        for k in ('n', 'gt', 'train'): tot[k] += s[k]
    print('-' * 84)
    print(f"{'TOTAL':<30}{tot['n']:>6}{'':>6}{tot['gt']:>6}{'':>6}{tot['train']:>7}")
    print(f"\ntrainable kept: {len(trainable_ids)} -> _trainable_ids.txt")

if __name__ == '__main__':
    main()
