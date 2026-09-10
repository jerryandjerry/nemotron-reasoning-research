#!/usr/bin/env python3
"""Step 4 — compile the 20-agent audit. Parse each axis's slice_*.md tables into per-puzzle scores,
print n / mean / score buckets, and list EVERY sub-10 id (worst first) as the triage work-list."""
import re, glob, os, statistics
HERE = os.path.dirname(os.path.abspath(__file__))

def parse(d):
    rows = {}
    for f in sorted(glob.glob(os.path.join(d, 'slice_*.md'))):
        for ln in open(f, encoding='utf-8'):
            if not ln.strip().startswith('|'): continue
            p = [c.strip() for c in ln.strip().strip('|').split('|')]
            if len(p) < 2: continue
            sid, m = p[0], re.search(r'\b(10|[0-9])\b', p[1])
            if not m or re.fullmatch(r'[-:\s]+', sid) or sid.lower() in ('id', ''): continue
            key = sid.strip('`[] ')
            reason = p[2] if len(p) > 2 else ''
            rows[key] = (int(m.group(1)), reason)
    return rows

def show(name, rows):
    if not rows:
        print(f"\n{name}: NO ROWS PARSED (agents may still be writing)"); return
    scores = [s for s, _ in rows.values()]
    b = {'10': sum(s == 10 for s in scores), '7-9': sum(7 <= s <= 9 for s in scores),
         '4-6': sum(4 <= s <= 6 for s in scores), '0-3': sum(s <= 3 for s in scores)}
    print(f"\n{'='*70}\n{name}: n={len(rows)}  mean={statistics.mean(scores):.2f}  "
          f"buckets 10={b['10']} | 7-9={b['7-9']} | 4-6={b['4-6']} | 0-3={b['0-3']}\n{'='*70}")
    sub = sorted(((s, k, r) for k, (s, r) in rows.items()), key=lambda x: x[0])
    sub = [x for x in sub if x[0] < 10]
    print(f"sub-10 ids ({len(sub)}):")
    for s, k, r in sub:
        print(f"  [{s}] {k}: {r[:120]}")

R = parse(os.path.join(HERE, '_audit/reason_results'))
F = parse(os.path.join(HERE, '_audit/format_results'))
show('REASONING (cold)', R)
show('FORMAT (template)', F)
print(f"\nparsed reason={len(R)} format={len(F)} (each should be 693)")
