#!/usr/bin/env python3
"""Instrument the Engine: count branch steps on the SOLUTION path, and how many of them happen
AFTER the query first becomes (operands pinned + query-op locked + result-digits pinned), under the
EXISTING (sound) _answer_ready criterion. This bounds the saving a sound query-stop could give on the
winning path. We run with QUERY_MIN=False (real search) but tap _answer_ready as a passive probe.
"""
import glob, os, json, copy
import cot_generator as cg
import _gen_crypt as gc

PREFIXES = ('arithmetic','little_endian','mixed_concat','pure_concat','query_unseen_concat','mixed_concat_little_endian')

def all_dirs():
    out = []
    for d in sorted(glob.glob('*/')):
        name = d.rstrip('/').split('/')[-1]
        if not os.path.exists(os.path.join(d,'question.txt')): continue
        if any(name.startswith(p) for p in PREFIXES): out.append(name)
    return out

# We tap into the search by counting 'Try X=' and 'Branch on' lines, and separately re-run the
# engine to find, within the FINAL accepted reading, the branch index at which _answer_ready first
# returns non-None along the accepted path. Simpler: count lines globally and the position of the
# LAST 'Branch on' in the CoT relative to the Conclusion line of the winning §.

def winning_section(cot):
    """Return the text of the winning reading's search (between its '§N.4'/'enter the tree' and its Conclusion)."""
    lines = cot.splitlines()
    # find the LAST 'Conclusion of step' that is the accepted one is hard; use Summary to know mode
    return lines

if __name__ == '__main__':
    base = json.load(open('_baseline_rows.json'))
    names = all_dirs()
    over = [n for n in names if base.get(n,{}).get('tok',0) > 8192]
    # measure branch/try counts on over-cap puzzles
    import _probe_search as ps
    rows = []
    for n in over:
        cot = gc.gen_cot(n)
        s = ps.stats(cot)
        rows.append((n, s['tries'], s['branches'], s['combos'], base[n]['tok']))
    rows.sort(key=lambda r: r[4])
    print(f"{'puzzle':45} {'tries':>6} {'branch':>6} {'combos':>6} {'tok':>9}")
    for n,t,b,c,tok in rows[:8] + rows[len(rows)//2-2:len(rows)//2+2] + rows[-4:]:
        print(f"{n:45} {t:6} {b:6} {c:6} {tok:9}")
    # aggregate
    tt = sum(r[1] for r in rows); bb = sum(r[2] for r in rows); cc = sum(r[3] for r in rows)
    print(f"\nover-cap totals: tries={tt} branches={bb} combos={cc}  (n={len(rows)})")
    print(f"avg per over-cap puzzle: tries={tt/len(rows):.0f} branches={bb/len(rows):.0f} combos={cc/len(rows):.1f}")
