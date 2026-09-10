#!/usr/bin/env python3
"""Compare, for one puzzle, whether the LATEST CoT (track/tree_cot.txt) is a faithful
step-by-step log of tree.json, and how the OLD archive CoT (track/archive/tree_cot.txt) compares.

Prints a labeled report + a JSON blob. Reads only on-disk files; computes nothing about answer.txt.

Usage: python _compare_tree.py <puzzle_dir>
"""
import json, os, re, sys

def load_tree(p):
    f = os.path.join(p, 'tree.json')
    if not os.path.exists(f):
        return None
    d = json.load(open(f, encoding='utf-8'))
    types = {}
    boundlines = []
    sol = None
    def walk(n):
        nonlocal sol
        if isinstance(n, dict):
            types[n.get('type')] = types.get(n.get('type'), 0) + 1
            b = n.get('bound')
            if b and b.get('lines'):
                boundlines.extend(b['lines'])
            if n.get('assignment') and n.get('outcome') == 'success' and sol is None:
                sol = {'assignment': n['assignment'], 'variants': n.get('variants')}
            for k in (n.get('children') or []):
                walk(k)
    walk(d)
    return {'types': types, 'n_boundlines': len(boundlines), 'boundlines': boundlines, 'solution': sol}

def cot_metrics(path):
    if not os.path.exists(path):
        return None
    t = open(path, encoding='utf-8').read()
    lines = t.splitlines()
    sol = None
    m = re.search(r'Solution:\s*(.+?)\.?\s*$', t, re.M)
    if m:
        sol = {}
        for kv in re.findall(r'([A-Z])\s*=\s*(\d)', m.group(1)):
            sol[kv[0]] = int(kv[1])
    boxed = None
    mb = re.findall(r'\\boxed\{(.*?)\}', t)
    if mb:
        boxed = mb[-1]
    return {
        'n_lines': len(lines),
        'n_chars': len(t),
        'assume': len(re.findall(r'(?m)^\s*Assume ', t)),
        'try_var': len(re.findall(r'(?m)^\s*try [A-Z]=', t)),     # archive-style branch
        'backtrack': t.count('backtrack'),
        'lock': len(re.findall(r'so lock ', t)),
        'uses_in_glyph': t.count('∈'),                       # ∈  (tree.json notation)
        'uses_implies': t.count('⟹'),                        # ⟹  (tree.json notation)
        'uses_ascii_arrow': t.count(' -> '),                      # latest-solver notation
        'uses_noisy_tilde': t.count('~mul') + t.count('~add') + t.count('~sub'),
        'solution': sol,
        'boxed': boxed,
    }

def agree(a, b):
    if not a or not b:
        return None
    common = set(a) & set(b)
    if not common:
        return None
    return all(a[k] == b[k] for k in common)

def main(p):
    p = p.rstrip('/')
    tree = load_tree(p)
    latest = cot_metrics(os.path.join(p, 'track', 'tree_cot.txt'))
    archive = cot_metrics(os.path.join(p, 'track', 'archive', 'tree_cot.txt'))

    tree_branches = tree['types'].get('try_var', 0) if tree else None
    print(f"PUZZLE: {p}")
    print(f"  tree.json        : {'present' if tree else 'MISSING'}", end='')
    if tree:
        print(f" | node types={tree['types']} | bound-lines={tree['n_boundlines']}")
        print(f"                     tree branches (try_var)={tree_branches} | solution={tree['solution']['assignment'] if tree['solution'] else None} variants={tree['solution']['variants'] if tree['solution'] else None}")
    else:
        print()
    if latest:
        print(f"  LATEST cot.txt   : lines={latest['n_lines']} chars={latest['n_chars']} | Assume={latest['assume']} backtrack={latest['backtrack']} lock={latest['lock']}")
        print(f"                     notation: ∈={latest['uses_in_glyph']} ⟹={latest['uses_implies']} ' -> '={latest['uses_ascii_arrow']} ~tilde={latest['uses_noisy_tilde']}")
        print(f"                     Solution={latest['solution']} | boxed={latest['boxed']!r}")
    if archive:
        print(f"  ARCHIVE cot.txt  : lines={archive['n_lines']} chars={archive['n_chars']} | try X=v branches={archive['try_var']} backtrack={archive['backtrack']}")
        print(f"                     notation: ∈={archive['uses_in_glyph']} ⟹={archive['uses_implies']} ' -> '={archive['uses_ascii_arrow']} ~tilde={archive['uses_noisy_tilde']}")
        print(f"                     Solution={archive['solution']} | boxed={archive['boxed']!r}")

    # --- verdicts ---
    v = {}
    if tree and latest:
        # faithful-log signals: latest must use tree.json's notation AND have comparable branch count
        latest_logs_tree = (latest['uses_implies'] > 0 and latest['assume'] >= 0.5 * max(tree_branches, 1)) and latest['uses_ascii_arrow'] == 0
        v['latest_is_log_of_tree'] = bool(latest_logs_tree)
        v['latest_branches'] = latest['assume']
        v['tree_branches'] = tree_branches
        v['branch_ratio'] = round(latest['assume'] / tree_branches, 3) if tree_branches else None
    if tree and archive:
        archive_logs_tree = archive['uses_implies'] > 0 and archive['try_var'] >= 0.5 * max(tree_branches, 1)
        v['archive_is_log_of_tree'] = bool(archive_logs_tree)
    if tree and latest:
        v['latest_vs_tree_solution_agree'] = agree(latest['solution'], tree['solution']['assignment'] if tree['solution'] else None)
    if tree and archive:
        v['archive_vs_tree_solution_agree'] = agree(archive['solution'], tree['solution']['assignment'] if tree['solution'] else None)
    print(f"  VERDICT: {json.dumps(v)}")
    print("  JSON:", json.dumps({'puzzle': p, 'tree_types': tree['types'] if tree else None,
                                  'latest': latest, 'archive': archive, 'verdict': v}))

if __name__ == '__main__':
    main(sys.argv[1])
