#!/usr/bin/env python3
"""Faithful 'Pattern consistency check' block.

The per-bit family scan can leave a position with several rules that all fit the examples but
disagree on the query (ambiguity). huikang's chain heuristic then picks one — sometimes not the
generator's. The blind word-level hypothesis H (the single transform fitting ALL examples) is the
honest tie-break: among fitting rules, choose the one consistent with H. This module renders H as a
stated, forward-only block that openly revises the ambiguous positions. No gold is used (blind).
"""
import re, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _gen_bitm as W

# ---- per-position source of a unary form (which input bit, or a boundary fill) ----
def _src(name, i):
    """Return ('bit', j, off, wrap) or ('const', v): where output position i reads from."""
    neg = name.startswith('not_')
    base = name[4:] if neg else name
    if base == 'id': return ('bit', i, 0, False), neg
    m = re.match(r'rotl(\d)$', base)
    if m: k = int(m.group(1)); return ('bit', (i + k) % 8, k, True), neg
    m = re.match(r'rotr(\d)$', base)
    if m: k = int(m.group(1)); return ('bit', (i - k) % 8, -k, True), neg
    m = re.match(r'shl(\d)f([01])$', base)
    if m:
        k, f = int(m.group(1)), m.group(2)
        return (('bit', i + k, k, False) if i + k < 8 else ('const', int(f))), neg
    m = re.match(r'shr(\d)f([01])$', base)
    if m:
        k, f = int(m.group(1)), m.group(2)
        return (('bit', i - k, -k, False) if i - k >= 0 else ('const', int(f))), neg

def _form(name):
    """A generic per-position formula string for a unary form, e.g. 'in[i+1] (0 past the end)'."""
    neg = name.startswith('not_'); base = name[4:] if neg else name
    if base == 'id': s = 'in[i]'
    elif re.match(r'rotl(\d)$', base): s = f'in[i+{base[4]}] (wrapping)'
    elif re.match(r'rotr(\d)$', base): s = f'in[i-{base[4]}] (wrapping)'
    elif re.match(r'shl(\d)f([01])$', base): s = f'in[i+{base[3]}] ({base[5]} past the end)'
    elif re.match(r'shr(\d)f([01])$', base): s = f'in[i-{base[3]}] ({base[5]} before the start)'
    else: s = base
    return f'NOT({s})' if neg else s

def render_H(H):
    """One-line readable statement of the global rule."""
    if H[0] == 'U':
        return f'output bit i = {_form(H[1])}'
    if H[0] in ('AND', 'OR', 'XOR'):
        return f'output bit i = {H[0]}( {_form(H[1])} , {_form(H[2])} )'
    if H[0] == 'MAJ':
        return f'output bit i = majority( {_form(H[1])} , {_form(H[2])} , {_form(H[3])} )'
    if H[0] == 'CH':
        return f'output bit i = choice( selector {_form(H[1])} ? {_form(H[2])} : {_form(H[3])} )'
    if H[0] == 'T3':
        _, o1, u1, o2, u2, u3 = H
        return f'output bit i = {o1}( {_form(u1)} , {o2}( {_form(u2)} , {_form(u3)} ) )'
    return 'a single transformation of the input'

def per_pos_labels(exs):
    """Blind: H and its per-position rule labels (None if no global hypothesis)."""
    H = W.word_solve_full(exs)
    if H is None: return None, None
    pref = W.full_prior(exs)
    return H, pref

def _eval_label(lab, qb):
    """Evaluate a rule label on the 8-bit query (list of ints)."""
    import _gen_bitm_v3 as V
    if lab == 'default 1': return 1
    bits = ''.join(str(b) for b in qb)
    m = re.match(r'I(\d)$', lab)
    if m: return qb[int(m.group(1))]
    m = re.match(r'NOT(\d)$', lab)
    if m: return 1 - qb[int(m.group(1))]
    m = re.match(r'C(\d)$', lab)
    if m: return int(m.group(1))
    m = re.match(r'(AND|OR|XOR)(-NOT)?(\d)(\d)$', lab)
    if m:
        o, nt, a, b = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        x, y = qb[a], (1 - qb[b]) if nt else qb[b]
        return {'AND': x & y, 'OR': x | y, 'XOR': x ^ y}[o]
    m = re.match(r'Maj(\d)(\d)(\d)$', lab)
    if m:
        j, k, l = (int(m.group(i)) for i in (1, 2, 3)); return 1 if qb[j] + qb[k] + qb[l] >= 2 else 0
    return int(V._eval_ext(lab, bits))

def build_block(exs, query, picks_u):
    """Return (lines, revised_picks) or (None, None) if no revision needed / no hypothesis.
    blind: compares huikang's picks_u to H only via examples + the query input."""
    H, pref = per_pos_labels(exs)
    if H is None or pref is None or any(p == W.PREF_NONFAM for p in pref):
        return None, None
    qb = [int(c) for c in query]
    # function (truth-table) equality: AND50 == AND05 etc.
    dom = [[(x >> (7 - b)) & 1 for b in range(8)] for x in range(256)]
    def tt(lab): return tuple(_eval_label(lab, ib) for ib in dom)
    # revise = positions where huikang's pick is a DIFFERENT function from the global pattern's rule
    diff = [i for i in range(8) if tt(picks_u[i]) != tt(pref[i])]
    if not diff:
        return None, None
    consistent = [i for i in range(8) if i not in diff]
    incols = [''.join(e[0][j] for e in exs) for j in range(8)]   # input columns across examples
    outcols = [''.join(e[1][p] for e in exs) for p in range(8)]  # output columns across examples
    def col_of(lab):
        return ''.join(str(_eval_label(lab, [int(b) for b in e[0]])) for e in exs)
    L = ['Pattern consistency check']
    L.append(f'The resolved positions follow a single rule: {render_H(H)}.')
    if consistent:
        L.append('Positions ' + ' '.join(str(i) for i in consistent) + ' already match this rule.')
    for i in diff:
        col = col_of(pref[i])   # equals outcols[i] since pref fits every example
        L.append(f'Position {i}: {picks_u[i]} was selected, but {pref[i]} (column {col}) matches output '
                 f'column {i} and is what this rule gives at position {i}; revise to {pref[i]}.')
    L.append('')
    revised = {i: (pref[i] if i in diff else picks_u[i]) for i in range(8)}
    return L, revised

def build_confirm_block(exs, picks_u):
    """Confirmation-only block: state the global rule and show, per position, the rule's value in its
    reduced form (the shifted operands fall off the end / coincide, so a majority/choice collapses to a
    simpler op there). Used when a Maj/Ch pick is example-ambiguous (several triples fit the column) but
    already matches the single global rule. Each position spells out the reduced label with column
    evidence — so a degenerate position (e.g. AND where the rule is a majority) reads as what the rule
    gives there, not as a contradiction. Same per-position wording as build_block's revise lines."""
    H, pref = per_pos_labels(exs)
    if H is None or pref is None or any(p == W.PREF_NONFAM for p in pref):
        return None
    def col_of(lab):
        return ''.join(str(_eval_label(lab, [int(b) for b in e[0]])) for e in exs)
    L = ['Pattern consistency check',
         f'The resolved positions follow a single rule: {render_H(H)}.']
    for i in range(8):
        L.append(f'Position {i}: {picks_u[i]} (column {col_of(picks_u[i])}) matches output column {i} '
                 f'and is what this rule gives at position {i}.')
    L.append('')
    return L
