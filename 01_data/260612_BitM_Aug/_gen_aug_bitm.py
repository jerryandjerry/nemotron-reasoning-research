#!/usr/bin/env python3
"""Construct synthetic bit_manipulation puzzles with a known hidden rule H.
Reuses _gen_bitm's H representation + apply_word_full. A puzzle = pick H of a target type with
chosen feature knobs, generate random 8-bit example inputs + a query, apply H -> outputs (+ gold).
The hidden rule is NEVER written into question.txt. Feature knobs (transform, direction, magnitude,
fill, NOT-wrap, #examples) are injected by one global rate each, identical across paths (decorrelation)."""
import re, random
import _gen_bitm as W

# secondary-feature rates measured on the 1,497 originals (decorrelation: same rate in every bucket)
DIR_L_PROB = 0.54                       # direction: left 54% / right 46%
_MAG_W = [150, 134, 151, 136, 137, 135, 157]   # magnitude 1..7 SAMPLING weights, compensated so the KEPT
# distribution (after the bucket gate rejects degenerate large-shift rules) matches the originals'
# 19/18/20/15/10/9/9 — large magnitudes are over-sampled to survive at the right rate.

def _mag(rng):
    return rng.choices(range(1, 8), weights=_MAG_W)[0]

# ---- random unary operand form (shifted/rotated/NOT-ed copy), per the measured generator biases ----
def rand_unary(rng, feat, allow_id=False):
    """transform: 'rotate' or 'shift'; direction L/R; magnitude 1-7; fill (shift only); NOT-wrap by feat."""
    if allow_id and rng.random() < 0.02:
        base = 'id'
    else:
        kind = 'rotate' if rng.random() < feat['rotate_prob'] else 'shift'
        d = 'l' if rng.random() < DIR_L_PROB else 'r'
        k = _mag(rng)
        if kind == 'rotate':
            base = f'rot{d}{k}'
        else:
            f = '1' if rng.random() < feat['fill1_prob'] else '0'
            base = f'sh{d}{k}f{f}'
    if rng.random() < feat['notwrap_prob']:
        base = 'not_' + base
    return base

# ---- fallback (no-single-rule) construction: assign each output COLUMN its own random op ----
def _ev(lab, ib):
    """Evaluate a per-column label on input bits ib (list of 8 ints) -> 0/1."""
    m = re.match(r'I(\d)$', lab)
    if m: return ib[int(m.group(1))]
    m = re.match(r'NOT(\d)$', lab)
    if m: return 1 - ib[int(m.group(1))]
    m = re.match(r'(AND|OR|XOR)(-NOT)?(\d)(\d)$', lab)
    if m:
        o, nt, a, b = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        x, y = ib[a], (1 - ib[b]) if nt else ib[b]
        return {'AND': x & y, 'OR': x | y, 'XOR': x ^ y}[o]
    m = re.match(r'Maj(\d)(\d)(\d)$', lab)
    if m:
        j, k, l = (int(m.group(i)) for i in (1, 2, 3)); return 1 if ib[j] + ib[k] + ib[l] >= 2 else 0
    m = re.match(r'Ch(\d)(\d)(\d)$', lab)
    if m:
        s, a, b = (int(m.group(i)) for i in (1, 2, 3)); return ib[a] if ib[s] else ib[b]

def rand_label(rng):
    """A random per-column boolean function label (for fallback puzzles)."""
    fam = rng.choice(['I', 'NOT', 'AND', 'OR', 'XOR', 'AND-NOT', 'OR-NOT', 'XOR-NOT', 'Maj', 'Ch'])
    if fam in ('I', 'NOT'): return f'{fam}{rng.randint(0,7)}'
    if fam in ('AND', 'OR', 'XOR', 'AND-NOT', 'OR-NOT', 'XOR-NOT'):
        a = rng.randint(0, 7); b = rng.randint(0, 7)
        while b == a: b = rng.randint(0, 7)
        return f'{fam}{a}{b}'
    s = rng.sample(range(8), 3); return f'{fam}{s[0]}{s[1]}{s[2]}'

def construct_fallback(rng, n_ex):
    """Build a puzzle whose 8 output columns each follow a DIFFERENT random op, so (almost surely) no single
    word-level rule fits. Returns question text only; gold = the solver's own blind output (set by the driver,
    per the §4b 'plant the solver's deterministic output' rule). Caller verifies word_solve_full is None."""
    labels = [rand_label(rng) for _ in range(8)]
    seen = set(); ins = []
    tries = 0
    while len(ins) < n_ex + 1 and tries < 400:
        tries += 1
        x = ''.join(rng.choice('01') for _ in range(8))
        if x in seen: continue
        seen.add(x); ins.append(x)
    if len(ins) < n_ex + 1: return None
    def out(x):
        ib = [int(c) for c in x]
        return ''.join(str(_ev(labels[p], ib)) for p in range(8))
    query = ins[-1]; exins = ins[:n_ex]
    lines = ["In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. "
             "The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, "
             "and possibly majority or choice functions.", "",
             "Here are some examples of input -> output:"]
    for x in exins: lines.append(f"{x} -> {out(x)}")
    lines += ["", f"Now, determine the output for: {query}"]
    return "\n".join(lines)

# ---- build a random H of a given operation type ----
def rand_H(optype, rng, feat):
    if optype == 'U':
        return ('U', rand_unary(rng, feat, allow_id=True))
    if optype in ('AND', 'OR', 'XOR'):
        a = rand_unary(rng, feat); b = rand_unary(rng, feat)
        while b == a: b = rand_unary(rng, feat)
        return (optype, a, b)
    if optype == 'MAJ':
        us = []
        while len(us) < 3:
            u = rand_unary(rng, feat)
            if u not in us: us.append(u)
        return ('MAJ',) + tuple(us)
    if optype == 'CH':
        s = rand_unary(rng, feat); a = rand_unary(rng, feat); b = rand_unary(rng, feat)
        while a == s: a = rand_unary(rng, feat)
        while b == s or b == a: b = rand_unary(rng, feat)
        return ('CH', s, a, b)
    if optype == 'T3':
        o1 = rng.choice(['AND', 'OR', 'XOR']); o2 = rng.choice(['AND', 'OR', 'XOR'])
        u1 = rand_unary(rng, feat); u2 = rand_unary(rng, feat); u3 = rand_unary(rng, feat)
        while u3 == u2: u3 = rand_unary(rng, feat)
        return ('T3', o1, u1, o2, u2, u3)

# ---- per-position labels of H against the EXTENDED vocabulary (what the faithful solver targets) ----
_DOM = [[(x >> (7 - b)) & 1 for b in range(8)] for x in range(256)]
def H_perpos(H):
    strs = [format(x, '08b') for x in range(256)]
    outs = [W.apply_word_full(H, s) for s in strs]
    labs = []
    for p in range(8):
        tt = tuple(int(o[p]) for o in outs)
        lab = W.ext_label(tt)
        labs.append(lab)
    return labs

def path_of_H(H):
    """Resolution path = the advanced tier the per-position decomposition needs."""
    labs = H_perpos(H)
    if any(l is None for l in labs): return None
    if any('(' in l for l in labs): return '3term'
    if any(l.startswith('NOT-') for l in labs): return 'notwrap'
    if any(l.startswith('Ch') for l in labs): return 'ch'
    if any(l.startswith('Maj') for l in labs): return 'maj'
    return 'direct'

# ---- construct the puzzle text ----
def construct(H, n_ex, rng):
    """Random distinct example inputs + a fresh query; apply H -> outputs. Returns (text, gold)."""
    seen = set(); ins = []
    tries = 0
    while len(ins) < n_ex + 1 and tries < 400:
        tries += 1
        x = ''.join(rng.choice('01') for _ in range(8))
        if x in seen: continue
        seen.add(x); ins.append(x)
    if len(ins) < n_ex + 1: return None
    query = ins[-1]; exins = ins[:n_ex]
    exs = [(x, W.apply_word_full(H, x)) for x in exins]
    gold = W.apply_word_full(H, query)
    lines = ["In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. "
             "The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, "
             "and possibly majority or choice functions.", "",
             "Here are some examples of input -> output:"]
    for x, y in exs: lines.append(f"{x} -> {y}")
    lines += ["", f"Now, determine the output for: {query}"]
    return "\n".join(lines), gold

# ---- blind GT check: does the solver's first-fit hypothesis agree with H on the query? ----
def blind_gt(text, gold):
    exs = re.findall(r'([01]{8}) -> ([01]{8})', text)
    q = re.search(r'determine the output for: ([01]{8})', text).group(1)
    h2 = W.word_solve_full(exs)
    if h2 is None: return False, None
    pred = W.apply_word_full(h2, q)
    return pred == gold, h2

def features_of(H, n_ex):
    """Measured secondary features of H for the decorrelation table."""
    forms = [x for x in H[1:] if isinstance(x, str)]
    trans = []; dirs = []; nots = 0; fills = []
    for f in forms:
        neg = f.startswith('not_'); base = f[4:] if neg else f
        if neg: nots += 1
        if base == 'id': trans.append('id')
        elif base.startswith('rot'): trans.append('rotate'); dirs.append(base[3])
        elif base.startswith('sh'): trans.append('shift'); dirs.append(base[2]); fills.append(base[-1])
    return {'n_ex': n_ex, 'rotate': sum(1 for t in trans if t == 'rotate'),
            'shift': sum(1 for t in trans if t == 'shift'), 'notwrap': nots,
            'dir_L': dirs.count('l'), 'dir_R': dirs.count('r'), 'fill1': fills.count('1')}
