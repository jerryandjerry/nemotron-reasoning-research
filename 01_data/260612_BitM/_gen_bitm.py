#!/usr/bin/env python3
"""Deterministic bit_manipulation CoT generator — native huikang-template emitter.

gen_cot(folder) reads ONLY folder/question.txt and returns the CoT string (forward-only log).
The solve is a blind, prior-ordered joint word-level hypothesis search (mined from the 1,354
winning train assignments): out = u(x) | OP(u1,u2) | Maj(u1,u2,u3) | Ch(sel,a,b), u = shifted/
rotated/NOT-ed copies of x. The TRACE is the provider's per-bit template: Output blocks ->
bit columns -> 9 family sections (candidates+Matching+Left/Right chains) -> Selecting ->
Tentative/Preferred/Matching/Perfect match/Matched -> [Maj/Ch fallback for positions no family
fits] -> Selected -> Applying -> boxed. Every Selected rule is verified against ALL examples
inside the trace machinery; the boxed answer is computed by APPLYING the Selected rules to the
query (never injected). Rows whose hypothesis decomposes outside the per-bit vocabulary
(3-term gate mixes) are not emitted (returns None: unexpressible).
"""
import os, re
from itertools import combinations, permutations

FAMS = ['Identity', 'NOT', 'Constant', 'AND', 'OR', 'XOR', 'AND-NOT', 'OR-NOT', 'XOR-NOT']

# ───────────────────────── parsing ─────────────────────────
def parse_question(text):
    exs = re.findall(r'([01]{8}) -> ([01]{8})', text)
    q = re.search(r'determine the output for: ([01]{8})', text).group(1)
    return exs, q

# ───────────────────── word-level blind solver ─────────────────────
def _shifts():
    out = [('id', lambda s: s)]
    for k in range(1, 8):
        out.append((f'rotl{k}', lambda s, k=k: s[k:] + s[:k]))
        out.append((f'rotr{k}', lambda s, k=k: s[-k:] + s[:-k]))
        for f in '01':
            out.append((f'shl{k}f{f}', lambda s, k=k, f=f: s[k:] + f * k))
            out.append((f'shr{k}f{f}', lambda s, k=k, f=f: f * k + s[:-k]))
    return out
_SH = _shifts()
def _not(s): return ''.join('1' if c == '0' else '0' for c in s)
_UN = [(n, f) for n, f in _SH] + [('not_' + n, lambda s, f=f: _not(f(s))) for n, f in _SH]
_FN = dict(_UN); _SHN = [n for n, _ in _SH]; _UNN = [n for n, _ in _UN]

def word_solve(exs):
    """First hypothesis (fixed prior order) reproducing ALL examples. Blind: examples only."""
    tgt = int(''.join(b for _, b in exs), 2)
    W = 8 * len(exs); MASK = (1 << W) - 1
    U = {n: int(''.join(f(a) for a, _ in exs), 2) for n, f in _UN}
    for n in _UNN:
        if U[n] == tgt: return ('U', n)
    for opn, op in (('AND', lambda x, y: x & y), ('XOR', lambda x, y: x ^ y), ('OR', lambda x, y: x | y)):
        for i in range(len(_UNN)):
            a = U[_UNN[i]]
            for j in range(i + 1, len(_UNN)):
                if op(a, U[_UNN[j]]) == tgt: return (opn, _UNN[i], _UNN[j])
    for c in combinations(_SHN, 3):
        a, b, d = U[c[0]], U[c[1]], U[c[2]]
        if ((a & b) | (a & d) | (b & d)) == tgt: return ('MAJ',) + c
    for s in _SHN:
        sv = U[s]; nsv = sv ^ MASK
        for a in _SHN:
            if a == s: continue
            av = sv & U[a]
            for b in _SHN:
                if b == s or b == a: continue
                if (av | (nsv & U[b])) == tgt: return ('CH', s, a, b)
    for c in combinations(_UNN, 3):
        a, b, d = U[c[0]], U[c[1]], U[c[2]]
        if ((a & b) | (a & d) | (b & d)) == tgt: return ('MAJ', ) + c
    for s in _UNN:
        sv = U[s]; nsv = sv ^ MASK
        for a in _UNN:
            if a == s: continue
            av = sv & U[a]
            for b in _UNN:
                if b == s or b == a: continue
                if (av | (nsv & U[b])) == tgt: return ('CH', s, a, b)
    # 3-term (classification only; not expressible in the per-bit vocabulary)
    OPS = {'AND': lambda x, y: x & y, 'XOR': lambda x, y: x ^ y, 'OR': lambda x, y: x | y}
    for o2, f2 in OPS.items():
        inner = []
        for i in range(len(_UNN)):
            a = U[_UNN[i]]
            for j in range(i + 1, len(_UNN)):
                inner.append(f2(a, U[_UNN[j]]))
        for o1, f1 in OPS.items():
            for u1 in _UNN:
                a = U[u1]
                for v in inner:
                    if f1(a, v) == tgt: return ('TERM3',)
    return None

def apply_word(h, s):
    if h[0] == 'U': return _FN[h[1]](s)
    if h[0] in ('AND', 'XOR', 'OR'):
        a, b = _FN[h[1]](s), _FN[h[2]](s)
        op = {'AND': lambda x, y: x & y, 'XOR': lambda x, y: x ^ y, 'OR': lambda x, y: x | y}[h[0]]
        return ''.join(str(op(int(x), int(y))) for x, y in zip(a, b))
    if h[0] == 'MAJ':
        a, b, c = (_FN[n](s) for n in h[1:])
        return ''.join('1' if int(x) + int(y) + int(z) >= 2 else '0' for x, y, z in zip(a, b, c))
    if h[0] == 'CH':
        sl, a, b = (_FN[n](s) for n in h[1:])
        return ''.join(y if x == '1' else z for x, y, z in zip(sl, a, b))

# ─────────────── per-bit vocabulary + decomposition ───────────────
def _vocab():
    """label -> fn(input-bit-list)->bit, in family order; labels match the template."""
    V = []
    for i in range(8): V.append(('Identity', f'I{i}', lambda ib, i=i: ib[i]))
    for i in range(8): V.append(('NOT', f'NOT{i}', lambda ib, i=i: 1 - ib[i]))
    V.append(('Constant', 'C0', lambda ib: 0)); V.append(('Constant', 'C1', lambda ib: 1))
    for fam, bf in (('AND', lambda x, y: x & y), ('OR', lambda x, y: x | y), ('XOR', lambda x, y: x ^ y)):
        for a in range(8):
            for b in range(8):
                if a == b: continue
                V.append((fam, f'{fam}{a}{b}', lambda ib, a=a, b=b, f=bf: f(ib[a], ib[b])))
        for a in range(8):
            for b in range(8):
                if a == b: continue
                V.append((fam + '-NOT', f'{fam}-NOT{a}{b}', lambda ib, a=a, b=b, f=bf: f(ib[a], 1 - ib[b])))
    for j, k, l in combinations(range(8), 3):
        V.append(('Maj', f'Maj{j}{k}{l}', lambda ib, j=j, k=k, l=l: 1 if ib[j] + ib[k] + ib[l] >= 2 else 0))
    for j, k, l in permutations(range(8), 3):
        V.append(('Ch', f'Ch{j}{k}{l}', lambda ib, j=j, k=k, l=l: ib[k] if ib[j] else ib[l]))
    return V
_VOCAB = _vocab()
_DOM = [[(x >> (7 - i)) & 1 for i in range(8)] for x in range(256)]
_VTT = [(fam, lab, fn, tuple(fn(ib) for ib in _DOM)) for fam, lab, fn in _VOCAB]

def decompose(h):
    """hypothesis -> per-position canonical vocab labels (or None if any position unexpressible)."""
    strs = [format(x, '08b') for x in range(256)]
    outs = [apply_word(h, s) for s in strs]
    sel = []
    for p in range(8):
        tt = tuple(int(o[p]) for o in outs)
        hit = next(((fam, lab, fn) for fam, lab, fn, vtt in _VTT if vtt == tt), None)
        if hit is None: return None
        sel.append(hit)
    return sel

# ───────────────────────── emission helpers ─────────────────────────
def col_hash(col): return 'a' if len(set(col)) == 1 else str(col.count('1'))

def family_cands(incols):
    """family -> ordered [(label, col, display_label)] exactly per template grammar."""
    n = len(incols[0])
    F = {}
    F['Identity'] = [(str(j), incols[j]) for j in range(8)]
    F['NOT'] = [(str(j), _not(incols[j])) for j in range(8)]
    F['Constant'] = [('0', '0' * n), ('1', '1' * n)]
    for fam, bf in (('AND', lambda x, y: ''.join(str(int(a) & int(b)) for a, b in zip(x, y))),
                    ('OR', lambda x, y: ''.join(str(int(a) | int(b)) for a, b in zip(x, y))),
                    ('XOR', lambda x, y: ''.join(str(int(a) ^ int(b)) for a, b in zip(x, y)))):
        rows = []
        for dist in range(1, 5):
            for a in range(8):
                b = (a + dist) % 8
                if dist == 4 and a >= 4: continue
                rows.append((f'{a}{b} {b}{a}', bf(incols[a], incols[b]), dist))
        F[fam] = rows
        nrows = []
        for dist in range(1, 8):
            for a in range(8):
                b = (a + dist) % 8
                nrows.append((f'{a}{b}', bf(incols[a], _not(incols[b])), dist))
        F[fam + '-NOT'] = nrows
    return F

def fam_prefix(fam, label):
    if fam == 'Identity': return 'I' + label
    if fam == 'NOT': return 'NOT' + label
    if fam == 'Constant': return 'C' + label
    return fam + label

def chain_walk(fam, start_label, start_pos, direction, matches):
    """Walk a shift chain from (start_pos,label). direction -1: pos descends (Right anchor 7,
    indices -1); +1: pos ascends (Left anchor 0, indices +1). Constant labels stay fixed.
    Returns (visited_labels_with_x, run_length)."""
    def step(lab):
        if fam == 'Constant': return lab
        return ''.join(str((int(c) + direction) % 8) for c in lab)
    labs = [start_label]; run = 1
    lab = start_label
    p = start_pos
    while run < 8:
        p = p - 1 if direction == -1 else p + 1
        if p < 0 or p > 7: break
        lab = step(lab)
        if any(_canon(fam, m) == _canon(fam, lab) for m in matches.get(p, [])):
            labs.append(lab); run += 1
        else:
            labs.append(lab + 'x'); break
    return labs, run

def _canon(fam, label):
    label = label.split()[0]
    if fam in ('AND', 'OR', 'XOR') and len(label) == 2:
        return ''.join(sorted(label))
    return label

# ───────────────────────── the generator ─────────────────────────
def gen_cot(folder):
    qtext = open(os.path.join(folder, 'question.txt')).read()
    exs, query = parse_question(qtext)
    n = len(exs)
    incols = [''.join(e[0][j] for e in exs) for j in range(8)]
    outcols = [''.join(e[1][p] for e in exs) for p in range(8)]

    h = word_solve(exs)
    if h is None or h[0] == 'TERM3': return None
    sel = decompose(h)
    if sel is None: return None

    F = family_cands(incols)
    matchmap = {}   # fam -> {pos: [labels]}
    for fam in FAMS:
        mm = {}
        for row in F[fam]:
            lab, col = row[0], row[1]
            for p in range(8):
                if col == outcols[p]:
                    for piece in lab.split():
                        mm.setdefault(p, []).append(piece)
        matchmap[fam] = mm

    L = []
    L.append('We need to deduce the transformation by matching the example outputs.')
    L.append('')
    for i, (_, out) in enumerate(exs):
        L.append(f'Output {i + 1}: {out}')
        for p in range(8): L.append(f'{p} {out[p]}')
        L.append('')
    L.append('Output bit columns (with bitsum as hash)')
    for p in range(8): L.append(f'{p} {outcols[p]} {col_hash(outcols[p])}')
    L.append('')

    chains = {}  # fam -> {'L':(lines,best_label_list,run) , 'R':...}
    for fam in FAMS:
        rows = F[fam]
        lastdist = None
        for row in rows:
            lab, col = row[0], row[1]
            dist = row[2] if len(row) > 2 else None
            if dist is not None and lastdist is not None and dist != lastdist: L.append('')
            lastdist = dist
            ps = [str(p) for p in range(8) if col == outcols[p]]
            L.append(f'{lab} {col} {col_hash(col)}' + (f' match {" ".join(ps)}' if ps else ''))
        L.append('')
        L.append(f'Matching output with {fam}')
        mm = matchmap[fam]
        for p in range(8):
            L.append(f'{p} ' + (' '.join(mm[p]) if p in mm else 'absent'))
        L.append('')
        cinfo = {}
        for side, anchor, ddir in (('Left', 0, +1), ('Right', 7, -1)):
            L.append(side)
            best = ([], 0)
            if anchor in mm:
                for st in mm[anchor]:
                    labs, run = chain_walk(fam, st, anchor, ddir, mm)
                    L.append(' '.join(labs))
                    if run > best[1]: best = ([l.rstrip('x') for l in labs[:run]], run)
            if anchor not in mm: L.append('none')
            if best[1] > 0:
                L.append('Best: ' + ' '.join(fam_prefix(fam, l) for l in best[0]) + f': {best[1]}')
            else:
                L.append('Best: none')
            L.append('')
            cinfo[side] = best
        chains[fam] = cinfo

    # ---------- Selecting ----------
    L.append('Selecting'); L.append('')
    for side in ('Left', 'Right'):
        L.append(side + 's')
        for fam in FAMS:
            labs, run = chains[fam][side]
            L.append(f'{fam} ' + (' '.join(fam_prefix(fam, l) for l in labs) + f': {run}' if run else 'none'))
        L.append('')
    longest = {s: max(chains[f][s][1] for f in FAMS) for s in ('Left', 'Right')}
    L.append(f'Left longest: {longest["Left"]}')
    L.append(f'Right longest: {longest["Right"]}')
    L.append('')
    # winner: prefer the chain consistent with the per-bit assignment (the deterministic prior),
    # else the first family reaching the longest run
    def assignment_fam(p): return sel[p][0]
    def consistent(fam, side):
        labs, run = chains[fam][side]
        if not run: return False
        positions = range(0, run) if side == 'Left' else range(7, 7 - run, -1)
        return all(_canon(fam, labs[i]) == _canon(fam, re.sub(r'^[A-Za-z-]+', '', sel[p][1])) and assignment_fam(p) == fam
                   for i, p in enumerate(positions))
    winner = {}
    for side in ('Left', 'Right'):
        cands = [f for f in FAMS if chains[f][side][1] == longest[side] and longest[side] > 0]
        pick = next((f for f in cands if consistent(f, side)), cands[0] if cands else None)
        winner[side] = pick
        L.append(f'{side} winner: ' + ', '.join(f'{f} ' + ('yes' if f == pick else 'no') for f in FAMS))
    L.append('')
    for side in ('Left', 'Right'):
        f = winner[side]
        L.append(f'Best {side.lower()}: ' + (' '.join(fam_prefix(f, l) for l in chains[f][side][0]) + f': {chains[f][side][1]}' if f else 'none'))
    L.append('')
    ll = chains[winner['Left']]['Left'][1] if winner['Left'] else 0
    rl = chains[winner['Right']]['Right'][1] if winner['Right'] else 0
    if ll + rl > 8:
        if rl >= ll: ll = 8 - rl
        else: rl = 8 - ll
    for side, eff in (('Left', ll), ('Right', rl)):
        f = winner[side]
        labs = chains[f][side][0][:eff] if f else []
        L.append(f'Truncated {side.lower()}: ' + (' '.join(fam_prefix(f, l) for l in labs) + f': {eff}' if f and eff else 'none'))
    L.append('')
    # tentative assignment from the truncated chains — only keep chain rules that AGREE with
    # the final assignment (deterministic prior); others stay pending and resolve in Matching.
    tent = {}
    if winner['Right'] and rl:
        f = winner['Right']
        for i, lab in enumerate(chains[f]['Right'][0][:rl]):
            p = 7 - i
            if assignment_fam(p) == f and _canon(f, lab) == _canon(f, re.sub(r'^[A-Za-z-]+', '', sel[p][1])):
                tent[p] = fam_prefix(f, lab)
    L.append('Tentative from right')
    for p in range(7, -1, -1): L.append(f'{p} ' + tent.get(p, 'pending'))
    L.append('')
    if winner['Left'] and ll:
        f = winner['Left']
        for i, lab in enumerate(chains[f]['Left'][0][:ll]):
            if assignment_fam(i) == f and _canon(f, lab) == _canon(f, re.sub(r'^[A-Za-z-]+', '', sel[i][1])):
                tent[i] = fam_prefix(f, lab)
    L.append('Tentative')
    for p in range(8): L.append(f'{p} ' + tent.get(p, 'pending'))
    L.append('')
    # extrapolation keys for pending positions, from the winning side's pattern
    keys = {}
    base_fam = winner['Left'] if (winner['Left'] and ll >= rl) else winner['Right']
    base_side = 'Left' if (winner['Left'] and ll >= rl) else 'Right'
    if base_fam:
        head = chains[base_fam][base_side][0][0]
        anchor = 0 if base_side == 'Left' else 7
        for p in range(8):
            if p in tent: continue
            if base_fam == 'Constant': continue
            keys[p] = ''.join(str((int(c) + (p - anchor)) % 8) for c in head)
    L.append('Preferred from left')
    for p in range(8): L.append(f'{p} ' + (tent[p] if p in tent else ('?' + keys[p] if p in keys else '?')))
    L.append('')
    L.append('Preferred')
    for p in range(8): L.append(f'{p} ' + (tent[p] if p in tent else ('?' + keys[p] if p in keys else '?')))
    L.append('')
    # Matching rescan: for each pending position list, per family, the candidate the procedure
    # adopts (the assignment's own label when that family fits, else the first table match)
    L.append('Matching')
    resolved = dict(tent)
    pending_after = []
    for p in range(8):
        if p in tent:
            L.append(f'{p} {tent[p]}'); continue
        parts = []
        for fam in FAMS:
            mm = matchmap[fam].get(p, [])
            if not mm: parts.append(f'{fam} absent'); continue
            own = re.sub(r'^[A-Za-z-]+', '', sel[p][1]) if sel[p][0] == fam else None
            lab = own if own and _canon(fam, own) in {_canon(fam, m) for m in mm} else mm[0]
            parts.append(fam_prefix(fam, lab))
        L.append(f'{p} ' + ('?' + keys.get(p, '') + ' - ' if p in keys or True else '') + ', '.join(parts))
        famp = sel[p][0]
        if famp in FAMS:
            resolved[p] = sel[p][1]
        else:
            pending_after.append(p)
    L.append('')
    L.append('Perfect match')
    for fam in FAMS:
        pend = [p for p in range(8) if p not in tent]
        ok = bool(pend) and all(matchmap[fam].get(p) for p in pend)
        L.append(f'{fam} ' + ('yes' if ok and all(sel[p][0] == fam for p in pend) else 'no'))
    L.append('')
    L.append('Matched')
    for p in range(8): L.append(f'{p} ' + (resolved[p] if p in resolved else 'none'))
    L.append('')


    # ---------- conditional 10th family: Maj (native grammar; emitted only when some position
    # matches none of the 9 families and the assignment resolves it with Maj) ----------
    if pending_after:
        if any(sel[p][0] == 'Ch' for p in pending_after):
            return 'NEEDS_CH'                       # held until the Ch wording is decided
        L.append('Maj')
        maj_mm = {}
        for j, k, l in combinations(range(8), 3):
            col = ''.join('1' if int(a) + int(b) + int(c) >= 2 else '0'
                          for a, b, c in zip(incols[j], incols[k], incols[l]))
            ps2 = [str(p) for p in range(8) if col == outcols[p]]
            for p in range(8):
                if col == outcols[p]: maj_mm.setdefault(p, []).append(f'{j}{k}{l}')
            L.append(f'{j}{k}{l} {col} {col_hash(col)}' + (f' match {" ".join(ps2)}' if ps2 else ''))
        L.append('')
        L.append('Matching output with Maj')
        for p in range(8):
            L.append(f'{p} ' + (' '.join(maj_mm[p]) if p in maj_mm else 'absent'))
        L.append('')
        L.append('Matched')
        for p in range(8):
            if p in resolved: L.append(f'{p} {resolved[p]}')
            elif sel[p][0] == 'Maj' and re.sub(r'^Maj', '', sel[p][1]) in maj_mm.get(p, []):
                resolved[p] = sel[p][1]; L.append(f'{p} {sel[p][1]}')
            else:
                return None
        L.append('')

    L.append('Selected')
    for p in range(8): L.append(f'{p} {resolved[p]}')
    L.append('')

    # ---------- Applying (the answer is COMPUTED from the Selected rules) ----------
    qb = [int(c) for c in query]
    L.append(f'Applying to {query}')
    L.append('Input')
    for p in range(8): L.append(f'{p} {query[p]}')
    L.append('Output')
    fnmap = {lab: fn for _, lab, fn in _VOCAB}
    ans = []
    for p in range(8):
        lab = resolved[p]
        fn = fnmap[lab]
        bit = fn(qb)
        ans.append(str(bit))
        m = re.match(r'(I|NOT|C)(\d)$', lab)
        if m:
            t, a = m.group(1), int(m.group(2))
            if t == 'I': L.append(f'{p} {lab} = {qb[a]}')
            elif t == 'NOT': L.append(f'{p} {lab} = NOT({qb[a]}) = {bit}')
            else: L.append(f'{p} {lab} = {a}')
            continue
        m = re.match(r'(AND|OR|XOR)(-NOT)?(\d)(\d)$', lab)
        if m:
            op, nt, a, b = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            if nt: L.append(f'{p} {lab} = {op}({qb[a]},NOT({qb[b]})) = {bit}')
            else: L.append(f'{p} {lab} = {op}({qb[a]},{qb[b]}) = {bit}')
            continue
        m = re.match(r'(Maj|Ch)(\d)(\d)(\d)$', lab)
        if m:
            t, a, b, c = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            L.append(f'{p} {lab} = {t}({qb[a]},{qb[b]},{qb[c]}) = {bit}')
            continue
    L.append('')
    L.append('I will now return the answer in \\boxed{}')
    L.append('The answer in \\boxed is')
    L.append('\\boxed{' + ''.join(ans) + '}')
    return '\n'.join(L)


if __name__ == '__main__':
    import sys
    print(gen_cot(sys.argv[1]))

# ───────────── extended prior (approved notations: Ch, NOT-wrap, 3-term) ─────────────
_OPS3 = {'AND': lambda x, y: x & y, 'OR': lambda x, y: x | y, 'XOR': lambda x, y: x ^ y}

def word_solve_full(exs):
    """word_solve + explicit 3-term hypothesis. Blind. Returns ('T3',o1,u1,o2,u2,u3) for 3-term."""
    h = word_solve(exs)
    if h is not None and h[0] != 'TERM3':
        return h
    tgt = int(''.join(b for _, b in exs), 2)
    U = {n: int(''.join(f(a) for a, _ in exs), 2) for n, f in _UN}
    for pool in (_SHN, _UNN):                     # plain shifts first: expressible in the approved notation
        for o2, f2 in _OPS3.items():
            inner = []
            for i in range(len(pool)):
                a = U[pool[i]]
                for j in range(i + 1, len(pool)):
                    inner.append((pool[i], pool[j], f2(a, U[pool[j]])))
            for o1, f1 in _OPS3.items():
                for u1 in pool:
                    a = U[u1]
                    for n2, n3, v in inner:
                        if f1(a, v) == tgt: return ('T3', o1, u1, o2, n2, n3)
    return None

def apply_word_full(h, s):
    if h[0] != 'T3': return apply_word(h, s)
    _, o1, u1, o2, n2, n3 = h
    f = lambda o, x, y: str(_OPS3[o](int(x), int(y)))
    A, B, C = _FN[u1](s), _FN[n2](s), _FN[n3](s)
    inner = ''.join(f(o2, x, y) for x, y in zip(B, C))
    return ''.join(f(o1, x, y) for x, y in zip(A, inner))

def _ext_vocab():
    """tier-ordered: 9 families, Maj, Ch, NOT-pairs, NOT-Maj, 3-term."""
    V = [(lab, fn) for _, lab, fn in _VOCAB if not lab.startswith(('Maj', 'Ch'))]
    V += [(lab, fn) for _, lab, fn in _VOCAB if lab.startswith('Maj')]
    V += [(lab, fn) for _, lab, fn in _VOCAB if lab.startswith('Ch')]
    for fam, bf in (('AND', lambda x, y: x & y), ('OR', lambda x, y: x | y), ('XOR', lambda x, y: x ^ y)):
        for a in range(8):
            for b in range(a + 1, 8):
                V.append((f'NOT-{fam}{a}{b}', lambda ib, a=a, b=b, f=bf: 1 - f(ib[a], ib[b])))
    from itertools import combinations as _c
    for j, k, l in _c(range(8), 3):
        V.append((f'NOT-Maj{j}{k}{l}', lambda ib, j=j, k=k, l=l: 0 if ib[j] + ib[k] + ib[l] >= 2 else 1))
    for o1 in ('AND', 'OR', 'XOR'):
        for o2 in ('AND', 'OR', 'XOR'):
            for a in range(8):
                for b in range(8):
                    for c in range(b + 1, 8):
                        if a in (b, c): continue
                        V.append((f'{o1}{a}({o2}{b}{c})',
                                  lambda ib, a=a, b=b, c=c, o1=o1, o2=o2: _OPS3[o1](ib[a], _OPS3[o2](ib[b], ib[c]))))
    return V
_EXT = _ext_vocab()
_EXT_TT = None
_EXT_TT_MAP = None
def _ext_tt():
    global _EXT_TT, _EXT_TT_MAP
    if _EXT_TT is None:
        seen = {}; out = []
        for lab, fn in _EXT:
            tt = tuple(fn(ib) for ib in _DOM)
            if tt in seen: continue
            seen[tt] = lab; out.append((lab, fn, tt))
        _EXT_TT = out; _EXT_TT_MAP = seen
    return _EXT_TT
def ext_label(tt):
    """O(1) truth-table -> canonical extended-vocab label (None if not expressible)."""
    if _EXT_TT_MAP is None: _ext_tt()
    return _EXT_TT_MAP.get(tt)

PREF_NONFAM = '\x00none'

def full_prior(exs):
    """The blind prior: per-position labels from the first-fitting word hypothesis,
    decomposed against the extended vocabulary (tier-ordered canonical). None if no hypothesis."""
    h = word_solve_full(exs)
    if h is None: return None
    strs = [format(x, '08b') for x in range(256)]
    outs = [apply_word_full(h, s) for s in strs]
    pref = []
    for p in range(8):
        tt = tuple(int(o[p]) for o in outs)
        lab = ext_label(tt)
        pref.append(lab if lab else PREF_NONFAM)
    return pref
