#!/usr/bin/env python3
"""Build the three PROPOSED notation example files (forward-only logs, huikang grammar).
Base trace = the one solver (tier mode) up to the Maj scan; then the proposed tier sections,
computed from REAL scans of this puzzle; lock; Selected; Applying; boxed = rules applied."""
import sys, re, os
sys.path.insert(0, '.')
import _gen_bitm_v3 as V
from _store_types import Problem, Example
V._TIER3 = True; V._PREFERRED = None

def cols(exs):
    inc = [''.join(e[0][j] for e in exs) for j in range(8)]
    out = [''.join(e[1][p] for e in exs) for p in range(8)]
    return inc, out
def hsh(col): return 'a' if len(set(col)) == 1 else str(col.count('1'))

def ch_scan(inc, outc, pend, invert=False):
    """Forward scan: selector s asc, ones-branch a asc, zeros-branch b asc. Hits only."""
    n = len(inc[0]); hits = []  # (label, col)
    for s in range(8):
        ones = [i for i in range(n) if inc[s][i] == '1']
        zeros = [i for i in range(n) if inc[s][i] == '0']
        if not ones or not zeros: continue
        for a in range(8):
            if a == s: continue
            ca = ''.join('1' if c == '0' else '0' for c in inc[a]) if invert else inc[a]
            for b in range(8):
                if b == s or b == a: continue
                cb = ''.join('1' if c == '0' else '0' for c in inc[b]) if invert else inc[b]
                col = ''.join(ca[i] if inc[s][i] == '1' else cb[i] for i in range(n))
                if any(col == outc[p] for p in pend):
                    lab = f'{s}{a}{b}'
                    if lab not in [h[0] for h in hits]: hits.append((lab, col))
    return hits

def t3_scan(inc, outc, pend):
    """Forward scan of OP1(x, OP2(y,z)): family pairs in fixed order, a asc, (b,c) asc. Hits only."""
    OPS = {'AND': lambda x, y: x & y, 'OR': lambda x, y: x | y, 'XOR': lambda x, y: x ^ y}
    n = len(inc[0]); res = {}  # (o1,o2) -> [(label,col)]
    for o1 in ('AND', 'OR', 'XOR'):
        for o2 in ('AND', 'OR', 'XOR'):
            for a in range(8):
                for b in range(8):
                    for c in range(b + 1, 8):
                        if a in (b, c): continue
                        col = ''.join(str(OPS[o1](int(inc[a][i]), OPS[o2](int(inc[b][i]), int(inc[c][i])))) for i in range(n))
                        if any(col == outc[p] for p in pend):
                            res.setdefault((o1, o2), []).append((f'{a}({b}{c})', col))
    return res

def section(name, hits, outc):
    L = [name]
    for lab, col in hits:
        ms = [str(p) for p in range(8) if col == outc[p]]
        L.append(f'{lab} {col} {hsh(col)}' + (f' match {" ".join(ms)}' if ms else ''))
    L += ['', f'Matching output with {name}']
    for p in range(8):
        labs = [lab for lab, col in hits if col == outc[p]]
        L.append(f'{p} ' + (' '.join(labs) if labs else 'absent'))
    L.append('')
    return L

def hyp_tt(d):
    """Per-position truth tables of the BLIND word hypothesis (examples only)."""
    import _gen_bitm as W
    q = open(os.path.join(d, 'question.txt')).read()
    exs = re.findall(r'([01]{8}) -> ([01]{8})', q)
    h = W.word_solve(exs)
    strs = [format(x, '08b') for x in range(256)]
    if h is not None and h[0] != 'TERM3':
        outs = [W.apply_word(h, s) for s in strs]
    else:
        OPS = {'AND': lambda x, y: x & y, 'OR': lambda x, y: x | y, 'XOR': lambda x, y: x ^ y}
        tgt = int(''.join(b for _, b in exs), 2)
        U = {n: int(''.join(f(a) for a, _ in exs), 2) for n, f in W._UN}
        h3 = None
        for o2 in ('AND', 'OR', 'XOR'):
            for o1 in ('AND', 'OR', 'XOR'):
                for u1 in W._UNN:
                    a = U[u1]
                    for i in range(len(W._UNN)):
                        for j in range(i + 1, len(W._UNN)):
                            f = lambda o, x, y: {'AND': x & y, 'OR': x | y, 'XOR': x ^ y}[o]
                            if f(o1, a, f(o2, U[W._UNN[i]], U[W._UNN[j]])) == tgt:
                                h3 = (o1, u1, o2, W._UNN[i], W._UNN[j]); break
                        if h3: break
                    if h3: break
                if h3: break
            if h3: break
        o1, u1, o2, n2, n3 = h3
        def ap(s):
            f = lambda o, x, y: str({'AND': int(x) & int(y), 'OR': int(x) | int(y), 'XOR': int(x) ^ int(y)}[o])
            A, B, C = W._FN[u1](s), W._FN[n2](s), W._FN[n3](s)
            inner = ''.join(f(o2, x, y) for x, y in zip(B, C))
            return ''.join(f(o1, x, y) for x, y in zip(A, inner))
        outs = [ap(s) for s in strs]
    return {p: tuple(int(o[p]) for o in outs) for p in range(8)}, strs

def fn_tt(fn, strs):
    return tuple(fn([int(c) for c in s]) for s in strs)

def build(d, kind):
    q = open(os.path.join(d, 'question.txt')).read()
    exs = re.findall(r'([01]{8}) -> ([01]{8})', q)
    qq = re.search(r'determine the output for: ([01]{8})', q).group(1)
    gold = open(os.path.join(d, 'answer.txt')).read().strip()
    inc, outc = cols(exs)
    H, strs = hyp_tt(d)
    import _gen_bitm as W2
    pref = []
    for p in range(8):
        hit = next((lab for fam, lab, fn, vtt in W2._VTT if vtt == H[p] and not lab.startswith(('Maj', 'Ch'))), None)
        if hit is None:
            hit = next((lab for fam, lab, fn, vtt in W2._VTT if vtt == H[p] and lab.startswith('Maj')), None)
        pref.append(hit if hit else '\x00none')
    V._SECTION_PREF.clear(); V._PREFERRED = pref
    try:
        base = V.reasoning_bit_manipulation(Problem(id=d, category='bit_manipulation',
                examples=[Example(a, b) for a, b in exs], question=qq, answer=''))
    finally:
        V._PREFERRED = None; V._SECTION_PREF.clear()
    lines = base.splitlines()
    # cut after the Maj "Matching output with Maj" block (the engine's tier emits it when pending)
    mi = lines.index('Matching output with Maj')
    cut = mi + 9  # header + 8 position lines
    head = lines[:cut + 1]                      # includes trailing blank
    # pending = positions whose engine Matched shows default
    mtail = lines[cut:]
    mm = mtail[mtail.index('Matched') + 1: mtail.index('Matched') + 9]
    pend = [int(l.split()[0]) for l in mm if 'default 1' in l]
    sel = {int(l.split(' ', 1)[0]): l.split(' ', 1)[1] for l in mm}
    L = head + ['Matched']
    for p in range(8): L.append(f'{p} ' + ('none' if p in pend else sel[p]))
    L.append('')
    # tier sections per kind
    resolved = dict(sel)
    if kind == 'ch':
        hits = ch_scan(inc, outc, pend)
        L += section('Ch', hits, outc)
        for p in pend:
            cands = [l for l, c in hits if c == outc[p]]
            pick = next((l for l in cands if fn_tt(lambda ib, l=l: ib[int(l[1])] if ib[int(l[0])] else ib[int(l[2])], strs) == H[p]), cands[0])
            resolved[p] = 'Ch' + pick
    if kind == 'notmaj':
        from itertools import combinations as _cmb
        n = len(inc[0])
        # plain Maj already scanned (absent). NOT-wrap tier: inverted pair ops, then inverted Maj. Hits only.
        OPS = {'AND': lambda x, y: x & y, 'OR': lambda x, y: x | y, 'XOR': lambda x, y: x ^ y}
        nsec = {}
        for fam in ('AND', 'OR', 'XOR'):
            hh = []
            for a in range(8):
                for b in range(a + 1, 8):
                    col = ''.join(str(1 - OPS[fam](int(inc[a][i]), int(inc[b][i]))) for i in range(n))
                    if any(col == outc[p] for p in pend): hh.append((f'{a}{b}', col))
            if hh: nsec['NOT-' + fam] = hh
        mh = []
        for j, k, l2 in _cmb(range(8), 3):
            col = ''.join('0' if int(inc[j][i]) + int(inc[k][i]) + int(inc[l2][i]) >= 2 else '1' for i in range(n))
            if any(col == outc[p] for p in pend): mh.append((f'{j}{k}{l2}', col))
        if mh: nsec['NOT-Maj'] = mh
        for name, hh in nsec.items():
            L += section(name, hh, outc)
        for p in pend:
            done = False
            for name, hh in nsec.items():
                for lab, c in hh:
                    if c != outc[p]: continue
                    if name == 'NOT-Maj':
                        f = lambda ib, lab=lab: 0 if ib[int(lab[0])] + ib[int(lab[1])] + ib[int(lab[2])] >= 2 else 1
                    else:
                        fam = name[4:]
                        f = lambda ib, lab=lab, fam=fam: 1 - OPS[fam](ib[int(lab[0])], ib[int(lab[1])])
                    if fn_tt(f, strs) == H[p]:
                        resolved[p] = name + lab; done = True; break
                if done: break
            if not done:
                for name, hh in nsec.items():
                    for lab, c in hh:
                        if c == outc[p]: resolved[p] = name + lab; done = True; break
                    if done: break
    if kind == 't3':
        hits = ch_scan(inc, outc, pend)
        L += section('Ch', hits, outc)
        res = t3_scan(inc, outc, pend)
        OPS = {'AND': lambda x, y: x & y, 'OR': lambda x, y: x | y, 'XOR': lambda x, y: x ^ y}
        for (o1, o2), hh in sorted(res.items()):
            L += section(f'{o1}({o2})', hh, outc)
        for p in pend:
            done = False
            for (o1, o2), hh in sorted(res.items()):
                for lab, c in hh:
                    if c != outc[p]: continue
                    a, bc = int(lab[0]), lab.split('(')[1][:-1]
                    b, cc = int(bc[0]), int(bc[1])
                    f = lambda ib, a=a, b=b, cc=cc, o1=o1, o2=o2: OPS[o1](ib[a], OPS[o2](ib[b], ib[cc]))
                    if fn_tt(f, strs) == H[p]:
                        resolved[p] = f'{o1}{a}({o2}{b}{cc})'; done = True; break
                if done: break
            if not done:
                for (o1, o2), hh in sorted(res.items()):
                    for lab, c in hh:
                        if c == outc[p]:
                            a, bc = int(lab[0]), lab.split('(')[1][:-1]
                            resolved[p] = f'{o1}{a}({o2}{bc})'; done = True; break
                    if done: break
    L.append('Matched')
    for p in range(8): L.append(f'{p} {resolved[p]}')
    L.append('')
    L.append('Selected')
    for p in range(8): L.append(f'{p} {resolved[p]}')
    L.append('')
    # Applying
    qb = qq
    def bit(i): return qb[i]
    def nb(v): return '1' if v == '0' else '0'
    L.append(f'Applying to {qq}'); L.append('Input')
    for i, c in enumerate(qq): L.append(f'{i} {c}')
    L.append('Output'); ans = []
    for p in range(8):
        lab = resolved[p]
        m = re.match(r'NOT-(AND|OR|XOR)(\d)(\d)$', lab)
        if m:
            o, a, b = m.group(1), int(m.group(2)), int(m.group(3))
            f = lambda o, x, y: str({'AND': int(x) & int(y), 'OR': int(x) | int(y), 'XOR': int(x) ^ int(y)}[o])
            inner = f(o, bit(a), bit(b)); v = nb(inner)
            L.append(f'{p} {lab} = NOT({o}({bit(a)},{bit(b)})) = {v}'); ans.append(v); continue
        m = re.match(r'NOT-Maj(\d)(\d)(\d)$', lab)
        if m:
            j, k, l2 = (int(m.group(i)) for i in (1, 2, 3))
            inner = '1' if int(bit(j)) + int(bit(k)) + int(bit(l2)) >= 2 else '0'
            v = nb(inner)
            L.append(f'{p} {lab} = NOT(Maj({bit(j)},{bit(k)},{bit(l2)})) = {v}'); ans.append(v); continue
        m = re.match(r'Ch(\d)(\d)(\d)$', lab)
        if m:
            s, a, b = (int(m.group(i)) for i in (1, 2, 3))
            v = bit(a) if bit(s) == '1' else bit(b)
            L.append(f'{p} {lab} = Ch({bit(s)},{bit(a)},{bit(b)}) = {v}'); ans.append(v); continue
        m = re.match(r'(AND|OR|XOR)(\d)\((AND|OR|XOR)(\d)(\d)\)$', lab)
        if m:
            o1, a, o2, b, c = m.group(1), int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5))
            f = lambda o, x, y: str({'AND': int(x) & int(y), 'OR': int(x) | int(y), 'XOR': int(x) ^ int(y)}[o])
            inner = f(o2, bit(b), bit(c)); v = f(o1, bit(a), inner)
            L.append(f'{p} {lab} = {o1}({bit(a)},{o2}({bit(b)},{bit(c)})) = {v}'); ans.append(v); continue
        m = re.match(r'Maj(\d)(\d)(\d)$', lab)
        if m:
            j, k, l2 = (int(m.group(i)) for i in (1, 2, 3))
            v = '1' if int(bit(j)) + int(bit(k)) + int(bit(l2)) >= 2 else '0'
            L.append(f'{p} {lab} = Maj({bit(j)},{bit(k)},{bit(l2)}) = {v}'); ans.append(v); continue
        m = re.match(r'I(\d)$', lab)
        if m: v = bit(int(m.group(1))); L.append(f'{p} {lab} = {v}'); ans.append(v); continue
        m = re.match(r'NOT(\d)$', lab)
        if m:
            v = nb(bit(int(m.group(1)))); L.append(f'{p} {lab} = NOT({bit(int(m.group(1)))}) = {v}'); ans.append(v); continue
        m = re.match(r'C(\d)$', lab)
        if m: v = m.group(1); L.append(f'{p} {lab} = {v}'); ans.append(v); continue
        m = re.match(r'(AND|OR|XOR)(-NOT)?(\d)(\d)$', lab)
        if m:
            o, nt, a, b = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            y = nb(bit(b)) if nt else bit(b)
            f = lambda o, x, y: str({'AND': int(x) & int(y), 'OR': int(x) | int(y), 'XOR': int(x) ^ int(y)}[o])
            v = f(o, bit(a), y)
            L.append(f'{p} {lab} = {o}({bit(a)},{"NOT(" + bit(b) + ")" if nt else bit(b)}) = {v}'); ans.append(v); continue
        raise ValueError(lab)
    L += ['', 'I will now return the answer in \\boxed{}', 'The answer in \\boxed is', '\\boxed{' + ''.join(ans) + '}']
    out = '\n'.join(L)
    okay = ''.join(ans) == gold
    fp = os.path.join(d, 'track', 'tree_cot_PROPOSED.txt')
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    open(fp, 'w').write(out)
    print(f'{d} [{kind}]: boxed={"".join(ans)} gold={gold} {"GT-OK" if okay else "MISMATCH"} -> {fp} ({len(L)} lines)')

build('bitm_000b53cf', 'ch')
build('bitm_796c8b63', 'notmaj')
build('bitm_0245b9bb', 't3')
