#!/usr/bin/env python3
"""THE faithful solver — one blind deterministic function: question.txt -> CoT.

= huikang's UNSTEERED machinery + honest scan tiers (self-consistent, his own logic),
  followed by an explicit 'Pattern consistency check' that openly revises positions where the
  global word-level rule disagrees with huikang's chain pick on the query. No silent steering;
  no gold used (blind). The boxed answer is COMPUTED from the (possibly revised) Selected rules.
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _gen_bitm_v3 as V
import _pattern_check as PC
from _store_types import Problem, Example

def _apply_lines(revised, query):
    qb = [int(c) for c in query]
    L = [f'Applying to {query}', 'Input']
    for i, c in enumerate(query): L.append(f'{i} {c}')
    L.append('Output'); ans = []
    for p in range(8):
        lab = revised[p]; v = PC._eval_label(lab, qb); ans.append(str(v))
        m = re.match(r'I(\d)$', lab)
        if m: L.append(f'{p} {lab} = {qb[int(m.group(1))]}'); continue
        m = re.match(r'NOT(\d)$', lab)
        if m: L.append(f'{p} {lab} = NOT({qb[int(m.group(1))]}) = {v}'); continue
        m = re.match(r'C(\d)$', lab)
        if m: L.append(f'{p} {lab} = {m.group(1)}'); continue
        m = re.match(r'(AND|OR|XOR)(-NOT)?(\d)(\d)$', lab)
        if m:
            o, nt, a, b = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
            yb = f'NOT({qb[b]})' if nt else f'{qb[b]}'
            L.append(f'{p} {lab} = {o}({qb[a]},{yb}) = {v}'); continue
        m = re.match(r'Maj(\d)(\d)(\d)$', lab)
        if m:
            j, k, l = (int(m.group(i)) for i in (1, 2, 3))
            L.append(f'{p} {lab} = Maj({qb[j]},{qb[k]},{qb[l]}) = {v}'); continue
        m = re.match(r'Ch(\d)(\d)(\d)$', lab)
        if m:
            s, a, b = (int(m.group(i)) for i in (1, 2, 3))
            L.append(f'{p} {lab} = Ch({qb[s]},{qb[a]},{qb[b]}) = {v}'); continue
        m = re.match(r'NOT-(AND|OR|XOR)(\d)(\d)$', lab)
        if m:
            o, a, b = m.group(1), int(m.group(2)), int(m.group(3))
            L.append(f'{p} {lab} = NOT({o}({qb[a]},{qb[b]})) = {v}'); continue
        m = re.match(r'NOT-Maj(\d)(\d)(\d)$', lab)
        if m:
            j, k, l = (int(m.group(i)) for i in (1, 2, 3))
            L.append(f'{p} {lab} = NOT(Maj({qb[j]},{qb[k]},{qb[l]})) = {v}'); continue
        m = re.match(r'(AND|OR|XOR)(\d)\((AND|OR|XOR)(\d)(\d)\)$', lab)
        if m:
            o1, a, o2, b, c = m.group(1), int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5))
            L.append(f'{p} {lab} = {o1}({qb[a]},{o2}({qb[b]},{qb[c]})) = {v}'); continue
        L.append(f'{p} {lab} = {v}')
    L += ['', 'I will now return the answer in \\boxed{}', 'The answer in \\boxed is', '\\boxed{' + ''.join(ans) + '}']
    return '\n'.join(L)

def _ambiguous_majch(base, picks):
    """True if a Selected Maj/Ch position had >=2 equally-matching candidates in its scan tier
    (the 'arbitrary pick' case the global-rule confirmation justifies)."""
    for tier in ('Maj', 'Ch'):
        mt = re.search(rf'Matching output with {tier}\n((?:\d .+\n)+)', base)
        if not mt: continue
        for ln in mt.group(1).strip().split('\n'):
            parts = ln.split()
            cands = parts[1:]
            if len(cands) >= 2 and cands[0] != 'absent' and picks.get(int(parts[0]), '').strip().startswith(tier):
                return True
    return False

def gen_cot(folder):
    qtext = open(os.path.join(folder, 'question.txt')).read()
    exs = re.findall(r'([01]{8}) -> ([01]{8})', qtext)
    query = re.search(r'determine the output for: ([01]{8})', qtext).group(1)
    # huikang UNSTEERED + tiers (self-consistent, blind)
    V._TIER3 = True; V._PREFERRED = None; V._SECTION_PREF.clear()
    base = V.reasoning_bit_manipulation(Problem(id=os.path.basename(folder), category='bit_manipulation',
            examples=[Example(a, b) for a, b in exs], question=query, answer=''))
    m = re.search(r'\nSelected\n((?:\d .+\n){8})', base)
    picks = {int(l.split(' ', 1)[0]): l.split(' ', 1)[1] for l in m.group(1).strip().split('\n')}
    block, revised = PC.build_block(exs, query, picks)
    if block is None:
        head = base[:m.start()].rstrip('\n')
        Hc, prefc = PC.per_pos_labels(exs)
        # no single rule reproduces every example -> state the fallback explicitly (don't go silent)
        if Hc is None:
            fb = 'No single rule reproduces all examples; each output bit is taken from its own column match.'
            return head + '\n\n' + fb + '\n' + base[m.start():].lstrip('\n')
        # 3-step (composite) rule: state it (don't go silent) for ANY T3 rule -- BUT only when the stated rule
        # actually produces the boxed answer (on ambiguous puzzles word_solve_full's H fits the examples yet
        # disagrees with the per-column box on the query; stating it then would contradict the box).
        if Hc[0] == 'T3' or any(p == PC.W.PREF_NONFAM for p in prefc):
            box_base = re.findall(r'\\boxed\{([^}]*)\}', base)[-1]
            if PC.W.apply_word_full(Hc, query) == box_base:
                rl = f'The resolved positions follow a single rule: {PC.render_H(Hc)}.'
                return head + '\n\n' + rl + '\n' + base[m.start():].lstrip('\n')
            return base   # rule disagrees with the box (ambiguous) -> stay silent; the build filter rejects it
        # no revision needed; if an ambiguous Maj/Ch pick was made, state the global rule to justify it
        if _ambiguous_majch(base, picks):
            cblock = PC.build_confirm_block(exs, picks)
            if cblock is not None:
                return head + '\n\n' + '\n'.join(cblock) + '\n' + base[m.start():].lstrip('\n')
        return base                                            # huikang natural, no revision
    head = base[:m.start()].rstrip('\n')                       # everything up to (not incl.) Selected
    sel = 'Selected\n' + '\n'.join(f'{p} {revised[p]}' for p in range(8))
    return head + '\n\n' + '\n'.join(block) + '\n' + sel + '\n\n' + _apply_lines(revised, query)

if __name__ == '__main__':
    print(gen_cot(sys.argv[1]))
