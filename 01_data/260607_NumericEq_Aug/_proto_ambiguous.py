#!/usr/bin/env python3
"""PROTOTYPE — ambiguous_arithmetic augmentation (the one category never generated).

Construction (mirrors how ALL 65 originals work): the QUERY operator is a sub-family op whose
examples are confined to ONE sign regime, so >=2 sub variants fit them all; the query operands flip
the regime, so the fitting variants DISAGREE on the query. The solver's ordered ~sub tie-break
(|a-b| -> a-b -> -|a-b| -> b-a) picks the first fit; we plant gold = that pick (for ambiguous, the
gold is underdetermined by the examples — exactly like the 36 orig GT-True ambiguous, where gold
happens to equal our pick). Blind GT-match stays a REAL filter: a construction/prediction bug
falls out as GT-match=False.

The 4 pair-types and their measured share of the 65 originals:
  case1 pick |a-b|  vs a-b    (46%)  examples P>Q, unsigned       query P<Q  -> positive gold
  case2 pick a-b    vs -|a-b| (32%)  examples P<Q, signed glyph   query P>Q  -> positive gold
  case3 pick |a-b|  vs b-a    (11%)  examples P<Q, unsigned       query P>Q  -> positive gold
  case4 pick -|a-b| vs b-a    ( 9%)  examples P>Q, signed glyph   query P<Q  -> NEGATIVE gold
                                     (gate-safe: examples already demonstrate the glyph)
Fillers come from add/mul/concat ONLY (a second sub-family op would let the solver's
distinct-operator family elimination kill the ambiguity), and the puzzle-level signed-conversion
is disabled (sign-ness comes from the query op's own regime).

Writes one example per case to _examples/ambiguous_case{1..4}/ and verifies each:
  classify == ambiguous_arithmetic, intended pick & dissenter, blind-solve GT-match, v3 invariants.
"""
import os, sys, random, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _gen_aug import (INTRO, ARITH_SYMS, assign_syms, operand, write_operand, write_output,
                      frame_result, good, Reject, variant_for, ARITH_PROB, RIGHTWARD_PROB,
                      LEADZERO_PROB, fits)
GEN = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260606_Numeric_Equation'
sys.path.insert(0, GEN)
import _gen_eq as S                                   # classify / consistent_ops / gen_cot / verify

CASES = {
    1: dict(V='absolute difference',        regime='P>Q', signed=False,
            pick='absolute difference',      dissent='subtraction (a-b)'),
    2: dict(V='subtraction (a-b)',          regime='P<Q', signed=True,
            pick='subtraction (a-b)',        dissent='negated absolute difference'),
    3: dict(V='reverse subtraction (b-a)',  regime='P<Q', signed=False,
            pick='absolute difference',      dissent='reverse subtraction (b-a)'),
    4: dict(V='reverse subtraction (b-a)',  regime='P>Q', signed=True,
            pick='negated absolute difference', dissent='reverse subtraction (b-a)'),
}

def in_regime(P, Q, regime): return (P > Q) if regime == 'P>Q' else (P < Q)

def build_ambiguous(case, rng):
    C = CASES[case]
    fams = ['sub'] + rng.sample(['add', 'mul', 'concat'], rng.choice([1, 2])); qi = 0
    signed, fmt = C['signed'], 'none'
    if signed:
        fmt = rng.choices(['prefix', 'suffix'], weights=[0.85, 0.15])[0]
    reading = 'leftward' if (signed and fmt == 'suffix') else ('rightward' if rng.random() < RIGHTWARD_PROB else 'leftward')
    ro = rr = (reading == 'leftward')
    leadzero = (reading == 'leftward' and rng.random() < LEADZERO_PROB)
    variants = [C['V']] + [variant_for(f, False, rng) for f in fams[1:]]
    want_arith = rng.random() < ARITH_PROB
    syms = assign_syms(fams, rng, want_arith)
    qsym = syms[qi]

    nex = rng.choice([3, 4, 5])
    counts = [1] * len(fams)
    while sum(counts) < nex:
        cand = [i for i in range(len(fams)) if counts[i] < 3]
        if not cand: break
        counts[rng.choice(cand)] += 1
    lines = []; trip = {}
    lead_done = not leadzero
    for k, (fam, V, sym, c) in enumerate(zip(fams, variants, syms, counts)):
        exs = []
        for j in range(c):
            for _t in range(400):
                wz = (leadzero and not lead_done)
                P = operand(ro, rng, want_leadzero=wz); Q = operand(ro, rng)
                if rng.random() < 0.5 and wz: P, Q = Q, P
                if k == qi:                                      # the ambiguous op: confine to the regime
                    if P == Q or not in_regime(P, Q, C['regime']): continue
                R = frame_result(P, Q, V)
                if R is None: continue
                sa, sb = write_operand(P, ro), write_operand(Q, ro)
                bad = (not wz and (not good(sa) or not good(sb)))
                if R.startswith('-') and R.lstrip('-')[-1] == '0' and rr: bad = True
                if bad: continue
                try: o = write_output(R, rr, fmt if k == qi else 'none', sym)
                except Reject: continue
                exs.append((sa, sb, o))
                if wz: lead_done = True
                break
            else:
                raise Reject()
        trip[sym] = exs
        for sa, sb, o in exs: lines.append(f"{sa}{sym}{sb} = {o}")
    rng.shuffle(lines)

    # gates: every filler variant must reproduce its examples (the qop's V too, by construction)
    for fam, V, sym in zip(fams, variants, syms):
        if not all(fits(sa, sb, V, ro, rr, o) for (sa, sb, o) in trip[sym]):
            raise Reject()

    # query: OPPOSITE regime, pick disagrees with the dissenter
    for _t in range(400):
        Pq = operand(ro, rng); Qq = operand(ro, rng)
        if Pq == Qq or in_regime(Pq, Qq, C['regime']): continue
        raw_pick = frame_result(Pq, Qq, C['pick'])
        raw_dis = frame_result(Pq, Qq, C['dissent'])
        if raw_pick is None or raw_pick == raw_dis: continue
        if raw_pick.startswith('-') and raw_pick.lstrip('-')[-1] == '0' and rr: continue
        qsa, qsb = write_operand(Pq, ro), write_operand(Qq, ro)
        if not (good(qsa) and good(qsb)): continue
        break
    else:
        raise Reject()
    gold = write_output(raw_pick, rr, fmt, qsym)

    text = INTRO + "\n" + "\n".join(lines) + f"\nNow, determine the result for: {qsa}{qsym}{qsb}\n"
    return text, gold, reading, fmt, qsym

def main():
    rng = random.Random(20260612)
    print("case | reading | sign | gold | classify | pick==blind | GT-match")
    for case in (1, 2, 3, 4):
        while True:
            try: text, gold, reading, fmt, qsym = build_ambiguous(case, rng)
            except Reject: continue
            d = os.path.join(HERE, '_examples', f'ambiguous_case{case}')
            os.makedirs(os.path.join(d, 'track'), exist_ok=True)
            open(os.path.join(d, 'question.txt'), 'w').write(text)
            open(os.path.join(d, 'answer.txt'), 'w').write(gold)
            open(os.path.join(d, 'label.txt'), 'w').write('NE_aug_ambiguous_arithmetic')
            rec = S.load_folder(d)
            lab = S.classify(rec)
            if lab != 'ambiguous_arithmetic': continue           # construction missed -> retry (counted nowhere, prototype only)
            cot, _ = S.gen_cot(rec)                              # BLIND solve (gen_cot never reads answer.txt)
            open(os.path.join(d, 'track', 'tree_cot.txt'), 'w').write(cot)
            box = S.extract_final_answer(cot)
            ok = S.verify(gold, box)
            print(f"  {case}  | {reading:9s} | {fmt:6s} | {gold:>6s} | {lab} | box={box!r} | {'MATCH' if ok else 'MISS'}")
            break

if __name__ == '__main__':
    main()
