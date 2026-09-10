#!/usr/bin/env python3
"""Augmentation generator v2 — COMPLEMENT the original 732 (fill the thin tails / OOD), not mirror it.

We CONSTRUCT puzzles (so we hold the gold); the solver runs BLIND (sub-agents, gold withheld); GT-match =
does the blind solver recover the planted answer. Ambiguous/unsolvable puzzles fall out as GT-match=False.

Complement design (vs original):
  type:    exotic-std 30% + corners 20% (exotic+sign / exotic+concat / multi-exotic, 0% in original)
           + unseen 25% + concat 15% + deducible 10% (deducible is saturated in the original -> minimized)
  symbols: +/-/* down-weighted to ~5% (original 47%); rare punctuation lifted to ~4% each
  reading: ~60% rightward (original is 64% leftward)
  signed:  ~55% per-puzzle (original 38%), concentrated in rightward + exotic
  flow:    leading-0-skip minimized ~5% (saturated in original); trailing-sign-skip boosted ~18%
Structural (kept in-format): 2-digit operands, 3-5 example lines, 2-3 operators.
Folders NE_aug_NNNNNN ; labels NE_aug_<type> ; gold in answer.txt ; master _manifest.csv.
"""
import os, sys, csv, random, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
GEN = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260606_Numeric_Equation'
sys.path.insert(0, GEN)
from _gen_eq import raw_on, _rev, render_answer, fits, VOCAB   # solver helpers (arithmetic/string only)

def any_family_fits(exs, ro, rr):
    """True if ANY modeled variant reproduces every example under this reading (early-exit). Used to gate
    exotic ops (they must fit NONE)."""
    return any(all(fits(sa, sb, nm, ro, rr, o) for sa, sb, o in exs) for nm in VOCAB)

INTRO = "In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:"
ARITH_SYMS = {'add': '+', 'sub': '-', 'mul': '*'}
PUNCT = list('/"`%!<[$^\'{?\\#(}):>@&|]')                       # the EXACT 23 non-arith symbols of the original
ADD_V = ['addition', 'add+1', 'add-1', 'add+2', 'add-2']
MUL_V = ['multiplication', 'multiply+1', 'multiply-1', 'multiply+2', 'multiply-2']
SUB_SIGNED = ['subtraction (a-b)', 'reverse subtraction (b-a)', 'sub+1', 'sub-1', 'sub+2', 'sub-2',
              'rsub+1', 'rsub-1', 'rsub+2', 'rsub-2']
CONCAT_V = ['concatenation', 'reverse concatenation']
BASE_OF = {'add': 'addition', 'mul': 'multiplication', 'sub': 'absolute difference', 'concat': 'concatenation'}
# Secondary features are injected UNIFORMLY across types AND at the ORIGINAL corpus's LEVELS (not complemented
# below them). The originals already have arith/leadz/signed ~uniform across categories, so matching their level
# keeps the COMBINED set uniform; sitting below them (the old complement design) manufactured a feature<->type
# shortcut because aug fills each category to a different degree. Calibrated so aug ~ orig: arith~60%, leadz~37%, signed~40%.
ARITH_PROB = 0.62            # PUZZLE-LEVEL chance of >=1 canonical +/-/* symbol (forces ONE arith slot) -> arith ~60% UNIFORM across types (matches orig); not per-family, so arith% doesn't scale with #arith-eligible ops
SIGNED_PROB = 0.62           # want-signed rate -> visible-signed ~40% (a signed sub only shows a sign when its result is negative)
RIGHTWARD_PROB = 0.55        # reading UNIFORM across types; combined reading spread stays ~10% (deducible's mild leftward skew is real)
LEADZERO_PROB = 0.80         # of leftward puzzles -> leadz ~37% overall (matches orig); injected uniformly across types

class Reject(Exception): pass
def rev(s): return s[::-1]

def frame_result(P, Q, V):
    if V == 'maxmodmin': return str(max(P, Q) % min(P, Q)) if min(P, Q) else str(max(P, Q))
    return raw_on(str(P), str(Q), V, False)

def write_operand(P, ro): return rev(str(P)) if ro else str(P)

def write_output(R, rr, fmt, sym):
    if R is None: raise Reject()
    if R.startswith('-'):
        return render_answer(R if not rr else _rev(R), fmt, sym)
    return _rev(R) if rr else R

def good(s): return s and not (len(s) > 1 and s[0] == '0')

def assign_syms(fams, rng, want_arith):
    """Distinct symbol per family slot. If want_arith (a PUZZLE-LEVEL decision, uniform across types), force
    exactly ONE arith-eligible slot to its canonical +/-/*; all other slots take punctuation. This keeps the
    puzzle-level 'has an arith symbol' rate ~constant across types instead of scaling with #arith-eligible ops."""
    arith_idx = [i for i, f in enumerate(fams) if f in ARITH_SYMS]
    force = rng.choice(arith_idx) if (want_arith and arith_idx) else None
    syms = [None] * len(fams); pool = PUNCT[:]; rng.shuffle(pool); used = set()
    for i, fam in enumerate(fams):
        if i == force and ARITH_SYMS[fam] not in used:
            s = ARITH_SYMS[fam]
        else:
            s = pool.pop()
            while s in used: s = pool.pop()
        used.add(s); syms[i] = s
    return syms

def variant_for(fam, signed, rng):
    if fam == 'maxmodmin': return 'maxmodmin'
    if fam == 'concat': return rng.choice(CONCAT_V)
    if fam == 'sub': return rng.choice(SUB_SIGNED) if signed else 'absolute difference'
    if fam == 'add': return rng.choice(ADD_V)
    return rng.choice(MUL_V)

def operand(ro, rng, want_leadzero=False):
    """A 2-digit operand value. Never ends in 0 (a trailing-0 reverses to a leading-0 in the other reading,
    which the solver rules illegal -> an UNintended skip) unless we explicitly inject one."""
    while True:
        P = rng.randint(10, 99)
        if want_leadzero:
            if P % 10 == 0: return P
            continue
        if P % 10 == 0: continue
        return P

def build_puzzle(ptype, rng):
    # ---- choose families, query op, reading, sign, by type ----
    signed = False; fmt = 'none'
    leadzero = False
    if ptype == 'exotic':
        fams = ['maxmodmin'] + rng.sample(['add', 'mul', 'sub', 'concat'], rng.choice([1, 2])); qi = 0
    elif ptype == 'exotic_sign':                               # corner: exotic query + a SIGNED-sub example op
        fams = ['maxmodmin', 'sub'] + ([rng.choice(['add', 'mul'])] if rng.random() < 0.4 else []); qi = 0
        signed = True
    elif ptype == 'exotic_concat':                             # corner: exotic query + a concat op
        fams = ['maxmodmin', 'concat'] + ([rng.choice(['add', 'mul', 'sub'])] if rng.random() < 0.4 else []); qi = 0
    elif ptype == 'multi_exotic':                              # corner: 2 exotic ops (+ optional arith anchor)
        fams = ['maxmodmin', 'maxmodmin'] + ([rng.choice(['add', 'mul', 'sub'])] if rng.random() < 0.6 else []); qi = 0
    elif ptype == 'concat':
        fams = ['concat'] + rng.sample(['add', 'mul', 'sub'], rng.choice([1, 2])); qi = 0
    elif ptype == 'unseen':
        fams = rng.sample(['add', 'mul', 'sub'], rng.choice([1, 2]))   # 1-2 arith example ops (>=1 so the solver can guess the unseen op) + 1 query = 2-3 distinct
        qi = None                                              # query op is a fresh remaining arith family
    else:                                                      # deducible (OOD-flavored: rare symbols + rightward)
        fams = rng.sample(['add', 'mul', 'sub', 'concat'], rng.choice([2, 3])); qi = rng.randrange(len(fams))

    # signed: decide per-puzzle (~SIGNED_PROB); if signed, FORCE an example op to be a sub so it carries the sign
    want_signed = (ptype == 'exotic_sign') or (rng.random() < SIGNED_PROB)
    if want_signed and 'sub' not in fams:
        conv = [k for k in range(len(fams)) if fams[k] in ('add', 'mul')]   # convert an add/mul op -> sub (never the maxmodmin/concat special)
        if conv: fams[conv[0]] = 'sub'
    signed = want_signed and 'sub' in fams
    if signed:
        fmt = rng.choices(['prefix', 'suffix'], weights=[0.85, 0.15])[0]   # the negative sign is ALWAYS the operator's own symbol (real data: 404/404 sign==operator, 85% prefix / 15% suffix). NO literal '-' unless the operator symbol itself is '-' (prefix with sym='-' -> '-91'). 'minus' removed: it injected a foreign '-' sign for glyph operators.

    # reading: suffix-sign forces leftward (trailing-sign skip); else RIGHTWARD_PROB
    if signed and fmt == 'suffix':
        reading = 'leftward'
    else:
        reading = 'rightward' if rng.random() < RIGHTWARD_PROB else 'leftward'
    ro = rr = (reading == 'leftward')
    if reading == 'leftward' and rng.random() < LEADZERO_PROB:
        leadzero = True

    variants = [variant_for(f, signed and f == 'sub', rng) for f in fams]
    want_arith = rng.random() < ARITH_PROB                  # puzzle-level, UNIFORM across types

    # query op + holistic symbol assignment (>=1 arith slot total when want_arith; unseen's query is an extra slot)
    if ptype == 'unseen':
        qfam = rng.choice([f for f in ['add', 'mul', 'sub'] if f not in fams])
        qvar = BASE_OF[qfam]
        all_syms = assign_syms(fams + [qfam], rng, want_arith)
        syms, qsym = all_syms[:len(fams)], all_syms[-1]
    else:
        syms = assign_syms(fams, rng, want_arith)
        qfam, qvar, qsym = fams[qi], variants[qi], syms[qi]

    # ---- example lines ----
    nex = rng.choice([3, 4, 5])
    counts = [1] * len(fams)
    while sum(counts) < nex:
        cand = [i for i in range(len(fams)) if counts[i] < 3]     # cap 3 examples/op; stop if none can take more
        if not cand: break                                         # (e.g. a single-operator puzzle caps total at 3)
        counts[rng.choice(cand)] += 1
    lines = []; trip = {}
    lead_done = not leadzero
    for k, (fam, V, sym, c) in enumerate(zip(fams, variants, syms, counts)):
        exs = []
        need_neg = (signed and fam == 'sub')             # the signed-sub operator MUST show a negative (demonstrate the sign convention)
        for j in range(c):
            for _t in range(400):
                wz = (leadzero and not lead_done)            # inject one leading-0 operand
                P = operand(ro, rng, want_leadzero=wz); Q = operand(ro, rng)
                if rng.random() < 0.5 and wz: P, Q = Q, P
                R = frame_result(P, Q, V)
                if R is None: continue
                if need_neg and j == 0 and not R.startswith('-'): continue   # force the signed sub's 1st example negative
                sa, sb = write_operand(P, ro), write_operand(Q, ro)
                # validity: avoid UNintended leading-0 OPERAND (allow the injected one). Leading-0 RESULTS are
                # allowed now (real data ~7.5%: a positive result ending in 0 reverses to e.g. "0624" leftward).
                bad = (not wz and (not good(sa) or not good(sb)))
                if R.startswith('-') and R.lstrip('-')[-1] == '0' and rr: bad = True   # keep: negative + leading-0-magnitude not seen in real data
                if bad: continue
                try: o = write_output(R, rr, fmt, sym)
                except Reject: continue
                exs.append((sa, sb, o));
                if wz: lead_done = True
                break
            else:
                raise Reject()
        trip[sym] = exs
        for sa, sb, o in exs: lines.append(f"{sa}{sym}{sb} = {o}")
    rng.shuffle(lines)

    # ---- validity gates (generation-time; exact, correctness-preserving — just cheaper than full consistent_ops) ----
    for fam, V, sym in zip(fams, variants, syms):
        if V == 'maxmodmin':
            if any_family_fits(trip[sym], ro, rr): raise Reject()                 # exotic op must fit NO family
        elif not all(fits(sa, sb, V, ro, rr, o) for (sa, sb, o) in trip[sym]):
            raise Reject()                                                        # intended variant MUST reproduce its examples

    # the sign convention (how a negative is written) is ONLY deducible if an EXAMPLE shows a sign. The real
    # data never has a negative gold without a negative example (orig: 48/48). So a negative query is allowed
    # ONLY when some example already carries a sign glyph; otherwise force a non-negative query.
    ex_signed = any((not o[0].isdigit()) or (not o[-1].isdigit()) for exs in trip.values() for (sa, sb, o) in exs)

    # ---- query + gold ----
    for _t in range(400):
        Pq = operand(ro, rng); Qq = operand(ro, rng)
        Rq = frame_result(Pq, Qq, qvar)
        if Rq is None: continue
        if Rq.startswith('-') and Rq.lstrip('-')[-1] == '0' and rr: continue   # negative + leading-0-magnitude: skip (not in real data); positive leading-0 results are allowed
        if Rq.startswith('-') and not ex_signed: continue                     # negative gold requires a sign-bearing example
        break
    else:
        raise Reject()                                                        # no valid query found -> drop the puzzle
    qsa, qsb = write_operand(Pq, ro), write_operand(Qq, ro)
    if not (good(qsa) and good(qsb)): raise Reject()
    gold = write_output(Rq, rr, fmt, qsym)

    text = INTRO + "\n" + "\n".join(lines) + f"\nNow, determine the result for: {qsa}{qsym}{qsb}\n"
    LBL = {'exotic': 'exotic_operation', 'exotic_sign': 'exotic_signed_sub', 'exotic_concat': 'exotic_concat',
           'multi_exotic': 'multi_exotic', 'concat': 'concatenation', 'unseen': 'unseen_operator',
           'deducible': 'arithmetic_left_to_right' if reading == 'rightward' else 'arithmetic_right_to_left'}
    return text, gold, 'NE_aug_' + LBL[ptype], reading, ptype, signed, fmt, leadzero

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--n', type=int, default=500); ap.add_argument('--seed', type=int, default=20260607)
    ap.add_argument('--test', type=int, default=0)
    ap.add_argument('--start-idx', type=int, default=0)        # first id = start_idx+1 (so a 2nd batch doesn't clobber the 1st)
    ap.add_argument('--append', action='store_true')           # extend _manifest.csv instead of overwriting
    a = ap.parse_args()
    rng = random.Random(a.seed)
    # 4 categories ONLY (no forced corners — those hard-wire a feature to a category). Every secondary feature
    # (reading, sign, symbol, #ops) is injected at the SAME rate in every category, so nothing predicts it.
    MIX = [('deducible', 0.47), ('exotic', 0.16), ('unseen', 0.18), ('concat', 0.19)]  # unseen up-weighted for its ~76% GT-yield; over-generate then subsample to exact aug targets
    targets = {p: round(a.n * f) for p, f in MIX}; targets['exotic'] += a.n - sum(targets.values())
    if a.test:
        for p, _ in MIX:
            print(f"\n===== {p} =====")
            for _ in range(a.test):
                r = None
                while r is None:
                    try: r = build_puzzle(p, rng)
                    except Reject: r = None
                print(f"[{r[3]}/{'signed:'+r[6] if r[5] else 'pos'}{'/lead0' if r[7] else ''}] gold={r[1]!r}\n{r[0]}", end='')
        return
    manifest = []; idx = a.start_idx
    for ptype, _ in MIX:
        made = tries = 0
        while made < targets[ptype]:
            tries += 1
            if tries > targets[ptype] * 800: print(f"WARN {ptype}: {made}/{targets[ptype]}"); break
            try: text, gold, label, reading, pt, signed, fmt, lz = build_puzzle(ptype, rng)
            except Reject: continue
            idx += 1; pid = f"NE_aug_{idx:06d}"
            d = os.path.join(HERE, pid); os.makedirs(os.path.join(d, 'track'), exist_ok=True)
            open(d + '/question.txt', 'w').write(text); open(d + '/answer.txt', 'w').write(gold); open(d + '/label.txt', 'w').write(label)
            manifest.append({'id': pid, 'reading': reading, 'type': pt, 'label': label, 'signed': fmt if signed else 'no', 'leadzero': lz, 'gold': gold})
            made += 1
        print(f"{ptype}: {made}")
    existing = []
    if a.append and os.path.exists(HERE + '/_manifest.csv'):
        existing = list(csv.DictReader(open(HERE + '/_manifest.csv', newline='')))
    with open(HERE + '/_manifest.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['id', 'reading', 'type', 'label', 'signed', 'leadzero', 'gold']); w.writeheader()
        w.writerows(existing + manifest)
    print(f"TOTAL {len(existing) + len(manifest)} (new {len(manifest)}, kept {len(existing)}) -> {HERE}")

if __name__ == '__main__':
    main()
