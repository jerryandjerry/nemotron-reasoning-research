#!/usr/bin/env python3
"""Cryptarithm augmentation generator — COMPLEMENT the original 800's thin TRAINABLE tails.

We CONSTRUCT puzzles (so we hold the cipher+gold); the solver runs BLIND (gen_cot sees question.txt only,
never answer.txt); GT-match = does the blind solver recover the planted answer. Ambiguous/unsolvable ones
fall out as GT-match=False.

The "category" the model must REASON to is the QUERY-OPERATOR NATURE (the reasoning path):
  deduce_arith   query op is arithmetic AND seen in examples         -> full digit deduce
  query_concat   query op is concat (seen)                            -> short-circuit to symbol concat
  unseen_concat  query op unseen, no 4-digit RHS                      -> guess concat
  unseen_arith   query op unseen, a 4-digit RHS present, 1 free fam   -> guess arithmetic  (ABSENT in corpus)
  pure_concat    every op is concat                                   -> trivial (kept small; saturated already)

§4b DECORRELATION: every INCIDENTAL feature (reading, sign, sign-position, #operators, #examples,
operator symbol-class arith-vs-punct) is injected by ONE global knob at the SAME rate in every category,
so no surface cue predicts the reasoning path. No forced corner subtypes. The category label is derived
post-hoc for bookkeeping/decorrelation measurement only — it never drives feature injection.

Render uses the solver's OWN helpers: cg.apply_op + _gen_solutions.num_to_sym (RHS, inverse of rhs_int)
and render_operand (inverse of sym_val), so every written line reads back exactly to the intended value.
"""
import os, sys, csv, random, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
CRYPT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260525_Cryptarithm'
sys.path.insert(0, CRYPT)
import cot_generator as cg
import _gen_solutions as gs

POOL23 = sorted('''!"#$%&'()/:<>?@[\\]^`{|}''')          # the 23 operand-capable symbols (no +,-,*)
ARITH_SYM = {'noisy_add': '+', 'noisy_mul': '*', 'noisy_subtraction': '-'}   # canonical arith glyphs (operator-only)
ADD_V = ['add', 'add_p1', 'add_m1', 'add_p2', 'add_m2']
MUL_V = ['mul', 'mul_p1', 'mul_m1', 'mul_p2', 'mul_m2']
SUB_V_SIGNED = ['sub_signed', 'rsub_signed', 'neg_absdiff',
                'sub_signed_p1', 'sub_signed_m1', 'sub_signed_p2', 'sub_signed_m2']
ARITH_FAMS = ['noisy_add', 'noisy_mul', 'noisy_subtraction']

# ── global decorrelation knobs (one per feature; identical rate in EVERY category) ──────────────
LEFTWARD_PROB = 0.45      # reading
SIGNED_PROB   = 0.42      # a sub op forced to carry a negative example (mirrors corpus 42%)
SUFFIX_PROB   = 0.20      # of signed+leftward puzzles, sign at back (forces leftward -> keep modest)
ARITH_CLASS_PROB = 0.55   # an operator's symbol is an arith glyph (+,-,*) vs a punctuation symbol
N_OPS_W   = {2: 0.40, 3: 0.60}        # corpus: 2 ops 38%, 3 ops 62% (drop the rare 1-op)
N_EX_W    = {3: 0.30, 4: 0.36, 5: 0.34}

class Reject(Exception): pass

# ── renderers: inverse of the solver's readers ──────────────────────────────────────────────────
def render_operand(V, mode, dig2sym):
    """2-digit value V -> two symbols. Inverse of cg.sym_val (standard: d1=tens; leftward: d1=units)."""
    t, u = divmod(V, 10)
    return dig2sym[t] + dig2sym[u] if mode == 'standard' else dig2sym[u] + dig2sym[t]

def render_rhs_arith(variant, a, b, mode, fmt, dig2sym, op_sym):
    """Result symbols for an arithmetic op. Inverse of cg.rhs_int via the solver's own num_to_sym.
    Returns (rhs_string, result_int) or raises Reject if a digit is uncoverable (never, with a full cipher)."""
    r = cg.apply_op(variant, a, b)
    body, missing = gs.num_to_sym(r, dig2sym, op_sym, mode, fmt)
    if missing: raise Reject()
    return body, r

def render_rhs_concat(o1syms, o2syms, direction):
    return (o1syms + o2syms) if direction == 'fwd' else (o2syms + o1syms)

# ── operand sampler (2-digit, leading != 0 guaranteed by value range) ───────────────────────────
def operand(rng):
    return rng.randint(10, 99)

def mul_operand(rng):
    """operands whose product is comfortably >= 1000 (forces a 4-digit RHS)."""
    return rng.randint(32, 99)

# ── per-family example builder ──────────────────────────────────────────────────────────────────
def fam_variant(fam, signed, rng, force_4digit=False):
    if fam == 'noisy_add': return rng.choice(ADD_V)
    if fam == 'noisy_mul': return rng.choice(MUL_V)
    # subtraction
    return rng.choice(SUB_V_SIGNED) if signed else 'absdiff'

def build_examples_for_op(fam, variant, sym, c, mode, fmt, dig2sym, rng,
                          need_neg=False, need_4digit=False):
    """Build c example triples (lhs_o1, lhs_o2, rhs) for one operator. Enforces per-op needs:
    need_neg (>=1 negative result, for the puzzle's sign), need_4digit (>=1 >=4-digit RHS, for unseen_arith)."""
    exs = []
    got_neg = got_4d = False
    for j in range(c):
        for _t in range(600):
            if fam == 'noisy_mul' and (need_4digit and not got_4d):
                a, b = mul_operand(rng), mul_operand(rng)
            else:
                a, b = operand(rng), operand(rng)
            try:
                rhs, r = render_rhs_arith(variant, a, b, mode, fmt, dig2sym, sym)
            except Reject:
                continue
            if need_neg and not got_neg and j == c - 1 and r >= 0:    # last slot must secure the negative
                continue
            o1, o2 = render_operand(a, mode, dig2sym), render_operand(b, mode, dig2sym)
            exs.append((o1, o2, rhs, a, b, r))
            if r < 0: got_neg = True
            if len([ch for ch in rhs if ch in dig2sym.values()]) >= 4: got_4d = True
            break
        else:
            raise Reject()
    if need_neg and not got_neg: raise Reject()
    if need_4digit and not got_4d: raise Reject()
    return exs

def build_concat_examples(direction, sym, c, mode, dig2sym, rng):
    exs = []
    for j in range(c):
        a, b = operand(rng), operand(rng)
        o1, o2 = render_operand(a, mode, dig2sym), render_operand(b, mode, dig2sym)
        rhs = render_rhs_concat(o1, o2, direction)
        exs.append((o1, o2, rhs, a, b, None))
    return exs

# ── symbol assignment for operators (decorrelated symbol-class) ─────────────────────────────────
def pick_op_symbols(fams_or_concat, operand_syms, rng):
    """One distinct symbol per operator, drawn so symbol-CLASS (arith glyph vs punct) is set by a global
    knob independent of the family. Concat ops never get an arith glyph (corpus: concat is always punct-ish;
    keeps + - * meaning 'arith-looking'). Returns list of symbols."""
    used = set(operand_syms)
    punct_pool = [s for s in POOL23 if s not in used]; rng.shuffle(punct_pool)
    arith_avail = {'+', '-', '*'}
    out = []
    for f in fams_or_concat:
        if f != 'concat' and rng.random() < ARITH_CLASS_PROB and ARITH_SYM[f] in arith_avail:
            s = ARITH_SYM[f]; arith_avail.discard(s)
        else:
            s = punct_pool.pop()
        used.add(s); out.append(s)
    return out

# ── main constructor ────────────────────────────────────────────────────────────────────────────
def build_puzzle(category, rng):
    # ---- global incidental features (same knobs for every category) ----
    # reading defines the standard/little_endian labels; otherwise it is a global knob.
    if category in ('mixed_concat', 'arithmetic'):                    mode = 'standard'
    elif category in ('mixed_concat_little_endian', 'little_endian'): mode = 'little_endian'
    else:                                                             mode = 'little_endian' if rng.random() < LEFTWARD_PROB else 'standard'
    want_signed = rng.random() < SIGNED_PROB
    if want_signed and mode == 'little_endian' and rng.random() < SUFFIX_PROB:
        fmt = 'suffix'
    elif want_signed:
        fmt = 'prefix'
    else:
        fmt = 'none'
    n_ex = rng.choices(list(N_EX_W), weights=list(N_EX_W.values()))[0]

    # ---- cipher: full bijection digit 0-9 -> 10 distinct operand symbols ----
    operand_syms = rng.sample(POOL23, 10)
    digits = list(range(10)); rng.shuffle(digits)
    dig2sym = {d: operand_syms[i] for i, d in enumerate(digits)}

    # ---- choose example operator FAMILIES + the query operator, per reasoning path ----
    # Corpus-faithful complement categories ONLY (the thin TRAINABLE tails):
    #   mixed_concat / mixed_concat_little_endian : >=1 CONCAT example op + arithmetic ops; QUERY op is a
    #       SEEN arithmetic op (concat is never the query — that is the corpus shape, which still needs the
    #       full digit search). Reading is the only thing separating the two labels.
    #   query_unseen_concat : query op unseen, no >=4-digit RHS -> solver guesses concat.
    n_ops = rng.choices(list(N_OPS_W), weights=list(N_OPS_W.values()))[0]
    need_4digit_op = None
    if category in ('mixed_concat', 'mixed_concat_little_endian'):
        n_arith = max(1, n_ops - 1)                              # >=1 arith (the query op) + 1 concat
        arith = rng.sample(ARITH_FAMS, min(n_arith, 3))
        if want_signed and 'noisy_subtraction' not in arith:
            arith[rng.randrange(len(arith))] = 'noisy_subtraction'
        ex_fams = arith + ['concat']
        q_kind = ('seen_arith', rng.randrange(len(arith)))       # query op is one of the ARITH ops
    elif category in ('arithmetic', 'little_endian'):            # pure arithmetic (NO concat op); seen arith query
        ex_fams = rng.sample(ARITH_FAMS, min(n_ops, 3))
        if want_signed and 'noisy_subtraction' not in ex_fams:
            ex_fams[rng.randrange(len(ex_fams))] = 'noisy_subtraction'
        q_kind = ('seen_arith', rng.randrange(len(ex_fams)))
    elif category == 'pure_concat':                             # every op is concat; query op is a seen concat op
        ex_fams = ['concat'] * n_ops
        q_kind = ('concat_query', None)
    else:  # query_unseen_concat — query op unseen; NO >=4-digit RHS (so the solver guesses concat)
        ex_fams = rng.sample(ARITH_FAMS + ['concat'], min(n_ops, 3))
        if want_signed and 'noisy_subtraction' not in ex_fams and 'noisy_add' in ex_fams:
            ex_fams[ex_fams.index('noisy_add')] = 'noisy_subtraction'
        ex_fams = [f if f != 'noisy_mul' else 'noisy_add' for f in ex_fams]   # no mul -> no 4-digit RHS
        ex_fams = list(dict.fromkeys(ex_fams)) or ['noisy_add']
        q_kind = ('unseen_concat', None)

    # ---- distinct families per op already (sample without replacement); assign symbols ----
    op_syms = pick_op_symbols(ex_fams, operand_syms, rng)

    # ---- distribute n_ex example lines across the ops (cap 3 each, >=1 each), RANDOMLY — no yield-gaming.
    #      The natural distribution sometimes leaves the query op with a single example (ambiguous variant)
    #      or the cipher underdetermined; those legitimately fail GT-match and are dropped. That is real. ----
    counts = [1] * len(ex_fams)
    cap_total = 3 * len(ex_fams)
    target = min(n_ex, cap_total)
    while sum(counts) < target:
        cand = [i for i in range(len(ex_fams)) if counts[i] < 3]
        if not cand: break
        counts[rng.choice(cand)] += 1

    # which example op carries the negative (sign)
    neg_op_idx = None
    if want_signed:
        sub_idxs = [i for i, f in enumerate(ex_fams) if f == 'noisy_subtraction']
        if sub_idxs: neg_op_idx = sub_idxs[0]
        else: want_signed = False; fmt = 'none'

    # ---- build example lines ----
    lines = []
    op_meta = []   # (sym, fam, variant)
    seen_digits = set()
    for i, (fam, sym, c) in enumerate(zip(ex_fams, op_syms, counts)):
        if fam == 'concat':
            direction = rng.choice(['fwd', 'rev'])
            exs = build_concat_examples(direction, sym, c, mode, dig2sym, rng)
            op_meta.append((sym, fam, direction))
        else:
            variant = fam_variant(fam, want_signed and i == neg_op_idx, rng)
            exs = build_examples_for_op(fam, variant, sym, c, mode, fmt, dig2sym, rng,
                                        need_neg=(i == neg_op_idx),
                                        need_4digit=(need_4digit_op == fam))
            op_meta.append((sym, fam, variant))
        for (o1, o2, rhs, a, b, r) in exs:
            lines.append(f"{o1}{sym}{o2} = {rhs}")
            for ch in o1 + o2 + rhs:
                if ch in operand_syms: seen_digits.add({v: k for k, v in dig2sym.items()}[ch])

    # ---- query operator + gold ----
    kind, info = q_kind
    if kind in ('concat_query',):
        # query op = the concat example op
        cidx = [i for i, f in enumerate(ex_fams) if f == 'concat'][0]
        qsym = op_syms[cidx]; qdir = op_meta[cidx][2]
        a, b = operand(rng), operand(rng)
        o1, o2 = render_operand(a, mode, dig2sym), render_operand(b, mode, dig2sym)
        gold = render_rhs_concat(o1, o2, qdir)                 # pure-symbol answer; digits don't matter
        qline = f"{o1}{qsym}{o2}"
    elif kind == 'unseen_concat':
        used = set(operand_syms) | set(op_syms)
        qpool = [s for s in POOL23 if s not in used]; rng.shuffle(qpool)
        qsym = qpool.pop()
        a, b = operand(rng), operand(rng)
        o1, o2 = render_operand(a, mode, dig2sym), render_operand(b, mode, dig2sym)
        gold = o1 + o2                                          # solver guesses concat_fwd -> d1d2d3d4
        qline = f"{o1}{qsym}{o2}"
    elif kind == 'unseen_arith':
        free_fam = info
        used = set(operand_syms) | set(op_syms)
        qpool = [s for s in POOL23 if s not in used]; rng.shuffle(qpool)
        # query symbol class also decorrelated; never collide with a present arith glyph
        if rng.random() < ARITH_CLASS_PROB and ARITH_SYM[free_fam] not in op_syms:
            qsym = ARITH_SYM[free_fam]
        else:
            qsym = qpool.pop()
        base = gs.BASE[free_fam]                                # solver guesses the BASE variant (absdiff/add/mul)
        for _t in range(600):
            a, b = operand(rng), operand(rng)
            try:
                gold, r = render_rhs_arith(base, a, b, mode, fmt, dig2sym, qsym)
            except Reject:
                continue
            # every digit of the answer + operands must be covered by examples (else solver can't map it)
            need = set(int({v: k for k, v in dig2sym.items()}[ch]) for ch in
                       render_operand(a, mode, dig2sym) + render_operand(b, mode, dig2sym)
                       if ch in operand_syms)
            need |= set(k for k, v in dig2sym.items() if v in [ch for ch in gold if ch in operand_syms])
            if not need <= seen_digits:
                continue
            o1, o2 = render_operand(a, mode, dig2sym), render_operand(b, mode, dig2sym)
            qline = f"{o1}{qsym}{o2}"
            break
        else:
            raise Reject()
    else:  # seen_arith
        qidx = info; qsym = op_syms[qidx]; fam = ex_fams[qidx]; variant = op_meta[qidx][2]
        sym2dig = {v: k for k, v in dig2sym.items()}
        for _t in range(600):
            a, b = operand(rng), operand(rng)
            try:
                gold, r = render_rhs_arith(variant, a, b, mode, fmt, dig2sym, qsym)
            except Reject:
                continue
            need = set(sym2dig[ch] for ch in render_operand(a, mode, dig2sym) + render_operand(b, mode, dig2sym) if ch in operand_syms)
            need |= set(sym2dig[ch] for ch in gold if ch in operand_syms)
            if not need <= seen_digits:
                continue
            o1, o2 = render_operand(a, mode, dig2sym), render_operand(b, mode, dig2sym)
            qline = f"{o1}{qsym}{o2}"
            break
        else:
            raise Reject()

    rng.shuffle(lines)
    text = ("In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
            "Below are a few examples:\n" + "\n".join(lines) +
            f"\nNow, determine the result for: {qline}\n")

    # ---- post-hoc feature record (for decorrelation measurement) ----
    actual_signed = any(ch in op_syms for ln in lines for ch in ln.split(' = ', 1)[1]) if False else (fmt != 'none')
    meta = {'category': category, 'reading': 'leftward' if mode == 'little_endian' else 'rightward',
            'signed': fmt if fmt != 'none' else 'no', 'n_ops': len(ex_fams), 'n_ex': len(lines),
            'arith_syms': sum(1 for s in op_syms if s in '+-*'),
            'has_concat': int('concat' in ex_fams), 'mode': mode}
    return text, gold, meta

CATEGORIES = ['arithmetic', 'little_endian', 'pure_concat',
              'mixed_concat', 'mixed_concat_little_endian', 'query_unseen_concat']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', type=int, default=0)
    ap.add_argument('--seed', type=int, default=20260607)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    if a.test:
        for cat in CATEGORIES:
            print(f"\n===== {cat} =====")
            for _ in range(a.test):
                r = None
                while r is None:
                    try: r = build_puzzle(cat, rng)
                    except Reject: r = None
                text, gold, meta = r
                print(f"[{meta['reading']}/{meta['signed']}/ops{meta['n_ops']}] gold={gold!r}")
                print(text, end='')

if __name__ == '__main__':
    main()
