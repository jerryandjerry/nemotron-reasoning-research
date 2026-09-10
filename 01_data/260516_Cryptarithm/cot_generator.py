#!/usr/bin/env python3
"""
cot_generator.py
Input:  <puzzle_dir>/question.txt
Output: <puzzle_dir>/prefix.txt   — deterministic steps 1-9
        <puzzle_dir>/tree.json    — full DFS tree for step 10

Usage: python3 cot_generator.py <puzzle_dir>
"""

import json
import re
import sys
from pathlib import Path
from itertools import product as iproduct, permutations

# ── Operations ────────────────────────────────────────────────────────────────

# Each variant maps two operands to a value.
VARIANT_FN = {
    'mul':         lambda a, b: a * b,
    'mul_p1':      lambda a, b: a * b + 1,
    'mul_m1':      lambda a, b: a * b - 1,
    'add':         lambda a, b: a + b,
    'add_p1':      lambda a, b: a + b + 1,
    'add_m1':      lambda a, b: a + b - 1,
    'add_p2':      lambda a, b: a + b + 2,
    'absdiff':     lambda a, b: abs(a - b),
    'absdiff_p1':  lambda a, b: abs(a - b) + 1,
    'absdiff_m1':  lambda a, b: abs(a - b) - 1,
    'absdiff_m2':  lambda a, b: abs(a - b) - 2,
    'sub_signed':  lambda a, b: a - b,
    'rsub_signed': lambda a, b: b - a,
    'neg_absdiff': lambda a, b: -abs(a - b),
}

# A core is one family's worth of variants, in priority order. The variant is resolved
# at match time: among variants whose value matches the RHS, the earliest in this list
# wins (subtraction variants overlap in value, so order is the tie-break).
CORES = {
    'mul':               ['mul', 'mul_p1', 'mul_m1'],
    'add':               ['add', 'add_p1', 'add_m1', 'add_p2'],
    'noisy_subtraction': ['absdiff', 'absdiff_p1', 'absdiff_m1', 'absdiff_m2',
                          'sub_signed', 'rsub_signed', 'neg_absdiff'],
}

# core name → family name (each core carries its family's full variant set)
CORE_FAMILY = {'mul': 'noisy_mul', 'add': 'noisy_add', 'noisy_subtraction': 'noisy_subtraction'}
SHORT_FAM   = {'noisy_mul': '~mul', 'noisy_add': '~add', 'noisy_subtraction': '~sub'}   # CoT shorthand for the noisy operator

def core_cands(core_name, a, b):
    """Ordered (value, variant) candidates for a core given operands a, b."""
    return [(VARIANT_FN[v](a, b), v) for v in CORES[core_name]]

# Family → list of core names (in order of preference)
FAMILIES = [
    ('noisy_mul',           ['mul']),
    ('noisy_add',           ['add']),
    ('noisy_subtraction', ['noisy_subtraction']),
]
FAM_BY_NAME = {name: (name, cores) for name, cores in FAMILIES}

# All variant names (used for op_dict and reporting)
ALL_VARIANTS = [v for vs in CORES.values() for v in vs]

# ── Digit-count family filtering ───────────────────────────────────────────────
# Operands are 2-digit (leading digit != 0), so a, b in [10, 99]. Output ranges:
#   subtraction (|a-b| family):   magnitude 0..90    -> 1 or 2 digits
#   addition    (a+b family):     19..200            -> 2 or 3 digits
#   multiplication (a*b family):  >=99               -> 3 or 4 digits
#     (the only 2-digit product case is 10*10-1=99, which needs both operands == 10;
#      that is degenerate, so we treat a 2-digit RHS as ruling multiplication out.)
# Inverting: which families can produce an output with exactly d digits.
FAMILY_ORDER_DEFAULT = ['noisy_add', 'noisy_subtraction', 'noisy_mul']

def possible_families(d):
    """Families that can produce a d-digit |result| from two 2-digit operands."""
    if d == 1: return {'noisy_subtraction'}
    if d == 2: return {'noisy_add', 'noisy_subtraction'}
    if d == 3: return {'noisy_add', 'noisy_mul'}
    return {'noisy_mul'}  # d >= 4

def family_ranking(digit_counts, prior_family=None):
    """Hard filter then order. Candidates = intersection of possible_families over all
    of an operator's example RHS digit counts (a family that cannot produce some example's
    digit count cannot be the operator, so it is dropped entirely). Order: symbol prior
    first if it survives, then the default likelihood order (rarer producers last)."""
    if digit_counts:
        cand = set.intersection(*(possible_families(d) for d in digit_counts))
    else:
        cand = set(FAMILY_ORDER_DEFAULT)
    if not cand:                      # contradictory examples — fall back to trying all
        cand = set(FAMILY_ORDER_DEFAULT)
    ordered = []
    if prior_family in cand: ordered.append(prior_family)
    for f in FAMILY_ORDER_DEFAULT:
        if f in cand and f not in ordered: ordered.append(f)
    return ordered

def apply_op(name, a, b):
    fn = VARIANT_FN.get(name)
    return fn(a, b) if fn else None

def op_str(name, a, b):
    r = apply_op(name, a, b)
    return {
        'mul':          f"{a}×{b}={r}",
        'mul_p1':       f"{a}×{b}+1={r}",
        'mul_m1':       f"{a}×{b}-1={r}",
        'add':          f"{a}+{b}={r}",
        'add_p1':       f"{a}+{b}+1={r}",
        'add_m1':       f"{a}+{b}-1={r}",
        'add_p2':       f"{a}+{b}+2={r}",
        'absdiff':      f"|{a}-{b}|={r}",
        'absdiff_p1':   f"|{a}-{b}|+1={r}",
        'absdiff_m1':   f"|{a}-{b}|-1={r}",
        'absdiff_m2':   f"|{a}-{b}|-2={r}",
        'sub_signed':   f"{a}-{b}={r}",
        'rsub_signed':  f"{b}-{a}={r}",
        'neg_absdiff':  f"-|{a}-{b}|={r}",
    }.get(name, str(r))

# ── Puzzle data ───────────────────────────────────────────────────────────────

class Example:
    def __init__(self, lhs, rhs):
        self.lhs = lhs; self.rhs = rhs
        self.d1 = lhs[0]; self.d2 = lhs[1]
        self.op = lhs[2]
        self.d3 = lhs[3]; self.d4 = lhs[4]

class Puzzle:
    def __init__(self, examples, query, digit_syms, op_syms, letter_map, op_map, by_op):
        self.examples   = examples
        self.query      = query
        self.digit_syms = digit_syms
        self.op_syms    = op_syms
        self.letter_map = letter_map
        self.op_map     = op_map
        self.by_op      = by_op

# ── Parse ────────────────────────────────────────────────────────────────────

def parse(path):
    text = Path(path).read_text()
    examples, query = [], None
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("Now, determine"):
            query = line.split(": ", 1)[1].strip()
        elif " = " in line and not line.startswith("In"):
            lhs, rhs = line.split(" = ", 1)
            lhs, rhs = lhs.strip(), rhs.strip()
            if len(lhs) == 5:
                examples.append(Example(lhs, rhs))

    op_syms = []
    for ex in examples:
        if ex.op not in op_syms: op_syms.append(ex.op)
    if query[2] not in op_syms: op_syms.append(query[2])
    op_set = set(op_syms)

    digit_set = set()
    for ex in examples:
        digit_set.update([ex.d1, ex.d2, ex.d3, ex.d4])
        digit_set.update(s for s in ex.rhs if s not in op_set)
    digit_set.update([query[0], query[1], query[3], query[4]])

    digit_syms  = sorted(digit_set)
    letter_map  = {s: chr(ord('A') + i) for i, s in enumerate(digit_syms)}
    op_map      = {s: chr(ord('f') + i) for i, s in enumerate(op_syms)}   # operators are f, g, h (function names; not x, which reads as ×)
    by_op       = {}
    for ex in examples:
        by_op.setdefault(ex.op, []).append(ex)

    return Puzzle(examples, query, digit_syms, op_syms, letter_map, op_map, by_op)

# ── Value helpers ─────────────────────────────────────────────────────────────

def sym_val(d1, d2, asgn, mode):
    a, b = asgn.get(d1), asgn.get(d2)
    if a is None or b is None: return None
    return 10*a+b if mode == 'standard' else 10*b+a

def rhs_poly(rhs, lm, op_set, mode):
    """Expand RHS symbol string to polynomial, e.g. DBGI → 1000D+100B+10G+I, HH → 11H."""
    chars = list(rhs)
    sign_str = ""
    if chars and chars[0] in op_set:  sign_str = "-"; chars = chars[1:]
    elif chars and chars[-1] in op_set: sign_str = "-"; chars = chars[:-1]

    letters = [lm.get(s, s) for s in chars]
    n = len(letters)
    coeffs = [10**(n-1-i) for i in range(n)] if mode == 'standard' else [10**i for i in range(n)]

    # collect coefficients by letter (preserving first-appearance order)
    coeff_map = {}
    order = []
    for c, L in zip(coeffs, letters):
        if L not in coeff_map:
            coeff_map[L] = 0
            order.append(L)
        coeff_map[L] += c

    parts = [L if coeff_map[L] == 1 else f"{coeff_map[L]}{L}" for L in order]
    expr = "+".join(parts)
    return f"-({expr})" if sign_str else expr


def rhs_int(rhs, asgn, mode, op_set):
    sign, chars = 1, list(rhs)
    if chars and chars[0] in op_set:    sign = -1; chars = chars[1:]
    elif chars and chars[-1] in op_set: sign = -1; chars = chars[:-1]
    digits = [asgn.get(s) for s in chars]
    if any(d is None for d in digits): return None
    v = 0
    if mode == 'standard':
        for d in digits: v = v*10 + d
    else:
        for i, d in enumerate(digits): v += d * (10**i)
    return sign * v

def get_leading(puzzle, op_set, mode='standard'):
    """Symbols that sit in a most-significant (leading) position and must be non-zero.
    standard: tens digit is the left symbol (d1/d3), RHS MSB is rhs_d[0].
    little_endian: tens digit is the right symbol (d2/d4), RHS MSB is rhs_d[-1]."""
    r = set()
    for ex in puzzle.examples:
        if mode == 'standard': r.add(ex.d1); r.add(ex.d3)
        else:                  r.add(ex.d2); r.add(ex.d4)
        rhs_d = [s for s in ex.rhs if s not in op_set]
        if len(rhs_d) > 1: r.add(rhs_d[0] if mode == 'standard' else rhs_d[-1])
    if mode == 'standard': r.add(puzzle.query[0]); r.add(puzzle.query[3])
    else:                  r.add(puzzle.query[1]); r.add(puzzle.query[4])
    return r

# ── Concat check ──────────────────────────────────────────────────────────────

def check_concat(op_sym, examples):
    exs = [e for e in examples if e.op == op_sym]
    if not exs: return None
    ex = exs[0]
    if ex.rhs == ex.d1+ex.d2+ex.d3+ex.d4: return 'fwd'
    if ex.rhs == ex.d3+ex.d4+ex.d1+ex.d2: return 'rev'
    return None

# ── operator_dict.md ──────────────────────────────────────────────────────────

_OP_FAMILY = {
    'mul': 'noisy_mul', 'mul_p1': 'noisy_mul', 'mul_m1': 'noisy_mul',
    'add': 'noisy_add', 'add_p1': 'noisy_add', 'add_m1': 'noisy_add', 'add_p2': 'noisy_add',
    'absdiff': 'noisy_subtraction', 'absdiff_p1': 'noisy_subtraction',
    'absdiff_m1': 'noisy_subtraction', 'absdiff_m2': 'noisy_subtraction',
    'sub_signed': 'noisy_subtraction', 'rsub_signed': 'noisy_subtraction',
    'neg_absdiff': 'noisy_subtraction',
}

_OD = None   # sym -> description
_OT = None   # sym -> top family
_OCOUNT = None   # sym -> {variant: count seen in other puzzles}

def op_dict():
    global _OD, _OT, _OCOUNT
    if _OD: return _OD
    text = (Path(__file__).parent / "operator_dict.md").read_text()
    _OD, _OT, _OCOUNT, sym = {}, {}, {}, None
    for line in text.splitlines():
        # double-backtick fence (e.g. ``## `` ` ``  ...`` for the backtick symbol) first
        m = re.match(r'^## ``\s*(.+?)\s*``', line) or re.match(r'^## `(.+?)`', line)
        if m: sym = m.group(1); continue
        if sym and line.startswith('*operator'):
            _OD[sym] = line.strip().strip('*').replace('\\*', '*')
        if sym and re.match(r'^\| [a-z]', line):
            parts = line.split('|')
            op_name = parts[1].strip()
            try: _OCOUNT.setdefault(sym, {})[op_name] = int(parts[2].strip())
            except (IndexError, ValueError): pass
            if sym not in _OT and op_name in _OP_FAMILY:
                _OT[sym] = _OP_FAMILY[op_name]
    return _OD

def op_top_family(sym):
    op_dict()  # ensure _OT is populated
    return _OT.get(sym)

_V2FAM = None
def op_seen_families(sym):
    """Operation FAMILIES (noisy_mul/noisy_add/noisy_subtraction/concat) this symbol has been seen
    as in other puzzles, ordered by total frequency (summed over each family's variants).
    Discarded ops (gcd/lcm/mod/...) drop out."""
    global _V2FAM
    if _V2FAM is None:
        _V2FAM = {v: CORE_FAMILY[c] for c, vs in CORES.items() for v in vs}
        _V2FAM['concat_fwd'] = _V2FAM['concat_rev'] = 'concat'
    op_dict()
    fam_count = {}
    for v, c in (_OCOUNT.get(sym) or {}).items():
        f = _V2FAM.get(v)
        if f: fam_count[f] = fam_count.get(f, 0) + c
    return [f for f, _ in sorted(fam_count.items(), key=lambda kv: -kv[1])]

# ── Prefix: steps 1-9 ────────────────────────────────────────────────────────

def make_prefix(puzzle):
    lm = puzzle.letter_map
    om = puzzle.op_map
    op_set = set(puzzle.op_syms)
    L = []
    def E(s=""): L.append(s)

    # 1
    E("I need to infer the transformation rule from the examples.")
    E()

    # 2
    E("First, let me assign letters to each symbol:")
    for s in puzzle.digit_syms: E(f"  {s} -> {lm[s]}")
    E("Operators:")
    for s in puzzle.op_syms: E(f"  {s} -> {om[s]}")
    E()

    # 3 — letter-form equations first (continues from the symbol assignment)
    E("I will convert all the equations to letter form:")
    for i, ex in enumerate(puzzle.examples, 1):
        lhs_l = f"{lm[ex.d1]}{lm[ex.d2]} {om[ex.op]} {lm[ex.d3]}{lm[ex.d4]}"
        rhs_l = ''.join(om.get(s, lm.get(s, s)) for s in ex.rhs)
        E(f"  EX{i} \u300c{ex.lhs}\u300d = \u300c{ex.rhs}\u300d becomes {lhs_l} = {rhs_l}")
    q = puzzle.query
    E(f"  QUERY \u300c{q}\u300d becomes {lm[q[0]]}{lm[q[1]]} {om[q[2]]} {lm[q[3]]}{lm[q[4]]}")
    E()

    # 4 — operator background
    E("In Alice world, multiplication, addition, subtraction and concatenating are the most common operations, but the results are usually noisy (off by ±1 or ±2, negated, or with operands reversed). For the operators in this puzzle:")
    CANON = {'+': 'noisy_add', '-': 'noisy_subtraction', '*': 'noisy_mul'}   # +,-,* strongly indicate their own operation
    for s in puzzle.op_syms:
        fams = op_seen_families(s)
        if s in CANON:
            rest = [f for f in fams if f != CANON[s]]
            tail = f", but I have seen it represent [{', '.join(rest)}] in other puzzles, ordered by frequency" if rest else ""
            E(f"  {s}, denoted as {om[s]}, is usually {CANON[s]}{tail}")
        elif fams:
            E(f"  I have seen operator {om[s]} represent [{', '.join(fams)}] in other puzzles, ordered by frequency")
        else:
            E(f"  I have not seen operator {om[s]} in other puzzles; I will infer it from the examples")
    E("Anyway, I should analyze the pattern of the examples to figure out what they are.")
    E()

    # 5 — RHS operator check → determine mode
    confirmed_sub_ops = set()
    rhs_op, rhs_pos, rhs_ex_nums = None, None, []
    for i, ex in enumerate(puzzle.examples):
        if ex.rhs and ex.rhs[0] in op_set:
            if rhs_op is None: rhs_op, rhs_pos = ex.rhs[0], 0
            if ex.rhs[0] == rhs_op: rhs_ex_nums.append(i + 1)
        elif ex.rhs and ex.rhs[-1] in op_set:
            if rhs_op is None: rhs_op, rhs_pos = ex.rhs[-1], -1
            if ex.rhs[-1] == rhs_op: rhs_ex_nums.append(i + 1)

    mode = 'standard'
    if rhs_op:
        confirmed_sub_ops.add(rhs_op)
        ex_label = " and ".join(f"EX{n}" for n in rhs_ex_nums)
        pos_name = "leading position RHS[0]" if rhs_pos == 0 else "trailing position RHS[-1]"
        order    = "standard (left to right, tens digit first)" if rhs_pos == 0 else "little-endian (right to left, units digit first)"
        if rhs_pos == -1: mode = 'little_endian'
        E(f"{om[rhs_op]} appears in {ex_label} at {pos_name}. In Alice world, only a subtraction operator can produce itself as a sign prefix in the output. This confirms {om[rhs_op]} represents a subtraction operation. Denote it as ~sub.")
        if rhs_pos == 0:
            E(f"{om[rhs_op]} at {pos_name} suggests {order}. A leading sign is consistent with both standard and little-endian, so I start with standard; if no solution is found, I will retry little-endian (right to left, units digit first).")
        else:
            E(f"{om[rhs_op]} at {pos_name} confirms the reading order is {order}. A trailing sign can only occur in little-endian, so this is unambiguous.")
    else:
        E("No operator symbol appears in any RHS. Reading order defaults to standard (tens digit first). If the full DFS finds no solution under standard order, I will retry with little-endian order (units digit first).")
    E()

    # 6 — query operator check
    q_op = q[2]
    q_op_unseen = q_op not in puzzle.by_op
    if not q_op_unseen:
        ex_n = next(i+1 for i, e in enumerate(puzzle.examples) if e.op == q_op)
        E(f"Operator {om[q_op]} in QUERY appears in EX{ex_n}, so I should analyze the equations and find out what the operator means.")
    else:
        E(f"Operator {om[q_op]} in QUERY does not appear in any example, so there is no way I can deduce what {om[q_op]} means and I will have to guess. I know it can only be either concatenation or arithmetic. If the examples are arithmetically solvable, I will guess an arithmetic operator, otherwise this operation should be concatenation. I need to check the examples.")
    E()

    # 7 — concat check (skip operators already confirmed as subtraction in step 5)
    concat_results = {}
    ops_to_check = [op for op in puzzle.by_op if op not in confirmed_sub_ops]
    E("Let me also check if operators are concat operation:")
    for op_sym in ops_to_check:
        ct  = check_concat(op_sym, puzzle.examples)
        ex  = puzzle.by_op[op_sym][0]
        fwd = ''.join(lm[s] for s in [ex.d1, ex.d2, ex.d3, ex.d4])
        rev = ''.join(lm[s] for s in [ex.d3, ex.d4, ex.d1, ex.d2])
        out = ''.join(lm.get(s, s) for s in ex.rhs)
        if ct:
            E(f"  {om[op_sym]}: concat_fwd={fwd}, concat_rev={rev}. Output={out}. {ct.upper()} matches -> {om[op_sym]} is concat_{ct}.")
            concat_results[op_sym] = ct
        else:
            E(f"  {om[op_sym]}: concat_fwd={fwd}, concat_rev={rev}. Output={out}. Neither matches, so {om[op_sym]} is not concat.")
    E()

    arith_ops = [op for op in puzzle.by_op if op not in concat_results]
    if not arith_ops:
        E("All operators are concat. No digit assignment needed.")
    else:
        E(f"All {len(arith_ops)} operator(s) remaining are arithmetic. We need to keep solving.")
    E()

    if arith_ops:
        # 8 — polynomial equations
        mode_lbl = "standard" if mode == 'standard' else "little-endian"
        pos_desc = "left is tens digit, right is units digit" if mode == 'standard' else "left is units digit, right is tens digit"
        E(f"Writing equations (mode={mode_lbl}, {pos_desc}):")
        def lhs_operand(da, db, parens=False):
            a, b = lm[da], lm[db]
            if mode == 'standard':
                s = f"11{a}" if a == b else f"10{a}+{b}"
            else:
                s = f"11{b}" if a == b else f"10{b}+{a}"
            return f"({s})" if parens and '+' in s else s

        for i, ex in enumerate(puzzle.examples, 1):
            if ex.op in concat_results: continue
            rhs_p = rhs_poly(ex.rhs, lm, op_set, mode)
            if ex.op in confirmed_sub_ops:
                poly = f"{lhs_operand(ex.d1, ex.d2, parens=True)} - {lhs_operand(ex.d3, ex.d4, parens=True)}"
            else:
                poly = f"{om[ex.op]}({lhs_operand(ex.d1, ex.d2)}, {lhs_operand(ex.d3, ex.d4)})"
            E(f"  Ex{i}: {poly} = {rhs_p}")
        all_l = ', '.join(lm[s] for s in puzzle.digit_syms)
        E(f"{all_l} each represent a distinct digit 0-9, leading symbol is never 0.")
        # state known vs unknown operators
        known_parts   = [f"op {om[op]} is subtraction" for op in arith_ops if op in confirmed_sub_ops]
        unknown_parts = [f"op {om[op]}" for op in arith_ops if op not in confirmed_sub_ops]
        distinct = " The operators are also distinct." if len(arith_ops) >= 2 else ""
        if known_parts and unknown_parts:
            E(f"We know {', '.join(known_parts)}, and we don't know {', '.join(unknown_parts)}.{distinct}")
        elif known_parts:
            E(f"We know {', '.join(known_parts)}.{distinct}")
        else:
            E(f"Unknown: {', '.join(unknown_parts)}.{distinct}")
        E()
        E("The solution tree is huge, let me see if I can further prune some branches.")

        # 9 — heuristic pruning (only for operators not already confirmed in step 5)
        #     Hard filter: a family that cannot produce some example's RHS digit count
        #     cannot be the operator, so it is dropped. Candidates = intersection over
        #     all the operator's examples. Order: symbol prior first, else likelihood.
        for op_sym in arith_ops:
            if op_sym in confirmed_sub_ops: continue
            ex_digits = [(i+1, len([s for s in ex.rhs if s not in op_set]))
                         for i, ex in enumerate(puzzle.examples) if ex.op == op_sym]
            ol     = om[op_sym]
            counts = [d for _, d in ex_digits]
            prior  = op_top_family(op_sym)
            names  = family_ranking(counts, prior)

            def join_ex(ns):
                items = [f"EX{n}" for n in ns]
                if len(items) <= 1: return "".join(items)
                return ", ".join(items[:-1]) + " and " + items[-1]

            obs = []
            for d in sorted(set(counts)):
                exs_d = [n for n, dd in ex_digits if dd == d]
                lbl   = join_ex(exs_d)
                have  = "have" if len(exs_d) > 1 else "has"
                if d == 1:
                    obs.append(f"{lbl} {have} 1-digit RHS (only subtraction can be a single digit; addition is ≥2 digits and multiplication ≥3)")
                elif d == 2:
                    obs.append(f"{lbl} {have} 2-digit RHS (rules out multiplication, which gives ≥3 digits)")
                elif d == 3:
                    obs.append(f"{lbl} {have} 3-digit RHS (rules out subtraction, whose magnitude is at most 2 digits)")
                else:
                    obs.append(f"{lbl} {have} {d}-digit RHS (only multiplication of two 2-digit numbers reaches {d} digits)")
            E(f"  {ol}: " + "; ".join(obs) + ".")

            if len(names) == 1:
                E(f"    Only {names[0]} survives the digit-count filter. Try {ol} = {names[0]}. Denote it as {SHORT_FAM[names[0]]}.")
            else:
                note = f" ({prior} is its most common meaning)" if prior in names and names[0] == prior else ""
                cand_str = ', '.join(f"{n} (denote {SHORT_FAM[n]})" for n in names)
                E(f"    Surviving candidates: {cand_str}. Try {ol} = {names[0]} first{note}.")
        E("No further pruning found.")
        E()

    return "\n".join(L), mode, concat_results, confirmed_sub_ops

# ── DFS tree builder ──────────────────────────────────────────────────────────
#
# Variable-by-variable DFS. For the entry example, enumerate LHS symbols
# (d1, d2, d3, d4 deduped) one by one, enforcing no-reuse via a 'used' set.
# After all LHS symbols are assigned, compute the output and match it to the
# RHS symbols. Then process other examples (enumerate their free symbols,
# verify once fully determined). Finally fill any remaining unassigned symbols.
#
# Tree node types:
#   try_combo   — one per (op_name) combination tried
#   try_var     — one variable tried at one value; children = next level
#   output_match — transition node: LHS assigned → output computed → RHS matched
#   verify_ex   — all symbols known for an example; arithmetic check
#   solution    — leaf: all symbols assigned and all examples verified

def build_tree(puzzle, mode, concat_results, confirmed_sub_ops=None):
    op_set           = set(puzzle.op_syms)
    confirmed_sub_ops = confirmed_sub_ops or set()
    arith_ops        = [op for op in puzzle.by_op if op not in concat_results]
    lead      = get_leading(puzzle, op_set, mode)
    lm        = puzzle.letter_map

    if not arith_ops:
        return {"type": "root", "mode": mode, "children": []}, None, None

    def sym_count(op):
        s = set()
        for ex in puzzle.by_op[op]:
            s.update([ex.d1, ex.d2, ex.d3, ex.d4])
            s.update(c for c in ex.rhs if c not in op_set)
        return len(s)

    entry_op  = min(arith_ops, key=sym_count)
    entry_ex  = puzzle.by_op[entry_op][0]
    other_exs = [ex for ex in puzzle.examples if ex is not entry_ex]
    lhs_syms  = list(dict.fromkeys([entry_ex.d1, entry_ex.d2,
                                     entry_ex.d3, entry_ex.d4]))

    def family_order(op_sym):
        if op_sym in confirmed_sub_ops:
            return [FAM_BY_NAME['noisy_subtraction']]
        exs = puzzle.by_op.get(op_sym, [])
        digit_counts = [len([s for s in ex.rhs if s not in op_set]) for ex in exs]
        names = family_ranking(digit_counts, op_top_family(op_sym))
        return [FAM_BY_NAME[n] for n in names]

    root = {"type": "root", "mode": mode, "children": []}

    for fam_combo in iproduct(*[family_order(op) for op in arith_ops]):
        for core_tuple in iproduct(*[fam[1] for fam in fam_combo]):
            op_cores = dict(zip(arith_ops, core_tuple))
            label    = ", ".join(f"{puzzle.op_map[op]}={CORE_FAMILY[op_cores[op]]}" for op in arith_ops)
            A, B     = lm[entry_ex.d1], lm[entry_ex.d2]
            C, D     = lm[entry_ex.d3], lm[entry_ex.d4]
            efam     = CORE_FAMILY[op_cores[entry_op]]
            eq_str   = (f"{efam}(10{A}+{B}, 10{C}+{D})"
                        if mode == 'standard'
                        else f"{efam}(10{B}+{A}, 10{D}+{C})")

            final_variants = [None]  # captured by closure; set when a solution is found

            def _resv(fam, o1, o2):
                if fam == 'noisy_mul': return (o1[0]*o2[0]-1, o1[1]*o2[1]+1)
                if fam == 'noisy_add': return (o1[0]+o2[0]-1, o1[1]+o2[1]+2)
                m = max(o1[1]-o2[0], o2[1]-o1[0], 0); return (-m-2, m+1)  # subtraction

            def _feas_pin(s, v, dom):
                """Is every arithmetic example interval-satisfiable with s pinned to v,
                given the CURRENT contracted domains of the other symbols?"""
                def rng(t):
                    if t == s: return (v, v)
                    d = dom[t]; return (min(d), max(d))
                def opv(d1, d2):
                    a, b = rng(d1), rng(d2)
                    return (10*a[0]+b[0], 10*a[1]+b[1]) if mode == 'standard' else (10*b[0]+a[0], 10*b[1]+a[1])
                def rhsv(rhs):
                    sign = 1; chars = list(rhs)
                    if chars and chars[0] in op_set:    sign = -1; chars = chars[1:]
                    elif chars and chars[-1] in op_set: sign = -1; chars = chars[:-1]
                    n = len(chars)
                    coeffs = [10**(n-1-i) for i in range(n)] if mode == 'standard' else [10**i for i in range(n)]
                    lo = hi = 0
                    for c, ss in zip(coeffs, chars):
                        r = rng(ss); lo += c*r[0]; hi += c*r[1]
                    return (-hi, -lo) if sign < 0 else (lo, hi)
                for e in puzzle.examples:
                    if e.op not in op_cores: continue
                    syms_e = (e.d1, e.d2, e.d3, e.d4, *(c for c in e.rhs if c not in op_set))
                    if s not in syms_e: continue                      # this equation doesn't constrain s
                    if any(t != s and not dom[t] for t in syms_e): return False  # an operand has no value left
                    r  = _resv(CORE_FAMILY[op_cores[e.op]], opv(e.d1, e.d2), opv(e.d3, e.d4))
                    rh = rhsv(e.rhs)
                    if r[1] < rh[0] or rh[1] < r[0]: return False
                return True

            def contract(partial):
                """Fixpoint arc/box-consistency over ALL symbols' digit domains given the
                assigned `partial`: alldifferent + per-example interval feasibility, iterated
                to a fixpoint. Returns {sym: set of feasible digits}; an EMPTY set for some
                symbol means the node is infeasible (kept, so the dead-end can be narrated).
                Sound — only removes digits that cannot complete any solution."""
                used = set(partial.values())
                dom = {s: ({partial[s]} if s in partial else
                           ((set(range(1, 10)) if s in lead else set(range(10))) - used))
                       for s in puzzle.digit_syms}
                if any(not d for d in dom.values()): return dom
                changed = True
                while changed:
                    changed = False
                    for s in puzzle.digit_syms:                       # alldifferent
                        if len(dom[s]) == 1:
                            v = next(iter(dom[s]))
                            for t in puzzle.digit_syms:
                                if t != s and t not in partial and v in dom[t]:
                                    dom[t].discard(v); changed = True
                                    if not dom[t]: return dom    # first wipeout = proximate cause
                    for s in puzzle.digit_syms:                       # interval feasibility
                        if s in partial: continue
                        keep = {v for v in dom[s] if _feas_pin(s, v, dom)}
                        if keep != dom[s]:
                            dom[s] = keep; changed = True
                            if not keep: return dom               # first wipeout = proximate cause
                return dom

            def feasible_values(sym, partial, want_deriv=False):
                """Surviving digits for `sym` after propagating every arithmetic example to a
                fixpoint over the current domains (see contract). With want_deriv, also returns
                the per-example bound reasoning, computed against the OTHER symbols' CONTRACTED
                ranges — so the lines shown are exactly what yields the contracted domain, or, on
                a dead-end, what empties a variable's domain."""
                dom_ct  = contract(partial)
                empties = [s for s in puzzle.digit_syms if not dom_ct[s]]
                ok = set() if empties else set(dom_ct[sym])
                if not want_deriv:
                    return ok
                # On a dead-end, narrate the variable propagation actually wiped out
                # (sym itself if it is empty, else the first emptied symbol).
                target = sym if (not empties or sym in empties) else empties[0]
                def orng(o):
                    if o in partial: return (partial[o], partial[o])
                    d = dom_ct[o]
                    return (min(d), max(d)) if d else ((1 if o in lead else 0), 9)
                def rng(s, v):
                    return (v, v) if s == target else orng(s)
                def opv(d1, d2, v):
                    a, b = rng(d1, v), rng(d2, v)
                    return (10*a[0]+b[0], 10*a[1]+b[1]) if mode == 'standard' else (10*b[0]+a[0], 10*b[1]+a[1])
                def resv(fam, o1, o2):
                    if fam == 'noisy_mul': return (o1[0]*o2[0]-1, o1[1]*o2[1]+1)
                    if fam == 'noisy_add': return (o1[0]+o2[0]-1, o1[1]+o2[1]+2)
                    m = max(o1[1]-o2[0], o2[1]-o1[0], 0); return (-m-2, m+1)  # subtraction
                def rhsv(rhs, v):
                    sign = 1; chars = list(rhs)
                    if chars and chars[0] in op_set:    sign = -1; chars = chars[1:]
                    elif chars and chars[-1] in op_set: sign = -1; chars = chars[:-1]
                    n = len(chars)
                    coeffs = [10**(n-1-i) for i in range(n)] if mode == 'standard' else [10**i for i in range(n)]
                    lo = hi = 0
                    for c, s in zip(coeffs, chars):
                        r = rng(s, v); lo += c*r[0]; hi += c*r[1]
                    return (-hi, -lo) if sign < 0 else (lo, hi)
                used_digits = set(partial.values())
                naive = {v for v in range(10) if not (target in lead and v == 0) and v not in used_digits}
                def per_example_surv(e, fam):
                    s = set()
                    for v in naive:
                        r = resv(fam, opv(e.d1, e.d2, v), opv(e.d3, e.d4, v)); rh = rhsv(e.rhs, v)
                        if not (r[1] < rh[0] or rh[1] < r[0]): s.add(v)
                    return s
                def lett(d1, d2):
                    a, b = (d1, d2) if mode == 'standard' else (d2, d1)
                    return f"11{lm[a]}" if a == b else f"(10{lm[a]}+{lm[b]})"
                def add_line(e, ei, surv):
                    coeff = {}
                    pos = [(e.d1, 10), (e.d2, 1), (e.d3, 10), (e.d4, 1)] if mode == 'standard' else \
                          [(e.d1, 1), (e.d2, 10), (e.d3, 1), (e.d4, 10)]
                    for s, c in pos: coeff[s] = coeff.get(s, 0) + c
                    chars = [s for s in e.rhs if s not in op_set]; n = len(chars)
                    pl = [10**(n-1-i) for i in range(n)] if mode == 'standard' else [10**i for i in range(n)]
                    for c, s in zip(pl, chars): coeff[s] = coeff.get(s, 0) - c
                    cs = coeff.get(target, 0)
                    if cs == 0: return None
                    others = [s for s in coeff if s != target and coeff[s] != 0]
                    lo, hi = -2, 1
                    for o in others:
                        co = coeff[o]; r = orng(o)
                        if co > 0: lo -= co*r[1]; hi -= co*r[0]
                        else:      lo -= co*r[0]; hi -= co*r[1]
                    terms = " ".join((f"- {coeff[o]}{lm[o]}" if coeff[o] > 0 else f"+ {-coeff[o]}{lm[o]}") for o in others)
                    cond = ", ".join(f"{lm[o]}∈[{orng(o)[0]},{orng(o)[1]}]" for o in others)
                    cstr = f"{cs}{lm[target]}" if cs != 1 else lm[target]
                    return f"EX{ei} (add): {cstr} = [-2,1] {terms};  {cond}  ⟹  {cstr} ∈ [{lo},{hi}], so {lm[target]} ∈ {{{','.join(map(str, sorted(surv)))}}}"
                def iv_line(e, ei, fam, surv):
                    o1, o2 = lett(e.d1, e.d2), lett(e.d3, e.d4)
                    res_los = [resv(fam, opv(e.d1, e.d2, v), opv(e.d3, e.d4, v))[0] for v in naive]
                    res_his = [resv(fam, opv(e.d1, e.d2, v), opv(e.d3, e.d4, v))[1] for v in naive]
                    rl = [rhsv(e.rhs, v)[0] for v in naive]; rr = [rhsv(e.rhs, v)[1] for v in naive]
                    opc = '×' if fam == 'noisy_mul' else '∓'
                    word = 'mul' if fam == 'noisy_mul' else 'subtraction'
                    return (f"EX{ei} ({word}): {o1} {opc} {o2} ∈ [{min(res_los)},{max(res_his)}]; "
                            f"RHS ∈ [{min(rl)},{max(rr)}]  ⟹  {lm[target]} ∈ {{{','.join(map(str, sorted(surv)))}}}")
                deriv = []
                combined = set(naive)
                for ei, e in enumerate(puzzle.examples, 1):
                    if e.op not in op_cores: continue
                    if target not in ([e.d1, e.d2, e.d3, e.d4] + [s for s in e.rhs if s not in op_set]): continue
                    fam = CORE_FAMILY[op_cores[e.op]]
                    surv = per_example_surv(e, fam)
                    combined &= surv
                    if surv == naive: continue
                    line = add_line(e, ei, surv) if fam == 'noisy_add' else iv_line(e, ei, fam, surv)
                    if line: deriv.append(line)
                # any digit that satisfies every equation but is gone from the domain was removed
                # by all-different (taken by an already-pinned symbol) — state that explicitly.
                gap = sorted(combined - set(dom_ct[target]))
                if gap:
                    holders = []
                    for g in gap:
                        w = next((x for x in puzzle.digit_syms if x != target and dom_ct[x] == {g}), None)
                        holders.append(f"{g}(={lm[w]})" if w else str(g))
                    deriv.append(f"all-different: {', '.join(holders)} already taken  ⟹  {lm[target]} ∈ "
                                 f"{{{','.join(map(str, sorted(dom_ct[target])))}}}")
                return ok, {"var": lm[target], "domain": sorted(dom_ct[target]), "lines": deriv}

            # ── nested DFS helpers (closures over op_cores, mode, etc.) ──────

            def dfs_lhs(idx, partial, used, op_variants):
                if idx == len(lhs_syms):
                    return do_output(partial, used, op_variants)
                sym = lhs_syms[idx]
                cand, bound = feasible_values(sym, partial, want_deriv=True)
                if not bound["domain"]:
                    return [{"type": "dead_end", "var": bound["var"], "bound": bound}], None
                children, sol = [], None
                for v in range(0, 10):
                    if v in used: continue
                    if sym in lead and v == 0: continue
                    if v not in cand: continue
                    ch, s = dfs_lhs(idx + 1, {**partial, sym: v}, used | {v}, op_variants)
                    node  = {"type": "try_var", "var": lm[sym], "val": v,
                             "outcome": "success" if s else "fail", "children": ch}
                    if not children: node["bound"] = bound   # only first sibling carries the derivation
                    children.append(node)
                    if s: sol = s; break
                return children, sol

            def do_output(partial, used, op_variants):
                """Try each variant of the entry op's core (priority order) against the RHS."""
                if mode == 'standard':
                    lv = 10 * partial[entry_ex.d1] + partial[entry_ex.d2]
                    rv = 10 * partial[entry_ex.d3] + partial[entry_ex.d4]
                else:
                    lv = 10 * partial[entry_ex.d2] + partial[entry_ex.d1]
                    rv = 10 * partial[entry_ex.d4] + partial[entry_ex.d3]
                core_name        = op_cores[entry_op]
                out_syms         = [s for s in entry_ex.rhs if s not in op_set]
                has_neg          = bool(entry_ex.rhs and
                                        (entry_ex.rhs[0] in op_set or entry_ex.rhs[-1] in op_set))

                cands = core_cands(core_name, lv, rv)
                if len(puzzle.by_op[entry_op]) == 1:
                    # entry op has no other example to tell tying variants apart, so
                    # equal-value variants are interchangeable: keep the top-priority one.
                    seen = set(); cands = [(v, var) for v, var in cands
                                           if not (v in seen or seen.add(v))]

                attempt_nodes = []
                for out_val, variant in cands:
                    arith   = op_str(variant, lv, rv)

                    def fail(reason):
                        attempt_nodes.append({"type": "variant_try", "variant": variant,
                                              "arithmetic": arith, "out_val": out_val,
                                              "outcome": "fail", "reason": reason})

                    if has_neg and out_val >= 0:   fail("expected negative"); continue
                    if not has_neg and out_val < 0: fail(f"negative: {out_val}"); continue

                    abs_str = str(abs(out_val))
                    if len(abs_str) != len(out_syms):
                        fail(f"len {len(abs_str)} vs {len(out_syms)}"); continue

                    out_digits = ([int(c) for c in abs_str] if mode == 'standard'
                                  else list(reversed([int(c) for c in abs_str])))
                    if out_syms and out_syms[0] in lead and out_digits[0] == 0:
                        fail("leading zero on output"); continue

                    new_p, new_u = dict(partial), set(used)
                    conflict = None
                    for sym, dig in zip(out_syms, out_digits):
                        if sym in new_p:
                            if new_p[sym] != dig:
                                conflict = f"{lm[sym]} conflict: {new_p[sym]}≠{dig}"; break
                        elif dig in new_u:
                            conflict = f"digit {dig} reused"; break
                        else:
                            new_p[sym] = dig; new_u.add(dig)
                    if conflict is not None:
                        fail(conflict); continue

                    # variant matches the entry RHS — lock it, proceed
                    new_variants = {**op_variants, entry_op: variant}
                    ch, s = dfs_others(other_exs, 0, new_p, new_u, new_variants)
                    attempt_nodes.append({"type": "variant_try", "variant": variant,
                                          "arithmetic": arith, "out_val": out_val,
                                          "partial": {lm[k]: w for k, w in new_p.items()},
                                          "outcome": "success" if s else "fail",
                                          "children": ch})
                    if s:
                        node = {"type": "output_match", "core": core_name,
                                "outcome": "success", "children": attempt_nodes}
                        return [node], s

                node = {"type": "output_match", "core": core_name,
                        "outcome": "fail", "children": attempt_nodes}
                return [node], None

            def dfs_others(exs, idx, partial, used, op_variants):
                if idx == len(exs):
                    return fill_remaining(partial, used, op_variants)
                ex = exs[idx]
                if ex.op not in op_cores:
                    return dfs_others(exs, idx + 1, partial, used, op_variants)
                all_s = list(dict.fromkeys(
                    [ex.d1, ex.d2, ex.d3, ex.d4] +
                    [s for s in ex.rhs if s not in op_set]))
                free = [s for s in all_s if s not in partial]
                if not free:
                    return verify_and_next(ex, partial, used, exs, idx, op_variants)
                return dfs_free(free, 0, ex, partial, used, exs, idx, op_variants)

            def verify_and_next(ex, partial, used, exs, idx, op_variants):
                """All symbols known — determine/verify variant and proceed."""
                op_sym    = ex.op
                core_name = op_cores[op_sym]
                lv = sym_val(ex.d1, ex.d2, partial, mode)
                rv = sym_val(ex.d3, ex.d4, partial, mode)
                ev = rhs_int(ex.rhs, partial, mode, op_set)
                ex_n = puzzle.examples.index(ex) + 1

                if op_sym in op_variants:
                    locked = op_variants[op_sym]
                    got   = apply_op(locked, lv, rv)
                    arith = op_str(locked, lv, rv)
                    if got == ev:
                        ch, s = dfs_others(exs, idx + 1, partial, used, op_variants)
                        node  = {"type": "verify_ex", "ex": ex_n, "variant": locked,
                                 "arithmetic": arith, "expected": ev,
                                 "outcome": "pass", "children": ch}
                        return [node], s
                    node = {"type": "verify_ex", "ex": ex_n, "variant": locked,
                            "arithmetic": arith, "expected": ev, "got": got, "outcome": "fail"}
                    return [node], None
                else:
                    # variant not yet locked for this op: try every variant whose value
                    # matches (priority order); subtraction variants can tie on value.
                    matches = [v for val, v in core_cands(core_name, lv, rv) if val == ev]
                    if not matches:
                        cand_vals = sorted({val for val, _ in core_cands(core_name, lv, rv)})
                        node = {"type": "verify_ex", "ex": ex_n, "core": core_name,
                                "family": CORE_FAMILY[core_name], "expected": ev,
                                "cand_vals": cand_vals, "outcome": "fail",
                                "reason": f"no {CORE_FAMILY[core_name]} variant equals {ev}"}
                        return [node], None
                    # tying variants can only diverge at a later example of the same op;
                    # if none remains, they are interchangeable — keep the top-priority one.
                    if len(matches) > 1 and not any(e.op == op_sym for e in exs[idx + 1:]):
                        matches = matches[:1]
                    nodes = []
                    for variant in matches:
                        arith = op_str(variant, lv, rv)
                        new_variants = {**op_variants, op_sym: variant}
                        ch, s = dfs_others(exs, idx + 1, partial, used, new_variants)
                        # this example's equation is satisfied (that is why we lock here);
                        # branch success/failure is carried by s, shown in the children.
                        nodes.append({"type": "verify_ex", "ex": ex_n, "variant": variant,
                                      "variant_locked_here": True, "arithmetic": arith,
                                      "expected": ev, "outcome": "pass", "children": ch})
                        if s:
                            return nodes, s
                    return nodes, None

            def dfs_free(free_syms, idx, ex, partial, used, exs, ex_idx, op_variants):
                if idx == len(free_syms):
                    return verify_and_next(ex, partial, used, exs, ex_idx, op_variants)
                sym = free_syms[idx]
                cand, bound = feasible_values(sym, partial, want_deriv=True)
                if not bound["domain"]:
                    return [{"type": "dead_end", "var": bound["var"], "bound": bound}], None
                children, sol = [], None
                for v in range(0, 10):
                    if v in used: continue
                    if sym in lead and v == 0: continue
                    if v not in cand: continue
                    ch, s = dfs_free(free_syms, idx + 1, ex,
                                     {**partial, sym: v}, used | {v}, exs, ex_idx, op_variants)
                    node  = {"type": "try_var", "var": lm[sym], "val": v,
                             "outcome": "success" if s else "fail", "children": ch}
                    if not children: node["bound"] = bound
                    children.append(node)
                    if s: sol = s; break
                return children, sol

            def all_examples_ok(asgn, op_variants):
                if any(asgn.get(s) == 0 for s in lead):
                    return False
                for ex in puzzle.examples:
                    if ex.op not in op_cores: continue
                    if ex.op not in op_variants: return False
                    v = apply_op(op_variants[ex.op],
                                 sym_val(ex.d1, ex.d2, asgn, mode),
                                 sym_val(ex.d3, ex.d4, asgn, mode))
                    if v != rhs_int(ex.rhs, asgn, mode, op_set): return False
                return True

            def fill_remaining(partial, used, op_variants):
                missing = [s for s in puzzle.digit_syms if s not in partial]
                if not missing:
                    if all_examples_ok(partial, op_variants):
                        final_variants[0] = dict(op_variants)
                        return ([{"type": "solution",
                                  "assignment": {lm[s]: v for s, v in partial.items()},
                                  "variants": {puzzle.op_map[k]: v for k, v in op_variants.items()},
                                  "outcome": "success"}], partial)
                    return ([{"type": "solution", "outcome": "fail",
                              "reason": "final verify failed"}], None)

                avail = [d for d in range(10) if d not in used]
                for perm in permutations(avail, len(missing)):
                    trial = dict(partial)
                    trial.update(zip(missing, perm))
                    if all_examples_ok(trial, op_variants):
                        final_variants[0] = dict(op_variants)
                        return ([{"type": "solution",
                                  "assignment": {lm[s]: v for s, v in trial.items()},
                                  "variants": {puzzle.op_map[k]: v for k, v in op_variants.items()},
                                  "outcome": "success"}], trial)
                return ([{"type": "solution", "outcome": "fail",
                          "reason": "no valid assignment for remaining symbols"}], None)

            # ── run DFS for this combo ────────────────────────────────────────
            children, sol = dfs_lhs(0, {}, set(), {})
            combo_node = {
                "type":           "try_combo",
                "label":          label,
                "op_cores":       op_cores,
                "entry_ex":       puzzle.examples.index(entry_ex) + 1,
                "entry_equation": eq_str,
                "outcome":        "success" if sol else "fail",
                "assignment":     {lm[s]: v for s, v in sol.items()} if sol else None,
                "variants":       {puzzle.op_map[k]: v for k, v in (final_variants[0] or {}).items()} if sol else None,
                "children":       children,
            }
            root["children"].append(combo_node)
            if sol:
                op_names = final_variants[0] or {}
                return root, sol, op_names

    return root, None, None

# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 cot_generator.py <puzzle_dir>"); sys.exit(1)

    pdir   = Path(sys.argv[1])
    puzzle = parse(pdir / "question.txt")

    prefix, mode, concat_results, confirmed_sub_ops = make_prefix(puzzle)
    tree, solution, op_names = build_tree(puzzle, mode, concat_results, confirmed_sub_ops)

    if solution is None and mode == 'standard':
        tree_le, solution, op_names = build_tree(puzzle, 'little_endian', concat_results, confirmed_sub_ops)
        tree["little_endian_retry"] = tree_le

    (pdir / "prefix.txt").write_text(prefix)
    (pdir / "tree.json").write_text(json.dumps(tree, indent=2))

    print(f"Written: {pdir}/prefix.txt  {pdir}/tree.json")
    if solution:
        print(f"Solution: { {puzzle.letter_map[s]: v for s, v in solution.items()} }")
        print(f"Ops:      {op_names}")
    else:
        print("No solution found.")

if __name__ == "__main__":
    main()
