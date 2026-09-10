#!/usr/bin/env python3
"""Equation-numeric solver + honest CoT generator — the cryptarithm SUB-PROBLEM.

equation_numeric is the cryptarithm with the operands ALREADY SETTLED: the operands are literal
numbers, so the digit-DFS is gone. What remains is exactly the cryptarithm step that runs once an
example's operands are known — read off the operator's MEANING ("We now know all the operands in
EX1: 47 ~mul 47, RHS = 2209. 2209 = 47×47, so lock f = mul."). The CoT REUSES the cryptarithm cot's
section structure and wording verbatim wherever the literal-operand setting allows it; only the
digit-cipher-specific lines (digit legend, polynomial operand forms, the digit DFS, the Solution
line) are adapted, since the operands here are plain numbers.

The CoT is a FORWARD-ONLY trace: it is written as the search runs, never edited afterward, and no
step uses information a later step would discover. The reading order is NOT precomputed — the solver
tries left-to-right first (writing it out), and only if left-to-right leaves an operator with no rule
does it retry right-to-left (writing that out too); it keeps whichever reading explained more operators
(ties broken by whichever pins the query operator, else left-to-right).
Each operator's noisy operation is searched within OUR cryptarithm vocabulary (VOCAB: mul/add/sub
families + concat); an operator needing a rule outside it (max-mod-min, …) is honestly left
undetermined, never fabricated. Matching is by STRING (faithful to the generator: "01" != "1").

Operator letters f,g,h are assigned by ORDER OF FIRST APPEARANCE in the examples (cryptarithm rule),
not by ASCII. label.txt holds OUR problem-type label (see `classify`). Run `python3 _gen_eq.py` for
all puzzles, or `python3 _gen_eq.py <id>` to regenerate a single example.
"""
import os, re, sys, csv, math, collections

REPO = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/02_train/260512_huikang_085/repo'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data'
USED_CSV = f'{DATA}/260523_new_data/260523_huikang_newsolver.csv'
UNUSED_CSV = f'{DATA}/260514_huikang_update/huikang_unused.csv'
csv.field_size_limit(10 ** 7)
sys.path.insert(0, REPO)
from reasoners.equation_numeric import (_all_candidates, _common_candidates, _rare_candidates, _rev)

COMMON = [n for n, _ in _common_candidates(12, 34, "12", "34")]
RARE = [n for n, _ in _rare_candidates(12, 34, "12", "34")]
ALLNAMES = [n for n, _ in _all_candidates(12, 34, "12", "34")]
MODES = [(False, False), (True, True)]
MODE_DESC = {(False, False): "left-to-right order (tens digit first)",
             (True, True): "right-to-left order (units digit first)"}

# compact, cryptarithm-style expression of an operation on its (reading-frame) operands a,b
CEXPR = {
    'concatenation': lambda a, b: f"{a}∥{b}", 'reverse concatenation': lambda a, b: f"{b}∥{a}",
    'addition': lambda a, b: f"{a}+{b}", 'add+1': lambda a, b: f"{a}+{b}+1", 'add-1': lambda a, b: f"{a}+{b}-1",
    'add+2': lambda a, b: f"{a}+{b}+2", 'add-2': lambda a, b: f"{a}+{b}-2",
    'multiplication': lambda a, b: f"{a}×{b}", 'multiply+1': lambda a, b: f"{a}×{b}+1", 'multiply-1': lambda a, b: f"{a}×{b}-1",
    'multiply+2': lambda a, b: f"{a}×{b}+2", 'multiply-2': lambda a, b: f"{a}×{b}-2",
    'absolute difference': lambda a, b: f"|{a}-{b}|", 'negated absolute difference': lambda a, b: f"-|{a}-{b}|",
    'subtraction (a-b)': lambda a, b: f"{a}-{b}", 'reverse subtraction (b-a)': lambda a, b: f"{b}-{a}",
    'sub+1': lambda a, b: f"{a}-{b}+1", 'sub-1': lambda a, b: f"{a}-{b}-1",
    'sub+2': lambda a, b: f"{a}-{b}+2", 'sub-2': lambda a, b: f"{a}-{b}-2",
}
# huikang operation name -> cryptarithm variant name used on the lock line
CRYPT_NAME = {
    'addition': 'add', 'add+1': 'add_plus1', 'add-1': 'add_minus1', 'add+2': 'add_plus2', 'add-2': 'add_minus2',
    'multiplication': 'mul', 'multiply+1': 'mul_plus1', 'multiply-1': 'mul_minus1', 'multiply+2': 'mul_plus2', 'multiply-2': 'mul_minus2',
    'absolute difference': 'absdiff', 'negated absolute difference': 'neg_absdiff',
    'subtraction (a-b)': 'sub_signed', 'reverse subtraction (b-a)': 'rsub_signed',
    'sub+1': 'sub_signed_plus1', 'sub-1': 'sub_signed_minus1', 'sub+2': 'sub_signed_plus2', 'sub-2': 'sub_signed_minus2',
    'concatenation': 'concat_fwd', 'reverse concatenation': 'concat_rev',
}
FAM = {}
for _n in ('multiplication', 'multiply+1', 'multiply-1', 'multiply+2', 'multiply-2'): FAM[_n] = 'mul'
for _n in ('addition', 'add+1', 'add-1', 'add+2', 'add-2'): FAM[_n] = 'add'
for _n in ('absolute difference', 'negated absolute difference', 'subtraction (a-b)',
           'reverse subtraction (b-a)', 'sub+1', 'sub-1', 'sub+2', 'sub-2'): FAM[_n] = 'sub'
# ±2 members: huikang's _all_candidates stops at ±1, but Alice-world noise is "±1 or ±2", so we model
# AND compute the ±2 ops ourselves — the family is a PRIOR over the rules, not huikang's enumeration.
EXTRA_OPS = {'add+2': lambda a, b: a + b + 2, 'add-2': lambda a, b: a + b - 2,
             'multiply+2': lambda a, b: a * b + 2, 'multiply-2': lambda a, b: a * b - 2,
             'sub+2': lambda a, b: a - b + 2, 'sub-2': lambda a, b: a - b - 2}
ARITH_FAMS = ['noisy_mul', 'noisy_add', 'noisy_subtraction']      # the three arithmetic families
FAMNAME_TAG = {'noisy_mul': '~mul', 'noisy_add': '~add', 'noisy_subtraction': '~sub'}
PRIOR_TAG = {'noisy_mul': '~mul', 'noisy_add': '~add', 'noisy_subtraction': '~sub', 'concat': '~concat'}   # incl. concat, for the operator-prior list
CANON_FAM = {'+': 'noisy_add', '-': 'noisy_subtraction', '*': 'noisy_mul'}   # earth-analog strong priors
BASEOP = {'noisy_mul': 'multiplication', 'noisy_add': 'addition', 'noisy_subtraction': 'absolute difference'}


def fam_of(nm):
    """Operation name -> family bucket, in the cryptarithm vocabulary (operator_dict)."""
    f = FAM.get(nm)
    if f == 'mul': return 'noisy_mul'
    if f == 'add': return 'noisy_add'
    if f == 'sub': return 'noisy_subtraction'
    return 'concat' if 'concat' in nm else 'other'


# the operations we search — the Alice-world families {base, ±1, ±2} + concat. huikang's ALLNAMES only
# enumerates up to ±1, so the ±2 members (EXTRA_OPS) are appended explicitly; the family is OUR prior.
VOCAB = [nm for nm in ALLNAMES if fam_of(nm) != 'other'] + list(EXTRA_OPS)


def cexpr(nm, a, b):
    return CEXPR[nm](a, b) if nm in CEXPR else f"{nm}({a},{b})"


def cname(nm):
    return CRYPT_NAME.get(nm, nm)


def famtag_of(nm):
    return {'mul': '~mul', 'add': '~add', 'sub': '~sub'}.get(FAM.get(nm), '~?')


def nrank(nm):
    if nm in COMMON: return (0, COMMON.index(nm))
    if nm in RARE: return (1, RARE.index(nm))
    return (2, list(EXTRA_OPS).index(nm))   # ±2 variants — rarest, tried last


def rev_if(s, ro):
    return s[::-1] if ro else s


# ── parsing ─────────────────────────────────────────────────────────────────────
def parse_eq(s):
    m = re.match(r'(\d+)(\D)(\d+)$', s)
    return (m.group(1), m.group(2), m.group(3)) if m else None


def parse_prompt(prompt):
    examples = []; question = None
    for ln in prompt.splitlines():
        ln = ln.strip()
        if ln.lower().startswith('now, determine the result for'):
            question = ln.split(':', 1)[1].strip() if ':' in ln else None
        elif '=' in ln:
            inp, out = ln.split('=', 1)
            examples.append({'input_value': inp.strip(), 'output_value': out.strip()})
    return examples, question


# ── string-faithful operation machinery (matches the generator: "01" != "1") ─────
def out_norm(s):
    """Output string with its sign symbol stripped, normalised to a leading '-' if negative."""
    s = s.strip()
    if not s: return s
    if s[0] == '-': return s
    if not s[0].isdigit(): return '-' + s[1:]
    if not s[-1].isdigit(): return '-' + s[:-1]
    return s


def sign_in_place(s):
    """Show an operator-symbol sign as '-' in its WRITTEN position (left-to-right view): 51$ -> 51-, $51 -> -51, -91 -> -91."""
    s = s.strip()
    if not s: return s
    if not s[0].isdigit(): return '-' + s[1:]
    if not s[-1].isdigit(): return s[:-1] + '-'
    return s


def raw_on(sa, sb, nm, ro):
    """Operation nm on the reading-frame operands (reversed iff ro), as a STRING (no result reversal)."""
    ta, tb = rev_if(sa, ro), rev_if(sb, ro)
    if nm in EXTRA_OPS:                       # ±2 members huikang doesn't enumerate — compute them ourselves
        return str(EXTRA_OPS[nm](int(ta), int(tb)))
    return next((r for n, r in _all_candidates(int(ta), int(tb), ta, tb) if n == nm), None)


def target_of(out_raw, rr):
    """The output as it must read in this mode (right-to-left reading flips the digit string)."""
    t = out_norm(out_raw)
    return _rev(t) if rr else t


def fits(sa, sb, nm, ro, rr, out_raw):
    r = raw_on(sa, sb, nm, ro)
    return r is not None and r == target_of(out_raw, rr)


def consistent_ops(exs, ro, rr):
    """Operations from OUR vocabulary consistent with EVERY example under mode (ro,rr), in preference order
    (nrank) — except within ~sub we prefer literal subtraction a-b over the contrived -|a-b| (swap the two)."""
    swap = {'negated absolute difference': 'subtraction (a-b)', 'subtraction (a-b)': 'negated absolute difference'}
    return sorted([nm for nm in VOCAB if all(fits(sa, sb, nm, ro, rr, o) for sa, sb, o in exs)],
                  key=lambda nm: nrank(swap.get(nm, nm)))


def fam_variant_order(fam):
    """Variants of family `fam` in the decided preference order (identical to consistent_ops' ordering):
    for ~sub this is |a-b| > a-b > -|a-b| > b-a, then the ±1/±2 noisy variants."""
    swap = {'negated absolute difference': 'subtraction (a-b)', 'subtraction (a-b)': 'negated absolute difference'}
    return sorted([n for n in VOCAB if fam_of(n) == fam], key=lambda nm: nrank(swap.get(nm, nm)))


def first_variant_match(sa, sb, o, fam, ro, rr):
    """The first variant of family `fam` (in fam_variant_order) that produces this ONE example's output —
    i.e. what the greedy search locks when it breaks at the first match — or None."""
    return next((v for v in fam_variant_order(fam) if fits(sa, sb, v, ro, rr, o)), None)


def coverage(by, mode):
    return sum(1 for op, exs in by.items() if consistent_ops(exs, mode[0], mode[1]))


def choose_mode(by):
    """Pin the global reading: the mode (left-to-right vs right-to-left) under which the most operators are
    explainable by OUR vocabulary, left-to-right-first on a tie. (Used only by classify/get_prior, NOT by
    the CoT — the CoT discovers the reading sequentially.)"""
    return max(MODES, key=lambda m: (coverage(by, m), 1 if m == (False, False) else 0))


# ── operator_dict prior (OUR dictionary, built from this corpus) ────────────────
_PRIOR = None


def get_prior():
    global _PRIOR
    if _PRIOR is not None:
        return _PRIOR
    cnt = collections.defaultdict(collections.Counter)
    for path in (USED_CSV, UNUSED_CSV):
        for r in csv.DictReader(open(path, newline='')):
            if not str(r.get('category', '')).startswith('equation_numeric'):
                continue
            ex, _ = parse_prompt(r['prompt'])
            by = collections.defaultdict(list)
            for e in ex:
                p = parse_eq(e['input_value'])
                if not p: continue
                sa, op, sb = p
                by[op].append((sa, sb, e['output_value'].strip()))
            if not by:
                continue
            mode = choose_mode(by)
            for op, exs in by.items():
                fit = consistent_ops(exs, mode[0], mode[1])
                if fit:
                    cnt[op][fam_of(fit[0])] += 1
    _PRIOR = cnt
    return cnt


def op_fams_ordered(sym):
    return [f for f, _ in get_prior().get(sym, collections.Counter()).most_common()]


def op_top_family(sym):
    a = [f for f in op_fams_ordered(sym) if f in ARITH_FAMS]
    return a[0] if a else None


def op_seen_arith(sym):
    return [f for f in op_fams_ordered(sym) if f in ARITH_FAMS]


def sign_format(by):
    for op, exs in by.items():
        for sa, sb, o in exs:
            s = o.strip()
            if not s: continue
            if s[0] == '-': return 'minus', '-'
            if not s[0].isdigit(): return 'prefix', s[0]
            if not s[-1].isdigit(): return 'suffix', s[-1]
    return 'none', ''


def render_answer(final, fmt, qop):
    if not final.startswith('-'): return final
    mag = final[1:]
    if fmt == 'suffix': return mag + qop
    if fmt == 'prefix': return qop + mag
    return '-' + mag                      # 'minus' or 'none' (sign convention unknown): keep the minus


# ── digit-count pruning (cryptarithm idiom) ─────────────────────────────────────
def _rules_out(d):
    if d <= 1: return "rules out addition and multiplication, only a subtraction-type reaches 1 digit"
    if d == 2: return "rules out multiplication, which gives ≥3 digits"
    if d == 3: return "rules out subtraction, whose magnitude is at most 2 digits"
    return "only multiplication of two 2-digit numbers reaches 4 digits"


def _surv_for(d):
    if d <= 1: return {'noisy_subtraction'}
    if d == 2: return {'noisy_add', 'noisy_subtraction'}
    if d == 3: return {'noisy_add', 'noisy_mul'}
    return {'noisy_mul'}


# ── the CoT ─────────────────────────────────────────────────────────────────────
def gen_cot(d):
    raw_exs = d['examples']
    items = collections.defaultdict(list)                # op -> [(sa, sb, out_raw, exidx)]
    order = []                                           # operators in order of FIRST appearance
    for i, e in enumerate(raw_exs, 1):
        p = parse_eq(e['input_value'])
        if not p: continue
        sa, op, sb = p
        items[op].append((sa, sb, e['output_value'].strip(), i))
        if op not in order: order.append(op)
    q = parse_eq(d['question']); qsa, qop, qsb = q
    full_order = order + ([qop] if qop not in order else [])
    label = {op: chr(ord('f') + k) for k, op in enumerate(full_order)}
    trip = {op: [(a, b, o) for a, b, o, _ in items[op]] for op in items}
    fmt, signchar = sign_format(trip)
    seen = qop in items
    FAM_REPS = {'noisy_subtraction': ['absolute difference', 'subtraction (a-b)', 'reverse subtraction (b-a)'],
                'noisy_add': ['addition'], 'noisy_mul': ['multiplication']}

    L = []
    # opening
    L.append("I need to infer the transformation rule from the examples. The operands are given "
             "already, I only need to deduce the operators.")
    L.append("")
    # assign letters (operands are numbers, so only the operator symbols get letters)
    L.append("First, let me assign letters to each symbol:")
    for op in full_order:
        L.append(f"  {op} -> {label[op]}")
    L.append("")
    # letter form
    L.append("I will convert all the equations to letter form:")
    for i, e in enumerate(raw_exs, 1):
        p = parse_eq(e['input_value'])
        if not p: continue
        sa, op, sb = p; o = e['output_value'].strip()
        L.append(f"  EX{i} {sa}{op}{sb} = {o} becomes {sa} {label[op]} {sb} = {sign_in_place(o)}")
    L.append(f"  QUERY {qsa}{qop}{qsb} becomes {qsa} {label[qop]} {qsb}")
    L.append("")
    # prior knowledge about Alice World: (1) operations + variant sets, (2) distinctness, (3) reading-order rule
    L.append("Prior knowledge for this kind of question:")
    L.append("1. We only consider four kinds of operations — noisy_multiplication (~mul), noisy_addition (~add), "
             "noisy_subtraction (~sub), and noisy_concatenation (~concat). They are noisy because the result "
             "is one of a fixed set of variants: the exact value, off by ±1 or ±2, and — for subtraction — "
             "also its negated or operand-reversed form. Specifically:")
    L.append("   ~mul = [a×b, a×b±1, a×b±2], ~add = [a+b, a+b±1, a+b±2], "
             "~sub = [|a-b|, a-b, -|a-b|, b-a, a-b±1, a-b±2], ~concat = [ab, ba], all ordered by frequency. "
             "Any other operation is deemed exotic.")
    L.append("2. The arithmetic operators are distinct, but the concatenation operator may repeat.")
    L.append("3. If an equation contains a number with a leading 0, the operator must be ~concat, "
             "since a number with a leading 0 is invalid for an arithmetic operator.")
    L.append("4. If a symbol exists in the result (RHS), it must be a negative sign.")
    L.append("5. The reading order is either rightward or leftward (units digit first). "
             "I will check rightward first.")
    # operator prior (prior #6): what each symbol has meant in OTHER puzzles (the operator_dict)
    L.append("6. For the operators in this puzzle:")
    for op in full_order:
        fams = op_fams_ordered(op)
        tags = [PRIOR_TAG.get(f, f) for f in fams]
        if op in CANON_FAM:
            top = CANON_FAM[op]
            others = [PRIOR_TAG.get(f, f) for f in fams if f != top] or [PRIOR_TAG[f] for f in ARITH_FAMS if f != top]
            L.append(f"  {op}, denoted as {label[op]}, is usually {PRIOR_TAG[top]}, but it can also represent "
                     f"[{', '.join(others)}], ordered by frequency")
        elif fams:
            L.append(f"  {op}, denoted as {label[op]}, it represents [{', '.join(tags)}], ordered by frequency")
        else:
            L.append(f"  {op}, denoted as {label[op]}, has not appeared in other puzzles")
    L.append("  Anyway, I should analyze the pattern of the examples to figure out what they are.")
    L.append("")
    # sign check (only subtraction makes its own sign): prose lines + the right-to-left start note (trailing only) + unseen-query flag
    confirmed_sub = set()
    sign_hits = []
    for op in full_order:
        for (sa, sb, o, i) in items.get(op, []):
            s = o.strip()
            if not s or (s[0].isdigit() and s[-1].isdigit()): continue
            pos, pp = ('leading', 'prefix') if not s[0].isdigit() else ('trailing', 'suffix')
            sign_hits.append((op, label[op], i, pos, pp)); confirmed_sub.add(op); break
    for op, lab, i, pos, pp in sign_hits:
        L.append(f"{lab} appears in EX{i} output side (RHS), that means it can only be a negative sign, "
                 f"so {lab} = [~sub].")
    if confirmed_sub:                                    # distinctness (prior #2): no OTHER operator can be ~sub
        others = [label[op] for op in order if op not in confirmed_sub]
        if others:
            ops_str = (others[0] if len(others) == 1 else
                       f"{others[0]} and {others[1]}" if len(others) == 2 else
                       ", ".join(others[:-1]) + f" and {others[-1]}")
            L.append(f"By the distinct operator rule, {ops_str} cannot be ~sub.")
    if sign_hits:
        L.append("")

    # ── ONE forward-only attempt under a fixed reading, narrated as the four steps §n.1–§n.4:
    #    .1 write the equations in this reading's frame (operands AND output reversed iff right-to-left),
    #    .2 concat check, .3 digit-count prune, .4 search/lock. The reading m and branch index n are fixed
    #    by the CALLER; this never peeks at the other reading or at the answer.
    def attempt(m, n):
        ro2, rr2 = m
        rdir = "rightward" if not (ro2 or rr2) else "leftward, units digit first"   # rightward = read left-to-right (tens digit first); leftward = right-to-left (units digit first)
        L.append(f"§{n} reading {rdir}:")
        # ── solution state: an ordered candidate-FAMILY set per operator (prior order; a sign-confirmed op starts {~sub}).
        #    Each step prunes/locks it; we log it after §n.1/§n.2/§n.3 like a running system state.
        FAM4 = ['noisy_mul', 'noisy_add', 'noisy_subtraction', 'concat']
        cand = {}
        for op in full_order:                                # full_order INCLUDES the (possibly unseen) QUERY operator
            if op in confirmed_sub:
                cand[op] = ['noisy_subtraction']
            else:
                pr = [f for f in op_fams_ordered(op) if f in FAM4]
                cand[op] = pr + [f for f in FAM4 if f not in pr]
        if confirmed_sub:                                    # distinctness (prior #2): a sign-confirmed ~sub locks ~sub
            for op in full_order:                            # for that operator, so no OTHER operator can be ~sub
                if op not in confirmed_sub:
                    cand[op] = [c for c in cand[op] if c != 'noisy_subtraction']
        ftag = lambda c: PRIOR_TAG.get(c, cname(c))          # family -> ~tag; a locked concat variant -> its concat_fwd/rev name
        def cell(op):                                        # candidate list (ordered, try-order) -> [..]; the unseen QUERY operator stays a bare `unknown` (not a list — nothing narrows it)
            if not seen and op == qop:
                return f"{label[op]} = unknown"
            return f"{label[op]} = [{', '.join(ftag(c) for c in cand[op])}]"
        state = lambda: ", ".join(cell(op) for op in full_order)

        def lock_concat(op):
            """Lock op against the concat family — the SAME check+confirm contract as the §n.4 arithmetic
            lock. Decide the direction from the first example that can tell fwd from rev (concat_fwd !=
            concat_rev); an EQUAL-operand example (fwd == rev) cannot decide, so it is deferred to the END
            and only CONFIRMED there (mirrors ~sub walking its variant order instead of committing to the
            first match). Then confirm the chosen direction on every other example. Returns
            (variant_or_None, [log lines]); a lock that fails confirmation returns None (the caller — the
            leading-0 gut-check or §n.2 — handles it: illegal, or 'not concat')."""
            exs = items[op]
            cf = lambda sa, sb, o: (rev_if(sa, ro2) + rev_if(sb, ro2), rev_if(sb, ro2) + rev_if(sa, ro2), target_of(o, rr2))
            deferred = []; decider = None; after = []        # classify first (no narration): the first example that distinguishes fwd/rev decides
            for e in exs:
                fwd, rev, tg = cf(*e[:3])
                if decider is None and fwd == tg and rev == tg: deferred.append(e)   # BOTH directions match output -> can't tell which
                elif decider is None:                          decider = e
                else:                                          after.append(e)
            lines = []
            if decider is None:                              # no example has distinct operands -> op is concat, but the direction is undecidable
                sa0, sb0, o0, i0 = exs[0]; fwd, rev, tg0 = cf(sa0, sb0, o0)
                more = "" if len(exs) == 1 else " (likewise " + ", ".join(f"EX{e[3]}" for e in exs[1:]) + ")"
                lines.append(f"    {label[op]}: concat_fwd={fwd}, concat_rev={rev}. Output={tg0}. Both match{more}; no example has distinct operands to fix the direction, so default to concat_fwd (the more frequent direction).")
                return 'concatenation', lines
            for (sa, sb, o, i) in deferred:                  # there IS a later decider, so "check the next" is honest
                fwd, rev, _ = cf(sa, sb, o)
                lines.append(f"    {label[op]}: EX{i} operands {rev_if(sa, ro2)}, {rev_if(sb, ro2)} are equal, so concat_fwd = concat_rev = {fwd} = output — this example can't decide the direction; check the next.")
            sa0, sb0, o0, i0 = decider
            fwd, rev, tg0 = cf(sa0, sb0, o0)
            head = f"    {label[op]}: concat_fwd={fwd}, concat_rev={rev}. Output={tg0}. "
            if fwd == tg0:   var = 'concatenation'
            elif rev == tg0: var = 'reverse concatenation'
            else:            return None, lines + [head + f"Neither matches, so {label[op]} is not concat."]
            lines.append(head + f"{'FWD' if var == 'concatenation' else 'REV'} matches -> so lock {label[op]} = {cname(var)}.")
            confirm = after + deferred                       # deferred equal-operand examples are confirmed last
            if confirm:
                checks = []; failed = False
                for (sa, sb, o, i) in confirm:
                    a, b = rev_if(sa, ro2), rev_if(sb, ro2); tg = target_of(o, rr2)
                    val = a + b if var == 'concatenation' else b + a
                    checks.append((i, cexpr(var, a, b), val, tg))
                    if val != tg: failed = True; break
                exl = ", ".join(f"EX{i}" for (i, _, _, _) in checks)
                body = "; ".join(f"EX{i}: {e} = {v}" + ("" if v == t else f", not {t}") for (i, e, v, t) in checks)
                if failed:
                    lines.append(f"    {label[op]} also appears in {exl}; check {body} — {cname(var)} fails.")
                    return None, lines
                lines.append(f"    {label[op]} also appears in {exl}; check {body}. Confirmed.")
            return var, lines
        # §n.1 ── write equations in THIS reading's frame, then log the state
        L.append(f"  §{n}.1 writing equations:")
        for i, e in enumerate(raw_exs, 1):
            p = parse_eq(e['input_value'])
            if not p: continue
            sa, op, sb = p
            O = e['output_value'].strip()
            ta, tb = rev_if(sa, ro2), rev_if(sb, ro2)
            disp = sign_in_place(O) if not rr2 else target_of(O, rr2)     # left-to-right keeps the sign in place; right-to-left reverses
            L.append(f"    EX{i}: {label[op]}({ta}, {tb}) = {disp}")
        L.append(f"    QUERY: {label[qop]}({rev_if(qsa, ro2)}, {rev_if(qsb, ro2)})")
        # §n.1 gut-check — TWO separate passes, cheapest first:
        #   (a) tail-sign pass over the examples: a number's sign comes first, so an RHS ending in a sign kills the reading (cheap — just the last char);
        #   (b) leading-0 pass over the scan list [all examples, QUERY]: any leading-0 number FORCES its operator to be concat (prior #3) -> lock_concat (check + confirm), else illegal.
        for i, e in enumerate(raw_exs, 1):                # (a) tail-sign pass — run first, it is the cheaper check
            p = parse_eq(e['input_value'])
            if not p: continue
            O = e['output_value'].strip()
            disp = sign_in_place(O) if not rr2 else target_of(O, rr2)
            if disp and not disp[-1].isdigit():
                L.append(f"    Note: EX{i} RHS {disp} ends in the sign '{disp[-1]}', but a number's sign comes first — illegal.")
                L.append("    This reading order is illegal; skip it completely.")
                return None
        concat_locked = {}                                # (b) leading-0 pass — scan list is [all examples, QUERY]
        scan = []                                         # build the scan list: every example (in order), then the QUERY
        for i, e in enumerate(raw_exs, 1):
            p = parse_eq(e['input_value'])
            if not p: continue
            sa, op, sb = p
            O = e['output_value'].strip()
            ta, tb = rev_if(sa, ro2), rev_if(sb, ro2)
            disp = sign_in_place(O) if not rr2 else target_of(O, rr2)
            scan.append((f"EX{i}", op, (ta, tb, re.sub(r'\D', '', disp))))
        if seen:                                          # an unseen QUERY op has no example -> concat direction is undecidable (left to unseen_leading_zero)
            scan.append(("QUERY", qop, (rev_if(qsa, ro2), rev_if(qsb, ro2))))
        for tag, op, nums in scan:                        # any leading-0 number -> the operator must be concat (prior #3)
            for num in nums:
                if len(num) > 1 and num[0] == '0':
                    if op in concat_locked: break         # this op was already shown concat by an earlier leading-0 in the scan
                    L.append(f"    Note: {tag} contains {num}, a number with a leading 0, so {label[op]} must be concat. Check {label[op]} right now:")
                    variant, lines = lock_concat(op)      # the SAME lock-and-confirm used everywhere: check the family + confirm on ALL the op's examples
                    L.extend(lines)
                    if variant is None:                   # forced concat, but it does not hold across the examples -> the reading is impossible
                        L.append("    This reading order is illegal; skip it completely.")
                        return None
                    concat_locked[op] = variant; cand[op] = [variant]
                    break
        if not seen:
            L.append(f"    {state()} since {label[qop]} only appears in QUERY.")
        else:
            L.append(f"    {state()}.")
        L.append("")
        # §n.2 ── concat check: lock a matching concat, drop ~concat from the rest; then log the state
        L.append(f"  §{n}.2 quick check to see whether any operator is concatenation:")
        concat_ops = dict(concat_locked)             # ops already locked concat by the §n.1 leading-0 gut-check
        for op in order:
            if op in concat_locked: continue         # the gut-check already locked this op concat -> don't re-check it here
            variant, lines = lock_concat(op)         # SAME lock-and-confirm helper: check the first example, confirm on the rest
            L.extend(lines)
            if variant is not None:
                concat_ops[op] = variant; cand[op] = [variant]
            else:
                cand[op] = [c for c in cand[op] if c != 'concat']
        arith = [op for op in order if op not in concat_ops]
        variants = {op: concat_ops[op] for op in concat_ops}
        if not arith:
            L.append(f"    Every operator is concatenation; nothing remains to search. {state()}")
            return concat_ops, arith, variants
        L.append(f"    All {len(arith)} operator(s) remaining are arithmetic. {state()}")
        L.append("")
        # §n.3 ── digit-count prune: drop families the RHS digit-count rules out (+ distinctness vs HARD-pinned); log state
        L.append(f"  §{n}.3 prune solution trees by the RHS digit-count before entering:")

        def _surv(op):                               # (digit-count survivors as a set, the clause string)
            by_d = {}
            for sa, sb, o, i in items[op]:
                by_d.setdefault(len(target_of(o, rr2).lstrip('-')), []).append(i)
            s = None
            for dd in by_d:
                s = set(_surv_for(dd)) if s is None else s & set(_surv_for(dd))
            s = s or set(ARITH_FAMS)
            cl = "; ".join(f"{' and '.join('EX' + str(i) for i in by_d[dd])} {'has' if len(by_d[dd]) == 1 else 'have'} "
                           f"{dd}-digit RHS ({_rules_out(dd)})" for dd in sorted(by_d))
            return s, cl

        survc = {}; clause = {}
        for op in arith:
            s, cl = _surv(op)
            survc[op] = [f for f in cand[op] if f in s]   # digit-count survivors, kept in prior order, within current cand
            clause[op] = cl
        hard_taken = set()   # families pinned by sign or a forced digit-count/distinctness deduction
        oidx = {op: i for i, op in enumerate(order)}
        availh = lambda o: [f for f in survc[o] if f not in hard_taken]
        remaining = list(arith)
        while remaining:
            sub_ops = [o for o in remaining if o in confirmed_sub]
            dc_ops  = [o for o in remaining if o not in confirmed_sub and len(survc[o]) == 1]
            di_ops  = [o for o in remaining if o not in confirmed_sub and len(survc[o]) > 1 and len(availh(o)) == 1]
            if sub_ops:                              # sign already pinned it -> HARD
                op = min(sub_ops, key=lambda o: oidx[o]); fam = 'noisy_subtraction'; cand[op] = ['noisy_subtraction']
                L.append(f"    {label[op]}: already shown to be ~sub by its output sign.")
            elif dc_ops:                             # digit-count alone forces it -> HARD
                op = min(dc_ops, key=lambda o: oidx[o]); fam = survc[op][0]; cand[op] = [fam]
                L.append(f"    {label[op]}: {clause[op]}.")
                L.append(f"      Only {PRIOR_TAG[fam]} survives the digit-count filter. Try {label[op]} = {PRIOR_TAG[fam]}.")
            elif di_ops:                             # distinctness against HARD-pinned families forces it -> HARD
                op = min(di_ops, key=lambda o: oidx[o]); fam = availh(op)[0]; cand[op] = [fam]
                L.append(f"    {label[op]}: {clause[op]}.")
                L.append(f"      Surviving candidates: {', '.join(PRIOR_TAG[f] for f in survc[op])}. "
                         f"By the distinct operator rule {label[op]} must be the remaining operation, {PRIOR_TAG[fam]}.")
            else:                                    # nothing forced -> SOFT: announce this sign's OWN most-frequent survivor. distinctness acts at LOCK (§n.4), NOT on the try-order — another sign's UNCONFIRMED try must not delete a family here.
                op = min(remaining, key=lambda o: (len(availh(o)), oidx[o])); cand[op] = list(survc[op])
                fam = survc[op][0]
                L.append(f"    {label[op]}: {clause[op]}.")
                L.append(f"      Surviving candidates: {', '.join(PRIOR_TAG[f] for f in survc[op])}. "
                         f"Try {label[op]} = {PRIOR_TAG[fam]} first ({PRIOR_TAG[fam]} appears more frequently).")
            if op in confirmed_sub or op in dc_ops or op in di_ops: hard_taken.add(fam)
            remaining.remove(op)
        L.append(f"    No further pruning found. We have {state()} to traverse in order.")
        L.append("")
        # §n.4 ── backtracking search: try the candidate combo; lock the ops that fit; advance ONLY the
        #    ops that miss to their next candidate family (skipping families already locked -> distinct), and
        #    re-print the combo; repeat until everything locks or an op exhausts its candidates (-> unknown).
        L.append(f"  §{n}.4 enter the tree and search for an operator assignment:")
        locked = {}                                  # op -> chosen exact variant
        idx = {op: 0 for op in arith}                # current index into cand[op]
        dead = set()                                 # ops that exhausted their candidates -> unknown
        while True:
            asg = []
            for op in order:
                if op in concat_ops:  asg.append(f"{label[op]} = {cname(variants[op])}")
                elif op in locked:    asg.append(f"{label[op]} = {cname(locked[op])}")
                elif op in dead:      asg.append(f"{label[op]} = unknown")
                else:                 asg.append(f"{label[op]} = {FAMNAME_TAG[cand[op][idx[op]]]}")
            if not seen: asg.append(f"{label[qop]} = unknown")
            L.append(f"    Try operators {', '.join(asg)}:")
            missed = []
            for op in arith:
                if op in locked or op in dead: continue
                fam = cand[op][idx[op]]; tag = FAMNAME_TAG[fam]
                sa0, sb0, o0, i0 = items[op][0]
                ta0, tb0 = rev_if(sa0, ro2), rev_if(sb0, ro2); tg0 = target_of(o0, rr2)
                L.append(f"      We now know all the operands in EX{i0}: {ta0} {tag} {tb0}, RHS = {tg0}.")
                famfit = [v for v in consistent_ops(trip[op], ro2, rr2) if fam_of(v) == fam]
                if famfit:                           # this family fits -> lock its simplest fitting variant
                    chosen = famfit[0]
                    # honest in-order search: a variant EARLIER in the family order that also matches the ENTRY
                    # example but fails a later one is shown locked-then-ruled-out before we settle on `chosen`.
                    # (Variants that don't even match the entry example are skipped, exactly like mul/mul+1.)
                    for v in fam_variant_order(fam):
                        if v == chosen: break
                        if raw_on(sa0, sb0, v, ro2) != tg0: continue
                        L.append(f"        {tg0} = {cexpr(v, ta0, tb0)}, so lock {label[op]} = {cname(v)}.")
                        for (sa, sb, o, i) in items[op][1:]:
                            r = raw_on(sa, sb, v, ro2); tg = target_of(o, rr2)
                            if r != tg:
                                try: gap = abs(int(r) - int(tg))
                                except (ValueError, TypeError): gap = None
                                miss = f", far from {tg}" if (gap is not None and gap > 2) else f", not {tg}"
                                L.append(f"        {label[op]} also appears in EX{i}; check EX{i}: {cexpr(v, rev_if(sa, ro2), rev_if(sb, ro2))} = {r}{miss} — {cname(v)} fails.")
                                break
                    L.append(f"        {tg0} = {cexpr(chosen, ta0, tb0)}, so lock {label[op]} = {cname(chosen)}.")
                    rest = items[op][1:]
                    if rest:
                        exl = ", ".join(f"EX{i}" for (_, _, _, i) in rest)
                        vs = "; ".join(f"EX{i}: {cexpr(chosen, rev_if(sa, ro2), rev_if(sb, ro2))} = {raw_on(sa, sb, chosen, ro2)}"
                                       for (sa, sb, o, i) in rest)
                        L.append(f"        {label[op]} also appears in {exl}; check {vs}. Confirmed.")
                    locked[op] = chosen
                else:                                # this family misses -> replay the greedy lock-and-verify honestly
                    v0 = first_variant_match(sa0, sb0, o0, fam, ro2, rr2)
                    if v0 is None:                   # the FIRST example is already out of band for the whole family
                        for v in FAM_REPS.get(fam, []):
                            L.append(f"        {v}: {cexpr(v, ta0, tb0)} = {raw_on(sa0, sb0, v, ro2)}, far from {tg0} — no.")
                        L.append(f"        no {tag} variant gives {tg0}, so {label[op]} = {tag} fails.")
                    else:                            # EX{i0} forces v0; lock it, then verification breaks on a later example
                        L.append(f"        {tg0} = {cexpr(v0, ta0, tb0)}, so lock {label[op]} = {cname(v0)}.")
                        passes = []; fail = None
                        for (sa, sb, o, i) in items[op][1:]:
                            r = raw_on(sa, sb, v0, ro2); tg = target_of(o, rr2)
                            if r == tg: passes.append((i, cexpr(v0, rev_if(sa, ro2), rev_if(sb, ro2)), r))
                            else: fail = (i, sa, sb, o, r, tg); break
                        fi, fsa, fsb, fo, fr, ftg = fail
                        fexpr = cexpr(v0, rev_if(fsa, ro2), rev_if(fsb, ro2))
                        try: gap = abs(int(fr) - int(ftg))
                        except (ValueError, TypeError): gap = None
                        far = f", far from {ftg}" if (gap is not None and gap > 2) else ""
                        w = first_variant_match(fsa, fsb, fo, fam, ro2, rr2)
                        exl = ", ".join([f"EX{i}" for (i, _, _) in passes] + [f"EX{fi}"])
                        parts = [f"EX{i}: {e} = {r}" for (i, e, r) in passes] + [f"EX{fi}: {fexpr} = {fr}{far}"]
                        L.append(f"        {label[op]} also appears in {exl}; check {'; '.join(parts)}.")
                        if w is None:                # EX{fi} is out of band for the whole family — identical verdict to the v0-None branch
                            L.append(f"        no {tag} variant gives {ftg}, so {label[op]} = {tag} fails.")
                        else:                        # EX{fi} needs a different variant — one operation can't be both
                            wexpr = cexpr(w, rev_if(fsa, ro2), rev_if(fsb, ro2))
                            L.append(f"        {ftg} = {wexpr} needs {cname(w)}, not {cname(v0)}, so {label[op]} = {tag} fails.")
                    missed.append(op)
            if not missed:
                break
            lockedfams = {fam_of(locked[o]) for o in locked}
            adv = []
            for op in missed:                        # advance to the next candidate family not already locked elsewhere
                ni = idx[op] + 1
                while ni < len(cand[op]) and cand[op][ni] in lockedfams:
                    ni += 1
                if ni < len(cand[op]):
                    idx[op] = ni; adv.append(op)
                else:
                    dead.add(op)
            gone = [op for op in missed if op in dead]
            tail = ""
            if gone:
                tail = (" (" + ", ".join(label[op] for op in gone) + (" has" if len(gone) == 1 else " have") +
                        " no candidate family left, so " + ("it remains" if len(gone) == 1 else "they remain") + " unknown)")
            if adv:
                L.append("    Advance the unsolved operator(s) to the next candidate: " +
                         ", ".join(f"{label[op]} -> {FAMNAME_TAG[cand[op][idx[op]]]}" for op in adv) + tail + ".")
            else:
                L.append(f"    No candidate family is left for {', '.join(label[op] for op in gone)}; "
                         f"{'it remains' if len(gone) == 1 else 'they remain'} unknown.")
                break
        for op in arith:
            variants[op] = locked.get(op)            # locked exact variant, or None (unknown) if the op died
        return concat_ops, arith, variants

    # ── forward-only control: §1 = the reading the sign points to (trailing sign -> right-to-left, else
    #    left-to-right); §2 = the other, opened ONLY if §1 leaves an operator unresolved; keep whichever
    #    resolves MORE (§1 wins ties). Then a one-line Summary, then the query answer.
    rname = lambda m: "rightward" if not (m[0] or m[1]) else "leftward"
    start = (False, False)               # always check left-to-right first (prior #4); the §n.1 gut-check / search discards a wrong reading
    other = (True, True)
    Y = len(full_order)                  # the unseen query operator counts toward the total (it stays unknown until guessed)
    dctstr = lambda vmap: ", ".join(f"{label[op]} = {cname(vmap[op]) if vmap.get(op) else 'unknown'}" for op in full_order)
    nres = lambda vmap: len([op for op in full_order if vmap.get(op)])

    L.append("Now start to traverse the solution tree, reading order = [rightward, leftward]:")
    legal_where = "either §1 or §2"          # which reading(s) actually searched the query op; narrowed below when a reading is ruled illegal (drives the exotic verdict)
    res1 = attempt(start, 1)
    L.append("")
    if res1 is None:                         # §1's equations are malformed -> §1 is illegal; the other reading is the only legal one
        legal_where = "§2"
        res2 = attempt(other, 2)
        L.append("")
        concat_ops, arith, variants = res2 if res2 is not None else ({}, list(order), {op: None for op in order})
        mode = other; n2 = nres(variants)
        L.append(f"  Conclusion of step §2: {{{dctstr(variants)}}}, {n2} of {Y} resolved.")
        L.append("")
        L.append(f"Summary: §1 is illegal, so §2 is the only legal reading. "
                 f"Confirm reading order = {rname(other)}, {dctstr(variants)}")
    else:
        concat_ops, arith, variants = res1
        mode = start
        n1 = nres(variants)
        if n1 == Y:                          # §1 resolved EVERY operator (incl. the query op) -> stop, no §2
            L.append(f"  Conclusion of step §1: {{{dctstr(variants)}}}, {n1} of {Y} resolved. This is the answer.")
            L.append("")
            L.append(f"Summary: Confirm reading order = {rname(start)}, {dctstr(variants)}")
        else:                                # anything unresolved (a stuck example op, OR the unseen query op) -> try the other reading
            L.append(f"  Conclusion of step §1: {{{dctstr(variants)}}}, {n1} of {Y} resolved. Need to try reading {rname(other)}.")
            L.append("")
            res2 = attempt(other, 2)
            L.append("")
            if res2 is None:                 # §2 illegal -> keep §1
                legal_where = "§1"
                L.append(f"Summary: §2 is illegal, so §1 is the only legal reading. "
                         f"Confirm reading order = {rname(start)}, {dctstr(variants)}")
            else:                            # both legal -> the reading that resolves MORE operators wins (§1 on a tie)
                c2, a2, v2 = res2; n2 = nres(v2)
                L.append(f"  Conclusion of step §2: {{{dctstr(v2)}}}, {n2} of {Y} resolved.")
                L.append("")
                if n2 > n1:
                    verdict = "§2 solved more operators than §1, so §2 is preferred."
                    mode = other; concat_ops, arith, variants = c2, a2, v2
                elif n1 > n2:
                    verdict = "§1 solved more operators than §2, so §1 is preferred."
                else:
                    verdict = f"§1 and §2 both solved {n1} operator{'' if n1 == 1 else 's'}, §1 is preferred."
                L.append(f"Summary: {verdict} Confirm reading order = {rname(mode)}, {dctstr(variants)}")
    ro, rr = mode
    L.append("")

    # query answer: pick up the QUERY in the confirmed reading frame, apply the resolved/guessed operator
    qta, qtb = rev_if(qsa, ro), rev_if(qsb, ro)

    def emit_answer(expr, raw, final, ans):
        L.append(f"Now solve the QUERY {label[qop]}({qta}, {qtb}):")
        if rr:                               # solved in the reverse frame -> the written answer is the result with its digits reversed
            L.append(f"{expr} = {raw}; reverse the digits {raw} -> {final}; answer = {final}")
        else:
            L.append(f"{expr} = {raw}; answer = {final}")

    if seen and variants.get(qop) in ('concatenation', 'reverse concatenation'):
        hk = variants[qop]
        raw = raw_on(qsa, qsb, hk, ro); final = _rev(raw) if rr else raw
        ans = render_answer(final, fmt, qop)
        emit_answer(cexpr(hk, qta, qtb), raw, final, ans)
    elif seen and variants.get(qop):
        chosen = variants[qop]
        raw = raw_on(qsa, qsb, chosen, ro); final = _rev(raw) if rr else raw
        ans = render_answer(final, fmt, qop)
        emit_answer(cexpr(chosen, qta, qtb), raw, final, ans)
    elif seen:                                           # EXOTIC: query op unsolvable by the four families -> the one out-of-system operation is max-mod-min
        A, B = int(qta), int(qtb)                        # query operands already in the confirmed reading frame
        mx, mn = max(A, B), min(A, B)
        raw = str(mx % mn) if mn else str(mx)
        final = _rev(raw) if rr else raw
        ans = render_answer(final, fmt, qop)
        L.append(f"QUERY operator {label[qop]} is not solvable in {legal_where}, so {label[qop]} is an exotic operator.")
        L.append("The one operation that exists outside the system is max-mod-min(a,b) = max(a, b) mod min(a, b)")
        L.append(f"Now solve the QUERY {label[qop]}({qta}, {qtb}):")
        if rr:
            L.append(f"max({qta}, {qtb}) mod min({qta}, {qtb}) = {mx} mod {mn} = {raw}; reverse the digits {raw} -> {final}; answer = {final}")
        else:
            L.append(f"max({qta}, {qtb}) mod min({qta}, {qtb}) = {mx} mod {mn} = {raw}; answer = {final}")
    else:                                                # UNSEEN: query op never appears -> prior-based guess (binary rule)
        L.append(f"Query operator {label[qop]} is not determined by the examples, so I need to guess it.")
        locked_fams = [fam_of(variants[op]) for op in arith if variants.get(op)]
        used_fams = {f for f in locked_fams if f in ARITH_FAMS}
        if not used_fams:
            # NO operator solved arithmetically -> the operation is concatenation (guess the ~concat family)
            hk = 'concatenation'
            L.append(f"No operator here can be solved arithmetically, so I guess the unseen operator "
                     f"{label[qop]} is ~concat. Let me use {label[qop]} = {cname(hk)}.")
            raw = raw_on(qsa, qsb, hk, ro); final = _rev(raw) if rr else raw
            ans = render_answer(final, fmt, qop)
            emit_answer(cexpr(hk, qta, qtb), raw, final, ans)
        else:                                            # guess an ARITHMETIC operator (distinctness + symbol prior)
            present = []
            for f in locked_fams:
                if f in ARITH_FAMS and f not in present: present.append(f)
            ford = []
            for f in [op_top_family(qop)] + op_seen_arith(qop) + ARITH_FAMS:
                if f in ARITH_FAMS and f not in ford: ford.append(f)
            free = [f for f in ford if f not in used_fams]
            chosen_fam = (free or ford)[0]
            base = BASEOP[chosen_fam]
            tg = lambda f: FAMNAME_TAG.get(f, f)         # noisy_* -> ~mul/~add/~sub
            prior = f"By frequency order, {tg(chosen_fam)} is the pick"
            intro = f"At least one operator here can be solved arithmetically, so I'll guess {label[qop]} is an arithmetic operator too"
            pres = " and ".join(tg(f) for f in present); hv = "has" if len(present) == 1 else "have"
            if len(free) == 1:
                L.append(f"{intro}. Since {pres} {hv} appeared in the puzzle, by the distinct operator rule "
                         f"{label[qop]} must be the remaining operation, {tg(chosen_fam)}.")
            elif len(free) >= 2:
                L.append(f"{intro}. Since {pres} {hv} appeared in the puzzle, by the distinct operator rule "
                         f"{label[qop]} is {' or '.join(tg(f) for f in free)}; {prior}.")
            else:
                L.append(f"{intro}. Since {pres} {hv} appeared in the puzzle, every arithmetic operation is "
                         f"already used, so I take the most likely, {tg(chosen_fam)}.")
            raw = raw_on(qsa, qsb, base, ro); final = _rev(raw) if rr else raw
            ans = render_answer(final, fmt, qop)
            emit_answer(cexpr(base, qta, qtb), raw, final, ans)

    if ans != final:
        kind = {'suffix': 'operator-symbol suffix', 'prefix': 'operator-symbol prefix',
                'minus': 'leading minus', 'none': 'leading minus'}[fmt]
        L.append(f"The result is negative, so it is written with the {kind}: {final} -> {ans}.")
    L.append("")
    L.append("I will now return the answer in \\boxed{}, the answer is")
    L.append(f"\\boxed{{{ans}}}")
    return "\n".join(L), ans


# ── official Kaggle metric (for GT-match reporting) ─────────────────────────────
def extract_final_answer(text):
    bs = list(re.finditer(r'\\boxed\{', text)); ms = []
    for i, m in enumerate(bs):
        end = bs[i + 1].start() if i + 1 < len(bs) else len(text)
        seg = text[m.end():end]; lb = seg.rfind('}')
        ms.append(seg[:lb] if lb != -1 else seg)
    ne = [m.strip() for m in ms if m.strip()]
    return ne[-1] if ne else (ms[-1].strip() if ms else 'NOT_FOUND')


def verify(stored, pred):
    stored = stored.strip(); pred = pred.strip()
    if re.fullmatch(r'[01]+', stored): return pred.lower() == stored.lower()
    try: return math.isclose(float(stored), float(pred), rel_tol=1e-2, abs_tol=1e-5)
    except Exception: return pred.lower() == stored.lower()


def classify(d):
    """OUR problem-type label for a puzzle, derived from its structure — a map of WHAT KIND of
    problem it is, for our own understanding (there is no ground-truth category; all we truly have
    per puzzle is question.txt + answer.txt). Determined by what decides solvability:
      unseen_operator        – query operator never appears in the examples (must be guessed)
      unseen_leading_zero    – unseen query operator whose operand has a leading 0 in the chosen reading
                               (prior #3 forces ~concat, but concat direction is undecidable; relabeled, not scored out)
      exotic_operation       – query operator needs a rule outside our cryptarithm vocabulary
      no_rule_fits           – no operation (even the full op set) explains the examples
      concatenation          – query operator is concatenation
      ambiguous_arithmetic   – >1 common operation fits the examples but they disagree on the query
      arithmetic_left_to_right / arithmetic_right_to_left – the fitting operations agree on the query, so
                               the answer IS determined (the unambiguous arithmetic case)."""
    by = collections.defaultdict(list)
    for e in d['examples']:
        p = parse_eq(e['input_value'])
        if not p: continue
        sa, op, sb = p
        by[op].append((sa, sb, e['output_value'].strip()))
    q = parse_eq(d['question'])
    if not q or not by:
        return 'no_rule_fits'
    qsa, qop, qsb = q
    ro, rr = choose_mode(by)
    if qop not in by:
        qa = _rev(qsa) if rr else qsa
        qb = _rev(qsb) if rr else qsb
        if (len(qa) > 1 and qa[0] == '0') or (len(qb) > 1 and qb[0] == '0'):
            return 'unseen_leading_zero'   # leading-0 query operand + no example of this symbol => concat by prior #3 but direction undecidable
        return 'unseen_operator'
    ourfit = consistent_ops(by[qop], ro, rr)
    if not ourfit:
        allfit = [nm for nm in ALLNAMES if all(fits(sa, sb, nm, ro, rr, o) for sa, sb, o in by[qop])]
        return 'exotic_operation' if allfit else 'no_rule_fits'
    ch = ourfit[0]
    if 'concat' in ch:
        return 'concatenation'

    def qf(nm):
        r = raw_on(qsa, qsb, nm, ro)
        return _rev(r) if rr else r
    if any(nm in COMMON and nm != ch and qf(nm) != qf(ch) for nm in ourfit):
        return 'ambiguous_arithmetic'
    return 'arithmetic_right_to_left' if rr else 'arithmetic_left_to_right'


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None    # regenerate a single puzzle id
    recs = {}
    for path in (USED_CSV, UNUSED_CSV):
        for r in csv.DictReader(open(path, newline='')):
            if not str(r.get('category', '')).startswith('equation_numeric'):
                continue
            if only and r['id'] != only:
                continue
            ex, qn = parse_prompt(r['prompt'])
            recs[r['id']] = {'category': r['category'], 'prompt': r['prompt'], 'answer': r['answer'],
                             'huikang_cot': r.get('solver_cot', ''), 'examples': ex, 'question': qn}
    n = match = huik = 0; by_cat = collections.Counter(); by_cat_ok = collections.Counter()
    for pid, d in recs.items():
        cat = classify(d)
        cot, ans = gen_cot(d)
        ok = verify(d['answer'], extract_final_answer(cot))
        hk_ok = verify(d['answer'], extract_final_answer(d['huikang_cot'])) if d['huikang_cot'].strip() else False
        n += 1; match += ok; huik += hk_ok; by_cat[cat] += 1; by_cat_ok[cat] += ok
        folder = os.path.join(OUT_DIR, f"equation_numeric_{pid}")
        os.makedirs(os.path.join(folder, 'track'), exist_ok=True)
        open(os.path.join(folder, 'question.txt'), 'w').write(d['prompt'])
        open(os.path.join(folder, 'answer.txt'), 'w').write(d['answer'])
        open(os.path.join(folder, 'label.txt'), 'w').write(cat + "\n")
        open(os.path.join(folder, 'cot_huikang.txt'), 'w').write(d['huikang_cot'])
        open(os.path.join(folder, 'track', 'tree_cot.txt'), 'w').write(cot)
        if only:
            print(f"regenerated equation_numeric_{pid}  (label={cat})")
            print(f"  our answer = {extract_final_answer(cot)} | gold = {d['answer']} | {'OK' if ok else 'MISS'}")
    if only:
        return
    print(f"wrote {n} puzzle folders under {OUT_DIR}")
    print(f"OUR GT-match (official metric):     {match}/{n} = {100*match/n:.1f}%")
    print("by OUR problem-type label (label.txt):")
    for c in sorted(by_cat, key=lambda c: -by_cat[c]):
        print(f"   {c:32} {by_cat_ok[c]:>4}/{by_cat[c]:<4} ({100*by_cat_ok[c]//by_cat[c]}%)")
    print(f"huikang GT-match (his cot_huikang): {huik}/{n} = {100*huik/n:.1f}%")


if __name__ == '__main__':
    main()
