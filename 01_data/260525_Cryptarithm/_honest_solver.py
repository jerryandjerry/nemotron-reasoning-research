#!/usr/bin/env python3
"""Honest step-by-step cryptarithm solver / CoT generator.

Every line is one deduction justified only by facts already established:
  - a variable is narrowed by ONE equation given the others' CURRENT ranges (interval form),
  - pinned operands are substituted into the expressions, '= v' written on collapse,
  - all-different removals cite the symbol that took the digit (one condensed line),
  - once an equation's operands are known, the operator's MEANING is determined by trying each
    candidate variant (47×47=2209 ⟹ mul), then that operator is used exactly afterwards,
  - dead-ends show the equation that empties a variable, then backtrack,
  - branching only at genuine choice points (the least-unknown equation, smallest domain).
Reuses cot_generator for the setup preamble and _gen_solutions for the answer.
gen(name) -> (full_cot, A:symbol->digit | None, variants, mode)."""
import copy
from itertools import product as iproduct
import cot_generator as cg
import _gen_solutions as gs

CORE = {'noisy_mul': 'mul', 'noisy_add': 'add', 'noisy_subtraction': 'noisy_subtraction'}

# The 23-symbol digit alphabet (the full pool minus the operator-only symbols + - *), ASCII order.
# A digit that never appears in the examples has a symbol we never saw; it is a uniform random
# pick from this pool minus the symbols already used and minus this puzzle's operator symbols
# (the unseen digit-symbol is never an operator symbol). The mapping is randomized per puzzle and
# cannot be derived, so we guess deterministically (first unused, non-operator symbol) to stay in
# the game rather than emit '_' (a guaranteed-wrong answer).
POOL23 = sorted("""!"#$%&'()/:<>?@[\\]^`{|}""")

def vname(v):
    """Readable variant name for display: spell out the offset suffix (add_m1 -> add_minus1)."""
    return (v.replace('_m1', '_minus1').replace('_p1', '_plus1')
             .replace('_m2', '_minus2').replace('_p2', '_plus2'))

def fam_interval(fam, o1, o2):
    if fam == 'noisy_mul': return (o1[0]*o2[0]-1, o1[1]*o2[1]+1)
    if fam == 'noisy_add': return (o1[0]+o2[0]-1, o1[1]+o2[1]+2)
    m = max(o1[1]-o2[0], o2[1]-o1[0], 0); return (-m-2, m+1)

def exact_interval(var, o1, o2):
    a1, a2 = o1; b1, b2 = o2
    if var in ('mul', 'mul_p1', 'mul_m1'):
        return (a1*b1 + {'mul':0,'mul_p1':1,'mul_m1':-1}[var], a2*b2 + {'mul':0,'mul_p1':1,'mul_m1':-1}[var])
    if var in ('add', 'add_p1', 'add_m1', 'add_p2'):
        off = {'add':0,'add_p1':1,'add_m1':-1,'add_p2':2}[var]; return (a1+b1+off, a2+b2+off)
    adlo = max(0, a1-b2, b1-a2); adhi = max(a2-b1, b2-a1, 0)
    if var == 'absdiff': return (adlo, adhi)
    if var == 'absdiff_p1': return (adlo+1, adhi+1)
    if var == 'absdiff_m1': return (adlo-1, adhi-1)
    if var == 'absdiff_m2': return (adlo-2, adhi-2)
    if var == 'neg_absdiff': return (-adhi, -adlo)
    if var == 'sub_signed': return (a1-b2, a2-b1)
    if var == 'rsub_signed': return (b1-a2, b2-a1)
    raise ValueError(var)

def _set(s): return ','.join(map(str, sorted(s)))

class Engine:
    def __init__(self, p, mode, op_set, fam):
        self.p, self.mode, self.op_set, self.fam = p, mode, op_set, fam
        self.lm = p.letter_map; self.syms = p.digit_syms
        self.lead = cg.get_leading(p, op_set, mode)

    def opval(self, d1, d2, A): return 10*A[d1]+A[d2] if self.mode == 'standard' else 10*A[d2]+A[d1]
    def operand_iv(self, d1, d2, dom):
        a, b = dom[d1], dom[d2]
        return (10*min(a)+min(b), 10*max(a)+max(b)) if self.mode == 'standard' else (10*min(b)+min(a), 10*max(b)+max(a))
    def res_iv(self, e, dom, locked):
        o1, o2 = self.operand_iv(e.d1, e.d2, dom), self.operand_iv(e.d3, e.d4, dom)
        return exact_interval(locked[e.op], o1, o2) if e.op in locked else fam_interval(self.fam[e.op], o1, o2)
    def rhs_iv(self, rhs, dom):
        sign = 1; ch = list(rhs)
        if ch and ch[0] in self.op_set: sign = -1; ch = ch[1:]
        elif ch and ch[-1] in self.op_set: sign = -1; ch = ch[:-1]
        n = len(ch); co = [10**(n-1-i) for i in range(n)] if self.mode == 'standard' else [10**i for i in range(n)]
        lo = hi = 0
        for c, s in zip(co, ch): lo += c*min(dom[s]); hi += c*max(dom[s])
        return (-hi, -lo) if sign < 0 else (lo, hi)
    def opexpr(self, d1, d2, dom):
        t, u = (d1, d2) if self.mode == 'standard' else (d2, d1)
        tp = next(iter(dom[t])) if len(dom[t]) == 1 else None
        up = next(iter(dom[u])) if len(dom[u]) == 1 else None
        if tp is not None and up is not None: return str(10*tp + up)
        if t == u and tp is None: return f"11{self.lm[t]}"
        tens = str(10*tp) if tp is not None else f"10{self.lm[t]}"
        return f"({tens}+{str(up) if up is not None else self.lm[u]})"
    def rhs_expr(self, rhs, dom):
        """RHS as its digit-symbols (pinned ones substituted), so every RHS variable is visible."""
        ch = list(rhs); sign = ''
        if ch and ch[0] in self.op_set: sign = '−'; ch = ch[1:]
        elif ch and ch[-1] in self.op_set: sign = '−'; ch = ch[:-1]
        body = ''.join(str(next(iter(dom[c]))) if len(dom[c]) == 1 else self.lm[c] for c in ch)
        return sign + body
    def pin(self, sym, dom):
        return f"{self.lm[sym]} = {next(iter(dom[sym]))}" if len(dom[sym]) == 1 else f"{self.lm[sym]} in {{{_set(dom[sym])}}}"

    def narrow(self, sym, e, dom, locked):
        keep = set()
        for v in dom[sym]:
            d2 = {**dom, sym: {v}}
            r = self.res_iv(e, d2, locked); rh = self.rhs_iv(e.rhs, d2)
            if not (r[1] < rh[0] or rh[1] < r[0]): keep.add(v)
        return keep

    def _arith(self, v, a, b):
        return cg.op_str(v, a, b).rsplit('=', 1)[0]                # "11×17" out of op_str's "11×17=187"

    def _op_cands(self, e, dom, locked):
        """For an example whose 4 operand symbols are all pinned: return
        (a, b, rstr, opc, zl, keep, excluded, used) for variant determination.
        keep = variants consistent with the RHS that aren't already another operator's meaning."""
        A = {s: next(iter(dom[s])) for s in self.syms if len(dom[s]) == 1}
        a, b = self.opval(e.d1, e.d2, A), self.opval(e.d3, e.d4, A)
        rh = self.rhs_iv(e.rhs, dom)
        opc = '~mul' if 'mul' in self.fam[e.op] else ('~add' if 'add' in self.fam[e.op] else '~sub')
        zl = self.p.op_map[e.op]
        rstr = f"RHS = {rh[0]}" if rh[0] == rh[1] else f"RHS in [{rh[0]},{rh[1]}]"
        used = {var: op for op, var in locked.items()}            # variant -> the operator already using it
        cands = [(v, cg.apply_op(v, a, b)) for v in cg.CORES[CORE[self.fam[e.op]]]]
        cands = [(v, rv) for v, rv in cands if rh[0] <= rv <= rh[1]]
        excluded = [(v, rv) for v, rv in cands if v in used]      # operators are distinct: drop a reused meaning
        keep = [(v, rv) for v, rv in cands if v not in used]
        return a, b, rstr, opc, zl, keep, excluded, used

    def propagate(self, dom, locked, emit):
        changed = True
        while changed:
            changed = False
            singles = {}                                          # digit -> the symbol pinned to it
            for t in self.syms:
                if len(dom[t]) == 1:
                    dv = next(iter(dom[t]))
                    if dv in singles:                             # two symbols forced to the same digit -> all-different violated
                        emit(f"{self.lm[singles[dv]]} and {self.lm[t]} are both forced to {dv}, but every symbol is a distinct digit — backtrack."); return False
                    singles[dv] = t
            for s in self.syms:
                if len(dom[s]) == 1: continue
                taken = sorted(v for v in dom[s] if v in singles)
                if not taken: continue
                dom[s] = dom[s] - set(taken); changed = True       # silent unless it forces a pin / wipeout
                lst = ", ".join(f"{v} is taken by {self.lm[singles[v]]}" for v in taken)
                if not dom[s]:
                    emit(f"{lst}, since all values are unique, {self.lm[s]} in {{}} — backtrack."); return False
                if len(dom[s]) == 1:
                    emit(f"{lst}, since all values are unique, {self.lm[s]} = {next(iter(dom[s]))}.")
            for ei, e in enumerate(self.p.examples, 1):
                if e.op not in self.fam: continue
                esyms = list(dict.fromkeys((e.d1, e.d2, e.d3, e.d4, *[c for c in e.rhs if c not in self.op_set])))
                # narrow EVERY variable of this equation from the SAME current snapshot, then report once
                changes = [(s, self.narrow(s, e, dom, locked)) for s in esyms if len(dom[s]) > 1]
                changes = [(s, nd) for s, nd in changes if nd != dom[s]]
                if changes:
                    o1, o2 = self.opexpr(e.d1, e.d2, dom), self.opexpr(e.d3, e.d4, dom)
                    if e.op in locked:
                        opc = vname(locked[e.op])                                       # once locked, show the exact variant (mul, mul_plus1, add_minus1, ...): narrowing now uses it exactly
                    else:
                        fm = self.fam[e.op]
                        opc = '~mul' if 'mul' in fm else ('~add' if 'add' in fm else '~sub')   # still noisy: ~ marks the family, distinct from real ×/+ and the + inside operands
                    r = self.res_iv(e, dom, locked); rh = self.rhs_iv(e.rhs, dom)
                    lead = f"EX{ei}: LHS {o1} {opc} {o2} in [{r[0]},{r[1]}], RHS {self.rhs_expr(e.rhs, dom)} in [{rh[0]},{rh[1]}] -> "
                    empty = [s for s, nd in changes if not nd]
                    if empty:
                        concl = ("LHS can never equal RHS — backtrack" if (r[1] < rh[0] or rh[1] < r[0])
                                 else f"no value of {self.lm[empty[0]]} makes LHS = RHS — backtrack")
                        emit(lead + concl); return False
                    parts = [f"{self.lm[s]} = {next(iter(nd))}" if len(nd) == 1 else f"{self.lm[s]} in {{{_set(nd)}}}" for s, nd in changes]
                    for s, nd in changes: dom[s] = nd
                    changed = True
                    emit(lead + ", ".join(parts))
                # forced operator meaning: the moment THIS example's operands+RHS are all known and leave
                # exactly one consistent variant, lock it right here (eager, in every branch)
                if e.op not in locked and all(len(dom[s]) == 1 for s in (e.d1, e.d2, e.d3, e.d4)):
                    a, b, rstr, opc, zl, keep, excluded, used = self._op_cands(e, dom, locked)
                    if len(keep) <= 1:
                        emit(f"We now know all the operands in EX{ei}: {a} {opc} {b}, {rstr}.")
                        for v, rv in excluded:
                            emit(f"  {rv} = {self._arith(v, a, b)} would make {zl} = {vname(v)}, but {vname(v)} is already operator {self.p.op_map[used[v]]}'s meaning; operators are distinct, so rule it out.")
                        if not keep:
                            emit(f"  no {'remaining ' if excluded else ''}meaning of {zl} fits the range — backtrack."); return False
                        v, rv = keep[0]
                        emit(f"  {rv} = {self._arith(v, a, b)}, so lock {zl} = {vname(v)}.")
                        locked[e.op] = v; changed = True
        return True

    def verify(self, A, locked):
        for op in self.fam:
            for e in self.p.by_op[op]:
                if cg.apply_op(locked[op], self.opval(e.d1, e.d2, A), self.opval(e.d3, e.d4, A)) != cg.rhs_int(e.rhs, A, self.mode, self.op_set):
                    return False
        return True

    def pick_branch(self, d, un):
        unset = set(un)
        def esyms(e): return (e.d1, e.d2, e.d3, e.d4, *[c for c in e.rhs if c not in self.op_set])
        cands = [e for e in self.p.examples if e.op in self.fam and any(s in unset for s in esyms(e))]
        if cands:
            e = min(cands, key=lambda e: (sum(s in unset for s in esyms(e)), self.p.examples.index(e)))
            pool = [s for s in esyms(e) if s in unset]
            return min(pool, key=lambda x: (len(d[x]), self.syms.index(x)))
        return min(un, key=lambda x: (len(d[x]), self.syms.index(x)))

    def search(self, dom, locked, emit, depth=0):
        d = copy.deepcopy(dom); lk = dict(locked)   # propagate may add forced operator locks to lk
        if not self.propagate(d, lk, emit): return None
        # any operator still unlocked whose operands are known is AMBIGUOUS (>1 variant fits) -> try each
        for ei, e in enumerate(self.p.examples, 1):
            if e.op in self.fam and e.op not in lk and all(len(d[s]) == 1 for s in (e.d1, e.d2, e.d3, e.d4)):
                a, b, rstr, opc, zl, keep, excluded, used = self._op_cands(e, d, lk)
                emit(f"We now know all the operands in EX{ei}: {a} {opc} {b}, {rstr}.")
                for v, rv in excluded:
                    emit(f"  {rv} = {self._arith(v, a, b)} would make {zl} = {vname(v)}, but {vname(v)} is already operator {self.p.op_map[used[v]]}'s meaning; operators are distinct, so rule it out.")
                emit(f"  more than one meaning of {zl} lands in range, try each:")
                for v, rv in keep:
                    emit(f"  {zl} = {vname(v)} ({self._arith(v, a, b)} = {rv}):")
                    res = self.search(d, {**lk, e.op: v}, lambda m: emit("  " + m), depth+1)
                    if res: return res
                emit(f"  no meaning of {zl} fits — backtrack."); return None
        un = [s for s in self.syms if len(d[s]) > 1]
        if not un:
            A = {s: next(iter(d[s])) for s in self.syms}
            if all(op in lk for op in self.fam) and self.verify(A, lk): return (A, dict(lk))
            emit("all pinned but the equations don't all check — backtrack."); return None
        s = self.pick_branch(d, un)
        emit(f"No equation narrows a variable further. Branch on {self.lm[s]} in {{{_set(d[s])}}} (least-unknown equation first), try in order.")
        for val in sorted(d[s]):
            emit(f"Assume {self.lm[s]} = {val}:")
            r = self.search({**d, s: {val}}, lk, lambda m: emit("  " + m), depth+1)
            if r: return r
        return None


def honest_search(p, mode, op_set, concat_results, confirmed):
    arith_ops = [op for op in p.by_op if op not in concat_results]
    def fam_order(op):
        if op in confirmed: return ['noisy_subtraction']
        counts = [len([s for s in e.rhs if s not in op_set]) for e in p.by_op[op]]
        return cg.family_ranking(counts, cg.op_top_family(op))
    combos = list(iproduct(*[fam_order(op) for op in arith_ops]))
    log = []
    for ci, combo in enumerate(combos):
        fam = dict(zip(arith_ops, combo)); eng = Engine(p, mode, op_set, fam)
        head = "Try operators " if ci == 0 else "Try the next possibility — operators "
        log.append(head + ", ".join(f"{p.op_map[op]} = {fam[op]}" for op in arith_ops) + ":")
        dom = {s: (set(range(1, 10)) if s in eng.lead else set(range(10))) for s in p.digit_syms}
        res = eng.search(dom, {}, lambda m: log.append("  " + m))   # nest the whole search under this attempt
        if res:
            A, variants = res
            log.append("  Every equation checks: " + ", ".join(f"{p.op_map[op]} = {vname(variants[op])}" for op in arith_ops) +
                       ". Solution: " + ", ".join(f"{p.letter_map[s]} = {A[s]}" for s in sorted(p.digit_syms, key=lambda x: p.letter_map[x])) + ".")
            return log, A, variants
        log.append("  -> no consistent solution under these meanings" + ("; try the next possibility." if ci+1 < len(combos) else "."))
    return log, None, None


def gen(name):
    p = cg.parse(f'{name}/question.txt'); op_set = set(p.op_syms)
    prefix, mode, concat_results, confirmed = cg.make_prefix(p)
    arith_ops = [op for op in p.by_op if op not in concat_results]
    log = []; A = None; variants = None; final_mode = mode
    if arith_ops:
        log, A, variants = honest_search(p, mode, op_set, concat_results, confirmed)
        if A is None and mode == 'standard':
            log.append(""); log.append("No consistent assignment under standard order. Retry little-endian (units digit first).")
            l2, A, variants = honest_search(p, 'little_endian', op_set, concat_results, confirmed)
            log += l2; final_mode = 'little_endian' if A is not None else 'standard'
        elif A is not None:
            final_mode = mode
    ans_lines, ans_str = emit_answer(p, A, variants, concat_results, final_mode, op_set)   # ans_str = answer in symbols
    cot = (prefix.rstrip() + "\n\nNow search for a digit assignment.\n" + "\n".join(log) +
           "\n\n" + "\n".join(ans_lines) +
           f"\nI will now return the answer in \\boxed{{}}, the answer is \n\\boxed{{{ans_str}}}")
    return cot, A, variants, final_mode


def _fill_missing(A, op_set, ans, missing, lf):
    """Replace each '_' (a digit with no seen symbol) by a guessed symbol and narrate it honestly.
    The mapping is random per puzzle, so the unseen symbol can't be derived; we show the full operand
    alphabet, then the same alphabet with every symbol already used in this puzzle masked as 'X', then
    take the FIRST remaining candidate (first operand symbol in ASCII order not used in this puzzle).
    This is a deterministic rule over visible features, so the model can learn and reproduce it (a
    seeded-random pick would be unlearnable noise — the seed isn't in the prompt). Distinct symbols for
    distinct missing digits. `lf` renders the filled answer in letter form for the answer line.
    Returns (narration_lines, filled_answer) where the last line already states the answer."""
    if not missing:
        return [], ans
    used = set(A.keys()) | set(op_set)               # every symbol that appears in this puzzle
    cand = [s for s in POOL23 if s not in used]      # candidates (operators are in `used`, so excluded)
    guess = {}; ci = 0
    for d in dict.fromkeys(missing):                 # distinct missing digits, in order of appearance
        guess[d] = cand[ci] if ci < len(cand) else '?'
        ci += 1
    out = []; mi = 0
    for ch in ans:
        if ch == '_':
            out.append(guess[missing[mi]]); mi += 1
        else:
            out.append(ch)
    filled = ''.join(out)
    full = ' '.join(POOL23)
    row = ' '.join(s if s not in used else 'X' for s in POOL23)
    pairs = ", ".join(f"{d} -> {s}" for d, s in guess.items())
    digs = ", ".join(str(d) for d in guess)
    head = (f"But digit {digs} doesn't show up in the examples, so I have to guess what symbol it maps to."
            if len(guess) == 1 else
            f"But digits {digs} don't show up in the examples, so I have to guess what symbols they map to.")
    return ([head,
             "I know the full set of operand symbols, ordered by ASCII:",
             f"  {full}",
             "Excluding every symbol that already appears in this puzzle (marked X), I still have:",
             f"  {row}",
             "I know the symbol-to-digit mapping is assigned uniformly at random, so none of the remaining candidates is more likely than another.",
             f"Since I have to choose, I will just take the first one that is left: {pairs}, answer = {lf(filled)} = {filled}"],
            filled)


def emit_answer(p, A, variants, concat_results, mode, op_set):
    q = p.query; d1, d2, zs, d3, d4 = q[0], q[1], q[2], q[3], q[4]; L = []
    zl = p.op_map[zs]                         # the query operator's LETTER (raw symbol only ever in the answer)
    dig2sym = {v: k for k, v in A.items()} if A else {}
    s2l = {**p.letter_map, **p.op_map}        # symbol -> letter (digit symbols and the operator/sign symbols)
    lf = lambda ans: ''.join(s2l.get(c, c) for c in ans)   # symbol-form answer -> letter form ('_' for an unseen digit stays '_')
    if zs in concat_results:
        k = concat_results[zs]; ans = (d1+d2+d3+d4) if k == 'fwd' else (d3+d4+d1+d2)
        L.append(f"Query operator {zl} is concat_{k} (from the examples); answer = {lf(ans)} = {ans}"); return L, ans
    if zs in p.by_op and variants and zs in variants:
        var = variants[zs]
        a = 10*A[d1]+A[d2] if mode == 'standard' else 10*A[d2]+A[d1]
        b = 10*A[d3]+A[d4] if mode == 'standard' else 10*A[d4]+A[d3]
        r = cg.apply_op(var, a, b); ans, missing = gs.num_to_sym(r, dig2sym, zs, mode)
        expr = cg.op_str(var, a, b).rsplit('=', 1)[0]
        gl, ans = _fill_missing(A, op_set, ans, missing, lf)
        if not missing:
            L.append(f"Query operator {zl} = {vname(var)} (from the examples), so QUERY is {expr} = {r}; answer = {lf(ans)} = {ans}")
        else:
            L.append(f"Query operator {zl} = {vname(var)} (from the examples), so QUERY is {expr} = {r}.")
            L += gl
        return L, ans
    if A:
        # Guess the unseen query operator. Apply operator distinctness at the OPERATION level first:
        # drop any arithmetic family already represented by a locked operator. Then use the prior only
        # to choose among the families that remain — and not at all when only one operation is left.
        ARITH = ('noisy_mul', 'noisy_add', 'noisy_subtraction')
        canon = {'+': 'noisy_add', '-': 'noisy_subtraction', '*': 'noisy_mul'}
        locked = list((variants or {}).values())
        used_fams = {cg._OP_FAMILY.get(v) for v in locked} & set(ARITH)
        present = [vname(v) for v in locked if cg._OP_FAMILY.get(v) in ARITH]
        order = [f for f in dict.fromkeys([cg.op_top_family(zs)] + cg.op_seen_families(zs)) if f in ARITH]
        order += [f for f in ARITH if f not in order]            # ensure all three are present
        free = [f for f in order if f not in used_fams]
        chosen = (free or order)[0]
        base = gs.BASE.get(chosen, 'add')
        a = 10*A[d1]+A[d2] if mode == 'standard' else 10*A[d2]+A[d1]
        b = 10*A[d3]+A[d4] if mode == 'standard' else 10*A[d4]+A[d3]
        r = cg.apply_op(base, a, b); ans, missing = gs.num_to_sym(r, dig2sym, zs, mode)
        expr = cg.op_str(base, a, b).rsplit('=', 1)[0]
        prior = (f"since it is {zs}, it is most likely {chosen}" if canon.get(zs) == chosen
                 else f"I have most often seen {zs} represent {chosen}")
        intro = f"The examples are arithmetically solvable, so I'll guess {zl} is an arithmetic operator too"
        pres = " and ".join(present); hv = "has" if len(present) == 1 else "have"
        if not used_fams:                                        # nothing to exclude -> prior only
            L.append(f"{intro}; {prior}.")
        elif len(free) == 1:                                     # forced by distinctness -> no prior needed
            L.append(f"{intro}. Since {pres} {hv} appeared in the puzzle, by the distinct operator rule {zl} must be the remaining operation, {chosen}.")
        elif len(free) >= 2:                                     # narrowed, prior breaks the tie
            L.append(f"{intro}. Since {pres} {hv} appeared in the puzzle, by the distinct operator rule {zl} is {' or '.join(free)}; {prior}.")
        else:                                                    # all three arithmetic operations already used
            L.append(f"{intro}. Since {pres} {hv} appeared in the puzzle, every arithmetic operation is already used, so I fall back to the most likely, {chosen}.")
        gl, ans = _fill_missing(A, op_set, ans, missing, lf)
        if not missing:
            L.append(f"Guess {zl} = {vname(base)}, so QUERY is {expr} = {r}; answer = {lf(ans)} = {ans}")
        else:
            L.append(f"Guess {zl} = {vname(base)}, so QUERY is {expr} = {r}.")
            L += gl
        return L, ans
    ans = d1+d2+d3+d4
    L.append(f"The examples aren't arithmetically solvable; guess concat_fwd; answer = {lf(ans)} = {ans}"); return L, ans


if __name__ == '__main__':
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else 'arithmetic_34d1d16f'
    cot, A, variants, mode = gen(name)
    print(cot); print("\n--- solution:", A, "mode:", mode)
