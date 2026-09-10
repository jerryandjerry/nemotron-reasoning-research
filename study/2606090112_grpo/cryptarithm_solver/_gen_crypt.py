#!/usr/bin/env python3
"""Cryptarithm CoT generator — matured Alice template (§-tree, system-log, joint digit+operator search).

Reuses cot_generator (cg) for parse + arithmetic helpers. The search is the same honest
interval-propagation DFS as _honest_solver.py; only the PRESENTATION is the new template:
  - numbered prior knowledge, ~mul/~add/~sub/~concat tags, rightward/leftward readings as §1/§2,
  - §N.1 write equations + operator state + tail-sign gut-check, §N.2 concat, §N.3 digit prune,
  - §N.4 joint search: a full `State:` line right after each `Try` (operator combo / branch value),
    narrowings as deltas; [] = ordered try-list, {} = unordered candidate set,
  - Conclusion counts ALL unknowns (digits + operators); answer maps digits back to symbols.

gen_cot(name) -> full CoT string.
"""
import copy, os, re
from itertools import product as iproduct
from tokenizers import Tokenizer as _Tokenizer
import cot_generator as cg
from cot_generator import VARIANT_FN, CORES, apply_op, op_str

_HERE = os.path.dirname(os.path.abspath(__file__))
_TOK_PATH = os.path.join(_HERE, 'tokenizer.json')                       # portable: tokenizer co-located in this folder
if not os.path.exists(_TOK_PATH):                                       # fallback to the in-repo location
    _TOK_PATH = os.path.join(_HERE, '..', '..', '02_train', '260512_huikang_085', 'repo', 'tokenizer.json')
_TOK = _Tokenizer.from_file(_TOK_PATH)

def _fused_tokens(p):
    """Multi-symbol BPE tokens occurring in this puzzle's equations+query, in first-appearance order.
    A token counts if it spans >=2 cipher symbols of this puzzle — exactly what step-1 must un-merge."""
    pool = set(p.digit_syms) | set(p.op_syms); seen = []
    runs = [s for ex in p.examples for s in (ex.lhs, ex.rhs)] + [p.query]
    for run in runs:
        for a, b in _TOK.encode(run, add_special_tokens=False).offsets:
            seg = run[a:b]
            if len(seg) >= 2 and all(c in pool for c in seg) and seg not in seen:
                seen.append(seg)
    return seen

CORE = {'noisy_mul': 'mul', 'noisy_add': 'add', 'noisy_subtraction': 'noisy_subtraction'}
TAG  = {'noisy_mul': '~mul', 'noisy_add': '~add', 'noisy_subtraction': '~sub', 'concat': '~concat'}
SYMB = {'mul': 'a×b', 'mul_p1': 'a×b+1', 'mul_m1': 'a×b-1', 'mul_p2': 'a×b+2', 'mul_m2': 'a×b-2',  # abstract formula per variant (a, b)
        'add': 'a+b', 'add_p1': 'a+b+1', 'add_m1': 'a+b-1', 'add_p2': 'a+b+2', 'add_m2': 'a+b-2',
        'absdiff': '|a-b|', 'sub_signed': 'a-b', 'neg_absdiff': '-|a-b|', 'rsub_signed': 'b-a',
        'sub_signed_p1': 'a-b+1', 'sub_signed_m1': 'a-b-1', 'sub_signed_p2': 'a-b+2', 'sub_signed_m2': 'a-b-2'}
CANON = {'+': 'noisy_add', '-': 'noisy_subtraction', '*': 'noisy_mul'}            # symbol's canonical family — used ONLY for the unseen-operator guess
CDISP = {'fwd': 'a∥b', 'rev': 'b∥a'}                                              # concat direction display (numeric-equation notation)

def mul_decomp(name, a, b):
    """op_str, but an explicit a×b is expanded by partial products (numeric 4-stage):
    a×b[±k] = (t+u)×b[±k] = t×b + u×b[±k] = P + Q[±k] = result. Non-mul ops fall through unchanged."""
    s = op_str(name, a, b)
    if '×' not in s: return s
    formula, result = s.rsplit('=', 1)
    m = re.match(r'(\d+)×(\d+)([+-]\d)?$', formula)
    if not m: return s
    A, B, off = int(m.group(1)), int(m.group(2)), (m.group(3) or '')
    ks = '' if not off else f" {off[0]} {off[1:]}"          # ' + 1' / ' - 2' for the spaced stages
    if A >= 10 and A % 10:   t, u = (A // 10) * 10, A % 10; head = f"({t}+{u})×{B}"; mid = f"{t}×{B} + {u}×{B}"; tail = f"{t*B} + {u*B}"
    elif B >= 10 and B % 10: t, u = (B // 10) * 10, B % 10; head = f"{A}×({t}+{u})"; mid = f"{A}×{t} + {A}×{u}"; tail = f"{A*t} + {A*u}"
    else: return s
    return f"{A}×{B}{off} = {head}{ks} = {mid}{ks} = {tail}{ks} = {result.strip()}"
MRV_GLOBAL = True   # branch heuristic: True = global minimum-remaining-values; False = least-unknown-equation-first
QUERY_MIN  = False  # TEST scheme: stop the search the moment the query is answerable (don't resolve all 10 digits)
FIRST_MODE = 'plain'    # first-§N.4 narrowing presentation. 'plain'=no special block (every compact line now carries its own Form A/B witness, so the verbose ±2 expansion is redundant);
                        # 'verbose'=full ±2 interval arithmetic; 'simple'=one-line operative cut (lhs/rhs ranges + the boundary reason)
RULE1_CONCAT = True     # unseen query operator AND no example result reaches 4 digits => guess ~concat, short-circuit to the answer
UNITS_PRUNE  = False    # TEST: aggressive units-digit congruence pruning, ASSUMING pure operators (ignore ±k noise) — fits more puzzles under cap, loses noisy ones
LEADPAIR_D   = True      # SOUND rule D: a ~sub example with a 1-digit result forces |lead1-lead2| <= 2 (mined; not captured by the wide ~sub interval). Fires during branching => §N.5.
SPLIT_45     = True      # presentation: split the search into §N.4 (deterministic narrowing) and §N.5 (branch on the remaining unknowns)
FORWARD_CHECK = False    # one-ply lookahead before branching. Faithful version (full mini-cascade, no hidden steps) only saves the redundant State-line snapshot on dead branches => +7 trainable, not worth a format change. The agents' "+65" required hiding the cascade (a forward-only violation). OFF.
# (rough-operand computation was tested and removed 2026-06-02: rounding operands to tens makes product bounds too wide to narrow => search branches instead of resolving => trainable 312->133, total tokens +~190%. The ±2 was not the issue; the operand rounding was. Dead end.)

def vname(v):
    return (v.replace('_m1', '_minus1').replace('_p1', '_plus1')
             .replace('_m2', '_minus2').replace('_p2', '_plus2'))

def _runs(s):
    """Collapse each contiguous run of length >=3 to 'lo-hi'; singletons/pairs stay explicit.
    e.g. {0,2,3,4,5,6,7,8,9} -> '0,2-9' ; {2,3,4,7,8,9} -> '2-4,7-9' ; {5,6} -> '5,6'."""
    s = sorted(s); out = []; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[j] + 1: j += 1
        run = s[i:j + 1]
        out.append(f"{run[0]}-{run[-1]}" if len(run) >= 3 else ",".join(map(str, run)))
        i = j + 1
    return ",".join(out)

def fmt_set(s):  return "{%s}" % _runs(s)
def fmt_list(s): return "[%s]" % _runs(s)

def fam_interval(fam, o1, o2):                                   # sound superset of ALL the family's variants (±2 noise)
    if fam == 'noisy_mul': return (o1[0]*o2[0]-2, o1[1]*o2[1]+2)
    if fam == 'noisy_add': return (o1[0]+o2[0]-2, o1[1]+o2[1]+2)
    m = max(o1[1]-o2[0], o2[1]-o1[0], 0); return (-m-2, m+2)

def exact_interval(var, o1, o2):
    a1, a2 = o1; b1, b2 = o2
    if var in ('mul', 'mul_p1', 'mul_m1', 'mul_p2', 'mul_m2'):
        off = {'mul':0,'mul_p1':1,'mul_m1':-1,'mul_p2':2,'mul_m2':-2}[var]; return (a1*b1+off, a2*b2+off)
    if var in ('add', 'add_p1', 'add_m1', 'add_p2', 'add_m2'):
        off = {'add':0,'add_p1':1,'add_m1':-1,'add_p2':2,'add_m2':-2}[var]; return (a1+b1+off, a2+b2+off)
    if var in ('sub_signed', 'sub_signed_p1', 'sub_signed_m1', 'sub_signed_p2', 'sub_signed_m2'):
        off = {'sub_signed':0,'sub_signed_p1':1,'sub_signed_m1':-1,'sub_signed_p2':2,'sub_signed_m2':-2}[var]
        return (a1-b2+off, a2-b1+off)
    adlo = max(0, a1-b2, b1-a2); adhi = max(a2-b1, b2-a1, 0)
    if var == 'absdiff': return (adlo, adhi)
    if var == 'neg_absdiff': return (-adhi, -adlo)
    if var == 'rsub_signed': return (b1-a2, b2-a1)
    raise ValueError(var)


class Engine:
    """Same search as _honest_solver.Engine; emit strings follow the new template."""
    def __init__(self, p, mode, op_set, fam, concat_results, sign_ops):
        self.p, self.mode, self.op_set, self.fam = p, mode, op_set, fam
        self.concat_results, self.sign_ops = concat_results, sign_ops
        self.lm = p.letter_map; self.syms = p.digit_syms
        self.lead = cg.get_leading(p, op_set, mode)
        self._verbose_done = False     # the FIRST §N.4 narrowing is shown in full; the rest compact

    # ---- state line (every unknown: operators then digit symbols) ----
    def state_line(self, dom, locked):
        ops = []
        for op in self.p.op_syms:
            ol = self.p.op_map[op]
            if op in locked:                 ops.append(f"{ol}={SYMB[locked[op]]}")
            elif op in self.concat_results:  ops.append(f"{ol}={CDISP[self.concat_results[op]]}")
            elif op in self.fam:             ops.append(f"{ol}=[{TAG[self.fam[op]]}]")
            else:                            ops.append(f"{ol}=unknown")
        digs = [f"{self.lm[s]}={fmt_set(dom[s])}" for s in self.syms]
        return "Current State: " + " ".join(ops + digs)

    # ---- numeric helpers (identical to _honest_solver) ----
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
        # compact 2-char operand in the reading frame (tens-first); pinned positions show the digit.
        t, u = (d1, d2) if self.mode == 'standard' else (d2, d1)   # t = tens symbol, u = units symbol
        tc = str(next(iter(dom[t]))) if len(dom[t]) == 1 else self.lm[t]
        uc = str(next(iter(dom[u]))) if len(dom[u]) == 1 else self.lm[u]
        return tc + uc                                              # spacing is applied globally in _finalize (keyed on the puzzle letters)
    def rhs_expr(self, rhs, dom):
        ch = list(rhs); sign = ''
        if ch and ch[0] in self.op_set: sign = '-'; ch = ch[1:]
        elif ch and ch[-1] in self.op_set: sign = '-'; ch = ch[:-1]
        body = ''.join(str(next(iter(dom[c]))) if len(dom[c]) == 1 else self.lm[c] for c in ch)
        return sign + body

    def narrow(self, sym, e, dom, locked):
        keep = set()
        for v in dom[sym]:
            d2 = {**dom, sym: {v}}
            r = self.res_iv(e, d2, locked); rh = self.rhs_iv(e.rhs, d2)
            if not (r[1] < rh[0] or rh[1] < r[0]): keep.add(v)
        return keep

    def _units_syms(self, e):
        u1 = e.d2 if self.mode == 'standard' else e.d1     # units = the less-significant operand symbol
        u2 = e.d4 if self.mode == 'standard' else e.d3
        rch = [c for c in e.rhs if c not in self.op_set]
        if not rch: return None
        ur = rch[-1] if self.mode == 'standard' else rch[0]
        return u1, u2, ur

    def _units_narrow(self, e, dom):
        """SOUND units-digit congruence for mul/add: result_units ∈ {(u1∘u2 + k) mod 10 : k∈-2..2}.
        Widened by the ±2 noise so it never removes the true digit. Enumerates the ≤3 distinct units
        symbols (each domain ≤10), enforcing all-different among them."""
        fam = self.fam.get(e.op)
        if fam not in ('noisy_mul', 'noisy_add'): return []
        us = self._units_syms(e)
        if not us: return []
        u1, u2, ur = us
        base = (lambda a, b: a*b) if fam == 'noisy_mul' else (lambda a, b: a+b)
        distinct = list(dict.fromkeys([u1, u2, ur]))
        ok = {s: set() for s in distinct}
        for combo in iproduct(*[sorted(dom[s]) for s in distinct]):
            if len(set(combo)) != len(distinct): continue       # distinct symbols => distinct digits
            val = dict(zip(distinct, combo))
            if (base(val[u1], val[u2]) - val[ur]) % 10 in (0, 1, 2, 8, 9):   # within ±2 (mod 10)
                for s in distinct: ok[s].add(val[s])
        return [(s, ok[s]) for s in distinct if ok[s] != dom[s]]

    def _leadpair_lr(self, e):
        """The two operand LEADING symbols in the reading frame (tens position)."""
        return (e.d1, e.d3) if self.mode == 'standard' else (e.d2, e.d4)

    def _leadpair_D(self, e, dom):
        """SOUND rule D: a ~sub example showing a 1-digit result forces the operand leading digits to differ
        by at most 2 (1-digit magnitude ≤9, +2 noise => |a-b|≤11 => 10·|l1-l2|-9 ≤ 11 => |l1-l2| ≤ 2).
        The wide ~sub interval does NOT capture this. Returns [(sym, narrowed_domain), ...]."""
        if self.fam.get(e.op) != 'noisy_subtraction': return []
        if len([c for c in e.rhs if c not in self.op_set]) != 1: return []
        l1, l2 = self._leadpair_lr(e)
        if l1 == l2: return []
        n1 = {v for v in dom[l1] if min(dom[l2]) - 2 <= v <= max(dom[l2]) + 2}
        n2 = {v for v in dom[l2] if min(dom[l1]) - 2 <= v <= max(dom[l1]) + 2}
        return [(s, nd) for s, nd in ((l1, n1), (l2, n2)) if nd != dom[s]]

    def _arith(self, v, a, b):
        return mul_decomp(v, a, b).rsplit('=', 1)[0].strip()

    def _crosscheck(self, ei, op, v, A, emit):
        """After locking `op`=v, verify it on every OTHER example whose operands AND RHS are pinned in A,
        and emit one 'also appears in ...; check ...; Confirmed.' line (numeric-style)."""
        sibs = []
        for j, e2 in enumerate(self.p.examples, 1):
            if j == ei or e2.op != op: continue
            ssyms = (e2.d1, e2.d2, e2.d3, e2.d4, *[c for c in e2.rhs if c not in self.op_set])
            if not all(s in A for s in ssyms): continue
            a2, b2 = self.opval(e2.d1, e2.d2, A), self.opval(e2.d3, e2.d4, A)
            rv = cg.apply_op(v, a2, b2)
            if rv != cg.rhs_int(e2.rhs, A, self.mode, self.op_set): continue   # only confirm genuinely-consistent siblings
            sibs.append((j, self._arith(v, a2, b2), rv))
        if sibs:
            exl = ", ".join(f"EX{j}" for j, _, _ in sibs)
            vs = "; ".join(f"EX{j}: {x} = {r}" for j, x, r in sibs)
            emit(f"{self.p.op_map[op]} also appears in {exl}; check {vs}. Confirmed.")

    def _op_cands(self, e, dom, locked):
        A = {s: next(iter(dom[s])) for s in self.syms if len(dom[s]) == 1}
        a, b = self.opval(e.d1, e.d2, A), self.opval(e.d3, e.d4, A)
        rh = self.rhs_iv(e.rhs, dom)
        opc = TAG[self.fam[e.op]]
        zl = self.p.op_map[e.op]
        rstr = f"= {rh[0]}" if rh[0] == rh[1] else f"in [{rh[0]},{rh[1]}]"
        used = {var: op for op, var in locked.items()}
        cands = [(v, cg.apply_op(v, a, b)) for v in cg.CORES[CORE[self.fam[e.op]]]]
        cands = [(v, rv) for v, rv in cands if rh[0] <= rv <= rh[1]]
        excluded = [(v, rv) for v, rv in cands if v in used]
        keep = [(v, rv) for v, rv in cands if v not in used]
        return a, b, rstr, opc, zl, keep, excluded, used

    # ---- verbose first-§N.4 narrowing: show the interval arithmetic explicitly ----
    def _mul_chain(self, lo1, lo2, hi1, hi2):
        """[lo1×lo2-2, hi1×hi2+2] expanded by partial products (split a factor tens+units), deduped."""
        def parts(x, y):
            if x >= 10 and x % 10: t, u = (x//10)*10, x % 10; return f"({t}+{u})×{y}", f"{t*y} + {u*y}"
            if y >= 10 and y % 10: t, u = (y//10)*10, y % 10; return f"{x}×({t}+{u})", f"{x*t} + {x*u}"
            return f"{x}×{y}", f"{x}×{y}"
        lh, lt = parts(lo1, lo2); hh, ht = parts(hi1, hi2)
        seq = [f"[{lo1}×{lo2}, {hi1}×{hi2}] widened by the ±2 noise",
               f"[{lh} - 2, {hh} + 2]", f"[{lt} - 2, {ht} + 2]", f"[{lo1*lo2-2}, {hi1*hi2+2}]"]
        out = [seq[0]]
        for s in seq[1:]:
            if s != out[-1]: out.append(s)
        return " = ".join(out)

    def _join_and(self, items):
        return " and ".join(items) if len(items) == 2 else ", ".join(items)

    def _operand_str(self, d1, d2, dom):
        t, u = (d1, d2) if self.mode == 'standard' else (d2, d1)
        disp = self.opexpr(d1, d2, dom); iv = self.operand_iv(d1, d2, dom)
        tc = str(next(iter(dom[t]))) if len(dom[t]) == 1 else self.lm[t]
        uc = str(next(iter(dom[u]))) if len(dom[u]) == 1 else self.lm[u]
        note = self._join_and([f"{self.lm[x]}∈{fmt_set(dom[x])}" for x in dict.fromkeys((t, u)) if len(dom[x]) > 1])
        return f"{disp} = 10·{tc} + {uc}" + (f", with {note}" if note else "") + f", so {disp} ∈ [{iv[0]},{iv[1]}]"

    def _lhs_iv_str(self, e, dom, locked):
        o1, o2 = self.operand_iv(e.d1, e.d2, dom), self.operand_iv(e.d3, e.d4, dom)
        opc = TAG[self.fam[e.op]]
        oe1, oe2 = self.opexpr(e.d1, e.d2, dom), self.opexpr(e.d3, e.d4, dom)
        fam = self.fam.get(e.op)
        if e.op not in locked and fam == 'noisy_mul':
            return f"{opc}({oe1}, {oe2}) ∈ {self._mul_chain(o1[0], o2[0], o1[1], o2[1])}"
        if e.op not in locked and fam == 'noisy_add':
            return (f"{opc}({oe1}, {oe2}) ∈ [{o1[0]}+{o2[0]}, {o1[1]}+{o2[1]}] widened by the ±2 noise "
                    f"= [{o1[0]+o2[0]-2}, {o1[1]+o2[1]+2}]")
        r = self.res_iv(e, dom, locked)
        return f"{opc}({oe1}, {oe2}) ∈ [{r[0]},{r[1]}]"

    def _rhs_str(self, rhs, dom):
        ch = list(rhs); sign = ''
        if ch and ch[0] in self.op_set: sign = '-'; ch = ch[1:]
        elif ch and ch[-1] in self.op_set: sign = '-'; ch = ch[:-1]
        n = len(ch); coeffs = [10**(n-1-i) for i in range(n)] if self.mode == 'standard' else [10**i for i in range(n)]
        disp = self.rhs_expr(rhs, dom); iv = self.rhs_iv(rhs, dom)
        terms = []
        for c, s in zip(coeffs, ch):
            sym = str(next(iter(dom[s]))) if len(dom[s]) == 1 else self.lm[s]
            terms.append(f"{c}·{sym}" if c > 1 else f"{sym}")
        body = f"-({' + '.join(terms)})" if sign else ' + '.join(terms)
        note = self._join_and([f"{self.lm[s]}∈{fmt_set(dom[s])}" for s in dict.fromkeys(ch) if len(dom[s]) > 1])
        return f"{disp} = {body}" + (f", with {note}" if note else "") + f", so {disp} ∈ [{iv[0]},{iv[1]}]"

    def _why(self, changes, e, dom, locked):
        """A faithful boundary reason for a single one-end cut; else the general overlap statement."""
        if len(changes) == 1:
            s, nd = changes[0]; elim = sorted(set(dom[s]) - set(nd))
            if elim and nd:
                bv = elim[-1] if min(nd) > elim[-1] else (elim[0] if max(nd) < elim[0] else None)
                if bv is not None:
                    d2 = {**dom, s: {bv}}
                    rr = self.res_iv(e, d2, locked); rhh = self.rhs_iv(e.rhs, d2)
                    if rr[1] < rhh[0]: return f"{self.lm[s]}={bv} caps the LHS at {rr[1]}, below the RHS minimum {rhh[0]}, so "
                    if rhh[1] < rr[0]: return f"{self.lm[s]}={bv} forces the LHS to at least {rr[0]}, above the RHS maximum {rhh[1]}, so "
        return "keeping only the values for which the two ranges overlap, "

    def _opval_fixed(self, d1, d2, dom, s, v):
        dm = {**dom, s: {v}}
        if len(dm[d1]) == 1 and len(dm[d2]) == 1:
            return self.opval(d1, d2, {k: next(iter(x)) for k, x in dm.items()})
        return None

    def _pinned_lhs(self, e, dom):
        """If both operands are pinned, the lhs arithmetic 'a×b = base' (else None) — shown when a pinned lhs misses the rhs."""
        if not all(len(dom[s]) == 1 for s in (e.d1, e.d2, e.d3, e.d4)): return None
        A = {s: next(iter(dom[s])) for s in self.syms if len(dom[s]) == 1}
        a, b = self.opval(e.d1, e.d2, A), self.opval(e.d3, e.d4, A); fam = self.fam.get(e.op)
        if fam == 'noisy_mul': return f"{a}×{b} = {a*b}"
        if fam == 'noisy_add': return f"{a}+{b} = {a+b}"
        if fam == 'noisy_subtraction': return f"|{a}−{b}| = {abs(a - b)}"
        return None

    def _witA(self, ei, e, dom, locked, s, nd, r, rh):
        """Form A: per-value boundary witness (exactly narrow()'s overlap test, surfaced)."""
        lm = self.lm; fam = self.fam.get(e.op)
        operand = s in (e.d1, e.d2, e.d3, e.d4)
        if operand:
            side = lambda v: self.res_iv(e, {**dom, s: {v}}, locked); other = rh
        else:
            side = lambda v: self.rhs_iv(e.rhs, {**dom, s: {v}}); other = r
        elim = sorted(set(dom[s]) - set(nd)); kept = sorted(nd)
        in1 = s in (e.d1, e.d2); in2 = s in (e.d3, e.d4)
        one_op = operand and (in1 != in2)                           # s in exactly ONE operand => the co-operand is free of s (clean product); s in both => quadratic, no product
        ctx = ""; civ = None; sym_with_s = None
        if fam == 'noisy_mul' and one_op:
            cs = (e.d3, e.d4) if in1 else (e.d1, e.d2)               # the OTHER operand (co-operand), independent of s
            civ = self.operand_iv(cs[0], cs[1], dom)
            ctx = f"{self.opexpr(cs[0], cs[1], dom)} ∈ [{civ[0]},{civ[1]}]; "
            sym_with_s = (e.d1, e.d2) if in1 else (e.d3, e.d4)
        def wit(v, hi):                                              # raw-product arithmetic for one end (±k offset dropped; exact for drops, off by ≤ the noise for a keep landing on the noise band)
            iv = side(v); val = iv[1] if hi else iv[0]
            if operand and fam in ('noisy_mul', 'noisy_add', 'noisy_subtraction'):
                dmv = {**dom, s: {v}}
                o1 = self.operand_iv(e.d1, e.d2, dmv); o2 = self.operand_iv(e.d3, e.d4, dmv)
                if fam == 'noisy_mul':   a, b, ops = (o1[1], o2[1], '×') if hi else (o1[0], o2[0], '×')
                elif fam == 'noisy_add': a, b, ops = (o1[1], o2[1], '+') if hi else (o1[0], o2[0], '+')
                else:                                               # sub: pick the operand-order difference reaching this end
                    a, b = ((o1[1], o2[0]) if o1[1] - o2[0] >= o2[1] - o1[0] else (o2[1], o1[0])) if hi else \
                           ((o1[0], o2[1]) if o1[0] - o2[1] <= o2[0] - o1[1] else (o2[0], o1[1]))
                    ops = '−'
                base = {'×': a * b, '+': a + b, '−': a - b}[ops]; off = val - base
                if abs(off) <= 2:                                   # show the real ±k offset so the bound (and the comparison) is exact
                    offs = '' if off == 0 else (f'+{off}' if off > 0 else f'−{-off}')
                    return f"{lm[s]}={v}: {a}{ops}{b}{offs} = {val}", val
            return f"{lm[s]}={v}: {'lhs' if operand else 'rhs'} {'≤' if hi else '≥'} {val}", val
        if not (kept and elim):
            return ctx + "keep only the values whose ranges overlap" + f"; so {lm[s]}={fmt_set(nd)}"
        if operand and s in [c for c in e.rhs if c not in self.op_set]:   # s on BOTH sides -> both intervals move with s; show each boundary value's lhs-vs-rhs ranges
            Lv = lambda v: self.res_iv(e, {**dom, s: {v}}, locked)
            Rv = lambda v: self.rhs_iv(e.rhs, {**dom, s: {v}})
            lo = [v for v in elim if Lv(v)[1] < Rv(v)[0]]; hi = [v for v in elim if Rv(v)[1] < Lv(v)[0]]
            cl2 = []
            if lo:
                v = max(lo, key=lambda v: Lv(v)[1]); cl2.append(f"{lm[s]}={v}: lhs [{Lv(v)[0]},{Lv(v)[1]}], rhs [{Rv(v)[0]},{Rv(v)[1]}] — no overlap, drop")
            if hi:
                v = min(hi, key=lambda v: Lv(v)[0]); cl2.append(f"{lm[s]}={v}: lhs [{Lv(v)[0]},{Lv(v)[1]}], rhs [{Rv(v)[0]},{Rv(v)[1]}] — no overlap, drop")
            if not cl2:                                                  # no clean drop boundary (rare) -> show one kept overlap as justification
                vk = kept[0]; cl2.append(f"{lm[s]}={vk}: lhs [{Lv(vk)[0]},{Lv(vk)[1]}], rhs [{Rv(vk)[0]},{Rv(vk)[1]}] — overlap, keep")
            return "; ".join(cl2) + f"; so {lm[s]}={fmt_set(nd)}"
        low_d  = [v for v in elim if side(v)[1] < other[0]]         # side-high below rhs-low  => this value is too LOW
        high_d = [v for v in elim if side(v)[0] > other[1]]         # side-low above rhs-high  => this value is too HIGH
        cl = []                                                     # classify by the ACTUAL side interval (robust to sign / non-monotonic coef)
        if low_d:
            vd = max(low_d, key=lambda v: side(v)[1]); dd, _ = wit(vd, True);  cl.append(f"{dd} < {other[0]}, drop")
        if high_d:
            vd = min(high_d, key=lambda v: side(v)[0]); dd, _ = wit(vd, False); cl.append(f"{dd} > {other[1]}, drop")
        if len(low_d) + len(high_d) != len(elim):
            cl.append("others by overlap")
        return ctx + "; ".join(cl) + f"; so {lm[s]}={fmt_set(nd)}"

    def _collect(self, s, e, dom):
        """Collect the (add/sub) lhs as coef·s + other terms, e.g. '20·C + H + 3'. Returns (coef, expr)."""
        pos = [(e.d1, 10), (e.d2, 1), (e.d3, 10), (e.d4, 1)] if self.mode == 'standard' else [(e.d1, 1), (e.d2, 10), (e.d3, 1), (e.d4, 10)]
        if self.fam.get(e.op) == 'noisy_subtraction':
            pos = [(sym, pv if i < 2 else -pv) for i, (sym, pv) in enumerate(pos)]
        coef = sum(pv for sym, pv in pos if sym == s)
        const = 0; oth = []
        for sym, pv in pos:
            if sym == s: continue
            if len(dom[sym]) == 1: const += pv * next(iter(dom[sym]))
            else: oth.append((pv, sym))
        expr = f"{coef}·{self.lm[s]}" if abs(coef) != 1 else (self.lm[s] if coef == 1 else f"-{self.lm[s]}")
        for pv, sym in oth:
            expr += (" - " if pv < 0 else " + ") + (self.lm[sym] if abs(pv) == 1 else f"{abs(pv)}·{self.lm[sym]}")
        if const: expr += (" - " if const < 0 else " + ") + str(abs(const))
        return coef, expr

    def _iso(self, s, e, dom, locked, nd):
        """Form B: isolate s when the side moves as coef·s with a CONSTANT-width rest (linear). Returns text iff it reproduces nd, else None."""
        operand = s in (e.d1, e.d2, e.d3, e.d4)
        sidefn = (lambda v: self.res_iv(e, {**dom, s: {v}}, locked)) if operand else (lambda v: self.rhs_iv(e.rhs, {**dom, s: {v}}))
        other = self.rhs_iv(e.rhs, dom) if operand else self.res_iv(e, dom, locked)
        vals = sorted(dom[s])
        if len(vals) < 2: return None
        los = [sidefn(v)[0] for v in vals]; his = [sidefn(v)[1] for v in vals]
        d = vals[1] - vals[0]
        if (los[1] - los[0]) % d or (his[1] - his[0]) % d: return None
        coef = (los[1] - los[0]) // d
        if coef == 0 or (his[1] - his[0]) // d != coef: return None          # low/high must share the slope => width constant => truly linear
        rlo, rhi = los[0] - coef * vals[0], his[0] - coef * vals[0]
        for v, lo, hi in zip(vals, los, his):
            if lo - coef * v != rlo or hi - coef * v != rhi: return None
        bl, bh = other[0] - rhi, other[1] - rlo                              # overlap  <=>  coef·s in [bl,bh]
        if {v for v in vals if bl <= coef * v <= bh} != set(nd): return None # must reproduce narrow() exactly, else fall back to Form A
        cs = f"{coef}·{self.lm[s]}" if abs(coef) != 1 else (self.lm[s] if coef == 1 else f"-{self.lm[s]}")
        bridge = ""
        if operand and self.fam.get(e.op) in ('noisy_add', 'noisy_subtraction'):
            bridge = f"lhs = {self._collect(s, e, dom)[1]}, so "
        return f"{bridge}{cs} ∈ [{bl},{bh}] -> {self.lm[s]}={fmt_set(nd)}"

    def _narrow_witness(self, ei, e, dom, locked, s, nd, r, rh):
        iso = self._iso(s, e, dom, locked, nd)
        if iso is not None: return ('B', iso)
        return ('A', self._witA(ei, e, dom, locked, s, nd, r, rh))

    def _emit_verbose(self, ei, e, dom, locked, changes, emit):
        opc = TAG[self.fam[e.op]]
        oe1, oe2 = self.opexpr(e.d1, e.d2, dom), self.opexpr(e.d3, e.d4, dom)
        r = self.res_iv(e, dom, locked); rh = self.rhs_iv(e.rhs, dom)
        isect = (max(r[0], rh[0]), min(r[1], rh[1]))
        parts = ", ".join(f"{self.lm[s]}={fmt_set(nd)}" for s, nd in changes)
        if FIRST_MODE == 'simple':                          # one-line operative cut: ranges + boundary reason only
            emit(f"EX{ei}: {opc}({oe1}, {oe2}) = {self.rhs_expr(e.rhs, dom)}, with LHS ∈ [{r[0]},{r[1]}] and RHS ∈ [{rh[0]},{rh[1]}]. {self._why(changes, e, dom, locked)}{parts}.")
            return
        emit(f"EX{ei}: {opc}({oe1}, {oe2}) = {self.rhs_expr(e.rhs, dom)}. Work out each side's range from the current candidates:")
        emit(f"  LHS: {self._operand_str(e.d1, e.d2, dom)}; {self._operand_str(e.d3, e.d4, dom)}; then {self._lhs_iv_str(e, dom, locked)}.")
        emit(f"  RHS: {self._rhs_str(e.rhs, dom)}.")
        emit(f"  The two sides are equal, so the value is in [{r[0]},{r[1]}] ∩ [{rh[0]},{rh[1]}] = [{isect[0]},{isect[1]}]. {self._why(changes, e, dom, locked)}{parts}.")
        emit("The remaining narrowings use the same interval reasoning, shown compactly:")

    def propagate(self, dom, locked, emit):
        changed = True
        while changed:
            changed = False
            ad = True; groups = []                             # all-different naked-single to a FIXPOINT; buffer the whole cascade and emit it as ONE line (consecutive same-reason forcings share a clause; cascade steps join with ';')
            _ad = lambda g: "All-different: " + "; ".join(f"{rz}, so {', '.join(ls)}" for rz, ls in g)
            def add(reason, label):
                if groups and groups[-1][0] == reason: groups[-1][1].append(label)
                else: groups.append((reason, [label]))
            while ad:
                ad = False
                singles = {}
                for t in self.syms:
                    if len(dom[t]) == 1:
                        dv = next(iter(dom[t]))
                        if dv in singles:
                            coll = f"{self.lm[singles[dv]]} and {self.lm[t]} are both {dv}, but every symbol is a distinct digit — no."
                            emit((_ad(groups) + "; " + coll) if groups else ("All-different: " + coll)); return False
                        singles[dv] = t
                made = {}
                for s in self.syms:
                    if len(dom[s]) == 1: continue
                    taken = sorted(v for v in dom[s] if v in singles)
                    if not taken: continue
                    dom[s] = dom[s] - set(taken); changed = ad = True
                    reason = ", ".join(f"{v} is taken by {self.lm[singles[v]]}" for v in taken)
                    if not dom[s]:
                        add(reason, f"{self.lm[s]}={{}}"); emit(_ad(groups) + " — no."); return False
                    if len(dom[s]) == 1:
                        sv = next(iter(dom[s])); add(reason, f"{self.lm[s]}={{{sv}}}")
                        if sv in made:                         # this removal forces two symbols to the same digit -> contradiction, shown on the same line
                            emit(_ad(groups) + ", but every symbol is a distinct digit — no."); return False
                        made[sv] = s
            if groups:
                emit(_ad(groups) + ".")
            # hidden-single elimination: the N symbols take N DISTINCT digits. If only N distinct digits are
            # still available across all domains, every one of them must be used (a bijection) — so any digit
            # that fits only one symbol is forced there. Sound exactly when available==N (always true for 10 symbols).
            avail = [d for d in range(10) if any(d in dom[t] for t in self.syms)]
            N = len(self.syms)
            if len(avail) < N:                                 # pigeonhole: fewer distinct digits left than symbols
                emit(f"Elimination: only {len(avail)} distinct digits remain for {N} symbols — impossible, no."); return False
            if len(avail) == N:
                why = "all ten digits 0-9 are used" if N == 10 else f"the {N} symbols use up all {N} remaining digits"
                for d in avail:
                    holders = [t for t in self.syms if d in dom[t]]
                    if len(holders) == 1 and len(dom[holders[0]]) > 1:
                        s = holders[0]; dom[s] = {d}; changed = True
                        emit(f"Elimination: among the unsolved symbols only {self.lm[s]} can still be {d}, and {why}, so {self.lm[s]}={d}.")
            for ei, e in enumerate(self.p.examples, 1):
                if e.op not in self.fam: continue
                esyms = list(dict.fromkeys((e.d1, e.d2, e.d3, e.d4, *[c for c in e.rhs if c not in self.op_set])))
                changes = [(s, self.narrow(s, e, dom, locked)) for s in esyms if len(dom[s]) > 1]
                changes = [(s, nd) for s, nd in changes if nd != dom[s]]
                if changes:
                    o1, o2 = self.opexpr(e.d1, e.d2, dom), self.opexpr(e.d3, e.d4, dom)
                    opc = TAG[self.fam[e.op]]
                    r = self.res_iv(e, dom, locked); rh = self.rhs_iv(e.rhs, dom)
                    head = f"EX{ei} rhs {self.rhs_expr(e.rhs, dom)} in [{rh[0]},{rh[1]}], check lhs {opc}({o1}, {o2})"
                    empty = [s for s, nd in changes if not nd]
                    if empty:
                        if r[1] < rh[0] or rh[1] < r[0]:
                            pl = self._pinned_lhs(e, dom) if e.op not in locked else None
                            concl = (f"{pl}, beyond ±2 of the rhs [{rh[0]},{rh[1]}] — no" if pl
                                     else f"lhs [{r[0]},{r[1]}] and rhs [{rh[0]},{rh[1]}] do not overlap — no")
                        else:
                            concl = f"no digit of {self.lm[empty[0]]} is left — no"
                        emit(head + ": " + concl); return False
                    if FIRST_MODE != 'plain' and not self._verbose_done and e.op not in locked and self.fam.get(e.op):
                        self._emit_verbose(ei, e, dom, locked, changes, emit)   # first narrowing: full interval arithmetic
                        self._verbose_done = True
                        for s, nd in changes: dom[s] = nd
                        changed = True
                    else:
                        wits = [self._narrow_witness(ei, e, dom, locked, s, nd, r, rh) for s, nd in changes]
                        for s, nd in changes: dom[s] = nd
                        changed = True
                        if len(wits) == 1 and wits[0][0] == 'B':
                            emit(head + ": " + wits[0][1])              # Form B: one line
                        else:
                            emit(head + ":")
                            emit("     " + " | ".join(w[1] for w in wits))
                if LEADPAIR_D:                                    # sound rule D: ~sub 1-digit result => |lead1-lead2| <= 2
                    dch = self._leadpair_D(e, dom)
                    if dch:
                        l1, l2 = self._leadpair_lr(e)
                        lead2 = f"EX{ei}: {TAG[self.fam[e.op]]} shows a 1-digit result, so the leading digits differ by at most 2, |{self.lm[l1]}-{self.lm[l2]}| ≤ 2 -> "
                        if any(not nd for _, nd in dch):
                            bad = next(self.lm[s] for s, nd in dch if not nd)
                            emit(lead2 + f"no digit of {bad} is left — no"); return False
                        for s, nd in dch: dom[s] = nd
                        changed = True
                        emit(lead2 + ", ".join(f"{self.lm[s]}={fmt_set(nd)}" for s, nd in dch))
                if UNITS_PRUNE:                                   # aggressive units-digit congruence (assume pure)
                    uch = self._units_narrow(e, dom)
                    if uch:
                        us = self._units_syms(e); osym = '×' if self.fam[e.op] == 'noisy_mul' else '+'
                        lead2 = f"EX{ei}: units of {TAG[self.fam[e.op]]} (within ±2): ({self.lm[us[0]]}{osym}{self.lm[us[1]]}) mod 10 ≈ {self.lm[us[2]]} -> "
                        if any(not nd for _, nd in uch):
                            bad = next(self.lm[s] for s, nd in uch if not nd)
                            emit(lead2 + f"no digit of {bad} is left — no"); return False
                        for s, nd in uch: dom[s] = nd
                        changed = True
                        emit(lead2 + ", ".join(f"{self.lm[s]}={fmt_set(nd)}" for s, nd in uch))
                if e.op not in locked and all(len(dom[s]) == 1 for s in (e.d1, e.d2, e.d3, e.d4)):
                    a, b, rstr, opc, zl, keep, excluded, used = self._op_cands(e, dom, locked)
                    if len(keep) <= 1:
                        emit(f"We now know all the operands in EX{ei}: {opc}({a}, {b}) {rstr}.")
                        for v, rv in excluded:
                            emit(f"  {self._arith(v, a, b)} = {rv} would make {zl} = {SYMB[v]}, but {SYMB[v]} is already operator {self.p.op_map[used[v]]}'s meaning; operators are distinct, so rule it out.")
                        if not keep:
                            rh = self.rhs_iv(e.rhs, dom); tgt = str(rh[0]) if rh[0] == rh[1] else f"[{rh[0]},{rh[1]}]"; fam = self.fam[e.op]
                            if fam in ('noisy_add', 'noisy_mul'):                       # mul/add: the base is more than 2 from the rhs -> beyond ±2 (numeric L705)
                                base = a * b if fam == 'noisy_mul' else a + b
                                near = rh[0] if base <= rh[0] else rh[1]; sd = f"{near - base:+d}"
                                emit(f"  {self._arith(cg.CORES[CORE[fam]][0], a, b)} = {base}; rhs {tgt}; {near}-{base} = {sd}; |{sd}| > 2, beyond ±2 — no.")
                            else:                                                       # sub: walk the structural readings, each misses
                                for vv in cg.CORES[CORE[fam]][:4]:
                                    emit(f"  {self._arith(vv, a, b)} = {cg.apply_op(vv, a, b)}; rhs {tgt} — no.")
                            emit(f"  no {opc} variant gives {tgt}, so {zl} = {opc} fails."); return False
                        v, rv = keep[0]
                        emit(f"  {self._arith(v, a, b)} = {rv}, so lock {zl} = {SYMB[v]}.")
                        locked[e.op] = v; changed = True
                        A_pin = {s: next(iter(dom[s])) for s in self.syms if len(dom[s]) == 1}
                        self._crosscheck(ei, e.op, v, A_pin, lambda m: emit("  " + m))
        return True

    def verify(self, A, locked):
        for op in self.fam:
            for e in self.p.by_op[op]:
                if cg.apply_op(locked[op], self.opval(e.d1, e.d2, A), self.opval(e.d3, e.d4, A)) != cg.rhs_int(e.rhs, A, self.mode, self.op_set):
                    return False
        return True

    def _ex_syms(self):
        r = set()
        for e in self.p.examples:
            if e.op in self.fam:
                r |= {e.d1, e.d2, e.d3, e.d4, *(c for c in e.rhs if c not in self.op_set)}
        return r

    def _query_relevant(self):
        """Symbols whose value affects the answer or the operator lock: query operands + the examples that use the query operator."""
        q = self.p.query; qop = q[2]
        rel = set(s for s in (q[0], q[1], q[3], q[4]) if s not in self.op_set)
        for e in self.p.examples:
            if e.op == qop and e.op in self.fam:
                rel |= {e.d1, e.d2, e.d3, e.d4, *(c for c in e.rhs if c not in self.op_set)}
        return rel

    def _answer_ready(self, d, lk):
        """Return (A, lk) if the query is SOUNDLY computable now: query operands pinned, the query operator's
        variant VERIFIED against every one of its examples (so it can't be a wrong tentative lock), and the
        result-digit symbols pinned. Returns None (keep searching) otherwise — never trusts an unverified lock."""
        q = self.p.query; qop = q[2]
        if qop not in lk: return None
        if any(len(d[s]) > 1 for s in (q[0], q[1], q[3], q[4])): return None
        A = {s: next(iter(d[s])) for s in self.syms if len(d[s]) == 1}
        for e in self.p.examples:                       # verify EVERY locked operator against EVERY fully-pinned example
            esyms = (e.d1, e.d2, e.d3, e.d4, *[c for c in e.rhs if c not in self.op_set])
            pinned = all(s in A for s in esyms)
            if e.op == qop and not pinned: return None   # the query operator must be fully verified before we trust it
            if e.op in lk and pinned and \
               cg.apply_op(lk[e.op], self.opval(e.d1, e.d2, A), self.opval(e.d3, e.d4, A)) != cg.rhs_int(e.rhs, A, self.mode, self.op_set):
                return None                              # ANY locked+pinned example violated -> bad branch, do NOT early-stop
        a = 10*A[q[0]]+A[q[1]] if self.mode == 'standard' else 10*A[q[1]]+A[q[0]]
        b = 10*A[q[3]]+A[q[4]] if self.mode == 'standard' else 10*A[q[4]]+A[q[3]]
        r = cg.apply_op(lk[qop], a, b)
        val2sym = {next(iter(d[s])): s for s in self.syms if len(d[s]) == 1}
        if any(int(c) not in val2sym for c in str(abs(r))): return None   # a result digit has no pinned symbol
        return (A, dict(lk))

    def pick_branch(self, d, un):
        if MRV_GLOBAL:                                  # global MRV over ALL unsolved, ties broken by LETTER — fully
            return min(un, key=lambda x: (len(d[x]), self.lm[x]))   # derivable from the visible state (no hidden example filter, no symbol-index tiebreak)
        unset = set(un)                                 # default: least-unknown equation, then smallest var in it
        def esyms(e): return (e.d1, e.d2, e.d3, e.d4, *[c for c in e.rhs if c not in self.op_set])
        cands = [e for e in self.p.examples if e.op in self.fam and any(s in unset for s in esyms(e))]
        if cands:
            e = min(cands, key=lambda e: (sum(s in unset for s in esyms(e)), self.p.examples.index(e)))
            pool = [s for s in esyms(e) if s in unset]
            return min(pool, key=lambda x: (len(d[x]), self.syms.index(x)))
        return min(un, key=lambda x: (len(d[x]), self.syms.index(x)))

    def search(self, dom, locked, emit, depth=0):
        d = copy.deepcopy(dom); lk = dict(locked)
        if not self.propagate(d, lk, emit): return None
        for ei, e in enumerate(self.p.examples, 1):
            if e.op in self.fam and e.op not in lk and all(len(d[s]) == 1 for s in (e.d1, e.d2, e.d3, e.d4)):
                a, b, rstr, opc, zl, _, _, used = self._op_cands(e, d, lk)
                rh = self.rhs_iv(e.rhs, d); target = str(rh[0]) if rh[0] == rh[1] else f"[{rh[0]},{rh[1]}]"
                emit(f"We now know all the operands in EX{ei}: {opc}({a}, {b}) {rstr}.")
                for v in cg.CORES[CORE[self.fam[e.op]]]:            # walk the family's readings in their fixed order; lock the FIRST that yields the rhs and holds everywhere (no peeking at how many match)
                    rv = cg.apply_op(v, a, b)
                    if not (rh[0] <= rv <= rh[1]):
                        emit(f"  {self._arith(v, a, b)} = {rv}; rhs {target} — no."); continue
                    if v in used:
                        emit(f"  {self._arith(v, a, b)} = {rv} would make {zl} = {SYMB[v]}, but {SYMB[v]} is already operator {self.p.op_map[used[v]]}'s meaning; operators are distinct, so rule it out."); continue
                    emit(f"  {self._arith(v, a, b)} = {rv}; rhs {target}; match, try {zl} = {SYMB[v]} and check the other equations:")
                    res = self.search(d, {**lk, e.op: v}, lambda m: emit("  " + m), depth + 1)
                    if res:
                        emit(f"  every equation checks out, so lock {zl} = {SYMB[v]}.")
                        self._crosscheck(ei, e.op, v, res[0], lambda m: emit("  " + m))
                        return res
                emit(f"  no {opc} variant gives {target}, so {zl} = {opc} fails."); return None
        if QUERY_MIN:
            ar = self._answer_ready(d, lk)
            if ar is not None:
                emit("Query operator locked and the query operands (and the answer's digits) are all pinned — enough to answer; stop the search here, the other symbols are not needed.")
                return ar
        un = [s for s in self.syms if len(d[s]) > 1]
        if not un:
            A = {s: next(iter(d[s])) for s in self.syms}
            if all(op in lk for op in self.fam) and self.verify(A, lk): return (A, dict(lk))
            emit("all symbols pinned but the equations don't all check — no."); return None
        if len(un) == 1 and all(op in lk for op in self.fam):       # the last unknown: if MORE THAN ONE digit satisfies every example it is FREE (the examples don't pin it) -> take the smallest by rule, don't pretend it was deduced
            fs = un[0]; A_base = {t: next(iter(d[t])) for t in self.syms if len(d[t]) == 1}
            valid = [v for v in sorted(d[fs]) if self.verify({**A_base, fs: v}, lk)]
            if len(valid) > 1:
                if getattr(self, 'emit_n5', None): self.emit_n5()
                emit(f"{self.lm[fs]} is not pinned by any example; go with the smallest {self.lm[fs]}={valid[0]}.")
                return ({**A_base, fs: valid[0]}, dict(lk))
        s = self.pick_branch(d, un)
        if getattr(self, 'emit_n5', None): self.emit_n5()          # first branch => open §N.5 (deterministic narrowing above was §N.4)
        if FORWARD_CHECK and len(d[s]) > 1:
            # one-ply lookahead: a value that contradicts under a full propagate can never be part of a solution,
            # so shave it now with one line instead of opening a dead Try-subtree. Answer-preserving.
            live = []; dead = []
            for val in sorted(d[s]):
                d2 = copy.deepcopy(d); d2[s] = {val}; buf = []
                if self.propagate(d2, dict(lk), buf.append): live.append(val)
                else: dead.append((val, list(buf)))                # keep the FULL mini-cascade, not just the last line
            if dead:
                for val, lines in dead:
                    emit(f"Forward-check {self.lm[s]}={val}:")
                    for l in lines: emit("  " + l)                 # show every narrowing step that leads to the contradiction (no redundant State line)
                if not live:
                    emit(f"every value of {self.lm[s]} is ruled out — no."); return None
                d = {**d, s: set(live)}
        reason = "fewest candidates left" if MRV_GLOBAL else "least-unknown equation first"
        emit(f"No equation narrows a variable further. Branch on {self.lm[s]} in {fmt_list(d[s])} ({reason}).")
        for val in sorted(d[s]):
            nd = {t: ({val} if t == s else (dt - {val} if len(dt) > 1 else dt)) for t, dt in d.items()}
            # delta-state: s now takes {val}, so all-different only removes {val} from the OTHER variables that held it.
            # The rest is unchanged from the branch (the §N.4 anchor + narrowings above), so emit just the delta, not a full re-print.
            removed = [self.lm[t] for t in self.syms if t != s and val in d[t] and len(d[t]) > 1]
            tail = f" removes {val} from {', '.join(removed)}." if removed else ""
            emit(f"Try {self.lm[s]}={val}:{tail}")
            r = self.search(nd, lk, lambda m: emit("  " + m), depth+1)
            if r: return r
        return None


def search_reading(p, mode, n, op_set, concat_results, sign_ops, cand, emit):
    """Run §N.4 (deterministic narrowing) + §N.5 (branch on the remaining unknowns): iterate arithmetic
    family combos, emit 'Try operators' + state + search; the §N.5 header opens at the first branch.
    Combos come from cand[op] (§N.3, distinctness already applied to HARD-forced ops) and are FILTERED to
    distinct-family assignments — the arithmetic operators are distinct, so a combo is never tried twice with
    the same family on two operators (e.g. f=~mul and h=~mul)."""
    arith_ops = [op for op in p.by_op if op not in concat_results]
    combos = list(iproduct(*[cand[op] for op in arith_ops])) if arith_ops else [()]
    combos = [c for c in combos if len(set(c)) == len(c)]   # distinct-operator rule: drop any combo that repeats a family
    n5 = [False]
    def emit_n5():                                                  # first time any combo needs to branch, open §N.5
        if SPLIT_45 and not n5[0]:
            emit(f"  §{n}.5 search the remaining unknowns by branching, applying the §{n}.4 domains as given:")
            n5[0] = True
    for ci, combo in enumerate(combos):
        fam = dict(zip(arith_ops, combo))
        eng = Engine(p, mode, op_set, fam, concat_results, sign_ops)
        eng.emit_n5 = emit_n5
        if ci == 0:
            emit("    Try operators " + ", ".join(f"{p.op_map[op]} = {TAG[fam[op]]}" for op in arith_ops) + ":")
        else:
            emit("    Advance to the next family combination: " + ", ".join(f"{p.op_map[op]} = {TAG[fam[op]]}" for op in arith_ops) + ":")
        dom = {s: (set(range(1, 10)) if s in eng.lead else set(range(10))) for s in p.digit_syms}
        emit("    " + eng.state_line(dom, {}))
        res = eng.search(dom, {}, lambda m: emit("    " + m))
        if res:
            A, variants = res
            return A, variants, concat_results
    return None, None, concat_results


# ───────────────────────── orchestration (front matter + §-tree + answer) ─────────────────────────

# Fixed family order for the search (no per-symbol prior). Chosen by the 6-order sweep: add > sub > mul
# gives the best GT-match and the shortest CoT (add is the most common operator family). concat is detected
# in §N.2, not searched. The digit-count prune still rules families out FIRST; this order only breaks ties.
FAM_ORDER = ['noisy_add', 'noisy_subtraction', 'noisy_mul']

def fam_rank(counts):
    cand = set.intersection(*(cg.possible_families(d) for d in counts)) if counts else set(FAM_ORDER)
    if not cand: cand = set(FAM_ORDER)
    return [f for f in FAM_ORDER if f in cand]

def prior_fams(op):
    return FAM_ORDER + ['concat']   # uniform fixed order for the §N.1 state line (concat is checked in §N.2)

def _digit_count_clause(p, op, op_set):
    """The digit-count observation string for an operator's examples, e.g. 'EX1 and EX2 have 4-digit RHS (...)'.
    Returns (survivor_families, clause_string)."""
    ex_digits = [(i+1, len([s for s in e.rhs if s not in op_set])) for i, e in enumerate(p.examples) if e.op == op]
    counts = [d for _, d in ex_digits]
    def join_ex(ns):
        return " and ".join(f"EX{x}" for x in ns)
    obs = []
    for d in sorted(set(counts)):
        exs_d = [x for x, dd in ex_digits if dd == d]; have = "have" if len(exs_d) > 1 else "has"
        obs.append(f"{join_ex(exs_d)} {have} {d}-digit RHS (" + {
            1: "rules out addition and multiplication, only a subtraction-type reaches 1 digit",
            2: "rules out multiplication, which gives ≥3 digits",
            3: "rules out subtraction, whose magnitude is at most 2 digits",
        }.get(d, f"only multiplication of two 2-digit numbers reaches {d} digits") + ")")
    return fam_rank(counts), "; ".join(obs)

def prune_candidates(p, op_set, sign_ops, concat_results, om, out, n):
    """§N.3: digit-count prune + the DISTINCT-operator rule, ported from the numeric solver. Processes the
    arithmetic operators in a priority loop — sign-confirmed ~sub, then digit-count-forced (single survivor),
    then distinctness-forced (one survivor left once HARD-pinned families are removed), then soft (multiple
    survivors, try the first). Returns cand[op] = the family candidate list per arithmetic operator, with
    distinctness already applied to the HARD-forced ones so no distinct-violating combo is ever generated."""
    arith_ops = [op for op in p.by_op if op not in concat_results]
    survc = {}; clause = {}
    for op in arith_ops:
        if op in sign_ops:
            survc[op] = ['noisy_subtraction']; clause[op] = ""
        else:
            survc[op], clause[op] = _digit_count_clause(p, op, op_set)
    cand = {}; hard_taken = set(); oidx = {op: i for i, op in enumerate(arith_ops)}
    availh = lambda o: [f for f in survc[o] if f not in hard_taken]
    remaining = list(arith_ops)
    while remaining:
        sub_ops = [o for o in remaining if o in sign_ops]
        dc_ops  = [o for o in remaining if o not in sign_ops and len(survc[o]) == 1]
        di_ops  = [o for o in remaining if o not in sign_ops and len(survc[o]) > 1 and len(availh(o)) == 1]
        if sub_ops:                                  # sign already pinned it -> HARD
            op = min(sub_ops, key=lambda o: oidx[o]); fam = 'noisy_subtraction'; cand[op] = [fam]
            out.append(f"    {om[op]}: already shown to be ~sub by its output sign.")
        elif dc_ops:                                 # digit-count alone forces it -> HARD
            op = min(dc_ops, key=lambda o: oidx[o]); fam = survc[op][0]; cand[op] = [fam]
            out.append(f"    {om[op]}: {clause[op]}.")
            out.append(f"      Only {TAG[fam]} survives the digit-count filter. Try {om[op]} = {TAG[fam]}.")
        elif di_ops:                                 # distinctness vs HARD-pinned forces it -> HARD
            op = min(di_ops, key=lambda o: oidx[o]); fam = availh(op)[0]; cand[op] = [fam]
            out.append(f"    {om[op]}: {clause[op]}.")
            out.append(f"      Surviving candidates: {', '.join(TAG[f] for f in survc[op])}. By the distinct operator rule {om[op]} must be the remaining operation, {TAG[fam]}.")
        else:                                        # nothing forced -> SOFT: keep the full survivor list (distinctness acts when generating combos, not on the try-order)
            op = min(remaining, key=lambda o: (len(availh(o)), oidx[o])); cand[op] = list(survc[op]); fam = survc[op][0]
            out.append(f"    {om[op]}: {clause[op]}.")
            out.append(f"      Surviving candidates: {', '.join(TAG[f] for f in survc[op])}. Try {om[op]} = {TAG[fam]} first ({TAG[fam]} comes first in the operation order).")
        if op in sign_ops or op in dc_ops or op in di_ops: hard_taken.add(fam)
        remaining.remove(op)
    trav = ", ".join(
        f"{om[op]} = [{CDISP[concat_results[op]]}]" if op in concat_results
        else f"{om[op]} = [{', '.join(TAG[f] for f in cand[op])}]" if op in cand
        else f"{om[op]} = unknown"                      # an operator that appears ONLY in the query (unseen) — not determined here
        for op in p.op_syms)
    out.append(f"    No further pruning found. We have {trav} to traverse in order.")
    return cand

def front_matter(p):
    op_set = set(p.op_syms); lm = p.letter_map; om = p.op_map; L = []
    L.append("I need to infer the transformation rule from the examples. Both the operands and operators are unknown, so I need to deduce them both. Even a symbol that looks like an operator — /, <, >, %, &, |, ^ — can represent an operand here; the only rule is that the third symbol of the lhs is always the operator.")
    L.append("")
    sp = lambda s: ' '.join(s)                              # write every symbol separately (one token each)
    # (0) split the fused BPE tokens so step-1 un-merges them by lookup, not by re-deriving the split
    fused = _fused_tokens(p)
    if fused:
        L.append("Split the BPE merged tokens in the examples:")
        for seg in fused:
            L.append(f"  {seg}  becomes  {sp(seg)}")
        L.append("")
        L.append("Now I read each equation one symbol at a time, using the splits above:")
    else:
        L.append("First, I should read each equation one symbol at a time, writing out every symbol separately:")
    # (1) transcription: expand each MERGED equation into its individual symbols
    for i, ex in enumerate(p.examples, 1):
        L.append(f"  EX{i}  {ex.lhs} = {ex.rhs}  becomes  {sp(ex.lhs)} = {sp(ex.rhs)}")
    L.append(f"  QUERY {p.query}  becomes  {sp(p.query)}")
    # (2) operator = the 3rd of the 5 LHS/QUERY symbols
    n_ops = len(p.op_syms)
    op_phrase = (f"there is {n_ops} unique operator, which is {p.op_syms[0]}" if n_ops == 1
                 else f"there are {n_ops} unique operators, which are {' , '.join(p.op_syms)}")
    L.append(f"LHS and the QUERY are always 5 symbols and the third symbol is the operator, so in the puzzle {op_phrase}")
    L.append("Now denote them as unknown functions:")
    for s in p.op_syms: L.append(f"  {s} -> {om[s]}")
    # (3) operands = every other symbol, a letter each in order of appearance
    L.append("The other symbols are operands; assign a letter to each in the order of appearance:")
    for s in p.digit_syms: L.append(f"  {s} -> {lm[s]}")
    L.append("")
    # (4) convert each equation: spaced symbols -> spaced letters
    L.append("I will convert all the equations to letter form:")
    for i, ex in enumerate(p.examples, 1):
        lhs_let = f"{lm[ex.d1]} {lm[ex.d2]} {om[ex.op]} {lm[ex.d3]} {lm[ex.d4]}"
        rhs_let = ' '.join(om[c] if c in op_set else lm.get(c, c) for c in ex.rhs)   # show the op-symbol as its letter; the sign-check below derives that it's '-'
        L.append(f"  EX{i} {sp(ex.lhs)} = {sp(ex.rhs)} becomes {lhs_let} = {rhs_let}")
    q = p.query
    L.append(f"  QUERY {sp(q)} becomes {lm[q[0]]} {lm[q[1]]} {om[q[2]]} {lm[q[3]]} {lm[q[4]]}")
    L.append("")
    L.append("Prior knowledge for this kind of question:")
    L.append("1. We only consider four kinds of operations — noisy_addition (~add), noisy_subtraction (~sub), noisy_multiplication (~mul), and noisy_concatenation (~concat). They are noisy because the result has a tolerance of ±2 from the exact value and — for subtraction — may also appear as its absolute value or operand-reversed form. Specifically:")
    L.append("   ~add = [a+b, a+b±1, a+b±2], ~sub = [|a-b|, a-b, -|a-b|, b-a, a-b±1, a-b±2], ~mul = [a×b, a×b±1, a×b±2], ~concat = [ab, ba], all ordered by frequency. The absolute forms |a-b|, -|a-b| carry no noise — they are themselves the unsigned reading of the signed difference; the ±2 tolerance applies to the signed form a-b. Any other operation is deemed exotic.")
    L.append("2. The arithmetic operators are distinct, but the concatenation operator may repeat.")
    L.append("3. Each letter represents a distinct digit 0-9, and the leading symbol of any number is never 0.")
    L.append("4. If a symbol exists in the result (RHS), it must be a negative sign.")
    L.append("5. The reading order is either rightward or leftward (units digit first). I will check rightward first.")
    L.append("6. I should never return the answer as \"unknown\". When the final result is undetermined, I should make a best guess using what I know.")
    L.append("")
    # sign check
    sign_ops = []
    for i, ex in enumerate(p.examples, 1):
        if ex.rhs and (ex.rhs[0] in op_set or ex.rhs[-1] in op_set):
            sop = ex.rhs[0] if ex.rhs[0] in op_set else ex.rhs[-1]
            if sop not in sign_ops:
                sign_ops.append(sop)
                L.append(f"{om[sop]} appears in EX{i} output side (RHS), that means it can only be a negative sign, so {om[sop]} = [~sub].")
                others = [om[o] for o in p.op_syms if o != sop]
                if others:
                    ops_str = (others[0] if len(others) == 1 else
                               f"{others[0]} and {others[1]}" if len(others) == 2 else
                               ", ".join(others[:-1]) + f" and {others[-1]}")
                    L.append(f"By the distinct operator rule, {ops_str} cannot be ~sub.")
    if sign_ops: L.append("")
    return L, set(sign_ops)

def state_fams(p, op, sign_ops):
    """Operator family-candidate list for the §N.1 state line (after sign distinctness)."""
    if op in sign_ops: return ['~sub']
    if op not in p.by_op: return None   # unseen query operator
    excl = {'noisy_subtraction'} if sign_ops else set()
    return [TAG[f] for f in prior_fams(op) if f not in excl]

def op_state_str(p, sign_ops, drop_concat=False):
    parts = []
    note = ""
    for op in p.op_syms:
        ol = p.op_map[op]
        fs = state_fams(p, op, sign_ops)
        if fs is None:
            parts.append(f"{ol} = unknown"); note = f" since {ol} only appears in QUERY"
        else:
            if drop_concat: fs = [t for t in fs if t != '~concat']
            parts.append(f"{ol} = [{', '.join(fs)}]")
    return ", ".join(parts) + note + "."

def op_state_after_concat(p, sign_ops, concat_results):
    """Operator state restated after the §N.2 concat check: concat-locked ops show their direction,
    sign ops show [~sub], unseen ops show unknown, arithmetic ops show their families minus ~concat."""
    parts = []
    for op in p.op_syms:
        ol = p.op_map[op]
        if op in concat_results:  parts.append(f"{ol} = [{CDISP[concat_results[op]]}]")
        elif op in sign_ops:      parts.append(f"{ol} = [~sub]")
        elif op not in p.by_op:   parts.append(f"{ol} = unknown")
        else:
            excl = {'noisy_subtraction'} if sign_ops else set()
            fs = [TAG[f] for f in prior_fams(op) if f != 'concat' and f not in excl]
            parts.append(f"{ol} = [{', '.join(fs)}]")
    return ", ".join(parts)

def sign_fmt(p, op_set):
    """Where the negative sign sits in the example RHS: 'prefix' (front), 'suffix' (back), or 'none'.
    Mirrors _gen_eq.sign_format — the sign POSITION is a property of the puzzle, read off the examples,
    NOT of the reading direction."""
    for e in p.examples:
        if e.rhs and e.rhs[0] in op_set: return 'prefix'
        if e.rhs and e.rhs[-1] in op_set: return 'suffix'
    return 'none'

def reading_dir(mode, fmt):
    """The §N reading-header phrasing (exact wording from _gen_eq.attempt)."""
    if mode == 'standard': return 'rightward'
    if fmt == 'suffix': return 'leftward fully, so the negative sign flips to the front'
    if fmt == 'prefix': return 'leftward on digit only, so the negative sign stays in the front'
    return 'leftward on digit only, no negative sign in RHS'

def _reading_repr(r, mode, fmt):
    """Result r rendered in the reading frame (digit-space, '-' for the sign).
    standard: no reversal (a back sign can only occur leftward, so standard is always prefix).
    leftward suffix: reverse the WHOLE string (sign flips to back). leftward prefix/none: reverse digits only, sign stays front."""
    s = str(r)
    if mode == 'standard':
        return (s[1:] + '-') if (r < 0 and fmt == 'suffix') else s
    if r < 0 and fmt != 'suffix':
        return '-' + s[1:][::-1]
    return s[::-1]

def _remap_block(transformed, r, dig2sym, zs, zl):
    """Step-2 'remap the value back to symbols' lines, matching the numeric-CoT wording:
    first map the ciphered digits back to symbols, then (only if the result is negative) map the
    subtraction sign back to its operator symbol. Returns (lines, final_answer_string)."""
    dmap = ", ".join(f"{d} -> {dig2sym[int(d)]}" for d in dict.fromkeys(c for c in transformed if c.isdigit()))
    mid = ''.join('-' if c == '-' else dig2sym[int(c)] for c in transformed)        # digits -> symbols, sign kept as '-'
    lines = [f"map the digits back to symbols, {dmap}, so {transformed} -> {mid}"]
    if r < 0:
        ans = ''.join(zs if c == '-' else dig2sym[int(c)] for c in transformed)      # '-' -> operator symbol
        lines.append(f"Need to use the operator symbol for the subtraction sign; mapping the subtraction sign {zl} back to {zs}, so {mid} -> {ans}.")
    else:
        ans = mid
    return lines, ans

def _emit_query_tail(out, p, zl, a, b, opname, symb, d1, d2, d3, d4, mode, fmt, rdir_short, dig2sym, zs, op_set, A):
    """Shared answer tail for both the seen and unseen-arithmetic query: §step1 get the value
    (compute + reading-frame transform), §step2 remap the value back to symbols."""
    import _gen_solutions as gs
    r = cg.apply_op(opname, a, b)
    ans, missing = gs.num_to_sym(r, dig2sym, zs, mode, fmt)
    out.append(f"Now solve the QUERY {zl}({_poly(p, d1, d2, mode)}, {_poly(p, d3, d4, mode)}) = {zl}({a}, {b}), applying {zl}(a, b) = {symb},")
    expr, val = [x.strip() for x in mul_decomp(opname, a, b).rsplit("=", 1)]
    transformed = _reading_repr(r, mode, fmt)
    rev = f"; reading order is {rdir_short}, so the result {r} -> {transformed}" if mode != 'standard' else ""
    out.append(f"{zl}({a}, {b}) = {expr} = {val}{rev}")                              # step 1: get the value
    if missing:
        out[-1] += "."
        gl, ans = _fill_missing(A, op_set, ans, missing); out += gl
    else:
        lines, ans = _remap_block(transformed, r, dig2sym, zs, zl)                   # step 2: remap to symbols
        out += lines
    return ans

def emit_reading(p, mode, n, op_set, sign_ops, out):
    lm = p.letter_map; om = p.op_map
    out.append(f"§{n} reading {reading_dir(mode, sign_fmt(p, op_set))}:")
    # §N.1 writing equations
    out.append(f"  §{n}.1 writing equations:")
    def operand(d1, d2):
        a, b = (lm[d1], lm[d2]) if mode == 'standard' else (lm[d2], lm[d1])
        return a + b   # compact tens-first operand, same form as the §N.4 search
    def rhs_disp(rhs):
        return ''.join('-' if c in op_set else lm.get(c, c) for c in rhs)   # symbol-letters, operator symbol -> '-'
    for i, ex in enumerate(p.examples, 1):
        out.append(f"    EX{i}: {om[ex.op]}({operand(ex.d1, ex.d2)}, {operand(ex.d3, ex.d4)}) = {rhs_disp(ex.rhs)}")
    q = p.query
    out.append(f"    QUERY: {om[q[2]]}({operand(q[0], q[1])}, {operand(q[3], q[4])})")
    out.append(f"    {op_state_str(p, sign_ops)}")
    # gut-check: standard illegal if any RHS has a trailing operator symbol
    if mode == 'standard':
        for i, ex in enumerate(p.examples, 1):
            if ex.rhs and ex.rhs[-1] in op_set and ex.rhs[0] not in op_set:
                out.append(f"    Note: EX{i} RHS {''.join('-' if c in op_set else lm.get(c,c) for c in ex.rhs)} ends in the sign '-', but a number's sign comes first — illegal.")
                out.append("    This reading order is illegal; skip it completely.")
                return None, None, None
    # §N.2 concat check
    out.append(f"  §{n}.2 quick check to see whether any operator is concatenation:")
    concat_results = {}
    qop = p.query[2]
    for op in p.by_op:
        if op in sign_ops:
            out.append(f"    {om[op]}: already ~sub by its output sign, skip the concat check.")
            continue
        ct = cg.check_concat(op, p.examples)
        ex = p.by_op[op][0]
        fwd = ''.join(lm[s] for s in [ex.d1, ex.d2, ex.d3, ex.d4])
        rev = ''.join(lm[s] for s in [ex.d3, ex.d4, ex.d1, ex.d2])
        outp = ''.join('-' if s in op_set else lm.get(s, s) for s in ex.rhs)
        if ct:
            out.append(f"    {om[op]}: a∥b={fwd}, b∥a={rev}. RHS={outp}. {CDISP[ct]} matches -> so lock {om[op]} = {CDISP[ct]}.")
            concat_results[op] = ct
            if op == qop:   # the QUERY operator is concatenation -> the result is just the concatenation of its operands; the digit values never enter the answer, so stop the search here
                out.append(f"    Note: {om[qop]} is the QUERY operator, and it has been locked as {CDISP[ct]}. At this point other unknowns are irrelevant, because the answer is simply the concatenation of the query operands. Proceed to answer directly.")
                return 'CONCAT_SHORT', None, concat_results
        else:
            out.append(f"    {om[op]}: a∥b={fwd}, b∥a={rev}. RHS={outp}. Neither matches, so {om[op]} is not concat.")
    arith_ops = [op for op in p.by_op if op not in concat_results]
    out.append(f"    {op_state_after_concat(p, sign_ops, concat_results)}")
    if not arith_ops:
        out.append("    Every operator is concatenation; nothing remains to search.")
    # §N.3 digit-count prune + distinct-operator rule
    if arith_ops:
        out.append(f"  §{n}.3 prune solution trees by the RHS digit-count before entering:")
        cand = prune_candidates(p, op_set, sign_ops, concat_results, om, out, n)
        head4 = f"  §{n}.4 narrow the operand domains before branching:" if SPLIT_45 else f"  §{n}.4 enter the tree and search for an operator and digit assignment:"
        out.append(head4)
        A, variants, _ = search_reading(p, mode, n, op_set, concat_results, sign_ops, cand, out.append)
        return A, variants, concat_results
    return {}, {}, concat_results


def _conc_brace(p, variants, concat_results, A):
    parts = []
    for op in p.op_syms:
        ol = p.op_map[op]
        if variants and op in variants:   parts.append(f"{ol} = {SYMB[variants[op]]}")
        elif op in (concat_results or {}): parts.append(f"{ol} = {CDISP[concat_results[op]]}")
        else:                              parts.append(f"{ol} = unknown")
    if A:
        for s in sorted(p.digit_syms, key=lambda x: p.letter_map[x]):
            parts.append(f"{p.letter_map[s]}={A[s]}")
    return "{" + ", ".join(parts) + "}"

def _resolved_count(p, variants, concat_results, A):
    nop = len([op for op in p.op_syms if (variants and op in variants) or op in (concat_results or {})])
    ndig = sum(1 for s in p.digit_syms if A and s in A)   # count pinned symbols (partial under QUERY_MIN)
    return nop + ndig

def _query_min_answered(p, A, variants, concat_results, mode, op_set):
    """Can the query be answered from this (possibly partial) assignment? (operator locked, query operands + result-digit symbols pinned)"""
    if not A or A == 'CONCAT_SHORT' or not (variants and p.query[2] in variants): return False
    q = p.query
    if any(s not in A for s in (q[0], q[1], q[3], q[4])): return False
    a = 10*A[q[0]]+A[q[1]] if mode == 'standard' else 10*A[q[1]]+A[q[0]]
    b = 10*A[q[3]]+A[q[4]] if mode == 'standard' else 10*A[q[4]]+A[q[3]]
    r = cg.apply_op(variants[q[2]], a, b)
    val2sym = {v: s for s, v in A.items()}
    return all(int(c) in val2sym for c in str(abs(r)))

def emit_answer(p, A, variants, concat_results, mode, op_set, out):
    import _gen_solutions as gs
    q = p.query; d1, d2, zs, d3, d4 = q[0], q[1], q[2], q[3], q[4]
    zl = p.op_map[zs]
    fmt = sign_fmt(p, op_set)                                   # sign convention (front/back), read off the examples
    rdir_short = 'leftward fully' if fmt == 'suffix' else 'leftward on digit only'
    dig2sym = {v: k for k, v in A.items()} if A else {}
    s2l = {**p.letter_map, **p.op_map}
    lf = lambda ans: ''.join(s2l.get(c, c) for c in ans)
    # concat query
    if zs in (concat_results or {}):
        k = concat_results[zs]; ans = (d1+d2+d3+d4) if k == 'fwd' else (d3+d4+d1+d2)
        o1, o2 = lf(d1+d2), lf(d3+d4); conc = f"{o1}∥{o2}" if k == 'fwd' else f"{o2}∥{o1}"
        out.append(f"Now solve the QUERY {zl}({o1}, {o2}), applying {zl} = {CDISP[k]},")
        out.append(f"{zl}({o1}, {o2}) = {conc} = {lf(ans)}; map the letters back to symbols, so the answer is {ans}")
        return ans
    # arithmetic query (seen)
    if zs in p.by_op and variants and zs in variants:
        var = variants[zs]
        a = 10*A[d1]+A[d2] if mode == 'standard' else 10*A[d2]+A[d1]
        b = 10*A[d3]+A[d4] if mode == 'standard' else 10*A[d4]+A[d3]
        return _emit_query_tail(out, p, zl, a, b, var, SYMB[var], d1, d2, d3, d4, mode, fmt, rdir_short, dig2sym, zs, op_set, A)
    # unseen query operator — guess by distinctness + symbol-canonical prior, then the fixed family order
    if A:
        ARITH = ('noisy_mul', 'noisy_add', 'noisy_subtraction')
        a = 10*A[d1]+A[d2] if mode == 'standard' else 10*A[d2]+A[d1]
        b = 10*A[d3]+A[d4] if mode == 'standard' else 10*A[d4]+A[d3]
        out.append(f"Query operator {zl} only appears in QUERY, so I need to guess it.")
        present = []
        for f in [cg._OP_FAMILY.get(v) for v in (variants or {}).values()]:
            if f in ARITH and f not in present: present.append(f)
        if not present:                                          # no arithmetic operator solved -> guess concat
            ans = d1+d2+d3+d4; o1, o2 = lf(d1+d2), lf(d3+d4)
            out.append(f"No operator here can be solved arithmetically, so I guess the unseen operator {zl} is ~concat. Let me use {zl} = a∥b.")
            out.append(f"Now solve the QUERY {zl}({o1}, {o2}), applying {zl} = a∥b,")
            out.append(f"{zl}({o1}, {o2}) = {o1}∥{o2} = {lf(ans)}; map the letters back to symbols, so the answer is {ans}")
            return ans
        canon = CANON.get(zs)                                    # + -> ~add, - -> ~sub, * -> ~mul; None for other symbols
        ford = []
        for f in ([canon] if canon else []) + list(FAM_ORDER):   # canonical family FIRST, then the global default order
            if f in ARITH and f not in ford: ford.append(f)
        free = [f for f in ford if f not in present]
        chosen = (free or ford)[0]; base = gs.BASE.get(chosen, 'add'); tg = lambda f: TAG.get(f, f)
        prior = (f"the symbol '{zs}' usually means {tg(canon)}, so {tg(chosen)} is the pick" if (canon and chosen == canon)
                 else f"{tg(chosen)} comes first in the default order, so it is the pick")
        intro = f"At least one operator here can be solved arithmetically, so I'll guess {zl} is an arithmetic operator too"
        pres = " and ".join(tg(f) for f in present); hv = "has" if len(present) == 1 else "have"
        if len(free) == 1:
            out.append(f"{intro}. Since {pres} {hv} appeared in the puzzle, by the distinct operator rule {zl} must be the remaining operation, {tg(chosen)}.")
        elif len(free) >= 2:
            out.append(f"{intro}. Since {pres} {hv} appeared in the puzzle, by the distinct operator rule {zl} is {' or '.join(tg(f) for f in free)}; {prior}.")
        else:
            out.append(f"{intro}. Since {pres} {hv} appeared in the puzzle, every arithmetic operation is already used, so I take the most likely, {tg(chosen)}.")
        return _emit_query_tail(out, p, zl, a, b, base, SYMB[base], d1, d2, d3, d4, mode, fmt, rdir_short, dig2sym, zs, op_set, A)
    # no reading solved — blind concat fallback
    ans = d1+d2+d3+d4
    out.append(f"The examples aren't arithmetically solvable; guess {zl} = a∥b; answer = {ans}")
    return ans

def _poly(p, d1, d2, mode):
    a, b = (p.letter_map[d1], p.letter_map[d2]) if mode == 'standard' else (p.letter_map[d2], p.letter_map[d1])
    return a + b   # compact tens-first operand, same form as §N.1 / §N.4

def _finalize(L, letters):
    s = "\n".join(L).replace("RHS", "rhs").replace("LHS", "lhs")      # rhs/lhs are 1 token each vs 2 for the uppercase forms
    s = s.replace(" ∈ ", " in ").replace("∈", " in ").replace("·", "×")   # 'in' for membership; '×' is the single multiplication sign everywhere (no '·')
    if not letters: return s
    pat = re.compile(f"[{''.join(sorted(letters))}0-9]{{2,}}")        # a multi-symbol run (operand/RHS: A B, 3 B, D E F G) built from PUZZLE letters/digits; keyed on the letters so EX1/QUERY are never touched
    sp = lambda m: ' '.join(m.group(0)) if any(c in letters for c in m.group(0)) else m.group(0)   # space it only if it carries a letter; a pure-number run stays joined
    return ''.join(part if part.startswith('\\boxed') else pat.sub(sp, part)   # never touch the boxed answer (must stay exact for the metric)
                   for part in re.split(r'(\\boxed\{[^}]*\})', s))


def gen_cot(name):
    p = cg.parse(f'{name}/question.txt'); op_set = set(p.op_syms)
    L, sign_ops = front_matter(p)
    # Unseen query operator: the two checks run SEQUENTIALLY, mirroring how the log is produced.
    #   STEP 1 — is the query operator unseen?  If so, LOG it (always, before §1).
    #   STEP 2 — only if unseen, apply rule #1 (no example result reaches 4 digits => guess ~concat, answer now).
    #            If rule #1 is not met, log that and fall through into the search (an arithmetic guess needs the digits).
    qop = p.query[2]
    if qop not in p.by_op:                                       # STEP 1: unseen query operator — log it (always)
        zl = p.op_map[qop]
        L.append(f"Note: Query operator {zl} does not appear in the examples, so solving the whole puzzle can be a waste of time, because we need to guess the Query operator anyway.")
        L.append("We know that concatenating two 2-digit numbers always produces a 4-digit RHS, and that the operators are usually distinct. A good signal is therefore whether any example has a 4-digit RHS; if none does, there is a good chance the query operator is concatenation.")
        rhs_lens = [len([s for s in ex.rhs if s not in op_set]) for ex in p.examples]
        if RULE1_CONCAT and max(rhs_lens) < 4:                   # STEP 2: rule #1 met — no 4-digit RHS => guess ~concat
            lm = p.letter_map
            o1c, o2c = p.query[:2], p.query[3:5]
            o1l, o2l = ''.join(lm[c] for c in o1c), ''.join(lm[c] for c in o2c)
            ans = o1c + o2c; ansl = o1l + o2l
            L.append(f"There is no 4-digit RHS in any example, so {zl} is most likely concatenation. I'll guess {zl} = a∥b.")
            L.append(f"Now solve the QUERY {zl}({o1l}, {o2l}), applying {zl} = a∥b,")
            L.append(f"{zl}({o1l}, {o2l}) = {o1l}∥{o2l} = {ansl}; map the letters back to symbols, so the answer is {ans}")
            L.append("")
            L.append("I will now return the answer in \\boxed{}, the answer is")
            L.append(f"\\boxed{{{ans}}}")
            return _finalize(L, set(p.letter_map.values()))
        if max(rhs_lens) >= 4:                                       # STEP 2: rule #1 not met — a 4-digit RHS makes concat risky
            ex4 = next(i for i, n in enumerate(rhs_lens, 1) if n >= 4)
            L.append(f"There is a 4-digit RHS in EX{ex4}, so guessing that {zl} is concatenation is risky. I'll solve the puzzle and see what happens.")
        else:                                                        # only reachable with RULE1_CONCAT disabled (ablation)
            L.append(f"I'll solve the puzzle and see what happens.")
        L.append("")
    L.append("Now start to traverse the solution tree, reading order = [rightward, leftward]:")
    Y = len(p.digit_syms) + len(p.op_syms)
    example_syms = set()
    for e in p.examples: example_syms |= {e.d1, e.d2, e.d3, e.d4, e.op, *e.rhs}
    # unknowns that appear ONLY in the query (the unseen query operator, query-only operands) can never be
    # resolved by ANY reading — so they must not count toward "complete" or trigger a wasted second reading.
    unresolvable = [s for s in list(p.digit_syms) + list(p.op_syms) if s not in example_syms]
    Y_solv = Y - len(unresolvable)
    win = None  # (mode, A, variants, concat_results)
    short_circuit = False
    legal_readings = []
    for mode, n in [('standard', 1), ('little_endian', 2)]:
        out = []
        A, variants, concat_results = emit_reading(p, mode, n, op_set, sign_ops, out)
        L += out
        if A == 'CONCAT_SHORT':     # query op is concat -> short-circuit straight to the answer
            brace = "{" + ", ".join(f"{p.op_map[op]} = {CDISP[concat_results[op]] if op in concat_results else 'unknown'}" for op in p.op_syms) + "}"
            nres = sum(1 for op in p.op_syms if op in concat_results)
            L.append(f"  Conclusion of step §{n}: {brace}, {nres} of {len(p.op_syms)} resolved.")
            L.append("")
            win = (mode, {}, {}, concat_results); short_circuit = True
            break
        illegal = (A is None and variants is None and concat_results is None)
        if illegal:
            legal_readings.append((n, mode, 'illegal', 0))
            continue
        if QUERY_MIN and _query_min_answered(p, A, variants, concat_results, mode, op_set):
            nres = _resolved_count(p, variants, concat_results, A)
            L.append(f"  Conclusion of step §{n}: the query is answerable now ({nres} of {Y} unknowns resolved; the rest are not needed). This is the answer.")
            L.append("")
            win = (mode, A, variants, concat_results); break
        X = _resolved_count(p, variants, concat_results, A)
        legal_readings.append((n, mode, (A, variants, concat_results), X))
        complete = X >= Y_solv                                    # everything the examples CAN pin is pinned
        tail = ""
        if n == 1:
            if complete and unresolvable:
                roles = ", ".join(p.op_map.get(s) or p.letter_map[s] for s in unresolvable)
                verb = "appears" if len(unresolvable) == 1 else "appear"
                them = "it" if len(unresolvable) == 1 else "them"
                tail = f" {roles} {verb} only in the query, so no reading can determine {them} — this reading is already as complete as possible. This is the answer."
            elif complete:
                tail = " This is the answer."
            else:
                tail = " Need to try reading leftward."
        L.append(f"  Conclusion of step §{n}: {_conc_brace(p, variants, concat_results, A)}, {X} of {Y} resolved.{tail}")
        L.append("")
        if n == 1 and complete:
            win = (mode, A, variants, concat_results); break
    # confirm-reading label carries the leftward mode (matches the §N reading header): rightward / leftward fully / leftward on digit only
    conf_reading = lambda m: 'rightward' if m == 'standard' else ('leftward fully' if sign_fmt(p, op_set) == 'suffix' else 'leftward on digit only')
    if short_circuit:
        mode, A, variants, concat_results = win
        qop = p.query[2]; ro = conf_reading(mode)
        asg = ", ".join(f"{p.op_map[op]} = {CDISP[concat_results[op]] if op in (concat_results or {}) else 'unknown'}" for op in p.op_syms)
        L.append(f"Summary: Query operator {p.op_map[qop]} = {CDISP[concat_results[qop]]}, no need to solve the other unknowns; Confirm reading order = {ro}, {asg}")
    elif win is None:
        # reading-selection gate cascade (ported from the numeric solver): a reading that solves the QUERY
        # operator is preferred over one that merely resolves MORE total unknowns — the query answer needs the
        # query operator, so a reading that didn't solve it can't answer even if it pinned more digits.
        qop = p.query[2]; ql = p.op_map[qop]
        res_by_n = {n: (mode, res, X) for (n, mode, res, X) in legal_readings}
        legal_ns  = [n for n in (1, 2) if n in res_by_n and res_by_n[n][1] != 'illegal']
        if legal_ns:
            def qsolved(res):
                _, variants, concat_results = res
                return bool((variants and qop in variants) or (concat_results and qop in concat_results))
            if len(legal_ns) == 1:                                   # the other reading is illegal
                n = legal_ns[0]
                illn = [m for m in (1, 2) if m != n]
                verdict = f"§{illn[0]} is illegal, so §{n} is the only legal reading."
            else:                                                    # both legal -> 3-gate cascade
                r1, r2 = res_by_n[1][1], res_by_n[2][1]
                q1, q2 = qsolved(r1), qsolved(r2)
                n1c, n2c = res_by_n[1][2], res_by_n[2][2]
                if q1 != q2:                                         # Gate 1: only one reading solved the query operator
                    n = 2 if q2 else 1; lo = 1 if q2 else 2
                    verdict = f"§{n} solved the QUERY operator {ql}, which §{lo} did not, so §{n} is preferred."
                else:                                                # Gate 1 inconclusive -> count
                    pre = ("Both readings solve" if q1 else "Neither reading solves") + f" the QUERY operator {ql};"
                    if n1c != n2c:                                   # Gate 2: more unknowns
                        n = 2 if n2c > n1c else 1
                        verdict = pre + f" §{n} resolved more unknowns, so §{n} is preferred."
                    else:                                            # Gate 3: tie -> default reading §1
                        n = 1
                        verdict = pre + " §1 and §2 resolved the same number of unknowns, so §1 is preferred."
            mode, res, X = res_by_n[n]; A, variants, concat_results = res
            win = (mode, A, variants, concat_results)
            ro = conf_reading(mode)
            asg = ", ".join(f"{p.op_map[op]} = {SYMB[variants[op]] if variants and op in variants else (CDISP[concat_results[op]] if op in (concat_results or {}) else 'unknown')}" for op in p.op_syms)
            L.append(f"Summary: {verdict} Confirm reading order = {ro}, " + asg)
        else:
            win = ('standard', None, None, None)
            L.append("Summary: no legal reading solved the puzzle.")
    else:
        mode, A, variants, concat_results = win
        L.append(f"Summary: Confirm reading order = {conf_reading(mode)}, " + ", ".join(f"{p.op_map[op]} = {SYMB[variants[op]] if variants and op in variants else (CDISP[concat_results[op]] if op in (concat_results or {}) else 'unknown')}" for op in p.op_syms))
    L.append("")
    mode, A, variants, concat_results = win
    ans = emit_answer(p, A, variants, concat_results or {}, mode, op_set, L)
    L.append("")
    L.append("I will now return the answer in \\boxed{}, the answer is")
    L.append(f"\\boxed{{{ans}}}")
    return _finalize(L, set(p.letter_map.values()))


POOL23 = sorted('''!"#$%&'()/:<>?@[\\]^`{|}''')

def _fill_missing(A, op_set, ans, missing):
    """Replace each '_' (a digit with no seen symbol) by the first unused operand symbol and narrate it
    (the approved missing-digit format: full alphabet, masked row, first-of-remaining)."""
    used = set(A.keys()) | set(op_set)
    cand = [s for s in POOL23 if s not in used]
    guess = {}; ci = 0
    for d in dict.fromkeys(missing):
        guess[d] = cand[ci] if ci < len(cand) else '?'; ci += 1
    o = []; mi = 0
    for ch in ans:
        if ch == '_': o.append(guess[missing[mi]]); mi += 1
        else: o.append(ch)
    filled = ''.join(o)
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
             f"Since I have to choose, I will just take the first one that is left: {pairs}, so the answer is {filled}"],
            filled)


if __name__ == '__main__':
    import sys
    print(gen_cot(sys.argv[1] if len(sys.argv) > 1 else 'arithmetic_a4e4ec1d'))
