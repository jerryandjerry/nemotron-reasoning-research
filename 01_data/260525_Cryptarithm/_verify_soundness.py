#!/usr/bin/env python3
"""Soundness verification for the 3 proposed structural-pattern rules.

For each puzzle we read the GOLD assignment (tree_solution.txt: digit map, variants,
concat ops, reading mode). For every ~add and signed-~sub example we form the
net-coefficient identity  Sum_t c_t * v_t = (operand sum/diff) - RHS  and check the
claimed bound. We also hunt for the worst-case residual to see if any ±2-noise variant
in the GOLD vocabulary breaks a claim.
"""
import os, re, glob, collections
from pathlib import Path
import cot_generator as cg

DATA = Path('/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260525_Cryptarithm')

# variant -> family
ADD_VARS = {'add','add_p1','add_m1','add_p2','add_m2'}
MUL_VARS = {'mul','mul_p1','mul_m1','mul_p2','mul_m2'}
SUB_SIGNED_VARS = {'sub_signed','sub_signed_p1','sub_signed_m1','sub_signed_p2','sub_signed_m2'}
RSUB_VARS = {'rsub_signed'}
ABS_VARS = {'absdiff','absdiff_p1','absdiff_m1','absdiff_p2','absdiff_m2','neg_absdiff',
            'neg_absdiff_p1','neg_absdiff_m1','neg_absdiff_p2','neg_absdiff_m2'}

def parse_solution(d, op_syms):
    """Returns (asgn, variants{op_sym->variant}, concat{op_sym->kind}, mode).
       The solution file keys variants by x/y/z letters assigned in op_syms appearance order."""
    f = d/'tree_solution.txt'
    if not f.exists(): return None
    txt = f.read_text()
    asgn = {}
    for m in re.finditer(r'^\s*(\S+) \([A-Z]\) -> (\d+)\s*$', txt, re.M):
        asgn[m.group(1)] = int(m.group(2))
    # letter (x,y,z) -> op_sym, by appearance order
    letter_to_op = {chr(ord('x')+i): op for i, op in enumerate(op_syms)}
    variants = {}
    mv = re.search(r'Operator variants:\s*(.*)', txt)
    if mv:
        for part in mv.group(1).split(','):
            part=part.strip()
            if '=' in part:
                key,var = part.split('=',1); key=key.strip()
                op = letter_to_op.get(key, key)
                variants[op] = var.strip()
    concat = {}
    mc = re.search(r'Concat operators:\s*(.*)', txt)
    if mc:
        for part in mc.group(1).split(','):
            part=part.strip()
            if '=' in part:
                op,var = part.split('=',1); concat[op.strip()] = var.strip()
    mode = 'standard'
    mm = re.search(r'Reading order:\s*(\S+)', txt)
    if mm: mode = mm.group(1).strip()
    return asgn, variants, concat, mode

def operand_val(d1,d2,asgn,mode):
    a,b = asgn[d1],asgn[d2]
    return 10*a+b if mode=='standard' else 10*b+a

def rhs_signed_int(rhs, asgn, mode, op_set):
    sign=1; chars=list(rhs)
    if chars and chars[0] in op_set: sign=-1; chars=chars[1:]
    elif chars and chars[-1] in op_set: sign=-1; chars=chars[:-1]
    n=len(chars)
    co = [10**(n-1-i) for i in range(n)] if mode=='standard' else [10**i for i in range(n)]
    v=0
    for c,s in zip(co,chars): v += c*asgn[s]
    return sign*v

def net_coeffs(e, mode, op_set):
    """Return dict sym -> net coefficient c_t for the identity (operand1+operand2) - RHS
       (the ~add net form). Operand symbol weights are 10 (tens) /1 (units) by reading frame.
       RHS coeffs depend on length & sign. Returns (coeffs, rhs_sign)."""
    coeffs = collections.defaultdict(int)
    # operand contributions: both operands ADDED (this is the add net form)
    for (d1,d2) in [(e.d1,e.d2),(e.d3,e.d4)]:
        tw,uw = (10,1)
        if mode=='standard': t,u = d1,d2
        else:               t,u = d2,d1
        coeffs[t]+=tw; coeffs[u]+=uw
    # rhs
    chars=list(e.rhs); sign=1
    if chars and chars[0] in op_set: sign=-1; chars=chars[1:]
    elif chars and chars[-1] in op_set: sign=-1; chars=chars[:-1]
    n=len(chars)
    co=[10**(n-1-i) for i in range(n)] if mode=='standard' else [10**i for i in range(n)]
    for c,s in zip(co,chars):
        coeffs[s]-= sign*c
    return coeffs

def net_coeffs_sub(e, mode, op_set, reverse):
    """net form for a-b (reverse=False) or b-a (reverse=True): operand1 - operand2 (or swapped) - RHS."""
    coeffs = collections.defaultdict(int)
    op1 = (e.d1,e.d2); op2=(e.d3,e.d4)
    if reverse: op1,op2 = op2,op1
    for sgn,(d1,d2) in [(1,op1),(-1,op2)]:
        if mode=='standard': t,u=d1,d2
        else:               t,u=d2,d1
        coeffs[t]+=sgn*10; coeffs[u]+=sgn*1
    chars=list(e.rhs); rsign=1
    if chars and chars[0] in op_set: rsign=-1; chars=chars[1:]
    elif chars and chars[-1] in op_set: rsign=-1; chars=chars[:-1]
    n=len(chars)
    co=[10**(n-1-i) for i in range(n)] if mode=='standard' else [10**i for i in range(n)]
    for c,s in zip(co,chars):
        coeffs[s]-= rsign*c
    return coeffs

def main():
    dirs = sorted([Path(p).parent for p in glob.glob(str(DATA/'*/question.txt'))])
    add_residuals=[]; sub_residuals=[]; rsub_residuals=[]
    add_examples=0; sub_examples=0; rsub_examples=0
    add_violations=[]; sub_violations=[]; rsub_violations=[]
    n_solved=0
    # also gather: how many puzzles have a symbol on both op-and-rhs / both operands
    shared_op_rhs=0; shared_operands=0; total=0
    map_mismatch=0
    for d in dirs:
        try:
            p = cg.parse(str(d/'question.txt'))
        except Exception as ex:
            continue
        sol = parse_solution(d, p.op_syms)
        if sol is None: continue
        asgn,variants,concat,mode = sol
        if not asgn: continue
        op_set=set(p.op_syms)
        n_solved+=1
        total+=1
        # structural stats
        has_or=False; has_oo=False
        for e in p.examples:
            rhs_syms = set(c for c in e.rhs if c not in op_set)
            op_syms_e = {e.d1,e.d2,e.d3,e.d4}
            if op_syms_e & rhs_syms: has_or=True
            # symbol in both operands:
            if (e.d1 in (e.d3,e.d4)) or (e.d2 in (e.d3,e.d4)): has_oo=True
        if has_or: shared_op_rhs+=1
        if has_oo: shared_operands+=1
        for e in p.examples:
            if e.op in concat: continue
            var = variants.get(e.op)
            if var is None: continue
            # independent cross-check: does the puzzle-level variant actually reproduce THIS example?
            a = operand_val(e.d1,e.d2,asgn,mode); b = operand_val(e.d3,e.d4,asgn,mode)
            rv = rhs_signed_int(e.rhs,asgn,mode,op_set)
            fn = cg.VARIANT_FN.get(var)
            matches = (fn is not None and fn(a,b)==rv)
            if not matches:
                map_mismatch += 1
            if var in ADD_VARS:
                add_examples+=1
                coeffs = net_coeffs(e, mode, op_set)
                resid = sum(c*asgn[s] for s,c in coeffs.items())
                add_residuals.append(resid)
                if not (-2<=resid<=2):
                    add_violations.append((str(d.name), e.lhs, e.rhs, var, resid))
            elif var in SUB_SIGNED_VARS:
                sub_examples+=1
                coeffs = net_coeffs_sub(e, mode, op_set, reverse=False)
                resid = sum(c*asgn[s] for s,c in coeffs.items())
                sub_residuals.append(resid)
                if not (-2<=resid<=2):
                    sub_violations.append((str(d.name), e.lhs, e.rhs, var, resid))
            elif var in RSUB_VARS:
                rsub_examples+=1
                coeffs = net_coeffs_sub(e, mode, op_set, reverse=True)
                resid = sum(c*asgn[s] for s,c in coeffs.items())
                rsub_residuals.append(resid)
                if not (-2<=resid<=2):
                    rsub_violations.append((str(d.name), e.lhs, e.rhs, var, resid))
    print(f"puzzles with gold solution parsed: {n_solved}")
    print(f"variant-map / example mismatches (sanity, should be 0): {map_mismatch}")
    print(f"shared op<->rhs symbol present in {shared_op_rhs}/{total} = {100*shared_op_rhs/total:.1f}%")
    print(f"shared-across-operands present in {shared_operands}/{total} = {100*shared_operands/total:.1f}%")
    print()
    print("=== ~add net identity Sum c_t v_t (claim: in [-2,2]) ===")
    print(f"  add examples checked: {add_examples}")
    print(f"  residual range: [{min(add_residuals)},{max(add_residuals)}]" if add_residuals else "  none")
    print(f"  residual distribution: {collections.Counter(add_residuals)}")
    print(f"  VIOLATIONS (|resid|>2): {len(add_violations)}")
    for v in add_violations[:10]: print("    ", v)
    print()
    print("=== signed-sub a-b net identity (claim: in [-2,2]; 0 for exact sub_signed) ===")
    print(f"  sub_signed examples checked: {sub_examples}")
    print(f"  residual range: [{min(sub_residuals)},{max(sub_residuals)}]" if sub_residuals else "  none")
    print(f"  residual distribution: {collections.Counter(sub_residuals)}")
    print(f"  VIOLATIONS (|resid|>2): {len(sub_violations)}")
    for v in sub_violations[:10]: print("    ", v)
    print()
    print("=== rsub (b-a) net identity (claim: =0 exactly for rsub_signed) ===")
    print(f"  rsub_signed examples checked: {rsub_examples}")
    print(f"  residual range: [{min(rsub_residuals)},{max(rsub_residuals)}]" if rsub_residuals else "  none")
    print(f"  residual distribution: {collections.Counter(rsub_residuals)}")
    print(f"  VIOLATIONS (|resid|>2): {len(rsub_violations)}")
    for v in rsub_violations[:10]: print("    ", v)

if __name__=='__main__':
    main()
