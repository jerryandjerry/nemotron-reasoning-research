#!/usr/bin/env python3
"""Soundness verification for the carry-column thread rules, against GOLD tree_solution.txt.

For each of the 800 puzzles we load the TRUE digit map + variant + reading mode, then for each
proposed rule we check whether the rule (applied to the true operand/result digits) would ever
remove the TRUE value of a symbol (an UNSOUND removal). We also try to construct a counterexample.
"""
import os, re, glob
import cot_generator as cg

PREFIXES = ('arithmetic','little_endian','mixed_concat','pure_concat','query_unseen_concat','mixed_concat_little_endian')

def all_dirs():
    out=[]
    for d in sorted(os.listdir('.')):
        if os.path.isdir(d) and any(d.startswith(p) for p in PREFIXES) and os.path.exists(d+'/question.txt'):
            out.append(d)
    return out

# ---- complete variant evaluator (covers gold absdiff_pK / absdiff_mK that VARIANT_FN lacks) ----
def eval_variant(var, a, b):
    if var in cg.VARIANT_FN:
        return cg.VARIANT_FN[var](a, b)
    # absdiff_pK / absdiff_mK / rsub variants etc.
    m = re.match(r'(absdiff|sub_signed|rsub_signed|neg_absdiff|add|mul)(?:_([pm])(\d))?$', var)
    if not m:
        raise ValueError('unknown variant '+var)
    base, sgn, k = m.group(1), m.group(2), m.group(3)
    off = 0
    if sgn: off = int(k) * (1 if sgn=='p' else -1)
    if base=='absdiff': v=abs(a-b)
    elif base=='sub_signed': v=a-b
    elif base=='rsub_signed': v=b-a
    elif base=='neg_absdiff': v=-abs(a-b)
    elif base=='add': v=a+b
    elif base=='mul': v=a*b
    return v+off

def infer_family(a, b, R):
    """Gold-grounded family inference from true operands a,b and true result R.
    sub family: R matches absdiff/sub_signed/rsub/neg_absdiff within +-2.
    add: R within [a+b-2, a+b+2].  mul: R within [a*b-2, a*b+2].
    Returns the UNIQUE matching family, or None if 0 or >1 match (ambiguous/exotic)."""
    fams=set()
    if a*b-2 <= R <= a*b+2: fams.add('noisy_mul')
    if a+b-2 <= R <= a+b+2: fams.add('noisy_add')
    ad=abs(a-b)
    sub_vals=set()
    for base in (ad, a-b, b-a, -ad):
        for k in (-2,-1,0,1,2):
            sub_vals.add(base+k)
    if R in sub_vals: fams.add('noisy_subtraction')
    if len(fams)==1: return next(iter(fams))
    return None   # ambiguous (e.g. tiny operands where add~mul overlap) or exotic -> skip

def fam_of_variant(var):
    if var.startswith('mul'): return 'noisy_mul'
    if var.startswith('add'): return 'noisy_add'
    return 'noisy_subtraction'   # absdiff / sub_signed / rsub_signed / neg_absdiff

def load_gold(name):
    t = open(name+'/tree_solution.txt').read()
    if 'not solvable' in t:
        return None
    digit={}
    for ln in t.splitlines():
        m=re.match(r'\s*(\S+) \([A-Z]+\) -> (\d+)', ln)
        if m: digit[m.group(1)] = int(m.group(2))
    variants={}
    m=re.search(r'Operator variants: (.+)', t)
    if m:
        for kv in m.group(1).split(','):
            kv=kv.strip()
            if '=' in kv:
                op,var=kv.split('=',1); variants[op.strip()]=var.strip()
    concat={}
    m=re.search(r'Concat operators: (.+)', t)
    if m:
        for kv in m.group(1).split(','):
            kv=kv.strip()
            if '=' in kv:
                op,c=kv.split('=',1)
                concat[op.strip()]= 'fwd' if 'fwd' in c else 'rev'
    mode = re.search(r'Reading order: (\w+)', t).group(1)
    return digit, variants, concat, mode

def opval(d1,d2,A,mode):
    return 10*A[d1]+A[d2] if mode=='standard' else 10*A[d2]+A[d1]

def units_syms(e, mode, op_set):
    u1 = e.d2 if mode=='standard' else e.d1
    u2 = e.d4 if mode=='standard' else e.d3
    rch=[c for c in e.rhs if c not in op_set]
    if not rch: return None
    ur = rch[-1] if mode=='standard' else rch[0]
    return u1,u2,ur

def msb_sym(e, mode, op_set):
    rch=[c for c in e.rhs if c not in op_set]
    if not rch: return None
    return rch[0] if mode=='standard' else rch[-1]

def leadpair(e, mode):
    return (e.d1,e.d3) if mode=='standard' else (e.d2,e.d4)

# ============================ SOUNDNESS CHECKS ============================
def main():
    names = all_dirs()
    # counters per rule
    r1_units_examples=0; r1_units_violations=[]      # units congruence mod10
    r2_add_cases=0; r2_add_fire=0; r2_add_viol=[]    # add msb=1, guarded
    r2_add_guard_blocked=0; r2_add_unguarded_viol=[] # without guard
    r3_mul_cases=0; r3_mul_fire=0; r3_mul_viol=[]    # mul msb!=9 guarded
    r3_mul_unguarded_viol=[]
    r5_sub_examples=0; r5_sub_violations=[]; r5_bandsizes=[]  # sub units band
    n_puz=0
    for name in names:
        p = cg.parse(name+'/question.txt'); op_set=set(p.op_syms)
        g = load_gold(name)
        if g is None: continue
        digit, variants, concat, mode = g
        n_puz+=1
        A = digit
        for e in p.examples:
            op=e.op
            if op in concat: continue       # concat example, no arithmetic
            # need all operand+result syms present in A
            esyms=(e.d1,e.d2,e.d3,e.d4,*[c for c in e.rhs if c not in op_set])
            if not all(s in A for s in esyms): continue
            a = opval(e.d1,e.d2,A,mode); b=opval(e.d3,e.d4,A,mode)
            R = cg.rhs_int(e.rhs, A, mode, op_set)   # TRUE result from gold RHS digits
            if R is None: continue
            fam = infer_family(a, b, R)
            if fam is None: continue   # exotic / unmatched operator -> skip (no supported-family claim)
            # ---- RULE 1: units congruence mod10 for add/mul ----
            if fam in ('noisy_mul','noisy_add'):
                us = units_syms(e, mode, op_set)
                if us:
                    u1,u2,ur = us
                    U1,U2,Ur = A[u1],A[u2],A[ur]
                    base = (U1*U2) if fam=='noisy_mul' else (U1+U2)
                    r1_units_examples+=1
                    if (base - Ur) % 10 not in (0,1,2,8,9):
                        r1_units_violations.append((name, fam, a, b, R, U1,U2,Ur, (base-Ur)%10))
            # ---- RULE 2: add 3-digit MSB = 1 ----
            if fam=='noisy_add':
                rch=[c for c in e.rhs if c not in op_set]
                if len(rch)==3:
                    ms = msb_sym(e, mode, op_set)
                    true_msb = A[ms]
                    # guard: NOT (d1==d2 and d3==d4) i.e. not both operands repeated-symbol
                    guard_repeat = (e.d1==e.d2 and e.d3==e.d4)
                    r2_add_cases+=1
                    if not guard_repeat:
                        r2_add_fire+=1
                        if true_msb != 1:
                            r2_add_viol.append((name,fam,a,b,R,true_msb))
                    else:
                        r2_add_guard_blocked+=1
                    # unguarded soundness: would pin to 1 always
                    if true_msb != 1:
                        r2_add_unguarded_viol.append((name,fam,a,b,R,true_msb,guard_repeat))
            # ---- RULE 3: mul 4-digit MSB != 9 (guard: 4 distinct operand syms) ----
            if fam=='noisy_mul':
                rch=[c for c in e.rhs if c not in op_set]
                if len(rch)==4:
                    ms = msb_sym(e, mode, op_set)
                    true_msb = A[ms]
                    distinct4 = len({e.d1,e.d2,e.d3,e.d4})==4
                    r3_mul_cases+=1
                    if distinct4:
                        r3_mul_fire+=1
                        if true_msb==9:
                            r3_mul_viol.append((name,fam,a,b,R,true_msb))
                    # unguarded
                    if true_msb==9:
                        r3_mul_unguarded_viol.append((name,fam,a,b,R,true_msb,distinct4))
            # ---- RULE 5: sub units band ----
            if fam=='noisy_subtraction':
                us = units_syms(e, mode, op_set)
                if us:
                    u1,u2,ur=us
                    U1,U2,Ur=A[u1],A[u2],A[ur]
                    r5_sub_examples+=1
                    band = set()
                    for base in ((U1-U2)%10, (U2-U1)%10):
                        for k in (-2,-1,0,1,2):
                            band.add((base+k)%10)
                    r5_bandsizes.append(len(band))
                    if Ur not in band:
                        r5_sub_violations.append((name,fam,a,b,R,U1,U2,Ur))
    print("=== puzzles processed:", n_puz)
    print("\n[RULE 1 units mod10 add/mul]")
    print("  example-units examined:", r1_units_examples)
    print("  SOUNDNESS violations (true digit out of band):", len(r1_units_violations))
    for v in r1_units_violations[:10]: print("   ", v)
    print("\n[RULE 2 add 3-digit MSB=1]")
    print("  3-digit add cases:", r2_add_cases, " fired(after guard):", r2_add_fire, " guard-blocked:", r2_add_guard_blocked)
    print("  GUARDED violations (true MSB != 1 where fired):", len(r2_add_viol))
    for v in r2_add_viol[:10]: print("   ", v)
    print("  UNGUARDED violations (true MSB != 1, any):", len(r2_add_unguarded_viol))
    for v in r2_add_unguarded_viol[:10]: print("   ", v)
    print("\n[RULE 3 mul 4-digit MSB!=9]")
    print("  4-digit mul cases:", r3_mul_cases, " fired(after distinct-4 guard):", r3_mul_fire)
    print("  GUARDED violations (true MSB==9 where fired):", len(r3_mul_viol))
    for v in r3_mul_viol[:10]: print("   ", v)
    print("  UNGUARDED violations (true MSB==9, any):", len(r3_mul_unguarded_viol))
    for v in r3_mul_unguarded_viol[:12]: print("   ", v)
    print("\n[RULE 5 sub units band]")
    print("  sub example-units:", r5_sub_examples, " violations:", len(r5_sub_violations))
    import statistics
    if r5_bandsizes:
        print("  band size mean=%.2f min=%d max=%d" % (statistics.mean(r5_bandsizes), min(r5_bandsizes), max(r5_bandsizes)))

if __name__=='__main__':
    main()
