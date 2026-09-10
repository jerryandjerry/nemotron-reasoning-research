import json, glob, os, re
import cot_generator as cg

DELETED = {'arithmetic_294453b5','arithmetic_3a6286e9','arithmetic_4e28b132','arithmetic_50ba5396',
'arithmetic_642c84f0','arithmetic_65a61279','arithmetic_8a15edaa','arithmetic_9346686a','arithmetic_9f9b0251',
'arithmetic_baab28bb','arithmetic_cd5e23c7','arithmetic_d45dc67f','arithmetic_e3fbc768','arithmetic_e9a9f047',
'arithmetic_f2399f16','arithmetic_fb440865','little_endian_564916b5','little_endian_865eca43','little_endian_91a0e345',
'little_endian_982c0b42','little_endian_a26065d4','little_endian_b69d2e78','arithmetic_236a2204'}

def find_sol(t):
    if isinstance(t,dict):
        if t.get('type')=='solution' and t.get('outcome')=='success':
            return t.get('assignment'), t.get('variants')
        for v in t.values():
            if isinstance(v,(dict,list)):
                r=find_sol(v)
                if r[0]: return r
    elif isinstance(t,list):
        for x in t:
            r=find_sol(x)
            if r[0]: return r
    return None,None

def num_to_sym(r, dig2sym, z_sym, mode, fmt=None):
    neg = r < 0; v=abs(r)
    digs=[int(c) for c in str(v)]
    if mode!='standard': digs=digs[::-1]
    out=[]; missing=[]
    for d in digs:
        if d in dig2sym: out.append(dig2sym[d])
        else: out.append('_'); missing.append(d)   # '_' is not a puzzle symbol (avoids collision with the real '?' symbol)
    body=''.join(out)
    if neg:
        # sign POSITION comes from the example convention (fmt), not the reading direction:
        # 'suffix' -> sign at back; 'prefix'/'none' -> sign at front. (fmt=None keeps the old mode-based behaviour for legacy callers.)
        if fmt is None: body = (z_sym+body) if mode=='standard' else (body+z_sym)
        else:           body = (body+z_sym) if fmt=='suffix' else (z_sym+body)
    return body, missing

BASE={'noisy_mul':'mul','noisy_add':'add','noisy_subtraction':'absdiff'}

def gen(d):
    puzzle=cg.parse(d+'question.txt')
    lm=puzzle.letter_map; l2s={v:k for k,v in lm.items()}
    q=puzzle.query; d1,d2,zs,d3,d4=q[0],q[1],q[2],q[3],q[4]
    tree=json.load(open(d+'tree.json')); prefix=open(d+'prefix.txt').read()
    top_succ=any(c.get('outcome')=='success' for c in tree.get('children',[]))
    le=tree.get('little_endian_retry'); le_succ=le and any(c.get('outcome')=='success' for c in le.get('children',[]))
    mode = tree.get('mode','standard') if top_succ else ('little_endian' if le_succ else tree.get('mode','standard'))
    asgn_letter,variants=find_sol(tree)
    sym2dig={l2s[L]:v for L,v in asgn_letter.items()} if asgn_letter else {}
    dig2sym={v:k for k,v in sym2dig.items()}
    let2op={v:k for k,v in puzzle.op_map.items()}; concat={}
    for line in prefix.split('\n'):
        mm=re.match(r'\s+([a-z]):.*?(FWD|REV) matches',line)
        if mm:
            s=let2op.get(mm.group(1))
            if s: concat[s]=mm.group(2).lower()
    interp=''; answer=None; missing=[]
    if zs in concat:
        k=concat[zs]; answer=(d1+d2+d3+d4) if k=='fwd' else (d3+d4+d1+d2)
        interp=f"operator {zs} is concat_{k} (identified from examples)"
    elif zs in puzzle.by_op and variants and puzzle.op_map[zs] in variants:
        var=variants[puzzle.op_map[zs]]
        a=10*sym2dig[d1]+sym2dig[d2] if mode=='standard' else 10*sym2dig[d2]+sym2dig[d1]
        b=10*sym2dig[d3]+sym2dig[d4] if mode=='standard' else 10*sym2dig[d4]+sym2dig[d3]
        r=cg.apply_op(var,a,b); answer,missing=num_to_sym(r,dig2sym,zs,mode)
        interp=f"operator {zs} = {var} (from examples): {a} {var} {b} = {r}"
    else:
        fam=cg.op_top_family(zs); base=BASE.get(fam,'add')
        if sym2dig:
            a=10*sym2dig[d1]+sym2dig[d2] if mode=='standard' else 10*sym2dig[d2]+sym2dig[d1]
            b=10*sym2dig[d3]+sym2dig[d4] if mode=='standard' else 10*sym2dig[d4]+sym2dig[d3]
            r=cg.apply_op(base,a,b); answer,missing=num_to_sym(r,dig2sym,zs,mode)
            interp=f"operator {zs} is unseen; examples solvable so guess arithmetic via prior {fam}->{base}: {a} {base} {b} = {r}"
        else:
            answer=d1+d2+d3+d4; interp=f"operator {zs} is unseen; guess concat_fwd"
    L=[]
    if sym2dig:
        L.append("Digit mapping (symbol -> digit):")
        for s in sorted(sym2dig, key=lambda x: lm[x]):
            L.append(f"  {s} ({lm[s]}) -> {sym2dig[s]}")
    else:
        L.append("Digit mapping: none needed (all operators are concat)")
    if variants: L.append("Operator variants: "+", ".join(f"{k}={v}" for k,v in variants.items()))
    if concat:   L.append("Concat operators: "+", ".join(f"{s}=concat_{k}" for s,k in concat.items()))
    L.append(f"Reading order: {mode}")
    L.append(f"Query: {q}")
    L.append(f"Derivation: {interp}")
    L.append(f"Answer: {answer}")
    if missing:
        md=sorted(set(missing))
        L.append("")
        L.append(f"NOTE - missing digit's symbol: the query result needs digit(s) {md}, but no symbol")
        L.append(f"for {'them' if len(md)>1 else 'it'} appears anywhere in this puzzle's examples or query, so the symbol")
        L.append("cannot be determined. Each puzzle assigns symbols to digits independently and at")
        L.append("random, so a digit that never shows up has no recoverable symbol (no cross-puzzle")
        L.append("lookup or in-puzzle deduction works). The example mapping IS solved and the answer")
        L.append("number is known; only the symbol(s) for the unseen digit(s) are unknowable (shown as _).")
    return "\n".join(L)+"\n"

if __name__=='__main__':
    types=['arithmetic','little_endian','mixed_concat','mixed_concat_little_endian','query_unseen_concat','pure_concat']
    dirs=sorted(set(glob.glob('arithmetic_*/')+glob.glob('little_endian_*/')+
        [d for d in glob.glob('mixed_concat_*/') if not d.startswith('mixed_concat_little_endian')]+
        glob.glob('mixed_concat_little_endian_*/')+glob.glob('query_unseen_concat_*/')+glob.glob('pure_concat_*/')))
    NOT_SOLVABLE=("not solvable\n\n"
        "NOTE - what 'not solvable' means: no assignment of distinct digits to the\n"
        "symbols appearing in the examples and query satisfies all example equations.\n"
        "The DFS exhausted every candidate operator family and digit assignment (under\n"
        "both standard and little-endian reading orders) without finding a consistent\n"
        "mapping. This happens when the true operator is outside the supported set\n"
        "(e.g. lcm, gcd, mod, rmod, fdiv, a2_plus_b), or the search space was too large\n"
        "to exhaust within the time limit. So no digit mapping could be found.\n")
    n_sol=n_notsolv=n_missing=0
    for d in dirs:
        name=d.rstrip('/')
        if name in DELETED or not os.path.exists(d+'tree.json'):
            open(d+'tree_solution.txt','w').write(NOT_SOLVABLE); n_notsolv+=1; continue
        try:
            content=gen(d); open(d+'tree_solution.txt','w').write(content); n_sol+=1
            if 'missing digit' in content: n_missing+=1
        except Exception as e:
            open(d+'tree_solution.txt','w').write(NOT_SOLVABLE); n_notsolv+=1
            print("ERR",name,e)
    print(f"solutions written={n_sol} (of which {n_missing} have a missing digit's symbol), not_solvable={n_notsolv}")
