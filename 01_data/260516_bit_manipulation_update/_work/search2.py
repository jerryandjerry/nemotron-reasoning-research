"""Extended search."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from helpers import *
from search import all_candidate_ops, try_single_ops, try_xor_with_const, try_binary_ops, try_xor_with_const_op, try_add_const, try_sub_const, try_mul_const

def try_binary_xor_const(ex):
    """op(a, b) XOR const."""
    keys = list(all_candidate_ops(ex[0][0]).keys())
    found = []
    for k1 in keys:
        for k2 in keys:
            if k1 > k2: continue
            for op_name, op in [('AND',AND),('OR',OR),('XOR',XOR)]:
                # find required const from first example
                cs0 = all_candidate_ops(ex[0][0])
                base = op(cs0[k1], cs0[k2])
                req_const = XOR(base, ex[0][1])
                ok = True
                for inp, out in ex[1:]:
                    cs = all_candidate_ops(inp)
                    if XOR(op(cs[k1], cs[k2]), req_const) != out:
                        ok = False; break
                if ok: found.append((op_name, k1, k2, req_const))
    return found

def try_arith_shifts(ex):
    """out = (in op const) shifted/rotated."""
    found = []
    for k in range(0, 8):
        # rotl by k of (in + c)
        for c in range(256):
            ok = True
            for inp, out in ex:
                v = (b2i(inp)+c) & 0xFF
                rotated = rotl(format(v,'08b'), k)
                if rotated != out:
                    ok = False; break
            if ok: found.append(('rotl_add', k, c))
    return found

def try_per_bit_general(ex):
    """For each output bit, find a function of <=3 input bits (any boolean function)."""
    from itertools import combinations, product
    ins = [inp for inp,_ in ex]
    outs = [out for _,out in ex]
    n = len(ex)
    results = []
    for op in range(8):
        out_bits = [out[op] for out in outs]
        # try 1-var
        found_pos = []
        # try 2-var: all 16 boolean functions
        for j in range(8):
            in_bits = [inp[j] for inp in ins]
            for f in range(4):  # 0=const0, 1=identity, 2=NOT, 3=const1
                if f==0: computed = ['0']*n
                elif f==1: computed = in_bits
                elif f==2: computed = ['1' if b=='0' else '0' for b in in_bits]
                else: computed = ['1']*n
                if computed == out_bits:
                    found_pos.append(('1var', j, f))
        for j in range(8):
            for k in range(8):
                if j==k: continue
                for f in range(16):
                    bits_jk = [(int(inp[j]), int(inp[k])) for inp in ins]
                    computed = []
                    for a,b in bits_jk:
                        idx = a*2+b
                        bit = (f >> idx) & 1
                        computed.append(str(bit))
                    if computed == out_bits:
                        found_pos.append(('2var', j, k, f))
        # 3-var: 256 funcs * 8*7*6/6 combos = manageable for 8 examples
        for j,k,l in combinations(range(8), 3):
            for f in range(256):
                bits_jkl = [(int(inp[j]), int(inp[k]), int(inp[l])) for inp in ins]
                computed = []
                ok = True
                for a,b,c in bits_jkl:
                    idx = a*4+b*2+c
                    bit = (f >> idx) & 1
                    computed.append(str(bit))
                if computed == out_bits:
                    found_pos.append(('3var', j, k, l, f))
        results.append(found_pos)
    return results

def find_consistent_perbit_rules(per_bit_options):
    """If each output bit has at least one rule, return per-bit rules (preferring simpler)."""
    rules = []
    for opt in per_bit_options:
        if not opt:
            return None
        # pick simplest: 1var > 2var > 3var
        opt_sorted = sorted(opt, key=lambda x: (0 if x[0]=='1var' else 1 if x[0]=='2var' else 2))
        rules.append(opt_sorted[0])
    return rules
