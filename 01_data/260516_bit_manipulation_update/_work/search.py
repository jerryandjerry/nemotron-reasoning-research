"""Generic search across rules for any puzzle."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from helpers import *

def all_candidate_ops(s):
    """Generate dict of named candidate 8-bit operands derived from input s."""
    cands = {}
    cands['in'] = s
    cands['rev'] = reverse_bits(s)
    cands['not'] = NOT(s)
    cands['swap'] = swap_nibbles(s)
    cands['notrev'] = NOT(reverse_bits(s))
    cands['revswap'] = reverse_bits(swap_nibbles(s))
    for k in range(1,8):
        cands[f'shl{k}'] = shl(s,k)
        cands[f'shr{k}'] = shr(s,k)
        cands[f'rotl{k}'] = rotl(s,k)
        cands[f'rotr{k}'] = rotr(s,k)
    # Arithmetic shift right (sign extend)
    for k in range(1,8):
        cands[f'asr{k}'] = s[0]*k + s[:-k] if k<8 else s[0]*8
    return cands

def try_single_ops(ex):
    """Try single op match."""
    keys = list(all_candidate_ops(ex[0][0]).keys())
    found = []
    for k in keys:
        ok = True
        for inp, out in ex:
            if all_candidate_ops(inp)[k] != out:
                ok = False; break
        if ok: found.append(k)
    return found

def try_xor_with_const(ex):
    """out = in XOR const."""
    found = []
    for c in range(256):
        cs = format(c,'08b')
        ok = True
        for inp, out in ex:
            if XOR(inp, cs) != out:
                ok = False; break
        if ok: found.append(cs)
    return found

def try_binary_ops(ex):
    """Try binary ops over candidate operands."""
    keys = list(all_candidate_ops(ex[0][0]).keys())
    found = []
    for k1 in keys:
        for k2 in keys:
            if k1 > k2:  # symmetric ops; skip dup
                continue
            for op_name, op in [('AND',AND),('OR',OR),('XOR',XOR)]:
                ok = True
                for inp, out in ex:
                    cs = all_candidate_ops(inp)
                    if op(cs[k1], cs[k2]) != out:
                        ok = False; break
                if ok: found.append((op_name, k1, k2))
    return found

def try_xor_with_const_op(ex):
    """out = op(in) XOR const for various ops."""
    keys = list(all_candidate_ops(ex[0][0]).keys())
    found = []
    for k in keys:
        for c in range(256):
            cs = format(c,'08b')
            ok = True
            for inp, out in ex:
                if XOR(all_candidate_ops(inp)[k], cs) != out:
                    ok = False; break
            if ok: found.append((k, cs))
    return found

def try_add_const(ex):
    """out = (in + const) mod 256."""
    found = []
    for c in range(256):
        ok = True
        for inp, out in ex:
            if (b2i(inp)+c)&0xFF != b2i(out):
                ok = False; break
        if ok: found.append(c)
    return found

def try_sub_const(ex):
    """out = (const - in) mod 256."""
    found = []
    for c in range(256):
        ok = True
        for inp, out in ex:
            if (c - b2i(inp))&0xFF != b2i(out):
                ok = False; break
        if ok: found.append(c)
    return found

def try_mul_const(ex):
    found = []
    for c in range(256):
        ok = True
        for inp, out in ex:
            if (b2i(inp)*c)&0xFF != b2i(out):
                ok = False; break
        if ok: found.append(c)
    return found

def run_search(name, ex):
    print(f"=== Puzzle {name} ===")
    print(f"  {len(ex)} examples")
    # Single op
    r = try_single_ops(ex)
    if r:
        print(f"  Single op:", r)
    # XOR with constant
    r = try_xor_with_const(ex)
    if r:
        print(f"  XOR with const:", [c for c in r])
    # op XOR const
    r = try_xor_with_const_op(ex)
    if r:
        print(f"  op XOR const:", r[:5], '...' if len(r)>5 else '')
    # Binary ops
    r = try_binary_ops(ex)
    if r:
        print(f"  Binary ops:", r[:10], '...' if len(r)>10 else '')
    # Arithmetic
    r = try_add_const(ex)
    if r: print(f"  Add const:", r)
    r = try_sub_const(ex)
    if r: print(f"  Sub from const (c - in):", r)
    r = try_mul_const(ex)
    if r: print(f"  Mul const:", r)
