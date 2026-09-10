"""Brute force search over many transformation families."""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from helpers import *

def parse_prompt(prompt):
    examples = []
    query = None
    for line in prompt.strip().split('\n'):
        line = line.strip()
        if '->' in line and len(line) > 5:
            parts = line.split('->')
            inp = parts[0].strip()
            out = parts[1].strip()
            if len(inp)==8 and all(c in '01' for c in inp):
                examples.append((inp, out))
        elif line.startswith('Now, determine the output for:'):
            query = line.split(':')[1].strip()
    return examples, query

def gen_ops(s):
    """Returns dict of named transforms of s."""
    cands = {'in':s, 'rev':reverse_bits(s), 'not':NOT(s), 'swap':swap_nibbles(s)}
    for k in range(1,8):
        cands[f'shl{k}'] = shl(s,k)
        cands[f'shr{k}'] = shr(s,k)
        cands[f'rotl{k}'] = rotl(s,k)
        cands[f'rotr{k}'] = rotr(s,k)
        cands[f'asr{k}'] = s[0]*k + s[:-k]  # arithmetic shift right
    return cands

def search_2op_with_const(ex):
    """Returns list of (op, k1, k2, const) where op(c1, c2) XOR const matches all examples."""
    found = []
    keys = list(gen_ops(ex[0][0]).keys())
    for op_name, op in [('AND',AND),('OR',OR),('XOR',XOR)]:
        for k1 in keys:
            for k2 in keys:
                if k1 > k2 and op_name in ('AND','OR','XOR'): continue
                # Get base for first example
                cs0 = gen_ops(ex[0][0])
                base = op(cs0[k1], cs0[k2])
                req_c = XOR(base, ex[0][1])
                ok = True
                for inp, out in ex[1:]:
                    cs = gen_ops(inp)
                    if XOR(op(cs[k1], cs[k2]), req_c) != out:
                        ok = False; break
                if ok:
                    found.append((op_name, k1, k2, req_c))
    return found

def search_arith(ex):
    """Try a*in + b mod m for various m."""
    found = []
    for m in [256, 255, 254, 253, 251, 127, 128]:
        for a in range(0, m if m<256 else 256):
            for b in range(m if m<256 else 256):
                ok = True
                for inp, out in ex:
                    if (a * b2i(inp) + b) % m != b2i(out):
                        ok = False; break
                if ok: found.append((a, b, m))
    return found

def search_3op_xor(ex):
    """Try a XOR b XOR c with operands from candidate set."""
    found = []
    keys = list(gen_ops(ex[0][0]).keys())
    short_keys = ['in','rev','not','swap'] + [f'shl{k}' for k in range(1,5)] + [f'shr{k}' for k in range(1,5)] + [f'rotl{k}' for k in range(1,5)] + [f'rotr{k}' for k in range(1,5)]
    short_keys = [k for k in short_keys if k in keys]
    for k1 in short_keys:
        for k2 in short_keys:
            for k3 in short_keys:
                ok = True
                for inp, out in ex:
                    cs = gen_ops(inp)
                    if XOR(XOR(cs[k1], cs[k2]), cs[k3]) != out:
                        ok = False; break
                if ok: found.append((k1, k2, k3))
    return found

def search_op2_then_op(ex):
    """Try op2(op1(in, a), b) for various ops where a,b are derived ops."""
    found = []
    keys = ['in','rev','not','swap'] + [f'shl{k}' for k in range(1,8)] + [f'shr{k}' for k in range(1,8)] + [f'rotl{k}' for k in range(1,8)] + [f'rotr{k}' for k in range(1,8)]
    keys = list(set(keys))
    op1s = [('AND',AND),('OR',OR),('XOR',XOR)]
    op2s = [('AND',AND),('OR',OR),('XOR',XOR)]
    for k1 in keys:
        for k2 in keys:
            if k1 == k2: continue
            for k3 in keys:
                for op1name, op1 in op1s:
                    for op2name, op2 in op2s:
                        ok = True
                        for inp, out in ex:
                            cs = gen_ops(inp)
                            try:
                                inner = op1(cs[k1], cs[k2])
                                result = op2(inner, cs[k3])
                                if result != out:
                                    ok = False; break
                            except: ok = False; break
                        if ok:
                            found.append((op1name, k1, k2, op2name, k3))
                            if len(found) > 20: return found
    return found

if __name__ == '__main__':
    with open(r"C:/Users/YongsanHuang/SynologyDrive/00_Kaggle/2026_Nemotron/01_data/260516_bit_manipulation_update/_batch_02.json") as f:
        puzzles = json.load(f)

    target_id = sys.argv[1] if len(sys.argv)>1 else None
    for puzzle in puzzles:
        if target_id and puzzle['id'] != target_id: continue
        ex, q = parse_prompt(puzzle['prompt'])
        print(f"\n### {puzzle['id']} (query={q}, ans={puzzle['answer']}) ###")
        r = search_2op_with_const(ex)
        if r:
            print(f"  2-op + XOR const: {r[:5]}")
            for op, k1, k2, c in r[:3]:
                from search import all_candidate_ops
                derived = XOR((AND if op=='AND' else OR if op=='OR' else XOR)(gen_ops(q)[k1], gen_ops(q)[k2]), c)
                print(f"    derived for query: {derived}, ans match: {derived == puzzle['answer']}")
        r3 = search_3op_xor(ex)
        if r3:
            print(f"  3-op XOR: {r3[:5]}")
            for k1, k2, k3 in r3[:3]:
                derived = XOR(XOR(gen_ops(q)[k1], gen_ops(q)[k2]), gen_ops(q)[k3])
                print(f"    derived: {derived}, match: {derived == puzzle['answer']}")
