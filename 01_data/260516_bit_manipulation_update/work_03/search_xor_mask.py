"""Search for unary(n) XOR mask form or unary(n) + c."""
import sys
sys.path.insert(0, r'c:/Users/YongsanHuang/SynologyDrive/00_Kaggle/2026_Nemotron/01_data/260516_bit_manipulation_update/work_03')
from helpers import *

def make_unary_pool():
    pool = {'id': lambda n: n, 'not': lambda n: NOT(n), 'rev': lambda n: rev(n),
            'swap': lambda n: swap_nib(n)}
    for k in range(1, 8):
        pool[f'rol{k}'] = lambda n, k=k: rol(n, k)
        pool[f'ror{k}'] = lambda n, k=k: ror(n, k)
        pool[f'shl{k}'] = lambda n, k=k: (n << k) & 0xFF
        pool[f'shr{k}'] = lambda n, k=k: (n >> k) & 0xFF
    # composites
    pool['rev_not'] = lambda n: NOT(rev(n))
    pool['not_rev'] = lambda n: rev(NOT(n))
    return pool

OPS = {
    '^': lambda x, y: (x ^ y) & 0xFF,
    '&': lambda x, y: (x & y) & 0xFF,
    '|': lambda x, y: (x | y) & 0xFF,
    '+': lambda x, y: (x + y) & 0xFF,
    '-': lambda x, y: (x - y) & 0xFF,
}

def search_const(examples):
    pool = make_unary_pool()
    found = []
    for name, f in pool.items():
        for op_name, op in OPS.items():
            for c in range(256):
                ok = True
                for inp, out in examples:
                    ni = b(inp); no = b(out)
                    if op(f(ni), c) != no:
                        ok = False; break
                if ok:
                    found.append(f'{name} {op_name} {c:08b}({c})')
    return found

def search_two_const(examples):
    """unary(n) OP c1 OP2 c2 - not useful"""
    pass

def search_binary_const(examples):
    """((a OP b) OP c) where c is constant"""
    pool = make_unary_pool()
    names = list(pool.keys())
    found = []
    for na in names:
        fa = pool[na]
        for nb in names:
            fb = pool[nb]
            for op1_name, op1 in OPS.items():
                for op2_name, op2 in OPS.items():
                    for c in range(256):
                        ok = True
                        for inp, out in examples:
                            ni = b(inp); no = b(out)
                            v = op2(op1(fa(ni), fb(ni)), c)
                            if (v & 0xFF) != no:
                                ok = False; break
                        if ok:
                            found.append(f'({na} {op1_name} {nb}) {op2_name} {c:08b}')
                            if len(found) > 20: return found
    return found

if __name__ == "__main__":
    import json, re
    name = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'const'
    with open(r'c:/Users/YongsanHuang/SynologyDrive/00_Kaggle/2026_Nemotron/01_data/260516_bit_manipulation_update/_batch_03.json') as f:
        data = json.load(f)
    for entry in data:
        if entry['id'] != name: continue
        prompt = entry['prompt']
        examples = []
        query = None
        for line in prompt.split('\n'):
            line = line.strip()
            m = re.match(r'^([01]{8})\s*->\s*([01]{8})$', line)
            if m: examples.append((m.group(1), m.group(2)))
            elif line.startswith('Now, determine the output for:'):
                query = line.split(':', 1)[1].strip()
        print(f"Query: {query}, Gold: {entry['answer']}")
        if mode == 'const':
            found = search_const(examples)
        else:
            found = search_binary_const(examples)
        for f in found[:30]:
            print(f"  MATCH: {f}")
        if not found:
            print("  No matches.")
