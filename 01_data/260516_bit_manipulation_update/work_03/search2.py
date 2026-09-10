"""Search 2-operand composites: op(a(n), b(n)) over many ops."""
import sys
sys.path.insert(0, r'c:/Users/YongsanHuang/SynologyDrive/00_Kaggle/2026_Nemotron/01_data/260516_bit_manipulation_update/work_03')
from helpers import *

def make_unary_pool():
    pool = {'id': lambda n: n, 'not': lambda n: NOT(n), 'rev': lambda n: rev(n),
            'swap': lambda n: swap_nib(n), 'rev_not': lambda n: NOT(rev(n)),
            'not_rev': lambda n: rev(NOT(n))}
    for k in range(1, 8):
        pool[f'rol{k}'] = lambda n, k=k: rol(n, k)
        pool[f'ror{k}'] = lambda n, k=k: ror(n, k)
        pool[f'shl{k}'] = lambda n, k=k: (n << k) & 0xFF
        pool[f'shr{k}'] = lambda n, k=k: (n >> k) & 0xFF
    return pool

OPS = {
    '^': lambda x, y: (x ^ y) & 0xFF,
    '&': lambda x, y: (x & y) & 0xFF,
    '|': lambda x, y: (x | y) & 0xFF,
    '+': lambda x, y: (x + y) & 0xFF,
    '-': lambda x, y: (x - y) & 0xFF,
    'andnot': lambda x, y: (x & (~y)) & 0xFF,
    'ornot': lambda x, y: (x | (~y)) & 0xFF,
    'xnor': lambda x, y: (~(x ^ y)) & 0xFF,
}

def search(examples, with_unary_outer=False):
    pool = make_unary_pool()
    names = list(pool.keys())
    found = []
    for na in names:
        fa = pool[na]
        for nb in names:
            fb = pool[nb]
            for op_name, op in OPS.items():
                ok = True
                for inp, out in examples:
                    ni = b(inp); no = b(out)
                    if op(fa(ni), fb(ni)) != no:
                        ok = False; break
                if ok:
                    found.append(f'{na} {op_name} {nb}')
    if with_unary_outer:
        outer_pool = {'id': lambda n: n, 'not': lambda n: NOT(n), 'rev': lambda n: rev(n), 'swap': lambda n: swap_nib(n)}
        for ou_name, ou in outer_pool.items():
            for na in names:
                fa = pool[na]
                for nb in names:
                    fb = pool[nb]
                    for op_name, op in OPS.items():
                        ok = True
                        for inp, out in examples:
                            ni = b(inp); no = b(out)
                            if ou(op(fa(ni), fb(ni))) != no:
                                ok = False; break
                        if ok:
                            found.append(f'{ou_name}({na} {op_name} {nb})')
    return found

if __name__ == "__main__":
    import json, re
    name = sys.argv[1]
    with_outer = '--outer' in sys.argv
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
        found = search(examples, with_unary_outer=with_outer)
        for f in found[:30]:
            print(f"  MATCH: {f}")
        if not found:
            print("  No matches.")
