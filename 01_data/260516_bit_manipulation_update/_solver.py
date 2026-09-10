"""Hypothesis tester for 8-bit puzzles."""
import json
import re
from itertools import product

DATA_PATH = r'c:/Users/YongsanHuang/SynologyDrive/00_Kaggle/2026_Nemotron/01_data/260516_bit_manipulation_update/_batch_01.json'

def parse_puzzle(prompt):
    examples = []
    query = None
    for line in prompt.strip().split('\n'):
        m = re.match(r'^([01]{8})\s*->\s*([01]{8})\s*$', line.strip())
        if m:
            examples.append((m.group(1), m.group(2)))
        elif 'determine the output for:' in line:
            qm = re.search(r'([01]{8})', line)
            if qm:
                query = qm.group(1)
    return examples, query


def b2i(s): return int(s, 2)
def i2b(n): return format(n & 0xFF, '08b')

# Define a library of transformations: name -> function: int->int
def make_lib():
    lib = {}
    # shifts
    for k in range(1, 8):
        lib[f'shl_{k}_0'] = lambda x, k=k: (x << k) & 0xFF  # shift left fill 0
        lib[f'shr_{k}_0'] = lambda x, k=k: (x >> k) & 0xFF  # shift right fill 0 (logical)
        lib[f'shr_{k}_sgn'] = lambda x, k=k: ((x >> k) | (0xFF & ~((1 << (8-k))-1)) if x & 0x80 else (x >> k))  # sign extend
        lib[f'rol_{k}'] = lambda x, k=k: ((x << k) | (x >> (8-k))) & 0xFF
        lib[f'ror_{k}'] = lambda x, k=k: ((x >> k) | (x << (8-k))) & 0xFF
    # reverse, swap nibbles, NOT
    def rev(x):
        r = 0
        for i in range(8):
            if x & (1 << i):
                r |= 1 << (7-i)
        return r
    lib['rev'] = rev
    lib['swap_nib'] = lambda x: ((x & 0xF) << 4) | ((x & 0xF0) >> 4)
    lib['not'] = lambda x: (~x) & 0xFF
    # XOR with constants
    for c in [0xFF, 0xAA, 0x55, 0xF0, 0x0F, 0xCC, 0x33, 0x3C, 0xC3, 0x99, 0x66, 0x81, 0x18, 0x7E, 0x42, 0x24]:
        lib[f'xor_{c:02x}'] = lambda x, c=c: x ^ c
    # gray code style
    lib['xor_shr1'] = lambda x: x ^ (x >> 1)
    lib['xor_shl1'] = lambda x: x ^ ((x << 1) & 0xFF)
    lib['xor_shr2'] = lambda x: x ^ (x >> 2)
    lib['xor_shl2'] = lambda x: x ^ ((x << 2) & 0xFF)
    # arithmetic
    for k in [1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 42, 63, 64, 85, 127, 128, 170]:
        lib[f'add_{k}'] = lambda x, k=k: (x + k) & 0xFF
        lib[f'sub_{k}'] = lambda x, k=k: (x - k) & 0xFF
        lib[f'rsub_{k}'] = lambda x, k=k: (k - x) & 0xFF
        lib[f'mul_{k}'] = lambda x, k=k: (x * k) & 0xFF
    # NOT then shift, shift then NOT, rev then ...
    return lib

def try_compositions(examples, max_depth=2):
    """Try single transforms and pairs."""
    lib = make_lib()
    items = list(lib.items())
    # single
    for name, fn in items:
        ok = True
        for inp, out in examples:
            try:
                if fn(b2i(inp)) != b2i(out):
                    ok = False; break
            except Exception:
                ok = False; break
        if ok:
            return [name]
    if max_depth >= 2:
        for n1, f1 in items:
            for n2, f2 in items:
                ok = True
                for inp, out in examples:
                    try:
                        if f2(f1(b2i(inp))) != b2i(out):
                            ok = False; break
                    except Exception:
                        ok = False; break
                if ok:
                    return [n1, n2]
    return None

def apply_chain(chain, x):
    lib = make_lib()
    for name in chain:
        x = lib[name](x)
    return x


if __name__ == '__main__':
    data = json.load(open(DATA_PATH))
    for d in data:
        examples, query = parse_puzzle(d['prompt'])
        chain = try_compositions(examples, max_depth=1)
        if chain:
            qans = i2b(apply_chain(chain, b2i(query)))
            print(f"{d['id']}: chain={chain}, query->{qans}, expected={d['answer']}, match={qans==d['answer']}")
        else:
            print(f"{d['id']}: no single-transform match")
