"""Generic exploration: try a bunch of common rules against a set of examples."""
import sys
sys.path.insert(0, r'c:/Users/YongsanHuang/SynologyDrive/00_Kaggle/2026_Nemotron/01_data/260516_bit_manipulation_update/work_03')
from helpers import *

def try_all(examples, query=None):
    rules = {}
    # rotations
    for k in range(1, 8):
        rules[f"rol_{k}"] = lambda n, k=k: rol(n, k)
        rules[f"ror_{k}"] = lambda n, k=k: ror(n, k)
    # shifts
    for k in range(1, 8):
        rules[f"shl_{k}"] = lambda n, k=k: shl(n, k)
        rules[f"shr_{k}"] = lambda n, k=k: shr(n, k)
    # reverse / swap / not
    rules["reverse"] = rev
    rules["swap_nib"] = swap_nib
    rules["NOT"] = NOT
    rules["reverse_NOT"] = lambda n: rev(NOT(n))
    rules["NOT_reverse"] = lambda n: NOT(rev(n))
    # XOR masks
    for mask in range(256):
        rules[f"xor_{mask:08b}"] = lambda n, m=mask: n ^ m
    # XOR with shift/rotation of self
    for k in range(1, 8):
        rules[f"xor_shr_{k}"] = lambda n, k=k: n ^ shr(n, k)
        rules[f"xor_shl_{k}"] = lambda n, k=k: n ^ shl(n, k)
        rules[f"xor_rol_{k}"] = lambda n, k=k: n ^ rol(n, k)
        rules[f"xor_ror_{k}"] = lambda n, k=k: n ^ ror(n, k)
    # NOT compositions
    for k in range(1, 8):
        rules[f"NOT_rol_{k}"] = lambda n, k=k: NOT(rol(n, k))
        rules[f"NOT_ror_{k}"] = lambda n, k=k: NOT(ror(n, k))
    # arithmetic
    for c in range(-128, 128):
        rules[f"add_{c}"] = lambda n, c=c: (n + c) & 0xFF
    for c in range(2, 16):
        rules[f"mul_{c}"] = lambda n, c=c: (n * c) & 0xFF

    perfect = []
    near = []
    for name, r in rules.items():
        p, t, f = test(r, examples)
        if p == t:
            perfect.append(name)
        elif p >= t - 1:
            near.append((name, p, t))
    return perfect, near, rules

if __name__ == "__main__":
    import json
    name = sys.argv[1]
    with open(r'c:/Users/YongsanHuang/SynologyDrive/00_Kaggle/2026_Nemotron/01_data/260516_bit_manipulation_update/_batch_03.json') as f:
        data = json.load(f)
    for entry in data:
        if entry['id'] != name:
            continue
        prompt = entry['prompt']
        # parse examples
        lines = prompt.split('\n')
        examples = []
        query = None
        import re
        for line in lines:
            line = line.strip()
            m = re.match(r'^([01]{8})\s*->\s*([01]{8})$', line)
            if m:
                examples.append((m.group(1), m.group(2)))
            elif line.startswith('Now, determine the output for:'):
                query = line.split(':', 1)[1].strip()
        print("Examples:", examples)
        print("Query:", query)
        print("Gold:", entry['answer'])
        perfect, near, rules = try_all(examples, query)
        print("\nPerfect matches:")
        for p in perfect:
            print(f"  {p} -> query={s(rules[p](b(query)))}")
        if not perfect:
            print("Near matches (off by 1):")
            for n, p, t in near[:10]:
                print(f"  {n}: {p}/{t}")
