import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from helpers import *
from search import all_candidate_ops
from search2 import try_per_bit_general, find_consistent_perbit_rules, try_binary_xor_const

with open(r"C:/Users/YongsanHuang/SynologyDrive/00_Kaggle/2026_Nemotron/01_data/260516_bit_manipulation_update/_batch_02.json") as f:
    puzzles = json.load(f)

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

def apply_perbit_rules(inp, rules):
    out = ''
    for r in rules:
        if r[0]=='1var':
            _, j, f = r
            a = int(inp[j])
            if f==0: bit = 0
            elif f==1: bit = a
            elif f==2: bit = 1-a
            else: bit = 1
        elif r[0]=='2var':
            _, j, k, f = r
            a, b = int(inp[j]), int(inp[k])
            idx = a*2+b
            bit = (f >> idx) & 1
        elif r[0]=='3var':
            _, j, k, l, f = r
            a, b, c = int(inp[j]), int(inp[k]), int(inp[l])
            idx = a*4+b*2+c
            bit = (f >> idx) & 1
        out += str(bit)
    return out

for puzzle in puzzles:
    ex, q = parse_prompt(puzzle['prompt'])
    pb = try_per_bit_general(ex)
    rules = find_consistent_perbit_rules(pb)
    if rules:
        derived = apply_perbit_rules(q, rules)
        ok = derived == puzzle['answer']
        print(f"{puzzle['id']}: PER-BIT FOUND. derived={derived}, ans={puzzle['answer']}, match={ok}")
        for i, r in enumerate(rules):
            print(f"  out[{i}] = {r}")
    else:
        # report missing bits
        missing = [i for i,o in enumerate(pb) if not o]
        print(f"{puzzle['id']}: no per-bit solution. Missing bits: {missing}")
