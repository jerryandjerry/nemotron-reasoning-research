import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from helpers import *
from search import *
import json

with open(r"C:/Users/YongsanHuang/SynologyDrive/00_Kaggle/2026_Nemotron/01_data/260516_bit_manipulation_update/_batch_02.json") as f:
    puzzles = json.load(f)

def parse_prompt(prompt):
    """Returns (examples list, query)."""
    examples = []
    query = None
    lines = prompt.strip().split('\n')
    for line in lines:
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

for puzzle in puzzles:
    ex, q = parse_prompt(puzzle['prompt'])
    print(f"\n### {puzzle['id']} query={q} answer={puzzle['answer']} ###")
    run_search(puzzle['id'], ex)
