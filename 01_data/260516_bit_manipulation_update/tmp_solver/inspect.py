"""Print all pairs nicely + various transformations to help find patterns."""
import json, re

PATH = r"c:/Users/YongsanHuang/SynologyDrive/00_Kaggle/2026_Nemotron/01_data/260516_bit_manipulation_update/_batch_07.json"

def parse_examples(prompt):
    pairs = []
    for line in prompt.splitlines():
        m = re.match(r"^([01]{8})\s*->\s*([01]{8})\s*$", line.strip())
        if m:
            pairs.append((int(m.group(1), 2), int(m.group(2), 2)))
    qm = re.search(r"determine the output for:\s*([01]{8})", prompt)
    query = int(qm.group(1), 2)
    return pairs, query

def b(x):
    return format(x & 0xFF, '08b')

def popcount(x):
    return bin(x & 0xFF).count('1')

data = json.load(open(PATH))
for p in data:
    pairs, q = parse_examples(p['prompt'])
    gold = int(p['answer'], 2)
    print(f"\n=== {p['id']} ===")
    print(f"{'inp':>10} {'out':>10}  inX  outX  xor       and       or        ipop opop  out-in  out+in")
    for x, y in pairs:
        xor = x ^ y
        andv = x & y
        orv = x | y
        print(f"  {b(x)}  {b(y)}  {x:3d}  {y:3d}  {b(xor)}  {b(andv)}  {b(orv)}  {popcount(x):3d} {popcount(y):3d}  {(y-x)&0xFF:3d}  {(y+x)&0xFF:3d}")
    print(f"  query: {b(q)}  gold: {b(gold)} (xor: {b(q^gold)})")
