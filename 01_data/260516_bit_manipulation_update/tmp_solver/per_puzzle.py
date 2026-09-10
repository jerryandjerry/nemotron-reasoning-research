"""Explore one puzzle at a time with many hypotheses."""
import json, re, sys

PATH = r"c:/Users/YongsanHuang/SynologyDrive/00_Kaggle/2026_Nemotron/01_data/260516_bit_manipulation_update/_batch_07.json"

def parse_examples(prompt):
    pairs = []
    for line in prompt.splitlines():
        m = re.match(r"^([01]{8})\s*->\s*([01]{8})\s*$", line.strip())
        if m:
            pairs.append((int(m.group(1), 2), int(m.group(2), 2)))
    qm = re.search(r"determine the output for:\s*([01]{8})", prompt)
    return pairs, int(qm.group(1), 2)

def b(x): return format(x & 0xFF, '08b')

def explore(pid, pairs, query, gold):
    print(f"\n### {pid}  query={b(query)} gold={b(gold)}")
    print("input    out      | shift-derived comparisons")
    for x, y in pairs:
        s1 = (x >> 1) & 0xFF
        s2 = (x << 1) & 0xFF
        # x ^ shifted
        e1 = x ^ s1
        e2 = x ^ s2
        comb = (x | s1) & 0xFF
        # cellular automaton: each bit = MAJ(left, self, right) etc?
        # Bit positions: bit0=MSB ... bit7=LSB
        print(f"{b(x)} {b(y)}  | x>>1={b(s1)} x<<1={b(s2)} x^(x>>1)={b(e1)} x^(x<<1)={b(e2)}  x|x>>1={b(comb)}")
    print(f"query   {b(query)} gold={b(gold)}")

def main():
    data = json.load(open(PATH))
    targets = sys.argv[1:] if len(sys.argv) > 1 else None
    for p in data:
        if targets and p['id'] not in targets: continue
        pairs, q = parse_examples(p['prompt'])
        gold = int(p['answer'], 2)
        explore(p['id'], pairs, q, gold)

if __name__ == '__main__':
    main()
