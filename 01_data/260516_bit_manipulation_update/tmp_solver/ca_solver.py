"""Search for cellular-automaton-style rules: each output bit y[i] is a boolean
function of a window of input bits {x[i+d] for d in offsets}, with optional
wraparound (rotation) or zero-fill at boundaries.

This handles:
- per-bit "ECA" Rule X (3-cell window with rotation/zero fill)
- 5-cell windows
- Any output bit y[i] = f(x[i-1], x[i], x[i+1]) where f is one of 256 functions

We also test "shift by k then apply rule" by allowing the window to be
{x[i+a], x[i+b], x[i+c]} for fixed offsets a,b,c.

Output index convention: bit i is the MSB position i (i=0..7).
"""
import json, re, itertools

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

def bits(x):
    return [(x >> (7 - i)) & 1 for i in range(8)]  # bit 0 = MSB

def from_bits(bs):
    v = 0
    for i, bit in enumerate(bs):
        v |= (bit & 1) << (7 - i)
    return v

def get_window(bs, i, offsets, boundary='wrap'):
    n = len(bs)
    out = []
    for o in offsets:
        j = i + o
        if boundary == 'wrap':
            out.append(bs[j % n])
        elif boundary == 'zero':
            out.append(bs[j] if 0 <= j < n else 0)
        elif boundary == 'one':
            out.append(bs[j] if 0 <= j < n else 1)
        elif boundary == 'edge':
            out.append(bs[max(0, min(n-1, j))])
    return out

def find_ca_rule(pairs, offsets, boundary='wrap'):
    """For each bit position i, find boolean fn f: window -> bit such that
    y[i] = f(window of x at i). The function may be position-independent or
    position-dependent. We try position-independent first."""
    nbits = len(offsets)
    # collect (window_int, output_bit) samples across all bit positions
    samples = []
    for x, y in pairs:
        xb = bits(x)
        yb = bits(y)
        for i in range(8):
            w = get_window(xb, i, offsets, boundary)
            wint = 0
            for j, bit in enumerate(w):
                wint |= bit << (nbits - 1 - j)
            samples.append((wint, yb[i], i))
    # check consistency
    table = {}
    for wint, ob, _ in samples:
        if wint in table and table[wint] != ob:
            return None
        table[wint] = ob
    # also report which window values are determined
    return table

def apply_ca(x, table, offsets, boundary='wrap'):
    xb = bits(x)
    nbits = len(offsets)
    yb = [0]*8
    for i in range(8):
        w = get_window(xb, i, offsets, boundary)
        wint = 0
        for j, bit in enumerate(w):
            wint |= bit << (nbits - 1 - j)
        if wint not in table:
            return None
        yb[i] = table[wint]
    return from_bits(yb)

def search_ca(pairs):
    # Try many offset sets and boundary types
    candidates = []
    # 3-cell windows
    for offsets in [(-1,0,1), (0,1,2), (-2,-1,0), (-1,0,2), (-2,0,1), (-2,0,2),
                    (-3,0,3), (-1,1,2), (0,1,3), (-2,1,3)]:
        for boundary in ('wrap','zero','one'):
            tab = find_ca_rule(pairs, offsets, boundary)
            if tab is not None and len(tab) >= 2**len(offsets) - 1:  # mostly determined
                # require ALL bits determined OR allow partial if we can determine query
                candidates.append((offsets, boundary, tab))
            elif tab is not None:
                candidates.append((offsets, boundary, tab))
    # 2-cell windows
    for offsets in [(-1,0),(0,1),(-1,1),(-2,0),(0,2),(-2,2),(-3,0),(0,3),(-3,3),(-4,0),(0,4)]:
        for boundary in ('wrap','zero','one'):
            tab = find_ca_rule(pairs, offsets, boundary)
            if tab is not None:
                candidates.append((offsets, boundary, tab))
    return candidates

def main():
    data = json.load(open(PATH))
    for p in data:
        pairs, q = parse_examples(p['prompt'])
        gold = int(p['answer'], 2)
        found = False
        for offsets, boundary, tab in search_ca(pairs):
            # full lookup table required?  Check if applying to all training inputs works
            ok = True
            for x, y in pairs:
                pred = apply_ca(x, tab, offsets, boundary)
                if pred is None or pred != y:
                    ok = False; break
            if not ok: continue
            qpred = apply_ca(q, tab, offsets, boundary)
            if qpred is None:
                continue
            tag = 'MATCH' if qpred == gold else 'MISMATCH'
            print(f"{p['id']}: {tag} CA offsets={offsets} bnd={boundary} pred={b(qpred)} gold={b(gold)} tabsize={len(tab)}/{2**len(offsets)}")
            found = True
            break
        if not found:
            print(f"{p['id']}: UNSOLVED")

if __name__ == '__main__':
    main()
