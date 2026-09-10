"""Per-puzzle analyzer that tries more rule families with parameters."""
import json, re, itertools

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

def rotl(x, k):
    k %= 8
    return ((x << k) | (x >> (8-k))) & 0xFF if k else x & 0xFF

def rotr(x, k):
    k %= 8
    return ((x >> k) | (x << (8-k))) & 0xFF if k else x & 0xFF

def shl(x, k, fill=0):
    if k >= 8:
        return (0xFF if fill else 0) & 0xFF
    res = (x << k) & 0xFF
    if fill:
        mask = (1 << k) - 1
        res |= mask
    return res

def shr(x, k, fill=0):
    if k >= 8:
        return (0xFF if fill else 0) & 0xFF
    res = x >> k
    if fill:
        mask = ((1 << k) - 1) << (8 - k)
        res |= mask
    return res & 0xFF

def reverse_bits(x):
    s = format(x & 0xFF, '08b')
    return int(s[::-1], 2)

def swap_nibbles(x):
    return ((x << 4) | (x >> 4)) & 0xFF

def popcount(x):
    return bin(x & 0xFF).count('1')

def parity(x):
    return popcount(x) & 1

# Generate composite unary rules: f then g
SHIFTS = []
for k in range(1, 8):
    SHIFTS.append((f"rotl{k}", lambda x, k=k: rotl(x, k)))
    SHIFTS.append((f"rotr{k}", lambda x, k=k: rotr(x, k)))
    for f in (0, 1):
        SHIFTS.append((f"shl{k}f{f}", lambda x, k=k, f=f: shl(x, k, f)))
        SHIFTS.append((f"shr{k}f{f}", lambda x, k=k, f=f: shr(x, k, f)))

UNARY = [("id", lambda x: x), ("not", lambda x: (~x)&0xFF),
         ("rev", reverse_bits), ("swap", swap_nibbles),
         ("not_rev", lambda x: (~reverse_bits(x))&0xFF),
         ("rev_not", lambda x: reverse_bits((~x)&0xFF))]
UNARY.extend(SHIFTS)

def gen_simple_rules():
    rules = []
    for n, f in UNARY:
        rules.append((n, f))
    # XOR constant + unary
    for cn, cf in UNARY:
        for c in range(256):
            rules.append((f"({cn})^{c:02x}", lambda x, cf=cf, c=c: cf(x) ^ c))
    # u(x) op v(x)
    OPS = [('^', lambda a, b: a ^ b),
           ('&', lambda a, b: a & b),
           ('|', lambda a, b: a | b),
           ('~^', lambda a, b: (~(a ^ b)) & 0xFF),
           ('+', lambda a, b: (a + b) & 0xFF),
           ('-', lambda a, b: (a - b) & 0xFF)]
    for un, uf in UNARY:
        for vn, vf in UNARY:
            for on, of in OPS:
                rules.append((f"{un}{on}{vn}", lambda x, uf=uf, vf=vf, of=of: of(uf(x), vf(x))))
    return rules

def search(pairs, candidates):
    for n, f in candidates:
        ok = True
        for x, y in pairs:
            if f(x) != y:
                ok = False; break
        if ok:
            return n, f
    return None, None

def main():
    data = json.load(open(PATH))
    rules = gen_simple_rules()
    print(f"Trying {len(rules)} rules per puzzle")
    for p in data:
        pairs, q = parse_examples(p['prompt'])
        gold = int(p['answer'], 2)
        n, f = search(pairs, rules)
        if f is None:
            print(f"{p['id']}: UNSOLVED  query={b(q)} gold={b(gold)}")
        else:
            pred = f(q)
            print(f"{p['id']}: rule={n}  pred={b(pred)} gold={b(gold)} match={pred==gold}")

if __name__ == '__main__':
    main()
