"""More aggressive rule search:
- u(x) op v(x) op w(x)  three-way combos
- masks via popcount/parity
- x +/- constant then bit op
"""
import json, re

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
def rotl(x, k): k %= 8; return ((x << k) | (x >> (8-k))) & 0xFF if k else x & 0xFF
def rotr(x, k): k %= 8; return ((x >> k) | (x << (8-k))) & 0xFF if k else x & 0xFF
def shl(x, k, fill=0):
    if k >= 8: return 0xFF if fill else 0
    res = (x << k) & 0xFF
    if fill: res |= (1 << k) - 1
    return res
def shr(x, k, fill=0):
    if k >= 8: return 0xFF if fill else 0
    res = x >> k
    if fill: res |= (((1 << k) - 1) << (8 - k)) & 0xFF
    return res & 0xFF
def revbits(x): return int(format(x & 0xFF, '08b')[::-1], 2)
def swap_nib(x): return ((x << 4) | (x >> 4)) & 0xFF
def popcount(x): return bin(x & 0xFF).count('1')

UNARY = []
UNARY.append(("id", lambda x: x))
UNARY.append(("not", lambda x: (~x)&0xFF))
UNARY.append(("rev", revbits))
UNARY.append(("swap", swap_nib))
UNARY.append(("not_rev", lambda x: (~revbits(x))&0xFF))
UNARY.append(("rev_not", lambda x: revbits((~x)&0xFF)))
for k in range(1, 8):
    UNARY.append((f"rotl{k}", lambda x, k=k: rotl(x, k)))
    UNARY.append((f"rotr{k}", lambda x, k=k: rotr(x, k)))
    for f in (0,1):
        UNARY.append((f"shl{k}f{f}", lambda x, k=k, f=f: shl(x, k, f)))
        UNARY.append((f"shr{k}f{f}", lambda x, k=k, f=f: shr(x, k, f)))

OPS = [
    ('^', lambda a,b: a^b),
    ('&', lambda a,b: a&b),
    ('|', lambda a,b: a|b),
    ('+', lambda a,b: (a+b)&0xFF),
    ('-', lambda a,b: (a-b)&0xFF),
]

def gen_rules():
    rules = []
    # already covered: pairs of unaries with op
    # add: three-way:  u OP1 v OP2 w  (left-associative)
    # too expensive, skip; do (u OP v) const-mask
    for un, uf in UNARY:
        for vn, vf in UNARY:
            for on, of in OPS:
                rules.append((f"{un}{on}{vn}", lambda x, uf=uf, vf=vf, of=of: of(uf(x), vf(x))))
    # u(x) op v(x) then xor constant
    masks = list(range(256))
    return rules, masks

RULES, MASKS = gen_rules()
print(f"#rules={len(RULES)}")

def test_rule(fn, pairs):
    for x, y in pairs:
        if fn(x) != y: return False
    return True

def find_with_mask(pairs):
    # try (u(x) op v(x)) ^ c
    for rn, rf in RULES:
        # compute residual mask from first example
        x0, y0 = pairs[0]
        c = rf(x0) ^ y0
        ok = True
        for x, y in pairs[1:]:
            if (rf(x) ^ c) != y:
                ok = False; break
        if ok:
            return f"({rn})^{c:02x}", lambda x, rf=rf, c=c: rf(x) ^ c
    return None, None

def find_simple(pairs):
    for rn, rf in RULES:
        if test_rule(rf, pairs):
            return rn, rf
    return None, None

def main():
    data = json.load(open(PATH))
    for p in data:
        pairs, q = parse_examples(p['prompt'])
        gold = int(p['answer'], 2)
        n, f = find_simple(pairs)
        if f is None:
            n, f = find_with_mask(pairs)
        if f is None:
            print(f"{p['id']}: UNSOLVED  query={b(q)} gold={b(gold)}")
        else:
            pred = f(q)
            tag = "MATCH" if pred == gold else "MISMATCH"
            print(f"{p['id']}: {tag} rule={n} pred={b(pred)} gold={b(gold)}")

if __name__ == '__main__':
    main()
