"""Exhaustive rule searcher for 8-bit puzzles.

Bit convention from prompt: Bit 0 = MSB. We'll parse strings as standard MSB-left binary.
We'll convert each 8-bit string to an int 0..255 where the MSB is the leftmost char.

We try many parametric rule families and check against all examples.
"""
import json, itertools, re, sys

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

def bits_msb(x):
    # returns list b[0..7] where b[0] is MSB
    s = format(x & 0xFF, '08b')
    return [int(c) for c in s]

def from_bits_msb(bs):
    return int(''.join(str(b) for b in bs), 2)

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

# Build a large library of unary candidate rules
def candidate_rules():
    rules = []
    # identity, not, reverse, swap_nibbles
    rules.append(("identity", lambda x: x))
    rules.append(("not", lambda x: (~x) & 0xFF))
    rules.append(("reverse_bits", reverse_bits))
    rules.append(("swap_nibbles", swap_nibbles))
    rules.append(("not_reverse", lambda x: (~reverse_bits(x)) & 0xFF))
    rules.append(("reverse_not", lambda x: reverse_bits((~x) & 0xFF)))
    # rotations
    for k in range(1, 8):
        rules.append((f"rotl_{k}", lambda x, k=k: rotl(x, k)))
        rules.append((f"rotr_{k}", lambda x, k=k: rotr(x, k)))
    # shifts with fill
    for k in range(1, 8):
        for f in (0, 1):
            rules.append((f"shl_{k}_f{f}", lambda x, k=k, f=f: shl(x, k, f)))
            rules.append((f"shr_{k}_f{f}", lambda x, k=k, f=f: shr(x, k, f)))
    # XOR with constant
    for c in range(256):
        rules.append((f"xor_{c:02x}", lambda x, c=c: x ^ c))
    # XOR with shifted self
    for k in range(1, 8):
        for shift_name, shift_fn in [
            (f"rotl{k}", lambda x, k=k: rotl(x, k)),
            (f"rotr{k}", lambda x, k=k: rotr(x, k)),
            (f"shl{k}_f0", lambda x, k=k: shl(x, k, 0)),
            (f"shr{k}_f0", lambda x, k=k: shr(x, k, 0)),
            (f"shl{k}_f1", lambda x, k=k: shl(x, k, 1)),
            (f"shr{k}_f1", lambda x, k=k: shr(x, k, 1)),
        ]:
            rules.append((f"xor_self_{shift_name}", lambda x, sf=shift_fn: x ^ sf(x)))
            rules.append((f"and_self_{shift_name}", lambda x, sf=shift_fn: x & sf(x)))
            rules.append((f"or_self_{shift_name}", lambda x, sf=shift_fn: x | sf(x)))
            rules.append((f"xnor_self_{shift_name}", lambda x, sf=shift_fn: ((~(x ^ sf(x))) & 0xFF)))
    # arithmetic
    for c in range(-8, 9):
        rules.append((f"add_{c}", lambda x, c=c: (x + c) & 0xFF))
        rules.append((f"sub_from_{c & 0xFF}", lambda x, c=c: (c - x) & 0xFF))
    rules.append(("mul2", lambda x: (x * 2) & 0xFF))
    rules.append(("mul3", lambda x: (x * 3) & 0xFF))
    rules.append(("mul5", lambda x: (x * 5) & 0xFF))
    rules.append(("mul7", lambda x: (x * 7) & 0xFF))
    rules.append(("mul9", lambda x: (x * 9) & 0xFF))
    rules.append(("mul11", lambda x: (x * 11) & 0xFF))
    rules.append(("mul13", lambda x: (x * 13) & 0xFF))
    rules.append(("mul15", lambda x: (x * 15) & 0xFF))
    rules.append(("mul17", lambda x: (x * 17) & 0xFF))
    rules.append(("mul31", lambda x: (x * 31) & 0xFF))
    rules.append(("mul33", lambda x: (x * 33) & 0xFF))
    # negate
    rules.append(("neg", lambda x: (-x) & 0xFF))
    rules.append(("not_plus1", lambda x: ((~x)+1) & 0xFF))
    # per-nibble swaps and operations
    rules.append(("hi_nib_to_low_zero_hi", lambda x: (x >> 4) & 0xFF))
    rules.append(("low_nib_to_hi_zero_low", lambda x: (x << 4) & 0xFF))
    return rules

CANDIDATES = candidate_rules()

def test_rule(fn, pairs):
    for x, y in pairs:
        if fn(x) != y:
            return False
    return True

def find_rule(pairs):
    for name, fn in CANDIDATES:
        if test_rule(fn, pairs):
            return name, fn
    return None, None

def main():
    data = json.load(open(PATH))
    results = []
    for p in data:
        pairs, query = parse_examples(p['prompt'])
        name, fn = find_rule(pairs)
        gold = int(p['answer'], 2)
        if fn is None:
            results.append({
                'id': p['id'], 'pairs': len(pairs),
                'solved': False, 'query': b(query), 'gold': b(gold), 'rule': None,
            })
        else:
            pred = fn(query)
            results.append({
                'id': p['id'], 'pairs': len(pairs),
                'solved': pred == gold, 'query': b(query), 'gold': b(gold),
                'pred': b(pred), 'rule': name,
            })
    for r in results:
        print(r)

if __name__ == '__main__':
    main()
