def b2i(s): return int(s, 2)
def i2b(n): return format(n & 0xFF, '08b')
def reverse_bits(s): return s[::-1]
def swap_nibbles(s): return s[4:] + s[:4]
def rotl(s, k):
    k %= 8
    return s[k:] + s[:k]
def rotr(s, k):
    k %= 8
    if k == 0: return s
    return s[-k:] + s[:-k]
def shl(s, k, fill='0'):
    if k >= 8: return fill*8
    return s[k:] + fill*k
def shr(s, k, fill='0'):
    if k >= 8: return fill*8
    return fill*k + s[:-k] if k > 0 else s
def NOT(s): return ''.join('1' if c=='0' else '0' for c in s)
def XOR(a, b): return ''.join('1' if x!=y else '0' for x,y in zip(a,b))
def AND(a, b): return ''.join('1' if x=='1' and y=='1' else '0' for x,y in zip(a,b))
def OR(a, b): return ''.join('1' if x=='1' or y=='1' else '0' for x,y in zip(a,b))

def test_rule(rule_fn, examples):
    out = []
    for inp, expected in examples:
        got = rule_fn(inp)
        out.append((inp, expected, got, got == expected))
    return out

def show(results):
    allok = True
    for inp, expected, got, ok in results:
        mark = "OK" if ok else "FAIL"
        if not ok: allok = False
        print(f"  {inp} -> {expected} (got {got}) {mark}")
    print(f"  ALL OK" if allok else "  some failed")
    return allok
