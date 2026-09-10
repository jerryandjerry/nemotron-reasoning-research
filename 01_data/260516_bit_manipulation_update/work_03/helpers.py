"""Helper functions for testing bit-manipulation hypotheses."""

def b(s):
    """binary string to int"""
    return int(s, 2)

def s(n):
    """int to 8-bit binary string"""
    return format(n & 0xFF, '08b')

def rol(n, k):
    """rotate left k bits"""
    k %= 8
    return ((n << k) | (n >> (8 - k))) & 0xFF if k else n & 0xFF

def ror(n, k):
    """rotate right k bits"""
    k %= 8
    return ((n >> k) | (n << (8 - k))) & 0xFF if k else n & 0xFF

def rev(n):
    """reverse 8 bits"""
    r = 0
    for i in range(8):
        if n & (1 << i):
            r |= 1 << (7 - i)
    return r

def swap_nib(n):
    """swap nibbles"""
    return ((n & 0x0F) << 4) | ((n & 0xF0) >> 4)

def popcount(n):
    return bin(n & 0xFF).count('1')

def shr(n, k):
    return (n >> k) & 0xFF

def shl(n, k):
    return (n << k) & 0xFF

def NOT(n):
    return (~n) & 0xFF

def test(rule, examples):
    """Test a rule against examples. Returns (pass_count, total, fails)."""
    fails = []
    for inp, out in examples:
        ni = b(inp)
        no = b(out)
        try:
            pred = rule(ni) & 0xFF
        except Exception as e:
            fails.append((inp, out, f"ERR:{e}"))
            continue
        if pred != no:
            fails.append((inp, out, s(pred)))
    return len(examples) - len(fails), len(examples), fails
