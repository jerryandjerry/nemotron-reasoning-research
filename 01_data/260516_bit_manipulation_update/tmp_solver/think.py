"""Detailed per-puzzle analysis with focused hypothesis."""
from explore_one import PUZZLES, b, rotl, rotr, shl, shr, rev, swap

def popcount(x): return bin(x & 0xFF).count('1')

def show(pid):
    pairs, q, gold = PUZZLES[pid]
    print(f"\n=== {pid} ===")
    print(f"query={b(q)} ({q}) gold={b(gold)} ({gold})")
    print()
    for x, y in pairs:
        # Try: x AND (x>>1), x AND (x<<1), etc.
        ax = x & (x >> 1)
        bx = x & (x << 1) & 0xFF
        cx = x & (x >> 2)
        dx = x & (x << 2) & 0xFF
        # is y a simple shift of x?
        notes = []
        for k in range(1,8):
            if y == rotl(x, k): notes.append(f"rotl{k}")
            if y == rotr(x, k): notes.append(f"rotr{k}")
            if y == shl(x, k, 0): notes.append(f"shl{k}f0")
            if y == shr(x, k, 0): notes.append(f"shr{k}f0")
        if y == (~x) & 0xFF: notes.append("NOT")
        if y == rev(x): notes.append("REV")
        if y == swap(x): notes.append("SWAP")
        print(f"  {b(x)} -> {b(y)}  {' '.join(notes)}")
        if notes:
            print(f"        x&(x>>1)={b(ax)} x&(x<<1)={b(bx)}")

for pid in ['7b38ff97', '7c99ca45', '7c4db527', '7a3ed1ef']:
    show(pid)
