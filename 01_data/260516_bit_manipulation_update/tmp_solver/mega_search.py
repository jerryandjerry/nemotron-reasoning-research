"""Massive search across all puzzles.
Trying many 3-way and 4-way combinations."""
import sys, itertools
from explore_one import (UNARY, OPS, PUZZLES, b, rotl, rotr, shl, shr, rev, swap)

# Add more unary transforms
EXTRA = []
EXTRA.append(("gray", lambda x: x ^ (x >> 1)))
EXTRA.append(("gray2", lambda x: x ^ (x << 1) & 0xFF))
EXTRA.append(("popcount", lambda x: bin(x).count('1')))
EXTRA.append(("popcount_x", lambda x: bin(x).count('1') * x & 0xFF))
EXTRA.append(("hi_nib", lambda x: x >> 4))
EXTRA.append(("lo_nib", lambda x: x & 0xF))
EXTRA.append(("hi_dup", lambda x: ((x >> 4) << 4) | (x >> 4)))
EXTRA.append(("lo_dup", lambda x: ((x & 0xF) << 4) | (x & 0xF)))
ALL_UNARY = UNARY + EXTRA

def main():
    pid = sys.argv[1]
    pairs, q, gold = PUZZLES[pid]
    print(f"=== {pid} === query={b(q)} gold={b(gold)}")

    # (u op1 v) op2 (w op3 z)  -- 4 unaries, 3 ops
    # too big. Try: (u op1 v) op2 w
    matches = []
    seen = set()
    cnt = 0
    for un, uf in ALL_UNARY:
        for vn, vf in ALL_UNARY:
            for on, of in OPS:
                # check single
                fn = lambda x, uf=uf, vf=vf, of=of: of(uf(x), vf(x)) & 0xFF
                if all(fn(x) == y for x, y in pairs):
                    matches.append((f"{un}{on}{vn}", fn))
                # check XOR mask
                x0, y0 = pairs[0]
                c = (of(uf(x0), vf(x0)) ^ y0) & 0xFF
                if c:
                    fn2 = lambda x, fn=fn, c=c: fn(x) ^ c
                    if all(fn2(x) == y for x, y in pairs):
                        matches.append((f"({un}{on}{vn})^{c:02x}", fn2))
                # check (u op v) op w
                for wn, wf in ALL_UNARY:
                    for o2n, o2f in OPS:
                        fn3 = lambda x, uf=uf, vf=vf, wf=wf, of=of, o2f=o2f: o2f(of(uf(x), vf(x)), wf(x)) & 0xFF
                        if all(fn3(x) == y for x, y in pairs):
                            key = (un, on, vn, o2n, wn)
                            if key not in seen:
                                seen.add(key)
                                matches.append((f"({un}{on}{vn}){o2n}{wn}", fn3))
                                if len(matches) > 100: break
                    if len(matches) > 100: break
                if len(matches) > 100: break
            if len(matches) > 100: break
        if len(matches) > 100: break

    if not matches:
        print("NO MATCHES")
        return
    for n, fn in matches[:30]:
        pred = fn(q)
        tag = 'MATCH' if pred == gold else 'MISMATCH'
        print(f"  {tag} {n}: pred={b(pred)} gold={b(gold)}")
    print(f"total: {len(matches)}")

if __name__ == '__main__':
    main()
