"""For one puzzle, search 3-way bitwise expressions: (u(x) op1 v(x)) op2 w(x)."""
import sys
from explore_one import (UNARY, OPS, PUZZLES, b, rotl, rotr, shl, shr, rev, swap)

def main():
    pid = sys.argv[1]
    pairs, q, gold = PUZZLES[pid]
    print(f"=== {pid} === query={b(q)} gold={b(gold)}  examples={len(pairs)}")
    matches = []
    seen = set()
    # cap candidate count by trying common unaries first
    BASE_UNARY = UNARY[:6] + [u for u in UNARY if 'rot' in u[0] or 'sh' in u[0]]
    for un, uf in BASE_UNARY:
        for vn, vf in BASE_UNARY:
            for wn, wf in BASE_UNARY:
                for on, of in OPS:
                    for o2n, o2f in OPS:
                        fn = lambda x, uf=uf, vf=vf, wf=wf, of=of, o2f=o2f: o2f(of(uf(x), vf(x)), wf(x)) & 0xFF
                        ok = True
                        for x, y in pairs:
                            if fn(x) != y:
                                ok = False; break
                        if ok:
                            key = (un, on, vn, o2n, wn)
                            if key in seen: continue
                            seen.add(key)
                            matches.append((f"({un}{on}{vn}){o2n}{wn}", fn))
                            if len(matches) >= 50:
                                break
                    if len(matches) >= 50: break
                if len(matches) >= 50: break
            if len(matches) >= 50: break
        if len(matches) >= 50: break

    for n, fn in matches:
        pred = fn(q)
        tag = 'MATCH' if pred == gold else 'MISMATCH'
        print(f"  {tag} {n}: pred={b(pred)} gold={b(gold)}")
    print(f"total: {len(matches)} candidates")

if __name__ == '__main__':
    main()
