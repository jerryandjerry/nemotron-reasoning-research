"""Broader CA search: any number of offsets up to 4 cells.
Also try POSITION-DEPENDENT rules: y[i] = f_i(window) where each position has
its own function. This catches things like 'top nibble negates, bottom shifts'.
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
    return [(x >> (7 - i)) & 1 for i in range(8)]

def from_bits(bs):
    v = 0
    for i, bit in enumerate(bs):
        v |= (bit & 1) << (7 - i)
    return v

def get_window(xb, i, offsets, boundary):
    n = 8
    out = []
    for o in offsets:
        j = i + o
        if boundary == 'wrap':
            out.append(xb[j % n])
        elif boundary == 'zero':
            out.append(xb[j] if 0 <= j < n else 0)
        elif boundary == 'one':
            out.append(xb[j] if 0 <= j < n else 1)
    return out

def try_position_independent(pairs, offsets, boundary):
    """Find a position-independent boolean table over windows."""
    nbits = len(offsets)
    table = {}
    for x, y in pairs:
        xb = bits(x); yb = bits(y)
        for i in range(8):
            w = get_window(xb, i, offsets, boundary)
            wi = 0
            for j, b_ in enumerate(w):
                wi |= b_ << (nbits - 1 - j)
            if wi in table:
                if table[wi] != yb[i]:
                    return None
            else:
                table[wi] = yb[i]
    return table

def apply_pi(x, table, offsets, boundary):
    xb = bits(x)
    nbits = len(offsets)
    yb = [0]*8
    for i in range(8):
        w = get_window(xb, i, offsets, boundary)
        wi = 0
        for j, b_ in enumerate(w):
            wi |= b_ << (nbits - 1 - j)
        if wi not in table:
            return None
        yb[i] = table[wi]
    return from_bits(yb)

def try_position_dependent(pairs, offsets, boundary):
    """y[i] = f_i(window), each position has its own table."""
    nbits = len(offsets)
    tables = [{} for _ in range(8)]
    for x, y in pairs:
        xb = bits(x); yb = bits(y)
        for i in range(8):
            w = get_window(xb, i, offsets, boundary)
            wi = 0
            for j, b_ in enumerate(w):
                wi |= b_ << (nbits - 1 - j)
            if wi in tables[i] and tables[i][wi] != yb[i]:
                return None
            tables[i][wi] = yb[i]
    return tables

def apply_pd(x, tables, offsets, boundary):
    xb = bits(x)
    nbits = len(offsets)
    yb = [0]*8
    for i in range(8):
        w = get_window(xb, i, offsets, boundary)
        wi = 0
        for j, b_ in enumerate(w):
            wi |= b_ << (nbits - 1 - j)
        if wi not in tables[i]:
            return None
        yb[i] = tables[i][wi]
    return from_bits(yb)

def gen_offsets():
    offsets_list = []
    # singles
    for o in range(-4, 5):
        offsets_list.append((o,))
    # pairs
    for a in range(-4, 5):
        for d in range(1, 5):
            offsets_list.append((a, a+d))
    # triples
    for a in range(-3, 3):
        for d1 in range(1, 4):
            for d2 in range(d1+1, d1+4):
                offsets_list.append((a, a+d1, a+d2))
    # standard 3-cell:
    offsets_list.append((-1,0,1))
    # dedupe
    seen = set()
    uniq = []
    for o in offsets_list:
        t = tuple(o)
        if t in seen: continue
        seen.add(t); uniq.append(t)
    return uniq

def main():
    data = json.load(open(PATH))
    OFFSET_SETS = gen_offsets()
    print(f"Trying {len(OFFSET_SETS)} offset sets x 3 boundaries x [PI,PD]")
    for p in data:
        pairs, q = parse_examples(p['prompt'])
        gold = int(p['answer'], 2)
        found = None
        # Try position-independent first (simpler)
        for offsets in OFFSET_SETS:
            for boundary in ('wrap','zero','one'):
                tab = try_position_independent(pairs, offsets, boundary)
                if tab is None: continue
                # verify training works
                ok = True
                for x, y in pairs:
                    pred = apply_pi(x, tab, offsets, boundary)
                    if pred != y: ok = False; break
                if not ok: continue
                qpred = apply_pi(q, tab, offsets, boundary)
                if qpred is None: continue
                tag = 'MATCH' if qpred == gold else 'MISMATCH'
                found = ('PI', offsets, boundary, tag, qpred, len(tab), 2**len(offsets))
                break
            if found: break
        if found:
            tag = found[3]; rule = f"PI off={found[1]} bnd={found[2]} table={found[5]}/{found[6]}"
            print(f"{p['id']}: {tag} {rule} pred={b(found[4])} gold={b(gold)}")
            continue
        # Try position-dependent (more permissive)
        found_pd = None
        for offsets in OFFSET_SETS:
            for boundary in ('wrap','zero','one'):
                tabs = try_position_dependent(pairs, offsets, boundary)
                if tabs is None: continue
                # verify training
                ok = True
                for x, y in pairs:
                    pred = apply_pd(x, tabs, offsets, boundary)
                    if pred != y: ok = False; break
                if not ok: continue
                qpred = apply_pd(q, tabs, offsets, boundary)
                if qpred is None: continue
                tag = 'MATCH' if qpred == gold else 'MISMATCH'
                found_pd = ('PD', offsets, boundary, tag, qpred, sum(len(t) for t in tabs))
                break
            if found_pd: break
        if found_pd:
            print(f"{p['id']}: {found_pd[3]} PD off={found_pd[1]} bnd={found_pd[2]} entries={found_pd[5]} pred={b(found_pd[4])} gold={b(gold)}")
        else:
            print(f"{p['id']}: UNSOLVED")

if __name__ == '__main__':
    main()
