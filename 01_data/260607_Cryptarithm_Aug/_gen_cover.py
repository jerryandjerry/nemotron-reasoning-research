#!/usr/bin/env python3
"""Targeted coverage generator: for each still-missing achievable 3-symbol merged token, CONSTRUCT a short
trainable puzzle that embeds it (all-operand token -> pure_concat RHS; operator token -> arith operand-op-
operand span), verified two ways: the tokenizer actually produces the token, AND the blind solver (gen_cot,
gold-free) recovers the planted answer under the 7680 cap. Output -> puzzles4/."""
import os, sys, re, csv, json, random, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
CRYPT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260525_Cryptarithm'
sys.path.insert(0, HERE); sys.path.insert(0, CRYPT)
import cot_generator as cg
import _gen_solutions as gs
import _gen_crypt as gcmod
import _gen_crypt_aug as G
from tokenizers import Tokenizer
TOK = G._TOK if hasattr(G, '_TOK') else Tokenizer.from_file(os.path.join(CRYPT, '..', '..', '02_train', '260512_huikang_085', 'repo', 'tokenizer.json'))
POOL = set('''!"#$%&'()/:<>?@[\\]^`{|}+-*'''); OPS = set('+-*')
POOL23 = sorted('''!"#$%&'()/:<>?@[\\]^`{|}''')
TABLE = json.load(open(os.path.join(CRYPT, 'full_merge_table.json')))['map']
ALL = set(TABLE)
FAM_OF = {'+': 'noisy_add', '-': 'noisy_subtraction', '*': 'noisy_mul'}
BASE_VAR = {'noisy_add': 'add', 'noisy_subtraction': 'absdiff', 'noisy_mul': 'mul'}

def fused(text):
    out = set()
    for line in text.strip().splitlines():
        for run in line.replace(' = ', ' ').split():
            for a, b in TOK.encode(run, add_special_tokens=False).offsets:
                seg = run[a:b]
                if len(seg) >= 2 and all(c in POOL for c in seg): out.add(seg)
    return out & ALL

def extract(text):
    bs = list(re.finditer(r'\\boxed\{', text)); ms = []
    for i, m in enumerate(bs):
        end = bs[i + 1].start() if i + 1 < len(bs) else len(text)
        seg = text[m.end():end]; lb = seg.rfind('}'); ms.append(seg[:lb] if lb != -1 else seg)
    ne = [m.strip() for m in ms if m.strip()]
    return ne[-1] if ne else 'NF'

def try_blind(text, gold, tmp):
    os.makedirs(os.path.join(tmp, 'track'), exist_ok=True)
    open(os.path.join(tmp, 'question.txt'), 'w').write(text)
    try:
        cot = gcmod.gen_cot(tmp)
    except Exception:
        return None
    box = extract(cot); ntok = len(TOK.encode(cot, add_special_tokens=False).ids)
    if box == gold and ntok < 7680:
        return cot, ntok
    return None

def build_concat(T, rng):
    """all-operand T (2-3 symbols) embedded in a pure_concat RHS."""
    others = [s for s in POOL23 if s not in T]; rng.shuffle(others)
    operand_syms = list(dict.fromkeys(list(T) + others))[:10]
    digits = list(range(10)); rng.shuffle(digits)
    # leading symbols must be nonzero; ensure T[0] not mapped to 0
    dig2sym = {d: operand_syms[i] for i, d in enumerate(digits)}
    sym2dig = {v: k for k, v in dig2sym.items()}
    if sym2dig[T[0]] == 0: return None
    opsyms = [s for s in POOL23 if s not in operand_syms]; rng.shuffle(opsyms)
    o1, o2 = opsyms[0], opsyms[1]
    mode = 'standard'
    def operand(a, b):  # symbols a,b -> 2-sym operand string (standard: a=tens)
        return a + b
    # embedding line (concat fwd): operand_a = T0 T1, operand_b = T2 X  => RHS = T0 T1 T2 X
    X = rng.choice([s for s in operand_syms if s not in T])
    if len(T) == 3:
        la, lb = T[0] + T[1], T[2] + X
    else:
        la, lb = T[0] + T[1], rng.choice(operand_syms) + X
    if sym2dig[la[0]] == 0 or sym2dig[lb[0]] == 0: return None
    lines = [f"{la}{o1}{lb} = {la}{lb}"]
    # one more concat example (op2) + ensure op2 used; query uses o1 (concat)
    for _ in range(2):
        a = dig2sym[rng.randint(1, 9)] + dig2sym[rng.randrange(10)]
        b = dig2sym[rng.randint(1, 9)] + dig2sym[rng.randrange(10)]
        op = rng.choice([o1, o2])
        rhs = a + b if op == o1 else (b + a)
        lines.append(f"{a}{op}{b} = {rhs}")
    rng.shuffle(lines)
    qa = dig2sym[rng.randint(1, 9)] + dig2sym[rng.randrange(10)]
    qb = dig2sym[rng.randint(1, 9)] + dig2sym[rng.randrange(10)]
    gold = qa + qb
    text = ("In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
            "Below are a few examples:\n" + "\n".join(lines) + f"\nNow, determine the result for: {qa}{o1}{qb}\n")
    return text, gold, 'pure_concat'

def build_arith(T, rng):
    """operator token T (op glyph at some index) embedded as an arith operand-op-operand span."""
    oi = [i for i, c in enumerate(T) if c in OPS]
    if len(oi) != 1: return None
    i = oi[0]; opsym = T[i]; fam = FAM_OF[opsym]; var = BASE_VAR[fam]
    flank = [c for j, c in enumerate(T) if j != i]
    others = [s for s in POOL23 if s not in flank]; rng.shuffle(others)
    operand_syms = list(dict.fromkeys(flank + others))[:10]
    digits = list(range(10)); rng.shuffle(digits)
    dig2sym = {d: operand_syms[k] for k, d in enumerate(digits)}
    sym2dig = {v: k for k, v in dig2sym.items()}
    mode = 'standard'
    # place op at lhs pos3; flanks adjacent. i==1: d2=T0, d3=T2 ; i==0: d3=T1,d4=T2 ; i==2: d1=T0,d2=T1
    def operand_val(a, b): return 10 * sym2dig[a] + sym2dig[b]
    for _try in range(200):
        d1 = dig2sym[rng.randint(1, 9)]; d4 = dig2sym[rng.randrange(10)]
        if i == 1:   da, db = d1 + T[0], T[2] + d4
        elif i == 0: da, db = d1 + dig2sym[rng.randrange(10)], T[1] + T[2]
        else:        da, db = T[0] + T[1], dig2sym[rng.randint(1,9)] + d4
        if sym2dig[da[0]] == 0 or sym2dig[db[0]] == 0: continue
        a, b = operand_val(da[0], da[1]), operand_val(db[0], db[1])
        r = cg.apply_op(var, a, b)
        body, miss = gs.num_to_sym(r, dig2sym, opsym, mode, 'none')
        if miss or (r < 0): continue
        line1 = f"{da}{opsym}{db} = {body}"
        # 2nd example for same op (pin variant), random operands
        ex2 = None
        for _ in range(200):
            x = dig2sym[rng.randint(1,9)]+dig2sym[rng.randrange(10)]; y = dig2sym[rng.randint(1,9)]+dig2sym[rng.randrange(10)]
            rr = cg.apply_op(var, operand_val(x[0],x[1]), operand_val(y[0],y[1]))
            bb, mm = gs.num_to_sym(rr, dig2sym, opsym, mode, 'none')
            if mm or rr < 0: continue
            ex2 = f"{x}{opsym}{y} = {bb}"; break
        if not ex2: continue
        # query (same op)
        qx = dig2sym[rng.randint(1,9)]+dig2sym[rng.randrange(10)]; qy = dig2sym[rng.randint(1,9)]+dig2sym[rng.randrange(10)]
        rq = cg.apply_op(var, operand_val(qx[0],qx[1]), operand_val(qy[0],qy[1]))
        gold, mq = gs.num_to_sym(rq, dig2sym, opsym, mode, 'none')
        if mq or rq < 0: continue
        lines = [line1, ex2]; rng.shuffle(lines)
        text = ("In Alice's Wonderland, a secret set of transformation rules is applied to equations. "
                "Below are a few examples:\n" + "\n".join(lines) + f"\nNow, determine the result for: {qx}{opsym}{qy}\n")
        return text, gold, 'arithmetic'
    return None

def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--tokensfile', required=True); ap.add_argument('--idoff', type=int, default=300000)
    ap.add_argument('--append', action='store_true')
    a = ap.parse_args()
    targets = json.load(open(a.tokensfile))
    out = os.path.join(HERE, 'puzzles4')
    if os.path.exists(out) and not a.append: shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)
    existing = []
    mpath = os.path.join(out, '_manifest.csv')
    if a.append and os.path.exists(mpath):
        existing = list(csv.DictReader(open(mpath)))
    tmp = os.path.join(HERE, '_cover_tmp')
    rng = random.Random(20260610)
    manifest = []; idx = 0; covered = []
    for T in targets:
        has_op = any(c in OPS for c in T)
        builder = build_arith if has_op else build_concat
        ok = False
        for _att in range(400):
            res = builder(T, rng)
            if not res: continue
            text, gold, sub = res
            if T not in fused(text): continue
            tb = try_blind(text, gold, tmp)
            if not tb: continue
            cot, ntok = tb
            idx += 1; pid = f"crypt_aug_{a.idoff + idx:06d}"
            d = os.path.join(out, pid); os.makedirs(os.path.join(d, 'track'), exist_ok=True)
            open(os.path.join(d, 'question.txt'), 'w').write(text)
            open(os.path.join(d, 'answer.txt'), 'w').write(gold)
            open(os.path.join(d, 'label.txt'), 'w').write(sub)
            open(os.path.join(d, 'track', 'tree_cot.txt'), 'w').write(cot)
            from _gen_batch import COARSE
            manifest.append({'id': pid, 'subtype': sub, 'category': COARSE[sub], 'gold': gold,
                             'reading': 'standard', 'signed': 'no', 'n_ops': 2, 'n_ex': 2, 'arith_syms': sum(c in OPS for c in (text.split('Now')[0])), 'has_concat': int(sub == 'pure_concat'),
                             'target_token': T, 'ntok': ntok})
            covered.append(T); ok = True; break
        print(f"  {T!r:8} {'COVERED' if ok else 'FAILED (BPE-unreachable in a short trainable puzzle)'}")
    allman = existing + manifest
    if allman:
        with open(os.path.join(out, '_manifest.csv'), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(allman[0].keys())); w.writeheader(); w.writerows(allman)
    if os.path.exists(tmp): shutil.rmtree(tmp)
    print(f"\ncovered {len(covered)}/{len(targets)} targets -> {out}")

if __name__ == '__main__':
    main()
