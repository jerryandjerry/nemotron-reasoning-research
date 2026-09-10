#!/usr/bin/env python3
"""Generate the two scenarios the aug never covered (they only existed in GT-false originals):
  (1) single-operator (nops=1) ambiguous  — build_ambiguous(n_fill=0)
  (2) unseen_leading_zero                  — unseen query op + leading-0 query operand
Both are underdetermined, so (as with all ambiguous) the gold = the solver's own deterministic output,
making them GT-true by construction; we still blind-solve + GT-filter + classify-check downstream.
Writes new NE5k_003xxx folders + appends rows to _manifest_missing.csv. Seed 20260613."""
import os, sys, csv, random, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import _gen_aug as G
sys.path.insert(0, f'{os.path.dirname(HERE)}/260606_Numeric_Equation')
import _gen_eq as S

def solver(text):
    ex, qn = S.parse_prompt(text)
    cot, _ = S.gen_cot({'prompt': text, 'examples': ex, 'question': qn})
    return S.extract_final_answer(cot), S.classify({'examples': ex, 'question': qn})

def build_unseen_leadzero(rng):
    """concat-guess variant (the combo missing from training, cf. 4f06e1c0): examples are concat ops only
    (so NO arithmetic is solvable -> concat-guess branch, rightward), and the QUERY uses a fresh symbol with
    a LITERAL leading-0 operand (e.g. '07') -> classify() = unseen_leading_zero. gold = the solver's own
    deterministic concat_fwd output."""
    SYMS = ['+', '-', '*'] + G.PUNCT                    # include arith glyphs (used AS concat) so arith=True is covered
    nex = rng.choice([3, 4]); sym = rng.choice(SYMS); qsym = rng.choice([s for s in SYMS if s != sym])
    var = rng.choice(G.CONCAT_V)
    lines = []
    for _ in range(nex):
        P = G.operand(False, rng); Q = G.operand(False, rng)        # rightward: written as-is, 10-99
        o = G.write_output(G.frame_result(P, Q, var), False, 'none', sym)
        lines.append(f"{P}{sym}{Q} = {o}")
    qa = '0' + str(rng.randint(1, 9)); qb = str(G.operand(False, rng))   # literal leading-0 query operand
    rng.shuffle(lines)
    text = G.INTRO + "\n" + "\n".join(lines) + f"\nNow, determine the result for: {qa}{qsym}{qb}\n"
    gold, lab = solver(text)
    if lab != 'unseen_leading_zero': raise G.Reject()
    return text, gold, 'rightward'

def main():
    n_amb = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    n_lz = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    rng = random.Random(20260613); idx = 3000; man = []
    made_a = 0; tries = 0
    while made_a < n_amb:
        tries += 1
        if tries > n_amb * 500: print(f"WARN ambiguous stalled at {made_a}"); break
        try: text, gold, label, reading, pt, signed, fmt, lz = G.build_ambiguous(rng, n_fill=0)
        except G.Reject: continue
        # verify it really is single-operator
        if len({m.group(1) for ln in text.splitlines() for m in [re.match(r'^\s*\d+(\D)\d+\s*=', ln)] if m}) != 1: continue
        idx += 1; pid = f"NE5k_{idx:06d}"; d = f"{HERE}/{pid}"; os.makedirs(d + '/track', exist_ok=True)
        open(d + '/question.txt', 'w').write(text); open(d + '/answer.txt', 'w').write(gold); open(d + '/label.txt', 'w').write(label)
        man.append({'id': pid, 'reading': reading, 'type': 'ambiguous', 'label': label, 'signed': fmt if signed else 'no', 'leadzero': lz, 'gold': gold, 'case': 'single_op_ambiguous'}); made_a += 1
    made_l = 0; tries = 0
    while made_l < n_lz:
        tries += 1
        if tries > n_lz * 500: print(f"WARN unseen_leadzero stalled at {made_l}"); break
        try: text, gold, reading = build_unseen_leadzero(rng)
        except G.Reject: continue
        idx += 1; pid = f"NE5k_{idx:06d}"; d = f"{HERE}/{pid}"; os.makedirs(d + '/track', exist_ok=True)
        open(d + '/question.txt', 'w').write(text); open(d + '/answer.txt', 'w').write(gold); open(d + '/label.txt', 'w').write('NE_aug_unseen_leading_zero')
        man.append({'id': pid, 'reading': reading, 'type': 'unseen', 'label': 'NE_aug_unseen_leading_zero', 'signed': 'no', 'leadzero': True, 'gold': gold, 'case': 'unseen_leading_zero'}); made_l += 1
    with open(f'{HERE}/_manifest_missing.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['id', 'reading', 'type', 'label', 'signed', 'leadzero', 'gold', 'case']); w.writeheader(); w.writerows(man)
    print(f"single-op ambiguous: {made_a} | unseen_leading_zero: {made_l} -> _manifest_missing.csv")

if __name__ == '__main__':
    main()
