#!/usr/bin/env python3
"""Coverage-greedy + balanced final selection of the 1000-row trainable set.
Keep ALL 307 original trainable (fixed). Then select aug trainable to hit per-category targets
(arith 300, le 300, each concat type 100 — incl originals), with two objectives:
  Phase 1 (coverage): greedily pick aug puzzles that add NEW achievable merged tokens (2-sym already 100%;
          the real gap is 3-sym), respecting per-category caps -> maximize transcription coverage.
  Phase 2 (balance):  fill remaining category slots stratified by (reading, sign) so no incidental feature
          skews -> balance across readings/sign within each category.
Writes _final_selection.txt and prints distribution + coverage + per-axis balance tables."""
import os, sys, re, csv, json, math, glob, random
from collections import defaultdict, Counter
csv.field_size_limit(10 ** 7)
HERE = os.path.dirname(os.path.abspath(__file__))
CRYPT = '/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/01_data/260525_Cryptarithm'
sys.path.insert(0, CRYPT)
from tokenizers import Tokenizer
TOK = Tokenizer.from_file('/Users/home/Library/CloudStorage/SynologyDrive-1/00_Kaggle/2026_Nemotron/02_train/260512_huikang_085/repo/tokenizer.json')
import cot_generator as cg
POOL = set('''!"#$%&'()/:<>?@[\\]^`{|}+-*'''); OPS = set('+-*')
TABLE = json.load(open(os.path.join(CRYPT, 'full_merge_table.json')))['map']
ALL = set(TABLE); ACH = {t for t in ALL if sum(c in OPS for c in t) <= 1}
ACH2 = {t for t in ACH if len(t) == 2}; ACH3 = {t for t in ACH if len(t) == 3}
BASE = os.path.join(CRYPT, '260607_TrainData/260607_Cryptarithm_gtTrue.csv')
TARGET = {'arithmetic': 300, 'little_endian': 300, 'pure_concat': 100,
          'mixed_concat': 100, 'mixed_concat_little_endian': 100, 'query_unseen_concat': 100}
ORDER = list(TARGET)
PATH = {'arithmetic': 'deduce', 'little_endian': 'deduce', 'mixed_concat': 'deduce',
        'mixed_concat_little_endian': 'deduce', 'pure_concat': 'concat-short', 'query_unseen_concat': 'guess'}

def verify(s, p):
    s = (s or '').strip(); p = (p or '').strip()
    if re.fullmatch(r'[01]+', s): return p.lower() == s.lower()
    try: return math.isclose(float(s), float(p), rel_tol=1e-2, abs_tol=1e-5)
    except Exception: return p.lower() == s.lower()

def fused(text):
    out = set()
    for line in text.strip().splitlines():
        for run in line.replace(' = ', ' ').split():
            for a, b in TOK.encode(run, add_special_tokens=False).offsets:
                seg = run[a:b]
                if len(seg) >= 2 and all(c in POOL for c in seg): out.add(seg)
    return out & ALL

def feats(prompt, subtype, reading_override=None):
    examples, query = [], None
    for line in prompt.strip().splitlines():
        line = line.strip()
        if line.startswith('Now'): query = line.split(': ', 1)[1].strip()
        elif ' = ' in line and not line.startswith('In'):
            lhs, rhs = line.split(' = ', 1); lhs = lhs.strip(); rhs = rhs.strip()
            if len(lhs) == 5: examples.append((lhs, rhs))
    ops = []
    for lhs, rhs in examples:
        if lhs[2] not in ops: ops.append(lhs[2])
    if query and query[2] not in ops: ops.append(query[2])
    opset = set(ops); digs = set()
    for lhs, rhs in examples:
        for ch in (lhs[0], lhs[1], lhs[3], lhs[4]):
            if ch not in opset: digs.add(ch)
        for ch in rhs:
            if ch not in opset: digs.add(ch)
    if query:
        for ch in (query[0], query[1], query[3], query[4]):
            if ch not in opset: digs.add(ch)
    signed = any(rhs and (rhs[0] in opset or rhs[-1] in opset) for _l, rhs in examples)
    reading = reading_override or ('leftward' if subtype in ('little_endian', 'mixed_concat_little_endian') else 'standard')
    return {'signed': signed, 'n_ops': len(ops), 'n_ex': len(examples),
            'arith_syms': sum(1 for s in ops if s in '+-*'), 'n_digits': len(digs), 'reading': reading}

def main():
    rng = random.Random(20260609)
    # ---- original trainable (fixed) ----
    orig = defaultdict(list); orig_cov = set(); sym_count = Counter()
    with open(BASE, newline='') as fh:
        for r in csv.DictReader(fh):
            if r['category'] in ('cryptarithm_deduce', 'cryptarithm_guess') and r['GT-match'] == 'True' and int(r['token length']) < 7680:
                tks = fused(r['prompt']); f = feats(r['prompt'], r['new label'])
                orig[r['new label']].append({'id': r['id'], 'src': 'orig', 'tokens': tks, 'text': r['prompt'], **f})
                orig_cov |= tks
                for ch in r['prompt']:
                    if ch in POOL: sym_count[ch] += 1

    # ---- aug trainable pool (all batches) ----
    blind = {}
    for f in glob.glob(os.path.join(HERE, '_blind_out*', 'shard_*.jsonl')):
        for line in open(f):
            line = line.strip()
            if line: o = json.loads(line); blind[o['id']] = (o['box'], o['ntok'])
    aug = defaultdict(list)
    for mf, pdir in [('puzzles/_manifest.csv', 'puzzles'), ('puzzles2/_manifest.csv', 'puzzles2'), ('puzzles3/_manifest.csv', 'puzzles3'), ('puzzles4/_manifest.csv', 'puzzles4')]:
        mp = os.path.join(HERE, mf)
        if not os.path.exists(mp): continue
        for row in csv.DictReader(open(mp)):
            box, ntok = blind.get(row['id'], ('M', -1))
            if box in ('M', 'TIMEOUT') or str(box).startswith('ERR'): continue
            if not (verify(row['gold'], box) and 0 <= ntok < 7680): continue
            text = open(os.path.join(HERE, pdir, row['id'], 'question.txt')).read()
            aug[row['subtype']].append({'id': row['id'], 'src': 'aug', 'tokens': fused(text), 'text': text,
                                        **feats(text, row['subtype'], row['reading'])})
    print("per-category aug-trainable pool sizes:", {s: len(aug[s]) for s in ORDER})

    need = {s: TARGET[s] - len(orig[s]) for s in ORDER}
    selected = {s: [] for s in ORDER}
    covered = set(orig_cov)
    pool = {s: sorted(aug[s], key=lambda p: p['id']) for s in ORDER}   # deterministic order
    chosen_ids = set()

    # ---- Phase 0: FORCE-include the targeted coverage puzzles (puzzles4, idoff 300000) — they uniquely
    #      cover the rarest 3-sym tokens and must not be crowded out by the category cap. ----
    for s in ORDER:
        for p in pool[s]:
            if int(p['id'].rsplit('_', 1)[1]) >= 300000 and len(selected[s]) < need[s]:
                selected[s].append(p); chosen_ids.add(p['id']); covered |= p['tokens']

    # ---- Phase 1: token-centric coverage — cover the RAREST achievable 3-sym tokens first (a token covered
    #      by few puzzles must claim a slot before the abundant ones, else the category cap crowds it out). ----
    tok2p = defaultdict(list)
    for s in ORDER:
        for p in pool[s]:
            for t in (p['tokens'] & ACH3):
                tok2p[t].append((s, p))
    for t in sorted(ACH3, key=lambda t: len(tok2p[t])):              # rarest tokens first
        if t in covered or not tok2p[t]: continue
        cands = [(s, p) for (s, p) in tok2p[t] if p['id'] not in chosen_ids and len(selected[s]) < need[s]]
        if not cands: continue
        s, p = max(cands, key=lambda sp: len((sp[1]['tokens'] & ACH3) - covered))  # tie-break: most extra new tokens
        selected[s].append(p); chosen_ids.add(p['id']); covered |= p['tokens']
    cov_picks = sum(len(v) for v in selected.values())

    # ---- Phase 2: balance fill, stratified by (reading, signed) within each category ----
    for s in ORDER:
        rem = need[s] - len(selected[s])
        if rem <= 0: continue
        cand = [p for p in pool[s] if p['id'] not in chosen_ids]
        strata = defaultdict(list)
        for p in cand: strata[(p['reading'], p['signed'])].append(p)
        for v in strata.values(): rng.shuffle(v)
        # round-robin across strata to keep them balanced
        order = sorted(strata, key=lambda k: -len(strata[k]))
        while rem > 0 and any(strata[k] for k in order):
            for k in order:
                if rem <= 0: break
                if strata[k]:
                    p = strata[k].pop(); selected[s].append(p); chosen_ids.add(p['id']); covered |= p['tokens']; rem -= 1

    sel_all = [p for s in ORDER for p in selected[s]]
    open(os.path.join(HERE, '_final_selection.txt'), 'w').write('\n'.join(sorted(p['id'] for p in sel_all)) + '\n')
    combined = [p for s in ORDER for p in orig[s]] + sel_all

    # ---- report: distribution ----
    print("\n" + "=" * 72); print("FINAL BALANCED 1000 — distribution"); print("=" * 72)
    print(f"{'subtype':<30}{'orig':>6}{'aug':>6}{'final':>7}{'target':>8}")
    for s in ORDER:
        print(f"{s:<30}{len(orig[s]):>6}{len(selected[s]):>6}{len(orig[s])+len(selected[s]):>7}{TARGET[s]:>8}")
    print('-' * 60); print(f"{'TOTAL':<30}{sum(len(orig[s]) for s in ORDER):>6}{len(sel_all):>6}{len(combined):>7}{sum(TARGET.values()):>8}")
    print(f"(coverage phase chose {cov_picks} of {len(sel_all)} aug; balance phase filled the rest)")

    # ---- report: token coverage on FINAL 1000 ----
    fcov = set()
    for p in combined: fcov |= p['tokens']
    print("\n" + "=" * 72); print("MERGED-TOKEN COVERAGE — final 1000"); print("=" * 72)
    print(f"  2-symbol achievable: {len(fcov & ACH2)}/{len(ACH2)} = {100*len(fcov&ACH2)/len(ACH2):.1f}%")
    print(f"  3-symbol achievable: {len(fcov & ACH3)}/{len(ACH3)} = {100*len(fcov&ACH3)/len(ACH3):.1f}%")
    print(f"  all achievable:      {len(fcov & ACH)}/{len(ACH)} = {100*len(fcov&ACH)/len(ACH):.1f}%")
    miss3 = sorted(ACH3 - fcov)
    print(f"  still-missing 3-sym ({len(miss3)}): {miss3[:30]}")

    # ---- report: per-axis balance ----
    import statistics
    print("\n" + "=" * 80); print("BALANCE — per reasoning-path (FINAL 1000)"); print("=" * 80)
    print(f"{'path':<14}{'n':>5}{'signed%':>8}{'left%':>7}{'arithSym%':>10}{'avgOps':>7}{'avgEx':>6}{'avgDig':>7}")
    for pth in ['deduce', 'concat-short', 'guess']:
        sub = [p for p, s in ((p, s) for s in ORDER for p in (orig[s] + selected[s])) if PATH[s] == pth]
    # rebuild properly
    bypath = defaultdict(list)
    for s in ORDER:
        for p in orig[s] + selected[s]: bypath[PATH[s]].append(p)
    for pth in ['deduce', 'concat-short', 'guess']:
        sr = bypath[pth]; n = len(sr)
        print(f"{pth:<14}{n:>5}{100*sum(p['signed'] for p in sr)/n:>7.0f}%{100*sum(p['reading']=='leftward' for p in sr)/n:>6.0f}%"
              f"{100*sum(p['arith_syms']>0 for p in sr)/n:>9.0f}%{statistics.mean(p['n_ops'] for p in sr):>7.2f}"
              f"{statistics.mean(p['n_ex'] for p in sr):>6.2f}{statistics.mean(p['n_digits'] for p in sr):>7.2f}")

    # symbol-frequency balance on final 1000
    fsym = Counter()
    for p in combined:
        for ch in p['text']:
            if ch in POOL: fsym[ch] += 1
    operand_freq = sorted((fsym[s] for s in fsym if s not in OPS))
    print(f"\nsymbol usage (final 1000): {len(fsym)}/26 symbols | operand-freq min/med/max "
          f"{operand_freq[0]}/{operand_freq[len(operand_freq)//2]}/{operand_freq[-1]}")
    print(f"\nselected aug ids -> _final_selection.txt ({len(sel_all)})")

if __name__ == '__main__':
    main()
