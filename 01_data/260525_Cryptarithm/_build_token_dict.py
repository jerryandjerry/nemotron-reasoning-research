#!/usr/bin/env python3
"""Build the merged-token -> symbols decomposition dictionary for the cryptarithm cipher alphabet.

The Nemotron (Tekken/Mistral-NeMo) BPE tokenizer MERGES runs of adjacent punctuation, so one token
can stand for 2+ cipher symbols (e.g. token '}-' = symbols '}' , '-'). This scans every puzzle's
equations + query, tokenizes them with the REAL model tokenizer, and records each token that spans
2+ cipher symbols together with its exact decomposition and occurrence count.

The decomposition is DETERMINISTIC and FINITE: this dict is the full lookup the model must learn to
split (via the symbol-by-symbol transcription step in the CoT). Output -> merged_token_dict.json.
Run: python3 _build_token_dict.py"""
import glob, os, json, collections, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cot_generator as cg
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
TOK  = os.path.join(HERE, '..', '..', '02_train', '260512_huikang_085', 'repo', 'tokenizer.json')
OUT  = os.path.join(HERE, 'merged_token_dict.json')
tok  = Tokenizer.from_file(TOK)

# cipher alphabet = every operand/operator symbol that appears across all puzzles
dirs = [d for d in sorted(glob.glob(os.path.join(HERE, '*_*'))) if os.path.exists(os.path.join(d, 'question.txt'))]
pool = set()
for d in dirs:
    p = cg.parse(os.path.join(d, 'question.txt'))
    pool |= set(p.digit_syms) | set(p.op_syms)

merged = collections.Counter()
def scan(line):
    e = tok.encode(line, add_special_tokens=False)
    for (a, b) in e.offsets:
        sub = line[a:b].strip()
        if len(sub) >= 2 and all(c in pool for c in sub):   # one token spanning 2+ cipher symbols
            merged[sub] += 1

for d in dirs:
    for line in open(os.path.join(d, 'question.txt')):
        line = line.strip()
        if line.startswith("Now, determine"):
            scan(line.split(": ", 1)[1].strip())
        elif " = " in line and not line.startswith("In"):
            lhs, rhs = line.split(" = ", 1)
            scan(lhs.strip()); scan(rhs.strip())

by_size = collections.Counter(len(k) for k in merged)
doc = {
    "meta": {
        "description": "Merged-token -> cipher-symbol decomposition. One token can encode 2+ symbols; "
                       "this is the deterministic lookup the model must learn to split.",
        "tokenizer": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 (md5 49f3ce2fc619568ee65707629bf23b53)",
        "scanned_puzzles": len(dirs),
        "alphabet_size": len(pool),
        "n_merged_tokens": len(merged),
        "total_occurrences": sum(merged.values()),
        "by_merge_size": {str(k): v for k, v in sorted(by_size.items())},
        "note": "single-symbol tokens are identity (token -> itself) and are omitted.",
    },
    # ordered by descending frequency; value = {symbols, count}
    "map": {t: {"symbols": list(t), "count": n} for t, n in merged.most_common()},
}

with open(OUT, 'w') as f:
    json.dump(doc, f, ensure_ascii=False, indent=2)
print(f"wrote {OUT}\n  {len(merged)} merged tokens | {sum(merged.values())} occurrences | "
      f"alphabet {len(pool)} | sizes {dict(sorted(by_size.items()))}")
