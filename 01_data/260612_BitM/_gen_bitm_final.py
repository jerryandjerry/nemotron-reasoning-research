#!/usr/bin/env python3
"""THE solver. One deterministic function: question.txt -> CoT (forward-only log).
= ported huikang machinery + blind word-prior steering + tier ladder
  (families -> Maj -> Ch -> NOT-wrap -> 3-term -> default 1 only after absent scans)."""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _gen_bitm_v3 as V
import _gen_bitm as W
from _store_types import Problem, Example

def gen_cot(folder):
    qtext = open(os.path.join(folder, 'question.txt')).read()
    exs = re.findall(r'([01]{8}) -> ([01]{8})', qtext)
    q = re.search(r'determine the output for: ([01]{8})', qtext).group(1)
    pref = W.full_prior(exs)
    V._TIER3 = True
    V._SECTION_PREF.clear(); V._PREFERRED = pref
    try:
        return V.reasoning_bit_manipulation(Problem(
            id=os.path.basename(folder), category='bit_manipulation',
            examples=[Example(a, b) for a, b in exs], question=q, answer=''))
    finally:
        V._PREFERRED = None; V._SECTION_PREF.clear()

if __name__ == '__main__':
    print(gen_cot(sys.argv[1]))
