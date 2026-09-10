#!/usr/bin/env python3
"""Self-test: run the solver on each bundled question.txt and compare its output, byte-for-byte,
to the golden cot.txt generated on the reference machine.

    python3 run_tests.py

PASS for a case means the solver reproduced the reference CoT exactly. A FAIL means the environment
differs from the reference (a different Python version, or a missing/edited file). Exit code is 0 iff
all pass.
"""
import sys, os, glob

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _gen_eq as g


def first_diff(expected, got):
    el, gl = expected.splitlines(), got.splitlines()
    for i in range(max(len(el), len(gl))):
        e = el[i] if i < len(el) else '<missing>'
        gg = gl[i] if i < len(gl) else '<missing>'
        if e != gg:
            return i + 1, e, gg
    return None


def main():
    cases = sorted(glob.glob(os.path.join(HERE, 'test_cases', '*', '')))
    if not cases:
        print("No test_cases/ found next to this script."); return 2
    npass = 0
    for d in cases:
        name = os.path.basename(d.rstrip('/'))
        expected = open(os.path.join(d, 'cot.txt')).read()
        try:
            got, _ = g.gen_cot(g.load_folder(d))                # solver reads ONLY question.txt
        except Exception as e:
            print(f"FAIL  {name:<40} solver raised {type(e).__name__}: {e}")
            continue
        if got == expected:
            print(f"PASS  {name:<40} ({len(expected.splitlines())} lines reproduced exactly)")
            npass += 1
        else:
            dd = first_diff(expected, got)
            where = f"line {dd[0]}" if dd else "length only"
            print(f"FAIL  {name:<40} output differs at {where}")
            if dd:
                print(f"        expected: {dd[1][:100]!r}")
                print(f"        got     : {dd[2][:100]!r}")
    total = len(cases)
    print(f"\n{npass}/{total} cases reproduced the reference CoT byte-for-byte.")
    if npass == total:
        print("OK — the solver behaves identically to the reference machine.")
        return 0
    print("MISMATCH — environment differs (check the Python version and that no file was edited).")
    return 1


if __name__ == '__main__':
    sys.exit(main())
