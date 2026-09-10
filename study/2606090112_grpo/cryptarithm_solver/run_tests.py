#!/usr/bin/env python3
"""Self-test: run the solver on each bundled question.txt and compare its output, byte-for-byte,
to the golden cot.txt generated on the reference machine.

    python3 run_tests.py

PASS for a case means the solver reproduced the reference CoT exactly. A FAIL (especially in the
"Split the BPE merged tokens" section) usually means a different `tokenizers` library version or a
missing/edited file — i.e. the environment differs from the reference. Exit code is 0 iff all pass.
"""
import sys, os, glob, tempfile, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _gen_crypt as gc


def solve(qtext):
    """Run the solver with ONLY question.txt in scope (fresh temp dir), return the CoT string."""
    d = tempfile.mkdtemp(prefix='selftest_')
    try:
        open(os.path.join(d, 'question.txt'), 'w').write(qtext)
        return gc.gen_cot(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def first_diff(expected, got):
    el, gl = expected.splitlines(), got.splitlines()
    for i in range(max(len(el), len(gl))):
        e = el[i] if i < len(el) else '<missing>'
        g = gl[i] if i < len(gl) else '<missing>'
        if e != g:
            return i + 1, e, g
    return None


def main():
    cases = sorted(glob.glob(os.path.join(HERE, 'test_cases', '*', '')))
    if not cases:
        print("No test_cases/ found next to this script."); return 2
    npass = 0
    for d in cases:
        name = os.path.basename(d.rstrip('/'))
        qtext = open(os.path.join(d, 'question.txt')).read()
        expected = open(os.path.join(d, 'cot.txt')).read()
        try:
            got = solve(qtext)
        except Exception as e:
            print(f"FAIL  {name:<36} solver raised {type(e).__name__}: {e}")
            continue
        if got == expected:
            print(f"PASS  {name:<36} ({len(expected.splitlines())} lines reproduced exactly)")
            npass += 1
        else:
            dd = first_diff(expected, got)
            where = f"line {dd[0]}" if dd else "length only"
            print(f"FAIL  {name:<36} output differs at {where}")
            if dd:
                print(f"        expected: {dd[1][:100]!r}")
                print(f"        got     : {dd[2][:100]!r}")
    total = len(cases)
    print(f"\n{npass}/{total} cases reproduced the reference CoT byte-for-byte.")
    if npass == total:
        print("OK — the solver behaves identically to the reference machine.")
        return 0
    print("MISMATCH — environment differs (check `tokenizers` version and that no file was edited).")
    return 1


if __name__ == '__main__':
    sys.exit(main())
