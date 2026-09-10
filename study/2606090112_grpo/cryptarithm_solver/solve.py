#!/usr/bin/env python3
"""Convenience wrapper: solve a cryptarithm puzzle and print its chain-of-thought.

Usage:
    python3 solve.py path/to/question.txt        # a single question file
    python3 solve.py path/to/puzzle_dir          # a dir containing question.txt
    echo "...puzzle text..." | python3 solve.py  # puzzle on stdin

The solver (gen_cot) reads ONLY question.txt and emits the forward-only DFS-search CoT
ending in \\boxed{<answer>}. It writes nothing back into your source puzzle dir (a temp
copy is used), so it is safe to point at read-only inputs.
"""
import sys, os, tempfile, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _gen_crypt as gc


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    tmp = tempfile.mkdtemp(prefix='cryptsolve_')
    try:
        if arg and os.path.isdir(arg):
            shutil.copy(os.path.join(arg, 'question.txt'), os.path.join(tmp, 'question.txt'))
        elif arg and os.path.isfile(arg):
            shutil.copy(arg, os.path.join(tmp, 'question.txt'))
        else:                                                  # stdin
            open(os.path.join(tmp, 'question.txt'), 'w').write(sys.stdin.read())
        print(gc.gen_cot(tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
