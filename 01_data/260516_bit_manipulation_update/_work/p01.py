import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from helpers import *

# Puzzle 1a862653
ex = [
("00000111","00001110"),
("11100001","11000000"),
("11011001","01100100"),
("10011011","01101101"),
("01001111","00111110"),
("11011011","01101101"),
("01000101","00000010"),
("10100011","01000101"),
]
print("Trying shl by 1:")
show(test_rule(lambda s: shl(s,1), ex))
print()
print("Trying rotl by 1:")
show(test_rule(lambda s: rotl(s,1), ex))
print()
print("Examples as integers:")
for inp, out in ex:
    print(f"  {b2i(inp):3d} -> {b2i(out):3d}  diff={b2i(out)-b2i(inp)}, ratio={b2i(out)/(b2i(inp) or 1):.3f}")
