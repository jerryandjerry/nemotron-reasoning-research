# EDA: Cryptarithm Puzzles

> Working dataset: 800 puzzles from `golden_cryptarithm_cot.csv` (ikev solver CoTs).

## Key finding

1. When the operator symbol appears in the output, it is always at `rhs[0]` or `rhs[-1]` — never in the middle. Verified across all 800 puzzles.
2. If operator appears in RHS, the operation is always a subtraction variant (absdiff, sub_signed, rsub_signed, neg_absdiff) — or unknown. No known non-subtraction operation ever produces the operator symbol in the output. Verified across all 800 puzzles.
   - Known (subtraction variant): 419 cases
   - Unknown: 48 cases
3. LHS always has exactly 5 symbols. Verified across all 800 puzzles. No exceptions.
4. query always have 5 symbols. Its position-3 symbol either never appears in any example, or appears only at position 3 of example LHS. Verified across all 800 puzzles. 0 violations.
5. When query position-3 never appears in example LHS (unseen operator), 161 cases total. Operation assigned by ikev cot:
   - addition: 35, multiplication: 30, absolute difference: 26, subtraction: 19, concatenation: 16, addition minus 1: 6, addition plus 1: 6, rsub_signed: 6, multiply then subtract 1: 5, gcd: 4, multiply then add 1: 4, lcm: 2, modulo: 2
6. If the operator symbol appears at rhs[-1] (last position), the puzzle is little-endian. Verified across all 800 puzzles using CSV only (solver_cot confirms "little-endian" in all 19 such puzzles). 0 violations.
7. 4-digit RHS (ignoring leading sign prefix) almost always means the operator is noisy_mul (mul, mul_p1, mul_m1). Verified across all 800 puzzles.
   - Total 4-digit RHS examples: 719
   - noisy_mul: 717 (99.72%)
   - Exceptions: 2
     - `little_endian_017a871e`: `a²+b` — squaring produces 4-digit output
     - `mixed_concat_86ccbdf7`: `addition` — zero-padded display of a small sum
   - Signal strength: strong prior to start search with noisy_mul, but not a hard rule.
8. Within a single puzzle, operators are almost always from different families (noisy_mul, noisy_add, general_subtraction, concat, other). Verified across all 800 puzzles.
   - Multi-operator puzzles: 788
   - Violations (two operators same family): 13 (1.65%)
     - 7 puzzles: two general_subtraction operators
     - 4 puzzles: two concat operators (concat_fwd + concat_rev)
     - 1 puzzle: two noisy_add operators
     - 1 puzzle: gcd + lcm
   - Signal strength: 98.35% prior — useful to guide search order (if one operator is identified, try a different family for the next), but not a hard constraint.
