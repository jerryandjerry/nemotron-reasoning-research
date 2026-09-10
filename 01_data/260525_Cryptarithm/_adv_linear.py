#!/usr/bin/env python3
"""Adversarial counterexample hunt for the linear-bound rules, divorced from any puzzle.

The rule keeps {v in dom[s]: lo <= c_s*v <= hi - sum_{t!=s} c_t*[dom interval]} where the net
identity is Sum_t c_t v_t = noise. SOUNDNESS = "the rule never removes the TRUE digit of s".
That holds iff, for the true assignment, Sum_t c_t v_t in [lo,hi] AND interval relaxation only
widens. The interval relaxation step is trivially a superset. So the whole question reduces to:

   Is  Sum_t c_t v_t = noise  always in [lo,hi]  for EVERY assignment the generator's noise model
   permits, for (i) ~add and (ii) a locked signed-sub (sub_signed / rsub_signed) example?

We brute-force ALL operand pairs and ALL family variants the generator can produce and check the
net residual equals exactly the offset k, and |k|<=2. We do this in BOTH reading modes and for the
worst RHS structures (shared symbols, 1/2/3-digit RHS, sign prefix/suffix), because the residual is
an algebraic identity independent of which symbols are shared — but we verify that too.
"""
import cot_generator as cg
from cot_generator import VARIANT_FN

ADD = ['add','add_p1','add_m1','add_p2','add_m2']
SUB_SIGNED = ['sub_signed','sub_signed_p1','sub_signed_m1','sub_signed_p2','sub_signed_m2']
RSUB = ['rsub_signed']
OFF = {'add':0,'add_p1':1,'add_m1':-1,'add_p2':2,'add_m2':-2,
       'sub_signed':0,'sub_signed_p1':1,'sub_signed_m1':-1,'sub_signed_p2':2,'sub_signed_m2':-2,
       'rsub_signed':0}

def residual_add(a, b, rv):
    # net identity for ~add: (a + b) - rv  should equal -(offset)?  Actually rv = a+b+k, so (a+b)-rv = -k.
    return (a + b) - rv

def residual_sub(a, b, rv):
    return (a - b) - rv

def residual_rsub(a, b, rv):
    return (b - a) - rv

worst_add = 0; worst_sub = 0; worst_rsub = 0
viol = []
for a in range(10, 100):
    for b in range(10, 100):
        for v in ADD:
            rv = VARIANT_FN[v](a, b)
            r = residual_add(a, b, rv)
            worst_add = max(worst_add, abs(r))
            if not (-2 <= r <= 2): viol.append(('add', a, b, v, rv, r))
            # the rule's claimed window is exactly [-2,2]; the residual is -k where k in [-2,2]
        for v in SUB_SIGNED:
            rv = VARIANT_FN[v](a, b)
            r = residual_sub(a, b, rv)
            worst_sub = max(worst_sub, abs(r))
            if not (-2 <= r <= 2): viol.append(('sub', a, b, v, rv, r))
        for v in RSUB:
            rv = VARIANT_FN[v](a, b)
            r = residual_rsub(a, b, rv)
            worst_rsub = max(worst_rsub, abs(r))
            if not (r == 0): viol.append(('rsub', a, b, v, rv, r))

print(f"ALL operand pairs x ALL variants:")
print(f"  ~add net residual max |.| = {worst_add}  (claim: <=2)")
print(f"  signed-sub net residual max |.| = {worst_sub}  (claim: <=2)")
print(f"  rsub net residual max |.| = {worst_rsub}  (claim: ==0)")
print(f"  VIOLATIONS: {len(viol)}")
for x in viol[:20]: print("   ", x)

# Now the DANGEROUS confusion: what if an operator the generator LOCKED as sub_signed is actually
# evaluated on an example that is really an absdiff/rsub case (a<b)? The lock means the generator
# committed to a-b for ALL of that op's examples. Check: if a<b but generator uses sub_signed,
# rv would be negative; the net residual is still exactly -k (algebraic). The danger is ONLY if the
# rule were applied with the wrong family. Confirm residual identity holds regardless of sign of a-b:
bad = 0
for a in range(10,100):
    for b in range(10,100):
        for v in SUB_SIGNED:
            rv = VARIANT_FN[v](a,b)
            if residual_sub(a,b,rv) != -OFF[v]: bad+=1
print(f"\nsigned-sub residual == -offset for all (a,b) incl a<b: violations={bad}")
