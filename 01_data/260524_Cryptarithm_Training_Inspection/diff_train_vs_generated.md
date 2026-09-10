# Diff: our training CoT vs our model's generated CoT (cryptarithm, newdata)

On the 68 val cryptarithm puzzles (all in our training set). Generated = newdata model, greedy.

## Aggregate
- token length: TRAINING median **7456** vs GENERATED median **7680**
- generated longer than its training target: **35/68**; generated hit 7680 ceiling: **51/68**
- identical from line 1 (longest common prefix): median **5** lines (of ~225 training lines)
- section where it FIRST diverges: {'legend': 66, 'equations': 2}
- generation has a line repeated >=4x consecutively (loop): **6/68**

## Concrete diff — id `b1b10e83`  (`-` training CoT, `+` model generated; first ~36 lines)
```diff
@@ -3,11 +3,10 @@
 First, let me assign letters to each symbol:
-  ! -> A
-  " -> B
-  # -> C
-  $ -> D
-  ) -> E
-  \ -> F
+  " -> A
+  # -> B
+  $ -> C
+  ) -> D
+  \ -> E
+  | -> F
   ` -> G
   { -> H
-  | -> I
 Operators:
@@ -15,10 +14,12 @@
   * -> g
+  ) -> h (but note: in the examples, ')' is used as an operator, so I'll use h for it)
+  $ -> i (but note: in the examples, '$' is used as an operator, so I'll use i for it)
 
 I will convert all the equations to letter form:
-  EX1 「"\+#|」 = 「"\#|」 becomes BF f CI = BFCI
-  EX2 「#!+#\」 = 「#!#\」 becomes CA f CF = CACF
-  EX3 「#"*#"」 = 「#){|」 becomes CB g CB = CEHI
-  EX4 「$\*)"」 = 「${$#」 becomes DF g EB = DHDC
-  EX5 「)\+`#」 = 「)\`#」 becomes EF f GC = EFGC
-  QUERY 「|"+#$」 becomes IB f CD
+  EX1 「"\+#|」 = 「"\#|」 becomes AE f BF = AEBF
+  EX2 「#!+#\」 = 「#!#\」 becomes BD f BE = BDBE
+  EX3 「#"*#"」 = 「#){|」 becomes BA g BA = BHDF
+  EX4 「$\*)"」 = 「${$#」 becomes CE g AD = CHCB
+  EX5 「)\+`#」 = 「)\`#」 becomes DE h EG = DEGD
+  QUERY 「|"+#$」 becomes FA AC
 
@@ -34,3 +35,2 @@
 Let me also check if operators are concat operation:
-  f: concat_fwd=BFCI, concat_rev=CIBF. Output=BFCI. FWD matches -> f is concat_fwd.
-  g: concat_fwd=CBCB, concat_rev=CBCB. Output=CEHI. Neither matches, so g is not concat.
+  f: concat_fwd=AEBF, concat_rev=BFAE. Output=AEBF. FWD matches -> f is concat_fwd.
```

generated continues to 7680 tokens; tail:
```
"concat.\n  i: not present in any example, so I have to guess what it means. Since it is $, it is most likely noisy_add or noisy_mul; I'll check the pattern later.\n  g: concat_fwd=BABA, concat_rev=ABAB."
```