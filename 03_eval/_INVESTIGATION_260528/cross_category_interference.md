# Cross-Category Parameter Interference Investigation

**Runs compared:** moe, alleq, alleqgt, cryptn
**Same-training-data categories (moe/alleq/alleqgt):** cryptarithm, bit_manipulation, cipher, gravity, numeral, unit_conversion
**cryptn shares data only for non-cryptarithm 5; cryptn has DIFFERENT cryptarithm training data**

## T1. Flipped puzzles per same-data category

A flip = verdict differs across moe/alleq/alleqgt (these 3 share training data on this category).
For the 5 non-cryptarithm categories, cryptn also shares data, so we also report flips across all 4.

### T1 Summary table

```
        category  n_puzzles  flips_moe_alleq_alleqgt flips_4runs_incl_cryptn  acc_moe  acc_alleq  acc_alleqgt  acc_cryptn
     cryptarithm         71                        3  N/A (cryptn diff data) 0.028169   0.014085     0.014085    0.000000
bit_manipulation        160                       18                      22 0.768750   0.756250     0.768750    0.781250
          cipher        158                        2                       2 0.993671   0.993671     0.987342    0.987342
         gravity        160                        2                       2 0.993750   1.000000     0.993750    1.000000
         numeral        158                        0                       0 1.000000   1.000000     1.000000    1.000000
 unit_conversion        159                        1                       1 1.000000   0.993711     1.000000    1.000000
```

## T2. Per-flip details (moe vs alleq vs alleqgt)


### cryptarithm  —  3 flipped puzzles

- **24e1f1d5** (cryptarithm_deduce)  GT=`^#(!`
  - prompt: `In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples: %!+<} = %!<} <^*"^ = ["( !\*%[ = #}(\ ^<*<< = ^"< Now, determine the result for: ^#+(!`
  - moe pred=`^#(!` ✓=True  tlen=732
  - alleq pred=`^(!#` ✓=False  tlen=729
  - alleqgt pred=`^#(!` ✓=True  tlen=724
  - cryptn pred=`3` ✓=False  tlen=7680
  - first-diff char idx: moe↔alleq=148, moe↔alleqgt=162, alleq↔alleqgt=148
- **4d8df95b** (cryptarithm_deduce)  GT=`)`
  - prompt: `In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples: /:-%( = -"^ (?-^/ = -/" ^/*@! = ^)(/ Now, determine the result for: ??-??`
  - moe pred=`?` ✓=False  tlen=498
  - alleq pred=`)` ✓=True  tlen=622
  - alleqgt pred=`-` ✓=False  tlen=507
  - cryptn pred=`3` ✓=False  tlen=7680
  - first-diff char idx: moe↔alleq=187, moe↔alleqgt=221, alleq↔alleqgt=187
- **b1b10e83** (cryptarithm_deduce)  GT=`|"#$`
  - prompt: `In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples: "\+#| = "\#| #!+#\ = #!#\ #"*#" = #){| $\*)" = ${$# )\+`# = )\`# Now, determine the resul`
  - moe pred=`|"#$` ✓=True  tlen=780
  - alleq pred=``}` ✓=False  tlen=880
  - alleqgt pred=`}}` ✓=False  tlen=880
  - cryptn pred=`{`` ✓=False  tlen=2584
  - first-diff char idx: moe↔alleq=249, moe↔alleqgt=235, alleq↔alleqgt=235

### bit_manipulation  —  18 flipped puzzles

- **3456da40** (bit_manipulation)  GT=`00001111`
  - prompt: `In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or ch`
  - moe pred=`01001111` ✓=False  tlen=6509
  - alleq pred=`00001111` ✓=True  tlen=6400
  - alleqgt pred=`01001111` ✓=False  tlen=6509
  - cryptn pred=`00001111` ✓=True  tlen=6400
  - first-diff char idx: moe↔alleq=7428, moe↔alleqgt=-1, alleq↔alleqgt=7428
- **3ec3f1b4** (bit_manipulation)  GT=`10111011`
  - prompt: `In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or ch`
  - moe pred=`10111011` ✓=True  tlen=6927
  - alleq pred=`51` ✓=False  tlen=7680
  - alleqgt pred=`10111011` ✓=True  tlen=6921
  - cryptn pred=`10111011` ✓=True  tlen=6928
  - first-diff char idx: moe↔alleq=6937, moe↔alleqgt=8104, alleq↔alleqgt=6937
- **3d24aef4** (bit_manipulation)  GT=`10100101`
  - prompt: `In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or ch`
  - moe pred=`10100100` ✓=False  tlen=6523
  - alleq pred=`10100100` ✓=False  tlen=6523
  - alleqgt pred=`10100101` ✓=True  tlen=6523
  - cryptn pred=`10100001` ✓=False  tlen=6703
  - first-diff char idx: moe↔alleq=-1, moe↔alleqgt=8248, alleq↔alleqgt=8248
- **81323d52** (bit_manipulation)  GT=`10110001`
  - prompt: `In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or ch`
  - moe pred=`10110001` ✓=True  tlen=6727
  - alleq pred=`10111001` ✓=False  tlen=6723
  - alleqgt pred=`10110001` ✓=True  tlen=6727
  - cryptn pred=`10110001` ✓=True  tlen=6727
  - first-diff char idx: moe↔alleq=8544, moe↔alleqgt=-1, alleq↔alleqgt=8544
- **4c327b55** (bit_manipulation)  GT=`11011100`
  - prompt: `In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or ch`
  - moe pred=`11011110` ✓=False  tlen=6739
  - alleq pred=`11011110` ✓=False  tlen=6739
  - alleqgt pred=`11011100` ✓=True  tlen=6723
  - cryptn pred=`11011110` ✓=False  tlen=6739
  - first-diff char idx: moe↔alleq=7563, moe↔alleqgt=2538, alleq↔alleqgt=2538
- **8c743940** (bit_manipulation)  GT=`01100100`
  - prompt: `In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or ch`
  - moe pred=`01100100` ✓=True  tlen=5987
  - alleq pred=`01100100` ✓=True  tlen=5987
  - alleqgt pred=`10100100` ✓=False  tlen=6073
  - cryptn pred=`01100100` ✓=True  tlen=5987
  - first-diff char idx: moe↔alleq=-1, moe↔alleqgt=4521, alleq↔alleqgt=4521
- **6f91481e** (bit_manipulation)  GT=`11000000`
  - prompt: `In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or ch`
  - moe pred=`10100000` ✓=False  tlen=7027
  - alleq pred=`11000000` ✓=True  tlen=6979
  - alleqgt pred=`10100000` ✓=False  tlen=7021
  - cryptn pred=`11000000` ✓=True  tlen=6979
  - first-diff char idx: moe↔alleq=2838, moe↔alleqgt=3064, alleq↔alleqgt=2838
- **cc5011ac** (bit_manipulation)  GT=`11011101`
  - prompt: `In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or ch`
  - moe pred=`11011101` ✓=True  tlen=6686
  - alleq pred=`51` ✓=False  tlen=7680
  - alleqgt pred=`11011101` ✓=True  tlen=6559
  - cryptn pred=`11011101` ✓=True  tlen=6689
  - first-diff char idx: moe↔alleq=3205, moe↔alleqgt=5960, alleq↔alleqgt=3205
- **52862572** (bit_manipulation)  GT=`00000010`
  - prompt: `In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or ch`
  - moe pred=`00000010` ✓=True  tlen=6949
  - alleq pred=`15` ✓=False  tlen=7680
  - alleqgt pred=`00000010` ✓=True  tlen=7039
  - cryptn pred=`00000000` ✓=False  tlen=6997
  - first-diff char idx: moe↔alleq=5418, moe↔alleqgt=6984, alleq↔alleqgt=5418
- **af5e4060** (bit_manipulation)  GT=`11001111`
  - prompt: `In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or ch`
  - moe pred=`11001111` ✓=True  tlen=6805
  - alleq pred=`01111111` ✓=False  tlen=7047
  - alleqgt pred=`011111?1` ✓=False  tlen=7000
  - cryptn pred=`01111111` ✓=False  tlen=6831
  - first-diff char idx: moe↔alleq=3269, moe↔alleqgt=3269, alleq↔alleqgt=3479
- **b634898d** (bit_manipulation)  GT=`11000000`
  - prompt: `In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or ch`
  - moe pred=`11000000` ✓=True  tlen=7152
  - alleq pred=`11000000` ✓=True  tlen=7154
  - alleqgt pred=`11000100` ✓=False  tlen=7223
  - cryptn pred=`11000000` ✓=True  tlen=7152
  - first-diff char idx: moe↔alleq=8452, moe↔alleqgt=5197, alleq↔alleqgt=5197
- **2f7f58de** (bit_manipulation)  GT=`00100100`
  - prompt: `In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or ch`
  - moe pred=`00100111` ✓=False  tlen=6299
  - alleq pred=`00100100` ✓=True  tlen=6284
  - alleqgt pred=`00100100` ✓=True  tlen=6284
  - cryptn pred=`00100100` ✓=True  tlen=6284
  - first-diff char idx: moe↔alleq=626, moe↔alleqgt=626, alleq↔alleqgt=-1
- **7f73016f** (bit_manipulation)  GT=`01010110`
  - prompt: `In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or ch`
  - moe pred=`11010110` ✓=False  tlen=6933
  - alleq pred=`11010110` ✓=False  tlen=6934
  - alleqgt pred=`01010110` ✓=True  tlen=6940
  - cryptn pred=`11110110` ✓=False  tlen=6955
  - first-diff char idx: moe↔alleq=8447, moe↔alleqgt=5988, alleq↔alleqgt=5988
- **af0fd8f6** (bit_manipulation)  GT=`10100110`
  - prompt: `In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or ch`
  - moe pred=`10100110` ✓=True  tlen=6244
  - alleq pred=`11001101` ✓=False  tlen=6267
  - alleqgt pred=`11100110` ✓=False  tlen=6218
  - cryptn pred=`10100110` ✓=True  tlen=6244
  - first-diff char idx: moe↔alleq=1165, moe↔alleqgt=1199, alleq↔alleqgt=1165
- **20052c2f** (bit_manipulation)  GT=`01100110`
  - prompt: `In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or ch`
  - moe pred=`01110110` ✓=False  tlen=6888
  - alleq pred=`01100110` ✓=True  tlen=6716
  - alleqgt pred=`01100111` ✓=False  tlen=6697
  - cryptn pred=`01100111` ✓=False  tlen=6697
  - first-diff char idx: moe↔alleq=912, moe↔alleqgt=912, alleq↔alleqgt=7808
- **2817d770** (bit_manipulation)  GT=`01100110`
  - prompt: `In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or ch`
  - moe pred=`01100110` ✓=True  tlen=7123
  - alleq pred=`7` ✓=False  tlen=7680
  - alleqgt pred=`7` ✓=False  tlen=7680
  - cryptn pred=`01100110` ✓=True  tlen=7149
  - first-diff char idx: moe↔alleq=3483, moe↔alleqgt=3483, alleq↔alleqgt=-1
- **9238e8d6** (bit_manipulation)  GT=`11101010`
  - prompt: `In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or ch`
  - moe pred=`10101010` ✓=False  tlen=6193
  - alleq pred=`11101010` ✓=True  tlen=6135
  - alleqgt pred=`10101010` ✓=False  tlen=6181
  - cryptn pred=`11101010` ✓=True  tlen=6133
  - first-diff char idx: moe↔alleq=1213, moe↔alleqgt=2384, alleq↔alleqgt=1213
- **d68daa97** (bit_manipulation)  GT=`01011001`
  - prompt: `In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or ch`
  - moe pred=`10111001` ✓=False  tlen=6134
  - alleq pred=`10111001` ✓=False  tlen=6134
  - alleqgt pred=`01011001` ✓=True  tlen=6091
  - cryptn pred=`10111001` ✓=False  tlen=6134
  - first-diff char idx: moe↔alleq=-1, moe↔alleqgt=3580, alleq↔alleqgt=3580

### cipher  —  2 flipped puzzles

- **8d10c393** (cipher)  GT=`princess chases the hidden map`
  - prompt: `In Alice's Wonderland, secret encryption rules are used on text. Here are some examples: ycslrzc giwwivy giuryc -> student follows forest vmejul mpjhmzry curjysur -> wizard imagines treasure kmul ycsl`
  - moe pred=`princess chases the hidden map` ✓=True  tlen=2215
  - alleq pred=`princess cases the hidden map` ✓=False  tlen=2210
  - alleqgt pred=`princess cases the island map` ✓=False  tlen=2213
  - cryptn pred=`princess castles the hidden map` ✓=False  tlen=2216
  - first-diff char idx: moe↔alleq=3482, moe↔alleqgt=3880, alleq↔alleqgt=3482
- **13db9692** (cipher)  GT=`princess writes treasure`
  - prompt: `In Alice's Wonderland, secret encryption rules are used on text. Here are some examples: nps inqrjzs umois wqsrnsi -> the strange mouse creates eqrzmj wqsrnsi nqsrioqs -> dragon creates treasure inoes`
  - moe pred=`princess rites treasure` ✓=False  tlen=1760
  - alleq pred=`princess writes treasure` ✓=True  tlen=1761
  - alleqgt pred=`princess rites treasure` ✓=False  tlen=1760
  - cryptn pred=`princess pirates treasure` ✓=False  tlen=1761
  - first-diff char idx: moe↔alleq=3347, moe↔alleqgt=3424, alleq↔alleqgt=3347

### gravity  —  2 flipped puzzles

- **0040ff76** (gravity)  GT=`154.62`
  - prompt: `In Alice's Wonderland, the gravitational constant has been secretly changed. Here are some example observations: For t = 1.37s, distance = 14.92 m For t = 4.27s, distance = 144.96 m For t = 3.28s, dis`
  - moe pred=`158.607` ✓=False  tlen=4159
  - alleq pred=`154.631` ✓=True  tlen=4159
  - alleqgt pred=`154.631` ✓=True  tlen=4159
  - cryptn pred=`154.631` ✓=True  tlen=4159
  - first-diff char idx: moe↔alleq=4447, moe↔alleqgt=1200, alleq↔alleqgt=1200
- **b5fdebcc** (gravity)  GT=`80.01`
  - prompt: `In Alice's Wonderland, the gravitational constant has been secretly changed. Here are some example observations: For t = 4.05s, distance = 85.84 m For t = 3.12s, distance = 50.94 m For t = 4.25s, dist`
  - moe pred=`79.999` ✓=True  tlen=2568
  - alleq pred=`79.999` ✓=True  tlen=2568
  - alleqgt pred=`79.002` ✓=False  tlen=2568
  - cryptn pred=`80.002` ✓=True  tlen=2568
  - first-diff char idx: moe↔alleq=-1, moe↔alleqgt=2901, alleq↔alleqgt=2901

### numeral  —  0 flipped puzzles

_No flips._

### unit_conversion  —  1 flipped puzzles

- **3b6db97b** (unit_conversion)  GT=`24.33`
  - prompt: `In Alice's Wonderland, a secret unit conversion is applied to measurements. For example: 8.6 m becomes 7.92 16.44 m becomes 15.14 24.68 m becomes 22.73 6.83 m becomes 6.29 Now, convert the following m`
  - moe pred=`24.306` ✓=True  tlen=1767
  - alleq pred=`50.726` ✓=False  tlen=1867
  - alleqgt pred=`24.306` ✓=True  tlen=1767
  - cryptn pred=`24.306` ✓=True  tlen=1767
  - first-diff char idx: moe↔alleq=161, moe↔alleqgt=-1, alleq↔alleqgt=161

## T3. Five most interesting flips (deep dive)


### b634898d (bit_manipulation / bit_manipulation)
GT=`11000000`
moe ✓=True pred=`11000000`
alleq ✓=True pred=`11000000`
alleqgt ✓=False pred=`11000100`
cryptn ✓=True pred=`11000000`

first-diff idx: moe↔alleq=8452, moe↔alleqgt=5197

**moe near divergence:**
```
I7
2 ?03
3 ?14
4 ?25
5 ?36
6 AND70
7 AND01

Matching
0 I6
1 I7
2 ?03 - Identity absent, NOT absent, Constant C0, AND07, OR absent, XOR absent, AND-NOT absent, OR-NOT absent, XOR-NOT absent
3 ?14 - Identity absent, NOT absent, Constant C0, AND absent, OR absent, XOR absent, AND-NOT absent, OR-NOT absent, XOR-NOT absent
4 ?25 - Identity absent, NOT a
```
**alleq near divergence:**
```
I7
2 ?03
3 ?14
4 ?25
5 ?36
6 AND70
7 AND01

Matching
0 I6
1 I7
2 ?03 - Identity absent, NOT absent, Constant C0, AND07, OR absent, XOR absent, AND-NOT03, OR-NOT absent, XOR-NOT absent
3 ?14 - Identity absent, NOT absent, Constant C0, AND absent, OR absent, XOR absent, AND-NOT absent, OR-NOT absent, XOR-NOT absent
4 ?25 - Identity absent, NOT absent
```

**moe near divergence (vs alleqgt):**
```
01100000 2
32 100000011 3
43 000110000 2
54 000001111 4
65 001000000 1
76 000011000 2

Matching output with AND-NOT
0 absent
1 70
2 01 71 03 15 51 05 06 75
3 01 71 03 15 51 05 06 75
4 01 71 03 15 51 05 06 75
5 01 71 03 15 51 05 06 75
6 01 71 03 15 51 05 06 75
7 02 04 07

Left
none
Best: none

Right
02 71 60x
04 73x
07 76x
Best: AND-NOT02 AND-NOT71:
```
**alleqgt near divergence:**
```
01100000 2
32 100000011 3
43 000110000 2
54 000001111 4
65 001000000 1
76 000011000 2

Matching output with AND-NOT
0 absent
1 70
2 01 71 03 15 51 05 60 55x
3 01 71 03 15 51 05 60 55x
4 01 71 03 15 51 05 60 55x
5 01 71 03 15 51 05 60 55x
6 01 71 03 15 51 05 60 55x
7 02 04 07

Left
none
Best: none

Right
02 71 60 57x
04 73x
07 76x
Best: AND-NOT02 AN
```

### 2817d770 (bit_manipulation / bit_manipulation)
GT=`01100110`
moe ✓=True pred=`01100110`
alleq ✓=False pred=`7`
alleqgt ✓=False pred=`7`
cryptn ✓=True pred=`01100110`

first-diff idx: moe↔alleq=3483, moe↔alleqgt=3483

**moe near divergence:**
```
put with OR
0 absent
1 23 32 34 43 35 53
2 04 40
3 15 51
4 26 62
5 13 31 37 73
6 04 40
7 15 51

Left
none
Best: none

Right
15 04 73 62 51 40 37 26 15x
51 40 37 26 15 04 73 62 51x
Best: OR15 OR04 OR73 OR62 OR51 OR40 OR37 OR26: 8

XOR
01 10 1110000000 3
12 21 1101000110 5
23 32 0101010001 4
34 43 0100010101 4
45 54 1001001101 5
56 65 0000101010 3
67
```
**alleq near divergence:**
```
put with OR
0 absent
1 23 32 34 43 35 53
2 04 40
3 15 51
4 26 62
5 13 31 37 73
6 04 40
7 15 51

Left
none
Best: none

Right
15 04 73 62 51 40 37 26 15 04 73 62 51 40 37 26 15 04 73 62 51 40 37 26 15 04 73 62 51 40 37 26 15 04 73 62 51 40 37 26 15 04 73 62 51 40 37 26 15 04 73 62 51 40 37 26 15 04 73 62 51 40 37 26 15 04 73 62 51 40 37 26 15 04 73 6
```

**moe near divergence (vs alleqgt):**
```
put with OR
0 absent
1 23 32 34 43 35 53
2 04 40
3 15 51
4 26 62
5 13 31 37 73
6 04 40
7 15 51

Left
none
Best: none

Right
15 04 73 62 51 40 37 26 15x
51 40 37 26 15 04 73 62 51x
Best: OR15 OR04 OR73 OR62 OR51 OR40 OR37 OR26: 8

XOR
01 10 1110000000 3
12 21 1101000110 5
23 32 0101010001 4
34 43 0100010101 4
45 54 1001001101 5
56 65 0000101010 3
67
```
**alleqgt near divergence:**
```
put with OR
0 absent
1 23 32 34 43 35 53
2 04 40
3 15 51
4 26 62
5 13 31 37 73
6 04 40
7 15 51

Left
none
Best: none

Right
15 04 73 62 51 40 37 26 15 04 73 62 51 40 37 26 15 04 73 62 51 40 37 26 15 04 73 62 51 40 37 26 15 04 73 62 51 40 37 26 15 04 73 62 51 40 37 26 15 04 73 62 51 40 37 26 15 04 73 62 51 40 37 26 15 04 73 62 51 40 37 26 15 04 73 6
```

### 6f91481e (bit_manipulation / bit_manipulation)
GT=`11000000`
moe ✓=False pred=`10100000`
alleq ✓=True pred=`11000000`
alleqgt ✓=False pred=`10100000`
cryptn ✓=True pred=`11000000`

first-diff idx: moe↔alleq=2838, moe↔alleqgt=3064

**moe near divergence:**
```
 16 61
5 23 32 56 65 67 76 46 64 06 60 05 50 16 61
6 23 32 56 65 67 76 46 64 06 60 05 50 16 61
7 23 32 56 65 67 76 46 64 06 60 05 50 16 61

Left
57 60 71 02x
75 06 17 20x
17 20x
71 02x
15 26 37 40x
51 62 73 04x
Best: AND57 AND60 AND71: 3

Right
23 12x
32 21x
56 45x
65 54x
67 56 45x
76 65 54x
46 35x
64 53x
06 75 64 53x
60 57 46 35x
05 74x
50 47x
16 
```
**alleq near divergence:**
```
 16 61
5 23 32 56 65 67 76 46 64 06 60 05 50 16 61
6 23 32 56 65 67 76 46 64 06 60 05 50 16 61
7 23 32 56 65 67 76 46 64 06 60 05 50 16 61

Left
57 60x
75 06x
17 20x
71 02x
15 26 37 40x
51 62 73 04x
Best: AND15 AND26 AND37: 3

Right
23 12x
32 21x
56 45x
65 54x
67 56x
76 65x
46 35x
64 53x
06 75x
60 57x
05 74x
50 47x
16 05x
61 50x
Best: AND23: 1

OR

```

**moe near divergence (vs alleqgt):**
```
60 AND71: 3

Right
23 12x
32 21x
56 45x
65 54x
67 56 45x
76 65 54x
46 35x
64 53x
06 75 64 53x
60 57 46 35x
05 74x
50 47x
16 05 74x
61 50 47x
Best: AND06 AND75 AND64: 3

OR
01 10 101111010 6
12 21 101101100 5
23 32 111101111 8
34 43 111100011 6
45 54 101100011 5
56 65 011100101 5
67 76 011111110 7
70 07 101111010 6

02 20 101011110 6
13 31 111100011
```
**alleqgt near divergence:**
```
60 AND71: 3

Right
23 12x
32 21x
56 45x
65 54x
67 56 45x
76 65 54x
46 35x
64 53x
06 75 64 53x
60 57 46 35x
05 74x
50 47x
16 05 74x
61 50 47x
Best: AND67 AND56: 2

OR
01 10 101111010 6
12 21 101101100 5
23 32 111101111 8
34 43 111100011 6
45 54 101100011 5
56 65 011100101 5
67 76 011111110 7
70 07 101111010 6

02 20 101011110 6
13 31 111100011 6
24 
```

### 52862572 (bit_manipulation / bit_manipulation)
GT=`00000010`
moe ✓=True pred=`00000010`
alleq ✓=False pred=`15`
alleqgt ✓=True pred=`00000010`
cryptn ✓=False pred=`00000000`

first-diff idx: moe↔alleq=5418, moe↔alleqgt=6984

**moe near divergence:**
```
111 4
51 11101010 5
62 11011101 6
73 11111111 a

05 11111111 a
16 10110111 6
27 11111010 6
30 11010111 6
41 01111010 5
52 10001111 5
63 11111100 6
74 11111111 a

06 11101111 7 match 5
17 10011111 6
20 11110111 7 match 6 7
31 11101011 6
42 00011111 5
53 10111110 6
64 11111101 7
75 11110111 7 match 6 7

07 11101010 5
10 10010111 5
21 11111010 6
32 11
```
**alleq near divergence:**
```
111 4
51 11101010 5
62 11011101 6
73 11111111 a

05 11111111 a
16 10110111 6
27 11111010 6
30 11010111 6
41 01111010 5
52 10001111 5
63 11111100 6
74 00011111 5

06 11101111 7 match 5
17 10011111 6
20 11110111 7 match 6 7
31 11101011 6
42 00011111 5
53 10111110 6
64 00111101 5
75 11110111 7 match 6 7

07 11101010 5
10 10010111 5
21 11111010 6
32 11
```

**moe near divergence (vs alleqgt):**
```
0101101 5
76 11010000 3

Matching output with XOR-NOT
0 37 73
1 04 40
2 70 15 51 07
3 26 62
4 37 73
5 absent
6 absent
7 absent

Left
37 40 51 62 73 04x
73 04 15 26 37 40x
Best: XOR-NOT37 XOR-NOT40 XOR-NOT51 XOR-NOT62 XOR-NOT73: 5

Right
none
Best: none

Selecting

Lefts
Identity none
NOT none
Constant none
AND none
OR OR35: 1
XOR none
AND-NOT none

```
**alleqgt near divergence:**
```
0101101 5
76 11010000 3

Matching output with XOR-NOT
0 37 73
1 04 40
2 70 15 51 07
3 26 62
4 37 73
5 absent
6 absent
7 absent

Left
37 40 51 62 73 04 15 26 37x
73 04 15 26 37 40 51 62 73x
Best: XOR-NOT37 XOR-NOT40 XOR-NOT51 XOR-NOT62 XOR-NOT73 XOR-NOT04 XOR-NOT15 XOR-NOT26: 8

Right
none
Best: none

Selecting

Lefts
Identity none
NOT none
Constant
```

### 7f73016f (bit_manipulation / bit_manipulation)
GT=`01010110`
moe ✓=False pred=`11010110`
alleq ✓=False pred=`11010110`
alleqgt ✓=True pred=`01010110`
cryptn ✓=False pred=`11110110`

first-diff idx: moe↔alleq=8447, moe↔alleqgt=5988

**moe near divergence:**
```
bsent, AND absent, OR absent, XOR absent, AND-NOT absent, OR-NOT absent, XOR-NOT absent
3 ?62 - Identity absent, NOT2, Constant absent, AND absent, OR absent, XOR absent, AND-NOT absent, OR-NOT62, XOR-NOT absent
4 ?73 - Identity absent, NOT3, Constant absent, AND absent, OR absent, XOR absent, AND-NOT73, OR-NOT absent, XOR-NOT absent
5 OR-NOT04
6 O
```
**alleq near divergence:**
```
bsent, AND absent, OR absent, XOR absent, AND-NOT absent, OR-NOT absent, XOR-NOT absent
3 ?62 - Identity absent, NOT2, Constant absent, AND absent, OR62, XOR absent, AND-NOT absent, OR-NOT62, XOR-NOT absent
4 ?73 - Identity absent, NOT3, Constant absent, AND absent, OR absent, XOR absent, AND-NOT73, OR-NOT absent, XOR-NOT absent
5 OR-NOT04
6 OR-NOT
```

**moe near divergence (vs alleqgt):**
```
a match 6
76 111111111 a match 6

Matching output with OR-NOT
0 67
1 07
2 absent
3 02 62 52
4 03 63 53
5 04
6 70 15 75 10 32 65 76
7 36 26

Left
67 70 01x
Best: OR-NOT67 OR-NOT70: 2

Right
36 25x
26 15 04 73x
Best: OR-NOT26 OR-NOT15 OR-NOT04: 3

XOR-NOT
01 101111101 7
12 101100011 5
23 011111111 8
34 100111000 4
45 010100010 3
56 110111111 8
67 111
```
**alleqgt near divergence:**
```
a match 6
76 111111111 a match 6

Matching output with OR-NOT
0 67
1 07
2 absent
3 02 62 52
4 03 63 53
5 04
6 70 15 75 10 32 65 76
7 36 26

Left
67 70x
Best: OR-NOT67: 1

Right
36 25x
26 15 04 73x
Best: OR-NOT26 OR-NOT15 OR-NOT04: 3

XOR-NOT
01 101111101 7
12 101100011 5
23 011111111 8
34 100111000 4
45 010100010 3
56 110111111 8
67 111001010 5 mat
```

## T4. Mean output_token_len per run (per same-data category)

```
        category   n  mean_tlen_moe  mean_tlen_alleq  mean_tlen_alleqgt  mean_tlen_cryptn  d_alleq_vs_moe  d_alleqgt_vs_moe  d_cryptn_vs_moe
     cryptarithm  71     827.323944       745.929577         744.056338       5790.225352      -81.394366        -83.267606      4962.901408
bit_manipulation 160    6685.706250      6710.956250        6692.150000       6690.687500       25.250000          6.443750         4.981250
          cipher 158    2090.892405      2090.841772        2090.639241       2090.569620       -0.050633         -0.253165        -0.322785
         gravity 160    3099.775000      3102.912500        3101.656250       3102.381250        3.137500          1.881250         2.606250
         numeral 158     955.556962       955.556962         955.556962        955.563291        0.000000          0.000000         0.006329
 unit_conversion 159    2312.823899      2315.188679        2312.773585       2315.094340        2.364780         -0.050314         2.270440
```

## T5. Numeric-eq format bleed into non-numeric-eq outputs

Search phrases (numeric-eq specific): `§1`, `§2`, `lock_concat`, `reading order =`, `concat_fwd`

Searching ALL puzzles in same-data NON-numeric-eq categories (no sampling — exhaustive).

### Count matrix (phrase occurrences per run × category)

```
phrase                    concat_fwd  lock_concat  reading order =    §1    §2
cat              run                                                          
bit_manipulation alleq           0.0          0.0              0.0   0.0   0.0
                 alleqgt         0.0          0.0              0.0   0.0   0.0
                 cryptn          0.0          0.0              0.0   0.0   0.0
                 moe             0.0          0.0              0.0   0.0   0.0
cipher           alleq           0.0          0.0              0.0   0.0   0.0
                 alleqgt         0.0          0.0              0.0   0.0   0.0
                 cryptn          0.0          0.0              0.0   0.0   0.0
                 moe             0.0          0.0              0.0   0.0   0.0
cryptarithm      alleq           0.0          0.0              0.0   0.0   0.0
                 alleqgt         0.0          0.0              0.0   0.0   0.0
                 cryptn         71.0          0.0             71.0  71.0  53.0
                 moe             0.0          0.0              0.0   0.0   0.0
gravity          alleq           0.0          0.0              0.0   0.0   0.0
                 alleqgt         0.0          0.0              0.0   0.0   0.0
                 cryptn          0.0          0.0              0.0   0.0   0.0
                 moe             0.0          0.0              0.0   0.0   0.0
numeral          alleq           0.0          0.0              0.0   0.0   0.0
                 alleqgt         0.0          0.0              0.0   0.0   0.0
                 cryptn          0.0          0.0              0.0   0.0   0.0
                 moe             0.0          0.0              0.0   0.0   0.0
unit_conversion  alleq           0.0          0.0              0.0   0.0   0.0
                 alleqgt         0.0          0.0              0.0   0.0   0.0
                 cryptn          0.0          0.0              0.0   0.0   0.0
                 moe             0.0          0.0              0.0   0.0   0.0
```

### Sample snippets (40 of 266 hits):

- run=**cryptn** cat=**cryptarithm** id=`0df82d52` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(GI, ID) = GIID     EX2: g(DG, FA) = DGFA ...`
- run=**cryptn** cat=**cryptarithm** id=`24e1f1d5` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(DA, FJ) = DAFJ     EX2: g(FI, BI) = -BE  ...`
- run=**cryptn** cat=**cryptarithm** id=`5690981d` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(BD, DE) = -BD     EX2: f(HB, DB) = II    ...`
- run=**cryptn** cat=**cryptarithm** id=`a1220274` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(CC, EF) = -BG     EX2: f(HE, DB) = -DD   ...`
- run=**cryptn** cat=**cryptarithm** id=`62fc7798` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(EC, GA) = IG     EX2: f(EE, IF) = GDI    ...`
- run=**cryptn** cat=**cryptarithm** id=`2f46a715` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(JJ, BC) = GJGB     EX2: g(AG, IB) = IBAG ...`
- run=**cryptn** cat=**cryptarithm** id=`52395e9a` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(HJ, BC) = BD     EX2: g(GD, FA) = AHJ    ...`
- run=**cryptn** cat=**cryptarithm** id=`afb53516` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(HD, FB) = -CG     EX2: g(IF, EG) = BA    ...`
- run=**cryptn** cat=**cryptarithm** id=`fcad9241` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(DF, DC) = DDC     EX2: f(DF, CI) = IE    ...`
- run=**cryptn** cat=**cryptarithm** id=`432b1110` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(EF, BE) = EBE     EX2: f(BF, AD) = BFBD  ...`
- run=**cryptn** cat=**cryptarithm** id=`39a1f5e9` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(AI, AC) = GH?     EX2: g(AI, IE) = -AI-  ...`
- run=**cryptn** cat=**cryptarithm** id=`4d8df95b` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(DC, EJ) = -BH     EX2: f(JE, DH) = -DB   ...`
- run=**cryptn** cat=**cryptarithm** id=`da10a947` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(DG, A) = DH     EX2: f(IB, CD) = AF     E...`
- run=**cryptn** cat=**cryptarithm** id=`258b796b` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: g(DN, DD) = A     EX2: f(BM, BC) = B     EX...`
- run=**cryptn** cat=**cryptarithm** id=`91e9dc52` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(CH, AE) = CEB     EX2: g(AB, HB) = D     ...`
- run=**cryptn** cat=**cryptarithm** id=`b0206bb7` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(DG, CB) = EE     EX2: g(AE, CF) = EGFH   ...`
- run=**cryptn** cat=**cryptarithm** id=`bc83b0a1` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(AA, FD) = AAFD     EX2: f(CH, AE) = CHAE ...`
- run=**cryptn** cat=**cryptarithm** id=`9081e954` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(EB, EH) = EBEH     EX2: f(ID, DB) = IDDB ...`
- run=**cryptn** cat=**cryptarithm** id=`6de4855a` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(EA, AE) = LL     EX2: g(BK, KD) = JML    ...`
- run=**cryptn** cat=**cryptarithm** id=`771472d6` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(DB, EH) = DBEH     EX2: g(HC, DD) = CGD  ...`
- run=**cryptn** cat=**cryptarithm** id=`db564ba9` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(GD, II) = -AE     EX2: f(HI, FB) = -CH   ...`
- run=**cryptn** cat=**cryptarithm** id=`a11311a4` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(AH, HG) = CFH     EX2: g(EC, AE) = -DB   ...`
- run=**cryptn** cat=**cryptarithm** id=`76c48f67` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(BI, J) = ADJ     EX2: g(DI, AG) = CEF    ...`
- run=**cryptn** cat=**cryptarithm** id=`dafd11de` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(AG, -D) = EHC     EX2: f(GG, AF) = EEA   ...`
- run=**cryptn** cat=**cryptarithm** id=`9eacc9a2` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(DG, DF) = -I     EX2: g(II, HD) = EJE    ...`
- run=**cryptn** cat=**cryptarithm** id=`81b6d789` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(EH, DB) = EDF     EX2: g(GA, AI) = IAGA  ...`
- run=**cryptn** cat=**cryptarithm** id=`7f5758a8` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(FH, FH) = FH     EX2: g(FB, FD) = DE     ...`
- run=**cryptn** cat=**cryptarithm** id=`db66c42c` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(AS, GF) = -HP     EX2: f(FM, MP) = -AN   ...`
- run=**cryptn** cat=**cryptarithm** id=`712fe3e4` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(FD, HE) = HJI     EX2: g(BJ, CH) = -DI   ...`
- run=**cryptn** cat=**cryptarithm** id=`6fc35fc2` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(EB, BC) = DBHH     EX2: f(CH, DD) = CAI  ...`
- run=**cryptn** cat=**cryptarithm** id=`236a2204` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(CD, BC) = BEFH     EX2: g(JF, GE) = HF   ...`
- run=**cryptn** cat=**cryptarithm** id=`042f1e53` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(CF, AI) = CFIA     EX2: g(FC, AI) = CF   ...`
- run=**cryptn** cat=**cryptarithm** id=`0f6436da` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(FA, FI) = AC     EX2: f(CG, BE) = FIA    ...`
- run=**cryptn** cat=**cryptarithm** id=`4e8982d6` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(UN, UI) = AU     EX2: h(JP, IM) = IJM    ...`
- run=**cryptn** cat=**cryptarithm** id=`c9463bb4` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(DF, AB) = DIC     EX2: g(DA, AG) = DEGH  ...`
- run=**cryptn** cat=**cryptarithm** id=`5501c054` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(DB, GD) = CE     EX2: f(BA, CG) = G-     ...`
- run=**cryptn** cat=**cryptarithm** id=`a9d91ec1` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(AE, GF) = EEAF     EX2: f(CE, GJ) = BA   ...`
- run=**cryptn** cat=**cryptarithm** id=`36d2d728` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(EF, GB) = EFGB     EX2: g(GJ, EB) = FEFB ...`
- run=**cryptn** cat=**cryptarithm** id=`f5126d48` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(BF, GB) = JI     EX2: f(BG, II) = D     E...`
- run=**cryptn** cat=**cryptarithm** id=`d5602cc5` phrase=`§1`
  - `...tion tree, reading order = [rightward, leftward]: §1 reading rightward:   §1.1 writing equations:     EX1: f(GG, BH) = CIAH     EX2: g(EG, BB) = CI   ...`

## T6. Token-prob divergence (2 flipped puzzles)


### 24e1f1d5 (cryptarithm)
First-differing token position: **45** of 729

Tokens [40..50):
```
  idx |      moe_tok       p |    alleq_tok       p
   40 |         '\n'  1.0000 |         '\n'  1.0000
   41 |          '"'  0.9624 |          '"'  0.9473
   42 |        ' ->'  1.0000 |        ' ->'  1.0000
   43 |         ' F'  0.9988 |         ' F'  0.9999
   44 |         '\n'  1.0000 |         '\n'  1.0000
   45 |          '['  0.6448 |          '('  0.4852 <<
   46 |        ' ->'  1.0000 |        ' ->'  1.0000
   47 |         ' G'  1.0000 |         ' G'  1.0000
   48 |         '\n'  1.0000 |         '\n'  1.0000
   49 |          '#'  0.4190 |          '['  0.7086
```

### 3456da40 (bit_manipulation)
First-differing token position: **5872** of 6400

Tokens [5867..5877):
```
  idx |      moe_tok       p |    alleq_tok       p
 5867 |          ','  1.0000 |          ','  1.0000
 5868 |       ' NOT'  1.0000 |       ' NOT'  1.0000
 5869 |        ' no'  1.0000 |        ' no'  1.0000
 5870 |          ','  1.0000 |          ','  1.0000
 5871 |  ' Constant'  1.0000 |  ' Constant'  1.0000
 5872 |        ' no'  0.9914 |       ' yes'  0.9993 <<
 5873 |          ','  1.0000 |          ','  1.0000
 5874 |       ' AND'  1.0000 |       ' AND'  1.0000
 5875 |        ' no'  1.0000 |        ' no'  1.0000
 5876 |          ','  1.0000 |          ','  1.0000
```

## T7. Net same-data wins/losses vs moe

Aggregate across all 6 same-data categories:
```
run_vs_moe  wins  losses  net
     alleq     8      11   -3
   alleqgt     6       8   -2
    cryptn     7       7    0
```

Per-category breakdown:
```
             cat      vs  wins  losses  net
     cryptarithm   alleq     1       2   -1
     cryptarithm alleqgt     0       1   -1
     cryptarithm  cryptn     0       2   -2
bit_manipulation   alleq     5       7   -2
bit_manipulation alleqgt     5       5    0
bit_manipulation  cryptn     6       4    2
          cipher   alleq     1       1    0
          cipher alleqgt     0       1   -1
          cipher  cryptn     0       1   -1
         gravity   alleq     1       0    1
         gravity alleqgt     1       1    0
         gravity  cryptn     1       0    1
         numeral   alleq     0       0    0
         numeral alleqgt     0       0    0
         numeral  cryptn     0       0    0
 unit_conversion   alleq     0       1   -1
 unit_conversion alleqgt     0       0    0
 unit_conversion  cryptn     0       0    0
```

## Bottom line

**Yes, there is cross-category parameter interference. It is small in magnitude but real, and concentrated in bit_manipulation.**

- Total flips across moe/alleq/alleqgt (where training data was byte-identical): **26**
  - cryptarithm: 3
  - bit_manipulation: 18
  - cipher: 2
  - gravity: 2
  - numeral: 0
  - unit_conversion: 1

- **Interference strongest in bit_manipulation** (18 flips on 160 puzzles = 11%). This is the longest-trajectory category (avg ~6.7k tokens), so the longer the search, the more sensitive to parameter drift.
- **Numeral is rock-solid** (0 flips, 100% across all 4 runs) — short deterministic outputs are immune to interference.
- **Cipher / gravity / unit_conversion**: 1–2 flips each, basically noise.
- **Cryptarithm** is essentially unsolvable by any run (~1–3% acc); the 3 'flips' are accidents of bad models guessing.

- **Format-bleed evidence (T5):** zero numeric-eq phrases (§1, concat_fwd, reading order =) appeared in bit_manipulation, cipher, gravity, numeral, unit_conversion outputs of ANY run. The 266 hits are 100% confined to **cryptn-on-cryptarithm**, which is by design — cryptn's new cryptarithm training data uses the same §-tree / concat_fwd format as numeric-eq, so the format bleed there is expected, not interference. **There is NO observable format leakage into the 5 non-cryptarithm same-data categories.**

- **Token-len drift (T4):** changes vs moe are < ~25 tokens on average for every same-data non-cryptarithm category (e.g. bit_manipulation: alleq +25, alleqgt +6, cryptn +5 tokens out of ~6700). Numeral & cipher unchanged to within 1 token. Cryptarithm's cryptn delta (+4963) is real but again by-design (different training data).

- **Net delta vs moe (T7):** alleq net=-3, alleqgt net=-2, cryptn net=0. Numeric-eq fine-tuning slightly HURTS unrelated categories on net, but the effect is tiny (single-digit on 866 puzzles).

- **Mechanism (T3/T6):** divergences appear deep in long bit_manipulation trajectories (e.g. position 5872 / 6400, position 8447 / 8544), often at search-decision points like ' no' vs ' yes' or '[' vs '(' with moderate probabilities (0.4–0.99). The model's internal search heuristic — which branch to pick when scoring candidate ops — gets perturbed by the new numeric-eq weights, and a single wrong branch cascades through the rest of the search. The parameter interference is therefore subtle: same prefix, then one biased coin-flip mid-search.

- **Where interference is strongest:** bit_manipulation (long DFS-style search where any perturbation can flip the chosen op). Where it is absent: numeral (deterministic short outputs).