| id | score | reasons |
|---|---|---|
| 200086 | 10 | deterministic forward search; operator locks verified; query h(11,48)=-35 -> '`% correct |
| 200106 | 10 | clean digit-count prune + branch-on-fewest; locks verified; query h(29,93)=2698 correct |
| 200111 | 10 | long exhaustive search, all branches pruned with stated reasons; g=|a-b| verified; query 3 correct |
| 200119 | 10 | forward-only; f/g/h locks verified; query g(55,38)=91 correct |
| 200146 | 10 | -|a-b| handled correctly; locks verified; query f(78,12)=-66 -> ]"" correct |
| 200150 | 10 | deterministic; f=-|a-b| verified across EX1/EX5; query -14 -> -%' correct |
| 200175 | 10 | clean mul/add/sub split; locks verified; query f(22,43)=944 correct |
| 200189 | 10 | thorough branch tree; a-b+1 lock verified; query f(33,48)=83 correct |
| 200200 | 10 | f=~add fail then advance handled; locks verified; query g(34,59)=91 correct |
| 200217 | 10 | deterministic; locks verified; query g(43,26)=67 correct |
| 200219 | 10 | forward; g=a-b-1 lock verified; query g(62,66)=-5 -> -> correct |
| 200220 | 10 | f=~add try-first then advance; |a-b| lock verified; query f(93,41)=135 correct |
| 200234 | 10 | |a-b| lock verified across examples; query f(35,97)=62 correct |
| 200245 | 10 | very long exhaustive f=~add elimination, deterministic; query f(45,32)=75 correct |
| 200248 | 10 | forward; a-b-1 lock verified; query g(47,43)=3 correct |
| 200258 | 10 | mul tolerance handled; g=a*b-1 verified; query f(15,31)=44 correct |
| 200270 | 10 | a*b+1 / a-b+2 locks verified; query g(89,37)=54 correct |
| 200273 | 10 | all-diff chain at end slightly under-shown for B but resolves uniquely; query g(96,22)=74 correct |
| 200281 | 10 | long deterministic mul-prune search; a*b-2 verified; query g(11,39)=48 correct |
| 200323 | 10 | forward; a*b-1 lock verified; query f(92,44)=4047 correct |
| 200334 | 10 | clean; locks verified; query g(97,51)=4948 correct |
| 200339 | 9 | cosmetic: top BPE-split preamble lists tokens not all matching this puzzle, but symbol reading + whole search correct; query f(17,61)=79 correct |
| 200358 | 10 | deterministic; g=a-b+2 verified; query f(95,29)=68 correct |
| 200359 | 9 | same cosmetic BPE-preamble mismatch; reading + search correct; query f(33,71)=2344 correct |
| 200360 | 10 | g=a*b-1 lock verified; query f(25,47)=22 correct |
| 200371 | 9 | cosmetic BPE-preamble mismatch; reading + search correct; query g(68,46)=116 correct |
| 200372 | 10 | forward; a+b-1/|a-b| verified; query f(71,90)=160 correct |
| 200378 | 10 | thorough; a-b-1 lock verified; query g(98,40)=136 correct |
| 200384 | 10 | deterministic; a+b+1 verified; query h(37,29)=67 correct |
| 200386 | 9 | cosmetic BPE-preamble mismatch; reading + search correct; query g(12,74)=886 correct |
| 200388 | 10 | f=~add fail then advance to ~sub handled; |a-b| verified; query g(92,70)=6439 correct |
| 200399 | 10 | forward; |a-b| lock verified; query g(13,24)=11 correct |
| 200402 | 10 | long exhaustive; a*b-2 verified; query f(46,15)=688 correct |
| 200409 | 10 | deterministic; a*b-1 verified; query g(96,62)=5951 correct |
| 200414 | 10 | g=~add fail then ~sub at EX5; locks verified; query f(61,28)=91 correct |
| 200422 | 10 | forward; a*b/|a-b| verified; query h(43,27)=16 correct |
| 200436 | 10 | h=~add a+b then a+b-1 variant handled; query f(68,69)=1 correct |
| 200447 | 10 | both g-family combos exhausted deterministically; a*b-2 verified; query f(77,66)=5080 correct |
| 200455 | 10 | a*b/a+b+2/|a-b| verified; query h(50,97)=47 correct |
| 200462 | 10 | g=b-a subtraction variant correctly tried+verified; query g(80,85)=5 correct |
| 200477 | 10 | both f-family combos exhausted; |a-b| verified; query g(31,81)=50 correct |
| 200487 | 10 | both g-family combos exhausted; a+b+2 verified; query g(56,70)=128 correct |
| 200498 | 10 | forward; a*b+1/|a-b|/a+b-1 verified; query g(33,54)=21 correct |
| 200509 | 10 | long exhaustive sub-search, all variants checked; a+b/|a-b| verified; query g(60,14)=46 correct |
| 200510 | 10 | very long deterministic tree, all branches pruned; query g(34,15)=50 correct |
| 200513 | 10 | extremely long but every branch deterministic; a-b+2/a*b verified; query g(63,84)=5292 correct |
| 200527 | 10 | forward; a-b-1/a+b-1/a*b+2 verified; query g(21,17)=37 correct |
| 200546 | 10 | both f-family combos exhausted; |a-b|/a+b-1 verified; query g(98,73)=170 correct |
| 200556 | 10 | deterministic; a+b+2/a-b-2/a*b+1 verified; query f(19,62)=83 correct |
| 200569 | 10 | a*b+1 variant tried+verified; query h(32,97)=3105 correct |
| 200575 | 10 | h=~add a+b variant tested; a*b+2 verified; query h(11,44)=55 correct |
| 200596 | 10 | forward; a-b+2/a*b/a*b-1 verified; query f(17,22)=-3 -> ') correct |
| 200607 | 10 | -|a-b| sub variant tried+verified; query f(44,33)=-11 -> -$$ correct |
| 200624 | 10 | -|a-b| verified; a*b-2/a+b-1 verified; query h(87,92)=178 correct |
| 200631 | 10 | exhaustive E-branch; a*b-2/a+b-2/|a-b| verified; query h(41,22)=19 correct |
| 200657 | 10 | f=~mul deterministic; a*b/a+b/-|a-b| verified; query f(91,21)=1911 correct |
| 200673 | 10 | forward; a+b+1/a*b/|a-b| verified; query g(83,75)=6225 correct |
| 200700 | 10 | long A/F/H branch tree, deterministic; a-b+2/a*b-2/a+b+2 verified; query h(11,23)=36 correct |
| 200722 | 10 | f=~add fail then ~mul handled; a*b-1/a+b/|a-b| verified; query g(14,41)=55 correct |
| 200737 | 10 | deterministic; a+b-1/a*b+2/|a-b| verified; query h(63,85)=22 correct |
| 200758 | 10 | forward; a+b/a*b/a-b-2 verified; query g(91,81)=7371 correct |
| 200759 | 10 | deterministic; a*b-2/a+b+2/a-b+2 verified; query h(22,17)=7 correct |
| 200763 | 10 | a-b sub variant tried+verified; a+b-2/a*b verified; query h(61,14)=47 correct |
| 200771 | 10 | both g-family combos exhausted; a*b+2/|a-b| verified; query g(15,39)=24 correct |
| 200797 | 10 | forward; |a-b|/a+b-1/a*b+1 verified; query g(86,11)=96 correct |
| 200811 | 10 | deterministic; a*b/a-b+2 verified; query g(96,17)=81 correct |
| 200843 | 10 | g=~mul all ±2 variants tested; a+b/a*b-2/a-b+1 verified; query g(95,52)=4938 correct |
| 200846 | 10 | long deterministic B/A/F tree; a*b-2/a+b/|a-b| verified; query h(45,41)=4 correct |
| 200876 | 10 | forward; |a-b|/a*b/a+b verified; query f(59,97)=38 correct |
