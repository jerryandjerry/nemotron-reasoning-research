| id | score | reasons |
|---|---|---|
| 001036 | 10 | deterministic; cipher+ops forced by digit-count + range pruning; query f(41,26)=1064 follows |
| 001066 | 10 | all ops/operands forced; dead branches abandoned correctly; query g(21,21)=43 consistent |
| 001093 | 10 | f,g,h all locked from examples; failing C=4 branch rejected, C=5 closes; query h(40,79)=119 |
| 001094 | 10 | every var pinned by example ranges; h=a×b+1 locked on EX3; query h(70,72)=5041 |
| 001101 | 10 | full A-branch search deterministic; f=a×b locked EX2; query f(71,57)=4047 |
| 001118 | 10 | exhaustive C/H search, g=a×b-2 locked+confirmed on EX4; query g(55,78)=4288 |
| 001130 | 10 | g=neg-sign by rhs; f=-\|a-b\| locked after a×b-2 failed g-check then a×b-1; query g(87,22)=1913 |
| 001134 | 10 | f,g,h forced; B/D=9 conflict rejected; query f(54,11)=65 consistent |
| 001138 | 10 | h=a-b+2 locked EX5, EX3 pins remaining; query f(13,75)=88 |
| 001165 | 10 | f,g,h locked; g=\|a-b\| matched EX2; query f(77,12)=91; unpinned G not in query |
| 001173 | 10 | f=a×b-1 locked EX1+confirmed EX5; g=-\|a-b\| matched; query f(82,48)=3935 |
| 001194 | 10 | full I/C/F search deterministic; f=a×b-2 locked+confirmed; query f(31,15)=463 |
| 001201 | 7 | severely underdetermined: only EX1 has operator f, operands A=1,C=2,D=4,E=3 are first-branch guesses; G=6,H=7 guessed; query f(12,20)=33 rides on guessed operand digits |
| 001213 | 7 | underdetermined: 2 add-eqs + concat, operands B=1,D=2,F=3,G=5 partly first-branch picks feeding query g(31,20)=51; C guessed (not in query) |
| 001217 | 10 | f=a×b-1 locked; G=6 forced by elimination not arbitrary; query f(63,11)=692 |
| 001221 | 10 | f,g,h locked; g=-\|a-b\| chosen after a-b failed EX4; query g(61,85)=-24 rendered <&] |
| 001230 | 5 | EX2 g(DD,DD)=0 gives no constraint for any digits; C=2,D=3,E=4 all guessed first-branch; query g(22,34)=12 depends entirely on arbitrary operand guesses |
| 001282 | 10 | full A=5 branch pins all 13 symbols; f=a×b+2,h=a-b+2 locked; query h(87,27)=62 |
| 001284 | 7 | only EX1 constrains f; A=2,C=7,B=8 first-branch guesses feed query f(23,78)=103; g locked on lone EX2 |
| 001304 | 9 | §1 rightward fails (E,I both forced 1), §2 leftward solves; B=3 branch-guess feeds query h(38,38); operator forced |
| 001334 | 10 | §1 fails, §2 leftward fully determined; f=a×b+2 locked; query h(91,19)=111 |
| 001346 | 10 | §1 fails (10 symbols, 9 digits), §2 leftward fully pins all; query f(24,52)=74→47 |
| 001347 | 6 | §2 leftward; F unconstrained by EX2 (5F-3F+1=21 for any F), F=6 guessed but F is in query h(18,68); answer rides on arbitrary F |
| 001359 | 10 | §1 illegal (rhs trailing sign), §2 leftward fully pins; f=a-b-2 locked; query f(10,27)=-19→`(- |
| 001363 | 10 | §2 leftward; h locked after a-b+1 failed J branches; query h(80,30)=52→25 |
| 001370 | 10 | §2 leftward fully determined; f=b∥a,g=a+b+1,h=a×b-2; query h(69,95)=6553→3556 |
| 001410 | 10 | §2 leftward, full D/G search pins all; f=a×b+1 locked; query f(50,18)=901→109 |
| 001423 | 10 | §1 illegal, §2 leftward fully; f=a×b+2,g=a-b-1 locked; query f(99,70)=6932→2396 |
| 001435 | 10 | §2 leftward; B=2 branch rejected, B=4 closes; query g(51,58)=108→801 |
| 001444 | 10 | §2 leftward fully pins; f=a-b-1 locked+confirmed EX2; query f(42,91)=-50→-#@ |
| 001458 | 10 | §2 leftward; g=a×b-1 locked on EX5; query f(43,20)=23→32 |
| 001476 | 10 | §2 leftward exhaustive; f=a+b-2,g=a-b+1 locked; query f(39,35)=72→27 |
| 001482 | 10 | §2 leftward, full F/B/J search pins all; g=a×b+1 locked; query g(41,88)=3609→9063 |
| 001503 | 10 | §2 leftward; g locked after a×b,a×b+1 failed F constraint; query f(97,13)=112→211 |
| 001507 | 10 | §1 illegal, §2 leftward fully pins; f=a+b-1,h=a-b+1; query h(36,26)=11; F guessed not in query |
| 001542 | 8 | sparse (3 eqs); B=2 branch-guess feeds query g(22,21) but anchored by EX1's tight 111 constraint; g=\|a-b\| from EX2; H guessed not in query |
| 001545 | 10 | §2 leftward fully pinned by propagation (no branching needed); f=a+b-1,g=a×b+1; query f(63,26)=88 |
| 001548 | 10 | §2 leftward full B search; f=a+b-1,g=a×b-1 locked; query f(73,83)=155→551 |
| 001589 | 10 | §2 leftward exhaustive; f=a-b+1 locked after many ~sub fails; query h(22,71)=1560→0651 |
| 001594 | 6 | §2 leftward; after locks F,G,H,I free among {1,4,6,7}, H=6 guessed+G=4 branch both feed query f(56,54); operands' 2nd digits arbitrary |
| 001604 | 10 | §2 leftward; A=6 forced (A=7 contradicts E=5 elim); query h(59,96)=156→651 |
| 001606 | 10 | §2 leftward fully pins; f=a+b+1,g=a∥b,h=a×b-2; query h(62,48)=2974→4792 |
| 001619 | 10 | §1 exhaustively fails, §2 leftward pins; g=a×b+2 locked; query g(88,98)=8626→6268; B guessed not in query |
| 001664 | 5 | only 1 arithmetic eq (EX1 sub, repdigit−2digit=−repdigit); f=a-b+1 derived from guessed A=2,B=3,C=4,D=1; query f(45,45)=1 depends on which sub-offset, undetermined |
| 001703 | 10 | §2 leftward fully pins; f=a+b,g=a×b+1,h=b∥a; query f(11,60)=71→17 |
| 001706 | 10 | §1 illegal, §2 leftward exhaustive ~sub; g=a-b-2 locked; query h(28,19)=49→94 |
| 001719 | 10 | §2 leftward; f locked after a×b/a×b±1 failed; query f(35,44)=1542→2451 |
| 001792 | 10 | §2 leftward exhaustive; f=a-b-2,g=b∥a,h=a+b-1; query f(81,26)=53→35 |
| 001805 | 10 | §1 illegal, §2 leftward; f locked after a×b-1 failed EX5; query f(45,59)=2653→3562 |
| 001826 | 10 | §2 leftward fully pins via propagation; f=a×b-1,g=a+b+1; query f(77,34)=2617→7162 |
| 001827 | 8 | §1 rightward; A=1,B=2 first-branch picks feed query g(12,39)=53; two add-eqs (EX4,EX5) anchor g=a+b+2; G guessed not in query |
| 001863 | 10 | §2 leftward minimal branching; f=a+b-2,g=a×b-2; query g(51,73)=3721→1273 |
| 001866 | 5 | sparse (1 add+1 sub+1 concat); B=1,A=0,C=3,D=4,E=5,F=6 all first-branch guesses; g=a-b+1 and query operands g(61,40) all ride on guesses |
| 001875 | 10 | §1 illegal, §2 leftward exhaustive; f=a+b+2,h=-\|a-b\| locked; query f(63,31)=96→69 |
| 001904 | 10 | §2 leftward; C=5 branch fully dead-ends, C=6 closes; g=a×b; query g(59,45)=2655→5562 |
| 001944 | 10 | §2 leftward; H=2 branches fail, F=9/H=3 closes; f=a+b; query f(77,57)=134→431 |
| 001966 | 10 | §2 leftward exhaustive H search; f=b∥a,g=a+b,h=a×b+2 locked+confirmed; query g(62,59)=121 |
| 001994 | 10 | §2 leftward; h=\|a-b\| locked EX3+confirmed EX4; query h(35,35)=0→< (operand-equal, robust) |
| 001999 | 10 | §2 leftward exhaustive; f=-\|a-b\| locked+confirmed EX5; query f(52,30)=-22→>$] |
| 002029 | 10 | §2 leftward; g=\|a-b\| matched EX2; query g(39,66)=27→72 |
| 002038 | 6 | §2 leftward; EX3 holds for any F (both sides 102+10F), so F unconstrained; F=4 guessed but F in query h(14,22); answer rides on arbitrary F |
| 002040 | 10 | §2 leftward; f=-\|a-b\| locked after h-variants failed; query f(28,86)=-58→>$] |
| 002064 | 10 | §2 leftward exhaustive; f=a-b+1,g=a×b locked; query f(38,74)=-35→-#` |
| 002094 | 10 | §2 leftward fully pins; f=a∥b,g=a×b+1,h=a+b-1; query h(61,56)=116→611 |
| 002174 | 10 | §2 leftward; h locked after a×b-1 failed verify; query h(66,34)=2242→2422 |
| 002188 | 9 | §2 leftward; B=8 closes first (B=9 untested), B feeds query f(38,71); operator f=a+b+2 forced; E=1 pin tightens B |
| 002206 | 10 | §1 illegal, §2 leftward exhaustive ~sub; f=a-b-1,h=a+b-1; query h(11,29)=39→93; D guessed not in query |
| 002207 | 10 | §2 leftward exhaustive; f=\|a-b\| locked; query h(42,33)=76→67 |
| 002209 | 10 | §2 leftward; f=a×b-2 locked on EX1; query f(57,49)=2791→1972 |
| 002251 | 6 | §2 leftward; g locked after a+b-1 failed EX2; B unconstrained (concat EX1 only), B=4 guessed but B in query g(41,79); answer rides on arbitrary B |
