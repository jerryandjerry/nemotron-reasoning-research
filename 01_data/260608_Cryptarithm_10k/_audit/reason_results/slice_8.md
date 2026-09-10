| id | score | reasons |
|----|-------|---------|
| crypt10k_008008 | 10 | Deterministic; §1 rightward exhausts, §2 leftward solves all 13; query h(13,83)=|a-b|=70 forced. |
| crypt10k_008012 | 10 | Clean DFS; f=a-b-2 locked at EX3; query f(95,53) forced. |
| crypt10k_008024 | 10 | Full deduction; f=a×b, g=b∥a, h=a+b+2 all locked; query forced. |
| crypt10k_008037 | 10 | Deterministic; g=a+b-2 locked; query g(76,27)=101 forced. |
| crypt10k_008058 | 10 | Long but deterministic; f=a+b+2, h=a×b+1 locked; query forced. |
| crypt10k_008071 | 9 | Query identical to EX1 so answer trivially forced; H=0 picked (not query-relevant); f=a+b is first ~add variant but EX1 reproduction makes answer robust. |
| crypt10k_008087 | 10 | Full deduction f=a+b-2 via EX1; query forced. |
| crypt10k_008103 | 10 | Deterministic; f=a+b+2 locked; query f(58,56)=116 forced. |
| crypt10k_008130 | 10 | Deterministic; g=a-b-1 locked; query operands independent of free I; answer forced. |
| crypt10k_008134 | 10 | Full deduction; g=a×b-1, f=a+b+2 locked; query forced. |
| crypt10k_008181 | 10 | §1 correctly skipped (rhs ends in sign); §2 leftward-fully; f=-|a-b|; query forced. |
| crypt10k_008216 | 10 | Deterministic; f=a×b+2; H free but not in query; answer forced. |
| crypt10k_008246 | 10 | Long deterministic search; f=b∥a, g=a×b; query g(50,20)=1000 forced. |
| crypt10k_008263 | 9 | Deterministic; h=a+b+1 forced by EX3 (H=0 only consistent value, though H=6 not explicitly shown failing); query forced. |
| crypt10k_008280 | 10 | Deterministic; f=a×b+1; G free but not in query; answer forced. |
| crypt10k_008282 | 10 | Full deduction; g=a×b+2; query g(39,93) forced. |
| crypt10k_008287 | 10 | Deterministic; h=a-b-1 locked; query h(47,94)=-48 forced; sign rendered with operator symbol. |
| crypt10k_008298 | 10 | Deterministic; h=a+b-2 locked; query h(75,65)=138 forced. |
| crypt10k_008305 | 10 | Deterministic; f=|a-b|; query f(14,14)=0 forced. |
| crypt10k_008307 | 10 | Full deduction; g=a+b-1 locked; query g(79,45)=123 forced. |
| crypt10k_008317 | 10 | Deterministic; f=a×b locked; query f(56,61)=3416 forced. |
| crypt10k_008354 | 10 | Deterministic; g=a+b-1, f=|a-b| locked; query g(45,65)=109 forced. |
| crypt10k_008374 | 10 | §1 ruled out by digit-count; §2 g=a-b-2; query g(44,62)=-20 forced; sign uses operator symbol. |
| crypt10k_008396 | 10 | Deterministic; f=a×b-1 locked; query f(73,90) forced. |
| crypt10k_008417 | 4 | Query-load-bearing E (operand "E A") appears ONLY in concat EX2 (no numeric constraint) and is set by first-branch E=3 (line 119); answer changes if E=7/9 — not forced. |
| crypt10k_008433 | 10 | Deterministic; f=a+b-1 locked; query f(22,65)=86 forced. |
| crypt10k_008455 | 10 | Long deterministic; h=a×b+2 locked; query h(89,87) forced. |
| crypt10k_008469 | 10 | Deterministic; f=a+b-1 locked; query f(34,66)=99 forced. |
| crypt10k_008519 | 10 | Deterministic; f=a×b+1; J forced by elimination; E/H/I free but not in query; answer forced. |
| crypt10k_008524 | 10 | §1 skipped (rhs ends in sign); §2 leftward-fully; g=a×b+1; query forced. |
| crypt10k_008536 | 10 | Deterministic; h=-|a-b| locked; query h(39,83)=-44 forced; sign uses operator symbol. |
| crypt10k_008551 | 10 | Deterministic; h=|a-b| locked; query h(19,42)=23 forced. |
| crypt10k_008573 | 10 | Deterministic; f=a+b+2 locked via 4-variant probe; query f(86,67)=155 forced. |
| crypt10k_008575 | 10 | §1 skipped (rhs ends in sign); §2 f=a×b+2; query forced. |
| crypt10k_008578 | 10 | Deterministic; h=a×b+2 locked; query h(71,65) forced. |
| crypt10k_008585 | 10 | Deterministic; g=a×b+2 locked; query g(59,94) forced. |
| crypt10k_008623 | 10 | §1 skipped (rhs ends in sign); §2 f=a+b-1; query f(55,81)=135 forced. |
| crypt10k_008638 | 10 | Long deterministic; h=a×b-1 locked; query h(51,48)=2447 forced. |
| crypt10k_008652 | 10 | Deterministic; f=a+b locked; query f(23,69)=92 forced. |
| crypt10k_008653 | 10 | Deterministic; f=a×b locked; query f(35,59)=2065 forced. |
| crypt10k_008656 | 4 | Query-load-bearing H (operand "H E") appears ONLY in concat EX3 and is set by first-branch H=5 (line 111); I also free; answer changes if H=8/9 — not forced. |
| crypt10k_008672 | 10 | Long deterministic; g=a+b+2; I=9/H=7 forced; query g(87,99)=188 forced. |
| crypt10k_008676 | 4 | Query-load-bearing H (operand "G H") appears ONLY in concat EX4; CoT states "H not pinned; smallest H=0" (line 141); answer changes if H differs — not forced. |
| crypt10k_008682 | 6 | Only ONE arithmetic equation (EX1); query operands F,E,B,C and the ~sub variant all rest on first-branch choices (B=1,D=3,A=0,C=4; G smallest); cipher underdetermined, answer not forced. Reasoning is rule-consistent but the puzzle/answer is under-constrained and the CoT does not flag it. |
| crypt10k_008689 | 10 | Deterministic; h=a×b+2 locked; query h(88,57) forced. |
| crypt10k_008728 | 10 | Deterministic; f=a×b+1 locked; query f(24,42)=1009 forced. |
| crypt10k_008748 | 10 | Deterministic; f=a×b+2 locked; query f(53,22)=1168 forced. |
| crypt10k_008757 | 10 | Deterministic; f=a+b-2, g=b-a locked; query f(65,91)=154 forced. |
| crypt10k_008763 | 10 | Deterministic; g=a+b-2 locked; query g(31,92)=121 forced. |
| crypt10k_008768 | 10 | §1 skipped (rhs ends in sign); §2 g=a-b+2; query g(52,57)=-3 forced; sign uses operator symbol. |
| crypt10k_008776 | 4 | Query-load-bearing I (operand "I D") appears ONLY in concat EX4; CoT states "I not pinned; smallest I=3" (line 255); answer changes if I differs — not forced. |
| crypt10k_008780 | 10 | Deterministic; f=a×b; I=8/H=9 forced; query f(21,89)=1869 forced. |
| crypt10k_008793 | 10 | Deterministic; f=a+b-2 locked; query f(59,99)=156 forced. |
| crypt10k_008814 | 10 | Deterministic; g=a-b+1 locked; query g(28,91)=-62 forced; sign uses operator symbol. |
| crypt10k_008827 | 10 | Long deterministic; f=a×b+2, g=|a-b| locked; I free but not in query; query f(93,27) forced. |
| crypt10k_008858 | 10 | Deterministic; g=a+b, f forced; I free but not in query; query g(53,19)=72 forced. |
| crypt10k_008879 | 10 | Deterministic; g=a-b+1, f=a×b+2 locked; query g(41,24)=18 forced. |
| crypt10k_008881 | 10 | Deterministic; g=a×b+1, f=|a-b| locked; query g(39,90)=3511 forced. |
| crypt10k_008895 | 10 | Deterministic; g=a×b+1 locked; query g(39,90) forced (duplicate map of 008881 family, independently derived). |
| crypt10k_008904 | 10 | Deterministic; g=-|a-b|, f=a+b+2 locked; query g(55,17)=-38 forced; sign uses operator symbol. |
| crypt10k_008908 | 10 | Deterministic; h=a×b, g=a+b locked; query h(36,88)=3168 forced. |
| crypt10k_008916 | 10 | Deterministic; f=a+b+1 locked; query f(66,78)=145 forced. |
| crypt10k_008932 | 10 | Deterministic; g=a+b+2, h=a×b-2 locked; B free but not in query; query g(49,67)=118 forced. |
| crypt10k_008934 | 10 | Deterministic; f=a×b-2, h=a+b+1 locked; query h(10,12)=23 forced. |
| crypt10k_008946 | 10 | Long deterministic; f=a×b-1 locked; query f(34,41)=1393 forced. |
| crypt10k_008960 | 10 | Deterministic; h=a×b+1, g=a+b+1 locked; D/E free but not in query; query h(79,29) forced. |
| crypt10k_008972 | 10 | Long deterministic; f=a×b+1, h=a+b locked; query f(94,98)=9213 forced. |
| crypt10k_008975 | 10 | Deterministic; f=a+b+2, g=a×b locked; query f(93,92)=187 forced. |
| crypt10k_008993 | 10 | Deterministic; f=a+b+1, h=a×b-1 locked; E free but not in query; query h(78,61)=4757 forced. |
| crypt10k_008996 | 10 | Deterministic; f=a×b-1, h=a+b locked; I free but not in query; query f(98,82)=8035 forced. |
