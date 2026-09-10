| id | score | reasons |
|---|---|---|
| crypt10k_000020 | 10 | Deterministic digit-count + distinct-op pruning, exhaustive branch search, operators locked only when all operands known; query h(27,12)=326 follows forced assignment. |
| crypt10k_000025 | 10 | Forward-only; all operators forced by digit-count, full constraint propagation to unique leaf; query g(51,60)=111 correct from chain. |
| crypt10k_000067 | 10 | All three ops forced; |a-b|, a×b-2, a+b-1 locked from fully-pinned operands; query h(56,38)=93 forced. |
| crypt10k_000073 | 10 | f=|a-b|/g=a×b+2 locked from EX1/EX4 with all operands known; query g(71,68)=4830 forced. |
| crypt10k_000107 | 10 | Clean two-op puzzle; f=a×b+1,g=a+b-2 locked; query f(57,58)=3307. Boxed string \\@} correct (3->\,0->@,7->}). |
| crypt10k_000146 | 10 | f tested -|a-b| (fails EX2) then b-a (passes) forward; negative sign rendered with op symbol "-"; query f(98,44)=-54 -> ->]. |
| crypt10k_000157 | 10 | Long exhaustive backtrack rejecting ~mul/~sub at each leaf by ±2 check; reaches unique f=b-a,g=a×b+2,h=a+b-2; query g(21,42)=884. |
| crypt10k_000176 | 10 | g=a×b lock confirmed across EX2/EX3; f=a+b+2 forced; query f(95,69)=166. |
| crypt10k_000182 | 10 | f=a-b,g=a+b+1,h=a×b+1 all locked from pinned operands; query h(43,52)=2237. |
| crypt10k_000184 | 10 | f tested then locked |a-b|, g=a×b+2,h=a+b; query g(78,12)=938 -> {`). |
| crypt10k_000189 | 10 | f a-b then -|a-b| (a-b fails other eqs) forward; neg sign rendered ^; query f(45,93)=-48 -> ^|<. |
| crypt10k_000193 | 10 | Same disambiguation a-b->-|a-b|; confirmed EX2/EX4; query f(45,93)=-48 -> ^|<. |
| crypt10k_000214 | 10 | g=a-b-2 confirmed EX3/EX5; f=a+b,h=a×b+1; query g(56,52)=2 -> <. |
| crypt10k_000221 | 9 | Reaches correct unique answer but at §line "lock g=a-b+2" the final F=6 is taken as first-branch without showing F=8 fails (it does fail EX2, just not demonstrated); minor determinism-presentation gap. Query g(77,56)=23. |
| crypt10k_000231 | 10 | g=a×b+2 confirmed EX2/EX3; f=a+b+2; query f(95,69)=166 -> ?{{. |
| crypt10k_000244 | 10 | f tested |a-b| (fails) then b-a; g=a×b+2,h=a+b-2; query g(67,68)=4558. |
| crypt10k_000246 | 10 | Exhaustive C-branch; f=a×b+1,g=a+b+1 locked; query g(12,26)=39. |
| crypt10k_000256 | 10 | Nested backtrack on F/I, F=4 both I branches fail then F=6 works; f=a×b,g=a+b-1,h=|a-b|; query h(50,57)=7. |
| crypt10k_000269 | 10 | Very long exhaustive search rejecting leaves by ±2; f=a-b,g=a+b-1,h=a×b-2; query h(67,73)=4889. |
| crypt10k_000280 | 10 | f=a×b-1 confirmed EX1/EX3, g/h locked; query h(63,76)=13 -> !]. |
| crypt10k_000286 | 10 | g=~add exhausted (variant test), then g=~sub path; f=|a-b|,g=a×b,h=a+b-2; query g(81,30)=2430. |
| crypt10k_000302 | 10 | E=2,3 fully exhausted before E=4 leaf; f=a×b-1,g=a+b-1,h=|a-b|; query h(63,76)=13. |
| crypt10k_000311 | 10 | g=a×b-1 chosen only after a×b/+1 fail ±2; f=a-b-1,h=a+b+1; query h(56,50)=107. |
| crypt10k_000316 | 10 | f=|a-b| confirmed EX1/EX3, g=a+b+2,h=a×b+1; query g(53,75)=130. |
| crypt10k_000334 | 10 | f=a+b+1,g=|a-b|,h=a×b-1 locked from pinned operands; query f(51,63)=115. |
| crypt10k_000335 | 10 | f tested -|a-b| forward; g=a×b-2 confirmed EX2 (a×b-1/-2 disambiguated); query g(52,70)=3638. |
| crypt10k_000359 | 10 | Exhaustive noisy-add variant testing (a+b±0/1/2) across many leaves; f=|a-b|,g=a+b+1; query g(41,77)=119. |
| crypt10k_000411 | 10 | f=a+b+1,g=a×b+1,h=|a-b| locked; query f(60,89)=150. Box }'^ correct (brace doubling is cosmetic). |
| crypt10k_000465 | 10 | h=a-b locked from EX4; f=a×b-2,g=a+b+1; query h(42,69)=-27 -> -[{ (neg sign = op symbol -). |
| crypt10k_000468 | 10 | Deep nested backtrack; f=a+b,g=a-b-2,h=a×b+2; query g(43,91)=-50 -> -&?. |
| crypt10k_000495 | 10 | Long exhaustive A/F/C/J search; f=a+b,g=a-b-2,h=a×b+2 locked; query g(43,91)=-50 -> -&?. |
| crypt10k_000500 | 10 | f=a+b+1,g=a×b,h=|a-b| locked; query h(94,81)=13 -> }| (brace doubling cosmetic). |
| crypt10k_000501 | 10 | Multiple ~mul variant rejections (±2); f=a×b-2,g=a-b+2,h=a+b-1; query g(55,87)=-30 -> "%{ (neg sign ="). |
| crypt10k_000529 | 10 | f tested a×b then a×b+1/-1/+2 at leaf; f=|a-b|,g=a×b+1,h=a+b; query f(80,46)=34. |
| crypt10k_000541 | 10 | g=a×b confirmed EX2/EX4 by ±2; f=b-a,g=a×b; query f(72,89)=17 -> )}. |
| crypt10k_000552 | 10 | g=a-b-2 locked from EX4; f=a×b+2 confirmed EX1/EX2; query f(39,62)=2420. |
| crypt10k_000566 | 10 | Deep H/J/A/C/G/E nested search to unique leaf; f=a×b-2,g=a-b+2,h=a+b; query f(74,41)=3032. |
| crypt10k_000605 | 10 | f=a×b-1 lock EX4, query g(96,97)=-1 -> /: (1-digit neg with op symbol /); deterministic. |
| crypt10k_000626 | 10 | Free var G (cancels in EX2, not query-relevant) set by stated smallest-rule; f=b-a,g=a×b; query f(98,67)=6566. |
| crypt10k_000647 | 10 | Exhaustive A/H branch; g determined a+b+2/+1/-1/-2 by ±2; f=a+b-1,g=a-b-1; query f(48,77)=124. |
| crypt10k_000656 | 10 | h=a×b+1 confirmed EX4/EX5; f=a+b-2,g=a-b,h=a×b+1; query h(78,18)=1405. |
| crypt10k_000681 | 10 | f=|a-b| confirmed EX4/EX5; g=a×b-1; query g(73,95)=6934. |
| crypt10k_000691 | 10 | f=a+b-2,g=|a-b|,h=a×b+2 locked; query h(57,15)=857 -> ?}#. |
| crypt10k_000702 | 10 | f=a+b then g=a×b+2 confirmed EX3/EX4 by ±2; query f(26,32)=58. |
| crypt10k_000708 | 10 | f tested |a-b| (fails) then leaf; f=|a-b|,g=a+b+2,h=a×b+2; query h(57,15)=857. |
| crypt10k_000721 | 10 | f=a-b-2 lock EX1, g=a+b,h=a×b; query f(56,55)=-1 -> -$ (neg with op symbol -). |
| crypt10k_000733 | 10 | f=a+b+2,g=|a-b|,h=a×b-1 locked from pinned operands; query f(25,67)=94. |
| crypt10k_000773 | 10 | g=a×b lock EX3, query f(72,89)=17 (b-a); negative-sign handling correct. |
| crypt10k_000788 | 10 | Very long exhaustive A/F/H/C search; f=a+b,g=a×b+1,h=a-b+2; query f(39,17)=56. |
| crypt10k_000843 | 10 | Deep H/J/A/B exhaustive; f=b-a (a-b fails), g=a×b; query f(72,89)=17 -> )}. |
| crypt10k_000866 | 10 | f=a-b-2 lock at C=8 leaf; g=a+b,h=a×b; query f(56,55)=-1 -> -$. |
| crypt10k_000867 | 10 | Exhaustive A/C/B search to unique leaf; f=a×b+1,g=a+b+1,h=|a-b|; query g(67,79)=147. |
| crypt10k_000871 | 10 | Long E/F/G exhaustive with ±2 rejections; f=a×b+1,g=a-b-1; query f(27,25)=676 -> |&|. |
| crypt10k_000879 | 10 | Massive A/C/B/J search rejecting leaves by ±2; f=a×b-2,g=a+b+1,h=|a-b|; query f(88,17)=103. |
| crypt10k_000887 | 10 | f=a×b-1,g=|a-b|,h=a+b+2 locked; query h(68,52)=122 -> \%%. |
| crypt10k_000907 | 10 | Deep A/B exhaustive; f=a-b-1,g=a×b,h=a+b-2 confirmed EX2/EX4; query h(14,51)=63. |
| crypt10k_000910 | 10 | f=a×b-1,g=|a-b|,h=a+b+2 confirmed EX3; query g(76,45)=31 -> [%. |
| crypt10k_000912 | 9 | Correct, but query f(76,66) syntactically uses undetermined I (set by smallest-rule; EX3 leaves I free); answer is invariant (I=6 or 8 both give 10) yet CoT doesn't note the invariance. |
| crypt10k_000918 | 10 | f=a×b-2,g=a-b+1,h=a+b+1 locked from pinned operands; query g(38,82)=-43 -> (<^ (neg sign =(). |
| crypt10k_000938 | 10 | Exhaustive A/C search; f=a+b+1,g=a×b-1,h=|a-b| confirmed EX3; query g(89,40)=3559. |
| crypt10k_000945 | 10 | f=~add path exhausted then f=~sub; f tested at leaf; f=|a-b|,g=a×b-2,h=a+b+2; query g(82,41)=3360. |
| crypt10k_000955 | 10 | f=a+b then leaf; f=a+b+1,g=a×b-1 confirmed EX1/EX3; query f(45,99)=145. |
| crypt10k_000960 | 10 | f=a+b lock EX1, g=a×b-1,h=|a-b| confirmed EX5; query h(94,49)=45 -> |\. |
| crypt10k_000963 | 10 | Exhaustive A/E/C/G search rejecting by ±2; f=|a-b|,g=a+b+1,h=a×b-1; query f(73,22)=51. |
| crypt10k_000971 | 10 | f=|a-b| confirmed EX4/EX5; g=a+b+1,h=a×b; query g(81,15)=97 -> !|. |
| crypt10k_000999 | 10 | Constraint propagation pins all in §1.4; f=|a-b|,g=a+b+2,h=a×b+2; query g(44,18)=64. |
| crypt10k_001052 | 10 | Exhaustive A/E/B search; f=a×b+2,g=|a-b|,h=a+b-1; query g(42,88)=46 -> `<. |
| crypt10k_001055 | 10 | Long A/J exhaustive; f=a+b+2,g=a×b+2; query f(40,71)=113 -> ##}. |
| crypt10k_001071 | 10 | All locks from pinned operands in §1.4; f=a-b-1,g=a+b-2,h=a×b-1; query g(19,67)=84 -> ^". |
| crypt10k_001075 | 10 | Exhaustive A/B/F search; f=a×b-2,g=a+b+1; query f(30,96)=2878 -> @[?[. |
