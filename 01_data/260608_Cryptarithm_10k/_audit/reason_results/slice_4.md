| id | score | reasons |
| --- | --- | --- |
| crypt10k_004037 | 10 | Deterministic: §1 fails to resolve, §2 leftward forces f=a×b-2, g=a+b via digit-count pruning + operand narrowing; query f(94,93)=8740 reversed correctly. Forward-only, self-contained. |
| crypt10k_004061 | 9 | Sound DFS; operators forced by digit-count/sign; query g(70,33)=39 correct. Minor: query operand digits A=7/B=3 reached via first complete branch (B=8/9 branches failed first), uniqueness not proven though g is query operator and forced. |
| crypt10k_004087 | 10 | §1 illegal (rhs sign at end), §2 leftward fully; g=-|a-b|, f=a+b-1 locked with confirmations; query g(86,82)=167 reversed to 761 correct. Clean. |
| crypt10k_004112 | 9 | Deep DFS (D,A branches); f=a+b, g=a×b-2 locked and confirmed; query g(33,54)=1780→0871 correct. Query operands D/A reached at first full solution, not proven unique. |
| crypt10k_004131 | 9 | g=|a-b|, f=a+b-2 forced; query f(86,10)=94→49 correct. Query operand A=8/G=6 reached via first surviving branch after several failures; not proven unique but f forced. |
| crypt10k_004153 | 9 | Long exhaustive A-branch; f=a+b, g=a×b-1, h=a-b all locked+confirmed; query g(28,82)=2295→5922 correct. Branch-terminating at first solution; otherwise rigorous. |
| crypt10k_004160 | 10 | f=a+b-2, g=a-b-1, h=a×b-1 each locked with EX confirmations; query h(53,66)=3497→7943 correct. Deterministic and self-contained. |
| crypt10k_004163 | 8 | g=a×b+1 (query op) forced by EX2 alone; but query operand A=6 reached after A=3,5 failed and solver stopped without testing A=8,9 — query-load-bearing AA not proven unique; answer g(66,10)=661→166 hinges on A. Reasoning otherwise sound. |
| crypt10k_004190 | 9 | f=a+b+2, g=|a-b| locked and confirmed across EX3/EX4/EX5; query f(91,84)=177→771 correct. Deep branch, first-solution stop. |
| crypt10k_004198 | 9 | f=a×b+1, g=a+b+2, h=a-b+2 locked+confirmed; query f(80,46)=3681→1863 correct. B/I/D branch, terminates at first full solution. |
| crypt10k_004206 | 9 | §1 illegal, §2 fully; f=a+b+2, g=a×b, h=a-b-2 locked; query g(91,69)=6279→9726 correct. Deep B/C/A/H/I branching, first-solution stop. |
| crypt10k_004249 | 9 | f=|a-b| distinguished from a-b; g=a+b-2, h=a×b-2 locked; query h(17,91)=1545→5451 correct. Very deep DFS, terminates at first solution. |
| crypt10k_004258 | 10 | f=a+b, g=a×b, h=|a-b| locked+confirmed; query h(12,23)=11→11 correct. Clean. |
| crypt10k_004260 | 10 | f=a×b+2, g=|a-b|, h=a+b+2 locked, f confirmed on EX3; query g(56,41)=15→51 correct. |
| crypt10k_004261 | 10 | f=a×b+2, g=a+b+1 locked+confirmed (EX1/EX5); query f(48,13)=626 palindrome correct. |
| crypt10k_004270 | 9 | f=a×b-2, g=a-b+1, h=a+b-2 locked; query g(98,46)=53→35 correct. Branch on I at end (first solution); otherwise forced. |
| crypt10k_004277 | 10 | §1 illegal, §2 fully; f=a-b, g=a+b+2, h=a×b-2 locked+confirmed; query f(21,78)=-57→"75-" with sign symbol correct. Negative-result rendering consistent. |
| crypt10k_004278 | 9 | f=a×b-2, g=a+b locked; query g(22,38)=60→06 correct. Minor: \boxed{}'} has cipher "}" adjacent to brace (rendering quirk, not a reasoning flaw). |
| crypt10k_004286 | 10 | f=a+b-2, g=a×b+2, h=a-b locked; query g(36,18)=650→056 correct. Clean. |
| crypt10k_004340 | 10 | f=a+b+2, g=-|a-b| locked+confirmed; query g(80,18)=-62→"-26"→"-<%" correct, sign rendered as operator. |
| crypt10k_004355 | 10 | §1 illegal, §2 fully; f=|a-b| explicitly rejected (eqs don't all check) then f=a-b locked+confirmed; query f(74,69)=5→"^" correct. Strong discrimination. |
| crypt10k_004379 | 10 | f=a+b-2, g=a×b+1, h=|a-b| locked+confirmed; query h(12,43)=31→13 correct. |
| crypt10k_004390 | 10 | f=|a-b|, g=a+b+2, h=a×b+2 locked+confirmed; query f(94,49)=45→54 correct. |
| crypt10k_004396 | 9 | f=|a-b|, g=a+b+2, h=a×b+2 locked; query f(56,62)=6→"]" wait ")" correct. A reached after A-branch failures, first-solution stop. |
| crypt10k_004397 | 10 | f=a-b-2, g=a+b-2, h=a×b-2 locked; query h(83,43)=3567→7653 correct. |
| crypt10k_004413 | 10 | §1 illegal, §2 fully; f=a+b-1, g=a-b-1, h=a×b+2 locked; query f(94,69)=162→261 correct. |
| crypt10k_004417 | 10 | §1 illegal, §2 fully; f=-|a-b|, g=a×b+1 locked+confirmed (EX1/EX3); query g(98,29)=2843→3482 correct. |
| crypt10k_004451 | 9 | f=a×b-1, g=|a-b|, h=a+b locked; query h(52,30)=82→28 correct. Minor: \boxed{{|} cipher "{" adjacent to brace (rendering quirk). |
| crypt10k_004483 | 10 | f=-|a-b|, g=a×b+2, h=a+b locked+confirmed; query h(13,18)=31→13 correct. |
| crypt10k_004500 | 9 | f=|a-b|, g=a+b+2 locked; query g(33,37)=72→27 correct. D-branch first-solution stop; otherwise forced. |
| crypt10k_004511 | 10 | f=a×b+1, g=a+b-2, h=|a-b| locked; query h(18,29)=11→11 correct. |
| crypt10k_004514 | 10 | f=a×b, g=a+b-2 locked; query g(22,92)=112→211 correct. |
| crypt10k_004535 | 10 | f=|a-b|, g=a×b+2, h=a+b+1 locked; query f(29,66)=37→73 correct. |
| crypt10k_004537 | 10 | §1 illegal, §2 fully; f=a+b, g=-|a-b|, h=a×b+2 locked; query g(87,75)=-12→"21-"→"":-" correct, sign as operator. |
| crypt10k_004538 | 9 | f=a+b, g=a-b+2 locked; query g(84,51)=35→53 correct. Minor: \boxed{}:} cipher "}" adjacent to brace (rendering quirk). G-branch first-solution stop. |
| crypt10k_004558 | 10 | f=-|a-b|, g=a×b, h=a+b-2 locked; query g(99,12)=1188→8811 correct. |
| crypt10k_004562 | 10 | §1 illegal, §2 fully; f=a+b+1, g=a-b+1 locked+confirmed (EX4); query f(90,33)=124→421 correct. |
| crypt10k_004567 | 10 | f=a-b+1, g=a×b+1, h=a+b+1 locked; query h(22,41)=64→46 correct. |
| crypt10k_004569 | 9 | §1 illegal, §2 fully; f=a+b+2, g=-|a-b| locked+confirmed; query f(13,27)=42→24 correct. Minor: \boxed{(}} cipher "}" adjacent to brace (rendering quirk). |
| crypt10k_004574 | 10 | f=a+b, g=|a-b| locked+confirmed (EX4); query f(45,76)=121 palindrome correct. |
| crypt10k_004603 | 10 | f=a+b-1, g=b-a (operand-reversed sub variant) locked+confirmed, h=a×b-2; query f(59,83)=141 palindrome correct. Good handling of b-a. |
| crypt10k_004607 | 9 | f=|a-b|, g=a+b-2 locked+confirmed (EX3/EX5); query f(12,47)=35→53 correct. I-branch at end (first solution); operators forced. |
| crypt10k_004610 | 9 | f=|a-b|, g=a+b-1, h=a×b-2 locked; query h(69,20)=1378→8731 correct. Minor: \boxed{&?$}} cipher "}" adjacent to brace (rendering quirk). |
| crypt10k_004626 | 10 | f=a+b-2, g=a-b-2 locked; query g(18,85)=-69→"-96"→"-<[" correct, sign as operator. |
| crypt10k_004628 | 10 | f=a+b-1, g=a×b+2, h=|a-b| locked+confirmed (EX5); query g(88,41)=3610→0163 correct. |
| crypt10k_004649 | 10 | f=a×b-1, g=a+b locked; query f(28,25)=699→996 correct. |
| crypt10k_004655 | 10 | §1 illegal, §2 fully; f=a×b, g=a-b locked; query f(97,79)=7663→3667 correct. |
| crypt10k_004677 | 10 | §1 illegal, §2 fully; f=a-b+1, g=a×b, h=a+b+2 locked; query g(53,10)=530→035 correct (leading-zero output fine). |
| crypt10k_004682 | 9 | f=a+b+1, g=|a-b|, h=a×b+1 locked+confirmed (EX3); query h(29,80)=2321→1232 correct. B-branch first-solution stop. |
| crypt10k_004696 | 9 | f=|a-b|, g=a+b+1, h=a×b locked; query h(36,44)=1584→4851 correct. Branch-terminating at first solution; operators forced by digit count. |
| crypt10k_004703 | 10 | f=a+b-2, g=a×b+2, h=a-b locked; query f(97,43)=138→831 correct. Minor: \boxed{?<}} cipher "}" adjacent to brace (rendering quirk). |
| crypt10k_004719 | 10 | f=a×b-2, g=|a-b|, h=a+b-1 locked+confirmed (EX4); query g(73,56)=17→71 correct. Minor brace adjacency only. |
| crypt10k_004721 | 10 | f=a×b-1, g=|a-b|, h=a+b locked; query g(68,28)=40→04 correct. |
| crypt10k_004731 | 10 | f=a-b+1, g=a+b+1, h=a×b-2 locked; query g(89,43)=133→331 correct. |
| crypt10k_004771 | 10 | f=a+b+2, g=a-b+1, h=a×b-2 locked; query g(49,45)=5→"\"" correct. |
| crypt10k_004780 | 10 | f=a×b, g=|a-b|, h=a+b-2 locked; query g(99,64)=35→53 correct. |
| crypt10k_004783 | 10 | f=|a-b|, g=a+b-1, h=a×b+2 locked+confirmed (EX3); query h(37,80)=2962→2692 correct. |
| crypt10k_004798 | 10 | f=a+b, g=a×b-1, h=|a-b| locked+confirmed (EX5); query f(76,85)=161 palindrome correct. |
| crypt10k_004841 | 10 | f=a+b+1, g=|a-b| locked+confirmed (EX3); query g(65,51)=14→41 correct. |
| crypt10k_004848 | 10 | f=a+b+1, g=-|a-b| locked+confirmed (EX3/EX4/EX5); query g(15,37)=-22→"-22"→"($$" correct, sign rendered as operator "(". |
| crypt10k_004851 | 10 | §1 illegal, §2 fully; f=a+b-2, g=a×b+2, h=-|a-b| locked+confirmed (EX5); query h(51,54)=-3→"3-"→"]-" correct. |
| crypt10k_004868 | 10 | f=a+b+2, g=a×b-2, h=|a-b| locked; query f(38,25)=65→56 correct. |
| crypt10k_004870 | 10 | f=a-b-2, g=a+b+2, h=a×b-2 locked+confirmed (EX3/EX5); query g(56,18)=76→67 correct. |
| crypt10k_004881 | 10 | f=|a-b|, g=a×b+2, h=a+b+1 locked; query h(95,38)=134→431 correct. |
| crypt10k_004890 | 10 | f=|a-b|, g=a×b-2, h=a+b+1 locked; query f(59,78)=19→91 correct. |
| crypt10k_004901 | 10 | f=a×b-1, g=a+b+2, h=-|a-b| locked; query h(75,70)=-5→"-5"→"`:" correct, sign as operator. |
| crypt10k_004902 | 10 | f=a+b, g=a-b-2, h=a×b-2 locked; query g(25,59)=-36→"-63"→""|>" correct, sign as operator. |
| crypt10k_004938 | 10 | §1 illegal, §2 fully; f=a+b-2, g=a×b-1, h=-|a-b| locked+confirmed (EX5); query g(44,18)=791→197 correct. |
| crypt10k_004941 | 10 | f=a+b-2, g=|a-b|, h=a×b-1 locked; query g(15,51)=36→63 correct. |
| crypt10k_004958 | 10 | f=a×b, g=a+b-2, h=a-b locked; query h(92,97)=-5→"-5"→"!@" correct, sign as operator. |
