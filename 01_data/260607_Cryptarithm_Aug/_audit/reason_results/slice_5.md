| id | score | reasons |
|---|---|---|
| 100569 | 10 | Deterministic; g=sub by rhs sign, §1 rightward exhausted, §2 leftward locks f=a+b+1,g=a-b+1, query g(59,17)=43->34->)] consistent. |
| 100585 | 10 | Sign/concat/digit-count pruning sound; leftward search locks all 12, query g(59,63)=123->321->^!/ consistent. |
| 100609 | 10 | §1 rightward fails (10 symbols, 9 digits), §2 leftward locks f=axb+1,g=a+b-2, query g(48,33)=79->97->!? consistent. |
| 100615 | 10 | Deterministic; g=sub, leftward branch finds f=|a-b|,g=axb-1,h=a+b+1, query h(63,44)=108->801->`&$ consistent. |
| 100618 | 10 | §1 rightward flagged illegal (rhs ends in '-'), §2 leftward-full locks f=a+b+1,g=a-b+1, query 97->79->'? consistent. |
| 100622 | 10 | §1 illegal sign-position, §2 leftward locks f=a+b-1,g=|a-b|,h=axb-2, query h(23,40)=918->819->|}@ consistent. |
| 100626 | 10 | Leftward locks f=a-b+2,g=a+b-1, query f(31,46)=-13->-31, sign rendered as operator | -> |[' consistent. |
| 100634 | 10 | Deterministic search, locks f=a+b+2,g=axb, query g(61,94)=5734->4375->!}][ consistent. |
| 100641 | 10 | Locks f=a+b-1,g=|a-b|,h=axb-2, query h(36,31)=66->66->?? consistent. |
| 100650 | 10 | g=sub, leftward locks f=a-b+2,g=axb-1, query f(22,55)=-31->-13, sign->@ -> @\^ consistent. |
| 100656 | 10 | §1 illegal sign, §2 leftward-full locks f=a+b+1,g=a-b-1, query f(23,48)=72->27->!| consistent. |
| 100658 | 10 | Exhaustive D-branching locks f=|a-b|,g=axb+2,h=a+b+1, query h(96,53)=150->051->({] consistent. |
| 100665 | 10 | g=sub, leftward locks f=a+b-1,g=-|a-b|,h=axb+1, query f(74,80)=153->351->}[| consistent. |
| 100669 | 10 | Locks f=|a-b|,g=axb-1,h=a+b+2, query f(65,22)=43->34->:` consistent. |
| 100673 | 10 | §1 illegal, §2 leftward-full locks f=a+b-2,g=axb-2,h=a-b-1, query h(99,90)=8->8->> consistent. |
| 100678 | 10 | Deterministic; locks f=a+b-1,g=|a-b|,h=axb-2, query f(56,18)... query f maps 79... (h leftward) consistent. |
| 100684 | 10 | f=sub, leftward locks f=a-b,g=a+b+2,h=axb, query g(63,43)=108->801->:/$ consistent. |
| 100687 | 10 | Locks f=a+b-2,g=|a-b|,h=axb+2, query f(79,91)=168->861->$^! consistent. |
| 100711 | 10 | Locks f=axb+2,g=|a-b|,h=a+b+2, query h(81,84)=167->761->'%? consistent. |
| 100735 | 10 | Locks f=axb-2,g=a+b+1, query g(49,98)=148->841->`'? consistent; rejected A=2 branch on all-different. |
| 100740 | 10 | Locks f=axb,g=a+b+2, query g(93,90)=185->581->'{/ consistent. |
| 100762 | 10 | §1 illegal, §2 leftward-full locks f=a+b,g=axb-2,h=a-b-1, query g(88,91)=8006->6008->'??< consistent. |
| 100770 | 10 | §1 illegal, §2 leftward-full locks f=b-a,g=a+b,h=axb-1, query h(90,94)=8459->9548->|?': consistent. |
| 100781 | 10 | §1 illegal, §2 leftward-full locks f=axb-2,g=a-b, query f(61,22)=1340->0431->]$\& (leading-zero is faithful reverse). |
| 100810 | 10 | Locks f=axb+2,g=a+b+1,h=a-b+1, query h(64,47)=18->81->%/ consistent. |
| 100832 | 10 | f=sub, leftward locks f=a+b+2,g=-|a-b|, query g(24,43)=-19->-91, sign->- -> -`) consistent. |
| 100851 | 10 | Locks f=axb+2,g=a+b-1,h=a-b, query h(44,88)=-44->-44, sign->- -> -<< consistent. |
| 100854 | 10 | Locks f=axb+2,g=|a-b|,h=a+b+1, query h(96,78)=175->571->[#| consistent. |
| 100855 | 10 | Locks f=a+b+2,g=|a-b|, query f(55,29)=86->68->'> consistent. |
| 100861 | 10 | Locks f=a+b-1,g=axb,h=|a-b|, query f(21,42)=62->26->}) consistent. |
| 100866 | 10 | Locks f=a+b,g=|a-b|,h=axb, query g(61,21)=40->04->%& (leading-zero faithful reverse). |
| 100872 | 10 | Locks f=axb+2,g=a+b-1, query f(69,39)=2693->3962->%#$! consistent. |
| 100884 | 10 | f=sub, leftward locks f=a+b+1,g=a-b-1, query g(80,32)=47->74->@{ consistent. |
| 100890 | 10 | Locks f=axb,g=a+b-2,h=a-b-2, query g(22,97)=117->711->{)) consistent. |
| 100902 | 10 | Locks f=axb-1,g=a+b-1,h=|a-b|, query f(62,71)=4401->1044->^`!! consistent. |
| 100912 | 10 | Locks f=axb-2,g=a+b+2, query f(31,98)=3036->6303->@`]` consistent. |
| 100931 | 10 | Locks f=|a-b|,g=axb-1, query g(53,15)=794->497-><@| consistent. |
| 100933 | 10 | f=sub, leftward locks f=axb-1,g=a-b, query g(96,97)=-1->-1, sign->/ -> /\ consistent. |
| 100940 | 10 | Locks f=-|a-b|,g=axb,h=a+b, query g(79,59)=4661->1664->/$$) consistent. |
| 100950 | 10 | Locks f=-|a-b|,g=axb+1,h=a+b-2, query h(12,42)=52->25->}) consistent. |
| 100974 | 10 | Locks f=|a-b|,g=a+b+1,h=axb-1, query f(13,27)=14->41->/! consistent. |
| 101003 | 10 | §1 illegal, §2 leftward-full locks f=a+b,g=axb+2,h=a-b+2, query h(87,68)=21->12-><! consistent. |
| 101031 | 10 | Locks f=a+b-1,g=axb+2,h=|a-b|, query f(56,18)=73->37->$! consistent. |
| 101032 | 10 | Locks f=|a-b|,g=a+b-1,h=axb-2, query f(25,36)=11->11->?? consistent. |
| 101040 | 10 | g=sub, leftward locks f=axb-1,g=a+b+2,h=a-b+1, query f(96,15)=1439->9341->%!|$ consistent. |
| 101045 | 10 | Locks f=a-b-2,g=axb-2,h=a+b, query g(66,11)=724->427->^&! consistent. |
| 101052 | 10 | Locks f=a+b+1,g=|a-b|,h=axb+1, query g(59,97)=38->83->%{ consistent. |
| 101057 | 10 | §1 illegal, §2 leftward-full locks f=-|a-b|,g=a+b+1,h=axb-2, query g(73,41)=115->511->?<< consistent. |
| 101063 | 10 | g=sub, leftward locks f=a+b+1,g=axb+2,h=a-b, query f(90,47)=138->831->)[^ consistent. |
| 101076 | 10 | Locks f=a+b,g=|a-b|,h=axb, query h(68,66)=4488->8844-><<&& consistent. |
| 101080 | 10 | g=sub, leftward locks f=a+b-1,g=axb+1,h=a-b-2, query h(64,93)=-31->-13, sign->& -> &@` consistent. |
| 101085 | 10 | §1 illegal, §2 leftward-full locks f=-|a-b|,g=axb-2,h=a+b-1, query f(81,69)=-12->21-, sign->) -> }!) consistent. |
| 101104 | 10 | Locks f=a+b-2,g=axb-2, query f(45,24)=67->76-><^ consistent. |
| 101105 | 10 | Locks f=axb-2,g=a+b+2, query g(31,43)=76->67->:! consistent. |
| 101149 | 10 | Locks f=axb+2,g=|a-b|,h=a+b+1, query g(11,24)=13->31-><( consistent. |
| 101150 | 10 | Locks f=a-b+2,g=a+b+2,h=axb+2, query f(62,68)=-4->-4, sign->[ -> [\ consistent. |
| 101153 | 10 | Locks f=|a-b|,g=a+b-2,h=axb-1, query g(49,71)=118->811->&%% consistent. |
| 101165 | 10 | f=sub, leftward locks f=a-b+1,g=axb+1,h=a+b+2, query g(26,80)=2081->1802->{?:\ consistent. |
| 101178 | 10 | §1 illegal, §2 leftward-full locks f=a+b+2,g=a-b, query f(48,36)=86->68->[) consistent. |
| 101180 | 10 | Locks f=a+b-1,g=axb-2,h=|a-b|, query g(44,14)=614->416->'%! consistent. |
| 101191 | 10 | Locks f=axb-1,g=a+b-1, query f(88,38)=3343->3433-><#<< consistent. |
| 200009 | 10 | Rightward solves directly; f=b-a,g=axb+2, query f(11,35)=24 (no reverse) ->}@ consistent. |
| 200015 | 10 | Rightward; f=a+b-2,g=axb+2,h=|a-b|, query h(48,89)=41->@] consistent. |
| 200022 | 10 | Rightward; f=|a-b|,g=axb-1,h=a+b-2, query f(72,51)=21->@{ consistent. |
| 200024 | 10 | g=sub; rightward locks f=a+b,g=a-b+1,h=axb+1, query h(43,22)=947->#^> consistent. |
| 200033 | 10 | Rightward locks f=axb+1,g=|a-b|,h=a+b+1, query h(50,73)=124->(}[ consistent. |
| 200042 | 10 | h=sub; rightward locks f=a+b+2,g=axb-2,h=-|a-b|, query g(23,17)=389->@^\ consistent. |
| 200055 | 10 | h=sub; rightward locks f=a+b-1,g=axb,h=a-b-1, query h(13,96)=-84, sign->` -> `{( consistent. |
| 200070 | 10 | f=sub; rightward locks f=a-b-1,g=a+b+2, query g(51,20)=73->`( consistent. |
