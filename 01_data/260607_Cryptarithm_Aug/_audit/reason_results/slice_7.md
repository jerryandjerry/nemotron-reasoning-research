| id | score | reasons |
|---|---|---|
| 200925 | 10 | Deterministic throughout; failed D=3 branch re-locks f cleanly in the surviving D=6 branch; query 71+53-2=122 -> ?}} consistent. |
| 200936 | 10 | Forward-only digit-count prune + domain narrowing; all locks verified against examples; query 41×59+2=2421 consistent. |
| 200950 | 10 | Clean rightward solve; operator locks cross-checked on other equations; query |56-83|=27 consistent. |
| 200961 | 10 | Small 2-operator puzzle, fully deterministic; f=a+b, g=|a-b|; query 60+62=122 consistent. |
| 200976 | 10 | Long DFS with many faithful dead-ends; -|a-b| only matches at the end; query -|94-33|=-61 consistent. |
| 200984 | 10 | Rightward solve, all three operators locked and cross-checked; query 95-35+2=62 consistent. |
| 200987 | 10 | Deep search, every branch closed by a stated rule; f=a×b-1 cross-checked; query 31-11+1=21 consistent. |
| 201004 | 10 | f=|a-b|,g=a+b+1,h=a×b+2 all verified; query 47×53+2=2493 consistent. |
| 201015 | 10 | Deterministic branching, locks cross-checked on EX2/EX4; query 64×24+1=1537 consistent. |
| 201025 | 10 | Negative-sign->~sub used; g=a-b+2 rendered with operator symbol for the sign; query 32-70+2=-36 -> `$^ consistent. |
| 201026 | 10 | Full 10-symbol elimination J=0; all-different deductions sound; query 13+90-1=102 consistent. |
| 201027 | 10 | Tight rightward solve, no late branching needed; query 94+66+1=161 consistent. |
| 201028 | 10 | Long but faithful DFS; f=a+b+1,g=a×b-2,h=|a-b| cross-checked; query 55+39+1=95 consistent. |
| 201041 | 10 | Negative-sign g=-|a-b|; failed B=4 branch faithfully discarded; query 99×11-2=1087 consistent. |
| 201045 | 10 | Multi-equation cross-check on f=a×b; query 15×21=315 consistent. |
| 201064 | 10 | f=|a-b| found only after exhausting B/D branches; query 30×53-2=1588 consistent. |
| 201067 | 10 | Deep DFS, negative-sign f locked late as a-b+1; query 56-22+1=35 consistent. |
| 201070 | 10 | Clean 2-operator solve; query 32+17+2=51 consistent. |
| 201091 | 10 | f=-|a-b| rendered via operator symbol for sign; query 50-20 -> -30 -> -\& consistent. |
| 201126 | 10 | Negative-sign f=a-b+1; query 22×13-2=284 consistent. |
| 201147 | 10 | Branch order by fewest-candidates; locks cross-checked; query |60-65|=5 consistent. |
| 201162 | 10 | Deterministic; query 21×37=777 -> {{{ consistent. |
| 201221 | 10 | Clean rightward solve; query |60-65|=5 consistent. |
| 201222 | 10 | §1 declared illegal (rhs ends in '-'); §2 leftward-fully flips sign correctly; query 82×26-2=2130 reversed to 0312 consistent. |
| 201225 | 10 | §1 exhausted (0/13) -> §2 leftward-digit-only; query 95×52+2=4942 reversed to 2494 consistent. |
| 201231 | 10 | §1 illegal -> §2 leftward-fully; g=a-b+2 sign rendered; query 69-88+2=-17 reversed to 71- -> &{- consistent. |
| 201243 | 10 | §1 exhausted -> §2; many faithful sub-variant dead-ends; query 41+46-2=85 reversed to 58 consistent. |
| 201254 | 10 | §1 exhausted -> §2; f locked a×b+1; query 25×98+1=2451 reversed to 1542 consistent. |
| 201255 | 10 | §1 illegal -> §2 leftward-fully; query 95-81-1=13 reversed to 31 consistent. |
| 201261 | 10 | §1 exhausted -> §2; query 32+60-1=91 reversed to 19 consistent. |
| 201303 | 10 | §1 exhausted -> §2; -|a-b| matched at EX4; query 44+83-2=125 reversed to 521 consistent. |
| 201308 | 10 | §1 exhausted -> §2; long DFS, f=a-b verified; query 42+77+1=120 reversed to 021 consistent. |
| 201315 | 10 | §1 exhausted -> §2; query 53×40+2=2122 reversed to 2212 consistent. |
| 201324 | 10 | §1 exhausted -> §2; negative-sign h; query 35-67=-32 reversed to -23 -> -?@ consistent. |
| 201330 | 10 | §1 exhausted -> §2; query 62×78-2=4834 reversed to 4384 consistent. |
| 201345 | 10 | §1 exhausted -> §2; query |93-86|=7 single-digit consistent. |
| 201374 | 10 | §1 exhausted -> §2; query 27×53+1=1432 reversed to 2341 consistent. |
| 201382 | 10 | §1 exhausted -> §2; query 36+94-1=129 reversed to 921 consistent. |
| 201389 | 10 | §1 exhausted -> §2; I=3 D-branches faithfully fail then I=5 solves; query 44-89-2=-47 reversed to -74 -> %/\ consistent. |
| 201390 | 10 | §1 exhausted -> §2; query 31+46=77 single-pair consistent. |
| 201408 | 10 | §1 exhausted -> §2; locks cross-checked; query 97+18=115 reversed to 511 consistent. |
| 201412 | 10 | §1 exhausted -> §2; query 59+25+2=86 reversed to 68 consistent. |
| 201432 | 10 | §1 exhausted -> §2; all three -2 operators locked; query 94+82-2=174 reversed to 471 consistent. |
| 201445 | 10 | §1 tries both f-families then exhausts -> §2; query |19-49|=30 reversed to 03 consistent. |
| 201448 | 10 | §1 exhausted -> §2; long DFS, f=a×b verified; query 18-28+1=-9 single-digit -> /} consistent. |
| 201468 | 10 | §1 exhausted -> §2; query 16+39+2=57 reversed to 75 consistent. |
| 201476 | 10 | §1 exhausted -> §2; query 20+33+1=54 reversed to 45 consistent. |
| 201477 | 10 | §1 exhausted -> §2; query |88-20|=68 reversed to 86 consistent. |
| 201497 | 10 | §1 exhausted -> §2; negative-sign f=a-b-1; query 47-62-1=-16 reversed to -61 -> -)> consistent. |
| 201499 | 10 | §1 illegal -> §2 leftward-fully; query 10+20+1=31 reversed to 13 consistent. |
| 201504 | 10 | §1 exhausted -> §2; query 59×36+2=2126 reversed to 6212 consistent. |
| 201535 | 10 | §1 exhausted -> §2; long faithful DFS; query |85-20|=65 reversed to 56 consistent. |
| 201561 | 10 | §1 exhausted -> §2; query 56+50-1=105 reversed to 501 consistent. |
| 201569 | 10 | §1 exhausted -> §2; query 52×60+1=3121 reversed to 1213 consistent. |
| 201572 | 10 | §1 exhausted -> §2; query 71-26-2=43 reversed to 34 consistent. |
| 201575 | 10 | §1 tries both g-families then exhausts A=1..9 -> §2; query 23+44+2=69 reversed to 96 consistent. |
| 201587 | 10 | §1 illegal -> §2 leftward-fully; many faithful sub-variant dead-ends; query 37-47-2=-12 reversed to 21- -> @:- consistent. |
| 201599 | 10 | §1 exhausted -> §2; locks cross-checked; query 58×89-1=5161 reversed to 1615 consistent. |
| 201605 | 10 | §1 exhausted -> §2; query |48-52|=4 single-digit consistent. |
| 201607 | 10 | §1 exhausted -> §2; query 21+63+1=85 reversed to 58 consistent. |
| 201621 | 10 | §1 exhausted -> §2; long DFS, f=|a-b| found last; query 84+30-2=112 reversed to 211 consistent. |
| 201623 | 10 | §1 tries both f/h-families then exhausts A/B -> §2; query 26+22-2=46 reversed to 64 consistent. |
| 201629 | 10 | §1 exhausted -> §2; negative-sign f=a-b-2; query 45-93-2=-50 reversed to -05 -> |?\ consistent. |
| 201630 | 10 | §1 illegal -> §2 leftward-fully; query 96-29+2=69 reversed to 96 consistent. |
| 201631 | 10 | §1 exhausted -> §2; query |15-50|=35 reversed to 53 consistent. |
| 201636 | 10 | §1 illegal -> §2 leftward-fully; query 29×26=754 reversed to 457 consistent. |
| 201637 | 10 | §1 exhausted -> §2; locks cross-checked; query 18×64+2=1154 reversed to 4511 consistent. |
| 201644 | 10 | §1 illegal -> §2 leftward-fully; query 48+99+1=148 reversed to 841 consistent. |
| 201655 | 9 | §1 illegal -> §2 leftward-fully, sound; but final operand E is undetermined and guessed by a stated "smallest E=5" rule ("E is not pinned by any example; go with the smallest E=5") and the query answer depends on E (operand 15). Principled, not arbitrary, so minor only. |
