| id | score | reasons |
|---|---|---|
| 002277 | 10 | clean §1→§2 DFS; leftward f=a+b, g=b∥a cross-checked; query 127→721→%#$ follows |
| 002303 | 10 | C=0 by elimination, ~mul digit-count prune, full DFS; query 7239→9327 consistent |
| 002315 | 10 | h=~sub from rhs sign, b-a derived with proper |a-b|→b-a backtrack; query 0→[ |
| 002316 | 10 | forward DFS, f=a+b-2/g=a×b-1/h=b∥a all cross-checked; query 150→051→['/ |
| 002332 | 10 | rightward illegal (rhs ends in sign) handled; f=b-a via backtrack; query 0→[ |
| 002352 | 10 | h=~sub, full forward DFS, all 12 resolved; query 180→081→{)[ |
| 002362 | 10 | long but forward; f=a×b+1/g=b∥a/h=a+b-2; query 3079→9703→^"?> |
| 002439 | 9 | rightward illegal handled; f locked from single EX1 (only appears once) but branch fully resolves; query 150→051→#>` |
| 002452 | 10 | f=a∥b, g=a+b cross-checked EX3/EX4; query 105→501→?@{ |
| 002493 | 10 | g=~sub, f=a∥b, rightward solves g=a-b+1; query -10→)]/ with sign as operator symbol |
| 002497 | 10 | F=0 elimination, f=a+b-1/g=a-b-2/h=a∥b full DFS; query 143→341→${: |
| 002518 | 10 | f=a+b-2/g=b∥a/h=a×b+2 forward; query 3018→8103→#!?@ |
| 002543 | 10 | rightward illegal handled; f=-|a-b| via backtrack; query 138→831→\]< |
| 002581 | 10 | proper f=a×b→a×b-1 backtrack; all pinned; query 56→65→#@ |
| 002591 | 10 | extensive correct backtracking; f=|a-b|/g=a+b+2/h=b∥a; query 35→53→>& |
| 002593 | 10 | f=~sub from rhs; f=a-b-2/g=a×b-1/h=a∥b; query 479→974→[:^ |
| 002594 | 9 | line 179 guesses smallest H=4 but H not in query (answer-irrelevant), rule-based; query 106→601→@`! |
| 002620 | 10 | f=a+b+2/g=a×b/h=b∥a cross-checked; query 8820→0288→><// |
| 002622 | 5 | C,G,H unconstrained (only in concat-identity eqs); H IS used in query g(48,87) yet picked H=8 by silent branch order (lines 177-180), not a stated rule → answer rests on arbitrary choice |
| 002628 | 9 | rightward illegal handled; line 153 smallest G=4 guess but G answer-irrelevant; query 813→318→`$& |
| 002639 | 9 | line 100 smallest I=8 guess but I not in query (answer-irrelevant); f=a+b+1 from single EX1; query 57→@{ |
| 002689 | 5 | E,F only in concat-identity eqs (unconstrained); E IS used in query f(56,46) yet E=5 picked first by branch order (line 135) with E=7 never refuted → answer hinges on undetermined variable |
| 002726 | 10 | f=a×b+2/g=a+b-2/h=a∥b cross-checked; query 65→56→\< |
| 002760 | 10 | f=b∥a, g=a+b+2 forward DFS; query 93→39→>% |
| 002761 | 10 | f=a+b-2/g=a×b+1/h=b∥a long forward DFS; query 545→545→"'" (J=5→",D=4→') |
| 002782 | 10 | g in rhs→~sub absent here; f=a+b-2 etc; query 545 palindrome→"'" |
| 002795 | 10 | rightward illegal handled; exhaustive ~sub DFS, A=7 by elimination; query -4→4-→`- |
| 002849 | 10 | f=a+b-2/g=a×b-2/h=b∥a cross-checked; query 1258→8521→#${\ |
| 002855 | 10 | rightward illegal handled; B,D pinned by EX1 given A; query 36→63→|( |
| 002875 | 10 | f=a×b+2/g=b∥a/h=|a-b|; query 12→21→># |
| 002879 | 9 | rightward illegal; line 105 smallest C=8 guess but C answer-irrelevant; query -5→(< |
| 002910 | 10 | f=a+b-2/g=a×b-2/h=b∥a cross-checked; query 102→201→#$| |
| 002920 | 10 | f=b∥a/g=a+b+2/h=a×b-2 cross-checked; query 87→78→@{ |
| 002922 | 10 | f=b∥a/g=a+b-1/h=|a-b| cross-checked; query 25→52→:) |
| 002925 | 10 | g in rhs→~sub; f=a+b-1/g=a-b+1/h=b∥a with g-variant backtrack; query 62→26→)# |
| 002930 | 10 | rightward illegal; long exhaustive but forward DFS; query 640→046→<}[ |
| 002953 | 10 | f in rhs→~sub; f=a-b-1/g=b∥a/h=a+b-2 with backtrack; query 8→8→] |
| 002970 | 10 | f=a∥b/g=a×b+2/h=a+b-2 cross-checked; query 4878→8784→{<{/ |
| 002980 | 9 | f in rhs→~sub; line 136 smallest I=4 guess but I answer-irrelevant; query -5→(< |
| 003004 | 10 | h in rhs→~sub; f=a+b+1/g=a∥b/h=a-b with backtrack; query 122→221→??< |
| 003013 | 10 | f=a+b/g=a×b-1/h=a∥b cross-checked; query 58→85→]@ |
| 003036 | 10 | g in rhs→~sub; f=a∥b/g=a-b+2/h=a+b+2; query 74→47→%! |
| 003066 | 10 | f=a×b+2/g=b∥a/h=a+b+2 cross-checked; query 91→19→/^ |
| 003102 | 10 | g in rhs→~sub; f=a×b+1/g=a-b+2/h=a∥b with f-variant backtrack; query 60→06→() |
| 003109 | 10 | f=a×b-2/g=a+b+2/h=b∥a with f-variant backtrack; query 158→851→?`% |
| 003144 | 10 | f in rhs→~sub; f=a-b+1/g=a∥b/h=a×b+2; query 2→2→\ |
| 003164 | 10 | f=a∥b/g=a+b+2/h=a×b-1 with h-variant backtrack; query 101 palindrome→])] |
| 003193 | 10 | f=a×b/g=a∥b forward DFS; query 1242→2421→{/{] |
| 003201 | 7 | query op absent from examples; concat guessed by no-4-digit-rhs heuristic (rule #6 fallback) — answer is an unverifiable guess on load-bearing operator; cipher not solved |
| 003205 | 7 | query op absent; concat heuristic guess, answer rests on guess |
| 003214 | 7 | query op absent; concat heuristic guess, answer rests on guess |
| 003216 | 7 | query op absent; concat heuristic guess, answer rests on guess |
| 003223 | 7 | query op absent; concat heuristic guess, answer rests on guess |
| 003227 | 7 | query op absent; concat heuristic guess, answer rests on guess |
| 003229 | 7 | query op absent; concat heuristic guess, answer rests on guess |
| 003234 | 7 | query op absent; concat heuristic guess, answer rests on guess |
| 003252 | 7 | query op absent; concat heuristic guess, answer rests on guess |
| 003254 | 7 | query op absent; concat heuristic guess, answer rests on guess |
| 003260 | 7 | query op absent; concat heuristic guess, answer rests on guess |
| 003262 | 7 | query op absent; concat heuristic guess, answer rests on guess |
| 003265 | 7 | query op absent; concat heuristic guess, answer rests on guess |
| 003266 | 7 | query op absent; concat heuristic guess, answer rests on guess |
| 003272 | 7 | query op absent; concat heuristic guess, answer rests on guess |
| 003274 | 7 | query op absent; concat heuristic guess, answer rests on guess |
| 003278 | 7 | query op absent; concat heuristic guess, answer rests on guess |
| 003287 | 7 | query op absent; concat heuristic guess, answer rests on guess |
| 003290 | 7 | query op absent; concat heuristic guess, answer rests on guess |
| 003291 | 7 | query op absent; concat heuristic guess, answer rests on guess |
| 003301 | 7 | query op absent; concat heuristic guess, answer rests on guess |
| 003310 | 7 | query op absent; concat heuristic guess, answer rests on guess |
