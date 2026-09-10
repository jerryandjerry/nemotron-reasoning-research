| id | score | reasons |
|----|-------|---------|
| 100002 | 10 | Deterministic throughout; operator pruning, domain narrowing, branching by fewest-candidates, locks verified across equations. |
| 100018 | 10 | f fixed as ~sub by output sign; all deductions follow; reaches boxed answer cleanly. |
| 100024 | 10 | Tries |a-b| then a-b for f, rejects/accepts by matching rhs; forward-only. |
| 100025 | 10 | Two-operator case; digit-count filter forces mul/add uniquely; arithmetic verified. |
| 100027 | 10 | Clean elimination; |a-b| locked after confirmation; correct query. |
| 100038 | 10 | Long backtracking but every branch follows stated rules; locks confirmed. |
| 100046 | 10 | Rejects g=a-b ("equations don't all check") before accepting -|a-b|; rigorous. |
| 100064 | 10 | g=~sub fails on one branch and backtracks; final |a-b| confirmed across EX. |
| 100089 | 10 | g=~sub rejected on A=6/A=7 branches; a-b+1 found and verified. |
| 100094 | 10 | Deterministic; concat check, digit-count filter, locks confirmed. |
| 100121 | 10 | Very deep backtracking over add-noise variants; each rejection justified; consistent. |
| 100129 | 10 | mul/add forced by digit counts; arithmetic checks out. |
| 100138 | 10 | f=~sub by sign; mul/add/sub locked and confirmed; correct. |
| 100144 | 10 | h=mul locked via EX5 13×13; subsequent locks verified. |
| 100149 | 10 | Extensive search; every f/g failure has explicit no-variant justification. |
| 100166 | 10 | Long mul search; f tries |a-b| (fails EX4) then a-b confirmed; consistent. |
| 100167 | 10 | All operators forced; arithmetic and cross-checks correct. |
| 100170 | 10 | f=~sub repeatedly rejected until |a-b| matches all; forward-only. |
| 100172 | 10 | F forced to 9 by EX2; |a-b|/mul/add locked and confirmed; correct. |
| 100182 | 10 | -|a-b| accepted after rejecting |a-b| and a-b; rigorous. |
| 100188 | 10 | mul/add only survivors; multiple add-variant trials each justified. |
| 100191 | 10 | Deep mul backtracking; every g=mul beyond-tolerance rejection shown. |
| 100192 | 10 | a×b+2 chosen after a×b trial fails cross-check; a-b confirmed. |
| 100200 | 10 | mul/sub/add forced; a×b-1 confirmed across EX3/EX4. |
| 100202 | 10 | Long add/sub/mul search; |a-b| confirmed; correct query. |
| 100209 | 10 | add/mul forced by digit counts; locks confirmed; correct. |
| 100216 | 10 | E forced to 1; all three operators locked and verified. |
| 100218 | 10 | f=~mul fails on C=3 branch, backtracks; a×b-2 confirmed. |
| 100222 | 10 | g=~sub by sign; a×b+2 confirmed across EX2/EX3; consistent. |
| 100227 | 10 | Deterministic; a+b+2 confirmed across EX2; correct. |
| 100229 | 10 | Tries |a-b| (fails), then b-a confirmed across EX2/EX4; rigorous. |
| 100232 | 10 | All operators forced; |a-b| confirmed; correct query. |
| 100253 | 10 | f=~mul on B=9/D=0 branch, a×b-1 locked; consistent. |
| 100263 | 10 | Multiple g=~sub failures justified; a-b+1 confirmed; correct. |
| 100268 | 10 | g=~mul fails E=2, succeeds E=3; a-b confirmed; consistent. |
| 100271 | 10 | f=~sub by sign; a-b confirmed across EX3; correct. |
| 100276 | 10 | f=~sub forced by 1-digit rhs; |a-b| confirmed across EX1/EX4. |
| 100280 | 10 | sub/mul/add forced; |a-b| confirmed across EX2; correct. |
| 100283 | 10 | a-b and a+b+2 locked and confirmed; correct query. |
| 100296 | 10 | Three operators locked; arithmetic verified; consistent. |
| 100315 | 10 | mul/add/abs forced; locks confirmed; correct. |
| 100317 | 10 | h=~sub by sign; a-b+2 confirmed; negative sign rendered as operator symbol consistently. |
| 100321 | 10 | |a-b|, a+b+2, a×b-1 locked and verified; correct query. |
| 100333 | 10 | a×b-1, |a-b|, a+b+2 all confirmed; consistent. |
| 100334 | 10 | Multiple g-variant trials across branches; |a-b| confirmed; correct. |
| 100354 | 10 | a-b-1, a+b-2, a×b+2 locked; arithmetic verified. |
| 100370 | 10 | a+b, a×b+2, a-b confirmed; correct query. |
| 100375 | 10 | a-b+2, a+b+1 locked; correct query. |
| 100387 | 10 | a-b-2, a+b, a×b-1 confirmed; correct. |
| 100394 | 10 | h=~sub fails twice, backtracks; |a-b| confirmed; a×b locked. |
| 100396 | 10 | f=~mul fails one branch; a×b locked, a-b+2 confirmed; correct. |
| 100397 | 10 | f=~mul fails, then a×b-1; g=b-a confirmed; correct query. |
| 100400 | 10 | h=~sub fails, |a-b| confirmed; a×b, a+b+1 locked; correct. |
| 100417 | 10 | Rightward yields 0/13, leftward(digit) solves fully; result digit-reversed correctly. |
| 100421 | 10 | Leftward; tries |a-b|,a-b before -|a-b| confirmed; result reversed correctly. |
| 100427 | 10 | Leftward chosen (rightward 0/12); |a-b| confirmed; result reversed. |
| 100447 | 10 | Leftward; a×b-1, a+b-2, |a-b| confirmed; result reversed. |
| 100468 | 10 | Leftward; a×b-1, a+b, |a-b| confirmed; query 0 handled correctly. |
| 100469 | 10 | Leftward; a×b-1, a+b-1 locked; 1471 reversed to 1741 correctly. |
| 100473 | 10 | Leftward; multiple sub/mul failures justified; a×b locked; result reversed. |
| 100482 | 10 | Leftward; a+b-1, a×b, a-b-2 confirmed; correct query. |
| 100500 | 10 | Leftward; a×b, -|a-b| confirmed; 1156 reversed to 6511 correctly. |
| 100501 | 10 | Leftward; tries a-b+2 then a-b confirmed; -10 reversed to -01 with sign-front. |
| 100518 | 10 | Leftward; a+b+1, |a-b| confirmed; result reversed. |
| 100533 | 10 | Rightward correctly ruled illegal (sign would be last); leftward-fully solves; result reversed. |
| 100540 | 10 | Rightward 0/12; leftward solves; a×b+1, a+b-1, |a-b| confirmed; result reversed. |
| 100541 | 10 | Leftward; tries f=~sub fail then a-b+1; all locked; result reversed. |
| 100543 | 10 | Leftward; a+b+2, a-b confirmed; query 0 handled. |
| 100552 | 10 | Leftward; a×b+1, a+b confirmed; 4159 reversed to 9514 correctly. |
