# Reasoning audit — slice_8

| id | score | reasons |
| --- | --- | --- |
| 201682 | 10 | deterministic forward DFS; operators locked only after all operands known; query arithmetic/reversal/symbol-render all consistent |
| 201691 | 10 | digit-count + distinct-op pruning sound; h=a-b+2 locked & verified; negative result rendered via operator symbol consistently |
| 201731 | 10 | §1 ~sub fails by real computation; §2 exhaustive branching; g=\|a-b\| locked from EX2 and verified |
| 201743 | 10 | sound digit-count pruning and branching; f=a×b-1, g=\|a-b\|, h=a+b-2 all locked & verified |
| 201757 | 10 | §1 ~add variants exhausted by real computation; §2 locks all 3 ops with checks; 100->001 leading-zero reversal per rule |
| 201760 | 10 | §1 correctly ruled illegal (rhs ends in sign); §2 leftward-fully sound; sign rendered via operator symbol |
| 201767 | 10 | g sign-forced ~sub; -\|a-b\| chosen after a-b verify-fail (deterministic order), verified on EX2/EX4 |
| 201825 | 10 | long honest multi-branch DFS with ±2 product rejections; f/g/h verified |
| 201843 | 10 | full forward search; f=a+b+1, g=a-b+2, h=a×b locked and cross-confirmed |
| 201850 | 10 | §1 illegal; §2 locks f=a×b+1, g=a-b; 4-digit reversal correct |
| 201856 | 10 | failed ~mul branch explicitly rejected; f=a×b, g=a+b-1, h=\|a-b\| verified |
| 201873 | 10 | sound digit-count pruning; g=a+b; 120->021 leading-zero reversal per rule, consistent |
| 201879 | 10 | §1 illegal; §2 leftward-fully; two ~sub variants rejected before f=a-b lock |
| 201887 | 10 | 2-operator puzzle; deterministic H/G branching; g=a×b verified on EX4 |
| 201895 | 10 | forward branching; f=\|a-b\|, g=a+b-2, h=a×b+1 locked with checks |
| 201900 | 10 | §1 illegal; §2 locks f=a-b-1, g=a×b; negative rendered via operator symbol # (boxed '#') |
| 201905 | 10 | sound; h=\|a-b\| locked from EX4 (\|44-41\|=3); single-digit render consistent |
| 201908 | 10 | §1 illegal; §2 deterministic; f=a+b, g=a-b+2 verified |
| 201919 | 10 | long honest search; h=a×b+1 locked & confirmed on EX3/EX4 |
| 201935 | 10 | multi-family traversal handled; deterministic tie-break; query 123->321 consistent |
| 201938 | 10 | exhaustive F-branch search; g=a+b+2, f=a×b+1, h=\|a-b\| verified |
| 201972 | 10 | full DFS with ±2 product rejections; ops locked and confirmed |
| 201979 | 10 | exhaustive F then A/D/E branching; f=\|a-b\|, g=a×b+1 verified |
| 201991 | 10 | branching with real ~mul rejections; f=a×b-1, g=a-b-2, h=a+b locked |
| 201993 | 10 | long exhaustive ~sub-variant checks; f=\|a-b\|, g=a+b verified |
| 201995 | 10 | forward; f=a+b-2, g=\|a-b\|, h=a×b+2 locked; 1900->0091 reversal consistent |
| 202004 | 10 | deterministic branch to B/C/D/J/G/H; f=a+b-2, g=\|a-b\| verified |
| 202007 | 10 | §1 ~add exhausted by computation; §2 locks f=a+b-1, g=\|a-b\| |
| 202008 | 10 | honest DFS, every product computed & rejected when off by >2; 4-digit reversal correct |
| 202010 | 10 | sound; f=a×b+1, g=\|a-b\|, h=a+b-2 locked and verified |
| 202014 | 10 | ~sub variant \|a-b\| rejected then a-b accepted; rigorous; h=a+b query correct |
| 202024 | 10 | forward; f=a+b-2, g=\|a-b\|, h=a×b-1 locked with cross-checks; 20->02 reversal consistent |
| 202030 | 10 | §1 illegal; §2 locks f=a×b+2, g=-\|a-b\|, h=a+b+1 verified |
| 202045 | 10 | exhaustive E/B/D branching; f=a-b+2, g=a×b-2, h=a+b+2 verified |
| 202047 | 10 | long DFS with ±2 rejections; ~mul/~add/~sub locked & confirmed; 4-digit reversal correct |
| 202056 | 10 | §1 illegal; §2 locks g=a×b+1; 4601->1064 reversal correct |
| 202061 | 10 | forward; h=-\|a-b\| locked; negative rendered via operator symbol ` (boxed `&]) |
| 202062 | 10 | sound branching; f=a+b-2, g=a×b+1, h=\|a-b\| verified |
| 202067 | 10 | clean; f=\|a-b\|, g=a+b locked from EX1; 158->851 reversal consistent |
| 202071 | 10 | exhaustive F-branch; h=a-b+2; negative -60 sign stays front (digit-only), operator '-' consistent |
| 202094 | 10 | forward G/A/E branching; f=a+b, g=a×b locked with checks |
| 202101 | 10 | multi-family (~add→~sub); long honest DFS; f=\|a-b\|, g=a×b-2; 4-digit reversal correct |
| 202104 | 10 | sound; f=\|a-b\|, g=a+b-2 locked and verified on EX2/EX3 |
| 202107 | 10 | forward; f=a+b+2, g=a-b+2 locked from pinned operands |
| 202115 | 10 | long exhaustive DFS with ±2 rejections; h=a-b query correct |
| 202135 | 10 | §1 illegal; huge honest search over H/E/B/F; f=a-b+2, g=a+b-1, h=a×b-1 |
| 202151 | 10 | g=\|a-b\| locked; palindrome 77 render consistent |
| 202156 | 10 | forward; f=a×b+2, g=a+b+2, h=\|a-b\| locked with checks |
| 202188 | 10 | §1 illegal; §2 exhaustive; f=a+b+2, g=a×b, h=a-b-2 verified; 4-digit reversal correct |
| 202195 | 10 | forward; g=a×b locked; 3456->6543 reversal correct |
| 202197 | 10 | §1 illegal; §2 long DFS over C/I/A/E/B/F; g=a-b-2 verified |
| 202206 | 10 | forward; g=a+b-1 locked; palindrome 77 consistent |
| 202208 | 10 | §1 ~sub fails by computation; §2 g=b-a chosen after \|a-b\| verify-fail (frequency order), f=a+b verified |
| 202211 | 10 | ~sub variants tried in order; f=a-b+2; negative rendered via operator '-', consistent |
| 202215 | 10 | exhaustive E/F/B/C/D search; f=a+b, g=a-b, h=a×b+2 verified |
| 202222 | 10 | forward; ~mul variant rejected; h=a+b+2; 80->08 reversal consistent |
| 202226 | 10 | §1 illegal; long faithful DFS with ±2 rejections; f=a×b-1, g=-\|a-b\| |
| 202241 | 10 | §1 illegal; honest DFS; h=a+b-1, g=-\|a-b\| verified |
| 202250 | 10 | forward branching; f=\|a-b\|, g=a+b-1 locked with checks |
| 202286 | 10 | sound; f=\|a-b\|, g=a+b-1, h=a×b verified on EX2 |
| 202332 | 10 | huge exhaustive search; multiple real ±2 product rejections; 9 symbols fully resolved |
| 202338 | 10 | forward; f=a×b-2, g=a-b+1, h=a+b-2 locked and verified |
| 202339 | 10 | h variants (a+b±) tried in order; ±2 product rejections; 107->701 reversal correct |
| 202347 | 10 | clean forward A-branch; f=a×b-1, g=a+b locked & confirmed; 9 symbols resolved |
| 202350 | 10 | §1 illegal; §2 locks f=a-b, g=a+b+2, h=a×b-2; negative leftward-fully sign flips to back, consistent |
| 202359 | 10 | exhaustive C/B/G/D branching; f=a×b-2, g=a+b; 60->06 reversal consistent |
| 202364 | 10 | §1 illegal; ~sub \|a-b\| rejected then a-b accepted; h=a-b query correct |
| 202377 | 10 | forward G/A branching; f=a+b-2, g=a×b+2, h=a-b; 650->056 reversal consistent |
| 202379 | 10 | full ~sub-variant ladder; f=b-a after \|a-b\| verify-fail (deterministic), g=a×b; 5820->0285 reversal consistent |
