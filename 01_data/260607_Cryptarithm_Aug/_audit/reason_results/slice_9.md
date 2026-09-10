| id | score | reasons |
|----|-------|---------|
| 202384 | 10 | Deterministic: §1 fails, §2 leftward search resolves all 13 vars; ops locked by checking noisy variants; query g(91,23)=68→86 correct. |
| 202392 | 10 | §1 illegal (sign at end), §2 leftward fully; rhs digit-count pruning + distinct-op deduction; query h(24,94)=2255→5522 correct. |
| 202396 | 10 | Forward DFS, all 10 vars resolved; |a-b| verified against every example before lock; query f(17,13)=223→322 correct. |
| 202413 | 10 | §1 illegal, §2 resolves 11 vars; g=a-b-1 confirmed on EX4; query f(85,12)=99 correct. |
| 202424 | 10 | Long but rule-driven branch/prune; ops locked by variant check; query h(33,14)=19→91 correct. |
| 202440 | 10 | §1 (~mul) exhausted then fails, §2 resolves 12; locks a×b/+1/|a-b| via examples; query f(50,11)=550→055 correct. |
| 202457 | 10 | g pinned ~sub by rhs sign; §2 resolves 12; h=a-b+2 from EX4; query h(23,66)=-41→-14, sign mapped to operator symbol. |
| 202470 | 10 | §2 resolves 13; f/g/h all confirmed; query h(24,20)=45→54 correct. |
| 202514 | 10 | Deep branch search, 13 vars; |a-b|,a+b+1,a×b-2 locked + cross-checked; query f(60,96)=36→63 correct. |
| 202560 | 10 | §1 illegal, §2 fully; |a-b| rejected globally then a-b accepted (forward); query f(74,69)=5 correct. |
| 202579 | 10 | h ~sub by sign; §2 resolves 12; -|a-b| chosen after |a-b|/a-b fail; query h(84,29)=-55. |
| 202585 | 10 | h ~sub; §1 hits all-different contradiction (correctly voids branch), §2 resolves 12; query h(25,92)=-67→-76. |
| 202597 | 10 | §1 G=0 elimination then all-diff fail, §2 resolves 13; a+b/|a-b|/a×b+1 locked; query g(17,83)=66 correct. |
| 202647 | 10 | F=0 elimination; §2 resolves 13; ops locked from EX2/EX1/EX3; query f(48,70)=119→911 correct. |
| 202648 | 10 | g ~sub; §1 exhausts ~mul and fails, §2 resolves 10; a-b accepted after |a-b| fails; query g(63,93)=-30→-03. |
| 202669 | 10 | g ~sub; §1 exhausts ~add variants and fails, §2 resolves 11; query f(56,62)=6 correct. |
| 202686 | 10 | §2 leftward resolves 11; f=a+b-2,g=a-b locked + EX4 cross-check; query f(24,86)=108→801 correct. |
| 202688 | 10 | §2; b-a accepted after |a-b| global fail (forward); EX5 cross-check; query g(60,64)=4 correct. |
| 202753 | 10 | Concat: g=a∥b verified on EX2 rhs, query short-circuits to GHHF. Sound. |
| 202770 | 10 | Concat: g=b∥a verified on EX2; h left unknown (not query op), valid; query HDCH correct. |
| 202798 | 10 | Concat: f=b∥a verified on EX1; query IDBD correct. |
| 202818 | 10 | Concat: f=a∥b verified on EX1; query HIGF correct. |
| 202839 | 10 | Concat: h=b∥a verified on EX5; query CHGI correct. |
| 202853 | 10 | Concat: f=a∥b verified on EX1; query FAAB correct. |
| 202905 | 10 | Concat: h=b∥a verified on EX3; query FCFG correct. |
| 202935 | 10 | Concat: f=b∥a verified on EX1; query EICA correct. |
| 202964 | 10 | Concat: h=b∥a verified on EX3; query BIHD correct. |
| 202995 | 10 | Concat: h=a∥b verified on EX3; query IGFC correct. |
| 203022 | 10 | Concat: g=a∥b verified on EX3; query HDHF correct. |
| 203058 | 10 | Concat: f=b∥a verified on EX1; query GJFG correct. |
| 203097 | 10 | Concat: g=b∥a verified on EX2; query GDCF correct. |
| 203165 | 10 | Concat: h=b∥a verified on EX3; query CFFC correct. |
| 203186 | 10 | Concat: g=b∥a verified on EX2; query DDFI correct. |
| 203187 | 10 | Concat: g=a∥b verified on EX2; query CFAI correct. |
| 203235 | 10 | Concat: g=a∥b verified on EX3; query FHFI correct. |
| 203245 | 10 | Concat: g=b∥a verified on EX4; query EDEH correct. |
| 203342 | 10 | Concat: g=b∥a verified on EX2; query GICB correct. |
| 203406 | 10 | Concat: f=b∥a verified on EX1; query FCHG correct. |
| 203469 | 10 | Concat: f=b∥a verified on EX1; query GEFA correct. |
| 203595 | 10 | Concat: g=b∥a verified on EX2; query AGGF correct. |
| 203598 | 10 | Concat: f=a∥b verified on EX1; query EEEF correct. |
| 203710 | 10 | Concat: h=b∥a verified on EX3; query AEHI correct. |
| 203759 | 10 | Concat: f=b∥a verified on EX1; query EIGH correct. |
| 203856 | 10 | Concat: g=b∥a verified on EX2; query ECIE correct. |
| 204022 | 10 | Concat: g=b∥a verified on EX2; query CBFB correct. |
| 204098 | 10 | Concat: f=a∥b verified on EX1; query FDGH correct. |
| 204170 | 10 | Concat: g=a∥b verified on EX3; query EFEF correct. |
| 204240 | 10 | Concat: h=a∥b verified on EX3; query DGAH correct. |
| 204857 | 10 | Concat: f=a∥b verified on EX1; query GHAB correct. |
| 205012 | 10 | Concat: g=a∥b verified on EX3; query EAED correct. |
| 300001 | 10 | Single-op ~mul; resolves 8; query needs digit 7 (unseen) → documented first-unused-ASCII guess; query-only G guessed smallest. Rule-based, valid. |
| 300002 | 10 | Single-op ~add resolved fully; query f(80,61)=141 correct. |
| 300003 | 10 | Single-op ~sub; exhaustive variant rejection; |a-b| confirmed on EX1; query f(14,14)=0 correct. |
| 300004 | 10 | Single-op ~add; a+b+2 locked; query f(99,32)=133 correct. |
| 300005 | 10 | Single-op ~sub; |a-b| confirmed on EX2; query-only F guessed smallest; query f(34,32)=2 correct. |
| 300006 | 10 | Concat: f=a∥b verified on EX1; query JEJD correct. |
| 300007 | 10 | Concat: f=a∥b verified on EX1; query HGCE correct. |
| 300008 | 10 | Single-op ~sub; exhaustive rejection, |a-b| confirmed on EX2; query-only G guessed; query f(20,37)=17 correct. |
| 300009 | 10 | Concat: g=a∥b verified on EX2; query FGFH correct. |
| 300010 | 10 | Single-op ~add; a+b locked on EX1; query-only H guessed smallest; query f(39,60)=99 correct. |
| 300011 | 10 | Concat: f=a∥b verified on EX1; query GHHA correct. |
| 320001 | 10 | Concat: f=a∥b verified on EX1; query IBCA correct. |
| 320002 | 10 | Concat: f=a∥b verified on EX1; query CECG correct. |
| 320003 | 10 | Concat: f=a∥b verified on EX1; query FFFH correct. |
| 320004 | 10 | Concat: f=a∥b verified on EX1; query GHHI correct. |
| 330001 | 10 | Concat: f=a∥b verified on EX1; query HBIA correct. |
