| id | score | reasons |
|---|---|---|
| crypt10k_001953 | 9 | Deterministic; operators forced by digit-count, full cipher resolved, query (f=a+b) derives from locked values. Minor: §1.5 C=6 branch line 104 locks f=a+b+2 inside a branch later dropped (EX4 fails line 107); harmless since the surviving C=7 branch re-derives f=a+b. |
| crypt10k_001997 | 10 | Fully deterministic; sign-rule + distinct-operator forces f=~add/g=~mul, complete cipher, query checks. |
| crypt10k_001998 | 9 | Deterministic exhaustive branch search; f=\|a-b\| forced by EX1 exact match, query uses only f/B/H/D which are forced in the found solution. Branches B=2..4 properly rejected. |
| crypt10k_002002 | 10 | Operators all forced by digit-count, full resolution, query g=a×b derives from locked cipher. |
| crypt10k_002015 | 10 | Sign-rule + digit-count force all 3 operators; thorough branch exhaustion; query h=a+b-2 forced. |
| crypt10k_002039 | 10 | Deterministic; all operators forced, query f=a+b+2 from locked values. |
| crypt10k_002044 | 10 | Exhaustive search with proper rejection; query g=a×b+1 derives from full cipher. |
| crypt10k_002052 | 10 | Sign-rule forces g=~sub; ~sub variant disambiguation finds f=-\|a-b\|; query consistent. |
| crypt10k_002063 | 10 | Deterministic; f=a-b-1 locked; negative query rendered with operator-symbol sign per rule 4; consistent. |
| crypt10k_002065 | 10 | All operators forced (1-digit→sub, 4-digit→mul); full resolution; query g=a×b+2 forced. |
| crypt10k_002096 | 10 | Deterministic; f=\|a-b\| forced by EX3 match; query f(78,79)=1 consistent. |
| crypt10k_002100 | 10 | Sign-rule forces f=~sub; complete cipher; query f=a-b-2 forced. |
| crypt10k_002111 | 10 | Sign-rule + digit-count force all operators; full resolution; query g=a×b-2 forced. |
| crypt10k_002128 | 10 | Sign-rule forces f=~sub; f=-\|a-b\| disambiguated; negative query rendered correctly. |
| crypt10k_002162 | 10 | Operators forced; thorough; query f=a×b+2 from locked cipher. |
| crypt10k_002161 | 10 | Proper operator-family backtracking (f=~add fails→f=~mul); query f=a×b-2 forced. |
| crypt10k_002186 | 10 | Deterministic exhaustive search; query f=a×b-2 from full cipher. |
| crypt10k_002189 | 10 | All operators forced; query f=a×b-2 consistent. |
| crypt10k_002197 | 10 | Very extensive nested branching all properly exhausted; query g=\|a-b\| forced. |
| crypt10k_002204 | 10 | Sub/mul forced by digit-count; ~sub variants rejected properly; query f=\|a-b\| forced. |
| crypt10k_002206 | 10 | Sign-rule forces g=~sub; extensive ~sub variant disambiguation; query h=a×b+1 forced. |
| crypt10k_002210 | 10 | All operators forced; query h=a+b+2 from locked cipher. |
| crypt10k_002211 | 10 | Sign-rule + digit-count; Elimination rule used legitimately; query f=a+b-1 forced. |
| crypt10k_002220 | 10 | Operator-family backtracking (f=~add fails→f=~sub); query g=a+b-2 forced. |
| crypt10k_002236 | 10 | Sign-rule forces f=~sub; f=-\|a-b\| disambiguated; query h=a×b-1 forced. |
| crypt10k_002243 | 10 | Sign-rule forces h=~sub; thorough ~sub rejection; query g=a+b-2 forced. |
| crypt10k_002245 | 10 | Sign-rule forces g=~sub; extensive ~sub variant checks; negative query rendered correctly. |
| crypt10k_002247 | 10 | All operators forced; query f=a+b-1 from full cipher. |
| crypt10k_002249 | 10 | g=~add variants (a+b, a+b+1, a+b-1) properly tried/rejected; query g=a+b-1 forced. |
| crypt10k_002272 | 10 | Sign-rule forces g=~sub; thorough; query h=a+b from locked cipher. |
| crypt10k_002290 | 10 | Operator-family backtracking; Elimination rule legitimate; query h=a×b forced. |
| crypt10k_002309 | 10 | Two operator-family combos tried; multiple A/B values exhausted; query f=a×b-2 forced. |
| crypt10k_002332 | 10 | f=~mul rejected for A=3,4 then A=5 solution; query f=a×b-2 forced. |
| crypt10k_002342 | 10 | Very extensive search; g=a-b confirmed across EX2/EX3/EX5; query f=a+b-1 forced. |
| crypt10k_002377 | 10 | Operators forced; query g=\|a-b\| from full cipher. |
| crypt10k_002384 | 10 | Sign-rule forces g=~sub; multiple branches exhausted; query f=a+b-1 forced. |
| crypt10k_002387 | 10 | f=a+b/g=\|a-b\| both confirmed; query f=a+b forced. |
| crypt10k_002418 | 10 | Sign-rule forces g=~sub; f=a×b confirmed across EX1/EX2; query f=a×b forced. |
| crypt10k_002451 | 10 | Sign-rule forces g=~sub; ~mul variants disambiguated; query g=a-b+1 forced. |
| crypt10k_002469 | 10 | Sign-rule forces g=~sub; extensive search; query g=\|a-b\| forced. |
| crypt10k_002480 | 10 | All operators forced (1-digit→sub); query g=a×b from full cipher. |
| crypt10k_002519 | 10 | Sign-rule forces g=~sub; deep branching properly exhausted; query h=a+b-1 forced. |
| crypt10k_002539 | 10 | Sign-rule forces h=~sub; mul-variant rejection; query g=a+b-1 forced. |
| crypt10k_002564 | 10 | G=3,4 + B values exhausted before G=5 solution; query g=a×b forced. |
| crypt10k_002591 | 10 | Sign-rule forces h=~sub; operator-family backtracking; query f=a×b-2 forced. |
| crypt10k_002596 | 10 | Sign-rule forces g=~sub; thorough A-value exhaustion; query g=a-b-2 forced. |
| crypt10k_002636 | 10 | All operators forced; very extensive A-value search; query f=a+b from full cipher. |
| crypt10k_002642 | 10 | All operators forced (1-digit→sub); deep branching; query h=a+b-1 forced. |
| crypt10k_002648 | 10 | Operators forced; query f=a+b+1 from full cipher. |
| crypt10k_002652 | 10 | Sign-rule forces g=~sub; multi-branch search; query h=a+b-1 forced. |
| crypt10k_002677 | 10 | Sign-rule forces g=~sub; exhaustive A-value + ~sub-variant rejection; negative query rendered correctly. |
| crypt10k_002686 | 10 | Sign-rule forces g=~sub; f=a×b/g=-\|a-b\| disambiguated; query f=a×b forced. |
| crypt10k_002691 | 10 | All operators forced; query g=a×b-2 from full cipher (boxed render has leading } artifact, not a reasoning flaw). |
| crypt10k_002693 | 10 | Sign-rule forces f=~sub; f=-\|a-b\| disambiguated; query h=a×b forced. |
| crypt10k_002701 | 10 | Operators forced; uses stated smallest-guess rule for unpinned I (which is forced anyway); query g=a×b-2 forced. |
| crypt10k_002707 | 10 | All operators forced; multi-branch search; query g=a×b-2 forced. |
| crypt10k_002720 | 10 | Sign-rule forces f=~sub; extensive A/B/C/D exhaustion; query h=a+b+1 forced. |
| crypt10k_002741 | 10 | All operators forced; query h=\|a-b\| from full cipher. |
| crypt10k_002763 | 10 | Operators forced; deep B/D branching; query h=\|a-b\| forced. |
| crypt10k_002765 | 10 | Sign-rule forces h=~sub; very extensive A-value exhaustion; query h=a-b+2 forced. |
| crypt10k_002773 | 10 | Sign-rule forces f=~sub; thorough A/B/C/D search; query h=a+b+1 forced. |
| crypt10k_002775 | 10 | Operators forced; query g=\|a-b\| from full cipher. |
| crypt10k_002783 | 10 | All operators forced; multi-branch; query f=a+b-2 forced. |
| crypt10k_002910 | 10 | Sign-rule forces f=~sub; thorough; query g=a×b+2 forced. |
| crypt10k_002933 | 10 | Operators forced; deep C/B/E branching; query f=a×b forced. |
| crypt10k_002951 | 10 | All operators forced (1-digit→sub); Elimination legitimate; query f=a+b forced. |
| crypt10k_002957 | 10 | Operators forced; query f=a×b-1/g=a+b-1 confirmed; query g=a+b-1 forced. |
| crypt10k_002978 | 10 | Sign-rule forces g=~sub; very extensive A/B exhaustion; ~sub-variant disambiguation; negative query rendered correctly. |
| crypt10k_002986 | 10 | All operators forced; query g=a+b-1 from full cipher. |
| crypt10k_002988 | 10 | Operators forced; deep B/E/D branching; query g=a×b forced. |
