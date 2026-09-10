| id | score | reasons |
|---|---|---|
| crypt_aug_000013 | 10 | Deterministic DFS; concat/digit-count pruning sound, every drop justified by overlap/±2 check, locks confirmed on a second example, query arithmetic and symbol map correct. |
| crypt_aug_000019 | 10 | Forward-only; rhs=E E A forces E=1 etc., all branches pruned by stated rules, final assignment consistent, f=a+b-1 query correct. |
| crypt_aug_000023 | 10 | Each operand-domain narrowing follows from a real interval/±2 test; locks rechecked; query g(34,12)=410 mapped correctly. |
| crypt_aug_000030 | 10 | g forced ~sub by rhs sign, h locked by a∥b match; long search all justified; -67 rendered with operator symbol correctly. |
| crypt_aug_000050 | 10 | Digit-count filter leaves only ~add; lock f=a+b-1 reconfirmed on EX1/EX2; query 59+78-1=136 correct. Minor: G branch resolved by elimination not shown inline, but conclusion consistent. |
| crypt_aug_000062 | 10 | f=~sub via 1-digit rhs, g=~add via 3-digit; |a-b| lock verified; query 88+55+2=145 correct. |
| crypt_aug_000099 | 10 | Clean DFS, mul lock 16x81+1=1297 verified, g=-|a-b| confirmed on EX3; query -|27-86|=-59 correct. |
| crypt_aug_000105 | 10 | All eliminations and overlap drops justified; f=a×b-1 lock confirmed; query 74×43-1=3181 correct. |
| crypt_aug_000109 | 10 | Long but every drop has an arithmetic basis; f=a×b-1 reconfirmed via EX3; query correct. |
| crypt_aug_000124 | 10 | g=~add by distinct-operator rule, mul lock f=a×b+2; query 54×85+2=4592 correct. |
| crypt_aug_000136 | 10 | f=b∥a concat lock, g=~mul, h=~sub; final assignment consistent; query 21×27-2=565 correct. |
| crypt_aug_000150 | 10 | f=~sub by sign, branch+guess (I smallest) is a stated repeatable rule; query 65-67-2=-4 rendered with operator symbol. |
| crypt_aug_000153 | 10 | Two-family advance (~add then ~sub) deterministic; mul/|a-b| locks verified; query 92×94-2=8646 correct. |
| crypt_aug_000182 | 10 | f=a∥b, g=a×b-2, h=a+b-2 all locked and reconfirmed; query 63×47-2=2959 correct. |
| crypt_aug_000189 | 10 | Elimination-driven, h=a×b-2 (20×10-2=198), g=a+b+1; query 90+42+1=133 correct. |
| crypt_aug_000197 | 10 | g=b∥a, f=a×b+2 (87×87+2=7571), h=a-b+2; query 58×67+2=3888 correct. |
| crypt_aug_000204 | 10 | g=~sub by sign, h=a∥b; locks f=a+b+1, g=a-b-1 confirmed on EX5; query 74-95-1=-22 rendered correctly. |
| crypt_aug_000234 | 10 | f=~sub, g=~add via digit count; |a-b| lock reconfirmed on EX3/EX4; query |67-31|=36 correct. |
| crypt_aug_000237 | 10 | f=b∥a, g=~sub, h=~mul; mul lock 42×42=1764, g=|a-b|; query 12×95-1=1139 correct. |
| crypt_aug_000285 | 10 | g=~sub by sign; long DFS with every drop justified; f=a+b+1, g=a-b+2 locked; query 38+97+1=136 correct. |
| crypt_aug_000287 | 10 | f=a∥b lock, g=~add then ~mul; h=-|a-b| confirmed; query -|25-16|=-9 rendered correctly. |
| crypt_aug_000301 | 10 | f=~mul, g=~add via digit count; locks reconfirmed on EX2/EX5; query 55+25-2=78 correct. |
| crypt_aug_000302 | 10 | f=~sub, g=~mul, h=b∥a; f=a-b+2, g=a×b-2 locked and confirmed; query 90×52-2=4678 correct. |
| crypt_aug_000303 | 10 | f=a+b-1 (74+37-1=110), h=a×b (57×18=1026); query 15×54=810 correct. |
| crypt_aug_000304 | 10 | f=a×b+1 (40×52+1=2081), g=a+b; query 73+84=157 correct. |
| crypt_aug_000323 | 10 | f=~mul, g=b∥a; long search, mul lock 49×75=3675; query 11×71=781 correct. |
| crypt_aug_000338 | 10 | f=a×b+1 (40×52+1=2081), g=a+b, h=b∥a; query 23×30-2=688 correct. |
| crypt_aug_000347 | 10 | f=~mul, g=a∥b; mul lock 33×87+1=2872; query 73×13+1=950 correct. |
| crypt_aug_000388 | 10 | f=a+b, g=b∥a, h=a×b-2; h verified via the ±0/±1/±2 ladder; query 50×48-2=2398 correct. |
| crypt_aug_000398 | 10 | f=b∥a, g=~sub; g=|a-b| confirmed on EX3; query |82-32|=50 correct. |
| crypt_aug_000420 | 10 | f=b∥a, g=~add; long DFS, g=a+b-1 lock (90+42-1=131); query 26+11-1=36 correct. |
| crypt_aug_000446 | 10 | f=a×b+1, g=a+b+2, h=b∥a; both arithmetic locks reconfirmed on a second example; query 86+34+2=122 correct. |
| crypt_aug_000461 | 10 | h=~sub by sign; f=a×b+1 (65×82+1=5331), h=a-b+2; query 82×22+1=1805 correct. |
| crypt_aug_000464 | 10 | f=~sub, g=b∥a, h=~mul; f=a-b confirmed on EX2/EX5; query 43-97=-54 rendered correctly. |
| crypt_aug_000488 | 10 | f=~mul, g=b∥a, h=~add; f=a×b+2, h=a+b-1 locked; query 67+23-1=89 correct. |
| crypt_aug_000493 | 10 | f=a∥b, g=~mul, h=~sub; mul lock 25×18+1=451, h=a-b-2; query 43-88-2=-47 rendered correctly. |
| crypt_aug_000533 | 10 | f=~sub, g=b∥a; under-constrained tails resolved by stated smallest-guess rule; h=a×b ladder applied; query |11-11|=0 -> @. |
| crypt_aug_000543 | 10 | f=-|a-b| (-|85-40|=-45), g=a×b-2, h=b∥a; query -|57-41|=-16 rendered correctly. |
| crypt_aug_000545 | 10 | f=a∥b, g=a+b+2, h=a-b+2; deep search, final smallest-guess for E is a stated rule; query 10+98+2=110 correct. |
| crypt_aug_000555 | 10 | g=~sub by sign; f=a+b-2 confirmed on EX1/EX3, g=a-b-1; query 69-26-1=42 correct. |
| crypt_aug_000558 | 10 | h=~sub by sign; f=b∥a, g=a×b-1 (75×44-1=...); query 31×67-1=2076 correct. |
| crypt_aug_000571 | 10 | f=a×b+1 (60×94+1=5641) reconfirmed on EX3, h=|a-b|; query 99×63+1=6238 correct. |
| crypt_aug_000574 | 10 | g=~sub by sign; f=b∥a, g=a-b-1, h=a+b+2 confirmed on EX2; query 82-54-1=27 correct. |
| crypt_aug_000604 | 10 | f=a×b+1 (33×44+1=1453), g=a+b; query 20×47+1=941 correct. |
| crypt_aug_000650 | 10 | f=~sub by sign; minimal puzzle, f=a-b; query 11-11=0 -> <. |
| crypt_aug_000652 | 10 | f=~sub by sign; f=a-b confirmed on EX4; query 77-77=0 -> !. |
| crypt_aug_000655 | 10 | f=a+b-1 (90+15-1=104) confirmed on EX1, h=b∥a; query 32+92-1=123 correct. |
| crypt_aug_000664 | 10 | f=~sub by sign; f=b-a confirmed on EX4; query 38+58+2=98 correct. |
| crypt_aug_000669 | 10 | f=a×b-2 reconfirmed on EX3/EX4, g=b∥a; query 23×30-2=688 correct. |
| crypt_aug_000687 | 10 | f=a×b+2 (41×98+2=4020), g=a∥b; query 74×76+2=5626 correct. |
| crypt_aug_000696 | 10 | f=b∥a, g=~mul; g=a×b+2 confirmed on EX2; query 72×95+2=6842 correct. |
| crypt_aug_000739 | 10 | f=a+b+1, g=a∥b, h=|a-b|; deep DFS all justified; query |94-19|=75 correct. |
| crypt_aug_000740 | 10 | g=a×b-1 (75×58-1=4349) reconfirmed on EX4, h=a+b+1; query 21+19+1=41 correct. |
| crypt_aug_000761 | 10 | h=~sub by sign; f=b∥a, g=a×b-2, h=a-b+1; query 49-95+1=-45 rendered correctly. |
| crypt_aug_000783 | 10 | h=~sub by sign; f=a×b (40×80=3200), h=a-b; query 41×32=1312 correct. |
| crypt_aug_000803 | 10 | g=~sub by sign; f=a∥b, g=a-b-1, h=a×b; query 64×27=1728 correct. |
| crypt_aug_000814 | 10 | g=~sub by sign; f=b∥a, g=b-a confirmed on EX4; query 50-32=18 correct. |
| crypt_aug_000852 | 10 | f=a∥b, g=a×b (97×22=2134), h=a+b-2; query 73×71=5183 correct. |
| crypt_aug_000859 | 10 | f=a×b+2 (32×75+2=2402), g=b∥a; query 90×86+2=7742 correct. |
| crypt_aug_000862 | 10 | f=~sub by sign; deep DFS, empty-candidate prune (B={}) correctly detected; f=a-b-2; query 25+62+2=89 correct. |
| crypt_aug_000870 | 10 | f=~mul, g=~sub via digit counts; f=a×b+1 (17×65+1=1106), g=|a-b|; query 76×66+1=5017 correct. |
| crypt_aug_000882 | 10 | f=~sub by sign; f=-|a-b| (-|94-63|=-31) reconfirmed on EX4, g=a+b-1; query -|75-83|=-8 rendered correctly. |
| crypt_aug_000892 | 10 | f=~sub by sign; f=-|a-b|, h=a+b+1 confirmed on EX5; query 60+99+1=160 correct. |
| crypt_aug_000940 | 10 | h=~sub by sign; f=b∥a, g=a+b+1, h=a-b+2; query 99+91+1=191 correct. |
| crypt_aug_000941 | 10 | f=a×b-1 (32×73-1=2335) reconfirmed on EX1, h=a+b+2; query 72+93+2=167 correct. |
| crypt_aug_000960 | 10 | g=~sub by sign; f=b∥a, g=-|a-b| confirmed on EX4, h=a×b-2; query 97×21-2=2035 correct. |
| crypt_aug_000988 | 10 | g=~sub by sign; minimal puzzle, g=-|a-b| (-|42-35|=-7); query -|56-53|=-3 rendered correctly. |
| crypt_aug_000992 | 10 | g=~sub by sign; f=a+b (91+90=181), g=a-b-2 confirmed on EX2, h=b∥a; query 14+48=62 correct. |
| crypt_aug_001009 | 10 | f=a+b-2 (50+65-2=113), g=b∥a, h=a×b-2; query 11+11-2=20 correct. |
