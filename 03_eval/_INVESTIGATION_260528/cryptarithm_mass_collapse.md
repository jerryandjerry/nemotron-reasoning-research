# Cryptarithm Mass-Collapse Investigation — `cryptn` Run (0/55 on val cryptarithm_deduce)

**Date:** 2026-05-28  
**Run analyzed:** `2605270226_newdata_cryptnumeq_outproj_950val` (cryptn)  
**Training CSV:** `01_data/260526_Cryptarithm_TrainData/260527_huikang_NumericEq.csv` (7077 rows)  
**Builder:** `01_data/260526_Cryptarithm_TrainData/_build_traincsv.py`  
**Base:** `01_data/260526_Numeric_Equation/260527_NumericEq_TrainData/260527_huikang_NumericEq_allEq_gtTrue.csv`  
**Cryptarithm CoT source:** `01_data/260525_Cryptarithm/<cat>_<id>/track/tree_cot.txt`  
**Tokenizer:** `02_train/260512_huikang_085/repo/tokenizer.json`  
**Gate:** `oversampling=0` iff `token_length >= 7680` OR `GT-match != True`

---

## T1. Token-length distribution of the 800 new cryptarithm CoTs

All 800 puzzle folders were tokenized via `Tokenizer.from_file(tokenizer.json)`. 0 puzzle id(s) did not appear in the cryptn training csv.

### `cryptarithm_deduce` (n=639, over-CAP=287, 44.9%)

| bucket (tokens) | count | cum% |
|---|---:|---:|
|      0–999    |    1 |   0.2% |
|   1000–1999   |    9 |   1.6% |
|   2000–2999   |  109 |  18.6% |
|   3000–3999   |   85 |  31.9% |
|   4000–4999   |   64 |  41.9% |
|   5000–5999   |   36 |  47.6% |
|   6000–6999   |   28 |  52.0% |
|   7000–7999   |   29 |  56.5% |
|   8000–8999   |   23 |  60.1%  **(>= CAP)** |
|   9000–9999   |   16 |  62.6%  **(>= CAP)** |
|  10000–10999  |   18 |  65.4%  **(>= CAP)** |
|  11000–11999  |   15 |  67.8%  **(>= CAP)** |
|  12000–12999  |    8 |  69.0%  **(>= CAP)** |
|  13000–13999  |    7 |  70.1%  **(>= CAP)** |
|  14000–14999  |    3 |  70.6%  **(>= CAP)** |
|  15000–15999  |    6 |  71.5%  **(>= CAP)** |
|  16000–16999  |    8 |  72.8%  **(>= CAP)** |
|  17000–17999  |    9 |  74.2%  **(>= CAP)** |
|  18000–18999  |   10 |  75.7%  **(>= CAP)** |
|  19000–19999  |   11 |  77.5%  **(>= CAP)** |
|  20000–20999  |    8 |  78.7%  **(>= CAP)** |
|  21000–21999  |    6 |  79.7%  **(>= CAP)** |
|  22000–22999  |    4 |  80.3%  **(>= CAP)** |
|  23000–23999  |    5 |  81.1%  **(>= CAP)** |
|  25000–25999  |    3 |  81.5%  **(>= CAP)** |
|  26000–26999  |    3 |  82.0%  **(>= CAP)** |
|  27000–27999  |    2 |  82.3%  **(>= CAP)** |
|  28000–28999  |    5 |  83.1%  **(>= CAP)** |
|  29000–29999  |    1 |  83.3%  **(>= CAP)** |
|  30000–30999  |    4 |  83.9%  **(>= CAP)** |
|  31000–31999  |    2 |  84.2%  **(>= CAP)** |
|  32000–32999  |    1 |  84.4%  **(>= CAP)** |
|  33000–33999  |    3 |  84.8%  **(>= CAP)** |
|  34000–34999  |    3 |  85.3%  **(>= CAP)** |
|  35000–35999  |    2 |  85.6%  **(>= CAP)** |
|  36000–36999  |    1 |  85.8%  **(>= CAP)** |
|  37000–37999  |    1 |  85.9%  **(>= CAP)** |
|  38000–38999  |    1 |  86.1%  **(>= CAP)** |
|  39000–39999  |    1 |  86.2%  **(>= CAP)** |
|  40000–40999  |    1 |  86.4%  **(>= CAP)** |
|  41000–41999  |    2 |  86.7%  **(>= CAP)** |
|  42000–42999  |    1 |  86.9%  **(>= CAP)** |
|  43000–43999  |    1 |  87.0%  **(>= CAP)** |
|  44000–44999  |    1 |  87.2%  **(>= CAP)** |
|  45000–45999  |    1 |  87.3%  **(>= CAP)** |
|  46000–46999  |    1 |  87.5%  **(>= CAP)** |
|  47000–47999  |    1 |  87.6%  **(>= CAP)** |
|  49000–49999  |    1 |  87.8%  **(>= CAP)** |
|  50000–50999  |    3 |  88.3%  **(>= CAP)** |
|  51000–51999  |    1 |  88.4%  **(>= CAP)** |
|  52000–52999  |    2 |  88.7%  **(>= CAP)** |
|  53000–53999  |    2 |  89.0%  **(>= CAP)** |
|  54000–54999  |    2 |  89.4%  **(>= CAP)** |
|  55000–55999  |    1 |  89.5%  **(>= CAP)** |
|  58000–58999  |    3 |  90.0%  **(>= CAP)** |
|  59000–59999  |    2 |  90.3%  **(>= CAP)** |
|  62000–62999  |    1 |  90.5%  **(>= CAP)** |
|  63000–63999  |    2 |  90.8%  **(>= CAP)** |
|  64000–64999  |    1 |  90.9%  **(>= CAP)** |
|  65000–65999  |    1 |  91.1%  **(>= CAP)** |
|  69000–69999  |    1 |  91.2%  **(>= CAP)** |
|  70000–70999  |    2 |  91.5%  **(>= CAP)** |
|  72000–72999  |    1 |  91.7%  **(>= CAP)** |
|  73000–73999  |    1 |  91.9%  **(>= CAP)** |
|  74000–74999  |    1 |  92.0%  **(>= CAP)** |
|  76000–76999  |    2 |  92.3%  **(>= CAP)** |
|  77000–77999  |    3 |  92.8%  **(>= CAP)** |
|  79000–79999  |    1 |  93.0%  **(>= CAP)** |
|  89000–89999  |    1 |  93.1%  **(>= CAP)** |
|  93000–93999  |    1 |  93.3%  **(>= CAP)** |
|  94000–94999  |    1 |  93.4%  **(>= CAP)** |
|  96000–96999  |    1 |  93.6%  **(>= CAP)** |
| 100000–100999 |    2 |  93.9%  **(>= CAP)** |
| 101000–101999 |    1 |  94.1%  **(>= CAP)** |
| 105000–105999 |    2 |  94.4%  **(>= CAP)** |
| 106000–106999 |    1 |  94.5%  **(>= CAP)** |
| 113000–113999 |    1 |  94.7%  **(>= CAP)** |
| 116000–116999 |    1 |  94.8%  **(>= CAP)** |
| 117000–117999 |    1 |  95.0%  **(>= CAP)** |
| 129000–129999 |    1 |  95.1%  **(>= CAP)** |
| 132000–132999 |    1 |  95.3%  **(>= CAP)** |
| 136000–136999 |    1 |  95.5%  **(>= CAP)** |
| 138000–138999 |    1 |  95.6%  **(>= CAP)** |
| 154000–154999 |    1 |  95.8%  **(>= CAP)** |
| 163000–163999 |    1 |  95.9%  **(>= CAP)** |
| 173000–173999 |    1 |  96.1%  **(>= CAP)** |
| 176000–176999 |    2 |  96.4%  **(>= CAP)** |
| 184000–184999 |    2 |  96.7%  **(>= CAP)** |
| 186000–186999 |    1 |  96.9%  **(>= CAP)** |
| 190000–190999 |    1 |  97.0%  **(>= CAP)** |
| 195000–195999 |    1 |  97.2%  **(>= CAP)** |
| 221000–221999 |    1 |  97.3%  **(>= CAP)** |
| 229000–229999 |    1 |  97.5%  **(>= CAP)** |
| 234000–234999 |    1 |  97.7%  **(>= CAP)** |
| 246000–246999 |    1 |  97.8%  **(>= CAP)** |
| 251000–251999 |    1 |  98.0%  **(>= CAP)** |
| 274000–274999 |    1 |  98.1%  **(>= CAP)** |
| 306000–306999 |    1 |  98.3%  **(>= CAP)** |
| 309000–309999 |    1 |  98.4%  **(>= CAP)** |
| 325000–325999 |    1 |  98.6%  **(>= CAP)** |
| 334000–334999 |    1 |  98.7%  **(>= CAP)** |
| 338000–338999 |    1 |  98.9%  **(>= CAP)** |
| 343000–343999 |    1 |  99.1%  **(>= CAP)** |
| 352000–352999 |    1 |  99.2%  **(>= CAP)** |
| 421000–421999 |    1 |  99.4%  **(>= CAP)** |
| 442000–442999 |    1 |  99.5%  **(>= CAP)** |
| 497000–497999 |    1 |  99.7%  **(>= CAP)** |
| 729000–729999 |    1 |  99.8%  **(>= CAP)** |
| 1347000–1347999 |    1 | 100.0%  **(>= CAP)** |

### `cryptarithm_guess` (n=161, over-CAP=96, 59.6%)

| bucket (tokens) | count | cum% |
|---|---:|---:|
|      0–999    |    1 |   0.6% |
|   1000–1999   |    3 |   2.5% |
|   2000–2999   |    7 |   6.8% |
|   3000–3999   |   21 |  19.9% |
|   4000–4999   |   14 |  28.6% |
|   5000–5999   |    7 |  32.9% |
|   6000–6999   |    8 |  37.9% |
|   7000–7999   |    9 |  43.5% |
|   8000–8999   |    4 |  46.0%  **(>= CAP)** |
|   9000–9999   |    6 |  49.7%  **(>= CAP)** |
|  10000–10999  |    3 |  51.6%  **(>= CAP)** |
|  11000–11999  |    4 |  54.0%  **(>= CAP)** |
|  12000–12999  |    2 |  55.3%  **(>= CAP)** |
|  13000–13999  |    4 |  57.8%  **(>= CAP)** |
|  14000–14999  |    2 |  59.0%  **(>= CAP)** |
|  15000–15999  |    1 |  59.6%  **(>= CAP)** |
|  16000–16999  |    2 |  60.9%  **(>= CAP)** |
|  18000–18999  |    2 |  62.1%  **(>= CAP)** |
|  19000–19999  |    2 |  63.4%  **(>= CAP)** |
|  22000–22999  |    2 |  64.6%  **(>= CAP)** |
|  23000–23999  |    3 |  66.5%  **(>= CAP)** |
|  24000–24999  |    1 |  67.1%  **(>= CAP)** |
|  25000–25999  |    1 |  67.7%  **(>= CAP)** |
|  30000–30999  |    1 |  68.3%  **(>= CAP)** |
|  31000–31999  |    1 |  68.9%  **(>= CAP)** |
|  33000–33999  |    2 |  70.2%  **(>= CAP)** |
|  35000–35999  |    1 |  70.8%  **(>= CAP)** |
|  36000–36999  |    1 |  71.4%  **(>= CAP)** |
|  37000–37999  |    1 |  72.0%  **(>= CAP)** |
|  39000–39999  |    1 |  72.7%  **(>= CAP)** |
|  41000–41999  |    1 |  73.3%  **(>= CAP)** |
|  43000–43999  |    1 |  73.9%  **(>= CAP)** |
|  48000–48999  |    2 |  75.2%  **(>= CAP)** |
|  50000–50999  |    3 |  77.0%  **(>= CAP)** |
|  51000–51999  |    1 |  77.6%  **(>= CAP)** |
|  53000–53999  |    1 |  78.3%  **(>= CAP)** |
|  54000–54999  |    1 |  78.9%  **(>= CAP)** |
|  55000–55999  |    1 |  79.5%  **(>= CAP)** |
|  57000–57999  |    1 |  80.1%  **(>= CAP)** |
|  62000–62999  |    2 |  81.4%  **(>= CAP)** |
|  65000–65999  |    2 |  82.6%  **(>= CAP)** |
|  70000–70999  |    1 |  83.2%  **(>= CAP)** |
|  72000–72999  |    1 |  83.9%  **(>= CAP)** |
|  86000–86999  |    1 |  84.5%  **(>= CAP)** |
|  96000–96999  |    1 |  85.1%  **(>= CAP)** |
| 108000–108999 |    1 |  85.7%  **(>= CAP)** |
| 109000–109999 |    1 |  86.3%  **(>= CAP)** |
| 116000–116999 |    1 |  87.0%  **(>= CAP)** |
| 121000–121999 |    1 |  87.6%  **(>= CAP)** |
| 131000–131999 |    1 |  88.2%  **(>= CAP)** |
| 143000–143999 |    1 |  88.8%  **(>= CAP)** |
| 146000–146999 |    1 |  89.4%  **(>= CAP)** |
| 185000–185999 |    1 |  90.1%  **(>= CAP)** |
| 192000–192999 |    1 |  90.7%  **(>= CAP)** |
| 197000–197999 |    1 |  91.3%  **(>= CAP)** |
| 225000–225999 |    1 |  91.9%  **(>= CAP)** |
| 257000–257999 |    1 |  92.5%  **(>= CAP)** |
| 276000–276999 |    1 |  93.2%  **(>= CAP)** |
| 323000–323999 |    1 |  93.8%  **(>= CAP)** |
| 348000–348999 |    1 |  94.4%  **(>= CAP)** |
| 400000–400999 |    1 |  95.0%  **(>= CAP)** |
| 444000–444999 |    1 |  95.7%  **(>= CAP)** |
| 731000–731999 |    1 |  96.3%  **(>= CAP)** |
| 737000–737999 |    1 |  96.9%  **(>= CAP)** |
| 806000–806999 |    1 |  97.5%  **(>= CAP)** |
| 846000–846999 |    1 |  98.1%  **(>= CAP)** |
| 912000–912999 |    1 |  98.8%  **(>= CAP)** |
| 1355000–1355999 |    1 |  99.4%  **(>= CAP)** |
| 2848000–2848999 |    1 | 100.0%  **(>= CAP)** |

**Observation:** see the long tail — many cryptarithm CoTs blow far past 7680 tokens.

## T2. GT-match on the new CoTs (boxed-answer vs `answer.txt`)

Extraction: last `\boxed{…}` (rfind end-brace), verified via `_build_traincsv.verify` (binary exact ∣ numeric ±1% ∣ string lower-case exact).

### Overall by folder-category

| folder category | n | GT-True | rate |
|---|---:|---:|---:|
| arithmetic | 315 | 200 | 63.5% |
| little_endian | 276 | 168 | 60.9% |
| mixed_concat | 108 | 23 | 21.3% |
| mixed_concat_little_endian | 26 | 7 | 26.9% |
| pure_concat | 59 | 59 | 100.0% |
| query_unseen_concat | 16 | 0 | 0.0% |

### By training category (`cryptarithm_deduce` vs `cryptarithm_guess`)

| training cat | n | GT-True | rate |
|---|---:|---:|---:|
| cryptarithm_deduce | 639 | 406 | 63.5% |
| cryptarithm_guess | 161 | 51 | 31.7% |

### By token-length bucket

| bucket | n | GT-True | rate |
|---|---:|---:|---:|
|      0–999    | 2 | 1 | 50.0% |
|   1000–1999   | 12 | 6 | 50.0% |
|   2000–2999   | 116 | 68 | 58.6% |
|   3000–3999   | 106 | 69 | 65.1% |
|   4000–4999   | 78 | 54 | 69.2% |
|   5000–5999   | 43 | 29 | 67.4% |
|   6000–6999   | 36 | 21 | 58.3% |
|   7000–7999   | 38 | 23 | 60.5% |
|   8000–8999   | 27 | 16 | 59.3%  **(>= CAP)** |
|   9000–9999   | 22 | 14 | 63.6%  **(>= CAP)** |
|  10000–10999  | 21 | 10 | 47.6%  **(>= CAP)** |
|  11000–11999  | 19 | 10 | 52.6%  **(>= CAP)** |
|  12000–12999  | 10 | 7 | 70.0%  **(>= CAP)** |
|  13000–13999  | 11 | 4 | 36.4%  **(>= CAP)** |
|  14000–14999  | 5 | 3 | 60.0%  **(>= CAP)** |
|  15000–15999  | 7 | 3 | 42.9%  **(>= CAP)** |
|  16000–16999  | 10 | 3 | 30.0%  **(>= CAP)** |
|  17000–17999  | 9 | 6 | 66.7%  **(>= CAP)** |
|  18000–18999  | 12 | 6 | 50.0%  **(>= CAP)** |
|  19000–19999  | 13 | 5 | 38.5%  **(>= CAP)** |
|  20000–20999  | 8 | 4 | 50.0%  **(>= CAP)** |
|  21000–21999  | 6 | 3 | 50.0%  **(>= CAP)** |
|  22000–22999  | 6 | 4 | 66.7%  **(>= CAP)** |
|  23000–23999  | 8 | 4 | 50.0%  **(>= CAP)** |
|  24000–24999  | 1 | 0 | 0.0%  **(>= CAP)** |
|  25000–25999  | 4 | 1 | 25.0%  **(>= CAP)** |
|  26000–26999  | 3 | 3 | 100.0%  **(>= CAP)** |
|  27000–27999  | 2 | 0 | 0.0%  **(>= CAP)** |
|  28000–28999  | 5 | 5 | 100.0%  **(>= CAP)** |
|  29000–29999  | 1 | 0 | 0.0%  **(>= CAP)** |
|  30000–30999  | 5 | 2 | 40.0%  **(>= CAP)** |
|  31000–31999  | 3 | 2 | 66.7%  **(>= CAP)** |
|  32000–32999  | 1 | 1 | 100.0%  **(>= CAP)** |
|  33000–33999  | 5 | 3 | 60.0%  **(>= CAP)** |
|  34000–34999  | 3 | 2 | 66.7%  **(>= CAP)** |
|  35000–35999  | 3 | 1 | 33.3%  **(>= CAP)** |
|  36000–36999  | 2 | 2 | 100.0%  **(>= CAP)** |
|  37000–37999  | 2 | 0 | 0.0%  **(>= CAP)** |
|  38000–38999  | 1 | 0 | 0.0%  **(>= CAP)** |
|  39000–39999  | 2 | 2 | 100.0%  **(>= CAP)** |
|  40000–40999  | 1 | 1 | 100.0%  **(>= CAP)** |
|  41000–41999  | 3 | 2 | 66.7%  **(>= CAP)** |
|  42000–42999  | 1 | 0 | 0.0%  **(>= CAP)** |
|  43000–43999  | 2 | 1 | 50.0%  **(>= CAP)** |
|  44000–44999  | 1 | 1 | 100.0%  **(>= CAP)** |
|  45000–45999  | 1 | 1 | 100.0%  **(>= CAP)** |
|  46000–46999  | 1 | 1 | 100.0%  **(>= CAP)** |
|  47000–47999  | 1 | 0 | 0.0%  **(>= CAP)** |
|  48000–48999  | 2 | 0 | 0.0%  **(>= CAP)** |
|  49000–49999  | 1 | 1 | 100.0%  **(>= CAP)** |
|  50000–50999  | 6 | 2 | 33.3%  **(>= CAP)** |
|  51000–51999  | 2 | 1 | 50.0%  **(>= CAP)** |
|  52000–52999  | 2 | 1 | 50.0%  **(>= CAP)** |
|  53000–53999  | 3 | 3 | 100.0%  **(>= CAP)** |
|  54000–54999  | 3 | 1 | 33.3%  **(>= CAP)** |
|  55000–55999  | 2 | 1 | 50.0%  **(>= CAP)** |
|  57000–57999  | 1 | 0 | 0.0%  **(>= CAP)** |
|  58000–58999  | 3 | 1 | 33.3%  **(>= CAP)** |
|  59000–59999  | 2 | 0 | 0.0%  **(>= CAP)** |
|  62000–62999  | 3 | 0 | 0.0%  **(>= CAP)** |
|  63000–63999  | 2 | 0 | 0.0%  **(>= CAP)** |
|  64000–64999  | 1 | 1 | 100.0%  **(>= CAP)** |
|  65000–65999  | 3 | 2 | 66.7%  **(>= CAP)** |
|  69000–69999  | 1 | 1 | 100.0%  **(>= CAP)** |
|  70000–70999  | 3 | 0 | 0.0%  **(>= CAP)** |
|  72000–72999  | 2 | 2 | 100.0%  **(>= CAP)** |
|  73000–73999  | 1 | 1 | 100.0%  **(>= CAP)** |
|  74000–74999  | 1 | 1 | 100.0%  **(>= CAP)** |
|  76000–76999  | 2 | 1 | 50.0%  **(>= CAP)** |
|  77000–77999  | 3 | 3 | 100.0%  **(>= CAP)** |
|  79000–79999  | 1 | 1 | 100.0%  **(>= CAP)** |
|  86000–86999  | 1 | 0 | 0.0%  **(>= CAP)** |
|  89000–89999  | 1 | 1 | 100.0%  **(>= CAP)** |
|  93000–93999  | 1 | 1 | 100.0%  **(>= CAP)** |
|  94000–94999  | 1 | 1 | 100.0%  **(>= CAP)** |
|  96000–96999  | 2 | 1 | 50.0%  **(>= CAP)** |
| 100000–100999 | 2 | 2 | 100.0%  **(>= CAP)** |
| 101000–101999 | 1 | 1 | 100.0%  **(>= CAP)** |
| 105000–105999 | 2 | 2 | 100.0%  **(>= CAP)** |
| 106000–106999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 108000–108999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 109000–109999 | 1 | 1 | 100.0%  **(>= CAP)** |
| 113000–113999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 116000–116999 | 2 | 0 | 0.0%  **(>= CAP)** |
| 117000–117999 | 1 | 1 | 100.0%  **(>= CAP)** |
| 121000–121999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 129000–129999 | 1 | 1 | 100.0%  **(>= CAP)** |
| 131000–131999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 132000–132999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 136000–136999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 138000–138999 | 1 | 1 | 100.0%  **(>= CAP)** |
| 143000–143999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 146000–146999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 154000–154999 | 1 | 1 | 100.0%  **(>= CAP)** |
| 163000–163999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 173000–173999 | 1 | 1 | 100.0%  **(>= CAP)** |
| 176000–176999 | 2 | 0 | 0.0%  **(>= CAP)** |
| 184000–184999 | 2 | 0 | 0.0%  **(>= CAP)** |
| 185000–185999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 186000–186999 | 1 | 1 | 100.0%  **(>= CAP)** |
| 190000–190999 | 1 | 1 | 100.0%  **(>= CAP)** |
| 192000–192999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 195000–195999 | 1 | 1 | 100.0%  **(>= CAP)** |
| 197000–197999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 221000–221999 | 1 | 1 | 100.0%  **(>= CAP)** |
| 225000–225999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 229000–229999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 234000–234999 | 1 | 1 | 100.0%  **(>= CAP)** |
| 246000–246999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 251000–251999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 257000–257999 | 1 | 1 | 100.0%  **(>= CAP)** |
| 274000–274999 | 1 | 1 | 100.0%  **(>= CAP)** |
| 276000–276999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 306000–306999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 309000–309999 | 1 | 1 | 100.0%  **(>= CAP)** |
| 323000–323999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 325000–325999 | 1 | 1 | 100.0%  **(>= CAP)** |
| 334000–334999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 338000–338999 | 1 | 1 | 100.0%  **(>= CAP)** |
| 343000–343999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 348000–348999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 352000–352999 | 1 | 1 | 100.0%  **(>= CAP)** |
| 400000–400999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 421000–421999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 442000–442999 | 1 | 1 | 100.0%  **(>= CAP)** |
| 444000–444999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 497000–497999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 729000–729999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 731000–731999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 737000–737999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 806000–806999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 846000–846999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 912000–912999 | 1 | 1 | 100.0%  **(>= CAP)** |
| 1347000–1347999 | 1 | 0 | 0.0%  **(>= CAP)** |
| 1355000–1355999 | 1 | 1 | 100.0%  **(>= CAP)** |
| 2848000–2848999 | 1 | 0 | 0.0%  **(>= CAP)** |

### GT-mismatch examples (first 30)

| id | folder cat | train cat | answer | predicted | tok_len |
|---|---|---|---|---|---:|
| [01ef1e3e](01_data/260525_Cryptarithm/arithmetic_01ef1e3e/track/tree_cot.txt) | arithmetic | cryptarithm_guess | `[](` | `[]'` | 11240 |
| [02902eb3](01_data/260525_Cryptarithm/arithmetic_02902eb3/track/tree_cot.txt) | arithmetic | cryptarithm_guess | `>/` | `>}` | 24617 |
| [0a94b2de](01_data/260525_Cryptarithm/arithmetic_0a94b2de/track/tree_cot.txt) | arithmetic | cryptarithm_deduce | `@}`@` | `@}!@` | 2771 |
| [0c0c6320](01_data/260525_Cryptarithm/arithmetic_0c0c6320/track/tree_cot.txt) | arithmetic | cryptarithm_guess | `":!|` | `":!{` | 23625 |
| [0c30f561](01_data/260525_Cryptarithm/arithmetic_0c30f561/track/tree_cot.txt) | arithmetic | cryptarithm_guess | `-]` | `]` | 4395 |
| [18bce168](01_data/260525_Cryptarithm/arithmetic_18bce168/track/tree_cot.txt) | arithmetic | cryptarithm_guess | `@&`{` | `@&)<` | 14493 |
| [1a493b13](01_data/260525_Cryptarithm/arithmetic_1a493b13/track/tree_cot.txt) | arithmetic | cryptarithm_guess | `-]` | `]` | 3748 |
| [212e792d](01_data/260525_Cryptarithm/arithmetic_212e792d/track/tree_cot.txt) | arithmetic | cryptarithm_deduce | `)//` | `)%` | 2073 |
| [21ee162c](01_data/260525_Cryptarithm/arithmetic_21ee162c/track/tree_cot.txt) | arithmetic | cryptarithm_guess | `'&{` | `'&$` | 6334 |
| [223b2899](01_data/260525_Cryptarithm/arithmetic_223b2899/track/tree_cot.txt) | arithmetic | cryptarithm_deduce | `<[{?` | `<[#?` | 11588 |
| [236a2204](01_data/260525_Cryptarithm/arithmetic_236a2204/track/tree_cot.txt) | arithmetic | cryptarithm_deduce | ``>` | `{{])` | 63680 |
| [27fdca03](01_data/260525_Cryptarithm/arithmetic_27fdca03/track/tree_cot.txt) | arithmetic | cryptarithm_guess | `@!` | `@)` | 22772 |
| [294453b5](01_data/260525_Cryptarithm/arithmetic_294453b5/track/tree_cot.txt) | arithmetic | cryptarithm_deduce | `&(` | `##!)` | 55641 |
| [2d624cab](01_data/260525_Cryptarithm/arithmetic_2d624cab/track/tree_cot.txt) | arithmetic | cryptarithm_guess | `![)]` | `{!["` | 348572 |
| [2db53c85](01_data/260525_Cryptarithm/arithmetic_2db53c85/track/tree_cot.txt) | arithmetic | cryptarithm_guess | `)//:` | `){!'` | 131596 |
| [2e9b1b9d](01_data/260525_Cryptarithm/arithmetic_2e9b1b9d/track/tree_cot.txt) | arithmetic | cryptarithm_deduce | `<)` | `#)` | 5267 |
| [305a3436](01_data/260525_Cryptarithm/arithmetic_305a3436/track/tree_cot.txt) | arithmetic | cryptarithm_guess | `&{'` | `&@"` | 4987 |
| [340321c8](01_data/260525_Cryptarithm/arithmetic_340321c8/track/tree_cot.txt) | arithmetic | cryptarithm_deduce | `-!:` | `$:` | 23178 |
| [34d1d16f](01_data/260525_Cryptarithm/arithmetic_34d1d16f/track/tree_cot.txt) | arithmetic | cryptarithm_guess | `^|!` | `/!` | 6387 |
| [3559dfc9](01_data/260525_Cryptarithm/arithmetic_3559dfc9/track/tree_cot.txt) | arithmetic | cryptarithm_deduce | `[%[` | `[![` | 12960 |
| [36b5df1b](01_data/260525_Cryptarithm/arithmetic_36b5df1b/track/tree_cot.txt) | arithmetic | cryptarithm_guess | `%\/` | `\/` | 4631 |
| [3a6286e9](01_data/260525_Cryptarithm/arithmetic_3a6286e9/track/tree_cot.txt) | arithmetic | cryptarithm_deduce | `!{!$` | `!|||` | 229092 |
| [3b7148f6](01_data/260525_Cryptarithm/arithmetic_3b7148f6/track/tree_cot.txt) | arithmetic | cryptarithm_guess | `&&$` | `?}>` | 30067 |
| [3b97e6f6](01_data/260525_Cryptarithm/arithmetic_3b97e6f6/track/tree_cot.txt) | arithmetic | cryptarithm_deduce | `<` | `|>` | 3964 |
| [420d5352](01_data/260525_Cryptarithm/arithmetic_420d5352/track/tree_cot.txt) | arithmetic | cryptarithm_deduce | `&?` | `!?` | 4422 |
| [4a02017f](01_data/260525_Cryptarithm/arithmetic_4a02017f/track/tree_cot.txt) | arithmetic | cryptarithm_guess | `'&}` | `@!#` | 25038 |
| [4bfd095a](01_data/260525_Cryptarithm/arithmetic_4bfd095a/track/tree_cot.txt) | arithmetic | cryptarithm_deduce | `{<<<` | `{!!!` | 2253 |
| [4d04ce0c](01_data/260525_Cryptarithm/arithmetic_4d04ce0c/track/tree_cot.txt) | arithmetic | cryptarithm_deduce | `&:?` | `:?` | 16894 |
| [4e28b132](01_data/260525_Cryptarithm/arithmetic_4e28b132/track/tree_cot.txt) | arithmetic | cryptarithm_deduce | `:}` | `//>:` | 27769 |
| [50a694de](01_data/260525_Cryptarithm/arithmetic_50a694de/track/tree_cot.txt) | arithmetic | cryptarithm_deduce | `%]` | `#$` | 10044 |

## T3. Cross-tab survival of the 800 cryptarithm rows in cryptn

- `oversampling=2` (trained ×2): **265**
- `oversampling=0` (excluded): **535**
- other / non-numeric: 0

### Exclusion breakdown (rows with `oversampling=0`)

- over-CAP only (token>=7680 AND GT True): **192**
- GT-mismatch only (token<CAP AND GT False): **152**
- both (token>=CAP AND GT False): **191**
- total excluded: 535

### Moe baseline `oversampling` for the same 800 cryptarithm ids (`260514_huikang_golden_stripped.csv`)

| moe oversampling | count |
|---|---:|
| `0` | 0 |
| `1` | 800 |
| `2` | 0 |
| `other` | 0 |

### Moe oversampling, broken out by training cat

- **cryptarithm_deduce** (n=639): `1`=639
- **cryptarithm_guess** (n=161): `1`=161

In the moe golden CSV, **every** cryptarithm row got `oversampling=1` — no gate, no exclusions.

## T4. Per-puzzle val delta for the 71 val cryptarithm puzzles

val cryptarithm_deduce: 55 rows · val cryptarithm_guess: 16 rows.

### cryptarithm_deduce (n=55)

- included in cryptn training (`oversampling=2`): **16**
- excluded by token-CAP (>= 7680): **28**
- excluded by GT-mismatch only: **8**
- moe val correct on these ids: **1**
- cryptn val correct on these ids: **0**

### cryptarithm_guess (n=16)

- included in cryptn training (`oversampling=2`): **3**
- excluded by token-CAP (>= 7680): **10**
- excluded by GT-mismatch only: **3**
- moe val correct on these ids: **0**
- cryptn val correct on these ids: **0**

**Correlation table for cryptarithm_deduce (cryptn included × cryptn eval correct):**

|              | eval correct | eval wrong | total |
|---|---:|---:|---:|
| **included** | 0 | 16 | 16 |
| **excluded** | 0 | 39 | 39 |

**Correlation table for cryptarithm_guess (cryptn included × cryptn eval correct):**

|              | eval correct | eval wrong | total |
|---|---:|---:|---:|
| **included** | 0 | 3 | 3 |
| **excluded** | 0 | 13 | 13 |

### Per-puzzle table — `cryptarithm_deduce` val (all 55)

| id | cryptn_os | moe_os | tok_len | gt_match | moe_corr | cryptn_corr | reason if excluded |
|---|---|---|---:|---|---|---|---|
| [042f1e53](../../01_data/260525_Cryptarithm/) | `0` | `1` | 47504 | False | False | False | over-CAP + GT-miss |
| [0df82d52](../../01_data/260525_Cryptarithm/) | `0` | `1` | 2386 | False | False | False | GT-mismatch |
| [0f6436da](../../01_data/260525_Cryptarithm/) | `2` | `1` | 2627 | True | False | False |  |
| [10552d46](../../01_data/260525_Cryptarithm/) | `NOT_IN_CSV` | `NOT_IN_CSV` | None | None | False | False |  |
| [236a2204](../../01_data/260525_Cryptarithm/) | `0` | `1` | 63680 | False | False | False | over-CAP + GT-miss |
| [24750c4a](../../01_data/260525_Cryptarithm/) | `0` | `1` | 65140 | True | False | False | over-CAP |
| [24e1f1d5](../../01_data/260525_Cryptarithm/) | `2` | `1` | 2934 | True | False | False |  |
| [2f46a715](../../01_data/260525_Cryptarithm/) | `0` | `1` | 21585 | True | False | False | over-CAP |
| [2ff200fb](../../01_data/260525_Cryptarithm/) | `0` | `1` | 9692 | True | False | False | over-CAP |
| [36d2d728](../../01_data/260525_Cryptarithm/) | `2` | `1` | 5408 | True | False | False |  |
| [39a1f5e9](../../01_data/260525_Cryptarithm/) | `2` | `1` | 2385 | True | False | False |  |
| [41554020](../../01_data/260525_Cryptarithm/) | `2` | `1` | 3181 | True | False | False |  |
| [424b50d1](../../01_data/260525_Cryptarithm/) | `2` | `1` | 2669 | True | False | False |  |
| [4d8df95b](../../01_data/260525_Cryptarithm/) | `0` | `1` | 195352 | True | False | False | over-CAP |
| [4e8982d6](../../01_data/260525_Cryptarithm/) | `0` | `1` | 12821 | True | False | False | over-CAP |
| [51181706](../../01_data/260525_Cryptarithm/) | `0` | `1` | 94217 | True | False | False | over-CAP |
| [52395e9a](../../01_data/260525_Cryptarithm/) | `0` | `1` | 77901 | True | False | False | over-CAP |
| [55bc449c](../../01_data/260525_Cryptarithm/) | `0` | `1` | 2377 | False | False | False | GT-mismatch |
| [5690981d](../../01_data/260525_Cryptarithm/) | `0` | `1` | 6954 | False | False | False | GT-mismatch |
| [6de4855a](../../01_data/260525_Cryptarithm/) | `0` | `1` | 8983 | True | False | False | over-CAP |
| [712fe3e4](../../01_data/260525_Cryptarithm/) | `0` | `1` | 8642 | True | False | False | over-CAP |
| [7660ac93](../../01_data/260525_Cryptarithm/) | `0` | `1` | 11452 | True | False | False | over-CAP |
| [771472d6](../../01_data/260525_Cryptarithm/) | `0` | `1` | 2269 | False | False | False | GT-mismatch |
| [7f5758a8](../../01_data/260525_Cryptarithm/) | `0` | `1` | 106248 | False | False | False | over-CAP + GT-miss |
| [81b6d789](../../01_data/260525_Cryptarithm/) | `2` | `1` | 2937 | True | False | False |  |
| [8dac3984](../../01_data/260525_Cryptarithm/) | `0` | `1` | 26800 | True | False | False | over-CAP |
| [9081e954](../../01_data/260525_Cryptarithm/) | `0` | `1` | 15187 | True | False | False | over-CAP |
| [90feb0c5](../../01_data/260525_Cryptarithm/) | `0` | `1` | 1655 | False | False | False | GT-mismatch |
| [91e9dc52](../../01_data/260525_Cryptarithm/) | `2` | `1` | 5019 | True | False | False |  |
| [9346686a](../../01_data/260525_Cryptarithm/) | `0` | `1` | 63620 | False | False | False | over-CAP + GT-miss |
| [94367b1d](../../01_data/260525_Cryptarithm/) | `0` | `1` | 19222 | True | False | False | over-CAP |
| [969a6b00](../../01_data/260525_Cryptarithm/) | `0` | `1` | 50163 | True | False | False | over-CAP |
| [9a0daca9](../../01_data/260525_Cryptarithm/) | `2` | `1` | 4157 | True | False | False |  |
| [a11311a4](../../01_data/260525_Cryptarithm/) | `0` | `1` | 4058 | False | False | False | GT-mismatch |
| [a1220274](../../01_data/260525_Cryptarithm/) | `0` | `1` | 41162 | False | False | False | over-CAP + GT-miss |
| [a692ec38](../../01_data/260525_Cryptarithm/) | `0` | `1` | 3241 | False | False | False | GT-mismatch |
| [a8323497](../../01_data/260525_Cryptarithm/) | `NOT_IN_CSV` | `NOT_IN_CSV` | None | None | False | False |  |
| [afb53516](../../01_data/260525_Cryptarithm/) | `2` | `1` | 6693 | True | False | False |  |
| [b0206bb7](../../01_data/260525_Cryptarithm/) | `0` | `1` | 22367 | True | False | False | over-CAP |
| [b1b10e83](../../01_data/260525_Cryptarithm/) | `2` | `1` | 1814 | True | True | False |  |
| [bc83b0a1](../../01_data/260525_Cryptarithm/) | `2` | `1` | 2288 | True | False | False |  |
| [c37e694c](../../01_data/260525_Cryptarithm/) | `NOT_IN_CSV` | `NOT_IN_CSV` | None | None | False | False |  |
| [c413ac69](../../01_data/260525_Cryptarithm/) | `0` | `1` | 8461 | True | False | False | over-CAP |
| [c9463bb4](../../01_data/260525_Cryptarithm/) | `2` | `1` | 2288 | True | False | False |  |
| [cd5e23c7](../../01_data/260525_Cryptarithm/) | `0` | `1` | 19118 | False | False | False | over-CAP + GT-miss |
| [cf79c10e](../../01_data/260525_Cryptarithm/) | `0` | `1` | 46606 | True | False | False | over-CAP |
| [d29e19a4](../../01_data/260525_Cryptarithm/) | `0` | `1` | 17098 | True | False | False | over-CAP |
| [d5602cc5](../../01_data/260525_Cryptarithm/) | `0` | `1` | 7541 | False | False | False | GT-mismatch |
| [da062350](../../01_data/260525_Cryptarithm/) | `0` | `1` | 136616 | False | False | False | over-CAP + GT-miss |
| [dafd11de](../../01_data/260525_Cryptarithm/) | `2` | `1` | 3834 | True | False | False |  |
| [db66c42c](../../01_data/260525_Cryptarithm/) | `0` | `1` | 19538 | True | False | False | over-CAP |
| [efa7edc3](../../01_data/260525_Cryptarithm/) | `2` | `1` | 4609 | True | False | False |  |
| [f7828fc1](../../01_data/260525_Cryptarithm/) | `0` | `1` | 34929 | False | False | False | over-CAP + GT-miss |
| [fb440865](../../01_data/260525_Cryptarithm/) | `0` | `1` | 421805 | False | False | False | over-CAP + GT-miss |
| [fcad9241](../../01_data/260525_Cryptarithm/) | `2` | `1` | 2561 | True | False | False |  |

### Per-puzzle table — `cryptarithm_guess` val (all 16)

| id | cryptn_os | moe_os | tok_len | gt_match | moe_corr | cryptn_corr | reason if excluded |
|---|---|---|---:|---|---|---|---|
| [21ee162c](../../01_data/260525_Cryptarithm/) | `0` | `1` | 6334 | False | False | False | GT-mismatch |
| [258b796b](../../01_data/260525_Cryptarithm/) | `0` | `1` | 2848066 | False | False | False | over-CAP + GT-miss |
| [432b1110](../../01_data/260525_Cryptarithm/) | `0` | `1` | 1005 | False | False | False | GT-mismatch |
| [5501c054](../../01_data/260525_Cryptarithm/) | `2` | `1` | 7157 | True | False | False |  |
| [62fc7798](../../01_data/260525_Cryptarithm/) | `2` | `1` | 4568 | True | False | False |  |
| [6fc35fc2](../../01_data/260525_Cryptarithm/) | `0` | `1` | 54281 | False | False | False | over-CAP + GT-miss |
| [73e0994b](../../01_data/260525_Cryptarithm/) | `0` | `1` | 143539 | False | False | False | over-CAP + GT-miss |
| [76c48f67](../../01_data/260525_Cryptarithm/) | `0` | `1` | 23998 | True | False | False | over-CAP |
| [9eacc9a2](../../01_data/260525_Cryptarithm/) | `2` | `1` | 3096 | True | False | False |  |
| [a40497f9](../../01_data/260525_Cryptarithm/) | `0` | `1` | 9656 | False | False | False | over-CAP + GT-miss |
| [a9d91ec1](../../01_data/260525_Cryptarithm/) | `0` | `1` | 13859 | False | False | False | over-CAP + GT-miss |
| [d6f18fc8](../../01_data/260525_Cryptarithm/) | `0` | `1` | 2851 | False | False | False | GT-mismatch |
| [da10a947](../../01_data/260525_Cryptarithm/) | `0` | `1` | 12259 | True | False | False | over-CAP |
| [db564ba9](../../01_data/260525_Cryptarithm/) | `0` | `1` | 86649 | False | False | False | over-CAP + GT-miss |
| [e6af1658](../../01_data/260525_Cryptarithm/) | `0` | `1` | 11975 | False | False | False | over-CAP + GT-miss |
| [f5126d48](../../01_data/260525_Cryptarithm/) | `0` | `1` | 116851 | False | False | False | over-CAP + GT-miss |

## T5. Old vs new CoT structure — 3 cryptarithm_deduce puzzles

Selected: prefer (moe_corr=True AND cryptn_os=2); fall back to any deduce puzzle included in cryptn.

### Puzzle `b1b10e83`

**Lengths.** old (moe CoT) = **608** tokens / 53 lines  ·  new (260525) = **1814** tokens / 80 lines  ·  **Δ = +1206 tokens** (+198%).

**OLD CoT (moe row) — head 50 lines:**
```
We need to infer the transformation rule from the examples.

First, let me assign letters to each symbol:
" -> A
\ -> B
# -> C
| -> D
! -> E
) -> F
{ -> G
$ -> H
` -> I
Operators:
+ -> x
* -> y

Converting examples to letter form:

  「"\+#|」 = 「"\#|」:
    input:  " \ + # | : " -> A, \ -> B, + -> x, # -> C, | -> D -> AB x CD
    output: " \ # | : " -> A, \ -> B, # -> C, | -> D -> ABCD

  「#!+#\」 = 「#!#\」:
    input:  # ! + # \ : # -> C, ! -> E, + -> x, # -> C, \ -> B -> CE x CB
    output: # ! # \ : # -> C, ! -> E, # -> C, \ -> B -> CECB

  「#"*#"」 = 「#){|」:
    input:  # " * # " : # -> C, " -> A, * -> y, # -> C, " -> A -> CA y CA
    output: # ) { | : # -> C, ) -> F, { -> G, | -> D -> CFGD

  「$\*)"」 = 「${$#」:
    input:  $ \ * ) " : $ -> H, \ -> B, * -> y, ) -> F, " -> A -> HB y FA
    output: $ { $ # : $ -> H, { -> G, $ -> H, # -> C -> HGHC

  「)\+`#」 = 「)\`#」:
    input:  ) \ + ` # : ) -> F, \ -> B, + -> x, ` -> I, # -> C -> FB x IC
    output: ) \ ` # : ) -> F, \ -> B, ` -> I, # -> C -> FBIC

Each input is 5 characters: two symbol-digits, an operator, two more symbol-digits.
Operator y: unknown
Operator x: concatenation

The question operator is x, which is concatenation.

Converting question "|"+#$": | " + # $ : | -> D, " -> A, + -> x, # -> C, $ -> H -> DA x CH

Applying to DA x CH:
  concatenation(DA, CH) = DA || CH = DACH
  Converting back: D A C H : D -> |, A -> ", C -> #, H -> $ -> |"#$

```
**OLD CoT — tail 20 lines:**
```

  「)\+`#」 = 「)\`#」:
    input:  ) \ + ` # : ) -> F, \ -> B, + -> x, ` -> I, # -> C -> FB x IC
    output: ) \ ` # : ) -> F, \ -> B, ` -> I, # -> C -> FBIC

Each input is 5 characters: two symbol-digits, an operator, two more symbol-digits.
Operator y: unknown
Operator x: concatenation

The question operator is x, which is concatenation.

Converting question "|"+#$": | " + # $ : | -> D, " -> A, + -> x, # -> C, $ -> H -> DA x CH

Applying to DA x CH:
  concatenation(DA, CH) = DA || CH = DACH
  Converting back: D A C H : D -> |, A -> ", C -> #, H -> $ -> |"#$

I will now return the answer in \boxed{}
The answer in \boxed is
\boxed{|"#$}
```
**NEW CoT (260525) — head 50 lines:**
```
I need to infer the transformation rule from the examples. Both the operands and operators are unknown, so I need to deduce them both.

First, let me assign letters to each symbol:
  ! -> A
  " -> B
  # -> C
  $ -> D
  ) -> E
  \ -> F
  ` -> G
  { -> H
  | -> I
Operators:
  + -> f
  * -> g

I will convert all the equations to letter form:
  EX1 "\+#| = "\#| becomes BF f CI = BFCI
  EX2 #!+#\ = #!#\ becomes CA f CF = CACF
  EX3 #"*#" = #){| becomes CB g CB = CEHI
  EX4 $\*)" = ${$# becomes DF g EB = DHDC
  EX5 )\+`# = )\`# becomes EF f GC = EFGC
  QUERY |"+#$ becomes IB f CD

Prior knowledge for this kind of question:
1. We only consider four kinds of operations — noisy_multiplication (~mul), noisy_addition (~add), noisy_subtraction (~sub), and noisy_concatenation (~concat). They are noisy because the result is one of a fixed set of variants: the exact value, off by ±1 or ±2, and — for subtraction — also its negated or operand-reversed form. Specifically:
   ~mul = [a×b, a×b±1, a×b±2], ~add = [a+b, a+b±1, a+b±2], ~sub = [|a-b|, a-b, -|a-b|, b-a, a-b±1, a-b±2], ~concat = [ab, ba], all ordered by frequency. Any other operation is deemed exotic.
2. The arithmetic operators are distinct, but the concatenation operator may repeat.
3. Each letter represents a distinct digit 0-9, and the leading symbol of any number is never 0.
4. If a symbol exists in the result (RHS), it must be a negative sign.
5. The reading order is either rightward or leftward (units digit first). I will check rightward first.

Now start to traverse the solution tree, reading order = [rightward, leftward]:
§1 reading rightward:
  §1.1 writing equations:
    EX1: f(BF, CI) = BFCI
    EX2: f(CA, CF) = CACF
    EX3: g(CB, CB) = CEHI
    EX4: g(DF, EB) = DHDC
    EX5: f(EF, GC) = EFGC
    QUERY: f(IB, CD)
    f = [~add, ~sub, ~mul, ~concat], g = [~add, ~sub, ~mul, ~concat].
  §1.2 quick check to see whether any operator is concatenation:
    f: concat_fwd=BFCI, concat_rev=CIBF. Output=BFCI. FWD matches -> so lock f = concat_fwd = a∥b.
    g: concat_fwd=CBCB, concat_rev=CBCB. Output=CEHI. Neither matches, so g is not concat.
    f = [concat_fwd], g = [~add, ~sub, ~mul]
  §1.3 prune solution trees by the RHS digit-count before entering:
    g: EX3 and EX4 have 4-digit RHS (only multiplication of two 2-digit numbers reaches 4 digits).
      Only ~mul survives the digit-count filter. Try g = ~mul.
    No further pruning found. We have f = [concat_fwd], g = [~mul] to traverse in order.
```
**NEW CoT — tail 20 lines:**
```
      EX3: 98 ~mul 98 in [9603,9605], RHS 960I in [9602,9607] -> I={3-5}
      EX4: 1F ~mul 68 in [1019,1157], RHS 1019 in [1019,1019] -> F={5}
      We now know all the operands in EX4: ~mul(15, 68) = 1019.
        1019 = 15×68-1, so lock g = mul_minus1 = a×b-1.
      EX3: 98 mul_minus1 98 in [9603,9603], RHS 960I in [9603,9604] -> I={3}
      No equation narrows a variable further. Branch on A in [2,4,7] (the variable with the fewest candidates left), try in order.
      Try A=2:
        Current State: f=concat_fwd g=mul_minus1 A={2} B={8} C={9} D={1} E={6} F={5} G={2,4,7} H={0} I={3}
        No equation narrows a variable further. Branch on G in [4,7] (the variable with the fewest candidates left), try in order.
        Try G=4:
          Current State: f=concat_fwd g=mul_minus1 A={2} B={8} C={9} D={1} E={6} F={5} G={4} H={0} I={3}
  Conclusion of step §1: {f = concat_fwd, g = mul_minus1, A=2, B=8, C=9, D=1, E=6, F=5, G=4, H=0, I=3}, 11 of 11 resolved. This is the answer.

Summary: Confirm reading order = rightward, f = concat_fwd, g = mul_minus1

Now solve the QUERY f(IB, CD), applying f = concat_fwd = a∥b,
f(IB, CD) = IB∥CD = IBCD; map the letters back to symbols, so the answer is |"#$

I will now return the answer in \boxed{}, the answer is
\boxed{|"#$}
```

### Puzzle `24e1f1d5`

**Lengths.** old (moe CoT) = **533** tokens / 50 lines  ·  new (260525) = **2934** tokens / 108 lines  ·  **Δ = +2401 tokens** (+450%).

**OLD CoT (moe row) — head 50 lines:**
```
We need to infer the transformation rule from the examples.

First, let me assign letters to each symbol:
% -> A
! -> B
< -> C
} -> D
^ -> E
" -> F
[ -> G
( -> H
\ -> I
# -> J
Operators:
+ -> x
* -> y

Converting examples to letter form:

  「%!+<}」 = 「%!<}」:
    input:  % ! + < } : % -> A, ! -> B, + -> x, < -> C, } -> D -> AB x CD
    output: % ! < } : % -> A, ! -> B, < -> C, } -> D -> ABCD

  「<^*"^」 = 「["(」:
    input:  < ^ * " ^ : < -> C, ^ -> E, * -> y, " -> F, ^ -> E -> CE y FE
    output: [ " ( : [ -> G, " -> F, ( -> H -> GFH

  「!\*%[」 = 「#}(\」:
    input:  ! \ * % [ : ! -> B, \ -> I, * -> y, % -> A, [ -> G -> BI y AG
    output: # } ( \ : # -> J, } -> D, ( -> H, \ -> I -> JDHI

  「^<*<<」 = 「^"<」:
    input:  ^ < * < < : ^ -> E, < -> C, * -> y, < -> C, < -> C -> EC y CC
    output: ^ " < : ^ -> E, " -> F, < -> C -> EFC

Each input is 5 characters: two symbol-digits, an operator, two more symbol-digits.
Operator y: unknown
Operator x: concatenation

The question operator is x, which is concatenation.

Converting question "^#+(!": ^ # + ( ! : ^ -> E, # -> J, + -> x, ( -> H, ! -> B -> EJ x HB

Applying to EJ x HB:
  concatenation(EJ, HB) = EJ || HB = EJHB
  Converting back: E J H B : E -> ^, J -> #, H -> (, B -> ! -> ^#(!

I will now return the answer in \boxed{}
The answer in \boxed is
\boxed{^#(!}
```
**OLD CoT — tail 20 lines:**
```

```
**NEW CoT (260525) — head 50 lines:**
```
I need to infer the transformation rule from the examples. Both the operands and operators are unknown, so I need to deduce them both.

First, let me assign letters to each symbol:
  ! -> A
  " -> B
  # -> C
  % -> D
  ( -> E
  < -> F
  [ -> G
  \ -> H
  ^ -> I
  } -> J
Operators:
  + -> f
  * -> g

I will convert all the equations to letter form:
  EX1 %!+<} = %!<} becomes DA f FJ = DAFJ
  EX2 <^*"^ = ["( becomes FI g BI = GBE
  EX3 !\*%[ = #}(\ becomes AH g DG = CJEH
  EX4 ^<*<< = ^"< becomes IF g FF = IBF
  QUERY ^#+(! becomes IC f EA

Prior knowledge for this kind of question:
1. We only consider four kinds of operations — noisy_multiplication (~mul), noisy_addition (~add), noisy_subtraction (~sub), and noisy_concatenation (~concat). They are noisy because the result is one of a fixed set of variants: the exact value, off by ±1 or ±2, and — for subtraction — also its negated or operand-reversed form. Specifically:
   ~mul = [a×b, a×b±1, a×b±2], ~add = [a+b, a+b±1, a+b±2], ~sub = [|a-b|, a-b, -|a-b|, b-a, a-b±1, a-b±2], ~concat = [ab, ba], all ordered by frequency. Any other operation is deemed exotic.
2. The arithmetic operators are distinct, but the concatenation operator may repeat.
3. Each letter represents a distinct digit 0-9, and the leading symbol of any number is never 0.
4. If a symbol exists in the result (RHS), it must be a negative sign.
5. The reading order is either rightward or leftward (units digit first). I will check rightward first.

Now start to traverse the solution tree, reading order = [rightward, leftward]:
§1 reading rightward:
  §1.1 writing equations:
    EX1: f(DA, FJ) = DAFJ
    EX2: g(FI, BI) = GBE
    EX3: g(AH, DG) = CJEH
    EX4: g(IF, FF) = IBF
    QUERY: f(IC, EA)
    f = [~add, ~sub, ~mul, ~concat], g = [~add, ~sub, ~mul, ~concat].
  §1.2 quick check to see whether any operator is concatenation:
    f: concat_fwd=DAFJ, concat_rev=FJDA. Output=DAFJ. FWD matches -> so lock f = concat_fwd = a∥b.
    g: concat_fwd=FIBI, concat_rev=BIFI. Output=GBE. Neither matches, so g is not concat.
    f = [concat_fwd], g = [~add, ~sub, ~mul]
  §1.3 prune solution trees by the RHS digit-count before entering:
    g: EX2 and EX4 have 3-digit RHS (rules out subtraction, whose magnitude is at most 2 digits); EX3 has 4-digit RHS (only multiplication of two 2-digit numbers reaches 4 digits).
      Only ~mul survives the digit-count filter. Try g = ~mul.
    No further pruning found. We have f = [concat_fwd], g = [~mul] to traverse in order.
  §1.4 enter the tree and search for an operator and digit assignment:
```
**NEW CoT — tail 20 lines:**
```
            EX3: 30 mul D7 in [2610,2910], RHS 2J60 in [2860,2960] -> D={9}, J={8}
            all symbols pinned but the equations don't all check — no.
          Try A=8:
            Current State: f=concat_fwd g=mul A={8} B={5} C={2,3,8,9} D={2,3,8,9} E={6} F={1} G={7} H={0,2,3,8,9} I={4} J={0,2,3,8,9}
            EX3: 8H mul D7 in [2160,8633], RHS CJ6H in [2060,9969] -> C={2,3}
            EX3: 8H mul D7 in [2160,8633], RHS CJ6H in [2060,3969] -> D={2,3}
            No equation narrows a variable further. Branch on C in [2,3] (the variable with the fewest candidates left), try in order.
            Try C=2:
              Current State: f=concat_fwd g=mul A={8} B={5} C={2} D={2,3} E={6} F={1} G={7} H={0,2,3,9} I={4} J={0,2,3,9}
              All-different: 2 is taken by C, so D={3}.
              EX3: 8H mul 37 in [2960,3293], RHS 2J6H in [2060,2969] -> H={0}, J={9}
  Conclusion of step §1: {f = concat_fwd, g = mul, A=8, B=5, C=2, D=3, E=6, F=1, G=7, H=0, I=4, J=9}, 12 of 12 resolved. This is the answer.

Summary: Confirm reading order = rightward, f = concat_fwd, g = mul

Now solve the QUERY f(IC, EA), applying f = concat_fwd = a∥b,
f(IC, EA) = IC∥EA = ICEA; map the letters back to symbols, so the answer is ^#(!

I will now return the answer in \boxed{}, the answer is
\boxed{^#(!}
```

### Puzzle `afb53516`

**Lengths.** old (moe CoT) = **824** tokens / 65 lines  ·  new (260525) = **6693** tokens / 217 lines  ·  **Δ = +5869 tokens** (+712%).

**OLD CoT (moe row) — head 50 lines:**
```
We need to infer the transformation rule from the examples.

First, let me assign letters to each symbol:
[ -> A
& -> B
: -> C
# -> D
% -> E
> -> F
| -> G
/ -> H
" -> I
< -> J
Operators:
* -> x
+ -> y
- -> z

Digit mapping:
A = 3, B = 9, C = 1, D = 4, E = 2, F = 5, G = 6, H = 8, I = 7, J = 0
Reading order: little-endian (units digit first)

Converting examples to letter form:

  「[&-:#」 = 「-%>」:
    input:  [ & - : # : [ -> A, & -> B, - -> z, : -> C, # -> D -> AB z CD
    output: - % > : - -> -, % -> E, > -> F -> -EF

  「|:+/>」 = 「#"」:
    input:  | : + / > : | -> G, : -> C, + -> y, / -> H, > -> F -> GC y HF
    output: # " : # -> D, " -> I -> DI
    digits: GC = 16, HF = 58, addition(16, 58) = 74, DI = 74

  「#>+</」 = 「#[:」:
    input:  # > + < / : # -> D, > -> F, + -> y, < -> J, / -> H -> DF y JH
    output: # [ : : # -> D, [ -> A, : -> C -> DAC
    digits: DF = 54, JH = 80, addition(54, 80) = 134, DAC = 134

  「%"+%&」 = 「#|:」:
    input:  % " + % & : % -> E, " -> I, + -> y, % -> E, & -> B -> EI y EB
    output: # | : : # -> D, | -> G, : -> C -> DGC
    digits: EI = 72, EB = 92, addition(72, 92) = 164, DGC = 164

  「[&*>/」 = 「><&"」:
    input:  [ & * > / : [ -> A, & -> B, * -> x, > -> F, / -> H -> AB x FH
    output: > < & " : > -> F, < -> J, & -> B, " -> I -> FJBI
    digits: AB = 93, FH = 85, multiplication(93, 85) = 7905, FJBI = 7905

Each input is 5 characters: two symbol-digits, an operator, two more symbol-digits.
Operator x: multiplication
```
**OLD CoT — tail 20 lines:**
```
    output: > < & " : > -> F, < -> J, & -> B, " -> I -> FJBI
    digits: AB = 93, FH = 85, multiplication(93, 85) = 7905, FJBI = 7905

Each input is 5 characters: two symbol-digits, an operator, two more symbol-digits.
Operator x: multiplication
Operator y: addition
Operator z: rsub_signed

The question operator is x, which is multiplication.

Converting question "#%*</": # % * < / : # -> D, % -> E, * -> x, < -> J, / -> H -> DE x JH

Applying to DE x JH:
  DE = 24, JH = 80
  multiplication(24, 80) = 1920
  Converting back: J E B C : J -> <, E -> %, B -> &, C -> : -> <%&:

I will now return the answer in \boxed{}
The answer in \boxed is
\boxed{<%&:}
```
**NEW CoT (260525) — head 50 lines:**
```
I need to infer the transformation rule from the examples. Both the operands and operators are unknown, so I need to deduce them both.

First, let me assign letters to each symbol:
  " -> A
  # -> B
  % -> C
  & -> D
  / -> E
  : -> F
  < -> G
  > -> H
  [ -> I
  | -> J
Operators:
  - -> f
  + -> g
  * -> h

I will convert all the equations to letter form:
  EX1 [&-:# = -%> becomes ID f FB = -CH
  EX2 |:+/> = #" becomes JF g EH = BA
  EX3 #>+</ = #[: becomes BH g GE = BIF
  EX4 %"+%& = #|: becomes CA g CD = BJF
  EX5 [&*>/ = ><&" becomes ID h HE = HGDA
  QUERY #%*</ becomes BC h GE

Prior knowledge for this kind of question:
1. We only consider four kinds of operations — noisy_multiplication (~mul), noisy_addition (~add), noisy_subtraction (~sub), and noisy_concatenation (~concat). They are noisy because the result is one of a fixed set of variants: the exact value, off by ±1 or ±2, and — for subtraction — also its negated or operand-reversed form. Specifically:
   ~mul = [a×b, a×b±1, a×b±2], ~add = [a+b, a+b±1, a+b±2], ~sub = [|a-b|, a-b, -|a-b|, b-a, a-b±1, a-b±2], ~concat = [ab, ba], all ordered by frequency. Any other operation is deemed exotic.
2. The arithmetic operators are distinct, but the concatenation operator may repeat.
3. Each letter represents a distinct digit 0-9, and the leading symbol of any number is never 0.
4. If a symbol exists in the result (RHS), it must be a negative sign.
5. The reading order is either rightward or leftward (units digit first). I will check rightward first.

f appears in EX1 output side (RHS), that means it can only be a negative sign, so f = [~sub].
By the distinct operator rule, g, h cannot be ~sub.

Now start to traverse the solution tree, reading order = [rightward, leftward]:
§1 reading rightward:
  §1.1 writing equations:
    EX1: f(ID, FB) = -CH
    EX2: g(JF, EH) = BA
    EX3: g(BH, GE) = BIF
    EX4: g(CA, CD) = BJF
    EX5: h(ID, HE) = HGDA
    QUERY: h(BC, GE)
    f = [~sub], g = [~add, ~mul, ~concat], h = [~add, ~mul, ~concat].
  §1.2 quick check to see whether any operator is concatenation:
    g: concat_fwd=JFEH, concat_rev=EHJF. Output=BA. Neither matches, so g is not concat.
    h: concat_fwd=IDHE, concat_rev=HEID. Output=HGDA. Neither matches, so h is not concat.
```
**NEW CoT — tail 20 lines:**
```
        EX5: 93 ~mul 85 in [7904,7906], RHS 5G97 in [7905,7945] -> G={0}
        We now know all the operands in EX5: ~mul(93, 85) = 7905.
          7905 = 93×85, so lock h = mul = a×b.
        All-different: 3 is taken by I, so B={4}.
        We now know all the operands in EX2: ~add(16, 58) = 74.
          74 = 16+58, so lock g = add = a+b.
          g also appears in EX3, EX4; check EX3: 54+80 = 134; EX4: 72+92 = 164. Confirmed.
        We now know all the operands in EX1: ~sub(93, 41) = -52.
          more than one variant of f reaches it, so try each:
          rsub_signed(93, 41) = -52:
          every equation checks out, so lock f = rsub_signed = b-a.
  Conclusion of step §2: {f = rsub_signed, g = add, h = mul, A=7, B=4, C=2, D=9, E=8, F=1, G=0, H=5, I=3, J=6}, 13 of 13 resolved.

Summary: §2 resolved more unknowns than §1, so §2 is preferred. Confirm reading order = leftward, f = rsub_signed, g = add, h = mul

Now solve the QUERY h(CB, EG) = h(24, 80), applying h = mul(a, b) = a×b,
h(24, 80) = 24×80 = 1920; reading order is leftward, so the result 1920 -> 0291; map the digits back to symbols, 1 -> :, 9 -> &, 2 -> %, 0 -> <, so the answer is <%&:

I will now return the answer in \boxed{}, the answer is
\boxed{<%&:}
```

### Structural commentary

- The **new CoTs** are §-tree honest-DFS search traces produced by `_honest_solver.py` (`01_data/260525_Cryptarithm/_honest_solver.py`); they explicitly enumerate variable ranges, constraint propagation, locks, branches, and backtracks. State lines, interval shrinks, and per-equation locks are all written out token-by-token.
- The **old moe CoTs** (`260514_huikang_golden_stripped.csv`) are much more compact, narrative-style derivations that go directly to the symbol↔digit assignment with relatively little visible search.
- Net effect: new CoTs are **multi-x longer** in tokens. For ~60–70% of the 800 puzzles they exceed the 7680-token training cap (see T1) and get zeroed out by the gate, while moe trained on all of them at `oversampling=1`.

## T6. Per-category training-mass swing across runs

For each of the 4 runs, for each of the 9 categories: row count and **Σ oversampling**. `Δvs moe` = `Σos_run − Σos_moe`. **Bold** if `|Δ%| > 10`.

### Σ oversampling totals per run

| run | rows | Σ os |
|---|---:|---:|
| **moe** | 6906 | 7849 |
| **alleq** | 7077 | 8020 |
| **alleqgt** | 7077 | 7883 |
| **cryptn** | 7077 | 7613 |

### `bit_manipulation`

| run | rows | Σ os | Δ vs moe (abs) | Δ vs moe (%) |
|---|---:|---:|---:|---:|
| moe | 1354 | 1754 | +0 | +0.0% |
| alleq | 1354 | 1754 | +0 | +0.0% |
| alleqgt | 1354 | 1754 | +0 | +0.0% |
| cryptn | 1354 | 1754 | +0 | +0.0% |

### `cipher`

| run | rows | Σ os | Δ vs moe (abs) | Δ vs moe (%) |
|---|---:|---:|---:|---:|
| moe | 1576 | 1656 | +0 | +0.0% |
| alleq | 1576 | 1656 | +0 | +0.0% |
| alleqgt | 1576 | 1656 | +0 | +0.0% |
| cryptn | 1576 | 1656 | +0 | +0.0% |

### `cryptarithm_deduce`

| run | rows | Σ os | Δ vs moe (abs) | Δ vs moe (%) |
|---|---:|---:|---:|---:|
| moe | 639 | 639 | +0 | +0.0% |
| alleq | 639 | 639 | +0 | +0.0% |
| alleqgt | 639 | 639 | +0 | +0.0% |
| cryptn | 639 | 486 | **-153** | **-23.9%** |

### `cryptarithm_guess`

| run | rows | Σ os | Δ vs moe (abs) | Δ vs moe (%) |
|---|---:|---:|---:|---:|
| moe | 161 | 161 | +0 | +0.0% |
| alleq | 161 | 161 | +0 | +0.0% |
| alleqgt | 161 | 161 | +0 | +0.0% |
| cryptn | 161 | 44 | **-117** | **-72.7%** |

### `equation_numeric_deduce`

| run | rows | Σ os | Δ vs moe (abs) | Δ vs moe (%) |
|---|---:|---:|---:|---:|
| moe | 540 | 658 | +0 | +0.0% |
| alleq | 596 | 714 | +56 | +8.5% |
| alleqgt | 596 | 676 | +18 | +2.7% |
| cryptn | 596 | 676 | +18 | +2.7% |

### `equation_numeric_guess`

| run | rows | Σ os | Δ vs moe (abs) | Δ vs moe (%) |
|---|---:|---:|---:|---:|
| moe | 21 | 126 | +0 | +0.0% |
| alleq | 136 | 241 | **+115** | **+91.3%** |
| alleqgt | 136 | 142 | **+16** | **+12.7%** |
| cryptn | 136 | 142 | **+16** | **+12.7%** |

### `gravity`

| run | rows | Σ os | Δ vs moe (abs) | Δ vs moe (%) |
|---|---:|---:|---:|---:|
| moe | 975 | 1055 | +0 | +0.0% |
| alleq | 975 | 1055 | +0 | +0.0% |
| alleqgt | 975 | 1055 | +0 | +0.0% |
| cryptn | 975 | 1055 | +0 | +0.0% |

### `numeral`

| run | rows | Σ os | Δ vs moe (abs) | Δ vs moe (%) |
|---|---:|---:|---:|---:|
| moe | 650 | 730 | +0 | +0.0% |
| alleq | 650 | 730 | +0 | +0.0% |
| alleqgt | 650 | 730 | +0 | +0.0% |
| cryptn | 650 | 730 | +0 | +0.0% |

### `unit_conversion`

| run | rows | Σ os | Δ vs moe (abs) | Δ vs moe (%) |
|---|---:|---:|---:|---:|
| moe | 990 | 1070 | +0 | +0.0% |
| alleq | 990 | 1070 | +0 | +0.0% |
| alleqgt | 990 | 1070 | +0 | +0.0% |
| cryptn | 990 | 1070 | +0 | +0.0% |

## T7. Pre-/post-gate Σ os for cryptn cryptarithm rows

- **Pre-gate Σos** (assume every cryptarithm row = `oversampling=2`): **1600**  (2×800)
- **Post-gate Σos** (actual cryptn): **530**
  - cryptarithm_deduce: **486** (= 243×2)
  - cryptarithm_guess: **44** (= 22×2)
- **Mass thrown away by the gate**: 1070 (66.9% of intended mass)

Contrast: moe trained every cryptarithm row at `oversampling=1`, giving Σos = 800 for cryptarithm rows. Cryptn intended 1600, delivered 530 → cryptarithm got **less** total training mass than moe even though each surviving row was 2×.

---

## Bottom line

Top 2-3 reasons cryptarithm collapsed to 0/55 (deduce) in cryptn:

1. **The new 260525 CoTs are far too long.** 383 of 800 cryptarithm puzzles exceed the 7680-token CAP and were excluded by the gate. The §-tree honest-DFS solver produces multi-thousand-token search traces (max in our scan: see T2 table, tail puzzles in the 10k–200k+ range). Moe trained all 800 rows at `oversampling=1` (Σos=800); cryptn delivered Σos=530 (-33.8% vs moe).

2. **GT-mismatch on the surviving CoTs.** Of the 800 puzzles, only 457 have a correct boxed answer (57%). The honest solver is faithful by construction but only **derives** an answer for ~57% of the set (see `01_data/260525_Cryptarithm/SUMMARY.md` — derived 71–100%, guess-buckets 5–34%). The gate then zeroes the other ~43%. moe baseline had no such gate.

3. **Stacked: the gate prunes from both directions.** Of 535 cryptarithm rows excluded, 192 were over-CAP only, 152 GT-mismatch only, 191 both. The model sees almost none of the cryptarithm distribution — only 265 puzzles × 2 = 530 training samples, distributed across the two subcategories (243 deduce + 22 guess). On the val side, only **16 of 55 val_deduce ids** were even present in training with non-zero weight — unsurprisingly the trained model produces 0 correct.
