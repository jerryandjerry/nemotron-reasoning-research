# Training CoT vs. what the model generates (cryptarithm)

Same puzzle for all four: **id `042f1e53`**, ground truth `-}`.
Prompt examples: `/%*"} = /%"}` , `[/+}" = /%` , `&<-[} = [|` , `//*<< = //<<` ; query `&&-&?`.

## Summary

| | Training CoT it learned from | What the model generates at inference |
|---|---|---|
| **moe** (baseline) | **ikev CoT** (~1.4k chars): states the digit mapping **upfront** — `A=0,B=1,C=4,…` — because ikev was built *from the gold answer*; then "verifies" examples. No derivation/search. | **~640 tok, terminates, WRONG.** Imitates ikev: asserts a **trivial guessed mapping** `A=1,B=2,…,J=0`, does an incoherent "verify", computes → `&`. It learned to *assert* a mapping but has none to copy, so it guesses. |
| **newdata** (ours) | **our honest DFS** (~7.5k tok): legend → letter-form → **derives** the mapping by interval-propagation search + backtracking. | **7,680 tok = ceiling, TRUNCATED, no answer.** Imitates the DFS but **misreads the symbols** and **loops** the search; never reaches `\boxed{}` → `2` (garbage). |

Net: ikev-style teaches "assert a mapping" → model fabricates a trivial one (terminates, wrong). Our style teaches "search for the mapping" → model can't execute/terminate the search (loops, no answer). Both ≈ 0.

---

## Verbatim excerpts

### ikev CoT (moe's training target) — mapping given upfront, gold-derived
```
First, let me assign letters to each symbol:
/ -> A   % -> B   " -> C   } -> D   [ -> E   & -> F   < -> G   | -> H   ? -> I
Operators:  * -> x   + -> y   - -> z
Digit mapping:
A = 0, B = 1, C = 4, D = 2, E = 3, F = 6, G = 7, H = 5, I = 8     <-- the CORRECT mapping (ikev had the answer)
  「&<-[}」 = 「[|」: FG = 61, ED = 29, subtraction(61, 29) = 32, EH = 32 ✓
```

### moe model generates — asserts a TRIVIAL mapping, no derivation (~640 tok, terminates, wrong)
```
First, let me assign letters to each symbol:
/ -> A  % -> B  " -> C  } -> D  [ -> E  + -> F  & -> G  < -> H  | -> I  ? -> J
Operators:  * -> x   - -> y
Digit mapping:
A = 1, B = 2, C = 3, D = 4, E = 5, F = 6, G = 7, H = 8, I = 9, J = 0    <-- trivial guess, not derived
  「[/+}"」: EA = 51, F D = 64, subtraction(51, 64) = -13, ABCD = -13    <-- incoherent
... -> \boxed{&}   (gt -}, WRONG)
```

### our CoT (newdata's training target) — derives the mapping by search (~7.5k tok)
```
First, let me assign letters to each symbol:
  " -> A  % -> B  & -> C  / -> D  < -> E  ? -> F  [ -> G  | -> H  } -> I
Operators:  * -> f   + -> g   - -> h
I will convert all the equations to letter form:
  EX1 「/%*"}」 = 「/%"}」 becomes DB f AI = DBAI
  EX2 「[/+}"」 = 「/%」 becomes GD g IA = DB    ...
[then ~7k tokens of interval-propagation DFS: narrow, lock, backtrack ...]
```

### newdata model generates — misparses + loops (7,680 tok = ceiling, truncated, no answer)
```
First, let me assign letters to each symbol:
  " -> A  & -> B  / -> C  < -> D  ? -> E  [ -> F  \ -> G  | -> H  } -> I    <-- invented "\" and ", dropped %... 
  Operators:  % -> f  + -> g  - -> h                                        <-- read % as an operator (it's a digit op * here)
  EX1 「/%*"}」 = 「/%"}」 becomes CF f AI = CFAI
... runs the DFS and never stops; TAIL (at token 7680):
  EX2: LHS (10C+F) ~add (10A+I) in [20,139], RHS Cf in [11,88] -> A in {1,2}
  (cut off mid-search -> pred '2', WRONG)
```

Note: `042f1e53` is also a puzzle our *offline* solver got wrong (GT-match False), so even a faithful reproduction would miss — but the model never gets that far.
