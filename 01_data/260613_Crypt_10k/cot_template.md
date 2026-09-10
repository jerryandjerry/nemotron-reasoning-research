# Cryptarithm CoT Template (adapted from the matured Alice / equation-numeric system, 2026-05-25)

> **Revised 2026-06-14 (clean folder `260613_Crypt_10k`, rightward-only overhaul)** — synced to the current
> `_gen_crypt.gen_cot`; narration-only, 0 boxed-answer changes vs. the prior revision, GT 285/797, trainable
> (GT-True ∩ under-7680-tokens) 145. Changes from the prior (leftward) revision:
> 1. **Reading is RIGHTWARD ONLY.** Leftward (units-digit-first) reading is removed. Prior #5 now reads
>    `The reading order is rightward only.`; there is no §2 reading, no `Need to try reading leftward`, no
>    leftward tie-break/illegal verdict. The §-tree is **flat — `§1 … §5`** (not the old nested `§N.1 … §N.5`).
>    A §1 that fails to resolve the query is **simply unsolvable → default to `~concat`**.
> 2. **No `QUERY_MIN` early-stop.** The search never stops on "the query is answerable now"; every locked
>    operator is verified against **all** examples before it is accepted (`every equation checks out, so lock …`).
>    The old `enough to answer; stop searching` line and its Conclusion form are gone.
> 3. **§4 header is method-explicit:** `narrow each symbol's domain by comparing its lhs range with its rhs
>    range, dropping any value that leaves the two unable to meet:`
> 4. **Domain narrowing is shown as an explicit forward witness or a binary search** (no opaque interval
>    isolate). A monotone operand boundary is found by **binary search** (`binary search the largest E that
>    keeps lhs ≤ rhs:` then `If E = 5: … ≤ 999 — yes.` / `If E = 7: … > 999 — no.`); a single eliminated value
>    is a **forward boundary witness** (`If C = 9: … ≥ 2158. The largest rhs is 1999 < 2158. Impossible, so
>    C = {2-8}.`). A **two-sided** symbol (in an operand AND the rhs) shows the **binding** value — the closest
>    the two sides ever come — not the loosest.
> 5. **Dead branches end `… so no digit of X is left.`** (no `— this branch fails`). The **terminal** dead-end
>    of a genuinely unsolvable puzzle is annotated `… no digit of X is left, X is not solvable.` (the terminal
>    line drops the `so` and appends `, X is not solvable`) — this is added **only** to the last narrowing line
>    before `Summary: Equations are arithmetically unsolvable`, never to a §5 backtracking branch (which the
>    next `Try` resolves, keeping plain `… so no digit of X is left.`) nor a recoverable family combo.
>
> Earlier corrections still in force: prior #1 drops the exotic/`a mod b` clause; **exotic = arithmetic-
> unsolvable → default to `~concat`** (NOT `a mod b` — that was numeric's); a QUERY-only operator is named
> `is a QUERY-only operator`, **omitted** from `Current State:`, and resolved by default; no `guess` /
> `= unknown` verdict. Unchanged structure: step-0 BPE merge-split table + symbol-by-symbol reading + "denote
> unknown functions" / "assign a letter"; **locks write the formula only**; concat shown as `a∥b` / `b∥a`;
> unseen-query STEP-1/2a/2b concat short-circuit; missing-answer-digit ASCII default.

## Background

A cryptarithm puzzle (`question.txt`) gives example equations and one query. Unlike equation-numeric,
the operands are **ciphered symbols**, not literal numbers — so the CoT must deduce **both** the
symbol→digit mapping **and** the operators. We write a CoT that deduces the answer; it earns a score as
long as it follows this deduce process, wording, and is reasonable. Every step is **deterministic**;
defaulting is allowed only when nothing can be deduced, and the default must itself be deterministic (it is
never narrated as a `guess`). **Any random choice scores 0.**

## The CoT is a SYSTEM LOG, not a polished explanation (governing principle)

The CoT is a literal log of the solver as it runs: do a step → print what it did → print the resulting
running state; repeat. No refinement, no cover, no fake prints. Four rules:

1. **Forward-only.** A line may use ONLY values already derived above it. Never pre-compute a later step's
   result into the current line. (This is why a dead branch shows the **binding** probe value, not a value
   chosen with hindsight about where the boundary lands.)
2. **A `lock` / a narrow is a real state change — no fake prints.** When the log prints `lock f = …` or
   `-> A={5}`, the candidate set actually collapses at that moment, and every later line reflects it. If a
   line is printed, the state changed; if the state changed in a way the cadence reports, a line is printed.
   **Every lock moment writes the formula explicitly:** `… so lock [LABEL] = [FORMULA].` where `[FORMULA]` is
   the abstract rule in `a`/`b` — `a×b`, `a×b+1`, `a-b`, `|a-b|`, `-|a-b|`, `b-a`, `a+b`, `a+b+2`, `a∥b`,
   `b∥a`, etc. (the **formula only** — no `variant =` label such as `mul_plus1 =` or `concat_rev =`). This
   holds at **all** lock sites: the single-survivor lock (§5 branch / §4 propagate), the try-each branch
   that resolves, and the concat lock (§2).
3. **Log the running state.** The full `Current State:` line lists the candidate set of **every unknown** —
   the operators AND every digit symbol, space-separated with **no separator character** (`|` is avoided: it
   is itself an operand symbol and the absolute-value bar). **Cadence: print the full `Current State:` line
   only right after entering a `Try`** (the operator combo `Try operators …:`, and each branch value
   `Try X=v:`); the narrowings in between are shown as deltas (`-> A={5-9}`, or `so A = {2-9}`). A digit is
   **solved when its set is a singleton**; an operator when it locks.
4. **Never skip a step because the outcome is "already known."** Print the step, then its result.

Write the log honestly and do not fear mistakes — reviewers recompute every line.

## Unknowns (the key difference from equation-numeric)

Two kinds of unknown, both carried in the running state:
- **operators** `f, g, h` — candidate **families** (`~mul, ~add, ~sub, ~concat`), shown `[…]` (ordered by
  frequency); collapse to a locked **formula** (`a×b+1`, `a+b`, `a-b`, `|a-b|`, `a∥b`, …).
- **digit symbols** `A, B, …` — candidate **digit sets** ⊆ `{0-9}`, shown `{…}` (leading symbols `{1-9}`).

`X of Y resolved` counts **all** unknowns (every digit symbol + every operator).

## Brackets & set notation
- `[…]` = an **ordered list tried in order** (operator family lists; the branch try-list).
- `{…}` = an **unordered candidate set** (a digit symbol's running state). A digit set is rendered `[…]`
  only at the moment we branch on it (`Branch on C in [1-9] … Try C=1`).
- **Run-collapse:** a contiguous run of length ≥3 is written `lo-hi`; singletons/pairs stay explicit —
  `{0,2,3,4,5,6,7,8,9}` → `{0,2-9}`, `{2,3,4,7,8,9}` → `{2-4,7-9}`, `{5,6}` → `{5,6}`. Same for `[…]`.

## Sign / character legend (use the EXACT char; no look-alikes)

Same legend as equation-numeric. Token ids from the Nemotron tokenizer (vocab 131072).

| Meaning | Char | Codepoint | Token id |
|---|---|---|---|
| subtraction / minus / negative sign | `-` | U+002D | 1045 |
| addition / plus | `+` | U+002B | 1043 |
| multiplication | `×` | U+00D7 | 7581 |
| plus-or-minus (variant set only) | `±` | U+00B1 | 19120 |
| absolute value | `\|` | U+007C | 1124 |
| concatenation | `∥` | U+2225 | 33778+1165 |
| noisy-operator prefix | `~` | U+007E | 1126 |
| equals | `=` | U+003D | 1061 |
| maps-to / becomes / reverses-to | `->` | U+002D U+003E | 1941 |
| clause dash | `—` | U+2014 | 1674 |
| ≤ / ≥ (range witness) | `≤` `≥` | U+2264 U+2265 | — |

- Minus is ALWAYS `-` (U+002D), whether subtraction, a negative sign, or a puzzle operator symbol.
- **Puzzle digit symbols and operator symbols** are displayed verbatim (whatever the puzzle gives; ~26
  ASCII punctuation chars), and are NOT in this legend.

## Full deduce process (exact wording required; `code` = literal CoT lines, *italics* = notes)

### Opening, BPE un-merge, symbol reading, letter form (unnumbered prose)

The CoT first un-merges the BPE tokens (so the model transcribes correctly), reads each equation
symbol-by-symbol, names the operators, then letters the operands. Exact wording:

```
Each symbol can be a hidden operator or operand. I work out the digit cipher and each operator from the example equations, then apply them to the QUERY and return the answer. Even a symbol that looks like an operator — /, <, >, %, &, |, ^ — can represent an operand here; the only rule is that the third symbol of the lhs is always the operator.

Split the BPE merged tokens in the examples:
  [TOKEN]  becomes  [SPACED]
  ...
Now I read each equation one symbol at a time, using the splits above:
  EX[N]  [RAW_LHS] = [RAW_RHS]  becomes  [SPACED_LHS] = [SPACED_RHS]
  ...
  QUERY [RAW]  becomes  [SPACED]
lhs and the QUERY are always 5 symbols and the third symbol is the operator, so in the puzzle there are [N] unique operators, which are [OP_SYM] , [OP_SYM]
Now denote them as unknown functions:
  [OP_SYM] -> f
  [OP_SYM] -> g
The other symbols are operands; assign a letter to each in the order of appearance:
  [OPERAND_SYM] -> A
  [OPERAND_SYM] -> B
  ...

I will convert all the equations to letter form:
  EX[N] [SPACED_LHS] = [SPACED_RHS] becomes [LETTERS] f [LETTERS] = [LETTERS]
  ...
  QUERY [SPACED] becomes [LETTERS] g [LETTERS]
```
*The **step-0 BPE merge-split table** lists every multi-symbol token the tokenizer merges in this puzzle's
equations+query, in first-appearance order, each split to its individual symbols (`  [TOKEN]  becomes
[SPACED]`, two spaces). If the puzzle has NO merged token the header line is instead `First, I should read
each equation one symbol at a time, writing out every symbol separately:` (and there is no "using the splits
above"). The operator-count line is singular when N=1: `there is [N] unique operator, which is [OP_SYM]`.
Operators are denoted `f, g, h` by first appearance; operand symbols lettered `A, B, …` by first appearance.
The **convert-to-letter-form** block transcribes every symbol literally to its assigned letter, so an operator
symbol appearing in the RHS becomes that operator's **letter** (`… = g H F`). Only the **§1 'writing equations'**
block re-renders that RHS operator as the sign `-` (`… = -H F`, prior #4) — the form the gut-check/solve use.*

### Prior knowledge (unnumbered prose, numbered list)

```
Prior knowledge for this kind of question:
1. We only consider four kinds of operations — noisy_addition (~add), noisy_subtraction (~sub), noisy_multiplication (~mul), and noisy_concatenation (~concat). They are noisy because the result has a tolerance of ±2 from the exact value and — for subtraction — may also appear as its absolute value or operand-reversed form. Specifically:
   ~add = [a+b, a+b±1, a+b±2], ~sub = [|a-b|, a-b, -|a-b|, b-a, a-b±1, a-b±2], ~mul = [a×b, a×b±1, a×b±2], ~concat = [ab, ba], all ordered by frequency. The absolute forms |a-b|, -|a-b| carry no noise — they are themselves the unsigned reading of the signed difference; the ±2 tolerance applies to the signed form a-b.
2. The arithmetic operators are distinct, but the concatenation operator may repeat.
3. Each letter represents a distinct digit 0-9, and the leading symbol of any number is never 0.
4. If a symbol exists in the result (rhs), it must be a negative sign.
5. The reading order is rightward only.
6. If the QUERY operator only appears in the QUERY, we cannot solve it from the examples; we call it a QUERY-only operator. If the QUERY operator or operand appears in the examples but cannot be solved, the puzzle is arithmetically unsolvable and we default to the ~concat operation. In either case we still resolve it to a concrete answer.
```
*Prior knowledge is items 1–6. **#5 is now `The reading order is rightward only.`** (leftward removed).
**#6 names the two "can't-solve" outcomes and both default to a concrete answer:** a **QUERY-only operator**
(appears ONLY in the query → no examples to derive from → resolve by default) and a puzzle that is **not
arithmetic solvable** (an in-example operator/operand cannot be solved → default the answer to `~concat`).
There is **NO `a mod b`** and **NO "exotic operator" verdict** — an in-example unsolvable operator simply
makes the puzzle not-arith-solvable → blind `~concat`. Item 1 lists the families in the tie-break order
**`~add › ~sub › ~mul`** (concat is detected in §2). The **digit-count prune still runs first** (§3); the
order only breaks ties. The ordering is **not narrated** — the §3 SOFT note just says `Try [LABEL] = [~X]
first.`*

### Sign check (unnumbered prose; emitted ONLY when an operator symbol appears in an RHS)

```
[LABEL] appears in EX[N] output side (rhs), that means it can only be a negative sign, so [LABEL] = [~sub].
By the distinct operator rule, [other ops] cannot be ~sub.
```

### Traverse the solution tree (rightward only — a single, flat §1 … §5)

```
Now start to traverse the solution tree, reading order = [rightward]:
```
*There is exactly one reading. Its five sub-steps are numbered **flat** `§1 … §5` (no `§N.x` nesting):
§1 write equations + gut-check, §2 concat check, §3 digit-count prune, §4 narrow domains (deterministic),
§5 branch-and-search.*

**§1 — write equations + the gut-check.**
```
  §1 writing equations:
    EX[M]: [LABEL]([SA], [SB]) = [RHS]
    ...
    QUERY: [LABEL]([QSA], [QSB])
    f = [~add, ~sub, ~mul, ~concat], g = [~add, ~sub, ~mul, ~concat][, h is a QUERY-only operator].
```
*Compact operands, tens-first: `[SA]` = the operand's two symbols as written (e.g. `A B`). `[RHS]` = the
output symbol-string (an operator symbol in it shown as the sign `-`, e.g. `-A B`). The last line is the
operator candidate-family state (a sign-confirmed op shows `[~sub]`). **Gut-check** (legality, rightward
only): if any RHS **ends** in an operator symbol it cannot be read this way (a number's sign comes first) →
`Note: EX[i] rhs [R] ends in the sign '[s]', but a number's sign comes first — illegal.` Because there is
no alternative reading, an illegal §1 makes the puzzle unsolvable → `~concat`. A leading sign is legal.*

**§2 — concat check.**
```
  §2 quick check to see whether any operator is concatenation:
    [LABEL]: a∥b=[FWD letters], b∥a=[REV letters]. rhs=[RHS letters]. [Neither matches, so [LABEL] is not concat. | [FWD|REV] matches -> so lock [LABEL] = [a∥b | b∥a].]
    [LABEL]: already ~sub by its output sign, skip the concat check.    // sign-confirmed op skips the concat check
    Note: [LABEL] is the QUERY operator, and it has been locked as [a∥b|b∥a]. At this point other unknowns are irrelevant, because the answer is simply the concatenation of the query operands. Proceed to answer directly.    // SHORT-CIRCUIT: if the locked-concat op is the QUERY operator, skip the rest of the solve
    ...
    f = [...], g = [b∥a], h is a QUERY-only operator    // operator state restated: concat-locked ops show their formula, sign ops [~sub], a QUERY-only operator named as such, others their families minus ~concat
```
*The concat candidates are shown as the **formula** `a∥b` / `b∥a` (no `concat_fwd`/`concat_rev` label); the
fwd/rev concatenations and the rhs are spelled in spaced letters (`a∥b=A B A C`). The lock writes the formula
only: `… matches -> so lock [LABEL] = b∥a.` A locked concat op confirms on its other examples. Direction is
read off the first example whose operands differ; equal-operand examples are deferred. **Short-circuit:** if the
op that just locked as concat **is the QUERY operator**, emit `Note: [LABEL] is the QUERY operator, and it has
been locked as [a∥b|b∥a]. At this point other unknowns are irrelevant … Proceed to answer directly.` and skip
straight to the Conclusion + answer (the cipher is never fully solved). If **every** operator locks as concat
but the query op is QUERY-only, the bridge line is `Every operator is concatenation; nothing remains to search.`*

**§3 — digit-count prune (choose each operator's family to try first).**
```
  §3 prune solution trees by the rhs digit-count before entering:
    [LABEL]: already shown to be ~sub by its output sign.                 // if sign-confirmed
    [LABEL]: EX[M] [has|have] [d]-digit rhs ([what it rules out]); ... .   // otherwise
      // one survives:        Only [~X] survives the digit-count filter. Try [LABEL] = [~X].
      // distinctness forces: Surviving candidates: [~X], [~Y]. By the distinct operator rule [LABEL] must be the remaining operation, [~Z].
      // ≥2 (SOFT):           Surviving candidates: [~X], [~Y]. Try [LABEL] = [~X] first.
    No further pruning found. We have f = [...], g = [...] to traverse in order.
```
*Digit-count clauses (by rhs symbol count): 1-digit `rules out addition and multiplication, only a
subtraction-type reaches 1 digit`; 2-digit `rules out multiplication, which gives ≥3 digits`; 3-digit
`rules out subtraction, whose magnitude is at most 2 digits`; 4-digit `only multiplication of two 2-digit
numbers reaches 4 digits`.*

**§4 — narrow domains (deterministic; NO branching).**
```
  §4 narrow each symbol's domain by comparing examples' lhs range with its rhs range, dropping any value that leaves the two unable to meet:
    Try operators f = [~X], g = [~Y][, h = ...]:
    Current State: f=[~X] g=[~Y][ h=...] A={..} B={..} ... [every digit symbol, space-separated, NO separator char]
    Elimination: among the unsolved symbols only [VAR] can still be [d], and all ten digits 0-9 are used, so [VAR]=[d].   // hidden-single: [d] is usually 0 but is ANY forced digit (e.g. "only A can still be 3 … so A=3."); when fewer than 10 symbols remain the count clause reads "the [N] symbols use up all [N] remaining digits"
    EX[M]: [~X]([SA], [SB]) = [RHS].
        // (a) binary-search witness — a MONOTONE operand boundary (one side rises with the symbol):
        The largest rhs is [RHS] = [hi]; binary search the largest [VAR] that keeps lhs ≤ rhs:
        If [VAR] = [v]: [SA] ≥ [..], [SB] ≥ [..], so lhs [~X] ≥ a×b−2 = [..]×[..]−2 = ([t]+[u])×[..] − 2 = [t]×[..] + [u]×[..] − 2 = [P] + [Q] − 2 = [val] [≤|>] [hi] — [yes|no].
        ...                                                                                          // midpoint probes
        If [VAR] = [v]: … — yes, so [VAR] = {lo-hi}.                                                  // last probe carries the result
        // (the smallest-side form: "The smallest rhs is [RHS] = [lo]; binary search the smallest [VAR] that keeps lhs ≥ rhs:")
        // (b) single-value forward boundary witness — the eliminated value, computed forward, compared:
        If [VAR] = [v]: [SA] ≥ [..], [SB] ≥ [..], so lhs [~X] ≥ [..]. The largest rhs is [hi] < [val]. Impossible, so [VAR] = {..}.
        If [VAR] = [v]: the smallest rhs is [RHS] = [lo]. The largest lhs [~X]([SA], [SB]) is [val] < [lo]. Impossible, so [VAR] = {..}.   // rhs-symbol case
        // (b-sub) ~sub is NON-MONOTONE in the operand (|a-b| is V-shaped), so binary search is invalid -> ENUMERATE every value (§4 only). Bound is always -|a-b|-2 (lower) / |a-b|+2 (upper):
        ~sub is non-monotone in [VAR], so test each value:
        If [VAR] = 1: [SA] ≤ [..], [SB] = [..], so lhs ~sub ≥ −|a−b|−2 = −|[x]−[y]|−2 = −[d]−2 = [val] [>|≤] largest rhs [hi] — [no|yes].
        ... (every value 1..9 / 0..9) ...
        If [VAR] = 9: …, so lhs ~sub ≥ −|a−b|−2 = … — yes, so [VAR] = {.. possibly non-contiguous, e.g. {1,8,9} ..}.
        // (c) DEAD branch — every value of the symbol is eliminated:
        If [VAR] = [v]: … Impossible, so no digit of [VAR] is left.
    Advance to the next family combination: f = [~X], g = [~Z]:    // a tried combo died; try the next §3 combo
    Current State: ...                                            // fresh state for the new combo
    ...
    Current State: f=[~X] g=[~Z] A={..} ...                       // §4 closes by restating the narrowed domains (only if §4 narrowed something); an operator LOCKED mid-§4 shows its FORMULA here, not [~X] (e.g. "f=[~add] g=a×b")
```
*Opens with `Try operators …:` then the **full** `Current State:` line. Then only **forced** cuts. Each cut
is a **forward witness**: pin a value, compute that side forward (a `~mul` is expanded by partial products),
and compare to the other side's range. When the lhs is **monotone** in the symbol (`~add`/`~mul`, symbol on
one side), the survivors form a contiguous range and the boundary is found by **binary search** (probe
midpoints; `yes` keeps the side, `no` cuts it; the last probe states `so [VAR] = {lo-hi}.`). When a single
value is eliminated, the **forward boundary witness** form is used. **`~sub` is non-monotone** (its `|a-b|`
magnitude is V-shaped in the symbol, so survivors can be **non-contiguous** like `{1,8,9}` and a binary search
would be wrong) — so a `~sub` operand cut is shown by **enumerating every value** (`~sub is non-monotone in
[VAR], so test each value:` then one `If [VAR] = v:` line per digit), **§4 only** (§5's per-branch `~sub` cuts
stay as the single-value witness). The `~sub` lhs bound is always **`−|a−b|−2`** (lower) / **`|a−b|+2`**
(upper) — shown at the operand endpoints that maximise `|a-b|`. A **two-sided symbol** (in an operand AND
the rhs — non-monotone) shows the **binding** value, the one minimising the gap between the two sides (the
closest they ever come), and excludes any digit already locked to another symbol. A symbol on the **rhs only**
compares `the smallest rhs … the largest lhs …` (or `largest rhs … smallest lhs …`). A **dead** branch
(every value gone) ends `… so no digit of [VAR] is left.` No branching in §4. Multiple §3 combos are tried in
order with `Advance to the next family combination: …` between them. **A QUERY-only operator is OMITTED from
`Current State:`** — it has no examples; it is named `is a QUERY-only operator` in §1 / Conclusion / Summary.*

**§5 — branch-and-search (global MRV, backtracking).**
```
  §5 search the remaining unknowns by branching, applying the §4 domains as given:
    Current State: f=[~X] g=[~Y] A={..} ...                       // the post-§4 domains, printed ONCE at the §5 head (if §4 left unknowns)
    No equation narrows a variable further. Branch on [VAR] in [..] (fewest candidates left).
    Try [VAR]=[v]: removes [v] from [VAR2, VAR3, ...].            // all-different delta on entering the branch; the "removes …" clause is OMITTED (bare "Try [VAR]=[v]:") when [v] is in no other live domain
      EX[M]: [~X]([SA], [SB]) = [RHS].
          [the same §4 witnesses — forward boundary / binary search — now under the branch]
          ... Impossible, so [VAR] = {..}.                        // a narrowing inside the branch
          EX[M]: ~sub shows a 1-digit result, so the leading digits differ by at most 2, |[L1]-[L2]| ≤ 2 -> [L2]={..}   // LEADPAIR rule D (a ~sub example with a 1-digit rhs forces |lead1-lead2| ≤ 2)
          ... Impossible, so no digit of [VAR] is left.           // this branch dies -> backtrack to the next Try (NO "is not solvable")
      No equation narrows a variable further. Branch on [VAR]=... // nested branch
      Try [VAR]=[v]: removes ...
        All-different: [d] is taken by [VAR], so [VAR2]={..}; [e] is taken by [VAR2], so [VAR3]={..}.
        [VAR] is not pinned by any example; go with the smallest [VAR]=[v].   // the LAST unknown, when >1 digit satisfies every example -> take the smallest by rule (a stated default, not a deduction)
        We now know all the operands in EX[M]: [~X]([a], [b]) = [r].   // an example's operands are all pinned -> lock its operator
          [a]×[b][±k] = ([t]+[u])×[b][±k] = [t]×[b] + [u]×[b][±k] = [P] + [Q][±k] = [r], so lock [LABEL] = a×b[±k].   // ~mul: partial-product expansion, FORMULA only
          [arith] = [r]; rhs [r]; match, try [LABEL] = [FORMULA] and check the other equations:                      // walk the family's readings in order; first that hits the rhs
          every equation checks out, so lock [LABEL] = [FORMULA].                                                     // verified on ALL other examples
          [LABEL] also appears in EX[i]; check EX[i]: [expr] = [val]. Confirmed.                                       // post-lock cross-check: re-verify the locked op on each OTHER fully-pinned example
          // --- LOCK FAILURE (operands pinned but NO family reading lands within ±2) -> this operator family fails ---
          We now know all the operands in EX[M]: [~X]([a], [b]) in [lo,hi].                                          // ranged form when the rhs isn't a single value
          [a]×[b] = [base]; rhs [lo,hi]; [near]-[base] = [±d]; |[±d]| > 2, beyond ±2 — no.                            // mul/add: the base is more than 2 from the rhs
          [arith] = [v]; rhs [target] — no.                                                                          // sub: each structural reading misses
          [arith] = [r] would make [LABEL] = [SYMB], but [SYMB] is already operator [..]'s meaning; operators are distinct, so rule it out.   // a reading collides with another locked op
          no [~X] variant gives [target], so [LABEL] = [~X] fails.                                                    // family exhausted -> Advance to next combo, or unsolvable
```
*The §5 block opens with one `Current State:` line, then branches. BACKTRACKING with **global MRV**: branch
on the unset variable with the **fewest candidates left** (tie → symbol order). Each `Try [VAR]=[v]:` shows
the all-different delta `removes [v] from …` (the full `Current State:` is re-printed only when the branch is
re-entered after narrowing). An operator **locks** the moment one example's operands are all pinned:
- the forced single survivor writes the formula directly (`… = [r], so lock [LABEL] = [FORMULA].`), a `~mul`
  **expanded by partial products** (`56×49+1 = (50+6)×49 + 1 = 50×49 + 6×49 + 1 = 2450 + 294 + 1 = 2745`);
- otherwise the solver **walks the family's readings in their fixed order** (`|a-b| › a-b › -|a-b| › b-a ›
  ±1 › ±2` for ~sub; `a×b › a×b+1 › a×b-1 › …` for ~mul), prints `[arith] = [r]; rhs [r]; match, try [LABEL]
  = [FORMULA] and check the other equations:`, recurses to verify, and on success ends `every equation checks
  out, so lock [LABEL] = [FORMULA].` (the verification against ALL examples replaces the old QUERY_MIN
  early-stop — nothing is locked on the query alone).
When the operands are pinned but **no** family reading lands within ±2, the family **fails**: mul/add print
`[a]×[b] = [base]; rhs [lo,hi]; [near]-[base] = [±d]; |[±d]| > 2, beyond ±2 — no.`, ~sub walks its readings each
ending `— no`, a reading that would collide with another locked op is ruled out (`… would make [LABEL] = [SYMB],
but [SYMB] is already operator …'s meaning; operators are distinct, so rule it out.`), and the family closes with
`no [~X] variant gives [target], so [LABEL] = [~X] fails.` → Advance to the next combo (or, if none remain,
unsolvable→~concat). The lock/fail header is `We now know all the operands in EX[M]: [~X]([a], [b]) = [r]` (or
`… in [lo,hi]` when the rhs is still a range). A within-branch single-value cut may end with a **bare**
`Impossible.` (no `so [VAR] = {..}`) when the resulting domain is restated on the immediately following line.
A failed family/branch is rejected with **`— no`** (never `≠`; `=` only locks). Distinct-arithmetic: a family
locked by one operator is removed from the others. A `Try` value whose forward check exceeds the rhs both ways
ends `Impossible, so no digit of [VAR] is left.` and the search **backtracks** to the next sibling `Try` — this
is NOT a terminal failure, so it is **never** annotated `is not solvable`. Two extra deterministic lines may
appear (in **§4 or §5**, wherever the example narrows): **LEADPAIR rule D** — a `~sub` example with a **1-digit**
rhs forces the two operand leading digits to differ by ≤ 2 (`~sub shows a 1-digit result, so the leading digits
differ by at most 2, |[L1]-[L2]| ≤ 2 -> …`; may carry two updates, `… -> [L2]={..}, [L1]={..}`);
and the **free-digit default** — when only the LAST unknown is unset and **more than one** digit satisfies every
example, it is not pinned, so take the smallest by rule (`[VAR] is not pinned by any example; go with the
smallest [VAR]=[v].`) — a stated repeatable default, never narrated as a guess.*

**Contradiction / dead-end system-log lines** *(any §; all end in the canonical `— no` / `impossible` and are
forward, not enumerated):*
```
All-different: [X] and [Y] are both [d], but every symbol is a distinct digit — no.   // two pinned symbols collide
All-different: [d] is taken by [X], [e] is taken by [Y], so [VAR]={} — no.            // elimination empties a domain (the removed digits leave no value)
All-different: [d] is taken by [X], so [VAR2]={v}, [VAR3]={v}, but every symbol is a distinct digit — no.   // a removal cascade forces two symbols to the SAME value -> contradiction
Elimination: only [k] distinct digits remain for [N] symbols — impossible, no.        // pigeonhole: fewer distinct digits than symbols (e.g. all 10 symbols are leading -> all {1-9})
all symbols pinned but the equations don't all check — no.                            // a full assignment that fails verification -> backtrack
Note: EX[i] rhs [R] ends in the sign '[s]', but a number's sign comes first — illegal.   // gut-check: rhs cannot start with a sign
This reading order is illegal; skip it completely.                                    // follows the illegal gut-check; rightward-only -> unsolvable -> ~concat
```

**Conclusion** *(two forms for a solvable puzzle; the unsolvable case prints NO Conclusion line):*
```
  // (a) fully resolved — lists every unknown (operators AND digit symbols) with its value; X of Y counts every unknown:
  Conclusion: {f = [FORMULA], g = [FORMULA | is a QUERY-only operator], ..., A=[d], ..., [last]=[d]}, [X] of [Y] resolved. This is the answer.
  // (a') fully resolved but a QUERY-ONLY symbol/operator can't be example-determined — append the clause, still "This is the answer.":
  Conclusion: {...}, [X] of [Y] resolved. [Z] appears only in the query, so no reading can determine it — this reading is already as complete as possible. This is the answer.
  // plural when several query-only symbols/ops: "[Z1], [Z2], … appear only in the query, so no reading can determine them — …"
  // (b) partially resolved — some operators/digits pinned but the query operator/operands cannot be -> default to concat:
  Conclusion: {f = [FORMULA | [~..] | is a QUERY-only operator], ...}, [X] of [Y] resolved; cannot answer the query, so default to ~concat.
```
- **Everything resolved (X == Y):** append ` This is the answer.` (with the `[Z] appears only in the query …`
  clause first if a query-only symbol/operator was resolved only by default).
- **Anything unresolved (X < Y) but X > 0:** append `; cannot answer the query, so default to ~concat.`
  **Exception — the concat short-circuit:** if the **query operator was answered** (it locked as concat in §2,
  or it is the QUERY-only op resolved by the concat default), the Conclusion is **bare** (`… X of Y resolved.`)
  with **no** `cannot answer the query …` suffix — the query is answered even though X < Y.
- **Nothing pinned at all (X == 0):** there is **no Conclusion line** — the `Summary:` line states the
  unsolvable verdict directly.
- The label is plain **`Conclusion:`** (there is no `Conclusion of step §N` — that was the leftward form).

### Summary / Confirm (always the last reasoning line before the query answer)

*Three forms, by how §1 ended:*
```
// (a) the puzzle was SOLVED (every unknown resolved):
Summary: Confirm reading order = rightward, f = [FORMULA], g = [FORMULA][, ...]; cipher A=[d] B=[d] ...
// (b) the puzzle is UNSOLVABLE or only partially resolved (default to concat):
Summary: Equations are arithmetically unsolvable; default to ~concat.
// (c) the QUERY operator itself is concatenation (short-circuit — the cipher is never solved):
Summary: Query operator [LABEL] = [FORMULA], no need to solve the other unknowns; Confirm reading order = rightward, [op states]
```
*Form (b) is the single unified unsolvable summary (the partial-resolve Conclusion above feeds into it too).
Just **before** the (b) Summary, the terminal dead-end line — if it is a `… so no digit of X is left.` line — is
rewritten to `… no digit of X is left, X is not solvable.` (drops the `so`, appends `, X is not solvable`;
terminal only).*

### Query answer (apply the resolved/defaulted operator to the QUERY, then map digits back to symbols)

```
// solved arithmetic operator:
Now solve the QUERY [LABEL]([QSA], [QSB]) = [LABEL]([qa], [qb]), applying [LABEL](a, b) = [FORMULA],
[LABEL]([qa], [qb]) = [EXPR] = [RAW]                            // carries a trailing period (`… = [RAW].`) when it connects into the missing-answer-digit block below
map the digits back to symbols, [d] -> [SYM], ..., so [RAW] -> [SPACED] -> [ANS]
[Need to use the operator symbol for the subtraction sign; mapping the subtraction sign [LABEL] back to [OP_SYM], so [MID] -> [ANS].]

// defaulted to concat (unsolvable / partial / QUERY-only):
Now solve the QUERY [LABEL]([QSA], [QSB]), by default [LABEL] = a∥b.
[LABEL]([QSA], [QSB]) = [QSA]∥[QSB] = [letters]
map the letters back to symbols, [d] -> [SYM], ..., so [letters] -> [SPACED] -> [ANS]
```
- line 1 restates the operator as a function def `[LABEL](a, b) = [FORMULA]` (e.g. `g(a, b) = |a-b|`), ending
  with a comma — the **formula only**, no `variant =` label.
- `[EXPR]` applies the operator to the resolved query operands (e.g. `|44-62|`; a `~mul` is expanded by
  partial products); then `= [RAW]`.
- **map digits back to symbols** is cryptarithm-specific: each result digit is rendered as its mapped symbol
  (`1 -> @, 8 -> &, so 18 -> @ & -> @&`). For a **negative** result a second line re-attaches the sign as the
  operator symbol.
- **concat default** maps the raw symbol concatenation back (no digit values needed): `by default [LABEL] =
  a∥b.` → `[LABEL](…) = …∥… = [letters]` → `map the letters back to symbols, …`.
- `[ANS]` (the final symbol string) appears on the answer line and in `\boxed{}`.

### Unseen query operator (the QUERY operator never appears in an example)

Handled in two steps — STEP 1 is logged BEFORE §1, then STEP 2 branches on the 4-digit-rhs signal.

**STEP 1 — always logged for an unseen query operator (before the §1 header):**
```
Note: Query operator [LABEL] appears only in the QUERY, so it is a QUERY-only operator that we resolve by default.
We know that concatenating two 2-digit numbers always produces a 4-digit rhs, and that the operators are usually distinct. A good signal is therefore whether any example has a 4-digit rhs; if none does, there is a good chance the query operator is concatenation.
```
*(No `guess` / `waste of time` / `unknown` — a QUERY-only operator is named and resolved **by default**.)*

**STEP 2a — NO example has a 4-digit rhs → short-circuit to concat, NO search** (the common `query_unseen_concat` path):
```
There is no 4-digit rhs in any example, so by default [LABEL] is concatenation. I use [LABEL] = a∥b.
Now solve the QUERY [LABEL]([o1], [o2]), applying [LABEL] = a∥b,
[LABEL]([o1], [o2]) = [o1]∥[o2] = [letters]; map the letters back to symbols, so the answer is [ANS]
```
then the two-line `\boxed{}` suffix. (The cipher is never solved — the answer is the raw symbol concatenation,
so the §1–§5 search, prior knowledge, and digit deduction are all SKIPPED. A format check must not require them.)

**STEP 2b — some example HAS a 4-digit rhs → not safe to default to concat; solve, then resolve by default at the end:**
`There is a 4-digit rhs in EX[i], so concatenation is risky for [LABEL]; I'll solve the puzzle and see what happens.` → run §1–§5, then at the answer `Query operator [LABEL] appears only in the QUERY, so no example fixes it; I will assign it by default.`
- **no arithmetic operator was solved →** `No operator is solved arithmetically, so [LABEL] is ~concat. Let me use [LABEL] = a∥b.`
- **some arithmetic operator was solved →** infer it is arithmetic too and force the remaining family by
  distinctness:
  - **single forced family** (distinctness leaves exactly one): `At least one operator is solved arithmetically, so [LABEL] is arithmetic too. Since [~X] and [~Y] have appeared in the puzzle, by the distinct operator rule [LABEL] must be the remaining operation, [~Z].`
  - **free family choice** (≥2 families remain): `At least one operator is solved arithmetically, so [LABEL] is arithmetic too. Since [~X] has appeared in the puzzle, by the distinct operator rule [LABEL] is [~Y] or [~Z]; the symbol '[s]' usually means [~W], so [~W] is the pick.` — the canonical-symbol prior (`+`→~add, `-`→~sub, `*`→~mul; default order `~add › ~sub › ~mul` for an arbitrary glyph) makes the pick.
  — then the standard `applying …` query lines.

**No family fits at all (blind fallback):** `The examples aren't arithmetically solvable, so by default
[LABEL] = a∥b; answer = [ANS]` — concat on the raw symbols (needs no digit values). **Do NOT use the
equation-numeric `max-mod-min` fallback** (numeric-only — it needs literal operands).

### Missing answer digit (a result digit whose symbol never appears in the examples)

When the QUERY answer needs a digit that no example pins to a symbol, the cipher cannot map it — so it is
resolved **by default** (first unused operand symbol, in ASCII order). Emitted just before the suffix:
```
But digit [N] doesn't show up in the examples, so I assign its symbol by default.
I know the full set of operand symbols, ordered by ASCII:
  [the 23 operand symbols, space-separated]
Excluding every symbol that already appears in this puzzle (marked X), I still have:
  [same row, each used symbol shown as X]
I know the symbol-to-digit mapping is assigned uniformly at random, so none of the remaining candidates is more likely than another.
Since I have to choose, I will just take the first one that is left: [N -> s, ...], so the answer is [ANS]
```
Plural form when >1 digit is missing: `But digits [N, M] don't show up in the examples, so I assign their
symbols by default.` ("uniformly at random" describes the *prior* — it justifies that the deterministic
first-of-remaining pick is unbiased; it is **not** a non-deterministic guess.) **Render note:** the literal
word `ASCII` is protected from the operand-letter spacer in `_finalize` (its `CII` run would otherwise
become `ASC I I` whenever C and I are cipher letters).

### Suffix (always, two lines)

```
I will now return the answer in \boxed{}, the answer is
\boxed{[ANS]}
```

## Worked example
See `arithmetic_00457d26/track/tree_cot.txt` (the live generator output) — a complete rightward-only CoT for
a typical arithmetic puzzle: the §3 digit-count prune, `Advance to the next family combination` after the
first combo dies in §4, the §4/§5 forward and **binary-search** witnesses, nested MRV branching with
backtracking (`no digit of B is left` → next `Try`), all-different forcing, the eager `~mul` lock with
partial-product expansion, and the `~sub` lock via `match, try … and check the other equations` →
`every equation checks out`. For the unsolvable→concat path with the terminal `… is not solvable.`
annotation, see `little_endian_ff86cd34/track/tree_cot.txt`. Regenerate any puzzle with
`python3 _cryptarithm_solver/_gen_crypt.py <folder>`.
