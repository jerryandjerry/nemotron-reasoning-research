# Cryptarithm CoT Template (adapted from the matured Alice / equation-numeric system, 2026-05-25)

> **Revised 2026-06-13** — wording overhaul synced to the current `_gen_crypt.gen_cot` output (mirrors the
> equation_numeric 2026-06-12 change; narration-only, 0 boxed-answer changes): new opening (`Each symbol can
> be a hidden operator or operand…`); prior-knowledge item 6 now NAMES the two can't-solve verdicts
> (QUERY-only operator / exotic operator) and resolves both **by default** (replaces never-return-"unknown");
> prior #1 ends `…deemed exotic and we will try a mod b`; the unsolved-from-examples verdict prints
> `is a QUERY-only operator` (not `= unknown`); the unseen-query resolution uses **by default** (no
> `guess`/`waste of time`). Unchanged structure: step-0 BPE merge-split table + symbol-by-symbol reading +
> "denote unknown functions" / "assign a letter"; §N split into §N.4/§N.5; **locks write the formula only**;
> concat shown as `a∥b`/`b∥a`; unseen-query STEP-1/2a/2b concat short-circuit. (Stale wording is what the
> format auditors flag as false positives — see the augmentation audit `AUDIT_20260607.md`.)

## Background

A cryptarithm puzzle (`question.txt`) gives example equations and one query. Unlike equation-numeric,
the operands are **ciphered symbols**, not literal numbers — so the CoT must deduce **both** the
symbol→digit mapping **and** the operators. We write a CoT that deduces the answer; it earns a score as
long as it follows this deduce process, wording, and is reasonable. Every step is **deterministic**;
guessing is allowed only when nothing can be deduced, and the guess must itself be deterministic. **Any
random choice scores 0.**

## The CoT is a SYSTEM LOG, not a polished explanation (governing principle)

The CoT is a literal log of the solver as it runs: do a step → print what it did → print the resulting
running state; repeat. No refinement, no cover, no fake prints. Four rules:

1. **Forward-only.** A line may use ONLY values already derived above it. Never pre-compute a later step's
   result into the current line.
2. **A `lock` / a narrow is a real state change — no fake prints.** When the log prints `lock f = …` or
   `-> A={5}`, the candidate set actually collapses at that moment, and every later line reflects it. If a
   line is printed, the state changed; if the state changed in a way the cadence reports, a line is printed.
   **Every lock moment writes the formula explicitly:** `… so lock [LABEL] = [FORMULA].` where `[FORMULA]` is
   the abstract rule in `a`/`b` — `a×b`, `a×b+1`, `a-b`, `|a-b|`, `-|a-b|`, `b-a`, `a+b`, `a+b+2`, `a∥b`,
   `b∥a`, etc. (the **formula only** — no `variant =` label such as `mul_plus1 =` or `concat_rev =`). This
   holds at **all** lock sites: the single-survivor lock (§N.5 branch / §N.4 propagate), the try-each branch
   that resolves, and the concat lock (§N.2).
3. **Log the running state.** The full `Current State:` line lists the candidate set of **every unknown** —
   the operators AND every digit symbol, space-separated with **no separator character** (`|` is avoided: it
   is itself an operand symbol and the absolute-value bar). **Cadence: print the full `Current State:` line
   only right after entering a `Try`** (the operator combo `Try operators …:`, and each branch value
   `Try X=v:`); the narrowings in between are shown as deltas (`-> A={5-9}`). A digit is **solved when its
   set is a singleton**; an operator when it locks.
4. **Never skip a step because the outcome is "already known."** Print the step, then its result.

Write the log honestly and do not fear mistakes — reviewers recompute every line.

## Unknowns (the key difference from equation-numeric)

Two kinds of unknown, both carried in the running state:
- **operators** `f, g, h` — candidate **families** (`~mul, ~add, ~sub, ~concat`), shown `[…]` (ordered by
  frequency); collapse to a locked variant (`mul_plus1`, `add`, `sub_signed`, `absdiff`, `concat_fwd`, …).
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
The letter-form RHS shows an operator symbol in the RHS as the sign `-` (prior #4).*

### Prior knowledge (unnumbered prose, numbered list)

```
Prior knowledge for this kind of question:
1. We only consider four kinds of operations — noisy_addition (~add), noisy_subtraction (~sub), noisy_multiplication (~mul), and noisy_concatenation (~concat). They are noisy because the result has a tolerance of ±2 from the exact value and — for subtraction — may also appear as its absolute value or operand-reversed form. Specifically:
   ~add = [a+b, a+b±1, a+b±2], ~sub = [|a-b|, a-b, -|a-b|, b-a, a-b±1, a-b±2], ~mul = [a×b, a×b±1, a×b±2], ~concat = [ab, ba], all ordered by frequency. The absolute forms |a-b|, -|a-b| carry no noise — they are themselves the unsigned reading of the signed difference; the ±2 tolerance applies to the signed form a-b. Any other operation is deemed exotic and we will try a mod b.
2. The arithmetic operators are distinct, but the concatenation operator may repeat.
3. Each letter represents a distinct digit 0-9, and the leading symbol of any number is never 0.
4. If a symbol exists in the result (rhs), it must be a negative sign.
5. The reading order is either rightward or leftward (units digit first). I will check rightward first.
6. If the QUERY operator only appears in the QUERY, we cannot solve it from the examples; we call it a QUERY-only operator. If the QUERY operator appears in the examples but cannot be solved, it is outside the system, and we call it an exotic operator. In either case we still resolve it to a concrete answer.
```
*Prior knowledge is items 1–6 (#6 names the two "can't-solve" verdicts — QUERY-only operator and exotic
operator — and states both are still resolved to a concrete answer **by default**; replaces the old
never-return-"unknown" nudge). Item 1 lists the families in the tie-break order **`~add › ~sub › ~mul`** (concat is
detected in §N.2); chosen by a 6-order sweep — `add › sub › mul` gave the best GT-match and the shortest CoT.
The **digit-count prune still runs first** (§N.3); the order only breaks ties. The ordering is **not narrated**
— the §N.3 SOFT note just says `Try [LABEL] = [~X] first.` and the unseen-query guess says `I try [~X] (the
first arithmetic family not yet used).`*

### Sign check (unnumbered prose; emitted ONLY when an operator symbol appears in an RHS)

```
[LABEL] appears in EX[N] output side (RHS), that means it can only be a negative sign, so [LABEL] = [~sub].
By the distinct operator rule, [other ops] cannot be ~sub.
```

### §1 / §2 — traverse the solution tree (one branch per reading)

```
Now start to traverse the solution tree, reading order = [rightward, leftward]:
§[N] reading [rightward | leftward, units digit first]:
```
*`§1` = rightward (checked first, prior #5); `§2` = leftward, opened ONLY if `§1` fails to resolve every
unknown (no consistent full assignment) or is ruled illegal. Each branch has **five** sub-steps: §N.1 write
equations + gut-check, §N.2 concat check, §N.3 digit-count prune, §N.4 narrow operand domains (deterministic),
§N.5 branch-and-search.*

**§N.1 — write equations (this reading's frame) + the gut-check.**
```
  §[N].1 writing equations:
    EX[M]: [LABEL]([SA], [SB]) = [RHS]
    ...
    QUERY: [LABEL]([QSA], [QSB])
    f = [~add, ~mul, ~sub, ~concat], g = [~sub][, h = ...].
```
*Compact operands, tens-first: `[SA]` = the operand's two symbols in reading order (rightward = as written,
e.g. `AE`; leftward = reversed). `[RHS]` = the output symbol-string (an operator symbol in it shown as the
sign `-`, e.g. `-AB`). Same compact form as the §N.4/§N.5 search. The last line is the operator candidate-family
state (a sign-confirmed op shows `[~sub]`). **Gut-check** (legality): if any RHS **ends** in an operator
symbol it cannot be read this way (a number's sign comes first) → `This reading order is illegal; skip it
completely.` A leading sign is legal in both readings. (No literal leading-0 number, so no leading-0 pass.)*

**§N.2 — concat check (under this reading).**
```
  §[N].2 quick check to see whether any operator is concatenation:
    [LABEL]: a∥b=[FWD letters], b∥a=[REV letters]. rhs=[RHS letters]. [Neither matches, so [LABEL] is not concat. | [FWD|REV] matches -> so lock [LABEL] = [a∥b | b∥a].]
    [LABEL]: already ~sub by its output sign, skip the concat check.    // sign-confirmed op skips the concat check
    ...
    f = [...], g = [b∥a], h is a QUERY-only operator    // operator state restated: concat-locked ops show their formula, sign ops [~sub], a QUERY-only operator is named as such, others their families minus ~concat
```
*The concat candidates are shown as the **formula** `a∥b` / `b∥a` (no `concat_fwd`/`concat_rev` label); the
fwd/rev concatenations and the rhs are spelled in spaced letters (`a∥b=A B A C`). The lock writes the formula
only: `… matches -> so lock [LABEL] = b∥a.` A locked concat op confirms on its other examples. Direction is
read off the first example whose operands differ; equal-operand examples are deferred.*

**§N.3 — digit-count prune (choose each operator's family to try first).**
```
  §[N].3 prune solution trees by the RHS digit-count before entering:
    [LABEL]: already shown to be ~sub by its output sign.                 // if sign-confirmed
    [LABEL]: EX[M] [has|have] [d]-digit RHS ([what it rules out]); ... .   // otherwise
      // one survives:        Only [~X] survives the digit-count filter. Try [LABEL] = [~X].
      // distinctness forces: Surviving candidates: [~X], [~Y]. By the distinct operator rule [LABEL] must be the remaining operation, [~Z].
      // ≥2 (SOFT):           Surviving candidates: [~X], [~Y]. Try [LABEL] = [~X] first.
    No further pruning found. We have f = [...], g = [...] to traverse in order.
```
*Digit-count clauses (by RHS symbol count): 1-digit `rules out addition and multiplication, only a
subtraction-type reaches 1 digit`; 2-digit `rules out multiplication, which gives ≥3 digits`; 3-digit
`rules out subtraction, whose magnitude is at most 2 digits`; 4-digit `only multiplication of two 2-digit
numbers reaches 4 digits`.*

**§N.4 — narrow the operand domains before branching (deterministic; NO branching).**
```
  §[N].4 narrow the operand domains before branching:
    Try operators f = [~X], g = [~Y][, h = ...]:
    Current State: f=[~X] g=[~Y][ h=...] A={..} B={..} ... [every digit symbol, space-separated, NO separator char]
    EX[M] rhs [RHS letters] in [lo,hi], check lhs [~X]([SA], [SB]):
         [VAR]=[v]: [arith] [< | >] [bound], drop; so [VAR]={..}      // a per-value witness for each forced cut
    ...                                                               // narrowings = deltas, no Current State line
    All-different: [d] is taken by [VAR], so [VAR2]={..}.
```
*Opens with `Try operators f = [~X], g = [~Y]:` then the **full** `Current State:` line — every unknown,
operators then digit symbols, space-separated with **no separator char** (`|` is avoided: it is itself an
operand symbol). Then only **forced** interval cuts, each justified by a **witness** value that fails
(`A=2: 29×29+2 = 843 < 1101, drop; so A={3-9}`). No branching in §N.4. Run-collapsed sets (`{3-9}`, `{0,2-9}`).
**Current-State shorthand:** an operator with no resolved family is written `h=unknown` **in the
`Current State:` line only** — this is the candidate-set shorthand ("function still to be determined"), NOT
the forbidden `unknown`-as-a-verdict. The verdict/Summary/Conclusion lines still name it `is a QUERY-only
operator` (or `is an exotic operator`). Format auditors sometimes flag this `h=unknown` shorthand; it is
intentional and template-sanctioned.*

**§N.5 — search the remaining unknowns by branching (global MRV, backtracking).**
```
  §[N].5 search the remaining unknowns by branching, applying the §[N].4 domains as given:
    No equation narrows a variable further. Branch on [VAR] in [..] (fewest candidates left).
    Try [VAR]=[v]:
      Current State: ... [VAR]={v} ...                              // full Current State right after each Try
      ...
      We now know all the operands in EX[M]: [~X]([a], [b]) = [r].   // RHS still ranged -> "[~X]([a], [b]) in [lo,hi]"
        [r] = [EXPR], so lock [LABEL] = [FORMULA].                  // single survivor (forced); FORMULA only, no variant label
        [a]×[b][±k] = ([t]+[u])×[b][±k] = [t]×[b] + [u]×[b][±k] = [P] + [Q] = [r], so lock [LABEL] = a×b[±k].   // a ~mul lock expands by partial products
        // ~sub ambiguity (≥2 variants reach RHS): "more than one variant of [LABEL] reaches it, so try each:"
        //   [a]-[b] = [r]:                                         // try a variant, recurse under it
        //   every equation checks out, so lock [LABEL] = a-b.      // formula only, on the branch that resolves
        [LABEL] also appears in EX[i]; check EX[i]: [EXPR] = [VAL]. Confirmed.    // cross-check on each OTHER pinned example
```
*BACKTRACKING with **global MRV**: branch on the unset variable with the **fewest candidates left**
(tie → symbol order). Each `Try [VAR]=[v]:` is followed by the full `Current State:`. An operator locks the
moment one example's operands are all pinned — `… so lock [LABEL] = [FORMULA].` (**formula only**, no
`variant =` label). A `~mul` lock is shown **expanded by partial products**
(`53×51+1 = (50+3)×51 + 1 = 50×51 + 3×51 + 1 = 2550 + 153 + 1 = 2704`). `~sub` disambiguates by trying its
variants in order (`|a-b| > a-b > -|a-b| > b-a > ±1/±2`), the resolving branch ending
`every equation checks out, so lock [LABEL] = [FORMULA].` A failed family/branch is rejected with **`— no`**
(never `≠`; `=` only locks). Distinct-arithmetic: a family locked by one operator is removed from the others.*

**Conclusion of step §N** *(lists every unknown — all operators AND all digit symbols — with its resolved
value, or named `is a QUERY-only operator` when it appears only in the QUERY; `X of Y` counts every unknown):*
```
  Conclusion of step §[N]: {f = [FORMULA | is a QUERY-only operator], ..., A=[d], ..., [last]=[d]}, [X] of [Y] resolved.[ This is the answer. | Need to try reading [OTHER].]
```
- **§1 resolved everything (X == Y):** append `This is the answer.` — §2 not opened.
- **§1 left anything unresolved (X < Y — a stuck search, or an unseen query operator):** append `Need to
  try reading [OTHER].`, open §2.
- **§2's conclusion is bare.**

### Summary / Confirm

*If §1 resolved everything (so §2 never ran):*
```
Summary: Confirm reading order = [DIR], f = [FORMULA | is a QUERY-only operator], g = [FORMULA | is a QUERY-only operator][, ...]
```
*If §2 ran: whoever resolved MORE unknowns wins, §1 on a tie; a reading ruled illegal by the gut-check
loses outright. Verdict (then ` Confirm reading order = [DIR], …`):*
- *both legal:* `§X resolved more unknowns than §Y, so §X is preferred.` *(numeric says "operators"; ours
  says "unknowns" because the count includes digit symbols, not just operators)*
- *tie:* `§1 and §2 both resolved [N] unknowns, §1 is preferred.`
- *one illegal:* `§Y is illegal, so §X is the only legal reading.`

### Query answer (apply the resolved/guessed operator to the QUERY, in the confirmed frame, then map digits back to symbols)

```
Now solve the QUERY [LABEL]([QSA], [QSB]) = [LABEL]([qa], [qb]), applying [LABEL](a, b) = [FORMULA],
[LABEL]([qa], [qb]) = [EXPR] = [RAW][; reading order is [leftward …], so the result [RAW] -> [REV]]
map the digits back to symbols, [d] -> [SYM], ..., so [RAW|REV] -> [MID]
[Need to use the operator symbol for the subtraction sign; mapping the subtraction sign [LABEL] back to [OP_SYM], so [MID] -> [ANS].]
```
- line 1 restates the operator as a function def `[LABEL](a, b) = [FORMULA]` (e.g. `g(a, b) = a-b`), ending
  with a comma — the **formula only**, no `variant =` label. **Concat** uses the paren-free `[LABEL] = a∥b,`
  (or `b∥a,`) and a value line `[LABEL]([o1], [o2]) = [o1]∥[o2] = [letters]; map the letters back to symbols, so the answer is [ANS]`.
- `[EXPR]` applies the operator to the resolved query operands (e.g. `76-41`; a `~mul` is expanded by partial
  products `53×51+1 = (50+3)×51 + 1 = …`); then `= [RAW]`.
- the `; reading order is [leftward …], so the result …` clause appears **only under leftward**.
- **map digits back to symbols** is cryptarithm-specific: each result digit is rendered as its mapped symbol.
  For a **negative** result a second line re-attaches the sign as the operator symbol: `Need to use the
  operator symbol for the subtraction sign; mapping the subtraction sign [LABEL] back to [OP_SYM], so [MID] -> [ANS].`
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
so the §1/§2 search, prior knowledge, and digit deduction are all SKIPPED. A format check must not require them.)

**STEP 2b — some example HAS a 4-digit rhs → not safe to default to concat; solve, then resolve by default at the end:**
`There is a 4-digit rhs in EX[i], so concatenation is risky for [LABEL]; I'll solve the puzzle and see what happens.` → run §1/§2, then at the answer `Query operator [LABEL] appears only in the QUERY, so no example fixes it; I will assign it by default.`
- **no arithmetic operator was solved →** `No operator is solved arithmetically, so [LABEL] is ~concat. Let me use [LABEL] = a∥b.`
- **some arithmetic operator was solved →** name the arithmetic candidate set, then narrow by distinctness +
  symbol-canonical prior (`+`→~add, `-`→~sub, `*`→~mul; none for arbitrary symbols), default order `~add › ~sub › ~mul`:
  `At least one operator is solved arithmetically, so [LABEL] is arithmetic too, so [LABEL] = [[FORD]][ (the symbol '[s]' usually means [~Z])].`
  - *≥1 free:* ` Since [PRES], the distinct-operator rule leaves [LABEL] = [[FREE]], so in order I assign [LABEL] = [~Z].`
  - *none free:* ` Since [PRES], every arithmetic operation is already used, so I take the first, [LABEL] = [~Z].`
  — then the standard `applying …` query lines. (`[FORD]` = the full arithmetic order, canonical family first;
  `[FREE]` = that order minus the families already used; `[PRES]` = `f = ~X and g = ~Y` for the solved arith ops.)

**No family fits in either legal reading (blind fallback):** `The examples aren't arithmetically solvable, so by
default [LABEL] = a∥b; answer = [ANS]` — concat on the raw symbols (needs no digit values). **Do NOT use the
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
See `arithmetic_a4e4ec1d/track/tree_cot.txt` (the live generator output) — a complete CoT for a typical
arithmetic puzzle: sign-confirmed ~sub, the MRV digit search with backtracking, eager operator lock, the
~sub variant disambiguation (absdiff → sub_signed), and the digit→symbol answer. Regenerate any puzzle with
`python3 _gen_crypt.py <folder>`.
