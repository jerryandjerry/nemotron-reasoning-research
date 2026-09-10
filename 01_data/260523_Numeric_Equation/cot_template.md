# Equation-Numeric CoT Template (2026-05-24)

## Background

You are given a puzzle (`question.txt`) in an exam and must write a CoT that deduces the answer. No one knows the ground-truth answer, but the CoT earns a score as long as it follows this deduce process and wording and is reasonable. Every step must be **deterministic**; guessing is allowed only when nothing can be deduced, and the guess must itself be deterministic and explainable. **Any random choice scores 0.**

## The CoT is a SYSTEM LOG, not a polished explanation

This is the governing principle. **The CoT is a literal log of the solver as it runs: do a step → print what the step did → print the resulting running state; repeat. It is not a beautiful, customized, or after-the-fact writeup — there is no refinement and no cover.** Four rules follow directly:

1. **Forward-only.** A line may use ONLY values already derived by the lines above it. The code must NEVER pre-compute a later step's result and drop it into the current line. Concretely, all of these are violations:
   - consulting the concat check to decide whether the leading-0 gut-check should speak (the leading-0 step runs *before* the concat result exists);
   - writing the sign-re-attached form (`6}`) on the `answer =` line, when the re-attach is a later step;
   - announcing a "Try [op] = X first" family that was chosen by peeking at what eventually locks.
2. **A `lock` is a real state change — there are no fake prints.** When the log prints `lock X = …`, the running candidate set for X *actually* collapses to that value at that moment (`cand[X]` becomes `[that value]`), and every later line reflects the new state. If a line is printed, the state changed; if the state changed, a line is printed.
3. **Log the running state after each step.** Every step (sign check · concat check · digit-count prune · search) ends by printing the current candidate set for every operator (`f = […], g = […]`). That printed line **is** the running state at that point; the next step reads from it. Nothing is held in the solver's head that isn't on the page.
4. **Never skip a step because the outcome is "already known".** When the solver notices a condition (a leading 0, an RHS sign, a digit count), it must print the step that acts on it and *then* print that step's result — step by step — even when the result turns out fine. It must not silently run a check and move on. (E.g. a leading-0 number ALWAYS fires the gut-check: print "…must be concat, check it", then on the next line print the concat result — match → lock concat, or neither → illegal.)

Write the log honestly and do not fear mistakes — the reviewers recompute every line and will surface any error.

## Required Wording

- **f, g, h:** operators are lettered by order of first appearance.
- **Operators we search:** ~add = [a+b, a+b±1, a+b±2], ~mul = [a×b, a×b±1, a×b±2], ~sub = [|a-b|, a-b, -|a-b|, b-a, a-b±1, a-b±2], and ~concat [ab, ba], all ordered by frequency. Any other operation is deemed **exotic** — never fabricated.
- **Reading order is DISCOVERED, not precomputed.** Reading order = [rightward, leftward] (rightward = read the tens digit first, the normal reading; leftward = read the units digit first). Under **leftward** every operand AND the output is read with its digits reversed; in §N.1 the equations are written **already reversed** (e.g. `f(25, 34) = 9` for `52{43 = 9`), and §N.4 then uses those reversed numbers directly — it does not re-explain the reversal.
- **Distinctness is prior knowledge (item 2):** the arithmetic operators are distinct; the concatenation operator may repeat. HARD rule for pruning, NOT re-appended to the "Unknown:" line.
- **Rejections say "no", not "backtrack".** A family only dies on a number it cannot reach — "**far from** [target]", **NEVER `≠`**. Exact `=` is used ONLY to LOCK a variant (`[TG] = [EXPR], so lock …`), never to reject one. The word "fits" is not used.
- **An undetermined operator is "remains unknown".**
- **Indexing — `§` numbers the reading branches only.** The setup (opening, letters, equations, prior knowledge, operator prior, sign check) is **unnumbered prose**. The search tree is `§`-numbered: `§1` is always rightward (checked first, prior #5), `§2` is leftward, opened ONLY if `§1` leaves an operator unresolved (or is ruled illegal). Each branch has four sub-steps `§N.1`–`§N.4` and closes with `Conclusion of step §N: {…}, X of Y resolved.` — bare for §2, while §1 appends its stop/continue decision (`This is the answer.` if X == Y, else `Need to try reading [other].`). A reading whose §N.1 gut-check finds a malformed number (a trailing sign on the RHS, or a leading 0 on any operand/result of a non-concat operator) is declared **illegal and skipped entirely**. Otherwise the winner is chosen on a `Summary:` line (the reading that resolves MORE operators wins, §1 on a tie) that also confirms the reading + every operator's meaning; if §1 already resolved everything there is no comparison, just `Summary: Confirm reading order = …`. Then the query answer (`Now solve the QUERY …`). We use `§` (not `#`) because operator symbols span essentially all ASCII punctuation — `#` itself is an operator in 78 puzzles.

## Sign / character legend (use the EXACT char for each sign — no look-alikes)

Every mathematical sign in the CoT is written with ONE fixed Unicode character; never substitute a look-alike (e.g. NEVER the MINUS SIGN `−` U+2212 for subtraction — always the HYPHEN-MINUS `-` U+002D). Token ids are from the Nemotron tokenizer (vocab 131072), for reference.

| Meaning | Char | Codepoint | Token id | Example in the CoT |
|---|---|---|---|---|
| subtraction / minus / negative sign | `-` | U+002D | 1045 | `-12 = 23-35`, `-\|63-50\|`, `31-` |
| addition / plus | `+` | U+002B | 1043 | `a+b`, `23+35` |
| multiplication | `×` | U+00D7 | 7581 | `a×b`, `02×91` |
| plus-or-minus (variant set only) | `±` | U+00B1 | 19120 | `a×b±1` |
| absolute value (open/close) | `\|` | U+007C | 1124 | `\|a-b\|` |
| concatenation | `∥` | U+2225 | 33778+1165 (2 tok) | `65∥71 = 6571` |
| noisy-operator prefix | `~` | U+007E | 1126 | `~sub = […]`, `f = [~mul, ~add]` |
| equals | `=` | U+003D | 1061 | `2209 = 47×47` |
| maps-to / becomes / reverses-to | `->` | U+002D U+003E | 1941 | `) -> f`, `136 -> 631`, `g -> ~sub`, `6 -> }6` |
| at-least (digit-count clause) | `≥` | U+2265 | 65533 | `gives ≥3 digits` |
| clause dash (punctuation) | `—` | U+2014 | 1674 | `far from 41 — no.` |
| separator (punctuation) | `,` | U+002C | 1044 | `[~add, ~sub]`, `{f = absdiff, g = mul}` |

Notes:
- **Minus is ALWAYS `-` U+002D** — the SAME token (1045) whether it means subtraction, a negative sign, or the puzzle's literal `-` operator symbol; the model sees one consistent minus token. (The GT boxed answers use U+002D, so this is also required for the metric.)
- **`->` is a single token (1941), distinct from `-` (1045)** — the maps-to arrow never collides with the minus sign. Likewise word-joiner hyphens in compound words (`right-to-left`, `digit-count`, `max-mod-min`) tokenize into `-to`/`-count`/`-mod` pieces, never the minus token.
- **Puzzle OPERATOR SYMBOLS** (e.g. `}`, `)`, `(`, `@`, `#`, `$`, `|`, …, any of ~26 ASCII punctuation chars) are whatever the puzzle gives and are displayed verbatim — they are NOT in this legend.
- `∥` (concat) is the only sign that costs 2 tokens (byte-fallback); it is still distinct from the absolute-value bar `|` (U+007C, token 1124).

## Full Deduce Process (exact wording is required)

Lines the CoT writes are in `code blocks`; **bold labels** mark the branch that applies and *italics* are notes. `[PLACEHOLDERS]` come from `question.txt`. Every step is deterministic and uses only the lines above it.

### Opening, letters, letter form (unnumbered prose)

```
I need to infer the transformation rule from the examples. The operands are given already, I only need to deduce the operators.

First, let me assign letters to each symbol:
  [OP_SYM] -> f
  [OP_SYM] -> g

I will convert all the equations to letter form:
  EX[N] [RAW_LHS][OP_SYM][RAW_RHS] = [OUT] becomes [SA] f [SB] = [OUT]
  ...
  QUERY [QSA][OP_SYM][QSB] becomes [QSA] g [QSB]
```

### Prior knowledge for this kind of question (unnumbered prose)

```
Prior knowledge for this kind of question:
1. We only consider four kinds of operations — noisy_multiplication (~mul), noisy_addition (~add), noisy_subtraction (~sub), and noisy_concatenation (~concat). They are noisy because the result is one of a fixed set of variants: the exact value, off by ±1 or ±2, and — for subtraction — also its negated or operand-reversed form. Specifically:
   ~mul = [a×b, a×b±1, a×b±2], ~add = [a+b, a+b±1, a+b±2], ~sub = [|a-b|, a-b, -|a-b|, b-a, a-b±1, a-b±2], ~concat = [ab, ba], all ordered by frequency. Any other operation is deemed exotic.
2. The arithmetic operators are distinct, but the concatenation operator may repeat.
3. If an equation contains a number with a leading 0, the operator must be ~concat, since a number with a leading 0 is invalid for an arithmetic operator.
4. If a symbol exists in the result (RHS), it must be a negative sign.
5. The reading order is either rightward or leftward (units digit first). I will check rightward first.
```

### Prior knowledge #6 — operator prior (continues the numbered list; no blank line after item 5)

```
6. For the operators in this puzzle:
```
One line per operator:

- **`+ - *` (strong prior):** `[OP_SYM], denoted as f, is usually [~add|~sub|~mul], but it can also represent [TAG_LIST], ordered by frequency`
- **arbitrary symbol, seen before:** `[OP_SYM], denoted as g, it represents [TAG_LIST], ordered by frequency`
- **arbitrary symbol, never seen:** `[OP_SYM], denoted as g, has not appeared in other puzzles`

*`[TAG_LIST]` = the symbol's families as short tags (`~mul`, `~add`, `~sub`, `~concat`), in frequency order.*

```
  Anyway, I should analyze the pattern of the examples to figure out what they are.
```

### Sign check (unnumbered prose; emitted ONLY when an operator-sign appears in an RHS)

*If an output carries an operator-symbol sign (a non-digit at its start or end), that operator is subtraction — one line per such operator:*

```
[LABEL] appears in EX[N] output side (RHS), that means it can only be a negative sign, so [LABEL] = [~sub].
```

- **no sign and the query is seen:** nothing — go straight to the traversal (§1 is always rightward, prior #5).

### §1 / §2 — Traverse the solution tree (one branch per reading)

```
Now start to traverse the solution tree, reading order = [rightward, leftward]:
§[N] reading [rightward | leftward, units digit first]:
```

*`§1` = the reading checked first; `§2` (the other reading) is opened ONLY if `§1` leaves an operator unresolved. Each branch is the same four steps:*

**§N.1 — writing equations, then log the solution state** *(in this reading's frame; for leftward the operands AND output — examples and QUERY alike — written already reversed; the QUERY has no `= [OUT]`)*

```
  §[N].1 writing equations:
    EX[M]: [LABEL]([SA], [SB]) = [OUT]
    ...
    QUERY: [LABEL]([QSA], [QSB])
    f = [~add, ~mul, ~sub, ~concat], g = [~mul, ~sub, ~add][, h = unknown].
```
*The last line logs the **solution state**: each EXAMPLE operator's candidate-family set, in its symbol's prior order. The **unseen QUERY operator** (one that appears only in the QUERY) is shown as a bare `unknown` — it has no examples, so nothing narrows it; in §N.1 ONLY, the state line ends `… g = unknown since g only appears in QUERY.` (the note appears once, not in §N.2/§N.3). A **sign-confirmed** operator collapses to `[~sub]`. (This replaces the old `Unknown:/We know` prose.)*

**§N.1 gut-check (legality).** After writing the frame equations, run two passes over the **scan list = [every example (in order), then the QUERY]** — the QUERY's operands obey prior #3 exactly like an example's. Cheapest pass first.

- **(a) tail-sign pass** (examples only — the QUERY has no RHS) — a number's sign comes first, so an output ending in a sign can't be read this way (only happens rightward; leftward always brings the sign to the front):
```
    Note: EX[M] RHS [DISP] ends in the sign '[SIGN]', but a number's sign comes first — illegal.
    This reading order is illegal; skip it completely.
```
- **(b) leading-0 pass** (prior #3) — any number in the scan list (an operand like `02`, the result like `0075`, OR a QUERY operand) with a leading 0 forces that operator to be ~concat, since arithmetic numbers have no leading 0. Fire the SAME lock-and-confirm used everywhere (`lock_concat` — identical to §N.2 and the §N.4 arithmetic lock): print "must be concat, check it", then on the NEXT line print the concat result. `[TAG]` is `EX[M]` or `QUERY`; `[NUM]` is the leading-0 number. If neither direction matches, the operator is not concat and the reading is impossible:
```
    Note: [TAG] contains [NUM], a number with a leading 0, so [LABEL] must be concat. Check [LABEL] right now:
    [LABEL]: concat_fwd=[..], concat_rev=[..]. Output=[DISP]. Neither matches, so [LABEL] is not concat.
    This reading order is illegal; skip it completely.
```
  If a concat direction MATCHES instead, `lock_concat` locks it (`FWD/REV matches -> so lock [LABEL] = concat_…`, then confirms on the operator's other examples) and the reading continues legally — the leading 0 is explained. A leading 0 never directly picks a reading; it only kills one once concat is *ruled out*. There is **no special "we already lock ~sub" shortcut** — a sign-confirmed ~sub op runs the same `lock_concat` (its signed output matches no concat direction, so it lands on the "Neither matches → illegal" branch).

  **Unseen QUERY operator:** a leading 0 on a QUERY operand whose operator appears ONLY in the QUERY has no example to fix the concat direction (fwd/rev undecidable), so it is NOT scanned here — these are relabeled `unseen_leading_zero` and left as-is.

**Output-sign display.** `[DISP]` shows the output's sign as `-` in its WRITTEN position. Left-to-right keeps it where written: `51$` → `51-`, `$51` → `-51`, `-91` → `-91` (so a tail sign is visible to the gut-check). Right-to-left reverses the output — a **leading** sign stays leading, a **trailing** sign is carried to the front — so both `51-` and `-51` → `-15` (always sign-leading, digits reversed; `_rev` never moves a leading sign to the tail). Hence a trailing-operator-sign puzzle (e.g. `26$74=51$`, `$`=~sub) is illegal rightward (`51-` ends in a sign) and resolves leftward (`-15`).

**§N.2 — concat check (under this reading)**

```
  §[N].2 quick check to see whether any operator is concatenation:
    f: concat_fwd=[A∥B], concat_rev=[B∥A]. Output=[OUT]. [Neither matches, so f is not concat. | FWD matches -> so lock f = concat_fwd. | REV matches -> so lock f = concat_rev.]
    ...
    All [N] operator(s) remaining are arithmetic. f = [~add, ~mul], g = [~sub]
```
*(if EVERY operator turned out to be concatenation, that last line instead reads `Every operator is concatenation; nothing remains to search.` and §N.3 / §N.4 are skipped for this branch.)*

**Deciding the concat direction (`lock_concat`).** The direction is read off the **first example whose two operands differ** — only there do `concat_fwd` and `concat_rev` differ, so the output can pick one. An example with **equal operands** (e.g. `66∥66`) gives `concat_fwd = concat_rev = output`, so it can't tell the directions apart; it is **deferred** and only confirmed at the end (mirrors the ~sub lock walking its variant order rather than committing to the first match):
```
    f: EX1 operands 66, 66 are equal, so concat_fwd = concat_rev = 6666 = output — this example can't decide the direction; check the next.
    f: concat_fwd=2432, concat_rev=3224. Output=3224. REV matches -> so lock f = concat_rev.
    f also appears in EX3, EX5, EX1; check EX3: 47∥91 = 4791; EX5: 55∥29 = 5529; EX1: 66∥66 = 6666. Confirmed.
```
If **no** example has distinct operands, the op is concatenation but the direction is undecidable, so it **defaults to concat_fwd** (the more frequent direction):
```
    f: concat_fwd=6666, concat_rev=6666. Output=6666. Both match; no example has distinct operands to fix the direction, so default to concat_fwd (the more frequent direction).
```

**§N.3 — digit-count prune (choose each operator's family to try first)**

```
  §[N].3 prune solution trees by the RHS digit-count before entering:
    [LABEL]: already shown to be ~sub by its output sign.                                   // if sign-confirmed
    [LABEL]: EX[M] [has|have] [d]-digit RHS ([what the digit count rules out]); ... .        // otherwise
```

*`([what the digit count rules out])`, by the RHS digit count (an operator's several examples join with `; `):*
- **1-digit:** `rules out addition and multiplication, only a subtraction-type reaches 1 digit`
- **2-digit:** `rules out multiplication, which gives ≥3 digits`
- **3-digit:** `rules out subtraction, whose magnitude is at most 2 digits`
- **4-digit:** `only multiplication of two 2-digit numbers reaches 4 digits`

then exactly ONE per operator (indented under its line):

- **one family survives (HARD):** `Only [~X] survives the digit-count filter. Try [LABEL] = [~X].`
- **distinctness forces it off a HARD-pinned family (HARD):** `Surviving candidates: [~X], [~Y]. By the distinct operator rule [LABEL] must be the remaining operation, [~Z].`
- **≥2 still possible (SOFT):** `Surviving candidates: [~X], [~Y]. Try [LABEL] = [~X] first ([~X] appears more frequently).` — `[~X]` is THIS operator's OWN most-frequent survivor (`survc[op][0]`), and the "(… appears more frequently)" note is always present. **Distinctness does NOT prune this announcement:** an UNCONFIRMED try by another operator must not delete a family here (it might fail and free that family back up). Distinctness acts only at LOCK (§N.4 — a locked family is removed from the others). So two operators that genuinely share a top BOTH announce it.

*(The unseen QUERY operator is NOT pruned or narrated here — it has no examples, so it just stays bare `unknown` in the state line.)*

```
    No further pruning found. We have f = [~X, ~Y], g = [concat_fwd] to traverse in order.   // running solution state for EVERY operator: arith ops show their surviving families (tried in this order), a locked concat op shows its variant, the unseen QUERY operator shows bare unknown
```

**§N.4 — enter the tree and search (lock each operator against ITS OWN examples)**

```
  §[N].4 enter the tree and search for an operator assignment:
    Try operators f = [~X], g = [~Y][, h = [~Z]][, [QLABEL] = unknown]:   // current combo: each unsolved op's first candidate (a concat op shows its locked variant; the unseen QUERY operator trails as unknown). Two ops MAY show the same family here (e.g. both ~add) — that's fine, distinctness resolves at lock: one fails and falls through, or one locks and the other drops it.
      We now know all the operands in EX[M]: [TA] [~X] [TB], RHS = [TG].
      ...
```
*(operands [TA], [TB], target [TG] are in the reading frame from §N.1 — not re-narrated. This is a BACKTRACKING search over the candidate sets.)*

*Per operator in the combo, replay the greedy lock-and-verify against ITS OWN examples (iterate the family's variants in order, break at the first that produces the example) — ONE of:*
- **family fits** → lock its simplest fitting variant: `[TG] = [EXPR], so lock [LABEL] = [VARIANT].` (+ if it has other examples: `[LABEL] also appears in EX[i], EX[j]; check EX[i]: [EXPR] = [VAL]; …. Confirmed.`)
- **family misses** → there are two honest shapes (NEVER `≠`: a family only dies on a number it can't reach, i.e. **far from**; exact `=` is used ONLY to lock a variant, never to reject one):
  - **first example out of band** — no variant of the family produces it. Show each representative far (one line for ~mul/~add, three for ~sub: `absolute difference` / `subtraction (a-b)` / `reverse subtraction (b-a)`), then the verdict:
    ```
        [VARIANT]: [EXPR] = [VAL], far from [TG0] — no.
        no [~X] variant gives [TG0], so [LABEL] = [~X] fails.
    ```
  - **first example forces a variant, but verification breaks later** — lock the variant the first example yields, verify on the rest, and fail on the example the family genuinely can't reach (a `far from`), then the SAME verdict:
    ```
        [TG0] = [EXPR], so lock [LABEL] = [VARIANT].
        [LABEL] also appears in [EX list]; check EX[i]: [EXPR] = [VAL]; …; EX[j]: [EXPR] = [VAL], far from [TGj].
        no [~X] variant gives [TGj], so [LABEL] = [~X] fails.
    ```
    *(Edge case — if EX[j] is reachable by a different variant rather than out of band, the verdict instead reads `[TGj] = [WEXPR] needs [W], not [VARIANT], so [LABEL] = [~X] fails.` This does not occur in the current corpus but is kept as the honest fallback.)*

*After the combo, if any op missed: advance ONLY those ops to their next candidate family — skipping any family already locked by another operator (distinct-arithmetic rule) — keep locked ops fixed, and re-print `Try operators …`:*

```
    Advance the unsolved operator(s) to the next candidate: g -> [~Y2], h -> [~Z2][ (LABEL has no candidate family left, so it remains unknown)].
```
*Repeat until every op locks or exhausts its candidates. An op that exhausts is `unknown` — named in the `Advance …` tail, or (if no op can advance at all) `No candidate family is left for [LABEL]; it remains unknown.`*

**Conclusion of step §N** *(lists every operator — the example operators AND, when the query operator is unseen, the query operator too (it stays `unknown` until guessed) — with the resolved variant or `unknown`, and the count `X of Y` where Y counts EVERY listed operator. A step concludes ONLY itself — no cross-reading comparison here (that is the Summary's job). The one exception is §1, which must decide whether to stop or open §2.):*

```
  Conclusion of step §[N]: {f = [VARIANT|unknown], g = [VARIANT|unknown][, h = ...]}, [X] of [Y] resolved.[ This is the answer. | Need to try reading [OTHER DIR].]
```

- **§1 resolved EVERY operator (X == Y, the query op included):** append `This is the answer.` — §2 is not opened.
- **§1 left ANYTHING unresolved (X < Y — a stuck example op, OR the unseen query operator, which is always still `unknown`):** append `Need to try reading [OTHER DIR].`, then open §2. The rule is UNIFORM: `X of Y` with `X < Y` always triggers the other reading. An unseen query op is counted in Y but is never resolved by the search (it has no examples), so §1 always lands here and §2 is always opened. §2 can't pin the query op either → the readings tie on it → the Summary picks the reading by EXAMPLE-operator count (§1 on a tie), and the query op is guessed after the Summary.
- **§2's conclusion is always bare** (just `…, X of Y resolved.`) — the Summary picks the winner.

### Summary / Confirm (one line)

*If §1 resolved every operator (so §2 never ran), no comparison — just confirm:*

```
Summary: Confirm reading order = [DIR], f = [VARIANT|unknown], g = [VARIANT|unknown][, h = ...]
```

*If §2 ran and neither reading was ruled illegal, the Summary states which reading wins, then confirms: whoever resolved MORE operators wins, §1 on a tie. If one reading was declared illegal by the §N.1 gut-check, the other is the only legal reading and wins outright.*

```
Summary: [VERDICT] Confirm reading order = [DIR], f = [VARIANT|unknown], g = [VARIANT|unknown][, h = ...]
```

`[VERDICT]`, by case:

- **§2 resolved more:** `§2 solved more operators than §1, so §2 is preferred.`
- **§1 resolved more:** `§1 solved more operators than §2, so §1 is preferred.`
- **tie:** `§1 and §2 both solved [N] operator(s), §1 is preferred.`
- **one reading ruled illegal by the gut-check:** `§[X] is illegal, so §[Y] is the only legal reading.`

### Query answer (apply the resolved/guessed operator to the QUERY, in the confirmed reading frame)

*`[QA], [QB]` are the reading-frame query operands (reversed iff leftward — the same ones written in §N.1). The operator's meaning is already stated (in the Summary for a determined op, or in the guess reasoning below), so this step is just the computation.*

```
Now solve the QUERY [LABEL]([QA], [QB]):
[EXPR] = [RAW][; reverse the digits [RAW] -> [FINAL]]; answer = [FINAL]
```

- `[EXPR]` applies the resolved/guessed operator to `[QA], [QB]` (e.g. `57+79`, `|81-20|`, `10×89-1`, `57∥79`).
- the `; reverse the digits [RAW] -> [FINAL]` clause appears **only under leftward** (the output is written reversed too); under rightward it is omitted and `answer = [RAW]` (which equals `[FINAL]`).
- **The `answer =` line shows `[FINAL]` — the raw signed value (e.g. `-6`), NOT the operator-symbol form.** Re-attaching the operator's sign (`-6 -> 6}`) is a LATER step (the re-attach line below); a step may never pre-fill a value a later step produces. `[ANS]` (the re-attached form) appears only on that re-attach line and in `\boxed{}`.

*(No "ambiguity caveat." The §N.4 search locks the FIRST consistent variant and the query just applies it — the trace never tests other variants, so there is nothing to caveat. Probing other variants to flag ambiguity would be out-of-band peeking; do not add it.)*

**query operator CANNOT be deduced** — two cases, by whether the operator appears in the examples:

**(c1) UNSEEN query** (the operator never appears in the examples) — lead line, then a **binary rule**:

```
Query operator g is not determined by the examples, so I need to guess it.
```

- **at least one operator was solved arithmetically** → guess arithmetic. Intro `At least one operator here can be solved arithmetically, so I'll guess g is an arithmetic operator too`, then the distinctness / symbol-prior clause:
  - **forced by distinctness:** `[INTRO]. Since [~X] [and [~Y]] [has|have] appeared in the puzzle, by the distinct operator rule g must be the remaining operation, [~Z].`
  - **narrowed but not forced:** `[INTRO]. Since [~X] [has|have] appeared in the puzzle, by the distinct operator rule g is [~Y] or [~Z]; By frequency order, [~Z] is the pick.`
  - **every arithmetic family already used:** `[INTRO]. Since [~X] and [~Y] [has|have] appeared in the puzzle, every arithmetic operation is already used, so I take the most likely, [~Z].`
- **no operator was solved arithmetically** → guess **concatenation** (the family), then commit the forward order: `No operator here can be solved arithmetically, so I guess the unseen operator g is ~concat. Let me use g = concat_fwd.` The `Now solve the QUERY` line then uses `a∥b`. This is the guess, not a fallback.

For (c1), the guessed operator's value is what `[EXPR]` uses in the `Now solve the QUERY` line (arithmetic base op `a+b` / `|a-b|` / `a×b`, or `a∥b` / `b∥a` for concat).

**(c2) EXOTIC query** (the operator appears, but every modeled family was disproven for it in **every LEGAL reading**): the four-family search is exhausted → the operator is exotic. `[WHERE]` in the verdict below names only the reading(s) actually SEARCHED — `either §1 or §2` when both were legal, else just the one legal `§N` (a reading ruled illegal by the §N.1 gut-check was never searched, so it is never claimed; this is driven by the same legality state as the Summary). The search/model stays four-family (we do NOT add operations to it); the exotic branch applies one **hardcoded guess**, `max mod min`. This is OBSERVED, not derived — across the exotic puzzles the out-of-system operator is almost always `max(a,b) mod min(a,b)`, so we hardcode that as the exotic guess (same status as boxing any fixed number, just a better-informed pick; it is applied in the confirmed reading frame and can be wrong). No verification of the examples, no sign re-attach.

```
QUERY operator g is not solvable in [WHERE], so g is an exotic operator.
The one operation that exists outside the system is max-mod-min(a,b) = max(a, b) mod min(a, b)
Now solve the QUERY g([QA], [QB]):
max([QA], [QB]) mod min([QA], [QB]) = [MX] mod [MN] = [R][; reverse the digits [R] -> [FINAL]]; answer = [FINAL]

I will now return the answer in \boxed{}, the answer is
\boxed{[ANS]}
```

`[QA],[QB]` are the confirmed-frame query operands; `[MX],[MN]` = max/min of them; `[R]` = `[MX] mod [MN]`; under leftward the `; reverse the digits [R] -> [FINAL]` clause is added and `[ANS]=[FINAL]`, else `[ANS]=[R]`. Hardcoded guess, not a deduction — it is wrong on the few exotic operators that are not max-mod-min (e.g. cross-multiply).

### Sign re-attach + boxed

*Sign re-attach only if the result is negative (form matches the example outputs):*

```
The result is negative, so it is written with the [operator-symbol suffix|operator-symbol prefix|leading minus]: [FINAL] -> [ANS].
```

*Suffix (always, two lines):*

```
I will now return the answer in \boxed{}, the answer is
\boxed{[ANSWER]}
```
