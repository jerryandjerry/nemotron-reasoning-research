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
4. **Never skip a step because the outcome is "already known".** When the solver notices a condition (a leading 0, an rhs sign, a digit count), it must print the step that acts on it and *then* print that step's result — step by step — even when the result turns out fine. It must not silently run a check and move on. (E.g. a leading-0 number ALWAYS fires the gut-check: print "…must be concat, check it", then on the next line print the concat result — match → lock concat, or neither → illegal.)

Write the log honestly and do not fear mistakes — the reviewers recompute every line and will surface any error.

## Required Wording

- **f, g, h:** operators are lettered by order of first appearance.
- **Operators we search:** ~add = [a+b, a+b±1, a+b±2], ~mul = [a×b, a×b±1, a×b±2], ~sub = [|a-b|, a-b, -|a-b|, b-a, a-b±1, a-b±2, b-a±1, b-a±2], and ~concat [ab, ba], all ordered by frequency. Any other operation is deemed **exotic** — never fabricated.
- **Reading order is DISCOVERED, not precomputed.** Reading order = [rightward, leftward] (rightward = read the tens digit first, the normal reading; leftward = read the units digit first). Under **leftward** every operand AND the output is read with its digits reversed; in §N.1 the equations are written **already reversed** (e.g. `f(25, 34) = 9` for `52{43 = 9`), and §N.4 then uses those reversed numbers directly — it does not re-explain the reversal. **Leftward has two modes, declared in the §2 header from where the sign sits in the rhs:** *on digit only* (reverse the digits; a front sign stays in front) and *fully* (reverse the WHOLE rhs, so a back sign flips to the front and is read first — which is what makes a back-sign legal under leftward).
- **Distinctness is prior knowledge (item 2):** the arithmetic operators are distinct; the concatenation operator may repeat. HARD rule for pruning, NOT re-appended to the "Unknown:" line.
- **Rejections say "no", not "backtrack"; NEVER `≠`.** A candidate dies by showing its computed value against the rhs, then `— no`: for ~add/~mul the explicit diff `[TG]-[BASE] = [D]; |[D]| > 2, beyond ±2 — no`; for ~sub the value vs rhs `[EXPR] = [VAL]; rhs [TG] — no`. An exact match LOCKS a variant (`match, so lock`), never rejects one. The words "fits" and "far from" are not used.
- **An undetermined operator is "remains unknown".**
- **No function names — bare formulas only.** Operators lock to their FORMULA (`a+b-1`, `a-b`, `-|a-b|`, `a×b+1`, `b∥a`), never an opaque name (`add_minus1`, `sub_signed`, `neg_absdiff`, `concat_rev`); the model can't learn the names. This holds everywhere: lock lines, the §N.4 combo, Conclusion/Summary, and the query function-def `[LABEL](a, b) = [FORMULA]`.
- **Indexing — `§` numbers the reading branches only.** The setup (opening, letters, equations, prior knowledge, operator prior, sign check) is **unnumbered prose**. The search tree is `§`-numbered: `§1` is always rightward (checked first, prior #5), `§2` is leftward, opened ONLY if `§1` leaves an operator unresolved (or is ruled illegal). Each branch has four sub-steps `§N.1`–`§N.4` and closes with `Conclusion of step §N: {…}, X of Y resolved.` — bare for §2, while §1 appends its stop/continue decision (`This is the answer.` if X == Y, else `Need to try reading [other].`). A reading whose §N.1 gut-check finds a malformed number (a trailing sign on the rhs, or a leading 0 on any operand/result of a non-concat operator) is declared **illegal and skipped entirely**. Otherwise the winner is chosen on a `Summary:` line (the reading that resolves MORE operators wins, §1 on a tie) that also confirms the reading + every operator's meaning; if §1 already resolved everything there is no comparison, just `Summary: Confirm reading order = …`. Then the query answer (`Now solve the QUERY …`). We use `§` (not `#`) because operator symbols span essentially all ASCII punctuation — `#` itself is an operator in 78 puzzles.

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
| clause dash (punctuation) | `—` | U+2014 | 1674 | `rhs -7 — no.`, `beyond ±2 — no.` |
| separator (punctuation) | `,` | U+002C | 1044 | `[~add, ~sub]`, `{f = |a-b|, g = a×b}` |

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

Denote the operators as unknown functions:
  [OP_SYM] -> f
  [OP_SYM] -> g

I will convert all the equations to letter form:
  EX[N] [RAW_LHS][OP_SYM][RAW_rhs] = [OUT] becomes [SA] f [SB] = [OUT_LET]
  ...
  QUERY [QSA][OP_SYM][QSB] becomes [QSA] g [QSB]
```
*This is a **mechanical scan-and-replace**: every operator symbol is swapped for its letter, **including one that appears in the output** (`86[93 = [7` → `86 g 93 = g7`; `44-28 = -83` → `44 f 28 = f83`). A symbol in the rhs is NOT yet known to be a negative sign — that meaning is derived only later (prior #4 → sign-check), so it is written as its letter here, never as `-7`. Forward-only.*
```
```

### Prior knowledge for this kind of question (unnumbered prose)

```
Prior knowledge for this kind of question:
1. We only consider four kinds of operations — noisy_addition (~add), noisy_subtraction (~sub), noisy_multiplication (~mul), and noisy_concatenation (~concat). They are noisy because the result has a tolerance of ±2 from the exact value and — for subtraction — may also appear as its absolute value or operand-reversed form. Specifically:
   ~add = [a+b, a+b±1, a+b±2], ~sub = [|a-b|, a-b, -|a-b|, b-a, a-b±1, a-b±2, b-a±1, b-a±2], ~mul = [a×b, a×b±1, a×b±2], ~concat = [ab, ba], all ordered by frequency. The absolute forms |a-b|, -|a-b| carry no noise — they are themselves the unsigned reading of the signed difference; the ±2 tolerance applies to the signed forms a-b, b-a. Any other operation is deemed exotic.
2. The arithmetic operators are distinct, but the concatenation operator may repeat.
3. If an equation contains a number with a leading 0, the operator must be ~concat, since a number with a leading 0 is invalid for an arithmetic operator.
4. If a symbol exists in the result (rhs), it must be a negative sign.
5. The reading order is either rightward or leftward (units digit first). I will check rightward first.
```

### (No operator-prior block — the numbered list ends at item 5)

There is **no "6. For the operators in this puzzle:" block.** The family try-order is a single **fixed global order `[~add, ~sub, ~mul, ~concat]`**, used for every operator in the §N.1 state line and the §N.4 search — NOT a per-symbol frequency prior. A symbol's identity matters only for the **unseen-operator guess** (see Query answer): a `+`/`-`/`*` query operator is guessed `~add`/`~sub`/`~mul`; arbitrary symbols fall back to the global order.

### Sign check (unnumbered prose; emitted ONLY when an operator-sign appears in an rhs)

*If an output carries an operator-symbol sign (a non-digit at its start or end), that operator is subtraction — one line per such operator:*

```
[LABEL] appears in EX[N] output side (rhs), that means it can only be a negative sign, so [LABEL] = [~sub].
```

- **no sign and the query is seen:** nothing — go straight to the traversal (§1 is always rightward, prior #5).

### §1 / §2 — Traverse the solution tree (one branch per reading)

```
Now start to traverse the solution tree, reading order = [rightward, leftward]:
§[N] reading [rightward | leftward on digit only, so the negative sign stays in the front | leftward fully, so the negative sign flips to the front | leftward on digit only, no negative sign in rhs]:
```

*`§1` = the reading checked first; `§2` (the other reading) is opened ONLY if `§1` leaves an operator unresolved. Each branch is the same four steps:*

**§N.1 — writing equations, then log the solution state** *(in this reading's frame; for leftward the operands AND output — examples and QUERY alike — written already reversed; the QUERY has no `= [OUT]`)*

```
  §[N].1 writing equations:
    EX[M]: [LABEL]([SA], [SB]) = [OUT]
    ...
    QUERY: [LABEL]([QSA], [QSB])
    f = [~add, ~sub, ~mul, ~concat], g = [~add, ~sub, ~mul, ~concat][, h = unknown].
```
*The last line logs the **solution state**: each EXAMPLE operator's candidate-family set, in the fixed global order `[~add, ~sub, ~mul, ~concat]`. The **unseen QUERY operator** (one that appears only in the QUERY) is shown as a bare `unknown` — it has no examples, so nothing narrows it; in §N.1 ONLY, the state line ends `… g = unknown since g only appears in QUERY.` (the note appears once, not in §N.2/§N.3). A **sign-confirmed** operator collapses to `[~sub]`. (This replaces the old `Unknown:/We know` prose.)*

**§N.1 gut-check (legality).** After writing the frame equations, run two passes over the **scan list = [every example (in order), then the QUERY]** — the QUERY's operands obey prior #3 exactly like an example's. Cheapest pass first.

- **(a) tail-sign pass** (examples only — the QUERY has no rhs) — a number's sign comes first, so an output ending in a sign can't be read this way (only happens rightward; leftward always brings the sign to the front):
```
    Note: EX[M] rhs [DISP] ends in the sign '[SIGN]', but a number's sign comes first — illegal.
    This reading order is illegal; skip it completely.
```
- **(b) leading-0 pass** (prior #3) — any number in the scan list (an operand like `02`, the result like `0075`, OR a QUERY operand) with a leading 0 forces that operator to be ~concat, since arithmetic numbers have no leading 0. Fire the SAME lock-and-confirm used everywhere (`lock_concat` — identical to §N.2 and the §N.4 arithmetic lock): print "must be concat, check it", then on the NEXT line print the concat result. `[TAG]` is `EX[M]` or `QUERY`; `[NUM]` is the leading-0 number. If neither direction matches, the operator is not concat and the reading is impossible:
```
    Note: [TAG] contains [NUM], a number with a leading 0, so [LABEL] must be concat. Check [LABEL] right now:
    [LABEL]: a∥b=[..], b∥a=[..]. rhs=[DISP]. Neither matches, so [LABEL] is not concat.
    This reading order is illegal; skip it completely.
```
  If a concat direction MATCHES instead, `lock_concat` locks it (`a∥b/b∥a matches -> so lock [LABEL] = a∥b`, then confirms on the operator's other examples) and the reading continues legally — the leading 0 is explained. A leading 0 never directly picks a reading; it only kills one once concat is *ruled out*. Even a sign-confirmed ~sub op runs `lock_concat` **here** (the leading 0 forces the check, prior #3 — its signed output matches no concat direction, so it lands on the "Neither matches → illegal" branch). (In the plain §N.2 concat check, by contrast, a sign-confirmed ~sub is **skipped** — see §N.2.)

  **Unseen QUERY operator:** a leading 0 on a QUERY operand whose operator appears ONLY in the QUERY has no example to fix the concat direction (fwd/rev undecidable), so it is NOT scanned in §N.1. It is handled later at the GUESS step (Query answer (c1)): if the query op is guessed ARITHMETIC, a leading-0 operand is invalid for arithmetic (rule 3) → the confirmed reading is illegal → flip to the other reading; if guessed concat, a leading 0 is fine, so it is left as-is.

**Output-sign display.** `[DISP]` shows the output's sign as `-` in its WRITTEN position. Left-to-right keeps it where written: `51$` → `51-`, `$51` → `-51`, `-91` → `-91` (so a tail sign is visible to the gut-check). Right-to-left reverses the output — a **leading** sign stays leading, a **trailing** sign is carried to the front — so both `51-` and `-51` → `-15` (always sign-leading, digits reversed; `_rev` never moves a leading sign to the tail). Hence a trailing-operator-sign puzzle (e.g. `26$74=51$`, `$`=~sub) is illegal rightward (`51-` ends in a sign) and resolves leftward (`-15`).

**§N.2 — concat check (under this reading)**

```
  §[N].2 quick check to see whether any operator is concatenation:
    f: a∥b=[A∥B], b∥a=[B∥A]. rhs=[OUT]. [Neither matches, so f is not concat. | a∥b matches -> so lock f = a∥b. | b∥a matches -> so lock f = b∥a.]
    [LABEL]: already ~sub by its output sign, skip the concat check.   // a sign-confirmed ~sub op: a signed output can't be concat, so it is NOT re-tested here
    ...
    f = [~add, ~sub, ~mul], g = [~sub]
```
*(the state line just shows the remaining candidates — there is no "All N operator(s) remaining are arithmetic" prose, since not-concat does not prove arithmetic (an op may turn out exotic). If EVERY operator turned out to be concatenation, the last line instead reads `Every operator is concatenation; nothing remains to search.` and §N.3 / §N.4 are skipped for this branch.)*

**QUERY-OPERATOR CONCAT HALT (solver early-exit — a real control-flow stop, not a print).** The moment the operator that gets locked concat **is the QUERY operator** (in this §N.2, or in the §N.1 leading-0 gut-check), the solver has all it needs and **halts**: it does NOT run §N.3/§N.4 and does NOT try the other reading. (Gate = *the query op locks concat*; example count is irrelevant — a single example has an empty confirm, multiple examples show the `Confirmed.` line first.) Why it's sound: a concatenation is **reading-invariant** (reversing the reading flips both the lock direction `a∥b`↔`b∥a` and the operands, so the query answer is the same string either way), and the other operators never affect the query. The halt line, then a bare conclusion (only the query op resolved, the rest `unknown`), then a jump Summary:
```
    [for a multi-example query op, the lock_concat `Confirmed.` line appears here first]
    Note: [LABEL] is the QUERY operator, and it has been locked as [a∥b|b∥a]. At this point other unknowns are irrelevant, because the answer is simply the concatenation of the query operands. Proceed to answer directly.
  Conclusion of step §[N]: {[LABEL] = [a∥b|b∥a], others = unknown}, 1 of [Y] resolved.
Summary: Query operator [LABEL] = [a∥b|b∥a], no need to solve the other unknowns; Confirm reading order = [DIR], [dctstr]
```
Then the query answer applies the concat directly. (If the query op is NOT locked concat — i.e. the concat check says "not concat" — there is no halt; the solver continues normally.)

**Deciding the concat direction (`lock_concat`).** The direction is read off the **first example whose two operands differ** — only there do `a∥b` and `b∥a` differ, so the output can pick one. An example with **equal operands** (e.g. `66∥66`) gives `a∥b = b∥a = output`, so it can't tell the directions apart; it is **deferred** and only confirmed at the end (mirrors the ~sub lock walking its variant order rather than committing to the first match):
```
    f: EX1 operands 66, 66 are equal, so a∥b = b∥a = 6666 = output — this example can't decide the direction; check the next.
    f: a∥b=2432, b∥a=3224. rhs=3224. b∥a matches -> so lock f = b∥a.
    f also appears in EX3, EX5, EX1; check EX3: 47∥91 = 4791; EX5: 55∥29 = 5529; EX1: 66∥66 = 6666. Confirmed.
```
If **no** example has distinct operands, the op is concatenation but the direction is undecidable, so it **defaults to a∥b** (the more frequent direction):
```
    f: a∥b=6666, b∥a=6666. rhs=6666. Both match; no example has distinct operands to fix the direction, so default to a∥b (the more frequent direction).
```

**§N.3 — digit-count prune (choose each operator's family to try first)**

```
  §[N].3 prune solution trees by the rhs digit-count before entering:
    [LABEL]: already shown to be ~sub by its output sign.                                   // if sign-confirmed
    [LABEL]: EX[M] [has|have] [d]-digit rhs ([what the digit count rules out]); ... .        // otherwise
```

*`([what the digit count rules out])`, by the rhs digit count (an operator's several examples join with `; `):*
- **1-digit:** `rules out addition and multiplication, only a subtraction-type reaches 1 digit`
- **2-digit:** `rules out multiplication, which gives ≥3 digits`
- **3-digit:** `rules out subtraction, whose magnitude is at most 2 digits`
- **4-digit:** `only multiplication of two 2-digit numbers reaches 4 digits`

then exactly ONE per operator (indented under its line):

- **one family survives (HARD):** `Only [~X] survives the digit-count filter. Try [LABEL] = [~X].`
- **distinctness forces it off a HARD-pinned family (HARD):** `Surviving candidates: [~X], [~Y]. By the distinct operator rule [LABEL] must be the remaining operation, [~Z].`
- **≥2 still possible (SOFT):** `Surviving candidates: [~X], [~Y]. Try [LABEL] = [~X] first ([~X] comes first in the operation order).` — `[~X]` is the first surviving family in the fixed global order (`survc[op][0]`). **Distinctness does NOT prune this announcement:** an UNCONFIRMED try by another operator must not delete a family here (it might fail and free that family back up). Distinctness acts only at LOCK (§N.4 — a locked family is removed from the others). So two operators that genuinely share a top BOTH announce it.

*(The unseen QUERY operator is NOT pruned or narrated here — it has no examples, so it just stays bare `unknown` in the state line.)*

```
    No further pruning found. We have f = [~X, ~Y], g = [a∥b] to traverse in order.   // running solution state for EVERY operator: arith ops show their surviving families (tried in this order), a locked concat op shows its variant, the unseen QUERY operator shows bare unknown
```

**§N.4 — enter the tree and search (lock each operator against ITS OWN examples)**

```
  §[N].4 enter the tree and search for an operator assignment:
    Try operators f = [~X], g = [~Y][, h = [~Z]][, [QLABEL] = unknown]:   // current combo: each unsolved op's first candidate (a concat op shows its locked variant; the unseen QUERY operator trails as unknown). Two ops MAY show the same family here (e.g. both ~add) — that's fine, distinctness resolves at lock: one fails and falls through, or one locks and the other drops it.
      We now know all the operands in EX[M]: [~X]([TA], [TB]) = [TG].   // the family-level equation to solve (function notation), in the reading frame
      ...
```
*(operands [TA], [TB], rhs [TG] are in the reading frame from §N.1 — not re-narrated. This is a BACKTRACKING search over the candidate sets.)*

*Per operator in the combo, replay the greedy lock-and-verify against ITS OWN examples (iterate the family's variants in order, break at the first that produces the example) — ONE of:*
- **family fits** → lock the variant, computed **left-to-right** (the log is a forward computation, not a result-first assertion):
  - **~add / ~mul (±2-offset families):** compute the family BASE left-to-right, then the variant falls out of the diff `target - base`:  `[OP]: [A]+[B] = [BASE]; rhs [TG]; [TG]-[BASE] = [D], so lock [LABEL] = [FORMULA].` (e.g. `addition: 29+95 = 124; rhs 123; 123-124 = -1, so lock f = a+b-1.`; when [D]=0 the diff line is replaced by `match`, e.g. `30+58 = 88; rhs 88; match, so lock f = a+b.`). **Multiplication is computed by partial products** (split the operand with a non-zero units digit into tens+units): `multiplication: 85×97 = (80+5)×97 = 80×97 + 5×97 = 7760 + 485 = 8245; rhs …`. `[D]` is the signed diff (`0`, `+1`, `-1`, `+2`, `-2`); `[D]=0` → base variant.
  - **~sub (structural variants):** first print the order line `for ~sub, we check these in order [|a-b|, a-b, -|a-b|, b-a], and lock the first match.`, then a **left-to-right walk** over the variant order `|a-b| → a-b → -|a-b| → b-a → a-b±1 → a-b±2 → b-a±1 → b-a±2`. The absolute forms `|a-b|`, `-|a-b|` carry no noise; only the signed forms `a-b`, `b-a` get the ±k offsets. Each variant is computed and compared to the rhs **exactly** (no diff/tolerance line — the ±k are their own candidates): `[EXPR] = [VAL]; rhs [TG] — no.` for a mismatch, and the first that equals the rhs locks with **`match`**: `[EXPR] = [VAL]; rhs [TG]; match, so lock [LABEL] = [FORMULA].` (e.g. `|86-93| = 7; rhs -7 — no.` then `86-93 = -7; rhs -7; match, so lock g = a-b.`). No "far from" — the value and the rhs are both shown, so the mismatch is explicit.
  - **confirm (all families, unchanged):** if the op has other examples, append `[LABEL] also appears in EX[i], EX[j]; check EX[i]: [EXPR] = [VAL]; …. Confirmed.` (the variant-expression form).
- **family misses** → NEVER `≠`; a family only dies on a number it can't reach. Two cases by family type:
  - **~add / ~mul:** compute the base, the diff exceeds the ±2 tolerance:
    ```
        [OP]: [A]+[B] = [BASE]; rhs [TG]; [TG]-[BASE] = [D]; |[D]| > 2, beyond ±2 — no.
        no [~X] variant gives [TG], so [LABEL] = [~X] fails.
    ```
    (multiplication shows partial products as in the lock case; if EX1 is in band but a LATER example needs a different offset, lock EX1's variant then break on that example with its own `… beyond ±2 — no` or `… needs [W], not [V0] …` verdict.)
  - **~sub:** print the order line, then walk the 4 structural variants in order (`|a-b|`, `a-b`, `-|a-b|`, `b-a`), each compared to the rhs exactly, then the verdict (the ±k offsets miss too):
    ```
        for ~sub, we check these in order [|a-b|, a-b, -|a-b|, b-a], and lock the first match.
        |[A]-[B]| = [VAL]; rhs [TG0] — no.
        [A]-[B] = [VAL]; rhs [TG0] — no.
        -|[A]-[B]| = [VAL]; rhs [TG0] — no.
        [B]-[A] = [VAL]; rhs [TG0] — no.
        no [~X] variant gives [TG0], so [LABEL] = [~X] fails.
    ```
  - **first example locks a variant, but a later example breaks it** (~sub) — lock the entry's variant (left-to-right), verify on the rest, fail on the example it can't reach, then the verdict:
    ```
        [EXPR] = [VAL]; rhs [TG0]; match, so lock [LABEL] = [FORMULA].
        [LABEL] also appears in [EX list]; check EX[i]: [EXPR] = [VAL]; …; EX[j]: [EXPR] = [VAL]; rhs [TGj] — no.
        no [~X] variant gives [TGj], so [LABEL] = [~X] fails.
    ```
    *(Edge case — if EX[j] is reachable by a different variant, the verdict instead reads `[WEXPR] = [TGj] needs [W], not [FORMULA], so [LABEL] = [~X] fails.`)*

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

*If §2 ran and neither reading was ruled illegal, the Summary decides the reading by a forward, short-circuiting **gate cascade** — each gate's clause is appended to the verdict ONLY when that gate is actually reached, and the cascade STOPS at the first gate that decides (the verdict is accumulated clause-by-clause, never composed in one shot):*

*1. **Gate 1 — query operator:** if exactly ONE reading solved the QUERY operator, take it, and stop (Gates 2/3 are never evaluated or logged).*
*2. **Gate 2 — count:** else (both readings solved it, or neither did), whoever resolved MORE operators wins.*
*3. **Gate 3 — default:** else (count tie), §1.*

*If one reading was declared illegal by the §N.1 gut-check, the other is the only legal reading and wins outright (no cascade).*

```
Summary: [VERDICT] Confirm reading order = [DIR], f = [VARIANT|unknown], g = [VARIANT|unknown][, h = ...]
```

`[DIR]` is `rightward`, `leftward on digit only`, or `leftward fully` — the leftward mode (from where the sign sits in the rhs) is carried into the confirm line. `[VERDICT]`, by case:

- **Gate 1 decides** (query op solved in exactly one reading): `§[X] solved the QUERY operator [LABEL], which §[Y] did not, so §[X] is preferred.` (Gates 2/3 never reached, so never logged.)
- **Gate 1 inconclusive → Gate 2 decides:** `[Both readings solve|Neither reading solves] the QUERY operator [LABEL]; §[X] solved more operators, so §[X] is preferred.`
- **Gate 1 inconclusive → Gate 2 tie → Gate 3:** `[Both readings solve|Neither reading solves] the QUERY operator [LABEL]; §1 and §2 solved the same number of operators, so §1 is preferred.`
- **one reading ruled illegal by the gut-check:** `§[X] is illegal, so §[Y] is the only legal reading.`

### Query answer (apply the resolved/guessed operator to the QUERY, in the confirmed reading frame)

*`[QA], [QB]` are the reading-frame query operands (reversed iff leftward — the same ones written in §N.1). The operator's meaning is already stated (in the Summary for a determined op, or in the guess reasoning below), so this step is just the computation.*

```
Now solve the QUERY [LABEL]([QA], [QB]), applying [LABEL](a, b) = [FORMULA],
[LABEL]([QA], [QB]) = [EXPR] = [RAW][; reading order is [MODE], so the result [RAW] -> [STANDIN]]; answer = [FINAL]   // positive ends here; a NEGATIVE result omits 'answer =' and runs step 2 (sign re-attach) below
```

- the first line **restates the locked operator as a function def** `[LABEL](a, b) = [FORMULA]` (e.g. `f(a, b) = b∥a`, `g(a, b) = a-b`) — a **bare formula**, no variant name (the model can't learn names like `sub_signed`) so applying it is not a memory test (no operand-order / concat-direction slips).
- `[EXPR]` applies it to `[QA], [QB]` (e.g. `57+79`, `|81-20|`, `20∥39`); the line begins `[LABEL]([QA], [QB]) =` so the substitution is explicit. **A multiplication query is computed by partial products** (split into tens+units), with any ±k offset carried through each stage: `g(53, 49) = 53×49 = (50+3)×49 = 50×49 + 3×49 = 2450 + 147 = 2597`, `h(64, 56) = 64×56+1 = (60+4)×56 + 1 = 60×56 + 4×56 + 1 = 3360 + 224 + 1 = 3585`.
- the `; reading order is [MODE], so the result [RAW] -> [STANDIN]` clause appears **only under leftward** and names the mode discovered in §2: `leftward on digit only` (reverse the digits, the leading minus stays in front) or `leftward fully` (reverse the WHOLE rhs, so a back-sign flips to the front). `[STANDIN]` is the reversed result with the stand-in minus on the side the mode dictates (front for digit-only, back for fully); for a positive result `[STANDIN]=[FINAL]`. Under rightward the clause is omitted (no reversal).
- **A positive result finalizes here** with `; answer = [FINAL]` (rightward: `answer = [RAW]`). **A negative result does NOT print `answer =` here** — step 1 stops at the reversed `[STANDIN]`, and the operator symbol is attached as a separate step 2 (below); a step may never pre-fill a value a later step produces. `[ANS]` (the operator-symbol form) appears only on that step-2 line and in `\boxed{}`. The mode named in step 1 is what makes `[STANDIN]` derivable (e.g. `leftward fully` → `-30 -> 03-`, `leftward on digit only` → `-39 -> -93`).

*(No "ambiguity caveat." The §N.4 search locks the FIRST consistent variant and the query just applies it — the trace never tests other variants, so there is nothing to caveat. Probing other variants to flag ambiguity would be out-of-band peeking; do not add it.)*

**query operator CANNOT be deduced** — two cases, by whether the operator appears in the examples:

**(c1) UNSEEN query** (the operator never appears in the examples) — lead line, then a **binary rule**:

```
Query operator g is not determined by the examples, so I need to guess it.
```

- **at least one operator was solved arithmetically** → guess arithmetic. Intro `At least one operator here can be solved arithmetically, so I'll guess g is an arithmetic operator too`, then the distinctness / symbol-prior clause:
  - **forced by distinctness:** `[INTRO]. Since [~X] [and [~Y]] [has|have] appeared in the puzzle, by the distinct operator rule g must be the remaining operation, [~Z].`
  - **narrowed but not forced:** `[INTRO]. Since [~X] [has|have] appeared in the puzzle, by the distinct operator rule g is [~Y] or [~Z]; [PICK].` where `[PICK]` is the symbol prior when the symbol's canonical family is free — `the symbol '[OP]' usually means [~Z], so [~Z] is the pick` (+ → ~add, - → ~sub, * → ~mul) — else the global default `[~Z] comes first in the default order, so it is the pick`.
  - **every arithmetic family already used:** `[INTRO]. Since [~X] and [~Y] [has|have] appeared in the puzzle, every arithmetic operation is already used, so I take the most likely, [~Z].`
- **no operator was solved arithmetically** → guess **concatenation** (the family), then commit the forward order: `No operator here can be solved arithmetically, so I guess the unseen operator g is ~concat. Let me use g = a∥b.` The `Now solve the QUERY` line then uses `a∥b`.  This is the guess, not a fallback.
  - **leading-0 query operand after an ARITHMETIC guess:** once the query op is guessed arithmetic, concat is ruled out, so a leading-0 operand in the confirmed reading can't be explained — it is invalid for an arithmetic operator (rule 3), so the confirmed reading is illegal for the query. Flip to the other reading (only when it is legal AND removes the leading 0): `Note: QUERY contains [NUM], a number with a leading 0, which is invalid for an arithmetic operator, so the [DIR] reading is illegal here; read it [OTHER], where the operands are [QA], [QB].` then solve there. (A *concat* guess does NOT flip — a leading 0 is valid for concatenation.)

For (c1), the guessed operator's value is what `[EXPR]` uses in the `Now solve the QUERY` line (arithmetic base op `a+b` / `|a-b|` / `a×b`, or `a∥b` / `b∥a` for concat).

**(c2) EXOTIC query** (the operator appears, but every modeled family was disproven for it in **every LEGAL reading**): the four-family search is exhausted → the operator is exotic. The only out-of-system operation we model is `max-mod-min(a,b) = max(a,b) mod min(a,b)` (OBSERVED — across the exotic puzzles the out-of-system operator is almost always max mod min). It is now **locked and VERIFIED against the examples**, starting with the confirmed reading: if it reproduces every example there, that reading stands; if it does NOT *and both readings are legal*, the other reading is checked, and if max-mod-min fits there the **actual reading is revised to it** — an exotic operator can overturn the four-family reading (hence the disclaimer). `[WHERE]` names only the reading(s) actually SEARCHED by the four-family pass — `either §1 or §2` when both were legal, else the one legal `§N` (a reading ruled illegal by the §N.1 gut-check was never searched, and is never the revision target). If max-mod-min fits NEITHER legal reading the puzzle is genuinely unsolvable (none in the current corpus).

```
QUERY operator g is not solvable by the four families in [WHERE], so g is an exotic operator. Note: an exotic operator may change all the rule we deduce above.
The only operation that exists outside the system is max-mod-min(a,b) = max(a, b) mod min(a, b)
Lock g = max-mod-min, and verify it against the examples[, starting with] the confirmed reading ([DIR]):
  EX[i]: max([A], [B]) mod min([A], [B]) = [MX] mod [MN] = [V]; rhs [TG]; match.        // "— no." if the confirmed reading fails
  [g also appears in EX..; check EX..: [MX] mod [MN] = [V]; .. . Confirmed.]            // confirm tail when >1 example
[The confirmed reading does not match, so check the other reading ([OTHER]):           // ONLY when confirmed fails AND both readings legal
  EX[i]: max([A], [B]) mod min([A], [B]) = [MX] mod [MN] = [V]; rhs [TG]; match.
  g also appears in EX..; check ... . Confirmed.
So the actual reading is [OTHER].]
Now solve the QUERY g([QA], [QB]):
g([QA], [QB]) = max([QA], [QB]) mod min([QA], [QB]) = [MX] mod [MN] = [R][; reading order is [MODE], so the result [R] -> [FINAL]]; answer = [FINAL]

I will now return the answer in \boxed{}, the answer is
\boxed{[ANS]}
```

`[QA],[QB]` are the query operands in the FINAL (verified) reading; `[MX],[MN]` = max/min of them; `[R]` = `[MX] mod [MN]`; under leftward the `reading order is [MODE], so the result [R] -> [FINAL]` clause is added and `[ANS]=[FINAL]`, else `[ANS]=[R]`. max-mod-min is the one modeled out-of-system operation; the verification step makes it earned (not asserted) and lets its fit revise the reading.

### Sign re-attach + boxed

*Step 2 for a negative result — map the stand-in minus to the operator symbol. The §2 mode already fixed which side the sign sits on, so this is an IN-PLACE swap (no example re-citation). Two cases:*

- **the sign IS the operator's symbol — a glyph (prefix/suffix) OR the literal `-` operator:** ONE uniform two-step — step 1 reversed the result onto the sign's side, step 2 swaps the stand-in minus for that symbol (identity when the symbol is `-`):
```
[LABEL]([QA], [QB]) = [EXPR] = [RAW]; reading order is [MODE], so the result [RAW] -> [STANDIN]
Need to use the operator symbol for the subtraction sign; mapping the subtraction sign [LABEL] back to [SYM], so [STANDIN] -> [ANS].
```
  (digit-only/prefix glyph: `… -> -93` then `… so -93 -> \93.`; fully/suffix glyph: `… -> 03-` then `… so 03- -> 03!.`; literal `-`: `… -> -82` then `… so -82 -> -82.` — identity map, same structure; rightward: step 1 is just `… = [RAW];` with no reversal, step 2 maps `[RAW] -> [ANS]`.)
- **no example fixed a sign side (fmt `none`):** the sign can only stay a leading minus — step 1 stops at the stand-in (no `answer =`), step 2 (leftward only): `The result is negative; the sign is written as a leading minus, so the answer is [ANS].`

*Suffix (always, two lines):*

```
I will now return the answer in \boxed{}, the answer is
\boxed{[ANSWER]}
```
