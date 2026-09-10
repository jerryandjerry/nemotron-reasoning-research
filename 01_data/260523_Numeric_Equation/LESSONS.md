# Numeric Equation — Hard-Won Lessons (read before changing anything)

> The [README](README.md) tells you the pipeline and process. **This file is the accumulated judgment**:
> the dead ends already tried and rejected, the exact forms of dishonesty that kept getting caught, the
> tokenizer landmines, the design rationale, and the philosophy. An agent who skips this will re-walk every
> wrong path — and won't even know to check the non-obvious traps. Everything here was learned the
> expensive way.

---

## A. Dead ends — tried, rejected, DO NOT re-attempt

Each of these *looked* reasonable and was implemented, then proven wrong. The fix is in
[`agent_review_instruction.md`](agent_review_instruction.md) (#1–#10); the *why it's wrong* is here.

1. **Exotic guess = "lucky number" (most-common example output).** Scored exotic **0/17**. Replaced by a
   hardcoded `max(a,b) mod min(a,b)` guess. Framing that matters: this is **not** a 5th modeled family and
   **not** a reversal of "four families only" — it has the *same status as boxing a fixed number like
   2026*. We *observed* that the out-of-system operator is almost always max-mod-min and hardcoded it; it is
   never verified per-puzzle. → exotic **17/17**.

2. **Bidirectional sign gut-check** (reverse the WHOLE output, `-91`→`19-`). Broke **90** right-to-left
   puzzles by treating a leading sign as if it became trailing. The CORRECT sign check is **unidirectional**:
   a tail sign in the *written / left-to-right* form only. (`_rev` never moves a leading sign to the tail.)

3. **`trial_line` with `≠` / "fits".** It declared a family dead on an *in-band* example, e.g.
   `02×91 = 182 ≠ 183` then "no ~mul variant gives 183" — **false**, `mul+1 = 183` hits it. Removed
   entirely. Now: honest greedy lock→verify→fail; a family only dies on a value it genuinely can't reach
   (`far from`, gap > 2); an in-band miss says `not`; exact `=` is used **only** to lock. **Never `≠`, never
   the word "fits"** (reviewers grep for both).

4. **Query-operator-first frame tiebreak.** Superseded by the §N.1 gut-check; deleted.

5. **"Need to guess the query operator g" special-case that SKIPPED §2.** Rejected — the stop rule is
   **uniform**: §1 ends only if it resolved every operator (X == Y); an unseen query op is counted in Y but
   never resolved, so it *always* opens §2. Don't add a shortcut.

6. **The "ambiguity caveat"** (`too few examples to pin g — sub_signed also fit and disagree... this is a
   guess, not a unique determination`). DELETED as **leakage** (see §B). The greedy search NEVER tested
   `sub_signed`; "too few examples"/"this is a guess" are correctness verdicts only makeable by peeking at
   the answer. The `ambiguous_arithmetic` *label* stays (our taxonomy) but triggers **no** narration.

7. **§N.3 announcing `avail_soft[0]`** (the first survivor not already announced by an earlier sign, via a
   `taken` set "to keep trial headers distinct"). Wrong: another sign's **unconfirmed** try must not delete a
   family from this sign. **Distinctness is a LOCK-time constraint** — a family leaves a sign's candidate
   list only when *another sign LOCKS it* (§N.4 enforces that on advance). Now §N.3 announces the sign's own
   top survivor; two signs sharing a top both announce it and §N.4 honestly tries the colliding combo.

8. **Leading-0 gut-check gated on a concat PRE-CHECK** in its `if` condition. It silently ran §N.2's concat
   test to decide *whether to narrate*, and stayed silent when concat matched — pre-computing a later step's
   result (forward-only violation). Now a leading-0 number **always** fires the gut-check, step by step:
   announce first, then compute and conclude.

9. **The `confirmed_sub`-only QUERY leading-0 shortcut** (`we already lock ~sub, so not concat`). Replaced
   by the unified scan list `[all examples, then QUERY operands]` + the one shared `lock_concat` for *any*
   seen operator. A sign-confirmed ~sub op now runs the same `lock_concat` and lands on "neither matches →
   illegal".

10. **`lock_concat` deciding direction off `exs[0]` blindly.** When the first example had equal operands
    (`66∥66` → `concat_fwd == concat_rev == output`), it committed to `concat_fwd`, and if that failed
    confirmation it returned "not concat" — never trying `concat_rev`, which was the consistent direction.
    Now it decides off the **first example whose operands differ**, defers equal-operand examples, and
    defaults to `concat_fwd` only when *no* example distinguishes. (Mirror of the ~sub variant walk — see
    §D.)

11. **Making the §N.4 trial header always show distinct families.** Would require knowing which family fits
    before trying it = look-ahead = leakage. Left honest: the header may propose a colliding combo the
    search then backtracks (the "all taken … let the search check it" line says so). ~3 puzzles show this;
    the final locks are distinct and correct.

12. **`§1`–`§7` as GLOBAL sections.** Wrong mental model. `§` numbers **reading branches only** (`§1`
    rightward, `§2` leftward); the setup is unnumbered prose, and `§N.1`–`§N.4` are the four sub-steps
    *within* a reading.

---

## B. Leakage traps — the exact dishonesty patterns that kept getting caught

A "leak" = the CoT uses information it could not have at that point in an honest forward search. The human
caught all of these repeatedly; treat each as a tripwire.

- **Using a later step's result in an earlier line.** Pre-computing the answer, the re-attached sign, or the
  concat result and dropping it into a line above where it's derived. (Killed #8, #10, and the
  `answer = {re-attached}` bug — the `answer =` line must show the **raw signed value**, sign re-attached
  only on the following line.)
- **Asserting the other reading while in this one.** "How do you know it's little-endian while you're in
  standard mode?" — you don't yet. Each reading is narrated self-contained; the winner is chosen only at the
  `Summary:` line after both are shown.
- **Sufficiency / correctness verdicts.** "too few examples", "this is a guess", "not a unique
  determination", "X also fits and disagrees" — these are only knowable by checking the answer. Forbidden.
- **Mentioning alternatives the trace never tested.** If the code computed `alts` but the greedy search
  never walked them, they cannot appear in the CoT.
- **Calling a logic choice "accurate"/"correct" because it matches the stored answer.** The stored answer is
  not ground truth and is never a justification (see §C of the README). Report the GT-match number as a
  signal; never reason from it.

The antidote is the same every time: **the CoT is a literal system log.** Do a step → print what it did →
print the resulting state. If a line is printed, the state actually changed. Don't fear mistakes — the
fresh reviewers recompute every line, so an honest wrong step is fine; a smart-looking leak is not.
("Be stupid, don't be smart." "Write and search at the same time; nothing changes after the line is
printed.")

---

## C. Tokenizer landmines (verified against the real Nemotron tokenizer, vocab 131072, GPT-2-style byte BPE)

These are invisible unless you check token IDs. They were found by tokenizing the actual output. The
sign/character legend table in [`cot_template.md`](cot_template.md) pins every special char so look-alikes
are never substituted.

- **Two different minus signs.** `-` U+002D = token **1045**; `−` U+2212 (MINUS SIGN) = token **7813**. The
  `~sub` definition once used U+2212 while every computation, negative sign, and `\boxed{-1}` used U+002D —
  so "subtraction" was *two unrelated tokens*. **Unified everything to U+002D** (canonical because it's tied
  to the boxed-answer format and the literal `-` operator symbol; going the other way risks the metric).
- **The old `§N - Reading` separator was a real collision.** `" - "` tokenized to `Ġ-` = **1462** — the same
  token a minus sign produces. Removed; headers are now `§1 reading rightward:` (no separator).
- **The arrow `->` = token 1941, distinct from `-` (1045).** Safe; deliberately kept (changing to `→`
  U+2192, token 14835, gives zero benefit).
- **Compound hyphens are safe** (`left-to-right`, `digit-count`, `2-digit`, `max-mod-min`): the `-` absorbs
  into a `-X` word-piece, never the standalone minus token.
- **`∥` (concat) = 2-token byte-fallback (33778+1165)**, distinct from abs-value `|` (1124). A minor
  inefficiency, *not* a collision — left as-is.
- **Lesson:** any character that *looks* like an operator or sign must be checked against the tokenizer; a
  look-alike Unicode char can silently split your training signal in two.

---

## D. Design rationale — the "why" behind the non-obvious choices

- **THE foundational insight: operators are tested INDEPENDENTLY because the distinct-operator rule always
  holds.** Distinct arithmetic operators cannot share a family, so locking one constrains the rest — there
  is no need to search the *combination* of all operators at once. This is what makes the cheap greedy
  per-operator search valid and complete. If you ever doubt "why don't we try all combinations?", this is
  why.
- **Greedy search ⇒ zero ambiguity by construction.** §N.4 locks the *first* consistent variant and stops,
  then the query just applies it. It never explores alternatives, so there is nothing to caveat (see dead
  end #6).
- **Why `§` (not `#`).** Operator symbols span essentially all ASCII punctuation — the operator set is 26
  chars including `#` (which is an operator in 78 puzzles). `§` and letters are the only collision-free
  markers.
- **`~sub` variant order `|a-b| > a-b > -|a-b| > b-a`** (then ±1, ±2). Literal `a-b` is preferred over the
  contrived `-|a-b|`; this ordering was the best of all 24 permutations on the set. It is a **prior, not a
  derivation** — the sign of an ambiguous ~sub query is genuinely undeterminable, so a fixed order is the
  honest deterministic choice; the GT-match only *confirms* it.
- **Distinctness is a LOCK-time constraint** (see dead end #7), never an announcement-time one.
- **Reading order is GLOBAL** — one reading governs the whole puzzle; you don't mix readings across
  operators.
- **`unseen_leading_zero` puzzles are relabeled and LEFT AS-IS** (CoT unchanged, counted as misses), not
  fixed-in-CoT: an unseen operator with a leading-0 operand forces ~concat by the leading-0 rule, but with no
  example of that symbol the direction (fwd/rev) is undecidable — there is no deterministic answer, so we do
  not fabricate one. Exactly two such puzzles.

---

## E. The philosophy (the human stated these HARD — they override convenience)

- **"It is a system log — can you print a fake system log?"** The CoT is not a beautiful customized
  explanation; it is a log of the solver's running state.
- **"Stay true to the logic even if we lose points."** Honesty/determinism/forward-only is the only
  correctness criterion. The score is a sanity signal, never the target, never a justification.
- **"It is NEVER the goal to beat another solver's CoT."** The goal is an honest, real CoT with zero
  leakage — "not any."
- **"Do the right thing, don't cut the short path."** Spending more effort on the correct path is cheap;
  shipping a plausible-looking shortcut is expensive.
- **Fixes are human-directed:** when you find a defect, surface it with a clickable puzzle link, explain the
  root cause, and **wait** — do not implement a fix unilaterally. (Several "obvious" fixes were wrong; the
  human redirected them.)

---

## F. Practical gotchas (will waste your time otherwise)

- **The markdown formatter can race on [`cot_template.md`](cot_template.md)** — prefer editing it via a
  python/temp-file script over many rapid in-place edits.
- **macOS `split` lacks the GNU flags** (`-n l/10`, `--additional-suffix`). Shard with `awk` instead (the
  command is in README §6).
- **`OUT_DIR = dirname(__file__)`** in the generator — so a *copy* of `_gen_eq.py` in `/tmp` writes its
  output to `/tmp`, not the real folders. This is exactly what makes the "reconstruct the pre-fix baseline
  and diff" verification safe (README §7).
- **A narration-only change must hold the score and move zero boxed answers** — always prove it with the
  before/after answer diff before believing it's narration-only. (Several "pure narration" changes were
  caught silently moving an answer.)
- **Reviewers must be fresh / no-context** — an author auditing their own output is blind to their own
  leaks. The whole point of the 10-agent audit is independent recomputation.

---

## G. The bigger picture

This is the **operand-settled sub-problem of the cryptarithm work** — it reuses that method and wording
(the project memory links it as `project-cryptarithm-solver`). The same honesty discipline, the same
system-log principle, and the same fresh-reviewer audit apply there. If you're improving this, the
cryptarithm pipeline is the sibling to keep consistent with.
