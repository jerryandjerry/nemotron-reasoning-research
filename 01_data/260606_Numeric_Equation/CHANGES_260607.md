# CoT changes — 2026-06-07 (equation_numeric) — hand-off for the cryptarithm agent

**What this is.** A summary of every change made to the equation_numeric CoT this session, so the same
conventions can be applied to the cryptarithm CoT where the equivalent constructs exist (the two share the
"Alice-world" template: prior-knowledge block, reading-order search, family lock/verify, exotic/unknown
verdicts, explicit computation, `\boxed{}` answer).

**Status.** All changes are **narration-only**. GT-match held at **621/732 (84.8%)** and there are **0
boxed-answer diffs** vs the frozen `260526_Numeric_Equation/` at every step — nothing here changes an answer,
only the wording/structure of the reasoning log. Active folder: `01_data/260606_Numeric_Equation/`.

---

## 1. Exotic ≠ Unknown — two DISTINCT verdicts (the core change)

These were being conflated (both rendered "unknown"). They are different situations and must read
differently:

- **Exotic** = the operator IS in the examples, but it **exhausted all four families** (we tried every
  family and none fit). A verdict reached by elimination.
- **Unknown** = the operator **never appears in the examples** (it's only in the QUERY), so there is
  **nothing to try** — no search can run on it.

Deciding factor: *the search ran and exhausted* (→ exotic) vs *there was nothing to run on* (→ unknown).

**Rendering:**
- Exhausted operator (any operator, any reading): `[X] is an exotic operator` — **NOT** `[X] = unknown`.
  - death line: `No candidate family is left for g; so it is an exotic operator.`
  - state line / conclusion / summary: `… g is an exotic operator`
- Unseen operator (only in QUERY): stays `[X] = unknown`, then guessed.

**Counting:** an exotic operator is **NOT counted** as resolved. e.g. `{f = a×b, g is an exotic operator},
1 of 2 resolved` (g labelled exotic but still only "1 of 2"). Same for an unknown operator.

**Per-reading note:** "exotic" is a *per-reading* statement — exactly parallel to an operator being `a+b`
under one reading and `a×b` under another. A reading that gets discarded can label ordinary operators
"exotic" with **zero effect** on the answer (the real resolution only runs on the KEPT reading's exhausted
query operator). This is fine and intended — do not try to suppress it.

---

## 2. Prior-knowledge block additions

- **max-mod-min is established up front.** Prior #1 now ends:
  `… all ordered by frequency. Any other operation is deemed exotic and we will try max(a,b) mod min(a,b).`
  So when the verdict later says "lock g as max(a,b) mod min(a,b)", it follows from a stated rule instead
  of being introduced out of nowhere.
- **New prior point 6 (never give up):**
  `6. I should never return the answer as "unknown". When the final result is undetermined, I should make a
  best guess using what I know.`
  (This is a soft nudge against the model hallucinating `\boxed{unknown}`, which never appears in training.)
- **Removed** the "The absolute forms |a-b|, -|a-b| carry no noise …" sentence from prior #1 (it was extra
  explanation, not needed).

---

## 3. Exotic verdict line — folded to one line

**Before** (three lines):
```
QUERY operator g is not solvable by the four families in §2, so g is an exotic operator. Note: an exotic operator may change all the rule we deduce above.
The only operation that exists outside the system is max-mod-min(a,b) = max(a, b) mod min(a, b)
Lock g = max-mod-min, and verify it against the examples in the confirmed reading (leftward):
```
**After** (one line):
```
QUERY operator g is an exotic operator, so lock g as max(a,b) mod min(a,b), and verify it against the examples in the confirmed reading (leftward):
```
- Dropped: the "not solvable by the four families" clause (the per-reading state already said "is an exotic
  operator"), the separate "The only operation that exists outside the system …" line (folded — max-mod-min
  is now in prior #1), and the "Note: an exotic operator may change all the rule …" line.
- Reading-revision branch lead-in: `…, and verify it against the examples, starting with the confirmed
  reading (DIR):` then `The confirmed reading does not match, so check the other reading (OTHER):` (this
  carries the same meaning the dropped Note used to flag).
- **The lock spells the OPERATION** `max(a,b) mod min(a,b)`, never the name "max-mod-min". (General rule:
  locks are operations/formulas, never family names or labels — the model can't learn names.)

---

## 4. Every operation computed EXPLICITLY — everywhere, including confirm rows

Principle: a check/confirm row must show the operation applied to its operands, the same way the lock row
does — never a collapsed result.

- **max-mod-min confirm rows** now show the operation, not the bare reduction:
  - before: `EX3: 88 mod 23 = 19`
  - after:  `EX3: max(88, 23) mod min(88, 23) = 88 mod 23 = 19`
- **Multiplication confirm rows** now show full **partial products**, matching the lock row and the query
  answer (was collapsed):
  - before: `check EX3: 57×16-1 = 911; EX4: 51×33-1 = 1682`
  - after:  `check EX3: 57×16-1 = (50+7)×16 - 1 = 50×16 + 7×16 - 1 = 800 + 112 - 1 = 911; EX4: 51×33-1 = (50+1)×33 - 1 = 50×33 + 1×33 - 1 = 1650 + 33 - 1 = 1682`
- `+`, `−`, `∥` were already explicit (`82+86 = 168`, `|37-26| = 11`, `23∥64 = 2364`) — unchanged.

**Cryptarithm relevance:** any place the cryptarithm CoT confirms an operation across multiple examples
should show the operation explicitly the same way (especially multiplication by partial products), not a
bare `= result`.

---

## 5. Unseen-guess lead line reworded

- before: `Query operator g is not determined by the examples, so I need to guess it.`
- after:  `Query operator g only appears in QUERY, so I need to guess it.`

Reason: "not determined by the examples" collided with the *exotic* case (which is also "not determined" but
for a different reason). The new wording names the actual unseen condition and matches the §n.1 state note
(`g = unknown since g only appears in QUERY`).

---

## 6. Concat-halt short-circuit REMOVED (Fix 1)

A query operator that locks concatenation used to **early-exit** the solver (skip the rest of the search and
the other reading). That short-circuit was removed: a concat query operator is now locked normally and the
solver runs the **full resolution** like any other puzzle. Because concatenation is reading-invariant, the
boxed answer is unchanged (0 diffs); 72 CoTs that used to halt now show the complete trace.

The old halt text (`Note: g is the QUERY operator … other unknowns are irrelevant … Proceed to answer
directly.` + the jump Conclusion/Summary) is gone.

---

## 7. What to apply to cryptarithm

Apply the analogous change wherever the same construct exists in the cryptarithm CoT:
1. **exotic vs unknown** split (exhausted-search vs nothing-to-search), with the exotic operator **not
   counted** as resolved, and rendered "is an exotic operator".
2. **prior point "never return unknown — guess"** (if cryptarithm can emit a give-up).
3. **max-mod-min (or the cryptarithm equivalent out-of-system operation) established in the prior block**,
   then the verdict locks the **operation expression** directly, with no separate "only operation outside
   the system" / "Note: may change the rule" lines.
4. **explicit computation in confirm rows** — multiplication by partial products, mod by max()/min(), etc.,
   the same form as the lock row and the final answer; never a collapsed `= result`.
5. **reword any "not determined by the examples" give-up** to name the real condition.
6. **remove any reading-invariant early-exit short-circuit** in favor of the full resolution (only if
   cryptarithm has one).

Reference implementation for all of the above: `_gen_eq.py` and `cot_template.md` in this folder; a worked
example is `equation_numeric_1f0fbe5f/track/tree_cot.txt` (exotic) and `equation_numeric_0505ab22/...`
(multiplication partial products in confirm rows).
