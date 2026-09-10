# Cryptarithm scenario space — all axes & distribution

_Reference for the cryptarithm CoT/aug pipeline. Established 2026-06-13 (Chicago). Grounded in our own
`_gen_crypt.py` (§-tree DFS solver) and `_gen_crypt_aug.py` (§4b generator), measured over the original 800.
**Not** copied from numeric_equation — that puzzle only lent CoT format/wording, never content._

---

## 0. Ground-truth reality (read first)
No handed-down competition gold for cryptarithm. Puzzles are genuinely **ambiguous** — most have several
valid solutions; `answer.txt` is the **designer's pick**, one valid leaf. Trust is **consensus**. Our §-tree
solver (GT-free, sound) is the best reference; the ikev/"golden" solver is **oracle/GT-conditioned**, trusted
only as a *derivation-from-gold*. **GT-match ≈ 54% is the ambiguity ceiling, not solver weakness**
(179/179 on the uniquely-determinable puzzles).

## 1. Operation pool
Real puzzles use ~20–47 ops; **we deliberately model only four families** — the `"We only consider four kinds
of operations"` rule is a *useful lie* bounding the search/CoT length (the full library → branching explodes →
CoTs exceed the 7680 cap → unlearnable).
- **Modeled:** `~add` (a+b, ±1, ±2) · `~sub` (|a−b|, a−b, b−a, −|a−b|, ±1, ±2) · `~mul` (a×b, ±1, ±2) ·
  `~concat` (ab, ba). Arith ops distinct; concat may repeat.
- **Exotic fallback (ONE extra, only after the four families are exhausted):** `a mod b` (operand order;
  a<b ⇒ a). Cryptarithm's own content — **NOT** numeric's `max(a,b) mod min(a,b)` (`modulo(11,63)=11` proves it).
- Unmodeled by design: gcd, lcm, rmod, fdiv, rdiv, min, max, mul_half, a²+b, bitwise, … → exotic fallback or dropped.

---

## 2. THE AXES (complete)

### A1 — TYPE (primary; reasoning path the model must not shortcut; the 5 are never mixed)
| value | definition | resolution |
|---|---|---|
| **deduce** | operator(s) in examples, four-family | full joint digit+operator deduction |
| **concat** | query operator is concatenation | short-circuit to symbol concat |
| **query-only** | operator appears **only** in the QUERY | can't derive → **deterministic default** |
| **exotic** | operator **in examples** but no four-family fits | post-search: lock **`a mod b`**, verify vs its examples, apply to query |
| **ambiguous** | examples don't uniquely pin the cipher / a needed digit is absent | **honest deterministic default** |

**A1 sub-scenarios:**
- **deduce** → per-operator family {~add, ~sub, ~mul} × variant {base, +1, −1, +2, −2, |·|, b−a, −|·|}
- **concat** → {pure_concat (all ops concat), mixed_concat (concat + arith)} × direction {fwd `a∥b`, rev `b∥a`}
- **query-only** → {no 4-digit RHS → concat · 4-digit RHS → arithmetic-by-distinctness · no reading → blind concat}
  (the "4-digit RHS but no arith" branch is dead in the 800)
- **exotic** → modeled resolution is `a mod b`; the actual hidden op may be modulo/gcd/lcm/rmod/fdiv/rdiv/a²+b/…
  (we do **not** identify it — `a mod b` is the single deterministic fallback)
- **ambiguous** → {free operand digit → **smallest by rule** ("not pinned by any example; go with the
  smallest") · missing answer digit (absent from examples) → **first unused symbol** ("since I have to choose,
  take the first one left")}

### A2 — reading: rightward / leftward
- sub: leftward → {leftward-on-digit-only, leftward-fully} (where the sign sits under a leftward read)

### A3 — signed: none / prefix / suffix
- sign present? and, if so, sign-position (prefix vs suffix). Hard rule: an operator symbol in the **RHS** ⇒
  always a subtraction variant (doubles as the negative sign; 419/419, 0 exceptions).

### A4 — # operators: 1 / 2 / 3  (each operator a **distinct** family)

### A5 — # examples: 2 / 3 / 4 / 5

### A6 — operator symbol-class: arith glyph (`+ − *`) / punctuation (one of the 23 operand-capable symbols)

### A7 — cipher size: # distinct digit-symbols present (bijection 0–9 → symbols; sets how much of the cipher
the examples can pin → feeds the **ambiguous** type)

### A8 — BPE-merged-token coverage: which merged operand-symbol tokens appear (rare-token coverage axis;
cryptarithm-specific, drives transcription learning — no numeric analog)

_(No "leading-zero" axis: the cipher forbids a leading-0 by construction.)_

---

## 3. Distribution over the original 800
- **TYPE / query operator** (per ikev derivation-from-gold): four-family 705 · query-only 75 · **exotic 20**
  (modulo 7 / gcd 7 / rmod 4 / lcm 2 — only `a mod b` independently verified; gcd/lcm/rmod labels unconfirmed)
- **any exotic operator incl. example positions (→ four-family solver fails):** 44/800 ≈ 6%
- **ambiguous:** most of the ~46% that don't GT-match (multiple valid answers)
- **query-only branches:** no-4digit→concat 56 · 4digit→arith 88 · blind-concat 4 (≈31% GT-match — weak by nature)
- **A2–A6 rates:** reading R 62 / L 38 · signed none 60 / prefix 38 / suffix 2 · #ops 2:41 / 3:56 / 1:3 ·
  #ex 3:31 / 4:38 / 5:23 · arith-symbol yes 78 / no 22

---

## 4. Aug coverage today, the gap, and the deliverable format
Trainable aug = 849 (307 original + 542 aug) across the 6 subtypes (arithmetic, little_endian, pure_concat,
mixed_concat, mixed_concat_little_endian, query_unseen_concat), balanced on A2–A8 with rare-token coverage.
**TYPE only spans deduce / concat / query-only.**
- **exotic = 0** (4-family generator can't make it; the 24 original exotic puzzles were deleted)
- **unseen_arith ≈ 0** (generator path exists but corpus never populated it)
- **ambiguous** — present implicitly but not balanced as an explicit cell

### Scenario CELLS to balance (the full joint space)
A populated cell = (A1 type/sub-scenario) × (A2 reading) × (A3 signed) × (A4 #ops) × (A5 #ex) × (A6 symbol-class),
with A7/A8 (cipher size, BPE-token coverage) swept for transcription coverage. The A1 cells, named:
- **deduce**: arithmetic / little_endian (× family per op × variant)
- **concat**: pure_concat / mixed_concat (× fwd|rev)
- **query-only**: query-only-**concat** (no 4-digit RHS) / query-only-**arithmetic** = unseen_arith (4-digit RHS) / query-only-**blind**
- **exotic**: exotic-deduce (`a mod b`)
- **ambiguous**: free-digit (smallest) / missing-digit (first-unused)

### → REGENERATE TARGET: `01_data/260613_Cryt_10k/` — balanced ~10k, ONE FOLDER PER PUZZLE, original + aug
Order (dependencies): (1) Phase-1 wording rework applied + verified (0 boxed changes); (2) add the `a mod b`
exotic resolution to the solver; (3) add generation for the **empty cells** (exotic, query-only-arithmetic);
(4) generate a **balanced** set across every A1 cell × A2–A8; (5) **fold the original 800 in** (each as a
folder) + aug; (6) emit **one folder per puzzle**; (7) audit balance + 0-boxed-change.

**⚠ DELIVERABLE FORMAT — one folder per puzzle (not CSV-only):**
```
01_data/260613_Cryt_10k/<type>_<id>/
  question.txt   # THE ONLY solver input
  answer.txt     # gold (designer's pick)
  label.txt      # TYPE + sub-scenario + axis tags
  track/tree_cot.txt   # the §-tree DFS CoT
```
Includes **all original 800 + aug**, ~10k total, every puzzle its own folder (same shape as
`01_data/260525_Cryptarithm/<subtype>_<id>/`). The 10k harvest's CSV-only form is the deficiency this fixes.

## 5. CoT wording scheme (format mirrors numeric; content is cryptarithm's; narration-only, 0 boxed changes)
- Opening: `Each symbol can be a hidden operator or operand. I work out the digit cipher and each operator
  from the example equations, then apply them to the QUERY and return the answer.` (+ operator-symbol caveat)
- Prior #1 ends: `…Any other operation is deemed exotic and we will try a mod b.`
- Prior #6: verbatim two-verdict definition (QUERY-only operator / exotic operator; "In either case we still
  resolve it to a concrete answer.")
- Verdicts: `g is a QUERY-only operator` / `g is an exotic operator`; resolutions are deterministic **"by
  default"** — never `guess`, `unknown`-as-verdict, `waste of time`, or `fits`.
