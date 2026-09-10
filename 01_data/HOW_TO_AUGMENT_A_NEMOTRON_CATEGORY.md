# How to Augment a Nemotron CoT Category — synthetic puzzles, blind-solve, GT-match filter

A repeatable method to MANUFACTURE extra training data for any Nemotron category (equation_numeric,
cryptarithm, …). Built and run on equation_numeric; this generalizes it so the cryptarithm agent (or any
other) can apply it. Reference implementation: `01_data/260607_NumericEq_Aug/`
(`_gen_aug.py` generator · `_solve_blind.py` blind driver · `_build_aug_csv.py` CSV builder).

---

## 0. The principle — construct-with-gold, solve-BLIND, keep-the-recovered

We can't get the real test set, and the training target is to **reproduce our own solver's CoT**. So to make
more training data we:

1. **Construct** synthetic puzzles where WE know the answer (we built them from a chosen hidden rule).
2. Run the **solver BLIND** on each puzzle — it sees `question.txt` only, never the gold.
3. **GT-match** = does the blind solver recover the planted answer? Ambiguous / underdetermined / buggy ones
   fail and are filtered out.
4. The `(question, blind-CoT)` pairs that GT-match are clean augmentation.

The integrity is **separation**: the gold-holder (the generator / you) and the solver (run by blind
sub-agents) are kept apart. The gold is legitimate because *we built the puzzle*, and GT-match independently
confirms the blind solver agrees — so a high GT-match rate means the puzzles are genuinely solvable, not that
we leaked the answer.

> Earlier worry, resolved: "isn't knowing the gold cheating?" No — cheating would be the *solver* seeing the
> gold. Here the solver is blind; we only use the gold to *filter* which generated puzzles are good. An
> ambiguous puzzle the solver reads differently than we built just scores GT-match=False and is dropped.

---

## 1. Generate `question.txt` (synthetic, gold-bearing)

1. **Measure the real corpus first** and mirror its surface exactly: operand/token digit-counts, number of
   example lines, number of operators, the symbol pool, sign conventions, reading directions. (For
   equation_numeric: all operands 2-digit, 3–5 examples, 2–3 operators, a fixed 26-symbol pool, ~14%
   signed outputs, two reading orders.)
2. **Pick the hidden structure.** equation_numeric: an operation + noise-variant per operator + a global
   reading order. cryptarithm: the symbol→digit cipher (+ reading). This is the gold-bearing rule.
3. **Render the puzzle with the SOLVER'S OWN low-level string helpers**, so each written line reads back
   *exactly* to the intended value. equation_numeric used `raw_on` / `write_operand` / `write_output`, where
   `write_output` is the inverse of the solver's `target_of` (handles digit-reversal for leftward and the
   sign glyph). This guarantees the puzzle is a valid instance the solver will parse as intended.
4. **Compute the GOLD** = apply the hidden rule to the query, rendered the same way.
5. **Validity gates at generation time** — cheap and exact (don't call the full operator-search; it's slow):
   - each operator's examples must be reproduced by its *intended* variant (`all(fits(...))`);
   - an "exotic"/out-of-vocab operator must fit **no** modeled family (early-exit `any_family_fits`), else
     it isn't exotic. Reject + retry on failure.
6. Write per puzzle: `question.txt`, `answer.txt` (gold), `label.txt` (type), and append to a master
   manifest (id, gold, type, secondary features). **The hidden rule / gold is NEVER written into
   `question.txt`.**

Watch for infinite loops in the distribution logic (e.g. "assign N examples across K operators capped at 3
each" loops forever if N > 3K — cap it). Generating on a cloud-synced drive creates transient conflict
copies (`track 2/`); let sync settle before downstream steps read the files.

---

## 2. Blind-solve with the solver (sub-agents)

1. Write a **blind driver** that, for each id, reads **only** `question.txt`, runs the solver (`gen_cot`)
   with the answer field **empty** (the solver is forward-only and never uses the answer), writes
   `track/tree_cot.txt`, and records the boxed answer. It must never open `answer.txt` / the manifest.
2. **Spawn N sub-agents** (one Workflow), each runs the driver on a disjoint slice; the prompt explicitly
   bars reading any gold/answer/manifest file and asks them to confirm `read_any_gold=false`.
3. This makes the solve provably independent of the gold. (Running the same blind driver yourself is also
   valid — the blindness is in the driver — but sub-agents give a clean, demonstrable boundary.)

---

## 3. GT-match filter

- Extract each blind boxed answer with the **live metric** (last non-empty `\boxed{}`, close brace via
  `rfind('}')`) and `verify(gold, pred)` (binary→exact, float→±1%, else→string).
- Keep **GT-match=True** (solver recovered the planted answer); discard False. Yield is ~90%+ for determined
  types, lower for "guess" types — expected, not a defect.

---

## 4. Distribution control — the real design work

- **You do NOT know the test distribution.** The original corpus is your only *proxy*; do not state its
  proportions as if they were the test's. Decide the target training mix by **judgment + held-out eval**, not
  by assuming the test.
- **Complement, in moderation.** Boost the tails the original under-covers, but don't *swamp* — e.g. pushing
  exotic from ~2% to 24% of training is almost certainly too aggressive (the model over-predicts it). Pick a
  sane combined target (we used eq ≈ deducible 55% · exotic 10% · unseen 10% · ambiguous 10% · concat 15%).
- **Un-synthesizable types come from OVERSAMPLING the original** (e.g. "ambiguous" is hard to construct
  deterministically — oversample the original's ambiguous rows to hit the target share).
- **Build the training CSV** to match the main training CSV's columns so it appends: all original GT-True
  rows at 1× + fresh aug to fill the targets + oversampling for the un-synthesizable buckets. Don't train on
  GT-False rows (keep them at 0×).

---

## 4b. DECORRELATE every feature from the category — the make-or-break step

This is the part that quietly decides whether the augmentation helps or *hurts*. A category label is what we
want the model to **reason** its way to. If any *surface* feature of the puzzle predicts the category, the
model takes the shortcut — it reads the feature and skips the reasoning. The augmentation, if careless, is
exactly where such a shortcut gets manufactured, because we control the joint distribution.

> The trap, concretely: if "leftward reading" ends up tied to "unseen", the model learns *leftward ⇒ guess*
> and stops deducing on every leftward puzzle. **And it is NOT just reading** — the same applies to sign
> presence, leading-zero, operator count, symbol class (arith `+-*` vs punctuation), #example-lines, operand
> ranges. **No feature, primary or incidental, may correlate with the category.**

Four concrete techniques (all used in the eq run):

1. **Inject every secondary feature at the SAME rate in every category.** One global knob per feature
   (`SIGNED_PROB`, `RIGHTWARD_PROB`, `LEADZERO_PROB`, arith-symbol prob …), applied identically whether the
   puzzle is deducible, exotic, unseen, or concat. No per-category feature logic.
2. **Do NOT create forced "corner" subtypes that hard-wire a feature to a category.** We initially had
   `exotic_sign` / `exotic_concat` / `multi_exotic` as their own buckets — but "exotic_sign is always signed"
   means *signed ⇒ exotic*. **Delete them.** Let the exotic+sign and exotic+concat combinations arise
   *naturally*, at the same rate, inside **every** category. Corners you want covered should be emergent, not
   labeled.
3. **Balance the structural axes too, not just the cosmetic ones.** Operator count is a feature: if unseen is
   always 2 ops and deducible is 3, op-count predicts the category. Keep all categories on the same
   #distinct-operators range (we used 2–3 everywhere).
4. **Stratify the GT-match subsample so the filter doesn't re-introduce skew.** GT-yield differs by type
   (unseen ~75%, others ~97%), and the *survivors* of a low-yield type can be selectively biased (e.g. the
   filter drops more 1-operator unseen → survivors skew op-heavy). When you subsample the kept rows down to
   the target, **stratify by the at-risk feature** (we stratified by reading) so the final mix stays balanced.

**Then PROVE it: measure per-category feature distributions on the final set and show each category is ~equal
on every axis.** If deducible is 50% signed, exotic should be ~50% signed, unseen ~50%, concat ~50% — and the
same for reading, leading-zero, op-count, symbol class. A flat table across categories is the deliverable that
says "no shortcut exists." (Note: the *original* rows carry their own mild real-world skews you can't edit;
the goal is that the **aug doesn't amplify them**, and that no feature becomes a strong predictor in the
*combined* set. Measure the combined set, not just the aug.)

**Cover the scenarios the original could only carry as GT-False.** Decorrelation balances the cells that
*exist* in training; this is the dual rule for the cells that DON'T. The original drops every GT-False row
(trained at 0×) — but a scenario is not the same thing as its (possibly-ambiguous) answer, and some scenarios
appear in the corpus *only* on rows that happen to be GT-False. If you stop there, that scenario vanishes from
training entirely — which is the exact hole augmentation exists to fill. So: **enumerate the scenario cells
carried only by GT-False originals, and synthesize GT-True instances of them.** Make them GT-True by
construction — plant the gold = the solver's own deterministic output for that underdetermined case (the same
tie-break the solver will produce blind), so GT-match passes and the cell is trainable. Worked example (eq,
2026-06-13): a gap check (GT-False originals' 6-axis scenarios vs. what training actually contained) found the
aug never produced **single-operator (nops=1)** puzzles, so single-op `ambiguous` and the
`unseen_leading_zero` concat-guess combo existed only on excluded rows. Fix: generate GT-True single-op
ambiguous (`build_ambiguous(n_fill=0)`) + `unseen_leading_zero` (gold = solver's deterministic output), fold a
proportional count into the buckets — gap check then 0 missing. **The check to run: for every scenario cell
present in the *original* corpus (any GT-status), assert it is present in the *trained* set; any cell that is
original-only-GT-False is a synthesis target, not an omission.**

**Worked example — the trap and the fix, from the eq run.** First pass looked decorrelated on reading but a
combined-set measurement exposed two *manufactured* shortcuts: `arith-symbol` spread **42%** across categories
(exotic 16% vs ambiguous 58%) and `leading-zero` spread **31%**. Root cause: the aug *complemented* those
features (down-weighted arith 60%→10%, leadz 37%→3%) while the originals had them ~uniform — and because aug
fills each category to a different degree (exotic 88% aug, ambiguous 0% aug), a uniform-but-wrong-level aug
rate became a category predictor. Two more subtleties surfaced: a *per-operator* arith probability made arith
scale with operator count (so operator-heavy types read as more-arith) — fixed by a **puzzle-level** "≥1 arith
symbol" decision that forces exactly one arith slot; and the `leading-zero` level had to be raised to the
originals' ~37% (matching, not complementing). After the fix, combined spreads: arith **12%**, leadz **7%**,
reading **11%**, with signed/op-count residue (25% / 24%) that is *real in the originals* (ambiguous is 100%
oversampled-original and runs high; unseen genuinely has more operators) — and even those came out **lower
than the originals' own** spreads. Lesson: **measure the combined set on every axis, separate manufactured
skew (fix it) from real skew (preserve it), and prefer matching the originals' levels over complementing
them for secondary features.**

---

## 5. Audit the result

Run the 20-agent, 2-axis audit (reasoning-cold + format) from `HOW_TO_AUDIT_A_NEMOTRON_CATEGORY.md` on the
kept CoTs. The corner/OOD combinations are exactly where new solver bugs and template-doc gaps hide.

## Bonus — augmentation finds solver bugs

Synthetic OOD combinations exercise code paths the original corpus never hit. The equation_numeric run
surfaced a real solver **crash** (exotic query op *combined with* a sign-confirmed ~sub, at the digit-count
step) — diagnosed, fixed, and verified GT-neutral on the original 732. Expect this; it's a feature.

---

## For the cryptarithm agent

Apply the identical pipeline to cryptarithm:
1. **Generate** crypt `question.txt` with a known symbol→digit cipher (+ reading), rendered via the crypt
   solver's own helpers so each equation reads back exactly; gold = apply the cipher to the query; validity
   gates (the cipher must be consistent across examples, digits distinct, no leading-zero violations unless
   intended). Hidden cipher never written into the puzzle.
2. **Blind-solve** via sub-agents on `question.txt` only (gold withheld).
3. **GT-match** filter to the recovered ones.
4. **Distribution + decorrelation** (§4 / §4b): complement-in-moderation; then decorrelate **every** feature
   from the puzzle type — for cryptarithm that means reading, #equations, #distinct symbols, word/operand
   length, leading-symbol constraints, carry depth, *every* surface cue — injected at the same rate in each
   type, **no forced corner subtypes**, structural axes balanced, GT-match subsample stratified by the
   at-risk feature. Oversample un-synthesizable classes. **Also synthesize GT-True instances of any scenario
   the original carries only on GT-False (excluded) rows** (§4b) — for cryptarithm, ciphers/readings that are
   ambiguous in the originals — so those cells don't vanish from training. Then **measure per-type feature
   tables on the combined set and confirm they're flat** before you ship.
5. **Audit** the kept CoTs.
