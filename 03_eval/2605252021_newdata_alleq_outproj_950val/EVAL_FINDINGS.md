# Eval findings — numeric equation (run `2605252021_newdata_alleq_outproj_950val`)

Scope: the **numeric equation** puzzle only (literal-number operands — the type we built). In this eval
that is the `equation_transformation` rows whose operands are numbers. (The symbol-operand rows in that
same eval category are a different puzzle and are out of scope here.)

## 1. Headline

- **84** numeric equation puzzles in the val set. **All 84 are in our training set; 0 are held out.**
- **The model reproduces our boxed answer on 72/84 = 85.7%.** (Ground truth is not the question — we want
  the model to reproduce OUR answer; matching it, even if both are "wrong" vs GT, is success.)
- **But 0 of 84 reproduce our token trajectory verbatim.** Every generation diverges from what we taught.
  Box-matching puzzles still have median ~99% / mean ~97% trajectory similarity (small local drift); a few
  are far lower (e.g. 1322241b = 71%, box still correct). First divergence is inside the § search in 67/84
  and in the operator-prior block in 17/84.
- Full side-by-side token diff of all 84, sorted most-divergent-first: `numeric_trajectory_diff.html`.

## 2. Why the model diverges (three causes)

**(A) The operator-frequency prior — the model never learned it, and couldn't.** Our CoT gave each symbol
its *own* family ordering from corpus frequency. Across the 732 training CoTs that is **8 distinct
orderings** (the most common is only ~30% of operator-lines; **439/732 puzzles carry a non-default order**).
That ordering is not derivable from anything in the puzzle, so there is no signal to learn "this symbol →
this order." The model overwrites it with its own internal prior (e.g. 1322241b: ours `[~sub, ~add, …]` →
model `[~add, ~sub, …]`), which cascades through every candidate-state line → the dominant trajectory
divergence. It is usually **benign** (box still correct), but it's pure noise we were paying for.

**(B) The boxed-answer format is inconsistent → the model corrupts correct answers.** A negative result is
boxed plainly (`-56`, 704/732) in most puzzles, but in 28/732 the operator symbol is re-attached as the
sign (`!32`, `64:`) with an explicit "written with the operator-symbol prefix/suffix" line. The plain-minus
case was **silent**. The model learned the re-attach and **misfires it on plain-minus puzzles**, turning a
*correct* `-56` into `5-6` (9a9f6025), `-35`→`35-` (45dbc1cc), `-6`→`6-` (4179c322).

**(C) The lock→query gap.** The operator is locked hundreds of tokens before the query is computed, and the
query line didn't restate the rule, so the model slips on operand order — e.g. 2a73a462: locked
`concat_rev` (b∥a) but applied `39∥20` instead of `20∥39`.

## 3. Fixes applied (in the duplicate folder `01_data/260526_Numeric_Equation`, original `260523` untouched)

The generator there was also switched to read each puzzle from its own `question.txt`/`answer.txt` (no CSV
dependency); it only rewrites `label.txt` and `track/tree_cot.txt`.

- **③ One global traverse order for every symbol.** Replaced the per-symbol prior with a single fixed order,
  used by the prior block, the candidate state lines, the §N.4 search, and the unseen guess. Swept all 6
  family orders for best GT-match → **`[~add, ~sub, ~mul]`** (615/732). Eliminates the 8-orderings noise so
  the prior block + state lines are one learnable string. (Trade: 618→615 reference GT-match, since the
  per-symbol tiebreak/guess is gone; GT-match is reference-only.)
- **① Uniform sign narration.** Every negative result now states its sign convention, *including* the
  plain-minus case (previously silent) — so the model can't overgeneralize the operator-symbol re-attach.
- **② Restate the locked rule at the query.** The query line now reads
  `applying g = concat_rev (b∥a)` before computing, so applying the operator isn't a memory test.

All three are narration/ordering changes; ① and ② don't change any answer, ③ shifts only ambiguous/guess
tiebreaks. Verified on 1322241b (③), 9a9f6025 (①), 2a73a462 (②).

## 4. Next step

Retrain on `260526_Numeric_Equation` (regenerate the train CSV from it first) and re-eval on the same 84
numeric puzzles, comparing **trajectory similarity** and **box-reproduction**, not GT. The expectation:
the prior-block divergence collapses (one order), the plain-minus answers stop being corrupted, and the
concat-direction slips go away. None of this is proven until that retrain+re-eval — these are hypotheses
the eval will confirm or refute.

> Note: the `260526` README / LESSONS / cot_template still describe the old per-symbol prior and CSV input;
> they need a sync if these changes are kept.
