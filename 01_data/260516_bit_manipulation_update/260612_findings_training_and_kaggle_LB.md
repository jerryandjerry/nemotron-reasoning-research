# Findings for the next bit_manipulation attempt — training history + Kaggle LB context

> Written 2026-06-12 (Chicago). Audience: the agent tasked with improving bit_manipulation.
> Everything here is sourced from completed experiments in `study/SCORE_TRACKER.md` (Subs #1–#34),
> the run artifacts referenced below, and a live leaderboard pull on 2026-06-12.

## 1. Why bit_manipulation matters right now

Live leaderboard (pulled 2026-06-12 via Kaggle API): **#1 = 0.90, three teams at 0.88, ~20 teams at 0.87.**
Our best is **0.86** (four times: public huikang adapter Sub #13, our #16, Stage 2 #32, Stage 4 #34 — never above).
So 0.86 is OUR plateau, not the eval's ceiling — at least 0.04 of proven headroom exists, and one 0.87 team
is named "Lora is all you need", so adapter-class recipes reach 0.87+.

bit_manipulation is a prime suspect for the gap: the only real-generalization measurement we have puts it at
**12% (3/25) on out-of-train val IDs** (run #7's model), while its in-train eval looked fine. See §3 of
`README.md` in this folder for the root cause: **rule ambiguity** — in 1,147 of 1,280 puzzles, multiple rules
fit all 7 examples, and the huikang CoT never demonstrates choosing between candidates.

## 2. What has been tried on bit_manipulation, with LB outcomes

### Sub #29 — failcorr augmentation (this folder's CSV) — **FAILED, −0.03**
- **Data:** `260516_bit_manipulation_failcorr.csv` (md5 `a9c2349c3cc88eddeebbc11709a443fa`). Base = #16's exact
  data (`260514_huikang_golden_stripped.csv`, md5 `92a515f31cd2e89f9cb355c4ed09f927`); only change = per-bit
  `tried <X> → ✓✗✓✓✓✓✓✓ reject; chose <Y> → accept` block inserted before `Selected` on 1,280/1,354 bitman rows
  (+~140 tokens/row). Non-bitman rows byte-identical (verified by sampling; one cryptarithm_deduce row differs, minor).
- **Recipe:** byte-identical to #16 (r32/α32, no-tie, live out_proj, lr 2e-4→0, bs32, seq8192, 246 steps).
- **Result: 0.83 vs #16's 0.86 on the same recipe.** The augmentation is the only meaningful change.
- **Mechanism (post-hoc):** training loss ran ~2× #16's at step 79 (0.008 vs 0.004) and +50% at the end
  (0.0045 vs 0.0030) — the LoRA spent capacity learning the new ✓/✗ FORMAT rather than the underlying
  elimination skill. This is a direct verification of the "Taha anti-pattern" warned about in
  `02_train/STRATEGY_0.84_to_0.86.md`: Taha hit 98.9% on his bit-serial generator but his LB dropped —
  format clash + easy-memorization absorb the rank-32 budget.
- **Full details:** `study/2606071542_sft_bitman_failcorr_rtx6000/260607_2057_submission/TRAINING_INFO.md`
  and the Sub #29 entry in `study/SCORE_TRACKER.md`.

### Sub #5 — lkevincc cryptarithm data added — collateral bitman collapse (−69pp)
- Adding 735 lkevincc cryptarithm samples crashed **bit_manipulation 74.4% → 5.0%** per-category
  (score 0.68 overall). Cause was NOT order or surface template (verified); observed symptom was the
  operator-matching commit threshold collapsing (5.2 → 0.9 successful matches/output), and 35.5% of
  equation_transformation outputs leaked lkevincc template markers.
- **Lesson: bitman is highly sensitive to cross-category format contamination.** Any new data for OTHER
  categories can silently destroy bitman.
- **Details:** `study/2605141458_sft_huikang_golden_rtx6000/260514_2003_submission/EVALUATION.md` and
  `STUDY_COT_AND_ORDER.md` (same folder) — these contain the per-category eval methodology worth reusing.

### Baseline context
- #16 (`study/2605222039_sft_moe_outproj_rtx6000`) = 0.86: r32/α32, NO-tie, LIVE out_proj
  (mixer.training=False trick), huikang golden_stripped data, 246 steps. This is the canonical recipe —
  every successful run since copies it verbatim (see memory rule: copy SFT verbatim, change the strict minimum).
- Curriculum status: continual FT via INIT_ADAPTER_FROM produced Stage 2 (#32) = 0.86 and Stage 4 (#34) = 0.86
  on crypt10k_FULL data (Stage 3 #33 = 0.84 in between). Stage 5 (epoch 3, shuffle seed 33) is training now.
  Best current adapters: Stage 4 final (`study/2606111516_sft_crypt10k_FULL_ep2_contS3_rtx6000/`) and
  Stage 2 final (`study/2606100600_sft_eqHard_FULL_contS1_rtx6000/`).

## 3. Measurement methodology — read this before judging any experiment

- **Inference noise is ±0.01 max.** Proven twice with byte-identical zip resubmits: Sub #11 endpoint
  0.83→0.84, Sub #13 public adapter 0.85→0.86. A 0.01 move means nothing.
- **A ≥0.02 LB gap is a real weight difference** (project rule, repeatedly verified) — but real ≠ caused by
  your intervention. See next point.
- **Training trajectory variance alone moves the LB.** File-level verified (2026-06-12): runs #7 vs #10 had
  bit-identical step-1 forward (loss 0.403735 both), identical data/script/single-GPU, diverged from the 4th
  decimal of step-1 grad_norm (CUDA atomicAdd nondeterminism) and scored **0.84 vs 0.76**. So a single
  −0.03 (like Sub #29) is solid evidence of "did not help" but its exact magnitude includes a trajectory
  component. **Strong claims need either a big margin or a repeat run.**
- Train loss does NOT predict LB in the converged regime: loss 0.0013 scored 0.83 (#27); loss 0.027
  scored 0.86 (#32). Don't optimize for lower train loss.
- Per-category evaluation harness exists — see Sub #5's `EVALUATION.md` for the approach (per-category
  accuracy + output-pattern forensics). The 25 out-of-train bit_man val IDs used for the 12% measurement
  are described in this folder's README and its `_solver.py` / batch files.

## 4. Constraints and traps for the next attempt

1. **Don't insert new formats into existing CoTs** (Sub #29's failure). If teaching elimination, the trace
   must look like the original distribution — or the change must be small enough that the rank-32 budget
   isn't spent on format learning.
2. **Don't let other categories' data leak templates** (Sub #5's failure). Validate non-target categories
   byte-identical after any data build.
3. **Recipe is frozen:** copy the #16/Stage trainer verbatim (r32/α32, no-tie, live out_proj, lr 2e-4→0,
   bs32, seq8192, stratified interleave). Every deviation that was tried (tying, DDP, QLoRA-4bit,
   category-prefix, DPO-distill) lost ≥0.02.
4. **Verify out_proj is alive at stress step 1** (grad norm sum > 1e-6), else the run is silently capped
   at ~0.84. Routine §1.6.
5. **Budget check:** untested ideas with the most headroom per this history:
   - data-side disambiguation: generate/keep only puzzles whose example set admits a UNIQUE fitting rule
     (kills the ambiguity root cause instead of teaching elimination);
   - elimination taught in huikang's native phrasing (no new symbols/format, no ✓/✗ glyphs);
   - continual-FT a bitman-only stage from the current 0.86 adapter (curriculum pattern is proven:
     INIT_ADAPTER_FROM + fresh lr cycle);
   - measure per-category BEFORE submitting: the 25-ID out-of-train val set distinguishes 12% vs 20%+
     locally without burning a submission.

## 5. Artifact index

| What | Where |
|---|---|
| Failcorr design doc + ambiguity analysis | `01_data/260516_bit_manipulation_update/README.md` |
| Failcorr data | `01_data/260516_bit_manipulation_update/260516_bit_manipulation_failcorr.csv` |
| Generator + near-miss solver | `01_data/260516_bit_manipulation_update/_solver.py`, `_batch_*.json` |
| Sub #29 run (failcorr, 0.83) | `study/2606071542_sft_bitman_failcorr_rtx6000/` |
| Sub #5 bitman-collapse forensics | `study/2605141458_sft_huikang_golden_rtx6000/260514_2003_submission/EVALUATION.md` |
| #16 canonical recipe (0.86) | `study/2605222039_sft_moe_outproj_rtx6000/` |
| Score history (all 34 subs) | `study/SCORE_TRACKER.md` |
| Taha anti-pattern warning | `02_train/STRATEGY_0.84_to_0.86.md` |
| Training routine (must follow) | `02_train/TRAINING_ROUTINE.md` |
| New BitM workspace (started 2026-06-12 09:45) | `01_data/260612_BitM/` |
