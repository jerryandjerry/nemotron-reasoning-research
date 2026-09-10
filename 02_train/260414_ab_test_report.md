# A/B Test Report

## Baseline
- **Run:** `2604121541_sft_qlora_unsloth_konbu17_rtx6000` | **Score:** 0.70
- **Config:** Nemotron-3-Nano-30B-A3B, QLoRA Unsloth, r=32, alpha=32, lr=1.414e-4, warmup=0.05, adamw_torch, bs=22, ga=1, seq=2000, ep=2, seed=123, konbu17 data (2905 samples), standard CE loss

## Summary

| # | Date | Variant | Change | Score | Delta |
|---|---|---|---|---|---|
| — | 2026-04-12 | **baseline** | standard CE, 2907 samples, 2ep, lr1.414e-4 | **0.70** | — |
| 1 | 2026-04-13 | boxed5x | loss: 5x weight on `\boxed{}` region | 0.68 | -0.02 |
| 2 | 2026-04-13 | focal_g2 | loss: focal loss gamma=2 | 0.68 | -0.02 |
| 3 | 2026-04-13 | topk32 | loss: top-32 worst tokens only | 0.67 | -0.03 |
| 4 | 2026-04-13 | softmax_t5 | loss: log-sum-exp tau=5 | 0.50 | -0.20 |
| 5 | 2026-04-14 | alldata_2xbit_2xeq_ep1 | data: all 7365 (2x bit/eq), 1ep | 0.70 | 0.00 |
| 6 | 2026-04-14 | alldata_2xbit_2xeq_ep2_lr1e4 | test5 + 1 more epoch, lr=1e-4 | 0.70 | 0.00 | clealy overfit but doesn't hurt, which means hard problem is still hard, while easy problem is still 100% correct.

---

## Test 1: boxed5x — 5x weighted boxed-answer CE | 0.68 (-0.02)
- **Change:** `compute_loss_func` only. All else identical to baseline.
- **Rationale:** Kh0a's 0.73 notebook uses this. `\boxed{}` region is eval-critical — wrong token = wrong submission. 5x weight there, 1x elsewhere.
- **Effect:** Slightly worse. Non-boxed tokens still get weight 1x, but normalization (`sum / weighted_sum`) effectively lowers their contribution. -0.02 could be this dilution effect or noise.
- **Note:** Kh0a's 0.73 uses same loss but differs in data (custom distilled traces, category-balanced ~3910 samples), seq=3500, bs=2, dropout=0.1, early stopping w/ validation.
- **Source:** Kh0a notebook `llkh0a/nemotron-unsloth-sft-training-3-30-2`

---

## Test 2: focal_g2 — Focal loss gamma=2 | 0.68 (-0.02)
- **Change:** `compute_loss_func` only. All else identical to baseline.
- **Rationale:** Downweight confident tokens (high p), upweight uncertain tokens (low p). Smooth approximation of "maximize min logprob" (THK #689915).
- **Effect:** Slightly worse. Reduces gradient from tokens the model already predicts well. -0.02 similar to boxed5x — any reweighting away from uniform may slightly hurt.
- **Source:** THK discussion #689915, focal loss (Lin et al. 2017)

---

## Test 3: topk32 — Top-K=32 worst tokens | 0.67 (-0.03)
- **Change:** `compute_loss_func` only. All else identical to baseline.
- **Rationale:** Per sequence, average only the 32 highest-CE tokens. Train only on what the model gets wrong.
- **Effect:** Worse. Discards ~97% of training signal. The 32 worst tokens per sequence are often noise (rare symbols, unpredictable digits), not eval-important tokens.

---

## Test 4: softmax_t5 — Log-sum-exp tau=5 | 0.50 (-0.20)
- **Change:** `compute_loss_func` only. All else identical to baseline.
- **Rationale:** Replace mean-CE with soft-max: `(1/tau) * logsumexp(tau * CE)`. As tau grows, approaches max-CE per sequence.
- **Effect:** Much worse. Loss dominated by extreme outlier tokens; reasoning chain tokens have moderate CE and get near-zero gradient. Effectively destroys the fine-tuning.

---

## Findings (Tests 1–4: Loss Function)
- All 4 loss variants scored equal or worse than baseline.
- boxed5x and focal_g2 at -0.02 — unclear if significant or noise.
- topk32 and softmax_t5 clearly hurt — discarding/ignoring most training signal damages the model.
- The loss function alone does not explain the gap between our 0.70 and Kh0a's 0.73. Kh0a differs in multiple dimensions: training data, seq length, batch size, dropout, early stopping.

---

## Test 5: alldata_2xbit_2xeq ep1 | 0.70 (0.00)
- **Change:** Data only. Use all 6558 rows (no type-balanced caps), duplicate Bit Manipulation (607→1214) and Equation Transformation (200→400). 1 epoch to keep total steps comparable (335 vs baseline 266).
- **Rationale:** Baseline undersamples the two hardest categories (Bit Manipulation: all 607 used, Equation Transformation: all 200 used — both fully used, no headroom). Using all data + duplicating the minority categories gives more exposure to bit/eq reasoning. 1 epoch to avoid over-training on duplicates.
- **Effect:** No change. Exposing the model to 2.5x more samples (7365 vs 2907) and doubling bit/eq representation didn't move the score. Either the baseline data is already sufficient, or 1 epoch isn't enough for the model to benefit from the extra samples.
- **Data distribution:** Gravity 1511, Numeral 1491, Text Enc 1407, Unit 1342, Bit 1214 (2x), Eq 400 (2x) = 7365

---

## Test 6: alldata_2xbit_2xeq ep2 lr=1e-4 | 0.70 (0.00)
- **Change:** Continuation of test 5 for a 2nd epoch with lr=1e-4 (lower than test 5's 1.414e-4). Same data, resumed from test 5's final adapter (checkpoint-335).
- **Rationale:** Test whether a second pass over the same data with lower lr consolidates learning and improves score. Train loss dropped from test 5's 0.25 avg to 0.15 avg (after full 670 steps).
- **Effect:** No change. Despite much lower training loss (0.15 vs 0.25), eval score is identical. The model is fitting the training data better but not generalizing differently — likely overfitting on details that don't transfer to eval.

---

## Findings (Tests 5–6: Data)
- Using all data + duplicating underrepresented categories (Test 5) didn't improve over the baseline's type-balanced 2907-sample subset.
- Training longer on the same data (Test 6) lowered train loss dramatically but didn't improve eval — evidence of overfitting without generalization.
- Neither data scaling nor extended training moves the score off 0.70 with the current recipe. Suggests the bottleneck is elsewhere (data *quality* vs quantity, or model/config limits).

---

## Overall Findings
- **4/6 tests hurt, 2/6 matched baseline, 0/6 improved.** Score sits firmly at 0.70.
- Loss function reweighting: no win, sometimes catastrophic.
- Data scaling + duplication: no change, even with lower train loss.
- The 0.70→0.73 gap to Kh0a is not closed by loss function or by our data duplication strategy. Kh0a's advantage likely comes from **data quality** (custom distilled traces, not just more samples) + **longer seq** (3500) + **early stopping with validation**.
