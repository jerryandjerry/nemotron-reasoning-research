# Checkpoint Selection & Early Stopping for LLM SFT
**Date:** 2026-05-15 (Chicago)
**Context:** LoRA SFT on Nemotron-3-Nano-30B-A3B, 244-step training with linear LR decay to 0, 5 saved checkpoints (50/100/150/200/244). Classical early-stopping ("monitor val loss, halt") is rarely used; the question is **which checkpoint, or combination of checkpoints, to ship**.

This report catalogs the 7 techniques actually used in 2024–2026 practice, with sources.

---

## Three families

| Family | When applied | Cost |
|---|---|---|
| **Selection** | Post-training | ~free (just run eval) |
| **Weight averaging** | Post-training | Cheap (load + average + save) |
| **During-training tricks** | While training | Adds infra, persistent overhead |

---

## 1. Task-metric eval per checkpoint

**What:** At each saved checkpoint, run inference on a held-out probe set, score with the downstream metric (not val loss). Pick the checkpoint with the highest score.

**Why not val loss:** Perplexity / val-loss can keep improving while task accuracy and Pass@k diversity collapse. Reported widely for reasoning SFT.

**Implementation:** HF Trainer / TRL pattern — `load_best_model_at_end=True` + `metric_for_best_model="<your_metric>"`.

**Sources:**
- [TRL SFTTrainer docs](https://huggingface.co/docs/trl/sft_trainer)
- [apxml RLHF course — Evaluating SFT](https://apxml.com/courses/rlhf-reinforcement-learning-human-feedback/chapter-2-sft-phase-rlhf/evaluating-sft-performance)
- [Llama-Factory SFT docs](https://llamafactory.readthedocs.io/en/latest/getting_started/sft.html)

**Three ways to operationalize #1:**

- **(a) Eval with metric** — run `generate()` on a held-out prompt set, extract `\boxed{}`, compute exact-match. The "correct" metric for reasoning, but requires autoregressive decoding (slow, KV cache memory).
- **(b) Eval loss** — single teacher-forced forward pass on the (prompt + ground-truth completion) sequence; CE on completion tokens only (response-only mask). No generation, no KV cache. Cheap, but a proxy — val_loss-best ≠ task-best on reasoning runs.
- **(c) Save N checkpoints, run anything later** — train without inline eval; save N intermediate checkpoints to disk; after training, run (a) or (b) or weight averaging on the saved set.

**For (c): what is N, and which checkpoints?**

- **N=5 is a reasonable floor and ceiling for our scale.** All four post-hoc methods (#1 task-metric, #2 WiSE-FT, #3 last-K soup, #4 MWA) work with N=5: WiSE-FT needs 2, last-K soup uses K=4–6 in published recipes, MWA uses the same set as #1. Above ~N=10 the marginal value of an extra checkpoint is small and disk cost is real (each LoRA-r32 + lm_head ckpt ≈ 1.7 GB).
- **Evenly spaced by default.** `save_steps ≈ total_steps / N`, rounded to a clean multiple of 25 or 50 (TRAINING_ROUTINE rule). With linear LR decay, even spacing matches the schedule — each saved checkpoint represents a similar fraction of the cumulative weight change.
- **Always force-save the final step** (`FinalStepSaveCallback` equivalent) regardless of the `save_steps` boundary. Otherwise step N×save_steps may not land exactly on `num_steps`.

---

## 2. WiSE-FT for reasoning (early × late weight interpolation)

**What:** Pick an early-training checkpoint and a late-training checkpoint. Interpolate their weights linearly:
```
W_merged = (1 − δ) · W_early + δ · W_late
```
Default δ=0.5; sweep 0.3, 0.5, 0.7. For LoRA: `model.set_adapters(adapter_weights=[1-δ, δ])`.

**Why:** Directly targets the math/reasoning SFT failure mode where Pass@k diversity collapses as Pass@1 keeps rising. Recovers diversity while keeping Pass@1.

**Reported effect:** +5–7% on MATH500 with test-time scaling (Pass@8).

**Sources:**
- **Paper:** [arXiv 2504.10478 — "Robust LoRA Soup for Reasoning SFT"](https://arxiv.org/html/2504.10478v4)
- **Tooling:** [HF PEFT model merging docs](https://huggingface.co/docs/peft/en/developer_guides/model_merging)

---

## 3. Last-k uniform weight averaging ("trajectory soup")

**What:** Average the last K saved checkpoints, uniformly. `K = 4–8` typical.

**Why:** Linear LR decay to zero is itself an implicit EMA (Bergsma ICLR'25 — `decay-to-zero` ≈ averaging-the-tail). Bus explicit averaging on top can still help, esp. when LR decay schedule is short or aggressive.

**Reported effect:** +1.64% bench avg with K=6 (Wang Nov'25, contradicting the "D2Z is sufficient" claim).

**Sources:**
- **Pretraining view:** [Bergsma et al. — "Straight to Zero" (ICLR 2025), arXiv 2502.15938](https://arxiv.org/abs/2502.15938)
- **SFT view (counter-argument):** [Wang et al. — "How LR Decay Wastes Your Best Data", arXiv 2511.18903](https://arxiv.org/abs/2511.18903)

---

## 4. Metrics-weighted averaging (MWA)

**What:** Same as last-k average, but weight each checkpoint by its eval score (or loss). Worse-performing checkpoints contribute less.

**Why:** Uniform averaging gives equal weight to bad and good checkpoints. Score-weighted soup explicitly biases toward the better candidates.

**Reported effect:** +5.05% over last-checkpoint-only, +1.76% over uniform soup, with LoRA on benchmark.

**Source:**
- **Paper:** [arXiv 2504.18580 — "Metrics-Weighted Averaging for LLM Fine-Tuning"](https://arxiv.org/pdf/2504.18580)

---

## 5. EMA / BEMA — exponential moving average of weights

**What:** Maintain an EMA of model weights *during* training. Each step:
```
W_ema = β · W_ema + (1 − β) · W_current
```
At end of training, use `W_ema` rather than `W_current`. BEMA (bias-corrected variant) fixes EMA's lag in the early-training phase.

**Why:** EMA smooths late-training oscillations and acts like averaging across many "virtual" checkpoints continuously.

**Trade-off:** Requires running EMA in parallel during training (memory + compute). More expensive than post-hoc soup, but tighter integration.

**Sources:**
- **BEMA paper:** [arXiv 2508.00180 — "Bias-Corrected EMA for Stable Pretraining"](https://arxiv.org/abs/2508.00180) (NeurIPS / OpenReview 2025)

---

## 6. Multiple held-out probes (defensive selection)

**What:** Don't rely on a single eval set (especially not the Kaggle public leaderboard). Maintain 2–3 separate diverse held-out sets and pick the checkpoint best on the **average** across all of them.

**Why:** Single-eval selection overfits to the eval set's biases. Public-leaderboard selection overfits to leaderboard bias. Diverse probes generalize better.

**Reported in:** AIMO Prize-1 winners used AMC + AIME + MATH as separate eval probes to pick checkpoints, explicitly "to avoid overfitting to the public leaderboard."

**Source:**
- **AIMO-1 writeup:** [project-numina/aimo-progress-prize on GitHub](https://github.com/project-numina/aimo-progress-prize)

---

## 7. Rollback on late instability

**What:** If training loss spikes or grad-norm explodes late in training, use an *earlier* stable checkpoint rather than the final one.

**Why:** SFT runs occasionally destabilize near the end (loss spike, divergence on a bad micro-batch, etc.). The final checkpoint may be in a worse state than step 200 or 150.

**Reported in:** AIMO Prize-2 and Prize-3 winning solutions both note teams "used earlier checkpoints after catastrophic shifts" late in training. Real and repeated pattern.

**Note**: 
- Only one step spike is not a spike, usually at least 3 step can be seen as a divergence.
- Loss spike may be due to the ood data, loss+grad norm spike may imply a divergence.

**Sources:**
- **AIMO-2 winning solution:** [arXiv 2504.16891](https://arxiv.org/pdf/2504.16891)
- **AIMO-3 entropy-weighted SC writeup:** [Kaggle writeup](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/entropy-weighted-self-consistency-for-olympiad-lev)

---

## Recommendation for our setup

We have:
- 5 checkpoints saved per run (typically only steps 200 and 244 survive FIFO; steps 50/100/150 are deleted)
- Linear LR decay to 0
- A `\boxed{}` exact-match scorer can be implemented against a held-out slice
- 2-hour training runs (we can afford the eval time)

**Highest ROI (do these first):**
1. **#1 Task-metric eval** on every saved checkpoint, post-training. If step-200 ≥ step-244, submit that. Cost: ~10 min of inference on a 200-prompt held-out set.
2. **#2 WiSE-FT** of the best two scoring checkpoints at δ=0.5, then sweep δ=0.3 and 0.7. Cost: a few minutes of LoRA merging + 3 more eval runs.

**Lower priority:**
3. **#3 Last-k soup** — only meaningful if we keep more checkpoints on disk (raise FIFO limit from 2 to 4+).
4. **#5 EMA/BEMA** — adds infra. Not worth it at our 244-step scale; the post-hoc soup captures most of it.
5. **#4 MWA** — needs the eval scores from #1 anyway; one extra averaging pass after #1 is cheap.

**Operational changes that would unlock more of this:**
- Raise FIFO checkpoint retention from 2 → 5 so all of 50/100/150/200/244 survive.
- Add a post-training eval script that runs the `\boxed{}` scorer against each saved checkpoint.
- Keep a frozen 200–300 prompt held-out set (not in the training data) for selection.

---

## Honest caveats

- **For pretraining with linear D2Z**, the final checkpoint is theoretically near-optimal (Bergsma 2025). That argument is **weaker for SFT on tiny data** — loss going 0.46 → 0.002 in 244 steps is textbook memorization, exactly the regime where #2 and #3 pay off.
- Val-loss-best ≠ task-best is **well-established for reasoning** but not codified as a standard recipe. Llama-Factory / Axolotl just expose `early_stopping_patience` + `load_best_model_at_end` and leave the metric choice to you.
- No top Kaggle LLM solution I found *publicly* uses SWA/soup. Most rely on **self-consistency + careful eval on diverse held-outs** (#1 + #6), not weight averaging.
- Probability estimate for the 0.84 baseline: step-244 best ~60–70%, step-200 best ~20–30%, step-150 best ~5–10%, step-100/50 best <5%. The cost of testing all five is small; the upside is a free leaderboard bump if 200 > 244.

---

## Summary table

| # | Technique | When | Effect (reported) | Source |
|---|---|---|---|---|
| 1 | Task-metric eval per ckpt | Post | (baseline best of N) | TRL/Llama-Factory docs |
| 2 | WiSE-FT (early × late interp) | Post | +5–7% MATH500 Pass@8 | arXiv 2504.10478 |
| 3 | Last-k uniform soup | Post | +1.64% bench avg (K=6) | arXiv 2511.18903 |
| 4 | Metrics-weighted soup | Post | +5.05% over last | arXiv 2504.18580 |
| 5 | EMA / BEMA | During | Smoother end-of-training | arXiv 2508.00180 |
| 6 | Multiple held-out probes | Post | Defensive (no leaderboard overfit) | AIMO-1 (Numina/HF) |
| 7 | Rollback on instability | Post | Recovery from late spike | AIMO-2 / AIMO-3 |
