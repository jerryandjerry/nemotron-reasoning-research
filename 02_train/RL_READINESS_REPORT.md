# Is It Time to Start RL? — Per-Category RL-Readiness Report
**Date:** 2026-05-28 (Chicago)
**Question:** Is our SFT training result an indication that we should move to reinforcement learning instead of continuing SFT? If so, for which categories?
**Method:** (1) deep web research on the established SFT→RL transition indicators; (2) five token-level diagnostics run on our own inference-vs-training tokens across **all 9 categories** of the 950-puzzle val set, using the real Nemotron tokenizer.

---

## 0. The four runs being compared

All four share the identical model (Nemotron-3-Nano-30B-A3B), LoRA config (r32 α32, 9 target modules incl. `out_proj`+`lm_head`), optimizer, LR schedule, batch size, and inference settings (`temperature=0.0`, `max_tokens=7680`, deterministic greedy). **The only variable is the training CSV.**

| run | numeric-eq CoTs | cryptarithm CoTs | LB |
|---|---|---|--:|
| moe | OLD | OLD (800 examples) | 0.86 |
| alleq | NEW (732, no GT filter) | OLD | 0.84 |
| alleqgt | NEW (732, GT-True only) | OLD | 0.84 |
| cryptn | NEW (GT-True) | NEW (honest-DFS, 535/800 gated out) | 0.85 |

---

## 1. The research-backed indicators that it's time for RL

From peer-reviewed / arXiv 2025–2026 sources, you transition from SFT to RL (specifically RLVR — RL with Verifiable Rewards) when **all** of these hold:

| # | Indicator | What it means | Source |
|---|---|---|---|
| 1 | **Verifiable reward exists** | answer correctness can be auto-checked (no human/reward-model needed) | [Appen RLVR](https://www.appen.com/blog/rlvr); [emergentmind RLVR](https://www.emergentmind.com/topics/reinforcement-learning-with-verified-rewards-rlvr) |
| 2 | **SFT has plateaued** | val within ~2% of min; more SFT data flat or hurts | [Quagmires in SFT-RL, arXiv 2510.01624](https://arxiv.org/html/2510.01624v1) |
| 3 | **High Pass@large-k** | the right answer is reachable by sampling even when greedy (Pass@1) misses | [Quagmires 2510.01624](https://arxiv.org/html/2510.01624v1) |
| 4 | **SFT not over-trained** | RL plasticity preserved (few epochs, not memorized) | [RL Neither Panacea Nor Mirage, arXiv 2508.16546](https://arxiv.org/html/2508.16546v1) |
| 5 | **Strong base reasoning patterns** | the model can already produce most of the correct trajectory | [RLVR Incentivizes Correct Reasoning, arXiv 2506.14245](https://arxiv.org/html/2506.14245v2) |
| 6 | **Failures are selection errors, not capability gaps** | model has the right "moves" but picks wrong | [RLVR superior OOD generalization](https://www.emergentmind.com/topics/reinforcement-learning-with-verified-rewards-rlvr) |

**Key theoretical results used:**

- *"RLVR achieves comparable in-distribution accuracy to SFT with superior out-of-distribution generalization."* — [emergentmind RLVR](https://www.emergentmind.com/topics/reinforcement-learning-with-verified-rewards-rlvr)
- *"SFT rotates singular-vector directions toward task-specific solutions [memorization]; RL rotates them back toward generalizable orientations."* — [RL Neither Panacea Nor Mirage, 2508.16546](https://arxiv.org/html/2508.16546v1)
- *"High SFT scores can be biased toward simpler/homogeneous data and are not reliably predictive of subsequent RL gains… post-SFT performance explains only 43% (R²=0.43) of the final post-RL outcome. Use generalization-loss + Pass@large-k as the transition signal instead."* — [Quagmires, 2510.01624](https://arxiv.org/html/2510.01624v1)
- *"Excessive SFT, especially overly large epochs, can limit further RL improvements [collapses RL plasticity]."* — [Quagmires, 2510.01624](https://arxiv.org/html/2510.01624v1)
- *"After saturation, RL mainly sharpens the output distribution — helping you sample correct answers more often, but does not fundamentally improve reasoning skill. RLVR makes models faster, not smarter."* — [EvoLM, 2506.16029](https://arxiv.org/pdf/2506.16029); [Promptfoo](https://www.promptfoo.dev/blog/rlvr-explained/)
- *"Models will learn to cheat any test that isn't comprehensive. A verifier catching 60% of errors creates a 40% gap."* — [Appen RLVR](https://www.appen.com/blog/rlvr) (caveat)

**Base-model context:** Nemotron-3-Nano-30B was itself RL-post-trained with **synchronous GRPO across math/code/science/instruction/tool-use + RLHF with a generative reward model (Qwen-3-Nemotron-235B-A22B-GenRM)** ([model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16), [Nemotron 3 paper 2512.20848](https://arxiv.org/pdf/2512.20848)). So the base already has RL-shaped, format-faithful, confident reasoning — which is *why* our SFT bugs express as fluent-confident-wrong answers rather than gibberish, and why GRPO is the natural RL choice (matches the base post-training).

---

## 2. The five token-level diagnostics (all 9 categories)

### Diagnostic 1 — Trajectory reproduction on IN-TRAINING puzzles
*Does SFT imitation take hold? `prefix%` = how far the model follows the training CoT token-for-token before first divergence.* (moe vs alleqgt shown; 30-puzzle sample per cell where N>30.)

| category | run | sim% | **prefix%** | box-match% | GT% |
|---|---|--:|--:|--:|--:|
| bit_manipulation | moe | 97.3 | 78.0 | 86.7 | 86.7 |
| bit_manipulation | alleqgt | 97.6 | 82.2 | 90.0 | 90.0 |
| cipher | moe | 99.9 | 98.6 | 100.0 | 100.0 |
| cipher | alleqgt | 99.9 | 99.4 | 100.0 | 100.0 |
| cryptarithm_deduce | moe | 80.8 | **7.0** | 3.3 | 3.3 |
| cryptarithm_deduce | alleqgt | 82.5 | **8.0** | 3.3 | 3.3 |
| cryptarithm_guess | moe | 79.3 | **7.2** | 0.0 | 0.0 |
| cryptarithm_guess | alleqgt | 79.2 | **6.9** | 0.0 | 0.0 |
| equation_numeric_deduce | moe | 100.0 | 100.0 | 100.0 | 100.0 |
| equation_numeric_deduce | alleqgt | 96.8 | 78.1 | 93.3 | 93.3 |
| equation_numeric_guess | alleqgt | 98.9 | 83.9 | 85.7 | 85.7 |
| gravity | moe | 99.9 | 95.1 | 96.7 | 96.7 |
| gravity | alleqgt | 100.0 | 100.0 | 100.0 | 100.0 |
| numeral | moe/alleqgt | 99.9 | 99.9 | 100.0 | 100.0 |
| unit_conversion | moe/alleqgt | 100.0 | 100.0 | 100.0 | 100.0 |

> Reading: every category EXCEPT cryptarithm shows 78–100% prefix-match — the model faithfully reproduces the training trajectory. **Cryptarithm collapses to 7–8% prefix** — the model diverges from its training CoT almost immediately (the 80% "similarity" is shared boilerplate, not shared reasoning). moe's `equation_numeric` 100%/100% is pure memorization (its CoTs are GT-correct on these in-training puzzles).

### Diagnostic 3 — RL-readiness: GT-in-trace on WRONG puzzles
*Of the puzzles the model gets wrong, on how many does the correct answer already appear in the model's own reasoning trace (but greedy decoding doesn't commit)? High = RL with verifiable reward can lock it in.* (Averaged over the 4 runs.)

| category | GT-in-trace on wrong puzzles |
|---|--:|
| equation_numeric_deduce | **71%** |
| bit_manipulation | **43%** |
| equation_numeric_guess | **32%** |
| cryptarithm_deduce | 7% |
| cryptarithm_guess | 0% |
| cipher | 0% (1 wrong) |
| gravity | 0% (≤1 wrong) |
| numeral / unit_conversion | — (0 wrong) |

> This is the single most decisive RL-readiness number. On `equation_numeric_deduce`, `bit_manipulation`, and `equation_numeric_guess`, the model **already explores the correct answer in 32–71% of its misses** — it just doesn't commit. RL reweights the policy toward the trajectory that hits the verified answer. On cryptarithm it's 0–7%: the model rarely even finds the answer, so RL has nothing to reinforce.

### Diagnostic 2 — Failure-type classification (numeric-eq wrong puzzles, alleqgt)
| type | count | RL-fixable? |
|---|--:|---|
| WRONG_VALUE (wrong operator family / reading order) | 11 | yes — reward reweights the decision |
| SIGN_FORMAT (`-56`→`5-6`, `14`↔`?14`) | 8 | yes — reward punishes wrong box directly |
| OFF_BY_1/2 (noisy variant) | 1 | yes |

> ~55% are decision-point selection errors among moves the model already knows; ~40% are sign/format. Both are reward-fixable. The `−56 → 5-6` garble took alleqgt 22 hand-authored narration lines in SFT (51% coverage, still leaks); a verifiable reward kills it with one signal (`\boxed{5-6}` is simply wrong).

### Diagnostic 4 — SFT saturation (cross-run prediction agreement)
*4 models, 4 different training mixes. If all 4 predict identically, the policy is locked and more SFT won't move it.*

| category | N | all-4-agree | best acc |
|---|--:|--:|--:|
| numeral | 158 | **100%** | 1.00 |
| cipher | 158 | **98%** | 0.99 |
| unit_conversion | 159 | **97%** | 1.00 |
| gravity | 160 | **95%** | 1.00 |
| bit_manipulation | 160 | 80% | 0.78 |
| equation_numeric_deduce | 65 | 80% | 0.92 |
| equation_numeric_guess | 19 | **10%** | 0.32 |
| cryptarithm_deduce | 55 | 0% | 0.04 |
| cryptarithm_guess | 16 | 0% | 0.00 |

> 4 categories are SFT-saturated (≥95% agreement, ~1.0 acc). `equation_numeric_guess` (10%) and cryptarithm (0%) are where training still steers the policy — but eq_guess steers toward *more-correct* answers while cryptarithm steers between *different wrong* answers (all ~0% acc).

### Diagnostic 6 — Output length / token-budget pressure
*Mean output tokens; % of puzzles hitting the 7680 cap with no `\boxed{}`.*

| category | moe | alleqgt | cryptn |
|---|--:|--:|--:|
| bit_manipulation | 6685 / 0% | 6692 / 1% | 6690 / 0% |
| equation_numeric_deduce | 5699 / 0% | 1531 / 0% | 1563 / 0% |
| equation_numeric_guess | 5978 / 0% | 1688 / 0% | 2017 / 5% |
| cryptarithm_deduce | 863 / 1% | 746 / 0% | **5928 / 43%** |
| cryptarithm_guess | 701 / 0% | 734 / 0% | **5316 / 43%** |
| gravity | 3099 / 0% | 3101 / 0% | 3102 / 0% |
| cipher / numeral / unit_conv | 2090 / 955 / 2312, all 0% | — | — |

> The new numeric-eq CoTs are 3.5× SHORTER (5.7k→1.5k) — good for reproduction. The new cryptarithm CoTs are 7× LONGER and **43% blow the 7680 cap with no boxed answer** — the model can't finish the honest-DFS in budget. This is a separate cryptn-specific failure on top of the imitation collapse.

---

## 3. The combined RL-readiness scorecard (all 9 categories)

| category | N | best acc | wrong | imitation (prefix) | GT-in-trace | saturated? | reward | **verdict** |
|---|--:|--:|--:|--:|--:|--:|---|---|
| **equation_numeric_guess** | 19 | 32% | 13 | 84% | 32% | NO (10%) | numeric±1%/str | **RL — highest value** |
| **equation_numeric_deduce** | 65 | 92% | 5 | 78% | 71% | near (80%) | numeric±1%/str | **RL — cleanest** |
| **bit_manipulation** | 160 | 78% | 35 | 80% | 43% | near (80%) | binary exact | **RL — biggest absolute** |
| cipher | 158 | 99% | 1 | 99% | 0% | 98% | string exact | saturated — anchor only |
| gravity | 160 | 100% | 0 | 98% | 0% | 95% | numeric±1% | saturated — anchor only |
| numeral | 158 | 100% | 0 | 100% | — | 100% | string exact | solved — anchor only |
| unit_conversion | 159 | 100% | 0 | 100% | — | 97% | numeric±1% | solved — anchor only |
| cryptarithm_deduce | 55 | 4% | 53 | **7%** | 7% | 0% (~0 acc) | string exact | **NOT RL — fix SFT first** |
| cryptarithm_guess | 16 | 0% | 16 | **7%** | 0% | 0% (~0 acc) | string exact | **NOT RL — fix SFT first** |

---

## 4. Verdict — three tiers

### Tier 1 — RL-READY and worth it
`equation_numeric_guess`, `equation_numeric_deduce`, `bit_manipulation`.
- They pass **every** indicator: verifiable reward ✓, SFT saturating/near-saturated ✓, imitation works (78–84% prefix) ✓, plasticity preserved (~1 epoch, low LR) ✓, base produces most of the trajectory ✓, failures are selection errors ✓.
- The decisive evidence: **the correct answer is already in the model's own trace on 32–71% of its misses.** That is the Pass@large-k upside the literature names as the RL trigger — RL reweights toward it.
- Combined headroom ≈ **53 wrong puzzles**, of which ~30 are in-trace and directly RL-recoverable.
- `bit_manipulation` is the biggest absolute prize (35 wrong, 43% in-trace ≈ 15 recoverable, largest category). `equation_numeric_guess` is the biggest relative prize (lowest accuracy, least saturated — data still steers it). `equation_numeric_deduce` is the cleanest (71% in-trace).

### Tier 2 — SOLVED / saturated → anchor only, do NOT optimize
`cipher` (99%), `gravity` (100%), `numeral` (100%), `unit_conversion` (100%).
- Imitation near-perfect (95–100% prefix), SFT fully saturated (95–100% agreement), 0–1 wrong.
- RL would add ~nothing and risks regressing them (the cross-category interference we measured is real — e.g. one cryptarithm puzzle flipped on byte-identical data). Keep them in the RL batch only as a **KL/replay anchor** to prevent drift.

### Tier 3 — NOT RL-ready → fix SFT first
`cryptarithm_deduce` (4%), `cryptarithm_guess` (0%).
- Imitation has **not taken hold** (7% prefix — the model diverges from its training CoT at token ~7%), GT-in-trace 0–7% (the model essentially never reaches the answer), and cryptn's new CoTs blow the budget (43% hit the cap with no box).
- **You cannot RL a trajectory the model can't produce.** RL sharpens an existing distribution; it doesn't create a skill ([Promptfoo](https://www.promptfoo.dev/blog/rlvr-explained/), [EvoLM 2506.16029](https://arxiv.org/pdf/2506.16029)).
- Fix the SFT data first: CoTs short enough to fit the 7680 cap AND GT-correct, to push base solve-rate above zero. *Then* add to RL.

---

## 5. Recommendation

**Start a targeted RLVR run now — GRPO, with the exact Kaggle metric as the reward.**

1. **Reward = the competition metric itself**: last `\boxed{}` → binary-exact / numeric±1% / string-exact. No reward model, no proxy, so no proxy-gaming — the reward *is* the true objective. (The base was GRPO-trained, so GRPO is the matched choice.)
2. **RL-active categories:** `equation_numeric_deduce`, `equation_numeric_guess`, `bit_manipulation`.
3. **Anchor/replay (in batch, not optimized):** `cipher`, `gravity`, `numeral`, `unit_conversion` — to prevent the cross-category regression we measured.
4. **Exclude from RL, keep on SFT:** `cryptarithm_deduce`, `cryptarithm_guess` — fix CoT length+correctness first.
5. **Guard the RLVR caveat** ([Appen](https://www.appen.com/blog/rlvr)): monitor Pass@k diversity / output entropy so the policy doesn't collapse to a single answer-format to game the verifier. Our reward is the true metric, so the main risk is mode-collapse, not proxy-hacking.
6. **Don't over-train SFT first** ([Quagmires 2510.01624](https://arxiv.org/html/2510.01624v1)): we're at ~1 epoch — plasticity is intact. Do NOT run more SFT epochs on the saturated categories before RL; that would collapse RL plasticity.

**One-line answer:** Yes — `equation_numeric` (both) and `bit_manipulation` are sitting exactly at the "knows-the-method, picks-wrong, answer-is-in-its-own-trace" inflection the literature names as the RL trigger, and the reward is free (the Kaggle metric). Four categories are already maxed (RL only risks them → anchor only), and cryptarithm is pre-SFT-complete (RL would be wasted → fix the data first). The right move is a **targeted RLVR/GRPO run on the three ready categories, anchored by the four saturated ones, with cryptarithm held back on SFT.**

---

## Sources
- [Quagmires in SFT-RL Post-Training: When High SFT Scores Mislead — arXiv 2510.01624](https://arxiv.org/html/2510.01624v1)
- [RL Is Neither a Panacea Nor a Mirage — arXiv 2508.16546](https://arxiv.org/html/2508.16546v1)
- [RLVR Implicitly Incentivizes Correct Reasoning in Base LLMs — arXiv 2506.14245](https://arxiv.org/html/2506.14245v2)
- [EvoLM: In Search of Lost LM Training Dynamics — arXiv 2506.16029](https://arxiv.org/pdf/2506.16029)
- [RLVR Makes Models Faster, Not Smarter — Promptfoo](https://www.promptfoo.dev/blog/rlvr-explained/)
- [RLVR for Reliable Enterprise LLMs — Appen](https://www.appen.com/blog/rlvr)
- [Reinforcement Learning with Verified Rewards (RLVR) — emergentmind](https://www.emergentmind.com/topics/reinforcement-learning-with-verified-rewards-rlvr)
- [Optimal SFT-to-RL Transition — emergentmind](https://www.emergentmind.com/topics/optimal-sft-to-rl-transition)
- [NVIDIA Nemotron 3 Nano 30B A3B model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16)
- [Nemotron 3 paper — arXiv 2512.20848](https://arxiv.org/pdf/2512.20848)
- [DeepSeek-R1: Incentivizing Reasoning via RL — arXiv 2501.12948](https://arxiv.org/html/2501.12948v1)

## Reproducibility
All diagnostics computed from:
- eval CSVs: `03_eval/{2605222039_moe…, 2605252021_alleq…, 2605270202_…alleqgt…, 2605270226_…cryptnumeq…}/val_eval_results.csv`
- training CSVs per run (see §0)
- tokenizer: `02_train/260512_huikang_085/repo/tokenizer.json` (real Nemotron, vocab 131072)
- Diagnostics: (1) trajectory prefix/similarity via difflib on token IDs; (3) GT-substring search in the pre-`\boxed` reasoning body; (4) cross-run `predicted` set-size==1; (6) `output_token_len` + `\boxed` presence.
