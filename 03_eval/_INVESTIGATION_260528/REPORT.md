# Why Improving the Data Made the Score Worse — Full Investigation
**Date:** 2026-05-28 (Chicago)
**Scope:** 4 fine-tune runs (moe, alleq, alleqgt, cryptnumeq) on Nemotron Nano 30B MoE, val=950, focus on numeric_equation + cryptarithm.

---

## Artifacts

- **HTML token-by-token side-by-side**, real Nemotron tokenizer:
  - [token_diff_eq.html](token_diff_eq.html) — 18 numeric-eq disagreement puzzles × 4 runs (1.1 MB)
  - [token_diff_crypt.html](token_diff_crypt.html) — 3 cryptarithm disagreement puzzles × 4 runs (5.9 MB)
- **Sub-investigations** (full markdown):
  - [cryptarithm_3puzzle_forensic.md](cryptarithm_3puzzle_forensic.md) — per-puzzle / per-run forensic
  - [cryptarithm_mass_collapse.md](cryptarithm_mass_collapse.md) — training-mass swing analysis
- **Web research** — research-grounded mechanisms (inline in this report, §10).

---

## 1. The four runs at a glance

| run | training CSV | Σ oversampling | new numeric-eq | new cryptarithm |
|---|---|--:|---|---|
| moe | `260514_huikang_golden_stripped.csv` | 7849 | — (old) | — (old) |
| alleq | `260525_huikang_NumericEq_allEq.csv` | 8020 | yes (171 new at os=1, full 732) | — (old) |
| alleqgt | `260527_huikang_NumericEq_allEq_gtTrue.csv` | 7883 | yes (gtTrue filter, 113 wrong zeroed) | — (old) |
| **cryptn** | `260526_Cryptarithm_TrainData/260527_huikang_NumericEq.csv` | **7613** | yes (gtTrue) | **yes (800 swapped, gate zeroed 535)** |

Cryptarithm training mass relative to moe:
- moe / alleq / alleqgt: 639 (deduce) + 161 (guess) — **byte-identical training CoTs**
- **cryptn**: 486 (deduce, **−23.9%**) + 44 (guess, **−72.7%**)

This is the entire story in one table: alleqgt has byte-identical cryptarithm training to moe and yet flips one cryptarithm puzzle; cryptn deliberately drops cryptarithm training mass by 24% / 73% and collapses to 0/55. The mechanism in each case is different and both are below.

---

## 2. Top-line val table

| | val OVERALL | num-eq (84) | cryptarithm_deduce (55) | cryptarithm_guess (16) | Public LB |
|---|--:|--:|--:|--:|--:|
| moe | 0.8621 | 61 (72.6%) | 2 (3.6%) | 0 | **0.86** |
| alleq | 0.8611 | 63 (75.0%) | 1 (1.8%) | 0 | 0.84 |
| alleqgt | 0.8632 | 64 (76.2%) | 1 (1.8%) | 0 | 0.84 |
| cryptn | 0.8653 | 64 (76.2%) | **0 (0%)** | 0 | 0.85 |

---

## 3. The val set is a 71% data leak — this changes everything

**671 of 950 val ids are also in each run's training CSV** (verified by prompt-equality, not just id-match):

| category | val ∩ moe training | leak % |
|---|---|---|
| cipher | 158/158 | **100.0%** |
| cryptarithm_guess | 16/16 | **100.0%** |
| cryptarithm_deduce | 52/55 | 94.5% |
| equation_numeric_deduce | 60/65 | 92.3% |
| bit_manipulation | 135/160 | 84.4% |
| gravity | 97/160 | 60.6% |
| unit_conversion | 88/159 | 55.3% |
| numeral | 64/158 | 40.5% |
| **equation_numeric_guess** | **1/19** | **5.3%** |

The val set is **not held out** from training — the model is being scored mostly on puzzles it memorized. The Public LB likely has less overlap, which is why the val and LB rankings disagree.

`equation_numeric_guess` is the one category that is *actually* out-of-distribution (only 1/19 leaked). It's the category where our new CoTs improved most (1/19 → 5–6/19), and it's the *only* category where you can read real generalization gain off the val numbers.

---

## 4. Trajectory reproduction: in-training (follow) vs out-of-training (generalize)

Splitting val numeric-eq by whether the puzzle was IN that run's training set:

| run | in-train (n) | GT in-train | GT out-of-train | traj-rep in-train | traj-rep out-of-train |
|---|--:|--:|--:|--:|--:|
| moe | 61 | **61/61 (100%)** | **0/23 (0%)** | 60/61 (98%) | 5/23 (22%) |
| alleq | 84 | 63/84 (75%) | — | 74/84 (88%) | — |
| alleqgt | 70 | 64/70 (91%) | 0/14 (0%) | 64/70 (91%) | **10/14 (71%)** |
| cryptn | 70 | 64/70 (91%) | 0/14 (0%) | 64/70 (91%) | **12/14 (86%)** |

`trajectory_rep` = `model.predicted == box(our solver_cot)` (per [[feedback_eval_reproduction_not_gt]]).

Same for cryptarithm:

| run | in-train (n) | GT in-train | GT out-of-train | traj-rep in-train | traj-rep out-of-train |
|---|--:|--:|--:|--:|--:|
| moe | 68 | 2/68 (3%) | 0/3 (0%) | 3/68 (4%) | 0/3 (0%) |
| alleqgt | 68 | 1/68 (1%) | 0/3 (0%) | 2/68 (3%) | 0/3 (0%) |
| cryptn | **19** | **0/19 (0%)** | 0/52 (0%) | **0/19 (0%)** | 0/52 (0%) |

What this means:

- **moe achieves its numeric-eq score purely by memorization** (100% on its 61 in-training val puzzles, 0% on the 23 out-of-training).
- **Our variants lose ~9pp of memorization but gain ~3× generalization** (alleqgt 71% / cryptn 86% trajectory rep on unseen puzzles vs moe's 22%). The model genuinely learned our method as a method, not as a lookup.
- For cryptn cryptarithm, even the **19 in-training puzzles produce 0/19 GT and 0/19 trajectory rep** — the model trained on those CoTs and still cannot reproduce them. The new cryptarithm CoTs are not transferring as a learnable procedure (more in §7).

---

## 5. Numeric-eq: the 18 disagreement puzzles ([HTML diff](token_diff_eq.html))

Of the 84 val numeric-eq puzzles, 66 are unanimous across all 4 runs. The action is in 18:

**Wins (our variants beat moe) — 7 puzzles**, all unseen_operator or ambiguous. moe predicts a digit / fragment because it has no method; our variants emit our solver's answer:

| id | label | GT | moe → ours |
|---|---|---|---|
| [053f4545](../../01_data/260526_Numeric_Equation/equation_numeric_053f4545/track/tree_cot.txt) | unseen_operator | 731 | 71 → **731** |
| [1c4861e6](../../01_data/260526_Numeric_Equation/equation_numeric_1c4861e6/track/tree_cot.txt) | unseen_operator | 993 | 79 → **993** |
| [ea6d926a](../../01_data/260526_Numeric_Equation/equation_numeric_ea6d926a/track/tree_cot.txt) | unseen_operator | 87 | 37 → **87** |
| [7d8fe3a8](../../01_data/260526_Numeric_Equation/equation_numeric_7d8fe3a8/track/tree_cot.txt) | unseen_operator | 7644 | 7 → **7644** |
| [157228d7](../../01_data/260526_Numeric_Equation/equation_numeric_157228d7/track/tree_cot.txt) | ambiguous | 35 | `*35` → **35** |
| [5f6798e1](../../01_data/260526_Numeric_Equation/equation_numeric_5f6798e1/track/tree_cot.txt) | left-to-right | 34 | -34 → **34** |
| [d4b08884](../../01_data/260526_Numeric_Equation/equation_numeric_d4b08884/track/tree_cot.txt) | exotic | 2 | 42 → 67 / **2** (cryptn only) |

**Losses (moe beats our variants) — 9 puzzles**, in 2 buckets:

**Bucket A — sign-garble (alleq only, 3 puzzles)**. The model produces our explicit "leading minus" narration on negative cases like [9a9f6025](../../01_data/260526_Numeric_Equation/equation_numeric_9a9f6025/track/tree_cot.txt) but then *overrides it* with a hallucinated rule "operator-symbol prefix: -56 -> 5-6". The line "operator-symbol prefix" doesn't exist in our training data. It's the model **emergent-misaligning** to over-generalize the operator-symbol re-attach pattern (for `^22` / `?14`) onto a literal `-` sign:

```
OUR training (alleq):  "The result is negative; the sign is written as a leading minus, so the answer is -56."   →  \boxed{-56}
alleq model output:    "The result is negative, so it is written with the operator-symbol prefix: -56 -> 5-6."   →  \boxed{5-6}
```

**alleqgt fixes 2 of 3** of these by training on the gtTrue-filtered set (113 wrong CoTs zeroed). Less noise → model reproduces our narration faithfully. This is the **strongest empirical evidence in this report that noise in the training CoTs actively reinforces wrong-pattern emergence** — see §10 mechanism #2.

**Bucket B — operator/operand drift (all variants, 6 puzzles)**. The model invents a different operator family or skips the reading-order reversal. E.g. [75645166](../../01_data/260526_Numeric_Equation/equation_numeric_75645166/track/tree_cot.txt):

```
OUR training CoT:  "g(53, 49) = 53×49 = 2597; reading order is leftward, so the result 2597 -> 7952"  →  \boxed{7952}
all 3 model outputs: "g(35, 94) = 35×94-1 = 3289"  →  \boxed{3289}
```

The model used the rightward operands (not the leftward-reversed ones from our training CoT), applied `mul_minus1` instead of `mul`, and skipped the reversal. All three of our variants lose this — the format is short and structurally correct but the search procedure is incomplete.

---

## 6. Cryptarithm: the 3 disagreement puzzles ([HTML diff](token_diff_crypt.html), [forensic](cryptarithm_3puzzle_forensic.md))

| id | GT | moe | alleq | alleqgt | cryptn |
|---|---|---|---|---|---|
| 24e1f1d5 | `^#(!` | `^#(!` ✓ | `^(!#` ✗ | `^#(!` ✓ | `3` ✗ |
| 4d8df95b | `)` | `?` ✗ | `)` ✓ | `-` ✗ | `3` ✗ |
| b1b10e83 | `|"#$` | `|"#$` ✓ | `` `} `` ✗ | `}}` ✗ | `` {` `` ✗ |

Forensic findings (sub-agent):

1. **moe/alleq/alleqgt train on byte-identical cryptarithm CoTs for these 3 puzzles** (1324, 1264, 1497 chars). All 3-way differences are sampling / decoding noise — at the digit-assignment line, both moe and alleqgt assign digits at p ≈ 0.13–0.15 (uniform-ish). The model is **not deducing the digit substitutions, it's guessing**, and whose guess happens to be consistent decides win/loss.

2. **alleq/alleqgt drift into a "Reading order: little-endian" sub-template** that moe never uses (alleq 38/55, alleqgt 26/55, moe 0/55). This sub-template numerically reinterprets `AB = 15` and converts pure-concatenation puzzles into addition problems, locking the model into wrong addition on `b1b10e83`.

3. **For `4d8df95b`**: the OLD training CoT for this puzzle is itself unprincipled — it announces `\boxed{)}` without deriving it. alleq lands on it by accident because its little-endian sampled permutation maps digit 0 → `)`. The "win" is illusory.

4. **cryptn produces SHORT new-format CoTs on cryptarithm but hits the 7680-token CAP without `\boxed{}` on 24/55 deduce val puzzles**. Mean output_token_len: 5928 (cryptn) vs 864 (moe). Even when cryptn finishes, it commits structural bugs (e.g. on b1b10e83 it lists `) -> h` as a third operator, uses 8 letters instead of 9, applies wrong operator at the QUERY step).

---

## 7. Cryptarithm mass collapse: anatomy of cryptn 0/55 ([full analysis](cryptarithm_mass_collapse.md))

The cryptn run replaced cryptarithm CoTs with our new honest-DFS solver from `01_data/260525_Cryptarithm/`. The build script applied two gates:

- **CAP gate**: oversampling=0 if `token_length >= 7680`
- **GT gate**: oversampling=0 if `GT-match != True`

Result:

| | total | over-CAP only | GT-mismatch only | both | survived |
|---|--:|--:|--:|--:|--:|
| cryptarithm_deduce | 639 | 173 | 70 | 153 | **243** (×2 → Σos 486) |
| cryptarithm_guess | 161 | 19 | 82 | 38 | **22** (×2 → Σos 44) |
| **total** | **800** | 192 | 152 | 191 | **265** |

So **535 of 800 cryptarithm rows were excluded from training** (66.9%).

Root causes:

1. **The honest-DFS CoTs are far too long.** Token-length distribution of the 800 new CoTs: median 3.8k, p90 around 25k. **47.9% exceed CAP=7680.** One puzzle is 2.8M tokens, dozens are 100k+. Old (golden) CoTs averaged ~824 tokens; new CoTs ~6693 (+712%). The new format is structurally incompatible with the training budget.
2. **GT-match rate on the new CoTs is only 57%** (457/800). The honest solver's `guess_*` buckets sit at 5–34% GT-True (per `01_data/260525_Cryptarithm/SUMMARY.md`). The gate kills the rest.
3. **The CAP at training = the CAP at inference.** At inference the model often runs to 7680 tokens with no `\boxed{}` because it was trained on CoTs that were already at the boundary. 24 of 55 val cryptarithm_deduce outputs hit the cap with no boxed answer.

The cryptarithm mass for moe was 800 effective training examples; for cryptn it was 530 — and those 530 are in a format the model can't fit inside the budget at inference.

---

## 8. Why alleqgt loses 1 cryptarithm puzzle vs moe (identical training)

This is the user's exact framing of the surprise. `b1b10e83`: moe ✓ (`|"#$`), alleqgt ✗ (`}}`), cryptarithm training byte-identical.

The training data is identical; the **model weights are not**. alleqgt was full-precision-fine-tuned on `out_proj` weights with a *different* numeric-eq training mix (gtTrue-filtered, 113 wrong CoTs zeroed) than moe. The numeric-eq gradient pushed `out_proj` to a different point in weight space. That parameter shift changes:

1. The output distribution at every attention block (not just for numeric-eq inputs)
2. Therefore the residual-stream representation read by every layer's MoE router (Nemotron has 128 experts, top-6)
3. Therefore the experts activated on cryptarithm prompts
4. Therefore the predicted token at the digit-assignment step, where the probability was already near-uniform

`b1b10e83`'s training CoT for moe-and-alleqgt is identical, but the trained model trajectories diverge by token 741 in the alleqgt run — the model emits an extra `' addition'` token with p ≈ 1.0 because it locked into the "little-endian" sub-template that the alleqgt's parameter shift made more accessible. The cryptarithm training never told the model that — the new template emerged from learned interference.

This is the textbook signature of **intruder dimensions** in `out_proj` from a full-precision update (Shuttleworth et al., [arxiv 2410.21228](https://arxiv.org/pdf/2410.21228); VeFA, [arxiv 2510.19155](https://arxiv.org/html/2510.19155)) — see §10 #1.

---

## 9. Why "improving the data" looks worse on the LB

| run | val OVERALL | LB | delta vs moe (val) | delta vs moe (LB) |
|---|--:|--:|--:|--:|
| moe | 0.8621 | 0.86 | — | — |
| alleq | 0.8611 | 0.84 | −0.001 | **−0.02** |
| alleqgt | 0.8632 | 0.84 | +0.001 | **−0.02** |
| cryptn | 0.8653 | 0.85 | +0.003 | **−0.01** |

Four contributing factors:

1. **Val is mostly memorization** (71% leak). moe with old CoTs memorizes the in-training val perfectly. Our variants memorize 91% but generalize ~3× better. On val that wash to ±0.3pp; on a less-leaky test set the trade is different.
2. **Cross-domain interference** flipped one cryptarithm puzzle (alleqgt 1/55 vs moe 2/55) on byte-identical cryptarithm data — small but real.
3. **cryptn collapsed cryptarithm to 0/55** because 67% of cryptarithm training mass was thrown away by the CAP+GT gates. We didn't "improve" cryptarithm data — we removed it.
4. **Format leakage** (§10 #2): the 3.5× shorter numeric-eq CoTs and the new structural markers (§-tree, `lock_concat`, "leading minus" narration) shifted the model's general reasoning-format direction. Cryptarithm at inference inherited the shorter, differently-structured style — even though its training rows did not change.

---

## 10. The mechanisms (research-grounded)

(Full literature review with citations is in the previous Agent response. Summary here.)

### #1. Intruder dimensions in `out_proj` from full-precision update
You're full-precision-updating a single shared projection with no rank cap. Shuttleworth's "LoRA vs Full Fine-tuning: An Illusion of Equivalence" ([2410.21228](https://arxiv.org/pdf/2410.21228)) identified **intruder dimensions** — new high-ranking singular vectors that fine-tuning introduces *outside* the column space of the pre-trained weights. The VeFA paper ([2510.19155](https://arxiv.org/html/2510.19155)) makes the causality explicit: **forgetting is localized to those intruder dimensions and can be ablated to recover prior behavior**. Your alleqgt almost certainly created such dimensions in `out_proj` that didn't exist in moe, and those dimensions are what flipped `b1b10e83`.

### #2. Narrow SFT → broad behavior shift (Betley emergent misalignment)
Betley et al. ([2502.17424](https://arxiv.org/abs/2502.17424), Nature 2026) trained models on 6,000 narrow examples and produced broadly-shifted behavior across topics with no overlap. OpenAI's mech-interp follow-up identified that the shift is mediated by *a single low-dimensional persona direction*. Your 3.5× CoT-length compression + new structural markers on numeric-eq pushed an analogous "reasoning-format" direction shared by cryptarithm. Direct mechanism for the `5-6` sign-garble overgeneralization: the model learned a *new format rule* ("when result has a sign, write the sign between digits") from the operator-symbol-prefix pattern (`^22`, `?14`) and applied it to plain `-56` even though our training never contained that.

### #3. Oversampling-mass collapse below literature thresholds
Particula's 2026 review names **"1,000 examples per task is an absolute minimum"** as the practitioner consensus. Your cryptn cryptarithm_guess Σos dropped 161 → 44 (−73%) — below every published threshold. Below this floor the gradient is dominated by other categories, the projection update direction is wrong for cryptarithm, and the model can't maintain its cryptarithm circuit.

### #4. SFT has no KL constraint (RL's Razor)
"RL's Razor" ([2509.04259](https://arxiv.org/abs/2509.04259)) shows SFT can land arbitrarily far from the base policy; RL is implicitly KL-bounded. Your full-precision out_proj SFT has no anchoring, no Fisher penalty, no rank cap, no replay buffer. Drift in shared representations is mathematically unconstrained.

### Top 3 likely mechanisms here, in priority order:
1. Intruder dimensions in `out_proj` from full-precision update (§10 #1) → flips byte-identical cryptarithm verdicts
2. Format/persona shift from narrow short-CoT training (§10 #2) → emergent `5-6` rule, cryptarithm little-endian drift
3. Oversampling-mass collapse below threshold (§10 #3) → cryptn cryptarithm 0/55

---

## 11. Improvements that would NOT cost score (concrete prescriptions, ranked)

| priority | change | rationale | citation |
|---|---|---|---|
| 1 | **Floor cryptarithm oversampling at ≥ moe baseline** (don't let it drop below 639 + 161) | Below 1000 effective examples per category, the category collapses | [Particula 2026](https://particula.tech/blog/how-much-data-fine-tune-llm) |
| 2 | **Add 5–10% replay** of baseline cryptarithm CoTs even when changing numeric-eq | Cheapest defense against intruder dimensions | [GeRe 2508.04676](https://arxiv.org/pdf/2508.04676) |
| 3 | **Switch from full-precision out_proj to LoRA** on out_proj (rank ≤ 32) | Adding rank cap limits intruder-dimension creation | [Shuttleworth 2410.21228](https://arxiv.org/pdf/2410.21228) |
| 4 | **Keep numeric-eq CoT length closer to baseline** OR mix long+short variants | Avoid format/persona shift | [LS-Mixture SFT 2505.03469](https://arxiv.org/abs/2505.03469) |
| 5 | **For new cryptarithm CoTs**: raise the CAP to 16k OR rewrite the solver to fit under 4k | New 800-puzzle set: 47.9% exceed current CAP | this report §7 |
| 6 | **Iterate on the `5-6` sign-garble**: the gtTrue filter already mostly fixed it; combine alleqgt's data discipline with the cryptn additions | Wrong-CoT noise reinforces emergent rule | this report §5 |
| 7 | **Stop reporting val accuracy without breaking it down by in-/out-of-training** | Val is a 71% data leak; the headline number is a memorization measurement | this report §3-4 |

---

## 12. Specific puzzle-level recommendations (per [[feedback_user_directs_fix]] — surface, don't fix)

### Numeric-eq solver
- **Unseen-operator default**: should be `|a−b|` (absolute difference), not concat. Reference solver uses this; our solver doesn't. Fixes 4 puzzles per §5.
- **Sign re-attach**: must use the literal operator symbol on negatives (not just `-`); preserve leading zeros.
- **Reading-order tiebreak**: prefer the reading consistent across *all* examples, not the one matching EX1.
- (Audit found 13 wrong puzzles; the reference solves all 13. Our solver is +56 ahead net 619 vs 563.)

### Cryptarithm new-solver
- Either raise CAP or rewrite to fit under 4k tokens — the current 47.9% over-CAP rate means most training is silently zeroed.
- The "Reading order: little-endian" sub-template appears 26–38 times in alleq/alleqgt val outputs — it's emerging spontaneously and breaking pure-concat puzzles. Worth probing.
- `4d8df95b` training CoT ends with an undeduced `\boxed{)}` — that one row teaches the model to guess.

---

## 13. Where the "data improved → score got worse" framing breaks down

The data didn't get worse. Three different things happened:

1. **The numeric-eq data got better at teaching a method** (out-of-training generalization 22% → 86%) but slightly worse at memorizing in-training (100% → 91%). Val is mostly memorization, so val score is flat.
2. **One cryptarithm puzzle flipped from interference**, despite identical training data — a real but ~1-puzzle effect on a tiny denominator (55).
3. **cryptn aggressively gated cryptarithm training, throwing away 67% of it**. That's not "improving data" — that's reducing training mass below the threshold a category needs to survive.

The fix isn't "stop improving data." It's:
- Improve numeric-eq CoT method *without* shrinking format (preserve the 5-6k token length signature) — or accept the in-training memorization loss and measure on the LB.
- Improve cryptarithm CoTs *without* dropping training mass — fit them under the CAP and ensure GT-True before swapping.
- Add a replay buffer and switch to LoRA on out_proj — both are cheap.

---

## Files for downstream review

- HTML diffs: [token_diff_eq.html](token_diff_eq.html), [token_diff_crypt.html](token_diff_crypt.html)
- Cryptarithm 3-puzzle forensic: [cryptarithm_3puzzle_forensic.md](cryptarithm_3puzzle_forensic.md)
- Cryptarithm mass collapse: [cryptarithm_mass_collapse.md](cryptarithm_mass_collapse.md)
- Build script: [_build_diff_html.py](_build_diff_html.py)
