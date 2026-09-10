# Score Tracker

> Kaggle: NVIDIA Nemotron Model Reasoning Challenge
> Baseline target: huikang 0.85 winning run

## Update rules

- Only add an entry when user gives a real Kaggle score (don't pre-fill from training loss).
- Append to the end — never reorder.
- One entry per submission. Include date, score, study folder, data, method, **LoRA architecture**, and why.
- Link to a per-run `EVALUATION.md` or `TRAINING_INFO.md` when there's a substantial analysis.

---

## Submission #1 — 2026-05-13 — Score 0.84

- **Study folder:** (pretok run)
- **LoRA architecture:** r32/α32 on attn + Mamba + experts + shared_experts + lm_head; routed experts tied rank-direction (rank-1/expert)
- **Data:** pretok 7830 (0408 winning run)
- **Method:** bf16, CCE, MoE tie, lm_head LoRA, manual loop, micro=4
- **Why:** Baseline repro of huikang 0.85. Same pretokenized data, same order.

---

## Submission #2 — 2026-05-13 — Score 0.81

- **Study folder:** `2605131530_sft_huikang_textcsv_7830_rtx6000`
- **LoRA architecture:** r32/α32 on attn + Mamba + experts + shared_experts + lm_head; routed experts tied rank-direction (rank-1/expert)
- **Data:** textcsv 7830 (0413 reasoning files)
- **Method:** bf16, CCE, MoE tie, lm_head LoRA, manual loop, stratified batch
- **Why:** Huikang's latest data but performs worse. `reasoning/*.txt` was updated after the winning run (added "I will put my final answer", 0-indexed). −0.03 from data mismatch.

---

## Submission #3 — 2026-05-14 — Score 0.69

- **Study folder:** `2605132110_sft_qlora4bit_huikang_pretok_rtx6000`
- **LoRA architecture:** 4-bit QLoRA, r32/α32 on attn + Mamba + experts + shared_experts + lm_head; routed experts tied rank-direction (rank-1/expert)
- **Data:** pretok 7830 (0408 winning run)
- **Method:** 4bit, CCE with `.to(bf16)` cast, MoE tie, lm_head LoRA, manual loop
- **Why:** Broken. Unnecessary `.to(bf16)` cast on hidden_states before CCE. Accumulated precision loss over 245 steps corrupted weights.

---

## Submission #4 — 2026-05-14 — Score 0.84

- **Study folder:** `2605132323_sft_huikang_textcsv_0408_rtx6000`
- **LoRA architecture:** r32/α32 on attn + Mamba + experts + shared_experts + lm_head; routed experts tied rank-direction (rank-1/expert)
- **Data:** textcsv 7830 (0408 decoded from pretok)
- **Method:** bf16, CCE, MoE tie, lm_head LoRA, manual loop, huikang order, raw tokenizer
- **Why:** Matched baseline. Decoded pretok→text, verified token-identical. Proves the text CSV pipeline works when the data is correct.

---

## Submission #5 — 2026-05-14 — Score 0.68

- **Study folder:** `2605141458_sft_huikang_golden_rtx6000`
- **LoRA architecture:** r32/α32 on attn + Mamba + experts + shared_experts + lm_head; routed experts tied rank-direction (rank-1/expert)
- **Data:** `260514_huikang_golden.csv` (7,849 expanded). 6,171 huikang in_7830 + 735 lkevincc cryptarithm samples.
- **Method:** bf16, CCE, MoE tie, lm_head LoRA, manual loop, stratified batch, raw tokenizer
- **Why:** Adding 735 lkevincc cryptarithm samples crashed the score. Per-category eval: bit_manipulation 74.4% → 5.0% (−69pp), text_encryption −8.9pp, equation_transformation −5.8pp, gravity −3.1pp. Order and surface template are NOT the cause of bit_manipulation (verified). Observed symptom: operator-matching commit threshold collapsed (5.2 → 0.9 successful matches/output). 35.5% of equation_transformation outputs contained lkevincc template markers (`Digit mapping`, `✓`, `Reading order:`).
- **Detail:** [EVALUATION.md](2605141458_sft_huikang_golden_rtx6000/260514_2003_submission/EVALUATION.md), [STUDY_COT_AND_ORDER.md](2605141458_sft_huikang_golden_rtx6000/260514_2003_submission/STUDY_COT_AND_ORDER.md)

---

## Submission #6 — 2026-05-15 — Score 0.84

- **Study folder:** `2605151409_sft_huikang_golden_stripped_kaggleregex_rtx6000`
- **LoRA architecture:** r32/α32 on attn + Mamba + experts + shared_experts + lm_head; routed experts tied rank-direction (rank-1/expert)
- **Data:** `260514_huikang_golden_stripped.csv` (7,849 expanded). Same composition as #5 — 6,171 huikang + 735 lkevincc — but with `✓` stripped from 704 lkevincc rows and id=45076dc9 typo'd CoT fixed.
- **Method:** Same as #5 (bf16, CCE, MoE tie, lm_head LoRA, manual loop, stratified batch, raw tokenizer) **plus Kaggle-metric `\boxed` extractor** in the training script.
- **Why — full recovery from 0.68 → 0.84 with three small fixes:**
  1. **Kaggle-metric `rfind('}')` extractor** — 88 lkevincc rows previously training on truncated answers (e.g. answer `+}` was extracted as `+` by the old `\boxed{([^}]*)}` regex).
  2. **`✓` stripped** — 2,537 tick characters removed from 704 lkevincc rows.
  3. **id=45076dc9 typo'd CoT fixed** — `\boxed{6}}` → `\boxed{6}`.

  The 0.68 → 0.84 jump is the **same data + same architecture + same hyperparameters** — just the three small data/extraction defects fixed. The "bit_manipulation collapse" symptom in #5 was downstream of these defects, not a separate MoE phenomenon. My #5 analysis blaming MoE re-routing / enumerate-then-verify habit was wrong.

- **Detail:** [TRAINING_INFO.md](2605151409_sft_huikang_golden_stripped_kaggleregex_rtx6000/260515_1846_submission/TRAINING_INFO.md)

---

## Submission #7 — 2026-05-16 — Score 0.84

- **Study folder:** `2605152212_sft_huikang_lkall_stripped_kaggleregex_rtx6000`
- **LoRA architecture:** r32/α32 on attn + Mamba + experts + shared_experts + lm_head; routed experts tied rank-direction (rank-1/expert)
- **Data:** `260515_huikang_lkall_stripped.csv` (7,849 expanded). 6,106 huikang non-cryptarithm + **800 lkevincc cryptarithm** (the 65 huikang-covered IDs swapped to lkevincc's CoT + 735 lkevincc-only). One CoT style for cryptarithm instead of #6's mix of huikang's 65 + lkevincc's 735.
- **Method:** Byte-identical to #6 except cryptarithm CoT source. Same recipe (bf16, CCE, MoE tie, lm_head LoRA, manual loop, stratified batch, raw tokenizer, Kaggle-metric extractor, `✓` stripped, 45076dc9 fixed).
- **Why:** Tested whether using one consistent CoT style for all 800 cryptarithm puzzles (lkevincc) is as good as the mixed huikang+lkevincc composition. Result: **same score, 0.84**. The 65 huikang cryptarithm CoTs (positive operator identification in 54/65) are not contributing distinct signal over lkevincc's "default to concatenation" fallback at this training scale — the score is bottlenecked elsewhere, not by which solver authored those 65 CoTs. One-template training is now the cleaner default for future cryptarithm work.
- **Detail:** [TRAINING_INFO.md](2605152212_sft_huikang_lkall_stripped_kaggleregex_rtx6000/260516_0229_submission/TRAINING_INFO.md)

---

## Submission #8 — 2026-05-16 — Score 0.82

- **Study folder:** `2605161900_sft_huikang_ddp_nosync_off_rtx6000`
- **LoRA architecture:** r32/α32 on attn + Mamba + experts + shared_experts + lm_head; routed experts tied rank-direction (rank-1/expert)
- **Data:** `260514_huikang_update/huikang_7830.csv` (same as 0.78 DDP run and same content as Sub#4 textcsv).
- **Method:** DDP 2-GPU, bf16 + CCE + MoE tying + lm_head LoRA + per-MB normalization + contiguous rank split. **Four changes vs the 0.78 run**: (1) `use_gradient_checkpointing='unsloth'` → HF non-reentrant via `gradient_checkpointing_enable(use_reentrant=False)`, (2) drop `static_graph=True`, (3) add `find_unused_parameters=True`, (4) drop `no_sync()` (every MB triggers all-reduce). Final loss 0.0021 (Sub#4 was 0.002092, diff 0.4%).
- **Why — DDP fix recovers +0.04 from 0.78 but still -0.02 short of single-GPU 0.84:**
  - Previous DDP run scored 0.78 because Unsloth's reentrant gradient checkpointing produces wrong gradients under DDP even with `static_graph=True` (which is supposed to suppress reducer double-fire). Step-1 grad_norm was 1.4142 vs single-GPU's 1.2356 — a 14.5% structural error too large for FP noise. Switching to HF non-reentrant fixed step-1 grad_norm to **1.2354** (matches Sub#4 to 0.016%, FP-noise level) and made checkpoint losses at steps 50/100/200/244 all track Sub#4 within 5%.
  - **Adapter-diff analysis** (this run vs Sub#4 vs old broken DDP, 11,986 LoRA tensors): the fix cut LoRA L2 distance from Sub#4 by **2.1×** (20.9% → 9.9%), mean cosine from 0.896 → **0.973**. The drift direction between this run and the old broken DDP is mostly orthogonal (cos=0.25) — meaning the two DDP runs drift via *different* DDP mechanisms, not the same one. **The full 0.02 is DDP-attributable**: Subs #4/#6/#7 (three different single-GPU runs across different data) all score exactly 0.84, so single-GPU variance ≈ 0 at the Kaggle-score level → any score gap from a single-GPU baseline IS the DDP contribution.
  - The remaining 0.02 likely lives in a second DDP/MoE interaction (32% of residual L2 sits in MoE up_proj LoRA), distinct from the reentrant-ckpt bug that the S3-lite fix eliminated. Plausible candidates: gradient-bucket reduction over sparsely-routed MoE experts under non-reentrant ckpt, or `gradient_as_bucket_view=True` + in-place `_tie_grads` writes corrupting subsequent iterations. Could likely be debugged further if speed-vs-score tradeoff matters.
  - **Tradeoff vs single-GPU:** 1.49× speedup (159 min vs 237 min) at -0.02 score cost. Much better than the previous DDP attempt (1.88× speedup at -0.06 cost).
- **Detail:** [TRAINING_INFO.md](2605161900_sft_huikang_ddp_nosync_off_rtx6000/20260516_submission/TRAINING_INFO.md)

---

## Submission #9 — 2026-05-17 — Score 0.46

- **Study folder:** `2605161142_sft_9500_catprefix_rtx6000`
- **LoRA architecture:** r32/α32 on attn + Mamba + experts + shared_experts + lm_head; routed experts tied rank-direction (rank-1/expert)
- **Data:** `260516_9500_answer_only_catprefix.csv` — 9,500 official rows (full official set), each row's `solver_cot` REPLACED with a category-identifying prefix + the answer-only boilerplate (`I will now return the answer in \boxed{}\nThe answer in \boxed is\n\boxed{...}`). Median CoT 78 chars (vs ~6,700 tokens in run #7). 7 prefixes — one per huikang category, with cryptarithm_deduce/guess sharing and equation_numeric_deduce/guess sharing.
- **Method:** Byte-identical to run #7 except (a) different data file and (b) different ordering (stratified shuffle since no `index.jsonl` for the 9,500-row set; oversampling = 1× each → 297 steps). Hyperparameters, loss, optimizer, LR schedule, MoE tying, lm_head LoRA all identical. Final loss 0.0623 at step 297.
- **Kaggle ref:** 52728440
- **Why — the hypothesis is decisively falsified, −0.38 vs 0.84 baseline:**
  - Test: "can the model learn the task from (prompt + category prefix + boxed answer) alone, with no real chain-of-thought?" Answer: **no.** Stripping the multi-KB solver CoT and replacing with a one-sentence prefix + boilerplate costs **38 percentage points** on Kaggle LB.
  - Mechanism: without real CoT, the model has no in-distribution example of *deriving* the answer from the prompt's examples. It learns to predict `\boxed{...}` tokens from the prompt pattern alone — which works on training data (memorization) but collapses on novel test prompts. Training loss converged fine (0.0623) because predicting the boxed-answer tokens from the same prompt+prefix pattern is easy; what we lost was the procedural knowledge encoded in the per-bit / per-rule / per-example CoT traces.
  - The added 2,594 official rows (huikang's "unused" set) cannot compensate — those rows were excluded because huikang's solver failed to produce CoTs for them; we provided NO CoT, just a prefix, so they teach the model nothing about how to solve.
  - This is a hard NO on "prefix is a useful inference-time scaffolding tool when CoT is absent." The CoT is doing real work. Future experiments must keep meaningful reasoning traces.
- **Detail:** [TRAINING_INFO.md](2605161142_sft_9500_catprefix_rtx6000/260517_0537_submission/TRAINING_INFO.md)

---

## Submission #10 — 2026-05-18 — Score 0.78 (best soup; endpoint 0.76)

- **Study folder:** `2605181043_sft_huikang_lkall_stripped_soup_rtx6000`
- **LoRA architecture:** r32/α32 on attn + Mamba + experts + shared_experts + lm_head; routed experts tied rank-direction (rank-1/expert)
- **Data:** `260515_huikang_lkall_stripped.csv` (7,849 expanded) — identical to Sub #7.
- **Method:** Retrain of Sub #7 with soup-saving adapters at {50, 100, 150, 197, 200, 209, 221, 234, 246}. One training run → 5 submissions (endpoint + last5/wise × naive/SVD).

| Recipe | Method | Kaggle ref | Score | Δ vs endpoint |
|---|---|---|---|---|
| final-246 (endpoint) | — | 52794510 | 0.76 | — |
| soup-last5 {200,209,221,234,246} | naive | 52788565 | 0.78 | +0.02 |
| soup-last5 {200,209,221,234,246} | SVD | 52792293 | 0.78 | +0.02 |
| wise {50,150,246} | naive | 52788645 | 0.75 | −0.01 |
| wise {50,150,246} | SVD | 52792619 | 0.77 | +0.01 |

- **Why:** Endpoint scored 0.76 vs Sub #7's 0.84 on byte-identical data + script (step-1 forward bit-identical, step-1 grad_norm differs 1e-4 from CUDA `atomicAdd` non-determinism, compounding over 246 steps to a worse minimum — a low-tail draw, see Sub #11 which lands 0.83 on the same setup). Against this run's true 0.76 baseline, last-K soup lifts +0.02 by denoising the converged tail; wise (mixing early ckpts) is flat/negative.
- **Detail:** [TRAINING_INFO.md](2605181043_sft_huikang_lkall_stripped_soup_rtx6000/260518_1507_submission/TRAINING_INFO.md)

---

## Submission #11 — 2026-05-19 — Score 0.84 (endpoint 0.83/0.84 — see inference-noise note)

- **Study folder:** `2605182305_sft_huikang_golden_stripped_soup_rtx6000`
- **LoRA architecture:** r32/α32 on attn + Mamba + experts + shared_experts + lm_head; routed experts tied rank-direction (rank-1/expert)
- **Data:** `260514_huikang_golden_stripped.csv` (7,849 expanded) — byte-identical to Sub #6 (md5 `92a515f31cd2e89f9cb355c4ed09f927`; matched Sub #6's startup token counts 27,989,496 / 26,713,988).
- **Method:** Reproduction of Sub #6 (same script + config + data) with soup-saving adapters. One training run → 5 submissions + 1 endpoint resubmit.

| Recipe | Method | Kaggle ref | Score |
|---|---|---|---|
| final-246 (endpoint) | — | 52813620 | 0.83 |
| final-246 (endpoint, **identical file resubmit**) | — | 52865914 | **0.84** |
| soup-last5 {200,209,221,234,246} | naive | 52835478 | 0.83 |
| soup-last5 {200,209,221,234,246} | SVD | 52833850 | 0.84 |
| wise {50,150,246} | naive | 52835601 | 0.77 |
| wise {50,150,246} | SVD | 52820370 | 0.78 |

- **Why:** Reproduces Sub #6 — final loss 0.0029 matched exactly, per-step loss tracked Sub #6 within ±5e-4.
- **⚠️ Inference is non-deterministic (±0.01).** The byte-identical endpoint file scored **0.83 then 0.84** on two submissions (refs 52813620 / 52865914). Kaggle re-runs greedy decoding each submission; bf16 MoE atomicAdd non-determinism shifts a few generations → ±0.01 score wobble. **Consequence:** the "+0.01 from SVD-soup-last5" is within this noise floor — soup-last5 (0.83/0.84) is NOT distinguishable from the endpoint (0.83/0.84). Earlier "soup buys +0.01" claims are not supported once the metric's own ±0.01 noise is known.
- **What still holds:** wise {50,150,246} = 0.77/0.78 is a real loss (−6/7pp ≫ noise) — mixing early checkpoints hurts.
- **Detail:** [TRAINING_INFO.md](2605182305_sft_huikang_golden_stripped_soup_rtx6000/260519_0332_submission/TRAINING_INFO.md)

---

## Submission #12 — 2026-05-20 — Score 0.84

- **Study folder:** `260520_cross_soup_run6_run11` (no training — derived from Sub #6 + Sub #11 adapters)
- **LoRA architecture:** SVD-soup of #6 + #11 adapters; routed experts tied rank-direction (rank-1/expert), inherited from the source runs
- **Method:** Cross-trajectory SVD soup. SVD-merge (N=2, equal weight) of two independent 0.84 trajectories: Sub #6's final adapter ⊕ Sub #11's svd-soup-last5. Same QR+SVD pipeline. Tests whether averaging *across* trajectories (vs within one) beats 0.84.
- **Kaggle ref:** 52867158
- **Why — no gain (0.84, same as both components):** Averaging two independent 0.84 trajectories did not exceed 0.84. The two runs share identical data (only differ by training non-determinism), so their errors are likely correlated — cross-trajectory averaging only helps when trajectories carry *independent* error, which needs diverse data / different seeds, not the same data twice. Any sub-noise gain is also masked by the ±0.01 inference noise (Sub #11 note). Multi-trajectory soup on identical-data runs is not a path past 0.84.
- **Detail:** local folder `260520_cross_soup_run6_run11/` (`submission_svd-cross_run6final_run11last5.zip`).

---

## Submission #13 — 2026-05-22 — Score 0.85 → 0.86 (best-of-2; the public 0.86-wall adapter, submitted verbatim)

- **Study folder:** none — no training. Downloaded public artifact, repackaged in `study/_tmp_086adapter/submission.zip`.
- **LoRA architecture:** r32/α32 on attn + Mamba + experts + shared_experts + lm_head; routed experts tied **expert-direction** (rank-32/expert) — huikang / Tinker
- **Data / Method:** N/A. This is the **public adapter behind the 0.86 leaderboard wall** — `kienngx/nemotron-nano-30b-trained/Triton/tinker-adapter/1` (adapter_config.json 581 B + adapter_model.safetensors 3.55 GB), the exact directory the most-voted notebooks (`safar1/lb-score-0-86`, `mohamedamr992/0-86-adapter-packaging-workflow`) zip and submit. Per research lineage it is huikang's published Tinker-trained `nemotron-adapter`. Submitted **byte-for-byte unmodified** (config: r32/α32/dropout0, targets q/k/v/o/up/down/in/out_proj + lm_head, base `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`). Both draws are the identical file (same md5 `geM2zuXyQCETAdcmaQ7oc…`).

| Draw | Kaggle ref | Score |
|---|---|---|
| 1 | 52912925 | 0.85 |
| 2 (identical-file resubmit) | 52915203 | **0.86** |

- **Why — settles "why are we at 0.84 when everyone's at 0.86":** the wall is hundreds of teams re-submitting THIS public adapter (huikang's actual weights), not out-training us. Submitting it verbatim scored **0.85 then 0.86 on the byte-identical file** — confirming the gap to the wall is **pure ±0.01 inference noise**, and that our trained reproductions (0.84; Subs #6/#7/#11) sit ~1 noise notch below his real Tinker weights (0.85–0.86). Kaggle's public LB shows each team's *best* submission, so the 0.86 wall is best-of-N noisy draws of this same artifact; draw 2 got the lucky draw → **our public LB is now 0.86.**
- **Takeaway:** the 0.84→0.86 gap is not an out-trainable data/recipe gap — the wall is huikang's real published weights. To MATCH the wall on the LB, submit his adapter (best-of-N → 0.86, done). To BEAT it requires improving on his actual weights, not re-training a copy (bit_manipulation lever, per `STRATEGY_0.84_to_0.86.md`). Since the entire 0.86 wall is the same artifact, the **private LB will be crowded at 0.86** — final placement is decided by who exceeds it.

---

## Submission #14 — 2026-05-22 — Score 0.71 (expert-direction MoE tie)

- **Study folder:** `2605221141_sft_moe_fixtie_rtx6000`
- **LoRA architecture:** r32/α32 on attn + Mamba + experts + shared_experts + lm_head; routed experts tied **expert-direction** — one shared rank-32 factor across all 128 experts (up_proj.lora_A + down_proj.lora_B). Verified byte-level: experts TIED 128/128, rank-32. out_proj.lora_B = 0 (Mamba fast-path on).
- **Data:** `260514_huikang_golden_stripped.csv` (7,849 expanded, md5 `92a515f31cd2e89f9cb355c4ed09f927`)
- **Method:** Unsloth + manual loop + CCE + manual lm_head LoRA. 246 steps, lr 2e-4→0, seq 8192, bs 32. Final loss 0.0036. One run → endpoint + soup-last5.

| Recipe | Method | Kaggle ref | Score |
|---|---|---|---|
| final-246 (endpoint) | — | 52936690 | 0.71 |
| soup-last5 {200,209,221,234,246} | SVD | 52936816 | 0.71 |

- **Why:** First run with the corrected expert-direction tie (vs the rank-direction rank-1 tie of Subs #6/#7/#11). Endpoint and soup both scored 0.71 — Δ−0.13 vs the 0.84 baseline (≫ ±0.01 noise). A/B with Sub #15 (same run, tie off) isolates the tie as the cause.
- **Detail:** [TRAINING_INFO.md](2605221141_sft_moe_fixtie_rtx6000/260522_fixtie_submission/TRAINING_INFO.md)

---

## Submission #15 — 2026-05-22 — Score 0.84 (no MoE tie)

- **Study folder:** `2605221210_sft_moe_notie_rtx6000`
- **LoRA architecture:** r32/α32 on attn + Mamba + experts + shared_experts + lm_head; routed experts **UNTIED** — independent rank-32 LoRA per expert. Verified byte-level: experts UNTIED, rank-32. out_proj.lora_B = 0 (fast-path on).
- **Data:** same as Sub #14 (md5 `92a515f31cd2e89f9cb355c4ed09f927`)
- **Method:** same script/config as Sub #14 with `MOE_TIE=0`. 246 steps, final loss 0.0029. One run → endpoint + soup-last5 (soup pending).

| Recipe | Method | Kaggle ref | Score |
|---|---|---|---|
| final-246 (endpoint) | — | 52936939 | 0.84 |
| soup-last5 {200,209,221,234,246} | SVD | 52938502 | 0.84 |

- **Why:** No-tie rank-32 = 0.84 (endpoint and soup both 0.84), matching the baseline (Subs #6/#7/#11). A/B vs Sub #14 (same run, tie on): no-tie 0.84 vs expert-tie 0.71 (Δ−0.13).
- **Detail:** [TRAINING_INFO.md](2605221210_sft_moe_notie_rtx6000/260522_notie_submission/TRAINING_INFO.md)

---

## Submission #16 — 2026-05-23 — Score 0.86 (no tie + LIVE out_proj) ← first *trained* model to clear 0.84

- **Study folder:** `2605222039_sft_moe_outproj_rtx6000`
- **LoRA architecture:** r32/α32 on attn + Mamba + experts + shared_experts + lm_head; routed experts **UNTIED** rank-32; **Mamba `out_proj` LoRA LIVE** (‖B‖≈0.70, matching huikang's Tinker 0.86). Byte-identical to Sub #15 in every tensor **except `out_proj` is now trained** (it was exactly 0 in #15 and every prior run). Backbone key naming (Kaggle-compatible). Verified byte-level + independently audited.
- **Data:** `260514_huikang_golden_stripped.csv` (md5 `92a515f31cd2e89f9cb355c4ed09f927`, 6906 unique → 7849 expanded) — identical to Subs #14/#15. **Provenance (from the CSV's `source` column):** 6106 huikang "golden" non-cryptarithm rows (bit_manipulation, cipher, unit_conversion, gravity, numeral, equation_numeric_deduce/guess) + 800 cryptarithm rows = **735 lkevincc** (`source=lkevincc_golden`) **+ 65 huikang** (`source=pretok_0408_decoded`). So the cryptarithm CoT is a 735/65 lkevincc/huikang mix — NOT pure lkevincc.
- **Recipe — how `out_proj` is made to train (the whole point):** keep `is_fast_path_available=True`, and set each Mamba mixer's `.training=False` after `FastLanguageModel.for_training`. The fused kernel `mamba_split_conv1d_scan_combined` (which folds `out_proj` via the raw weight → LoRA bypassed → dead) is gated on `if self.training and cache_params is None:`; with the mixer's `.training=False`, the mixer takes its own efficient **unfused else-branch** (`causal_conv1d_fn` + `mamba_chunk_scan_combined` + `self.out_proj(scan_output)` as a MODULE call) → `out_proj` LoRA receives gradients. No model-file patching; uses the model's own CUDA kernels. **NOT** `is_fast_path_available=False` (that routes to pure-torch `torch_forward` → OOM 126 GB at seq 8192). Same ~88.7 GB VRAM and ~1 min/step as the baseline. Independently audited VALID.
- **Train config:** Unsloth FastLanguageModel + manual loop + CCE + manual lm_head LoRA, transformers 4.56.2 (base env). r=32 / α=32 / dropout=0; targets q/k/v/o/up/down/in/out_proj + lm_head. AdamW lr 2e-4→0 linear, betas (0.9, 0.95), wd 0. Batch 32 (micro 4, ga 8), seq 8192, 246 steps (1 epoch), stratified seed 42, FIFO ckpt + soup saves [50,100,150,197,200,209,221,234,246]. Final loss 0.0030, 4.19 hrs, RTX PRO 6000. Env `MOE_TIE=0 USE_MEM_EFF=0`.

| Recipe | Method | Kaggle ref | Score |
|---|---|---|---|
| final-246 (endpoint) | live out_proj | 52959510 | **0.86** |
| soup-last5 {200,209,221,234,246} | SVD, live out_proj | 52959990 | 0.84 |

- **Why — `out_proj` is the missing piece (0.84 → 0.86):** clean A/B vs Sub #15 (identical run, `out_proj` dead) — the only tensor difference is a trained `out_proj`. The endpoint hit **0.86 on its first submission**, the **first time any of our *trained* adapters cleared 0.84** (Subs #1/#4/#6/#7/#11/#15 all topped at 0.84 across many endpoint/soup/resubmit draws). This matches huikang's Tinker 0.86 (also live `out_proj`, ‖B‖≈0.70) — so we now reach the wall with **our own trained model**, not by resubmitting his weights (Sub #13). +0.02 ≈ 2× the ±0.01 noise floor and mechanistically expected (`out_proj` is the lone architectural diff). **Honest caveat:** the soup draw landed at 0.84 (within noise of its own endpoint), so a single 0.86 endpoint isn't bulletproof — but our dead-`out_proj` adapters never produced a 0.86 across many tries, so the weight of evidence is that live `out_proj` lifts the ceiling to ~0.86. **Tying is NOT needed and HURTS** (Sub #14 expert-tie = 0.71): the recipe is **no-tie + live `out_proj`**.
- **Detail:** [TRAINING_INFO.md](2605222039_sft_moe_outproj_rtx6000/260523_outproj_submission/TRAINING_INFO.md); mechanism + resolution in `02_train/260522_moe_expert_rank1_bug_report.md`.

---

## Submission #17 — 2026-05-23 — Score 0.84 (DDP 2-GPU replication of #16)

- **Study folder:** `2605231435_sft_moe_outproj_ddp_rtx6000`
- **LoRA architecture:** identical to #16 — r32/α32, no-tie, **live out_proj** (‖B‖ per layer = #16's, ratio 0.994). Only parallelization changed.
- **Data:** `260514_huikang_golden_stripped.csv` (md5 `92a515f31cd2e89f9cb355c4ed09f927`, 6906 unique → 7849 expanded) — identical to #16.
- **Method:** 2× RTX PRO 6000 DDP. HF non-reentrant gradient checkpointing (the DDP-correct ckpt; Unsloth reentrant is the old 0.78 DDP bug). Per-rank batch 16 (contiguous split), per-MB-mean loss, DDP all-reduce-mean = single-GPU /8. Dropped the ragged final batch (245 steps; LR_DENOM=246 preserves #16's LR schedule). Resumed from ckpt-200 after an attempt-1 NCCL crash at the ragged step. Final loss 0.0025. Env `MOE_TIE=0 USE_MEM_EFF=0`.

| Recipe | Method | Kaggle ref | Score |
|---|---|---|---|
| final-246 (endpoint) | DDP 2-GPU, live out_proj | 52968815 | 0.84 |

- **Why — measured: no DDP penalty; 0.84 is run-to-run trajectory variance, not DDP-attributable.** Adapter-diff (effective ΔW=B@A) of DDP-vs-#16 = **cos 0.9575 / relL2 0.292**; the baseline of two *single-GPU* runs of the same recipe (#6 vs #11, both 0.84) = **cos 0.9555 / relL2 0.298** — i.e. DDP drifts from #16 by the *same* amount (marginally less) than two single-GPU runs drift from each other. Step-1 was bit-identical (loss 0.403614, gn 0.1480), loss tracked #16 within 0.0009 for all 245 steps, all module-group norms preserved (ratio 0.98–1.00), out_proj fully trained. So DDP added no extra drift; both runs land in a flat loss basin (loss 0.0025) at different points (cos ~0.955). **0.84 is the modal score** for this recipe (#4/#6/#7/#11/#15 all 0.84); #16's 0.86 was the high-tail draw (#10's 0.76 the low tail). This revises Sub #8's "the full 0.02 is DDP-attributable" (which never measured single-vs-single weight drift). NOT inference noise (that's ±0.01); a real weight difference that is not DDP-specific.
- **Detail:** [EVALUATION.md](2605231435_sft_moe_outproj_ddp_rtx6000/2605231810_submission/EVALUATION.md); [TRAINING_INFO.md](2605231435_sft_moe_outproj_ddp_rtx6000/2605231810_submission/TRAINING_INFO.md).

---

## Submission #18 — 2026-05-24 — Score 0.85 (newsolver-data endpoint)

- **Study folder:** `2605231427_sft_newdata_outproj_rtx6000`
- **LoRA architecture:** identical recipe to #16 — r32/α32, no-tie, **live out_proj** (‖B‖≈0.72), backbone naming (Kaggle-compatible). Only the training data differs from #16.
- **Data:** `260523_huikang_newsolver.csv` (6906 unique). Same as #16's data EXCEPT the 800 cryptarithm CoTs are swapped to our own **"newsolver"** solver. These are real reasoning traces with answer-matching NOT enforced — 346 do **not** reach the gt answer and are **kept intentionally** ("true reasoning > fake answer-matched CoT"). Non-cryptarithm huikang-golden rows untouched.
- **Train config:** Unsloth + manual loop + CCE + lm_head LoRA, transformers 4.56.2. AdamW lr 2e-4→0, batch 32 (micro4 ga8), seq 8192, 236 steps, stratified seed 42. Final loss 0.0021. Env `MOE_TIE=0 USE_MEM_EFF=0 FRESH_START=1`. RTX PRO 6000.

| Recipe | Method | Kaggle ref | Score |
|---|---|---|---|
| final-236 (endpoint) | live out_proj, newsolver data | 52970103 | 0.85 |

- **Why:** swapping the 800 cryptarithm CoTs to our newsolver solver (mismatches kept) scored 0.85 — **below** #16's 0.86 (treating scoring as deterministic per user; not "noise"). The new crypt data did not beat the original. Endpoint-only (no soup). Part of a 3-variant data A/B: see #19 (gold-only 0.84) and #20 (gt2x 0.85).
- **Detail:** [TRAINING_INFO.md](2605231427_sft_newdata_outproj_rtx6000/2605231846_submission/TRAINING_INFO.md).

---

## Submission #19 — 2026-05-24 — Score 0.84 (gold-only data endpoint)

- **Study folder:** `2605231948_sft_newdata_goldonly_outproj_rtx6000`
- **LoRA architecture:** identical recipe to #16 — r32/α32, no-tie, **live out_proj** (‖B‖ mean 0.76 / max 0.88, verified locally), backbone naming. Code diff vs the proven trainer = only `CSV_PATH` + `_DATA_TAG` (stress guards are no-ops in real training). Only data differs.
- **Data:** `260523_huikang_newsolver_goldonly.csv` (6906 unique → **7904 expanded**). Cryptarithm CoTs (newsolver): **285 gold-matching kept at 3× oversampling**, 515 dropped (346 GT-mismatch + 169 over-long). Non-cryptarithm huikang-golden rows untouched.
- **Train config:** Unsloth + manual loop + CCE + lm_head LoRA, transformers 4.56.2. AdamW lr 2e-4→0, batch 32 (micro4 ga8), seq 8192, 247 steps, seed 42. Final loss 0.0052. Env `MOE_TIE=0 USE_MEM_EFF=0 FRESH_START=1`. RTX PRO 6000.

| Recipe | Method | Kaggle ref | Score |
|---|---|---|---|
| final-247 (endpoint) | live out_proj, gold-only data | 52978378 | 0.84 |
| last5-soup | weighted SVD: 0.5 final + 0.125×4 (steps 198–247) | 52999385 | 0.85 |
| wise-soup | weighted SVD: 0.1/0.2/0.7 (steps 49/148/247) | 53029586 | 0.84 |

- **Why:** the most aggressive crypt manipulation (drop 515 CoTs incl all mismatches + 3×-oversample the surviving 285 gold) scored **0.84 — worst of the 3 endpoints** and below 0.86. The weighted **last5-soup lifted it to 0.85** (+0.01 over its own endpoint); the **wise-soup held at 0.84** — both still under 0.86.
- **Detail:** [TRAINING_INFO.md](2605231948_sft_newdata_goldonly_outproj_rtx6000/260524_0137_submission/TRAINING_INFO.md).

---

## Submission #20 — 2026-05-24 — Score 0.85 (gt2x data endpoint)

- **Study folder:** `2605232159_sft_newdata_gt2x_outproj_rtx6000`
- **LoRA architecture:** identical recipe to #16 — r32/α32, no-tie, **live out_proj** (‖B‖ mean 0.76 / max 0.88, verified locally), backbone naming. Code diff vs the proven trainer = only `CSV_PATH` + `_DATA_TAG`. Only data differs.
- **Data:** `260523_huikang_newsolver_gt2x.csv` (6906 unique → **7806 expanded**). Cryptarithm CoTs (newsolver): **285 gold-matching at 2× + 187 at 1×** kept, 328 dropped. Milder than gold-only (broader crypt coverage, lower oversampling). Non-cryptarithm untouched.
- **Train config:** Unsloth + manual loop + CCE + lm_head LoRA, transformers 4.56.2. AdamW lr 2e-4→0, batch 32 (micro4 ga8), seq 8192, 244 steps, seed 42. Final loss 0.0057. Env `MOE_TIE=0 USE_MEM_EFF=0 FRESH_START=1`. RTX PRO 6000.

| Recipe | Method | Kaggle ref | Score |
|---|---|---|---|
| final-244 (endpoint) | live out_proj, gt2x data | 52978951 | 0.85 |
| last5-soup | weighted SVD: 0.5 final + 0.125×4 (steps 195–244) | 52999394 | 0.85 |
| wise-soup | weighted SVD: 0.1/0.2/0.7 (steps 49/146/244) | 53029591 | 0.85 |

- **Why:** milder crypt treatment (2× on 285 + keep 187 at 1×, drop only 328) scored **0.85 — same as newsolver (#18), above gold-only (#19), still below 0.86**. The weighted **last5-soup and wise-soup both = 0.85** (= its endpoint). Across the 3 variants the pattern is consistent: broader/less-manipulated cryptarithm CoT scores higher, but our newsolver crypt data never reaches the original-data 0.86 (best soup = 0.85). **Net: the data swap did not beat 0.86; the original 735-lkevincc/65-huikang crypt CoT (#16) remains the best training signal.**
- **Detail:** [TRAINING_INFO.md](2605232159_sft_newdata_gt2x_outproj_rtx6000/260524_0224_submission/TRAINING_INFO.md).

---

## Submission #21 — 2026-05-25 — Score 0.85 (cross-data SVD soups, all 3 weightings)

- **Study folder:** none — no training. Derived from the #16 / #18 / #20 endpoint adapters. Build scripts `study/_tmp_crosssoup_build_w.py` (save_file) + `study/_tmp_crosssoup_build_stream.py` (low-RAM streaming); zips in `study/260524_crosssoup_submission/`.
- **LoRA architecture:** weighted SVD soup — rank-32 best approximation of `Σ wᵢ·Bᵢ@Aᵢ` over the 3 endpoints; no-tie, live out_proj inherited (out_proj ‖B‖ mean 1.40 / 1.41 / 1.42 across the 3 weightings).
- **Method:** cross-trajectory weighted SVD merge of three independent endpoints — #16 (0.86, original 735-lkevincc/65-huikang crypt), #18 newsolver (0.85), #20 gt2x (0.85) — at three weightings increasingly anchored on the 0.86 endpoint. Each zip verified locally (CRC OK, 12011 tensors, live out_proj) before upload.

| Recipe (0.86 / newsolver / gt2x) | Kaggle ref | Score |
|---|---|---|
| 0.6 / 0.2 / 0.2 | 53031172 | 0.85 |
| 0.8 / 0.1 / 0.1 | 53032040 | 0.85 |
| 0.9 / 0.05 / 0.05 | 53032239 | 0.85 |

- **Why:** all three cross-data soups scored **0.85 — below #16's 0.86**, including the **0.9-anchored** soup (90% weight on the exact 0.86 endpoint). Blending in any of the 0.85 newsolver/gt2x deltas — plus the rank-32 SVD re-truncation of the combined 96-rank stack — costs the 0.86. Reinforces #12 (cross-trajectory soup of same-data runs = no gain): cross-data soup of these variants does not exceed the best single endpoint. The original-data #16 endpoint (0.86) remains the ceiling.

---

## Submission #22 — 2026-05-26 — Score 0.84 (numeric-equation new-solver CoT swap; numeq vs allEq A/B)

- **Study folders:** `2605251948_sft_newdata_numeq_outproj_rtx6000` (numeq) + `2605252021_sft_newdata_numeq_alleq_outproj_rtx6000` (allEq).
- **LoRA architecture:** identical recipe to #16 — r32/α32, no-tie, **live out_proj** (‖B‖ 0.74 both, verified locally), backbone naming. Code diff vs the proven trainer = only `CSV_PATH` + `_DATA_TAG` (+ `ADAPTER_SAVE_STEPS=[]`, endpoint-only).
- **Data:** the #16 (0.86) ID set with the **numeric-equation CoTs swapped to our new "transformation-rule" solver** (non-equation rows byte-identical to #16). Two mismatch-handling variants:
  - **numeq** `260525_huikang_NumericEq.csv` (md5 f86286eb, 7849 expanded, 246 steps): drops 171 low-confidence eq rows (102 GT-mismatch + 69 over-long via oversampling=0); 12 mismatches survive. Final loss 0.0032.
  - **allEq** `260525_huikang_NumericEq_allEq.csv` (md5 d2fe439d, 8020 expanded, 251 steps): keeps ALL 732 new-solver eq rows incl. 114 GT-mismatch. Final loss 0.0024.
- **Train config:** Unsloth + manual loop + CCE + lm_head LoRA, transformers 4.56.2. lr 2e-4→0, bs32 (micro4 ga8), seq8192, seed42. Env `MOE_TIE=0 USE_MEM_EFF=0 FRESH_START=1`. RTX PRO 6000.

| Variant | Kaggle ref | Score |
|---|---|---|
| numeq (drop 171 low-conf eq CoTs) | 53044093 | 0.84 |
| allEq (keep all 732 eq CoTs)      | 53041884 | 0.84 |

- **Why:** swapping the numeric-equation CoTs to the new solver scored **0.84 — below #16's 0.86, and below the crypt-swap variants (#18/#20 = 0.85)**. **numeq == allEq**, so mismatch-handling is irrelevant — the **equation-CoT *style*** drives the −0.02, not the mismatch count. Verified NOT a bug (code byte-identical to proven trainer; data valid — boxed answers present 618/618, match gt 609/618, encoding clean incl. intentional §/±/× symbols). The new eq CoTs are a weaker signal: **<half the length** (mean 4.6K chars vs original eq CoTs' 10.5K), **heavily templated** (identical "Prior knowledge: four kinds of operations…" rule-dump + §1/§1.1 section scaffolding across all 732), with **terser arithmetic** than the original's explicit working (new: `36×48 = 1728; answer = 1728`; orig: `(30+6)*48 = 30*48 + 6*48 = 1440+288 = 1728`). Confirms the project-wide pattern: **every** new-solver CoT swap underperforms the original detailed CoTs (crypt → 0.85, numeric-eq → 0.84); #16 (0.86) remains the ceiling.
- **Note:** numeq run#1 derailed from a non-deterministic bf16 grad explosion at step 19 (grad_norm 2830, no clipping) → relaunched as-is (same seed), run#2 converged clean (did not reproduce — confirmed CUDA-nondeterministic, not data/code). See `project_grad_explosion_nondeterministic` memory.
- **Detail:** [numeq TRAINING_INFO](2605251948_sft_newdata_numeq_outproj_rtx6000/260526_1517_submission/TRAINING_INFO.md), [allEq TRAINING_INFO](2605252021_sft_newdata_numeq_alleq_outproj_rtx6000/260526_1343_submission/TRAINING_INFO.md).

---

## Submission #23 — 2026-05-27 — Score 0.84 (allEq + gt-True filter)

- **Study folder:** `2605270202_sft_newdata_alleqgt_outproj_rtx6000`
- **LoRA architecture:** identical recipe to #16 — r32/α32, no-tie, **live out_proj**, backbone naming. Code diff vs the proven gt2x trainer = only `CSV_PATH` + `_DATA_TAG`.
- **Data:** `260527_huikang_NumericEq_allEq_gtTrue.csv` (md5 `9628b30d04e4747992a4c6a8c7abf450`, **7883 expanded → 247 steps**). Variant of #22's allEq with **GT-mismatching new-solver equation CoTs dropped** — keeps only gt-matching new-solver eq rows. Non-equation rows byte-identical to #16.
- **Train config:** Unsloth + manual loop + CCE + lm_head LoRA, transformers 4.56.2. lr 2e-4→0, bs32 (micro4 ga8), seq8192, seed42. Final loss **0.0029**, 3.93 hrs. Env `MOE_TIE=0 USE_MEM_EFF=0 FRESH_START=1`. RTX PRO 6000.

| Recipe | Method | Score |
|---|---|---|
| final-247 (endpoint) | live out_proj, allEq+gtTrue eq data | 0.84 |

**huikang val 4-way comparison** (950 rows, per-token `(token, prob)` + live `rfind` extractor — identical setup across all 4 → directly comparable):

| huikang_category | N | moe (#16) | alleq (#22 allEq) | alleqgt (#23) | cryptnumeq (#24) |
|---|---:|---:|---:|---:|---:|
| equation_numeric_deduce | 65 | **0.9231** | 0.8923 | 0.8923 | 0.8923 |
| equation_numeric_guess | 19 | 0.0526 | 0.2632 | **0.3158** | **0.3158** |
| cryptarithm_deduce | 55 | **0.0364** | 0.0182 | 0.0182 | 0.0000 |
| cryptarithm_guess | 16 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| bit_manipulation | 160 | 0.7688 | 0.7562 | 0.7688 | **0.7812** |
| cipher | 158 | **0.9937** | **0.9937** | 0.9873 | 0.9873 |
| gravity | 160 | 0.9938 | **1.0000** | 0.9938 | **1.0000** |
| numeral | 158 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| unit_conversion | 159 | **1.0000** | 0.9937 | **1.0000** | **1.0000** |
| **OVERALL (val)** | **950** | 0.8621 | 0.8611 | 0.8632 | **0.8653** |
| **Public LB** | | **0.86** | 0.84 | 0.84 | 0.85 |

- **Why:** dropping the 114 GT-mismatching new-solver eq CoTs (vs #22 allEq's keep-all) held the LB at **0.84** — no improvement over #22 allEq. Filtering by gt-match did not recover the −0.02 vs #16. Per-category: equation_numeric_deduce 0.8923 (≡ #22, −3.1pp vs #16), equation_numeric_guess 0.3158 (best of the 4). The −0.02 LB gap vs #16 traces to the equation-CoT *style*, not mismatch filtering — consistent with #22.
- **Detail:** eval CSV at `03_eval/2605270202_newdata_alleqgt_outproj_950val/val_eval_results.csv`.

---

## Submission #24 — 2026-05-27 — Score 0.85 (combined crypt + eq new-solver)

- **Study folder:** `2605270226_sft_newdata_cryptnumeq_outproj_rtx6000`
- **LoRA architecture:** identical recipe to #16 — r32/α32, no-tie, **live out_proj**, backbone naming. Code diff vs the gt2x trainer = `CSV_PATH` + `_DATA_TAG` + `csv.field_size_limit(2**31-1)` (this data has 0×-dropped runaway crypt CoTs up to 6.8M chars; `DictReader` must parse every row before the 0× drop or it aborts).
- **Data:** `260527_huikang_NumericEq.csv` (in `260526_Cryptarithm_TrainData/`, **7917 expanded → 248 steps**). **Combines both new-solver swaps in one file:** 800 cryptarithm new-solver (152 GT-mismatch kept at 2× oversample + remainder at 1×) + 732 numeric-equation new-solver (gt-True only). Non-{crypt,eq} huikang-golden rows untouched.
- **Train config:** Unsloth + manual loop + CCE + lm_head LoRA, transformers 4.56.2. lr 2e-4→0, bs32 (micro4 ga8), seq8192, seed42. Final loss **0.009**, 4.02 hrs. Env `MOE_TIE=0 USE_MEM_EFF=0 FRESH_START=1`. RTX PRO 6000.

| Recipe | Method | Score |
|---|---|---|
| final-248 (endpoint) | live out_proj, combined crypt+eq new-solver | 0.85 |

- **Why:** combining the new-solver crypt + new-solver eq swaps into one training scored **0.85** — best of the new-solver runs to date (#18 newsolver crypt 0.85, #20 gt2x 0.85, #22 numeq/allEq 0.84, #23 alleqgt 0.84) but still **−0.01 below #16's 0.86**. Highest val OVERALL of the 4-way (**0.8653**) and best `bit_manipulation` (0.7812). Per-category vs #16: `cryptarithm_deduce` dropped 0.0364 → 0.0000 (combining the new-solver swaps did not recover the original 65-huikang crypt CoT signal lost in #18/#20). +0.01 LB vs #23 alleqgt is consistent with the +1.5pp `bit_manipulation` + +5pp `equation_numeric_guess` val improvements. 4-way val comparison table in #23.
- **Detail:** eval CSV at `03_eval/2605270226_newdata_cryptnumeq_outproj_950val/val_eval_results.csv`.

---

## Submission #25 — 2026-05-30 — Score 0.80 → 0.81 (2-epoch resume from #16's 0.86 adapter on filtered cryptnumeq; **regressed −0.05 / −0.06 vs #16**)

- **Study folder:** `2605301254_sft_resume086_filt98_rtx6000` (one run, paused after epoch 1 for Sub #25, resumed for epoch 2 / Sub #26 — both submissions logged here per routine §6 *"soup runs = ONE entry"*)
- **LoRA architecture:** identical recipe to #16 — r32/α32, no-tie, **live out_proj** (verified: `out_proj.lora_B norms [0.71, 0.64, 0.62, 0.66, 0.67]` at resume start = #16's converged ~0.7 signature), backbone naming.
- **Starting adapter:** **Sub #16's 0.86 endpoint adapter** (`submission_moe_notie_outproj.zip`, unzipped to `/root/autodl-tmp/adapter_086_init/` for ep1; ep2 resumed from ep1's `checkpoint-69_*/` with warm AdamW + full optimizer state continuity).
- **Data:** `260530_huikang_NumericEq_5cat15_crg3x.csv` (ep1; md5 `5551b007…`) + `260530_huikang_NumericEq_5cat15_crg3x_ep2.csv` (ep2; md5 `525ed18f…`). Both: 1,715 unique → 2,201 expanded → 69 batches at bs=32. 4 new-trace cats (eq_d/eq_g/cr_d/cr_g) **kept FULL** (cr_g bumped 2× → 3×); 5 "avoid-forget" cats (bit_manipulation/cipher/gravity/unit_conversion/numeral) **sampled at 15% with oversampling forced to 1**. Ep2 used a NON-OVERLAPPING 15% sample of the 5 avoid-forget cats (eq+crypt rows identical to ep1, second pass with warm optimizer). All rows GT-match=True.
- **Train config:** Unsloth + manual loop + CCE + lm_head LoRA, transformers 4.56.2. bs32 (micro4 ga8), seq8192, seed42, no-tie, live out_proj. Env: `MOE_TIE=0 USE_MEM_EFF=0` (no FRESH_START); ep2 also `TOTAL_STEPS=138`. RTX PRO 6000.

| Recipe | Steps | LR schedule | Final loss | Time | Kaggle ref | Score | Δ vs #16 |
|---|---|---|---|---|---|---|---|
| ep1 endpoint | 0-69 | 5e-5 → 0 linear over 69 | 0.0716 | 1.07 hr | 53197260 | **0.80** | **−0.06** |
| ep2 endpoint (warm resume) | 69-138 | 2.5e-5 → 0 linear (138-step schedule) | 0.0538 | 1.02 hr | 53200903 | **0.81** | **−0.05** |
| **combined run** | 0-138 | hybrid (steep 5e-5→0 for ep1, then 2.5e-5→0 for ep2) | 0.0538 | **2.09 hr** | | | |

- **Why — the resume regressed significantly below the starting point, falsifying the design hypothesis:**
  - **Hypothesis:** train the 0.86 adapter further on cryptnumeq's new-trace eq + crypt CoTs (which #16 has never seen) while protecting the saturated cats via 15% downsampling, expecting LB gain because the model learns new traces without forgetting the easy categories.
  - **Result:** the model regressed by 0.05-0.06 LB. Both epochs are below #16 (0.86) AND below cryptnumeq Sub #24 (0.85, full training from base on the same data). Continuing for a second epoch only recovered +0.01.
  - **What the data is consistent with:**
    1. **15% downsampling of avoid-forget cats let the model forget them.** Sub #24 (cryptnumeq, 100% on these cats) scored 0.85. Our 15% downsample scored 0.05 lower. Bit_manipulation/cipher/gravity/unit_conversion/numeral have a real maintenance cost that cannot be cut to 15% without measurable LB loss.
    2. **The new-trace eq + crypt CoTs, even with full exposure (as in Sub #24), only reach 0.85** — they don't add information that compensates for forgetting on the other cats.
    3. **Combining both losses:** the 0.86 → 0.80/0.81 gap = (15% avoid-forget penalty) + (new-trace eq/crypt does not help further on top of #16's already-strong baseline).
  - The +0.01 from ep2 over ep1 is within ±0.01 inference noise; not a meaningful gain from the second epoch.
  - **Falsified:** the "downsample the easy cats to 15% and force concentration on hard cats while resuming from #16" recipe. The avoid-forget cats need more coverage than 15% to be maintained.
- **Detail:** [ep1 TRAINING_INFO](2605301254_sft_resume086_filt98_rtx6000/260530_1625_submission/TRAINING_INFO.md), [ep2 TRAINING_INFO](2605301254_sft_resume086_filt98_rtx6000/260530_1852_submission/TRAINING_INFO.md).

---

## Submission #26 — 2026-05-31 — Scores 0.51 / 0.56 / 0.54 / 0.56 (DPO distillation sweep, 4 ckpts, all ≤0.56 vs SFT seed 0.85)

- **Study folder:** `2605301130_dpo_distill_alldata_rtx6000`
- **LoRA architecture:** r32/α32, MoE tie ON (`fixtie`), `USE_MEM_EFF=1` → **out_proj LoRA inert during DPO training** (seed values preserved verbatim, not updated). Target modules same as SFT 0.85.
- **Starting adapter:** SFT 0.85 cryptnumeq seed (`/root/autodl-tmp/sft_seed_cryptnumeq`, 12011 tensors, missing/unexpected = 0/0).
- **Data:** PREFER `prefer_260527_huikang_NumericEq.csv` (chosen = `solver_cot`) + REJECT `reject_numericeq.csv` (rejected = `raw_output`). Joined on `id` = 6429 pairs; post-filter at `EASY_CATEGORIES_VAL_GE_98={numeral, gravity, unit_conversion, cipher, bit_manipulation}` × `ANCHOR_FRACTION=0.10` → **1439 pairs**. Tokens: chosen 4.06M, rejected 4.45M. Diff% distribution: **92.4% of pairs ≤10% character diff** (chosen ≈ rejected), 7.6% genuinely different (mostly cryptarithm).
- **Train config:** Manual loop, NO TRL/accelerate/HF Trainer. GroupDPO gradient-decomposition (arXiv 2604.15602): 2 inference_mode forwards → off-graph s = σ(−β·Δ) → 2 with-grad forwards each immediately backward at ±β·s. Apple `linear_cross_entropy` on patched CCE; LoRA fp32, base bf16, both Phase 1 + Phase 3 under matching bf16 autocast. Unsloth smart-offloaded GC preserved end-to-end (root-caused as the 46-test "memory wall" when severed by TRL DPOTrainer). bs=16 (micro=4 → 4 micro-steps/optim step), seq=8192, β=0.05, lr=5e-7 (Tülu-3 DPO default), max_grad_norm=1.0, 1 epoch = 90 optimizer steps, ~1.64 hr. Peak VRAM 87.47 GB / 95 GB cap.
- **Training-time metrics looked structurally correct:** mean loss 0.6819 < ln(2)=0.6931; reward gap (chosen−rejected) grew +0.017 (steps 1–30) → +0.101 (steps 61–90); chosen-wins rate 60% → 85%.

| Checkpoint | Step | Single-batch loss at save | Kaggle ref | Public score | Δ vs SFT 0.85 |
|---|---|---|---|---|---|
| ckpt-25 | 25 | 0.6873 | 53233187 | **0.51** | **−0.34** |
| ckpt-50 | 50 | 0.5792 | 53233307 | **0.56** | **−0.29** |
| ckpt-75 | 75 | 0.6998 | 53233299 | **0.54** | **−0.31** |
| ckpt-90 (final) | 90 | 0.7173 | 53233178 | **0.56** | **−0.29** |

- **Why:** First DPO submission; intended to lift the SFT 0.85 baseline by preferring gold `solver_cot` over the model's `raw_output` on the same prompts. Every checkpoint regressed ≥0.29 instead. Damage is **fast and structural** (already at 0.51 by step 25 = 28% trained), not slow drift. Training-time loss/reward stats looked healthy, so the failure is in the setup or data — not over-training. Root cause candidates pending diagnosis in the EVALUATION write-up:
  1. **92% of pairs are near-identical chosen vs rejected** — noisy near-zero DPO gradient dominates the few genuinely-different pairs (cryptarithm). The diff% histogram was visible BEFORE launch and not acted on.
  2. **out_proj training-vs-inference path mismatch.** SFT 0.85 was trained with `USE_MEM_EFF=0` (out_proj LoRA actively trained). DPO ran with `USE_MEM_EFF=1` (fused Mamba kernel bypasses out_proj). The seed's trained out_proj LoRA was preserved verbatim, but the rest of the LoRAs were updated against a model whose effective forward path doesn't include out_proj — possible representation drift between training and Kaggle inference (which exercises out_proj).
  3. Manual loop (no TRL): possible subtle bugs in the per-branch backward coefficient sign / per-token weighting that aren't visible in aggregate loss/reward stats.
- **Detail:** [TRAINING_INFO.md](2605301130_dpo_distill_alldata_rtx6000/260531_2319_submission/TRAINING_INFO.md). EVALUATION.md pending (see todo: out_proj path verification + diff_pct≥25 filter rebuild).

---

## Key takeaways

- **★ `out_proj` (live) is the 0.84→0.86 lever (Sub #16).** Training the Mamba `out_proj` LoRA — dead by default because the fused kernel bypasses it — lifts our OWN trained model to **0.86**, matching huikang's Tinker wall, with **no tie** (no-tie + live out_proj; tying HURTS, Sub #14 = 0.71). Fix = keep `is_fast_path_available=True` and set each mixer's `.training=False` so it takes the unfused else-branch (`out_proj` as a module call); no model-file patching, same VRAM/speed. **First trained adapter of ours to clear 0.84.** Mechanism: `02_train/260522_moe_expert_rank1_bug_report.md`.
- bf16 + CCE + MoE tying + lm_head LoRA + huikang config = 0.84 baseline (Subs #1 / #4 / #6 / #7) — **but that baseline has a DEAD `out_proj`; training it → 0.86 (Sub #16).**
- **The 0.86 LB wall = huikang's published adapter (Sub #13), and now ALSO reachable by our own training (Sub #16).** Submitting `kienngx/…/tinker-adapter/1` verbatim scored 0.85 then 0.86. Our trained repro topped out at 0.84 *until* we trained `out_proj` live (Sub #16 → 0.86). His real Tinker weights serve at 0.85–0.86 and also have a live `out_proj` (‖B‖≈0.70) — consistent. Going beyond 0.86 now needs better data/recipe on top of live `out_proj`.
- **Kaggle inference is non-deterministic: ±0.01 score noise on the SAME submission file.** Sub #11's byte-identical endpoint scored 0.83 then 0.84 (refs 52813620 / 52865914); Sub #13's byte-identical 0.86-adapter scored 0.85 then 0.86 (refs 52912925 / 52915203). Greedy decoding on a bf16 MoE model has atomicAdd non-determinism → a few generations flip → ±0.01. **Treat any single-submission Δ ≤ 0.01 as noise, not signal.**
- **Single-trajectory soup is NOT a proven win.** Once the ±0.01 inference noise is known, the "soup-last5 = +0.01 over endpoint" (Sub #10/#11) is inside the noise floor — soup-last5 and the endpoint are indistinguishable (both 0.83/0.84). Don't claim soup-last5 helps. (It doesn't hurt either — safe to submit, but no demonstrated edge.)
- **WiSE (first+mid+final) soup = real loss** (Sub #11 −6/7pp ≫ noise). Mixing early checkpoints into a converged endpoint hurts. SVD-merge beats naive only for this distant-ckpt case; within noise for last-K.
- **Cross-trajectory soup = no gain, even across genuinely different data** (Sub #12 identical-data 0.84; **Sub #21 cross-DATA** #16⊕#18⊕#20 all **0.85 ≤ #16's 0.86**, even 0.9-anchored on the 0.86 endpoint). #12 suggested "needs genuinely diverse trajectories (different data)" — #21 falsifies that: blending the 0.85 newsolver/gt2x deltas into the 0.86 endpoint + rank-32 SVD re-truncation still costs the 0.86. Soup is not a path past the best single endpoint here.
- Sub #10's endpoint 0.76 is a real low outlier (8pp ≫ ±0.01 inference noise) — a bad training-trajectory draw from CUDA backward non-determinism, not inference noise.
- Score drops have been **data composition / extraction bugs** (Subs #2 / #3 / #5 / #9) or **trajectory variance** (Sub #10). A handful of broken training rows can crash the score 16pp (Sub #5 → #6).
- Text CSV pipeline matches pretok when byte-identical; stratified batching is not a cause of any drop.
- When eval and training data disagree, **trust the experiment** — re-train with the fix and measure.

---

## Submission #27 — 2026-06-04 — Scores 0.83 / 0.84 / 0.83 (cryptarithm_gtTrue, 3 zips, all below #16's 0.86)

- **Study folder:** `2606032253_sft_cryptarithm_gtTrue_rtx6000`
- **Data:** `260601_Cryptarithm_gtTrue.csv` (68.7 MB, md5 `ec6c1404688adb804d8badc57118476d`, 7077 raw → 6441 unique (oversampling>0) → **7756 expanded → 243 steps**). **Base = `260514_huikang_golden_stripped.csv` (SAME as #16)**; eq + crypt rows have their CoTs swapped to `260601_new_solver`, plus +171 new eq puzzles appended. Source labels on the surviving rows: `pretok_0408_decoded` (5545 unique = huikang-golden's own internal source label, byte-identical to #16) + `260601_new_solver` (896 unique = the swapped eq/crypt rows). 636 rows zeroed: all `260601_new_solver` eq+crypt (451 GT-False + 185 GT-True over the 7680-token cap, all cats cryptarithm_*/equation_numeric_*); the 5 avoid-forget cats (cipher/bit_manipulation/unit_conversion/gravity/numeral) are **byte-identical to #16, untouched**. Expanded by cat: bit_manipulation 1754, cipher 1656, unit_conversion 1070, gravity 1055, equation_numeric_deduce 825, numeral 730, cryptarithm_deduce 498, equation_numeric_guess 116, cryptarithm_guess 52.
- **Method:** SAME recipe as #16 (verbatim copy of cryptnumeq SFT 0.85 trainer per [[feedback_copy_sft_verbatim]]). r32/α32 LoRA on q/k/v/o/up/down/in/out_proj + lm_head, **no-tie**, **live out_proj**. bf16 + CCE + AdamW(0.9, 0.95) + wd=0, lr 2e-4→0 linear, bs32 (micro4 ga8), seq8192, seed42. Env `MOE_TIE=0 USE_MEM_EFF=0 FRESH_START=1`. RTX PRO 6000. Routine §1.6 out_proj-alive check at stress step 1 = grad norm sum 0.1478 > 0 ✓. 4.09 hr wall, peak 72.2 GB VRAM, final loss 0.0013, 0 grad spikes. Weighted SVD soups (`svd_soup_weighted.py` md5 `51c8045e`) on adapter-saves [49, 146, 194, 207, 219, 231, 243].

| Recipe | Method | Kaggle ref | Score | Δ vs final-step |
|---|---|---|---|---|
| final-step (243) | endpoint | 53378383 | **0.83** | — |
| last5-soup {194,207,219,231,243} | wSVD, final=0.5 + others=0.125 | 53378388 | **0.84** | +0.01 |
| wise-soup {49,146,243} | wSVD, 0.1/0.2/0.7 | 53379681 | **0.83** | 0.00 |

- **Why — swapping huikang's eq+crypt CoTs for new-solver CoTs cost 0.02–0.03 LB; the avoid-forget cats are untouched, so the loss is fully in eq+crypt:**
  - **Recipe is byte-identical to #16:** same r/α, same targets, same optimizer/LR/bs/seq/steps (243 vs 246), same seed, same env, live out_proj verified at stress step 1. Loss / peak VRAM / wall time all in the #16 envelope. Recipe is not the lever.
  - **Base data IS huikang_golden_stripped — NOT a different dataset.** Earlier I claimed the base was switched to `pretok_0408_decoded` — that was wrong. `pretok_0408_decoded` is the internal source-string huikang used on his own golden rows, not a separate dataset. Verified by reading `01_data/260526_Numeric_Equation/260601_NumericEq_TrainData/_build_train.py` (line 19: `BASE = .../260514_huikang_golden_stripped.csv`). The 5 avoid-forget cats are **byte-identical to #16**.
  - **The actual data deltas vs #16:**
    1. **Eq rows (561 base + 171 appended = 732) have CoTs swapped** from huikang's originals to `260601_new_solver`'s.
    2. **Crypt rows have CoTs swapped** from huikang's originals to `260601_new_solver`'s, **GT-True ones at 2× oversample, GT-False zeroed**.
    3. **636 zeroed rows are all `260601_new_solver` eq+crypt** (185 GT-True over the 7680 cap + 451 GT-False) — not avoid-forget cats. The "GT-True filter" only affects swapped rows.
  - **#24 (0.85) is the closest baseline** — also a huikang-base + new-solver eq+crypt swap, scored 0.85. My run's −0.01 to −0.02 vs #24 is plausibly the GT-True filter on crypt: #24 kept 152 GT-mismatch crypt rows at 2×; I zeroed them. Whether feeding GT-mismatch crypts helps or hurts isn't settled by this run alone; could also be inference noise per [[feedback_noise_not_002]].
  - **#16 vs my run: −0.02 to −0.03 LB.** Since the avoid-forget cats are byte-identical, the loss is entirely from the eq+crypt CoT swap. Huikang's original eq+crypt CoTs train better than the new-solver substitutes on these categories, at least in this composition. This replicates the #24 finding with one more data point.
  - **last5-soup 0.84 ≈ final 0.83:** +0.01 = inference noise (per [[feedback_noise_not_002]]: noise is ±0.01 max, ≥0.02 is signal). Soup doesn't rescue the data deficit; wise-soup matches the endpoint at 0.83.
  - **Falsified:** "new-solver eq+crypt CoTs (with GT-True filter) ≥ huikang's original eq+crypt CoTs on the same base." Need a different angle to clear 0.85 — the new-solver CoT pool as currently generated does not lift these categories.
- **Correction note:** the initial version of this entry claimed the base data source was switched to `pretok_0408_decoded`. That was wrong — `pretok_0408_decoded` is huikang_golden's own internal label; the base IS huikang_golden_stripped (verified at `_build_train.py:19`). Corrected after user pushback ("are you sure?"). [[feedback_no_wrong_assumptions]] applied.
- **Detail:** [TRAINING_INFO.md](2606032253_sft_cryptarithm_gtTrue_rtx6000/260604_0330_submission/TRAINING_INFO.md).

---

## Submission #28 — 2026-06-07 — Score 0.84 (cryptarithm_gtTrue 260606 solver refresh, ≈ #27 noise)

- **Study folder:** `2606070339_sft_cryptarithm_gtTrue_260606swap_rtx6000`
- **Data:** `260607_Cryptarithm_gtTrue.csv` (md5 `e0e0c02d6f13a4cdc63893096c84a464`, 7077 raw → 6441 train rows → 7756 expanded → 243 steps). Sources within training: `pretok_0408_decoded` (5545 = huikang's internal label) + `260601_new_solver` (275 cryptarithm) + `260606_new_solver` (621 equation_numeric). Net delta vs the #27 (06-04) CSV: the 621 equation_numeric rows have their CoTs regenerated by the 260606 solver run (different `source` label); 275 cryptarithm rows are still labeled `260601_new_solver`. Expanded category breakdown: bit_manipulation 1754, cipher 1656, unit_conversion 1070, gravity 1055, equation_numeric_deduce 825, numeral 730, cryptarithm_deduce 498, equation_numeric_guess 116, cryptarithm_guess 52.
- **Method:** SAME recipe as #27 byte-identical (verbatim copy of cryptnumeq SFT 0.85 trainer per [[feedback_copy_sft_verbatim]]). r32/α32 LoRA on q/k/v/o/up/down/in/out_proj + lm_head, **no-tie**, **live out_proj**. bf16 + CCE + AdamW(0.9, 0.95) + wd=0, lr 2e-4→0 linear, bs32 (micro4 ga8), seq8192, seed42. Env `MOE_TIE=0 USE_MEM_EFF=0 FRESH_START=1`. RTX PRO 6000. Routine §1.6 out_proj-alive at stress step 1 = grad norm sum 0.1476 > 0 ✓. Wall 4.00 hr, peak 72.2 GB VRAM, final loss 0.0015, 0 grad spikes.

| Recipe | Method | Kaggle ref | Score | Δ vs #27 endpoint |
|---|---|---|---|---|
| final-step (243) | endpoint | 53460361 | **0.84** | +0.01 (within noise) |

- **Why — refreshing 621 eq CoTs from the 260606 solver did not move the LB out of #27's noise band:**
  - The 06-04 run (#27) scored 0.83 / 0.84 / 0.83 (endpoint / last5-soup / wise-soup). This run reused the same recipe + same crypt rows + same avoid-forget rows; the only data change is the 621 equation_numeric CoTs now come from the 260606 solver (vs `260601_new_solver` in #27).
  - The 0.84 result is +0.01 vs the #27 endpoint, which is inside the ±0.01 inference-noise floor (per byte-identical resubmits in #11 / #13). Per [[feedback_noise_not_002]], a <0.02 delta is not signal.
  - **What this falsifies / confirms:** refreshing the eq CoTs from the 260606 solver does not lift LB above the cryptarithm-gtTrue / new-solver-eq family's 0.83–0.85 plateau established by Subs #18–#24, #27. The eq+crypt new-solver substitution ceiling is unaffected by which version of the new-solver you use.
  - Same recipe carrying the same eq+crypt new-solver substitution lever ⇒ same LB band as #27. The 260606 solver is not a lever for this regime.
- **Detail:** [TRAINING_INFO.md](2606070339_sft_cryptarithm_gtTrue_260606swap_rtx6000/260607_1532_submission/TRAINING_INFO.md).

---

## Submission #29 — 2026-06-07 — Score 0.83 (bitman per-bit ✓/✗ verification on #16's data; **regressed −0.03 vs #16 — Taha anti-pattern verified**)

- **Study folder:** `2606071542_sft_bitman_failcorr_rtx6000`
- **Data:** `260516_bit_manipulation_failcorr.csv` (md5 `a9c2349c3cc88eddeebbc11709a443fa`, 35.0 MB, 6905 unique → 7847 expanded → 246 steps). **Base = `260514_huikang_golden_stripped.csv` (#16's exact data, md5 `92a515f31cd2e89f9cb355c4ed09f927`)**; the modification is a `Verifying candidates by per-example check / Bit i: tried <X> → ✓✗✓✓✓✓✓✓ reject; chose <Y> → ✓✓✓✓✓✓✓✓ accept` block inserted **between the rule-analysis section and the `Selected` block** on 1,280 of 1,354 bit_manipulation rows (the rest of the CoT is byte-identical). Each per-bit "tried" candidate is a real near-miss rule (fits N−1 of N examples). +~140 tokens per augmented row. Non-bit_manipulation rows verified byte-identical to #16 (5/5 sampled cats: cipher / unit_conversion / gravity / numeral / equation_numeric_deduce) **except one cryptarithm_deduce row id `b1b10e83` (CoT differs from #16; minor)**. Categories: −1 bit_manipulation row unique (−2 expanded) and −1 cryptarithm_deduce CoT changed; all other categories byte-identical to #16. Per the failcorr README: hypothesis was "12% → 20% on the 25 out-of-train bit_man val IDs = worth iterating; → 30% = strong signal; no movement = format change alone can't force verification at inference."
- **Method:** SAME recipe as #16 byte-identical (verbatim copy of cryptnumeq SFT 0.85 trainer per [[feedback_copy_sft_verbatim]]). r32/α32 LoRA on q/k/v/o/up/down/in/out_proj + lm_head, **no-tie**, **live out_proj**. bf16 + CCE + AdamW(0.9, 0.95) + wd=0, lr 2e-4→0 linear, bs32 (micro4 ga8), seq8192, seed42. Env `MOE_TIE=0 USE_MEM_EFF=0 FRESH_START=1`. RTX PRO 6000. Routine §1.6 out_proj-alive at stress step 1 = grad norm sum 0.1592 > 0 ✓. Wall 4.21 hr, peak 72.2 GB VRAM, final loss 0.0045, 0 grad spikes.

| Recipe | Method | Kaggle ref | Score | Δ vs #16 endpoint |
|---|---|---|---|---|
| final-step (246) | endpoint, per-bit verify augmentation | 53464866 | **0.83** | **−0.03** |

- **Why — the per-bit ✓/✗ verification augmentation cost 0.03 LB on identical underlying data; STRATEGY's Taha anti-pattern is verified by direct experiment:**
  - **The data delta is small and intentional:** 1,280 of 1,354 bit_manipulation CoTs (94.5%) got a `Verifying candidates / Bit i: tried <X> → ✓✗✓✓✓✓✓✓ reject; chose <Y> → ✓✓✓✓✓✓✓✓ accept` block inserted before the `Selected` choice; everything else (rule-analysis prefix, `Selected` block, `Applying to <query>` block, `\boxed{}` close) is preserved verbatim from #16. The non-bit_man rows are byte-identical to #16 (verified on 5 sample rows across 5 cats; 1 cryptarithm_deduce CoT also differs — minor secondary effect).
  - **Recipe is byte-identical to #16:** same r/α, same targets, same optimizer/LR/bs/seq, +3 more total steps (246 vs 246; trivial). Stress out_proj alive at 0.1592 (vs #16's typical ~0.15 band). Wall time / loss / peak VRAM in the #16 envelope.
  - **Result: 0.83 (–0.03 vs #16's 0.86).** Per [[feedback_noise_not_002]], a ≥0.02 LB gap is real signal, not inference noise. The training data is nearly identical to #16 by composition, yet the model trained on it lost 0.03 LB. The augmentation is the only meaningful change → **the augmentation caused the regression**.
  - **STRATEGY_0.84_to_0.86.md verbatim warning was accurate:** *"Taha hit 98.9% on his bit-serial generator but his trained model's LB dropped — he 'poisoned' it via (a) duplicate CoTs collapsing gradients and (b) format clash (rank-32 LoRA spent the epoch learning new formatting instead of math)."* The failcorr augmentation is a milder version of the same anti-pattern (insertion not wholesale rewrite), and it produced a milder version of the same outcome (−0.03 not whatever Taha's was).
  - **Training-loss early-warning signal that I missed:** at step 79 failcorr loss = 0.008 vs #16's 0.004 (+100%); at end failcorr loss = 0.0045 vs #16's 0.0030 (+50%). I read that elevation as the "augmentation overhead" of extra ✓/✗ tokens. The actual mechanism is plausibly the inverse — the LoRA capacity was being absorbed by the format-prediction task, not by the underlying math. The lower training loss on the augmented format (failcorr beat #27 at every step) was easy memorization, not generalization improvement.
  - **The failcorr README's own decision rule:** *"12% → 20% = worth iterating; 12% → 30% = strong signal; no movement = format change alone can't force verification at inference."* Public LB shows no movement (in fact regression). The "no movement" branch is verified: the format change alone does not force verification at inference; the model evidently learns to emit the ✓/✗ blocks but does not generalize the per-example verification mechanism.
  - **What this falsifies:** the hypothesis that "insert per-bit verification block before `Selected`, keep rest of huikang's skeleton" stays close enough to huikang's CoT to inherit the 0.86 anchor. It doesn't — even an insertion-only augmentation triggers enough LoRA-capacity reallocation to format learning to cost LB.
  - **What this confirms:** the path past 0.86 is not stylistic CoT modification on the same data. Past evidence said this already (#16's lkevincc CoT remains the best signal; #18–#24, #27 newsolver-class variants all 0.83–0.85). #29 adds bit-augmentation-class variant to that list.
- **Detail:** [TRAINING_INFO.md](2606071542_sft_bitman_failcorr_rtx6000/260607_2057_submission/TRAINING_INFO.md).

---

## Submission #30 — 2026-06-09 — Score 0.84 (cryptaug_1000: huikang base + 1458 NE_aug/crypt_aug rows + new_solver eq; **−0.02 vs #16, joins the 0.83–0.85 plateau**)

- **Study folder:** `2606072355_sft_cryptaug_1000_rtx6000`
- **Data:** `260607_Cryptarithm_1000_FULL.csv` (md5 `928065b844f19297ce144541e40620ad`, 75.75 MB, 8535 rows → 7931 train (oversampling>0) → 8687 expanded → 272 steps). Composition: **pretok_0408_decoded 5545 (huikang's original CoTs from #16's base)** + **260607_NE_aug 765** (numeric-equation augmentation) + **260607_crypt_aug 693** (1000-problem cryptarithm augmentation) + **260606_new_solver 621** (same eq solver source as #28) + **260601_new_solver 307** (same source as #27). 100% GT-True. 0 rows over 7680 cap. Categories (9): bit_manipulation 20.2%, cipher 19.1%, equation_numeric_deduce 14.6%, equation_numeric_generate 1.7%, cryptarithm_deduce 10.2%, cryptarithm_generate 1.4%, gravity 12.1%, numeral 8.4%, unit_conversion 12.3%. Net lever vs #16 = +1458 crypt_aug/NE_aug rows (≈17% of expanded) on top of an otherwise huikang-style base; eq new_solver overlap covers the #27/#28 lever too.
- **Method:** SAME recipe as #16 / #29 byte-identical (verbatim copy of bitman_failcorr trainer per [[feedback_copy_sft_verbatim]]; only `CSV_PATH` and `_DATA_TAG` changed). r32/α32 LoRA on q/k/v/o/up/down/in/out_proj + lm_head, **no-tie**, **live out_proj**. bf16 + CCE + AdamW(0.9, 0.95) + wd=0, lr 2e-4→0 linear, bs32 (micro4 ga8), seq8192, seed42. Env `MOE_TIE=0 USE_MEM_EFF=0 FRESH_START=1`. RTX PRO 6000. Routine §1.6 out_proj-alive at stress step 1 = 0.1288 > 0 ✓. Wall 4.43 hr, peak 72.1 GB VRAM, final loss 0.0234, 0 grad spikes. No soup build this run (saved 7 adapter snapshots [54, 163, 218, 231, 245, 258, 272] for potential post-hoc build).

| Recipe | Method | Kaggle ref | Score | Δ vs #16 endpoint |
|---|---|---|---|---|
| final-step (272) | endpoint, cryptaug_1000 data | 53492865 | **0.84** | **−0.02** |

- **Why — adding 1458 NE_aug/crypt_aug rows + new_solver eq on top of huikang's base landed at 0.84, same as #28, −0.02 vs #16's 0.86:**
  - **Recipe is byte-identical to #16 / #29.** Same r/α, targets, optimizer/LR/bs/seq, seed. The only intentional lever is the data composition. +26 more steps (272 vs 246) reflects the larger expanded count, not a recipe change.
  - **Result: 0.84 (−0.02 vs #16, same as #28, +0.01 vs #29).** Per [[feedback_noise_not_002]] the −0.02 gap to #16 is real signal, not inference noise; the +0.00 to #28 and +0.01 to #29 are inside the ±0.01 noise floor.
  - **What this falsifies:** the hypothesis that augmenting huikang's base with 1000 fresh cryptarithm problems + 765 numeric-equation augmentations + the 260607 crypt_aug variants would push the LB above the 0.86 anchor. It doesn't — even keeping 5545 of huikang's original rows as the majority (~64% of expanded data) is insufficient when the remaining ~36% comes from augmentation / new-solver classes that the prior runs have already proven to plateau at 0.83–0.85.
  - **What this confirms:** the 0.83–0.85 plateau established by Subs #18–#24, #27, #28 (and #29 from the format-augmentation angle) absorbs the cryptaug-class data too. The aug-class data ceiling is reproducible across NE_aug, crypt_aug, new_solver_eq, bitman_failcorr — every CoT-source substitution we've tried lands in the same band.
  - **What this implies for path past 0.86:** consistent with #29's conclusion — the lever past 0.86 is not "swap or augment CoTs on top of huikang's skeleton." The 0.86 anchor remains exclusive to #16's exact 260514_huikang_golden_stripped composition. Candidate next levers (not yet tested in this branch): rank > 32, longer training on the exact #16 data, RL/DPO from #16's adapter as base, or a non-CoT-substitution change to the training signal.
- **Detail:** [TRAINING_INFO.md](2606072355_sft_cryptaug_1000_rtx6000/260608_0448_submission/TRAINING_INFO.md).

---

## Submission #31 — 2026-06-10 — Score 0.84 (Stage 1 of 2-stage curriculum: eqHard_AUG0x; survived a single late-train grad spike)

- **Study folder:** `2606092318_sft_eqHard_AUG0x_rtx6000`
- **Data:** `260609_Crypt3k_eqHardened_AUG0x_FULL.csv` (md5 `6c2ad37022f1e6b91aa4ad5703847c07`, 99 MB, 11345 rows → 6473 oversampling>0 → 7229 expanded → 226 steps). AUG0x dampens augmented rows: cryptarithm_deduce 3041 raw → 280 expanded; equation_numeric_deduce 1989 → 599. Sources (raw): pretok_0408_decoded 5545 (huikang base) + 260608_crypt10k 2693 + 260607_NE_aug 1575 + 260601_new_solver 800 + 260606_new_solver 732. 100% GT-True. Stage 1 of user-specified 2-stage curriculum (Stage 2 = same data + aug-row oversampling on cryptarithm + numeric_equation; Stage 2 will continue from this Stage 1 adapter via `INIT_ADAPTER_FROM`).
- **Method:** SAME recipe as #16 / #29 / #30 byte-identical (verbatim copy of `train_cryptaug_1000.py` per [[feedback_copy_sft_verbatim]]; only `CSV_PATH` + `_DATA_TAG` + docstring changed). r32/α32 LoRA on q/k/v/o/up/down/in/out_proj + lm_head, **no-tie**, **live out_proj**. bf16 + CCE + AdamW(0.9, 0.95) + wd=0, lr 2e-4→0 linear, bs32 (micro4 ga8), seq8192, seed42. Env `MOE_TIE=0 USE_MEM_EFF=0 FRESH_START=1`. RTX PRO 6000. Routine §1.6 out_proj-alive at stress step 1 = 0.1430 > 0 ✓. Wall 3.78 hr, peak 71.96 GB VRAM, final loss 0.0057. Soup adapter saves [45, 136, 181, 192, 203, 215, 226]. **1 grad spike at step 221: `grad_norm=28086.92`** (lr 5.31e-06 at that step → effective per-param update ≈ 0.149 vs typical ~1e-4). Steps 222–226 recovered to grad ~0.01 and finished cleanly.

| Recipe | Method | Kaggle ref | Score | Δ vs #16 endpoint |
|---|---|---|---|---|
| final-step (226) | endpoint, eqHard_AUG0x data, post-spike adapter | 53539358 | **0.84** | **−0.02** |

- **Why — Stage 1 lands at the 0.83–0.85 plateau like every other aug-class data run (#28, #30, etc.); the spike did not derail the LB:**
  - **The data is aug-class.** Despite AUG0x oversampling dampening the aug rows heavily, the trainable mix still includes 260607_NE_aug + 260608_crypt10k + new_solver_eq alongside huikang's pretok_0408_decoded base. Per #28/#30/#29's pattern, every aug-class data variant lands at 0.83–0.85, never breaking 0.86. Stage 1's 0.84 is consistent with that plateau.
  - **The grad spike did not corrupt the adapter (or did so within tolerable LB bounds).** Per [[feedback_noise_not_002]] a ≥0.02 LB gap is real signal — 0.84 matches the aug-class plateau exactly, so we can't distinguish "spike harmless" from "spike caused exactly enough damage to hit the plateau the data already implies" without a no-spike reference. The matching match memory [[project_grad_explosion_nondeterministic]]: these rare bf16/MoE spikes don't reproduce on relaunch with same seed, and a fresh run might or might not hit the same number.
  - **What this implies for Stage 2:** Stage 1 adapter is healthy enough to seed Stage 2. The lever Stage 2 tests is whether oversampling the cryptarithm + numeric_equation aug rows on top of a Stage 1 adapter that's already absorbed AUG0x can push past 0.84 — i.e., whether targeted strengthening via continued training can exit the 0.83–0.85 plateau that single-shot training cannot.
- **Detail:** [TRAINING_INFO.md](2606092318_sft_eqHard_AUG0x_rtx6000/260610_0340_submission/TRAINING_INFO.md).

---

## Submission #32 — 2026-06-11 — Score 0.86 (Stage 2 of 2-stage curriculum: eqHard_FULL continued from Stage 1; **+0.02 vs Stage 1, MATCHES #16's 0.86 anchor — first run since #16 to break the 0.83–0.85 plateau**)

- **Study folder:** `2606100600_sft_eqHard_FULL_contS1_rtx6000`
- **Data:** `260609_Crypt3k_eqHardened_FULL.csv` (md5 `46b787fd33eea30d3766d72ea45a5e89`, 99 MB, 11345 rows → 10741 oversampling>0 → 11497 expanded → 360 steps). Same source mix as Stage 1's AUG0x (pretok_0408_decoded 5545 + 260608_crypt10k 2693 + 260607_NE_aug 1575 + 260601_new_solver 800 + 260606_new_solver 732) but with aug-row oversampling restored: cryptarithm_deduce 280 expanded (Stage 1) → 2682 (Stage 2, +2402), equation_numeric_deduce 599 → 1992 (+1393). The +4268 expanded rows are concentrated on cryptarithm + numeric_equation categories. 100% GT-True.
- **Method:** SAME recipe as Stage 1 byte-identical (verbatim copy + INIT_ADAPTER_FROM block) per [[feedback_copy_sft_verbatim]]. r32/α32 LoRA on q/k/v/o/up/down/in/out_proj + lm_head, **no-tie**, **live out_proj**. bf16 + CCE + AdamW(0.9, 0.95) + wd=0, lr 2e-4→0 linear over 360 steps (fresh schedule), bs32 (micro4 ga8), seq8192, seed42. Env `MOE_TIE=0 USE_MEM_EFF=0 FRESH_START=1 INIT_ADAPTER_FROM=/root/autodl-tmp/stage2_adapter_init` (extracted from Stage 1's submission.zip). RTX PRO 6000. Stage 1 adapter load verified at startup: 12010 LoRA tensors loaded, 0 missing, lora_B norm 1.46 (trained-adapter, not random init). Step-1 loss 0.043 (vs Stage 1's step-1 loss 0.57 from random — proves init came from a trained adapter). Wall 6.19 hr, peak 71.9 GB VRAM, final loss 0.0270, **0 grad spikes** (Stage 1 had 1 spike at step 221; Stage 2 was clean throughout).

| Recipe | Method | Kaggle ref | Score | Δ vs Stage 1 (0.84) | Δ vs #16 (0.86) |
|---|---|---|---|---|---|
| final-step (360) | endpoint, eqHard_FULL data, continual-FT from Stage 1 adapter | 53553649 | **0.86** | **+0.02** | **0.00 (tied)** |

- **Why — 2-stage curriculum SFT broke the 0.83–0.85 aug-class plateau and matched #16's 0.86 anchor; first run since #16 to reach 0.86, validates the curriculum hypothesis the user designed:**
  - **The user's hypothesis tested:** "first train without aug data to build the ability of all puzzles (Stage 1 AUG0x), then strengthen the numeric eq and cryptarithm" (Stage 2 FULL with aug oversampling on those two categories). The hypothesis was that single-shot SFT on the FULL data (aug-class plateau, 0.83–0.85) would not match #16 because the model can't allocate capacity to both the base puzzles and the oversampled aug rows simultaneously, but a 2-stage curriculum could.
  - **Result confirms the hypothesis.** Stage 1 (AUG0x = base puzzles only) → 0.84 (plateau, expected). Stage 2 (continual FT, FULL data with crypt + eq oversampling) → 0.86. The +0.02 lift is REAL per [[feedback_noise_not_002]]. **This is the first run since #16 to reach 0.86 in the project** (since 05-23). Sub #16, #17, #18, #19, #20, #21, #22, #23, #24, #25, #26, #27, #28, #29, #30, #31 all landed at 0.83–0.85.
  - **What this falsifies:** the prior conclusion (from #16, #18–#24, #27–#31) that "the path past 0.86 is not stylistic CoT modification on the same data" or "aug-class data has a hard 0.83–0.85 ceiling." Both are too strong — the 2-stage curriculum SHOWS aug-class data CAN reach 0.86 when applied via continual FT to a base-trained adapter. Single-shot on the same data hits the plateau; staged training escapes it.
  - **What this confirms / new findings:**
    1. **Curriculum SFT works on this competition.** Order matters: base-then-strengthen ≠ all-at-once. The single-shot FULL data (which would be Sub #30-style) plateaus; the 2-stage version (Sub #31 → #32) reaches the anchor.
    2. **The INIT_ADAPTER_FROM continual-FT pattern is sound.** PEFT key rename (Kaggle-compat `backbone.lm_head.*` ↔ training-time `lm_head.*`) was the only real engineering issue and is now handled in the trainer. Future stages can reuse the same pattern verbatim.
    3. **#16's 0.86 anchor is no longer a unique configuration.** It was reachable by another data composition + recipe combo, just not by single-shot SFT on aug-class data.
  - **What this implies for Stage 3:** Stage 3 (in progress / pending download) extends the same curriculum lever: continual FT from Stage 2's 0.86 adapter on `260609_Crypt10k_eqHardened_FULL.csv` (+7,307 more crypt10k rows, 18,804 expanded total, 588 steps). The test is whether further strengthening past 0.86 is possible by escalating the cryptarithm count. If Stage 3 lands at 0.87+, the curriculum lever is monotonic. If it lands back at 0.84–0.85, there's a sweet spot in aug-row count that Stage 2 hit and Stage 3 overshoots (over-strengthening collapses the LoRA capacity allocated to other categories).
- **Detail:** [TRAINING_INFO.md](2606100600_sft_eqHard_FULL_contS1_rtx6000/260610_1929_submission/TRAINING_INFO.md).

---

## Submission #33 — 2026-06-11 — Score 0.84 (Stage 3 of 3-stage curriculum: crypt10k_FULL continued from Stage 2; **−0.02 vs Stage 2, regressed back to 0.83–0.85 plateau — curriculum is NOT monotonic, Stage 2 was the sweet spot**)

- **Study folder:** `2606102132_sft_crypt10k_FULL_contS2_rtx6000`
- **Data:** `260609_Crypt10k_eqHardened_FULL.csv` (md5 `0654f4639b0b51c0c01f5fe0edebf4dd`, 170 MB, 18652 rows → 18048 oversampling>0 → 18804 expanded → 588 steps). Same source mix as Stage 2 + **+7,307 more 260608_crypt10k cryptarithm rows** (10,000 crypt10k total vs Stage 2's 2,693). Categories (raw) escalated for cryptarithm: cryptarithm_deduce 9639 (vs Stage 2's 3041), cryptarithm_guess 1161 (vs 452); other 7 categories unchanged. The +4,268 expanded rows vs Stage 2 are concentrated entirely on cryptarithm. 100% GT-True.
- **Method:** SAME recipe as Stage 1 / Stage 2 byte-identical (verbatim copy of Stage 2 trainer + CSV swap only — 3-line diff: docstring, CSV_PATH, _DATA_TAG). r32/α32 LoRA on q/k/v/o/up/down/in/out_proj + lm_head, **no-tie**, **live out_proj**. bf16 + CCE + AdamW(0.9, 0.95) + wd=0, lr 2e-4→0 linear over 588 steps (fresh schedule), bs32 (micro4 ga8), seq8192, seed42. Env `MOE_TIE=0 USE_MEM_EFF=0 FRESH_START=1 INIT_ADAPTER_FROM=/root/autodl-tmp/stage2_adapter_init` (extracted from Stage 2's submission.zip). Stage 2 adapter load verified at startup: 12010 LoRA tensors loaded, 0 missing, lora_B norm 2.14 (Stage 2's converged value vs 1.46 at Stage 2 init — adapter kept training between stages). Step-1 loss 0.044. Wall 9.74 hr, peak 72.1 GB VRAM, final loss 0.0150, **0 grad spikes**.

| Recipe | Method | Kaggle ref | Score | Δ vs Stage 2 (0.86) | Δ vs #16 (0.86) |
|---|---|---|---|---|---|
| final-step (588) | endpoint, crypt10k_FULL data, continual-FT from Stage 2 adapter | 53576276 | **0.84** | **−0.02** | **−0.02** |

- **Why — 3rd-stage continual FT with +7,307 more cryptarithm rows DROPPED LB by 0.02; the curriculum lever is NOT monotonic; Stage 2 was the sweet spot:**
  - **The user's hypothesis tested:** escalate the cryptarithm strengthening further by going from Stage 2's 2,693 crypt10k rows to 10,000 (3.7× more). If LoRA capacity allocation scales with more aug-rows, score should rise to 0.87+. If there's a saturation point, score should plateau at 0.86. If overshoot, score should drop.
  - **Result: −0.02 regression vs Stage 2.** The lever overshoots. Per [[feedback_noise_not_002]], the −0.02 gap is real signal, not inference noise. Stage 3's 0.84 lands back in the 0.83–0.85 aug-class plateau where Stage 1 (#31), Sub #28, #30, #29 all sat.
  - **What this falsifies:** the monotonic-curriculum hypothesis. More aug data → more lift is wrong. The lever has a sweet spot, and Stage 2 was on it. Pushing further trades off other categories.
  - **What this confirms / new findings:**
    1. **Optimal cryptarithm aug-row count is ~2,700 (Stage 2), not 10,000 (Stage 3).** The 4-fold increase in cryptarithm rows over-allocates LoRA capacity to cryptarithm token patterns at the expense of generalization on cipher / bit_manipulation / gravity / unit_conversion / numeral. By Stage 3's category distribution (cryptarithm 51% of expanded data), the model effectively becomes a cryptarithm specialist that loses non-cryptarithm performance.
    2. **The Stage 2 → Stage 3 deltas are diagnostic:** Stage 2's final loss was 0.0270; Stage 3's was 0.0150 (45% lower). Lower training loss correlates with worse LB — classic over-fitting signal. The +7,307 cryptarithm rows let the model memorize cryptarithm patterns deeper but at the cost of generalization.
    3. **The 0.86 ceiling is back.** Stage 2 reached it; Stage 3 falls off. To break 0.87, one needs a different lever entirely — not more of the same. Candidates: rank > 32, different LoRA target modules, RL/DPO from Stage 2's adapter, or a fundamentally different data composition.
  - **What this implies for the curriculum strategy going forward:** Stage 2 (Sub #32 at 0.86) is the current best continual-FT recipe in this branch. Future stages should explore *qualitatively different* levers (rank, modules, RL) rather than escalating the same crypt-row count. Or: stop at Stage 2 and use it as the submission for the 0.86-band.
- **Detail:** [TRAINING_INFO.md](2606102132_sft_crypt10k_FULL_contS2_rtx6000/260611_0744_submission/TRAINING_INFO.md).

---

## Submission #34 — 2026-06-12 — Score 0.86 (Stage 4: second epoch on crypt10k_FULL continued from Stage 3; **+0.02 vs Stage 3, back to the 0.86 wall**)

- **Study folder:** `2606111516_sft_crypt10k_FULL_ep2_contS3_rtx6000`
- **Data:** `260609_Crypt10k_eqHardened_FULL.csv` (md5 `0654f4639b0b51c0c01f5fe0edebf4dd`, 170 MB, 18804 expanded → 588 steps) — **identical CSV and composition as Stage 3 (Sub #33)**. 100% GT-True.
- **Method:** verbatim copy of Stage 3 trainer, 2-line diff (docstring, `_DATA_TAG`); per [[feedback_copy_sft_verbatim]]. r32/α32 LoRA on q/k/v/o/up/down/in/out_proj + lm_head, **no-tie**, **live out_proj**. bf16 + CCE + AdamW(0.9, 0.95) + wd=0, lr 2e-4→0 linear over 588 steps (fresh cycle), bs32 (micro4 ga8), seq8192, seed42. Env `MOE_TIE=0 USE_MEM_EFF=0 FRESH_START=1 INIT_ADAPTER_FROM=/root/autodl-tmp/stage3_adapter_init` (extracted from Stage 3's submission.zip). Stage 3 adapter load verified at startup: 12010 LoRA tensors loaded, 0 missing, lora_B norm 2.88 (vs 2.14 at Stage 3 init — adapter kept training between stages). Step-1 loss 0.025. Wall 9.92 hr, peak 72.2 GB VRAM, final loss 0.0118, 1 transient grad spike at step 577 (grad_norm 535.86 at lr=4.08e-06, loss stayed normal, recovered next step; script spike counter logged 0).

| Recipe | Method | Kaggle ref | Score | Δ vs Stage 3 (0.84) | Δ vs Stage 2 / #16 (0.86) |
|---|---|---|---|---|---|
| final-step (588) | endpoint, crypt10k_FULL data epoch 2, continual-FT from Stage 3 adapter | 53596511 | **0.86** | **+0.02** | **0.00 (tied)** |
| final-step (588), **byte-identical resubmit draw A** | noise-floor test, same zip | 53609145 | **0.86** | — | 0.00 |
| final-step (588), **byte-identical resubmit draw B** | noise-floor test, same zip | 53609157 | **0.86** | — | 0.00 |

- **Noise-floor addendum (2026-06-12):** the Stage 4 0.86 zip was resubmitted byte-identical twice more (user request: test whether inference noise could reach 0.87). Both draws = **0.86**. Combined with the original, the same weights scored **0.86 / 0.86 / 0.86** across three submissions — the true score sits solidly at 0.86, NOT on a 0.865 rounding boundary, so there is no 0.87 recoverable from inference variance. Consistent with the proven ±0.01 noise floor ([[feedback_noise_not_002]]); here the realized variance was 0.00 across three draws.

- **Why — another full epoch on the SAME data recovered +0.02 and re-tied the 0.86 wall; falsifies the "lower train loss → worse LB" overfit reading of Stage 3:**
  - **The user's hypothesis tested:** "I want to train for another epoch, I don't know if it is still underfit." Stage 4 = same data, same recipe, fresh lr 2e-4→0 cycle from Stage 3's adapter.
  - **Result: 0.86, +0.02 vs Stage 3.** Per [[feedback_noise_not_002]] the gap is real. Stage 4's train loss (0.0118) is the LOWEST of all four stages, yet its LB is the highest band — this falsifies the Stage 3 analysis that "lower training loss correlates with worse LB (over-fitting)". Train loss is not predictive of LB position within this recipe family.
  - **What this confirms / new findings:**
    1. **Stage 3's 0.84 was not a data-composition ceiling** — the same crypt10k_FULL data reaches 0.86 with one more epoch. The Stage 3 entry's conclusion that "10,000 crypt rows overshoots the sweet spot" is weakened; the 0.84 draw is at least partly trajectory/endpoint variance (cf. #16=0.86 vs #17=0.84 same recipe, #10 vs #11 endpoint spread).
    2. **The 0.86 wall holds.** Four configurations now tie it (public huikang adapter #13, #16, Stage 2 #32, Stage 4 #34); none exceed it. More epochs recover to the wall but do not break it.
    3. Continual-FT chain S1→S2→S3→S4 ends at 0.86 with lora_B norm grown 1.46→2.14→2.88; no instability (1 benign transient spike in 588 steps).
- **Detail:** [TRAINING_INFO.md](2606111516_sft_crypt10k_FULL_ep2_contS3_rtx6000/260612_0130_submission/TRAINING_INFO.md).

---

## Submission #35 — 2026-06-13 — Score 0.85 (Stage 5: third epoch on crypt10k_FULL continued from Stage 4, **shuffle seed 33**; within-noise of the 0.86 band)

- **Study folder:** `2606120939_sft_crypt10k_FULL_ep3_contS4_seed33_rtx6000`
- **Data:** `260609_Crypt10k_eqHardened_FULL.csv` (md5 `0654f4639b0b51c0c01f5fe0edebf4dd`, 18804 expanded → 588 steps) — **identical CSV/composition as Stage 3 & 4**. 100% GT-True.
- **Method:** verbatim copy of Stage 4 trainer, 3-line diff (docstring, `_DATA_TAG`, `SHUFFLE_SEED 42→33`); per [[feedback_copy_sft_verbatim]]. r32/α32 LoRA, no-tie, live out_proj, bf16+CCE, AdamW(0.9,0.95) wd=0, lr 2e-4→0 linear over 588 steps (fresh cycle), bs32 (micro4 ga8), seq8192, **stratified seed=33**. Env `MOE_TIE=0 USE_MEM_EFF=0 FRESH_START=1 INIT_ADAPTER_FROM=/root/autodl-tmp/stage4_adapter_init` (Stage 4's adapter). Load verified: 12010 LoRA tensors, lora_B norm 3.324 at init (vs 2.88 at Stage 4 init). Step-1 loss 0.011. Wall 9.89 hr, peak ~89 GB VRAM, final loss 0.0137, 0 grad spikes.

| Recipe | Method | Kaggle ref | Score | Δ vs Stage 4 (0.86) | Δ vs Stage 3 (0.84) |
|---|---|---|---|---|---|
| final-step (588) | endpoint, crypt10k_FULL epoch 3, continual-FT from Stage 4 adapter, seed 33 | 53622312 | **0.85** | −0.01 | +0.01 |

- **Why — a 3rd epoch (new batch order, seed 33) landed at 0.85, statistically indistinguishable from Stage 4's 0.86; the continual-FT chain oscillates inside the 0.84–0.86 band and does not exceed 0.86:**
  - **The user's question tested:** does training yet another epoch (with a fresh shuffle order to avoid a third identical seed-42 pass) push past 0.86? Result: 0.85 — within the proven ±0.01 inference-noise floor of Stage 4's 0.86 ([[feedback_noise_not_002]]: 0.01 is noise, not signal), so Stage 5 ≈ Stage 4.
  - **Continual-FT chain to date:** Stage 2 (#32)=0.86, Stage 3 (#33)=0.84, Stage 4 (#34)=0.86, Stage 5 (#35)=0.85. No monotonic trend; the chain samples the 0.84–0.86 band. lora_B norm grew 1.46→2.14→2.88→3.32 across stages with no instability. The seed-33 reshuffle changed nothing material.
  - **What this confirms:** more epochs on the same crypt10k_FULL data — regardless of batch order — recover to/near the 0.86 wall but never break it. Consistent with the LR-schedule literature finding (schedule/epoch knobs are within-noise once converged) and the 34-submission ceiling. Breaking 0.87 needs a different lever (rank, modules, RL, or data targeting the failing categories), not more passes.
- **Detail:** [TRAINING_INFO.md](2606120939_sft_crypt10k_FULL_ep3_contS4_seed33_rtx6000/260612_2034_submission/TRAINING_INFO.md).

---

## Submission #36 — 2026-06-13 — Score 0.79 (#16 reproduction + 5% warmup/cosine decay; **−0.07 collapse from an unclipped early grad explosion, NOT the schedule**)

- **Study folder:** `2606122141_sft_moe_outproj_warm5cos_repro16_rtx6000`
- **Data:** `260514_huikang_golden_stripped.csv` (md5 `92a515f31cd2e89f9cb355c4ed09f927`) — **exact #16 data**, 7849 expanded → 246 steps, seed=42.
- **Method:** byte-identical to #16 (`train_moe_outproj.py`) EXCEPT the LR schedule → 5% linear warmup + cosine decay to 0 over remaining 95% (#16 = linear 2e-4→0, no warmup). r32/α32, no-tie, live out_proj, bf16+CCE, AdamW(0.9,0.95) wd=0, **clip_grad_norm max_norm=1e9 (i.e. NO clipping, same as #16)**, bs32 (micro4 ga8), seq8192, random init. Env `MOE_TIE=0 USE_MEM_EFF=0 FRESH_START=1`. Wall 4.07 hr, final loss 0.003592 (vs #16's 0.0030).

| Recipe | Method | Kaggle ref | Score | Δ vs #16 (0.86) |
|---|---|---|---|---|
| final-step (246) | warmup5%+cosine, no grad clip | 53631833 | **0.79** | **−0.07** |

- **Why — the −0.07 is an unclipped grad explosion at step 67 (high-LR region), not the warmup/cosine schedule:**
  - **Step 67 had grad_norm = 3936.48** (vs ~0.01 typical), with `clip_grad_norm max_norm=1e9` = effectively no clipping, so the raw spike was applied. It occurred at **step 67/246 (27% in), lr ≈ 1.75e-4 (near peak)** → a huge gradient × high LR = a large damaging weight update + poisoned Adam second-moment state. Train loss recovered (re-fit the training CoTs to 0.0036) but **generalization was permanently damaged** → 0.79 LB.
  - **Why this spike hurt when prior spikes didn't:** the same family of non-deterministic bf16/MoE spikes hit Stage 1 (#31, step 221, grad 28086 → scored 0.84) and Stage 4 (#34, step 577, grad 535 → scored 0.86) — but BOTH were at ~98% of training where **lr ≈ 0**, so even unclipped the update was negligible. warm5cos's spike was **early, at high LR** → the update was large and destructive. Timing, not magnitude, decided the damage.
  - **NOT the schedule.** The LR-schedule literature predicts cosine-vs-linear and ±warmup are within-noise once converged (final loss 0.0036 ≈ #16's 0.0030 confirms convergence parity); a −0.07 drop is an order of magnitude beyond any schedule-shape effect. The schedule experiment is **confounded** by the spike — it cannot be read as a schedule result.
  - **Lesson / fix:** the `max_norm=1e9` (no clipping) inherited from #16 is a latent vulnerability — harmless when spikes land at lr≈0, catastrophic when they land early at high lr. Standard gradient clipping at **max_norm=1.0** (Pascanu et al. 2013; universal in LLM training) would have clipped 3936→1.0 with ZERO effect on normal steps (#16's grad norms are all <0.25), neutralizing this failure mode. To get a clean schedule result, relaunch with clip=1.0 (spike is non-deterministic, unlikely to recur; clip catches it if it does).
- **Detail:** [TRAINING_INFO.md](2606122141_sft_moe_outproj_warm5cos_repro16_rtx6000/260613_0234_submission/TRAINING_INFO.md).
