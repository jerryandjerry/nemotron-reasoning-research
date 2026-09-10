# Project Context: Nemotron Reasoning Challenge Fine-Tune Comparison
**Date:** 2026-05-28 | **Scope:** Complete project documentation review (260408–260526)

---

## I. The Competition

**Event:** NVIDIA Nemotron Model Reasoning Challenge (Kaggle)  
**Model:** Nemotron-3-Nano-30B-A3B (30B total, 3.5B active; hybrid Mamba2/Attention/128-expert MoE, 52 layers)  
**Task:** Multi-category reasoning with 6 categories: bit_manipulation, text_encryption, gravitational_constant, number_conversion, unit_conversion, equation_transformation (333% coverage across training decomposition: bit_manip, cipher, cryptarithm_deduce/guess, equation_numeric_deduce/guess, gravity, numeral, unit_conversion).

**Metric:** Plain accuracy (no per-category weighting). Answer extraction via `rfind('}')` last brace; comparison: binary→exact, numeric→±1% tolerance (`rel_tol=1e-2, abs_tol=1e-5`), else→case-insensitive string match.

---

## II. Public Leaderboard Score History

| Submission | Date | Score | Mechanism | Status |
|---|---|---|---|---|
| **#13** | 2026-05-22 | 0.85 / **0.86** | huikang's published Tinker adapter (resubmitted verbatim) | Public wall |
| **#16** | 2026-05-23 | **0.86** | Our own trained model (live `out_proj` + no MoE tie) | First *trained* model >0.84 |
| **#6, #7, #11** | 2026-05-15/16/19 | **0.84** | Baseline: bf16 + CCE + MoE tying + dead `out_proj` | Current baseline |
| **#18, #20** | 2026-05-24 | **0.85** | Newsolver cryptarithm CoT swap | Regressed from 0.86 |
| **#22** | 2026-05-26 | **0.84** | Numeric-equation new-solver CoT swap | Regressed from 0.86 |

**Key insight:** The public "0.86 wall" is **not an out-training target** — it is huikang's actual published Tinker weights (`kienngx/…/tinker-adapter/1`), which score 0.85–0.86 due to ±0.01 inference noise. Our own training now **reaches** 0.86 (Sub #16), not replicates it. Going beyond requires better data/recipe on top of the live-`out_proj` baseline.

---

## III. The 0.84→0.86 Breakthrough: `out_proj` LoRA

### Root Cause (Submission #16 report, "260522_moe_expert_rank1_bug_report.md")

Our adapters at 0.84 had **two issues:**

1. **Mamba `out_proj` LoRA never trained (dead, B=0):**
   - Cause: Unsloth's fused Mamba CUDA kernel (`is_fast_path_available=True`) bypasses the LoRA apply.
   - The fused `mamba_split_conv1d_scan_combined` applies `out_proj.weight` directly (base weight only) when `if self.training and cache_params is None`. LoRA stays in graph nowhere → no gradient → `lora_B` stays zero.
   - This limitation is in Unsloth itself, copied from huikang's own notebook (he hits the same ceiling at ~0.85).

2. **Routed experts collapsed to rank-1 (unintended artifact):**
   - Our MoE-tie code (copied from huikang) assumed batched `[128, rank, in]` expert layout (his Unsloth version).
   - Our Unsloth instance exposed separate `[rank, in]` per-expert tensors.
   - Same code: `mean(dim=0, keepdim=True)` collapsed the rank instead of tying across experts.
   - This was **environment-specific**, not a bug we authored. Fixed in Sub #15 (no-tie, rank-32).

### The Fix (Sub #16, verified)

**How to make `out_proj` train (no model-file patching):**
- Keep `is_fast_path_available=True`.
- Set each Mamba mixer's `.training=False` *after* `FastLanguageModel.for_training()`.
- This gates the mixer into its own unfused else-branch: `causal_conv1d_fn + mamba_chunk_scan_combined + self.out_proj(…) as a module call` → LoRA receives gradients.
- No change to VRAM (stays ~88.7 GB) or speed (~1 min/step).

**Result:** Sub #16 trained `out_proj.lora_B` to ‖B‖≈0.70 (matching huikang's Tinker 0.86 magnitude), scored **0.86 on its first submission** — the first *our-own-trained* adapter to clear 0.84.

### What Does NOT Work

- `is_fast_path_available=False` → pure-torch `torch_forward` → ~126 GB SSM tensor at seq 8192 → instant OOM.
- MoE expert-direction tie (Sub #14, rank-32 tied across all 128 experts) → **0.71** (−0.13 from no-tie). Huikang's Tinker may tie, but his 0.86 comes from live `out_proj`, not the tie. The tie hurts *our pipeline*.

---

## IV. Cryptarithm Crisis & Mitigation Strategy

### The Problem

Per SCORE_TRACKER §11 / NOTEPAD_0516:

| huikang category | Train n | Accuracy | Wrong | Issue |
|---|---|---|---|---|
| cryptarithm_deduce | 639 | 1.8% | 54/55 | "Non-functional" CoT template |
| cryptarithm_guess | 161 | 0% | 16/16 | Same template |
| vs equation_numeric_deduce | 540 | 92.3% | 5/65 | **Learnable with proper CoTs** |

**Root cause:** lkevincc's cryptarithm CoTs (median 715 tokens, "default to concatenation" fallback template) do NOT teach the model how to *solve* the puzzle — they teach pattern matching without derivation. The model memorizes when to emit `\boxed{…}` on training data but fails on novel test prompts.

**Impact:** 70/139 wrong answers (50% of all errors) live in cryptarithm. This is the **highest-yield lever** to push past 0.84 (NOTEPAD priority: A).

### Mitigation Attempts

| Submission | Data | Result | Cost | Takeaway |
|---|---|---|---|---|
| **#18** (newsolver base) | 469 kept, 331 capped | 0.85 | 3 days | −0.01 from original |
| **#19** (newsolver gold-only) | 285 correct × 3× | 0.84 (soup 0.85) | 3 days | Gold-only doesn't beat diverse |
| **#20** (newsolver gt2x) | 285 correct × 2× + 187 × 1× | 0.85 | 3 days | Milder oversampling still −0.01 |
| **#21** (cross-data soup) | #16 (0.86) + #18 (0.85) + #20 (0.85) | **0.85** all 3 weightings | 2 days | Blending in 0.85 variants costs the 0.86 |

**Verdict:** All newsolver cryptarithm CoT swaps underperformed the original 735-lkevincc/65-huikang mix. The new solver produces honest (human-verifiable) reasoning traces, but **shorter CoTs** (median 5,944 tokens), **heavily templated** (identical rule dumps), and **intentionally mismatched** (346 GT-mismatch kept). The original lkevincc solution, despite its 1.8% local-eval accuracy, contains implicit patterns the model can fit during training, which don't generalize but do exploit the public test set. **The data swap was not the lever.**

**Path forward:** A truly *working* cryptarithm CoT generator would require reverse-engineering the symbolic-to-digit constraint-propagation solver (huikang's `cryptarithm.py` is 36 KB; a fresh port would take 2–3 days).

---

## V. Numeric Equation CoT Swap (Sub #22)

| Variant | Data | Accuracy | Notes |
|---|---|---|---|
| **#16 (original)** | original eq CoTs, 10.5K chars median | **0.86** | Detailed working shown: `(30+6)*48 = 30*48 + 6*48 = 1440+288 = 1728` |
| **#22 numeq** | new "transformation-rule" solver, 4.6K chars median, 171 low-conf dropped | **0.84** | Terser: `36×48 = 1728; answer = 1728` |
| **#22 allEq** | new solver, all 732 eq CoTs, 114 GT-mismatch kept | **0.84** | Mismatch count irrelevant; style drives the −0.02 |

**Finding:** The new numeric-equation solver produces much shorter, heavily templated CoTs (`Prior knowledge: four kinds of operations…` identical rule dump + `§1/§1.1` scaffolding). The detailed arithmetic working in the original CoTs teaches the model explicit derivation; the new version's brevity doesn't compensate. Both numeq and allEq scored 0.84, confirming the architecture/style mismatch, not data-quality issues, is the problem.

**Pattern:** Every new-solver CoT swap regressed: crypt → 0.85, numeric-eq → 0.84. The **original detailed CoTs are the stronger training signal**. Sub #16 (0.86) remains the ceiling.

---

## VI. MoE Architecture Decisions

### LoRA Scope & Tying

**What works (Sub #6/#7/#11/#15/#16):**
- r32/α32 on attn (q/k/v/o) + Mamba (in_proj + **live** out_proj) + routed experts + shared_experts + lm_head.
- Routed experts: **UNTIED, independent rank-32 per expert** (no shared factor across the 128 experts).
- Final loss converges to ~0.0029–0.0032; validation matches other single-GPU runs within CUDA non-determinism.

**What fails (Sub #14):**
- MoE expert-direction tie (one shared rank-32 factor `lora_A` / `lora_B` across all 128 experts) → **0.71** (−0.13).
- huikang's Tinker *may* use tying (his writeup unclear), but his 0.86 comes from **live `out_proj`**, not the tie. Our pipeline: no-tie + live `out_proj` = 0.86; tie + anything = hurt.

### Inference Noise

**Critical discovery (Sub #11/#13):**
Byte-identical submission files scored ±0.01 apart:
- Sub #13 (0.86 Tinker adapter): ref 52912925 = 0.85, ref 52915203 = **0.86** (resubmitted same zip).
- Sub #11 (endpoint): ref 52813620 = 0.83, ref 52865914 = **0.84** (resubmitted same zip).

**Cause:** vLLM greedy decoding on bf16 MoE with `atomicAdd` non-determinism shifts a few token generations → ±0.01 score wobble. **Any single-submission Δ ≤ 0.01 should be treated as noise, not signal.**

**Consequence:** Claims like "soup-last5 buys +0.01" (Sub #10/#11) are **not supported** once the metric's own ±0.01 noise is known. The 0.84 endpoint and the soup-last5 both landed 0.83/0.84 — indistinguishable.

---

## VII. Soup & Post-Training Tricks (Null Results)

### Checkpoint Selection & Weight Averaging

**Attempted methods (all ≤ ±0.01 lift, within noise):**

| Technique | Result | Reference |
|---|---|---|
| Task-metric eval per checkpoint | Untested (would require held-out eval set) | 260515_checkpoint_selection_report.md |
| WiSE-FT (early × late interpolation) | Sub #11 (wise-soup −6/7pp) shows mixing early ckpts HURTS | — |
| Last-K uniform soup | Sub #10/#11: within ±0.01 noise; indistinguishable from endpoint | — |
| Metrics-weighted soup | Sub #11 SVD-last5: both scored 0.83/0.84 with endpoint | — |
| Cross-trajectory soup (same data) | Sub #12: no gain from #6 + #11 adapters | — |
| Cross-trajectory soup (diverse data) | Sub #21: 0.86 + (0.85 newsolver) + (0.85 gt2x) all → 0.85, even 0.9-weighted | — |

**Verdict:** Soup is not a path past the best single endpoint. Sub #21's finding (cross-data soup of 0.86+0.85+0.85 even at 0.9:0.1:0.05 weighting all scored 0.85) definitively rules out "blending independent trajectories beats the best." The ±0.01 inference noise and the rank-32 SVD re-truncation cost the 0.86.

---

## VIII. Known Bottlenecks & Hard Constraints

### Per-Category Accuracy Bottleneck (Sub #7 / NOTEPAD_0516)

**Val eval (950-sample set) with 0.84-scoring recipe:**

| Category | Train n | Accuracy | Teachable? |
|---|---|---|---|
| gravity, numeral, unit_conversion | 1,477 | 100% | ✓ Free points |
| cipher | 1,576 | 98.7% | ✓ Nearly saturated |
| equation_numeric_deduce | 540 | 92.3% | ✓ Strong signal + detailed CoTs |
| bit_manipulation | 1,354 | 72.5% | ✓ Learnable; ceiling ~85% with better CoTs |
| equation_numeric_guess | 21 | 5.3% | ⚠ Rare (21 samples) + ambiguous-operator task |
| cryptarithm_deduce | 639 | 1.8% | ✗ Template non-functional |
| cryptarithm_guess | 161 | 0% | ✗ Template non-functional |

**The 139 wrong answers distribute as: cryptarithm 70 (50%), bit_manipulation 44 (32%), others 25 (18%).**

### Identified Failure Modes

From 260408_training_lessons.md & 260522_moe_expert_rank1_bug_report.md:

| Failure | Category | Why | Evidence |
|---|---|---|---|
| Free-form CoT collapse | bit_manipulation, equation_transformation | LLM CoTs ramble without deriving rules; the model learns to emit verbose but incorrect reasoning | DeepSeek-R1 (32k tokens) only 44% bit_manip / 28% eq_trans; 0.72 LoRA still vulnerable |
| Short/garbage CoTs dilute signal | all | 504 samples <50 chars CoT, 2,432 <200 chars; model learns vague reasoning is OK | 260408 analysis: training data quality |
| Training data answer mismatches | cryptarithm, equation_numeric_guess | 91 rows where solver_cot's \boxed{} ≠ gold; model trains on inconsistency | 260515: 89.8% match in cryptarithm |
| Mamba `out_proj` dead by default | baseline architecture | Unsloth fused kernel bypasses LoRA; sub #6/#7/#11/#15 all have ‖B‖=0 | 260522 verified byte-level |
| Inference non-determinism | ±0.01 score wobble | bf16 MoE atomicAdd non-determinism; per-submission reroll | Sub #11/#13 same file scored ±0.01 apart |

---

## IX. Training Recipe (Final Validated, Sub #16 / #20 / #22)

### Model & Quantization
- **Base:** nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
- **Quantization:** bf16 (no 4-bit; 4-bit adds dequantization overhead + materializes 16 GB logits, no net VRAM savings)
- **Framework:** Unsloth FastLanguageModel (custom Mamba CUDA kernels, but fused path must gate correctly)

### LoRA Config
```
rank (r)        = 32
alpha (α)       = 32
dropout         = 0
target modules  = q_proj, k_proj, v_proj, o_proj (attention)
                + in_proj, out_proj (Mamba)          ← out_proj MUST train; use mixer.training=False
                + up_proj, down_proj (routed experts) ← UNTIED, rank-32 each
                + up_proj, down_proj (shared expert)
                + lm_head
moe_tying       = OFF (tying hurts: Sub #14 → 0.71)
```

### Training Hyperparameters
```
optimizer       = AdamW (betas 0.9/0.95, weight_decay 0)
learning_rate   = 2e-4 linear decay to 0
batch_size      = 32 (micro=4, grad_accum=8)
max_seq_length  = 8192
epochs          = 1 (244–251 steps total)
loss function   = Cross-entropy (not CCE; standard CE via masked completion tokens)
gradient_clip   = 1.0 (default)
seed            = 42 (stratified sampling per category)
checkpoint_save = {50, 100, 150, 197/198, 200, 209, 221, 234, 246/247} + final force-save
```

### GPU & Timing (RTX PRO 6000, 96 GB)
```
Peak VRAM       = 88.7 GB (model 60 + logits 0 (CCE) + activations ~28)
Time per step   = ~1 min (246 steps ≈ 4.1 hrs)
Data pipeline   = text CSV, stratified batch, raw Kaggle tokenizer
```

### Data Curation (Sub #16 final)
**CSV:** `260514_huikang_golden_stripped.csv` (7,849 expanded rows, 6,906 unique)
- **Composition:** 6,106 huikang non-cryptarithm (bit_manip, cipher, unit_conv, gravity, numeral, equation_numeric_deduce/guess) + 800 cryptarithm (735 lkevincc + 65 huikang)
- **QA fixes applied:**
  1. Kaggle-metric `rfind('}')` extractor (88 lkevincc rows previously truncated by `[^}]*` regex).
  2. `✓` tick stripped from 704 lkevincc rows (2,537 total chars removed).
  3. id=45076dc9 typo'd `\boxed{6}}` → `\boxed{6}` fixed.

These three small fixes alone recovered 0.68 → 0.84 (Sub #5 → Sub #6).

---

## X. Project-Wide Methodology: CoT Honesty

### The "System Log" Principle (from HOW_TO_CRACK_A_NEMOTRON_CATEGORY.md)

The CoT is a **literal log of the solver running**, not a polished explanation:
1. Do a step → print what it did → print the resulting state. Repeat.
2. No refinement, no cover, no fake prints, no leakage.
3. Forward-only: a line uses only values derived *above* it.
4. Deterministic: same logic everywhere; guessing only when nothing is deducible, and deterministically.

**Hard rules (non-negotiable):**
- "Same condition → 100% identical wording" (generator line-templates are the vocabulary).
- One shared routine per concept; no special-case shortcuts.
- Honesty > score; stay true to logic even if points are lost.
- Never hand-edit a CoT; only the generator produces them.
- Human-directed fixes: surface with a link, wait for direction.

### Fresh-Reviewer Audit

Correctness established by **independent multi-agent line-by-line recomputation**, not by score matching.

**Example (260523_Numeric_Equation):**
- 732 puzzles sharded into 10 lists.
- 10 fresh agents (one per shard) read spec + rubric, then for every puzzle recompute every line by hand.
- Goal: 732/732 = 10/10 (achieved).
- Defect log (#1–#10) catalogs every pitfall found and fixed.

**Key caveat:** Authors cannot audit their own work (blind spot risk). Reviewers must have zero shared context with the author.

---

## XI. Data & Category Insights

### Cryptarithm Puzzle Structure (260516_Cryptarithm/cryptarithm_investigation_report.md)

**Puzzle format:** 5-character operands (two 2-symbol numbers + 1 operator), 3–5 examples + 1 query.
- **Symbol↔digit mapping:** Uniform random per puzzle, no global rule (0/725 on 20+ metrics tested — MD5/SHA, ASCII, keyboard, permutation structure, all zero).
- **Operations:** 22 distinct families. Operator symbols have weak predictiveness (+→add 74%, ∗→mul 77%, −→sub 81%; others ~random).
- **Uniqueness:** 85% of 10-symbol puzzles uniquely solvable; <10-symbol puzzles often ambiguous (underdetermined system).

**Why ambiguity matters:** 
- Brute-force solution: 3.6M permutations for 10-symbol puzzles. Model cannot enumerate.
- Ground-truth choice rule: **Not derivable.** Tested 30+ hypotheses (min-digit-sum, lex-first, ASCII-based, etc.) — all ~33% accuracy on ambiguous puzzles.
- lkevincc solver: tiered operation search (TIER0 first), first valid permutation wins. Conditioned on gold answer for 93% of puzzles (not learnable).

**GT-match on our newsolver (260523_Cryptarithm/):** 454/800 (57%) — not a bug rate. Every solution satisfies all examples (0 INVALID); misses are ambiguity + underivability (unseen operators, exotic ops).

### Numeric Equation Puzzle Structure (260523_Numeric_Equation/)

**Simpler than cryptarithm:** Operands are literal numbers (no symbol cipher); only the **operator meaning** and **reading order** (left-to-right vs right-to-left) need to be deduced.

**Current state:** 618/732 GT-match (84.4%); **732/732 = 10/10 on fresh-reviewer audit** (zero defects).

**Design:**
- Reading order global (one per puzzle, not mixed).
- Four families: ~mul, ~add, ~sub, ~concat (each can be exact or off-by ±1/±2; ~sub also allows negation/reversal).
- Exotic fallback: `max(a,b) mod min(a,b)` when no family fits (hardcoded, non-generalizable, same status as boxing a fixed number).
- Unseen-operator guess: binary arithmetic/concat guess (only 2 outcomes, deterministic, learnable).

**CoT format:** System log with §-tree (§1 rightward, §2 leftward), gut-checks, concat detection, family search, greedy lock-and-verify with backtracking.

---

## XII. Strategic Implications & Next Steps

### Current Ceiling (Sub #16: 0.86)

**Achieved by:**
1. No MoE tie (rank-32 per expert; tying hurts to 0.71).
2. **Live Mamba `out_proj` LoRA** (mixer.training=False gates to unfused else-branch; +0.02 over dead out_proj).
3. Original cryptarithm/equation_numeric CoTs (detailed, structured).
4. Proper data QA (rfind extractor, tick-stripping, typo fixes).

### Paths Beyond 0.86

**A. Cryptarithm replacement (highest ROI, +3–5 pp local = +2–3 pp public):**
- Port huikang's `cryptarithm.py` solver or write brute-force search.
- Goal: 30–50% cryptarithm accuracy (up from 1.8%).
- Cost: 2–3 days. Expected push: 0.86 → ~0.88–0.89 public.

**B. Bit-manipulation refinement (+1–2 pp local):**
- Use huikang's bit-serial solver (`reasoners/bit_manipulation.py`) to regenerate CoTs.
- Current: 72.5% accuracy; ceiling ~85%.
- Cost: 1 day.

**C. Data augmentation (unknown, +1–3 pp):**
- huikang uses matching/concat/split augmentations (8,463 sub-skills per row).
- We don't. Adds task diversity.
- Cost: 1 retrain.

**D. Better equation_numeric_guess coverage (+0.5–1 pp):**
- Only 21 training samples; 18/19 val wrong.
- Generate more "guess" CoTs or brute-force operator search.
- Cost: 1 day.

**Paths that don't work (proven null):**
- Soup (all variants ≤ ±0.01 noise; cross-data soup cost the 0.86).
- New-solver CoT swap (all underperform original detailed CoTs).
- DDP parallelization (1.49× speedup at −0.02 cost; single-GPU is cleaner).
- MoE tying (0.84 → 0.71, direct −0.13).

---

## XIII. Code & Configuration Artifacts

### Key Files by Purpose

| Purpose | File |
|---|---|
| **Training routine** | `02_train/TRAINING_ROUTINE.md` (step-by-step, stress test, logging, checkpoint save rules, submission process) |
| **MoE fix details** | `02_train/260522_moe_expert_rank1_bug_report.md` (root cause, byte-level verification, recipe for live out_proj) |
| **Checkpoint selection** | `02_train/260515_checkpoint_selection_report.md` (7 post-training techniques; most ≤ noise floor) |
| **A/B test results** | `02_train/260414_ab_test_report.md` (loss weighting, data scaling, oversampling all null; baseline stuck at 0.70) |
| **4-bit QLoRA investigation** | `02_train/260514_4bit_investigation_report.md` (4-bit adds 16 GB logits, dequantization overhead; bf16 + CCE strictly better) |
| **Parallel training schemes** | `02_train/260409_parallel_training_report.md` (DDP OOM; Pipeline Parallel viable; FSDP + QLoRA not viable on this model) |
| **Answer extraction metric** | `01_data/ANSWER_EXTRACTION.md` (rfind boxed logic, verify 3-branch, extraction fallbacks) |
| **Kaggle metric report** | `01_data/260523_kaggle_metric_extractor_report.md` (live metric pulled 2026-05-23; our training matches rfind version) |
| **How to crack a category** | `01_data/HOW_TO_CRACK_A_NEMOTRON_CATEGORY.md` (8-phase method, system-log principle, fresh-reviewer audit) |
| **Numeric equation build** | `01_data/260523_Numeric_Equation/README.md` (index), `LESSONS.md` (12 dead ends + 6 leakage traps documented), `cot_template.md` (spec), `agent_review_instruction.md` (rubric + defect log #1–#10) |
| **Cryptarithm build** | `01_data/260523_Cryptarithm/SUMMARY.md` (honest solver, 57% GT-match on 800 puzzles, 10-agent audit clean) |
| **Score tracker** | `study/SCORE_TRACKER.md` (22 submissions, 2026-05-13 to 2026-05-26, with Kaggle refs, breakdowns, mechanisms) |
| **Strategic notebook** | `study/NOTEPAD_0516_path_to_higher_score.md` (per-category accuracy bottleneck analysis, priority-ranked levers A–F) |

### Training Scripts (Latest Working)

**Location:** `study/2605222039_sft_moe_outproj_rtx6000/train_moe_outproj.py` (0.86 endpoint recipe)

**Key env vars:**
```bash
MOE_TIE=0                 # Don't tie experts (tying → 0.71)
USE_MEM_EFF=0             # mixer.training=False (not is_fast_path_available=False)
FRESH_START=1             # Don't resume from checkpoint
```

---

## XIV. Summary: The Full Story

### The Gap (What happened 2026-05-13 to 2026-05-26)

1. **Started at 0.84** (Subs #1/#4/#6/#7): Reproduced huikang's recipe but always scored below his published 0.86. Assumed we were out-training the wall.

2. **Diagnosed the gap (Sub #16, 2026-05-23):**
   - Found: Mamba `out_proj` never trained (dead, B=0) — Unsloth fused kernel bypasses it.
   - Found: Our MoE-tie code unintentionally collapsed experts to rank-1 (environment mismatch with huikang's Unsloth version).
   - Fixed: `out_proj` LoRA by setting mixer.training=False; no-tie + live `out_proj` → **0.86**.
   - **First time our trained model cleared 0.84.**

3. **Proved the wall is not out-trainable (Sub #13):** Submitted huikang's published adapter verbatim, scored 0.85 then 0.86 (same file). The "0.86 wall" is his actual weights, not a training ceiling.

4. **Tested cryptarithm & numeric-equation swaps (Subs #18–#22):** All underperformed 0.86. The original detailed CoTs are the stronger signal; new "honest" but shorter/templated CoTs don't compensate.

5. **Tested soup, DDP, cross-data blending (Subs #10–#12, #17, #21):** All ≤ ±0.01 noise or regressed. The best single endpoint (0.86) is the ceiling.

### What We Actually Achieved

- **Reached the public-LB wall (0.86)** with our own trained model, not resubmission.
- **Identified the missing piece** (live `out_proj`): a 1-line fix (`mixer.training=False`) that bridged 0.84 → 0.86.
- **Proved what doesn't work:** soup, DDP (high overhead), new-solver swaps, MoE tying.
- **Quantified the bottleneck:** 70/139 wrong answers (50%) live in cryptarithm; 44 (32%) in bit_manipulation. Fixing cryptarithm with a working solver is the path past 0.86.

### The Real Leaderboard Situation (Not Public Knowledge)

- **Public LB (visible to all):** ~700 teams re-submitting huikang's Tinker adapter (Sub #13's `kienngx/…/tinker-adapter/1`). Best-of-N noisy draws → the "0.86 wall" is **not a training target, it's a shared artifact**.
- **Our standing:** Also at 0.86 now (Sub #16), but from our own training + the `out_proj` fix.
- **Private LB:** Unknown, but likely crowded at 0.86 (same artifact re-submitted many times). Final placement decided by who exceeds it.

---

## XV. References & Quotes

### Key Quotes (Verbatim)

From 260522_moe_expert_rank1_bug_report.md, §7 (the breakthrough):
> "**Headline:** Sub #16's endpoint = **0.86** — the first time any of our *trained* adapters cleared 0.84 (every prior run #1/#4/#6/#7/#11/#15 topped at 0.84 across many endpoint/soup/resubmit draws)."

From NOTEPAD_0516:
> "**The lever to push past 0.84 is a working cryptarithm CoT pipeline.** Not regex fixes, not soup, not prompting tricks. The model needs CoTs that solve the puzzle."

From 260523_Numeric_Equation/LESSONS.md, Philosophy (§E):
> "**It is a system log — can you print a fake system log?** The CoT is not a beautiful customized explanation; it is a log of the solver's running state."

From SCORE_TRACKER.md, Sub #21 (cross-data soup):
> "Reinforces #12 (cross-trajectory soup of same-data runs = no gain): cross-data soup of these variants does not exceed the best single endpoint. The original-data #16 endpoint (0.86) remains the ceiling."

---

**END OF REPORT**
