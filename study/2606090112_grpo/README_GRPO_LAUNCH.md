# GRPO run — crypt + numeric-eq distillation RL

**Folder:** `study/2606090112_grpo/`
**Prepared:** 2026-06-12 (CDT)

## Contents

| File | What |
|---|---|
| `train_grpo_v1.py` | Training script (SFT stack + GRPOTrainer + GSPO/DAPO/β=0.02 KL) |
| `grpo_rewards.py` | 7-reward per-step distillation stack (smoke test: near-gold +9.9, garbage −3.2) |
| `grpo_dataset_cryptEq.csv` | 928 rows (5.8 MB) — built by `_build_dataset.py` |
| `_build_dataset.py` | Reproducible dataset build from the 260609 FULL CSV |
| `cryptarithm_solver/` | Gold CoT generator + 10 test cases (verified 10/10 on this machine) |
| `NumericEq_solver/` | Gold CoT generator + 8 test cases (verified 8/8 on this machine) |

## Inputs

- **Seed LoRA** (user uploads to instance — 3.58 GB):
  `study/2606111516_sft_crypt10k_FULL_ep2_contS3_rtx6000/260612_0130_submission/submission_crypt10k_FULL_ep2_contS3_notie_outproj.zip`
  → unzip on instance to `/root/autodl-tmp/sft_seed_crypt10k_FULL_ep2_contS3/`
  (script reverses the submission-time `backbone.lm_head` key rename at load)
- **Data**: `grpo_dataset_cryptEq.csv` (5.8 MB) → gzip (~1.5 MB) + SFTP to `/root/autodl-tmp/data/`, gunzip there
- **Base model**: already on instance (`/root/autodl-tmp/Nemotron-3-Nano-30B-A3B`)

## Dataset facts

- Source: `260609_Crypt3k_eqHardened_AUG0x_FULL.csv` (11,345 rows)
- Kept: 4 categories × **oversampling ≥ 1** (per user: ov=0 = too-long AUG puzzles, excluded) × GT-match=True = **928 rows**
  - equation_numeric_deduce 563 / cryptarithm_deduce 280 / equation_numeric_guess 58 / cryptarithm_guess 27
- Gold CoT lengths: med 1,745 / p95 5,850 / max 7,648 tokens
- `cot_token_len` carried per row; the script drops puzzles whose gold CoT exceeds
  `MAX_COMPLETION_LEN − 256`. Fit table: budget 3584 → 79.3%, 4096 → 84.4%, 5120 → 91.4%, 7680 → 100%

## Decisions encoded in the script

| Knob | Value | Why |
|---|---|---|
| β (KL) | 0.02 | β=0 catastrophically forgets (arxiv 2509.07430); ref policy via PEFT disable_adapter |
| loss_type | dapo | length-bias fix |
| importance_sampling_level | sequence | GSPO |
| epsilon_high | 0.28 | DAPO clip-higher |
| lr | 5e-6, cosine, warmup 0.1 | Unsloth GRPO consensus |
| num_generations | 4 | × grad_accum 4 = 16 rollouts/step |
| MAX_PROMPT_LEN / MAX_COMPLETION_LEN | 512 / 3584 | prompts ≤60 tok; completion budget pending stress test |
| USE_MEM_EFF | 0 | out_proj LoRA trains (the 0.84→0.86 lever) |
| MOE_TIE | 0 | no-tie, matches seed |

**Note on oversampling weighting:** rewards multiply by `oversampling`, but GRPO group-normalized
advantage is scale-invariant within a group (all rollouts of one prompt share the weight), so the
36 rows at 2× have no practical effect. Harmless; the real prompt weighting lever in GRPO is row
duplication, which we're not using.

**Known bug fixed during prep:** `grpo_rewards.py` originally routed on `cat == 'cryptarithm'`;
dataset categories are `cryptarithm_deduce` etc. → D2/Q3/S6 would never fire. Fixed to
`startswith` (both copies: here + `02_train/grpo_rewards.py`).

## Launch checklist (per TRAINING_ROUTINE.md)

1. ☐ User: power on instance, share SSH port
2. ☐ User: upload seed zip (3.58 GB) to `/root/autodl-tmp/`
3. ☐ Me: unzip seed → `/root/autodl-tmp/sft_seed_crypt10k_FULL_ep2_contS3/`
4. ☐ Me: upload `train_grpo_v1.py` + `grpo_rewards.py` + gzipped dataset (~1.5 MB) via SFTP
5. ☐ Me: `test_imports.py` — verify unsloth/trl 0.22.2/vllm/peft + GRPOConfig knobs exist
6. ☐ Me: `STRESS_TEST=1 python train_grpo_v1.py` — VRAM check on 32 longest prompts
7. ☐ User: approve final config (routine §2)
8. ☐ Me: launch real run + 5-min polls
