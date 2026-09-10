# Training Info — Stage 3 crypt10k_FULL_contS2 SFT

**Finished:** 2026-06-11 07:44 Chicago (instance: 2026-06-11 20:35 UTC+8)
**Wall time:** 9.74 hr (584.2 min) for 588 steps
**Peak VRAM:** 72.1 GB (RTX PRO 6000 Blackwell, 96 GB)

## Stage 3: continual SFT from Stage 2 adapter

- **Init weights:** Stage 2 adapter loaded via `INIT_ADAPTER_FROM=/root/autodl-tmp/stage2_adapter_init`
- **Load verification at startup:** `12010 LoRA tensors loaded, 0 LoRA missing, 0 unexpected, sanity lora_B norm = 2.14`
- **Optimizer + scheduler:** reset (fresh 1-epoch cycle)
- **Sanity at step 1:** loss **0.045** (Stage 2's step 1 was 0.043, Stage 1's 0.57 from random — Stage 2 adapter loaded successfully)
- **out_proj alive at stress step 1:** 0.0108 > 0 ✓

## Method (verbatim copy of Stage 2 trainer + CSV swap only)

- LoRA bf16 + CCE + MoE tying (MOE_TIE=0 → no-tie, live out_proj)
- LoRA: r=32, α=32, dropout=0
- Targets: `q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, in_proj, out_proj, lm_head`
- Optimizer: AdamW (β=(0.9, 0.95), wd=0)
- LR: 2e-4 → 0 (linear decay over 588 steps)
- Batch: 32 effective (micro=4 × ga=8), seq=8192
- Seed: 42, stratified-interleave by category

## Data

- CSV: `260609_Crypt10k_eqHardened_FULL.csv` (md5 `0654f4639b0b51c0c01f5fe0edebf4dd`, 170 MB)
- Rows: 18652 total → 18048 with oversampling>0 → 18804 expanded → 588 steps
- 100% GT-True
- **vs Stage 2:** +7,307 more crypt10k cryptarithm rows (10,000 crypt10k total vs Stage 2's 2,693)
- Categories (9, raw): cryptarithm_deduce 9639, equation_numeric_deduce 1989, cipher 1576, bit_manipulation 1354, cryptarithm_guess 1161, unit_conversion 990, gravity 975, numeral 650, equation_numeric_guess 318
- Sources (raw): 260608_crypt10k 10000, pretok_0408_decoded 5545, 260607_NE_aug 1575, 260601_new_solver 800, 260606_new_solver 732

## Trajectory

- 0 grad spikes across all 588 steps (grad_norm peak ~0.06)
- Soup adapter saves: [118, 353, 470, 500, 529, 559, 588]
  - step 118  loss 0.0294
  - step 353  loss 0.0220
  - step 470  loss 0.0169
  - step 500  loss 0.0190
  - step 529  loss 0.0192
  - step 559  loss 0.0172
  - step 588  loss 0.0150 (final)

## Outputs

On instance `/root/autodl-tmp/260611_0744_submission/` (user downloads from AutoDL file manager):
- `submission_crypt10k_FULL_contS2_notie_outproj.zip` (3.6 GB)
- `checkpoint-588_loss0.0150_lr3.40e-07.zip` (9.7 GB)

Local in this folder:
- `train_log.txt` — full step-by-step log (2 MB)
- `requirements.txt` — pip freeze snapshot
- `train_crypt10k_FULL_contS2.py` — exact trainer used
- `test_imports.py` — pre-flight import check

## SwanLab

Project: https://swanlab.cn/@jerry4083/260410_Nemotron (network errors during training; offline-uploaded later)

## Reference

Trainer = verbatim copy of Stage 2 (`train_eqHard_FULL_contS1.py`) + CSV swap (3-line diff: docstring, CSV_PATH, _DATA_TAG). INIT_ADAPTER_FROM block identical to Stage 2's.

3-stage curriculum complete: Stage 1 (AUG0x, build base) → Stage 2 (FULL, strengthen aug cats) → Stage 3 (Crypt10k FULL, escalate cryptarithm strengthening).
