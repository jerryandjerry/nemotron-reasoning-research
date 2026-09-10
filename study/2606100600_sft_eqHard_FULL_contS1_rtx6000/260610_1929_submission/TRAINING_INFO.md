# Training Info — Stage 2 eqHard_FULL_contS1 SFT

**Finished:** 2026-06-10 ~19:29 Chicago (instance: 2026-06-11 08:29 UTC+8)
**Wall time:** 6.19 hr (371.2 min) for 360 steps
**Peak VRAM:** 71.9 GB (RTX PRO 6000 Blackwell, 96 GB)

## Stage 2: continual SFT from Stage 1 adapter

- **Init weights:** Stage 1 adapter loaded via `INIT_ADAPTER_FROM=/root/autodl-tmp/stage1_adapter_init` (extracted from Stage 1's submission.zip, ~3.6 GB)
- **Load verification at startup:** `12010 LoRA tensors loaded, 0 LoRA missing, 0 unexpected, sanity lora_B norm = 1.46` (trained-adapter, not random)
- **Optimizer + scheduler:** reset (fresh 1-epoch cycle, not continued from Stage 1)
- **Sanity at step 1:** loss **0.043** (vs Stage 1's step-1 loss 0.57 from random init — confirms adapter loaded)
- **out_proj alive at stress step 1:** 0.0159 > 0 ✓

## Method (verbatim copy of Stage 1 trainer + INIT_ADAPTER_FROM block)

- LoRA bf16 + CCE + MoE tying (MOE_TIE=0 → no-tie, live out_proj)
- LoRA: r=32, α=32, dropout=0
- Targets: `q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, in_proj, out_proj, lm_head`
- Optimizer: AdamW (β=(0.9, 0.95), wd=0)
- LR: 2e-4 → 0 (linear decay, fresh cycle over 360 steps)
- Batch: 32 effective (micro=4 × ga=8), seq=8192
- Seed: 42, stratified-interleave by category

## Data

- CSV: `260609_Crypt3k_eqHardened_FULL.csv` (md5 `46b787fd33eea30d3766d72ea45a5e89`, 99 MB)
- Rows: 11345 total → 10741 with oversampling>0 → 11497 expanded → 360 steps
- 100% GT-True
- **vs Stage 1's AUG0x:** same source mix; FULL adds **4268** more rows by oversampling the cryptarithm + numeric_equation aug rows
- Categories (9, expanded): cryptarithm_deduce **2682** (+2402 vs Stage 1), equation_numeric_deduce **1992** (+1393), bit_manipulation 1754 (same), cipher 1656 (same), unit_conversion 1070 (same), gravity 1055 (same), numeral 730 (same), cryptarithm_guess 318 (+291), equation_numeric_guess 240 (+182)
- Sources (raw): pretok_0408_decoded 5545, 260608_crypt10k 2693, 260607_NE_aug 1575, 260601_new_solver 800, 260606_new_solver 732

## Trajectory

- 0 grad spikes across all 360 steps (grad_norm peak 0.06)
- Soup adapter saves: [72, 216, 288, 306, 324, 342, 360]
  - step 72   loss 0.0245
  - step 216  loss 0.0149
  - step 288  loss 0.0176
  - step 306  loss 0.0166
  - step 324  loss 0.0125
  - step 342  loss 0.0170
  - step 360  loss 0.0270 (final)

## Outputs

On instance `/root/autodl-tmp/260610_1929_submission/` (user downloads from AutoDL file manager):
- `submission_eqHard_FULL_contS1_notie_outproj.zip` (3.6 GB) — final-step adapter for Kaggle
- `checkpoint-360_loss0.0270_lr5.56e-07.zip` (9.7 GB) — full resume checkpoint

Local in this folder (SFTP):
- `train_log.txt` — full step-by-step log (1.2 MB)
- `requirements.txt` — pip freeze snapshot
- `train_eqHard_FULL_contS1.py` — exact trainer used (includes INIT_ADAPTER_FROM block)
- `test_imports.py` — pre-flight import check

## SwanLab

https://swanlab.cn/@jerry4083/260410_Nemotron/runs/z7y8usym179ctnu0q8617

## Reference

Trainer = verbatim copy of Stage 1 (`train_eqHard_AUG0x.py`) + INIT_ADAPTER_FROM block.
- Reads `INIT_ADAPTER_FROM` env; if set, after `get_peft_model()` + lm_head LoRA + fp32 cast, loads adapter_model.safetensors from that path
- Reverses the §3 Kaggle-compat rename (`backbone.lm_head.*` → `lm_head.*`) so Stage 1's saved keys match Stage 2's freshly-created LoRA layers
- Casts loaded LoRA params to fp32 to match the fp32 cast above
- Asserts `len(lora_missing) == 0` (catches load failures)
- Optimizer + scheduler stay fresh (no resume from Stage 1's optimizer state)

The 2-stage curriculum (Stage 1 = build base ability on AUG0x; Stage 2 = strengthen cryptarithm + numeric_equation on FULL) is now complete.
