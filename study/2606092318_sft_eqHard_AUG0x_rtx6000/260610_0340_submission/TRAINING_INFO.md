# Training Info — Stage 1 eqHard_AUG0x SFT

**Finished:** 2026-06-10 ~03:40 Chicago (instance: 2026-06-10 16:31 UTC+8)
**Wall time:** 3.78 hr (226.6 min) for 226 steps
**Peak VRAM:** 71.96 GB (RTX PRO 6000 Blackwell, 96 GB)

## ⚠️ Grad spike at step 221 — Stage 2 NOT launched

**Anomaly:** at step 221, `grad_norm=28086.92` (3,000,000× the typical ~0.01 norm). Single-step spike; steps 222–226 recovered to normal grad ~0.01 and finished cleanly with final loss 0.0057.

| step | loss | grad_norm | lr |
|---|---|---|---|
| 220 | 0.0049 | 0.0080 | 6.19e-06 |
| **221** | **0.0078** | **28086.92** | **5.31e-06** |
| 222 | 0.0111 | 0.0091 | 4.42e-06 |
| 223 | 0.0063 | 0.0095 | 3.54e-06 |
| 224 | 0.0018 | 0.0043 | 2.65e-06 |
| 225 | 0.0078 | 0.0070 | 1.77e-06 |
| 226 | 0.0057 | 0.0108 | 8.85e-07 |

Effective update magnitude at step 221 ≈ `grad_norm × lr` = 28087 × 5.31e-6 ≈ **0.149 per affected parameter** (huge vs typical ~1e-4). The model recovered but I cannot guarantee the final adapter is uncorrupted without LB evaluation.

**Decision per user's instruction ("any spike → shutdown, don't relaunch") + memory `feedback_shutdown_on_finish_or_spike`:**
- Stage 1 finalized + submission folder created.
- **Stage 2 NOT launched** (would have init'd from a possibly-corrupted Stage 1 adapter).
- Instance shut down.

**Memory match:** `project_grad_explosion_nondeterministic` describes this exact pattern (rare bf16/MoE grad explosion, norm ~thousands, no clipping). Memory says these "don't reproduce on relaunch (same seed)." If you decide Stage 1 is unusable, a fresh relaunch with the same code/seed should not spike.

**Recommended next steps on wake:**
1. Submit Stage 1 zip to Kaggle to measure actual LB. If ≥ 0.84 → adapter survived the spike, can launch Stage 2 from it. If much lower → relaunch Stage 1 fresh.
2. Decide on Stage 2.

## Method (verbatim copy of #30/cryptaug_1000 trainer)

- LoRA bf16 + CCE + MoE tying (MOE_TIE=0 → no-tie, live out_proj via mixer.training=False)
- LoRA: r=32, α=32, dropout=0
- Targets: `q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, in_proj, out_proj, lm_head`
- Optimizer: AdamW (β=(0.9, 0.95), wd=0)
- LR: 2e-4 → 0 (linear decay)
- Batch: 32 effective (micro=4 × ga=8), seq=8192
- Seed: 42, stratified-interleave by category

## Data

- CSV: `260609_Crypt3k_eqHardened_AUG0x_FULL.csv` (md5 `6c2ad37022f1e6b91aa4ad5703847c07`, 99 MB)
- Rows: 11345 total → 6473 with oversampling>0 → 7229 expanded → 226 steps
- 100% GT-True; AUG0x oversampling heavily dampens augmented rows (cryptarithm_deduce 3041 raw → 280 expanded; eq_numeric_deduce 1989 → 599)
- Categories (9, expanded): bit_manipulation 1754, cipher 1656, unit_conversion 1070, gravity 1055, numeral 730, equation_numeric_deduce 599, cryptarithm_deduce 280, equation_numeric_guess 58, cryptarithm_guess 27
- Sources (raw): pretok_0408_decoded 5545, 260608_crypt10k 2693, 260607_NE_aug 1575, 260601_new_solver 800, 260606_new_solver 732

## Trajectory

- Stress test: VRAM 88.9 GB, 2.7 min, out_proj alive at 0.1430 > 0 ✓
- Soup adapter saves: [45, 136, 181, 192, 203, 215, 226]
  - step 45 loss 0.0052
  - step 136 loss 0.0055
  - step 181 loss 0.0147
  - step 192 loss 0.0079
  - step 203 loss 0.0087
  - step 215 loss 0.0021
  - step 226 loss 0.0057 (final)
- 1 grad spike (step 221), no loss derail

## Outputs

On instance `/root/autodl-tmp/260610_0340_submission/` (download from AutoDL file manager):
- `submission_eqHard_AUG0x_notie_outproj.zip` (3.6 GB) — final-step adapter
- `checkpoint-226_loss0.0057_lr8.85e-07.zip` (9.4 GB) — full resume checkpoint

Local in this folder (SFTP):
- `train_log.txt` — full step-by-step log
- `requirements.txt` — pip freeze snapshot
- `train_eqHard_AUG0x.py` — exact trainer used
- `test_imports.py` — pre-flight import check

## SwanLab

https://swanlab.cn/@jerry4083/260410_Nemotron/runs/grr1lqf5fgunf9q5uw6l0

## Reference

Verbatim copy of `study/2606072355_sft_cryptaug_1000_rtx6000/train_cryptaug_1000.py`, only changed:
- docstring
- line 65: `CSV_PATH = '/root/autodl-tmp/data/260609_Crypt3k_eqHardened_AUG0x_FULL.csv'`
- line 69: `_DATA_TAG = 'eqHard_AUG0x'`

Stage 2 trainer (`study/2606100600_sft_eqHard_FULL_contS1_rtx6000/train_eqHard_FULL_contS1.py`) is pre-written locally but NOT SFTP'd or launched. It adds an `INIT_ADAPTER_FROM` block to load Stage 1's adapter. Ready to use if user decides Stage 1 is healthy.
