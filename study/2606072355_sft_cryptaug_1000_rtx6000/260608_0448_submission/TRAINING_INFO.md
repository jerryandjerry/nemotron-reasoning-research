# Training Info — cryptaug_1000 SFT

**Finished:** 2026-06-08 04:48 Chicago (instance: 2026-06-08 17:45)
**Wall time:** 4.43 hr (265.7 min) for 272 steps
**Peak VRAM:** 72.1 GB (RTX PRO 6000 Blackwell, 96 GB)

## Method (verbatim copy of #16/bitman_failcorr trainer)

- LoRA bf16 + CCE + MoE tying (MOE_TIE=0 → no-tie, live out_proj via mixer.training=False)
- LoRA: r=32, α=32, dropout=0
- Targets: `q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, in_proj, out_proj, lm_head`
- Optimizer: AdamW
- LR: 2e-4 → 0 (linear decay)
- Batch: 32 effective (micro=4 × ga=8), seq=8192
- Seed: 42, stratified-interleave by category

## Data

- CSV: `260607_Cryptarithm_1000_FULL.csv` (md5 `928065b844f19297ce144541e40620ad`, 75.75 MB)
- Rows: 8535 total → 7931 oversampling>0 → 8687 expanded → 272 steps
- 100% GT-True
- 0 rows over 7680 token cap
- Categories (9): bit_manipulation 20.2%, cipher 19.1%, eq_d 14.6%, eq_g 1.7%, crypt_d 10.2%, crypt_g 1.4%, gravity 12.1%, numeral 8.4%, unit_conv 12.3%
- Sources: pretok_0408_decoded 5545, 260607_NE_aug 765, 260607_crypt_aug 693, 260606_new_solver 621, 260601_new_solver 307

## Training trajectory

- 0 grad spikes across all 272 steps (grad_norm peak 0.022)
- §1.6 out_proj check at stress step 1: 0.1288 > 0 ✓
- Soup adapter saves: [54, 163, 218, 231, 245, 258, 272]
  - step 54   loss 0.0242
  - step 163  loss 0.0194
  - step 218  loss 0.0080
  - step 231  loss 0.0115
  - step 245  loss 0.0136
  - step 258  loss 0.0161
  - step 272  loss 0.0234 (final)
- Stress test: VRAM 72.2 GB, 2.4 min, out_proj alive

## Outputs

In `260608_0448_submission/` on instance (user downloads from AutoDL file manager):
- `submission_cryptaug_1000_notie_outproj.zip` (3.6 GB) — final-step adapter for Kaggle
- `checkpoint-272_loss0.0234_lr7.35e-07.zip` (9.7 GB) — full resume checkpoint

Local in this folder (SFTP):
- `train_log.txt` — full step-by-step log
- `requirements.txt` — pip freeze snapshot
- `train_cryptaug_1000.py` — exact trainer used
- `test_imports.py` — pre-flight import check

## SwanLab

https://swanlab.cn/@jerry4083/260410_Nemotron/runs/yrulo5vspeo5nhft1qizr

## Reference

Verbatim copy of `study/2606081955_sft_bitman_failcorr_rtx6000/train_bitman_failcorr.py`, only changed:
- docstring
- line 65: `CSV_PATH = '/root/autodl-tmp/data/260607_Cryptarithm_1000_FULL.csv'`
- line 69: `_DATA_TAG = 'cryptaug_1000'`

No soup BUILD this run (user didn't request). Soup adapters (7) are saved for potential post-hoc build.
