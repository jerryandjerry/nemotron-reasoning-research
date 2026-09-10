# TRAINING_INFO — Stage 5: crypt10k_FULL third epoch (continued from Stage 4, shuffle seed 33)

- **Study folder:** `2606120939_sft_crypt10k_FULL_ep3_contS4_seed33_rtx6000`
- **Submission folder:** `260612_2034_submission` (training finished 2026-06-12 20:34 Chicago)
- **Run:** swanlab `260410_Nemotron/runs/4yn073akk7q74s9ko3njy`

## What was trained

"Another epoch" #2 — continual SFT on the **same** crypt10k_FULL data as Stage 3/4, initialized from Stage 4's final step-588 adapter (Sub #34 = 0.86). User-requested change: **shuffle seed 42 → 33**, so epoch 3 sees a fresh stratified batch order instead of repeating epochs 1–2's identical seed-42 order. Tests whether more epochs keep helping (S3→S4 was +0.02) with a new data ordering.

## Config

- Trainer: verbatim copy of Stage 4 `train_crypt10k_FULL_ep2_contS3.py`; only 3 changes — docstring, `_DATA_TAG='crypt10k_FULL_ep3_contS4'`, `SHUFFLE_SEED=33`. CSV path identical.
- Init: `INIT_ADAPTER_FROM=/root/autodl-tmp/stage4_adapter_init` (Stage 4 step-588 adapter, extracted from Stage 4 submission.zip; 12010 LoRA tensors loaded, 0 missing, **lora_B norm 3.324** at init — chain S1→S2→S3→S4→S5 grew 1.46→2.14→2.88→3.32; step-1 loss 0.011 vs Stage 4's 0.025, consistent with the more-trained init)
- Data: `/root/autodl-tmp/data/260609_Crypt10k_eqHardened_FULL.csv` (md5 0654f4639b0b51c0c01f5fe0edebf4dd), 18652 unique → 18804 expanded → 588 steps, **stratified-interleave seed=33**
- LoRA r=32 α=32 dropout=0; targets q/k/v/o_proj, up/down_proj, in_proj, out_proj, lm_head; MOE_TIE=0 (no-tie), live out_proj (mixer.training=False), CCE
- LR 2e-4 → 0 linear over 588 steps (fresh cycle), bf16, batch 32 (micro=4 × ga=8), seq 8192
- GPU: RTX PRO 6000 Blackwell 96 GB (port 21578)

## Result

- **Wall time:** 9.89 hr (593.2 min). Per-step `smi_after` peaked ~89.1 GB.
- **Final train loss:** 0.0137 at step 588 (Stage 4: 0.0118, Stage 3: 0.015, Stage 2: 0.027)
- **Grad spikes:** 0 across 588 steps (grad_norm stayed ~0.012–0.025; cleanest run of the chain — Stage 4 had 1 transient at step 577)
- **Soup adapters saved (7):** steps [118, 353, 470, 500, 529, 559, 588] = 20/60/80/85/90/95/100%
  - adapter-118 loss 0.0177 · adapter-353 loss 0.0102 · adapter-470 loss 0.0164 · adapter-500 loss 0.0123 · adapter-529 loss 0.0124 · adapter-559 loss 0.0101 · adapter-588 loss 0.0137
  - On instance disk in `adapter_output_crypt10k_FULL_ep3_contS4_notie_outproj/` (not zipped; available while instance exists)

## Artifacts

Instance `/root/autodl-tmp/260612_2034_submission/` (user downloads via AutoDL file manager):
- `submission_crypt10k_FULL_ep3_contS4_notie_outproj.zip` (3.84 GB) — final step-588 adapter, Kaggle-ready
- `checkpoint-588_loss0.0137_lr3.40e-07.zip` (10.41 GB) — full resume checkpoint

Local (this folder): `train_crypt10k_FULL_ep3_contS4.py`, `train_log.txt`, `requirements.txt`, this file.

## Context / hypothesis

Stage 4 (#34) recovered to 0.86 from Stage 3's 0.84. Stage 5 = a third epoch with a new batch order, testing whether the curriculum keeps climbing or has plateaued at the 0.86 wall (LB max across 34 submissions; leaderboard top is 0.90, so 0.86 is our recipe's ceiling, not the eval's). Per the LR-schedule literature review (2026-06-12), our per-stage re-warm-to-2e-4 sawtooth has no proven single-model advantage over one decay for same-data continuation, and LR *magnitude* — not schedule shape — is the dominant LoRA knob; if Stage 5 again lands ≤0.86, peak-LR tuning and the failing-category data gap are the higher-value levers than more epochs. Awaiting LB score.
