# TRAINING_INFO — Stage 4: crypt10k_FULL second epoch (continued from Stage 3)

- **Study folder:** `2606111516_sft_crypt10k_FULL_ep2_contS3_rtx6000`
- **Submission folder:** `260612_0130_submission` (training finished 2026-06-12 01:30 Chicago)
- **Run:** swanlab `260410_Nemotron/runs/pzpwp75o861vtdtojp57w`

## What was trained

"Train for another epoch" — continual SFT on the **same** crypt10k_FULL data as Stage 3 (Sub #33, 0.84), initialized from Stage 3's final step-588 adapter. Purpose: test whether Stage 3 was still underfit (user question). LB context going in: Stage 2 (Sub #32) = 0.86 at train loss 0.027; Stage 3 (Sub #33) = 0.84 at train loss 0.015 — lower train loss correlated with worse LB, an overfit signal; this run probes the other direction.

## Config

- Trainer: verbatim copy of Stage 3 `train_crypt10k_FULL_contS2.py`; only docstring + `_DATA_TAG='crypt10k_FULL_ep2_contS3'` changed (CSV path identical)
- Init: `INIT_ADAPTER_FROM=/root/autodl-tmp/stage3_adapter_init` (Stage 3 step-588 adapter; 12010 LoRA tensors loaded, 0 missing, lora_B norm 2.88 at init; step-1 loss 0.025 vs 0.57 random-init — proves trained init)
- Data: `/root/autodl-tmp/data/260609_Crypt10k_eqHardened_FULL.csv` (md5 0654f4639b0b51c0c01f5fe0edebf4dd), 18804 expanded examples → 588 steps, stratified-interleave seed=42
- LoRA r=32 α=32 dropout=0; targets q/k/v/o_proj, up/down_proj, in_proj, out_proj, lm_head; MOE_TIE=0 (no-tie), live out_proj (mixer.training=False), CCE
- LR 2e-4 → 0 linear over 588 steps (fresh cycle), bf16, batch 32 (micro=4 × ga=8), seq 8192
- GPU: RTX PRO 6000 Blackwell 96 GB (port 21578)

## Result

- **Wall time:** 9.92 hr (595.0 min). Script-reported peak VRAM 72.2 GB (per-step `smi_after` peaked ~89.1 GB during training).
- **Final train loss:** 0.0118 at step 588 (vs Stage 3's 0.015, Stage 2's 0.027)
- **Loss path:** 0.025 (step 1) → ~0.020 (mid) → 0.0118 (final); smooth decline
- **Grad spikes:** 1 transient at step 577 — grad_norm 535.86 (typical 0.017), loss stayed normal (0.0162), fully recovered next step (grad 0.0154, loss 0.0135). Occurred at lr=4.08e-06 (near end of schedule) so weight impact is minimal. Script's GRAD SPIKE counter logged 0. Same family as Stage 1's step-221 spike (28086, recovered, run still scored 0.84).
- **Soup adapters saved (7):** steps [118, 353, 470, 500, 529, 559, 588] = 20/60/80/85/90/95/100%
  - adapter-118 loss 0.0205 · adapter-353 loss 0.0171 · adapter-470 loss 0.0128 · adapter-500 loss 0.0152 · adapter-529 loss 0.0155 · adapter-559 loss 0.0140 · adapter-588 loss 0.0118
  - Saved on instance disk in `adapter_output_crypt10k_FULL_ep2_contS3_notie_outproj/` (not zipped; available while instance exists)

## Artifacts

Instance `/root/autodl-tmp/260612_0130_submission/` (user downloads via AutoDL file manager):
- `submission_crypt10k_FULL_ep2_contS3_notie_outproj.zip` (3.84 GB) — final step-588 adapter, Kaggle-ready
- `checkpoint-588_loss0.0118_lr3.40e-07.zip` (10.48 GB) — full resume checkpoint

Local (this folder): `train_crypt10k_FULL_ep2_contS3.py`, `train_log.txt`, `requirements.txt`, this file.

## Hypothesis going into submission

Sub #32→#33 trend (more steps on same data, lower train loss, −0.02 LB) predicts the overfit direction; if correct, this run's final adapter (train loss 0.0118, even lower) lands ≤0.84. If Stage 3's 0.84 was instead a bad-seed trajectory draw (cf. Sub #10 vs #11 endpoint variance), the fresh LR cycle could recover toward 0.86. The 0.02 gap is real signal either way (noise is ±0.01 max, proven by byte-identical resubmits #11/#13). The 7 soup saves preserve the post-hoc option to test earlier checkpoints (e.g. adapter-118 at 20%) if the endpoint disappoints.
