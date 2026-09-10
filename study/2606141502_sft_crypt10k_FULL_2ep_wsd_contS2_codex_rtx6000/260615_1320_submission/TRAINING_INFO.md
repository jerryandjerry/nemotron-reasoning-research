# Crypt10k FULL 2ep WSD contS2

- **Run folder:** `2606141502_sft_crypt10k_FULL_2ep_wsd_contS2_codex_rtx6000`
- **AutoDL instance:** `ssh -p 47429 root@connect.bjb1.seetacloud.com`
- **Remote artifact folder:** `/root/autodl-tmp/260615_1320_submission`
- **Task:** Crypt10k FULL continuous 2-epoch LoRA SFT from Stage 2 adapter, fresh optimizer/scheduler/RNG.

## Inputs

- **Data:** `/root/autodl-tmp/data/260609_Crypt10k_eqHardened_FULL.csv`
- **Init LoRA:** `/root/autodl-tmp/stage2_adapter_init`
- **Base model:** `/root/autodl-tmp/Nemotron-3-Nano-30B-A3B`

## Config

- **Epochs:** 2 continuous epochs
- **Samples:** 18,804
- **Total steps:** 1,176
- **Batch:** 32, micro batch 4, gradient accumulation 8
- **Sequence length:** 8192
- **LoRA:** rank 32, alpha 32, no-tie, out_proj enabled
- **Optimizer:** AdamW
- **Peak LR:** `1.5e-4`
- **Schedule:** WSD, 5% warmup then stable LR then final 30% linear decay to 0
- **Warmup steps:** 59
- **Stable until step:** 823
- **Decay steps:** 353
- **Gradient clipping:** `max_norm=1.0`
- **Fresh start:** `FRESH_START=1`

## Result

- **Final step:** 1176 / 1176
- **Final loss:** 0.009984
- **Final grad_norm:** 0.0145
- **Final LR:** `4.25e-07`
- **Elapsed time:** 1245.3 min, 20.75 hr
- **Adapter saves:** 235, 706, 941, 1000, 1058, 1117, 1176
- **Final checkpoint:** `checkpoint-1176_loss0.0100_lr4.25e-07`

## Artifacts

Large files are on the AutoDL instance for download:

- `/root/autodl-tmp/260615_1320_submission/submission.zip`
- `/root/autodl-tmp/260615_1320_submission/submission_crypt10k_FULL_2ep_wsd_contS2_notie_outproj.zip`
- `/root/autodl-tmp/260615_1320_submission/checkpoint-1176.zip`

Local small files saved here:

- `train_crypt10k_FULL_2ep_wsd_contS2.py`
- `train_log_crypt10k_2ep_wsd_notie_outproj.txt`
- `train_crypt10k_FULL_2ep_wsd_contS2.nohup.log`
- `requirements_crypt10k_FULL_2ep_wsd_contS2.txt`
- `TRAINING_INFO.md`

## Validation

- `submission.zip`: CRC passed, 2 files, 3.6 GB.
- `submission_crypt10k_FULL_2ep_wsd_contS2_notie_outproj.zip`: CRC passed, same content as `submission.zip`.
- `checkpoint-1176.zip`: CRC passed, 8 files, 9.8 GB.
- Submission adapter config uses `metric/nemotron-3-nano-30b-a3b-bf16`.
- Submission adapter has 12011 tensors.
- Submission adapter has `backbone.lm_head` keys and no stale `base_model.model.lm_head` keys.
- Submission adapter includes `out_proj` LoRA tensors.
