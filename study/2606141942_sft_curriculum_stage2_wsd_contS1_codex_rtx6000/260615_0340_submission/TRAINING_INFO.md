# Curriculum Stage 2 SFT

- **Run folder:** `2606141942_sft_curriculum_stage2_wsd_contS1_rtx6000`
- **AutoDL instance:** `ssh -p 22079 root@connect.bjb1.seetacloud.com`
- **Remote artifact folder:** `/root/autodl-tmp/260615_0340_submission`
- **Task:** Stage 2 curriculum LoRA SFT from Stage 1 adapter, fresh optimizer/scheduler/RNG.

## Inputs

- **Data:** `/root/autodl-tmp/data/260614_STAGE2_FULL.csv`
- **Data SHA256:** `8b55754443aa7f7fcdaaf8a43681f3d546eea298f305c5dd21a9bed7bffb1c9f`
- **Init LoRA:** `/root/autodl-tmp/adapter_output_curriculum_stage1_notie_outproj/adapter-230_loss0.0019_lr8.70e-07`
- **Base model:** `/root/autodl-tmp/Nemotron-3-Nano-30B-A3B`

## Config

- **Epochs:** 1
- **Total steps:** 431
- **Batch:** 32, micro batch 4, gradient accumulation 8
- **Sequence length:** 8192
- **LoRA:** rank 32, alpha 32, no-tie, out_proj enabled
- **Optimizer:** AdamW
- **Peak LR:** `1.5e-4`
- **Schedule:** WSD, 5% warmup then stable LR then final 30% linear decay to 0
- **Warmup steps:** 22
- **Stable until step:** 302
- **Decay steps:** 129
- **Gradient clipping:** `max_norm=1.0`
- **Fresh start:** `FRESH_START=1`

## Result

- **Final step:** 431 / 431
- **Final loss:** 0.005282
- **Final grad_norm:** 0.0112
- **Final LR:** 0
- **Elapsed time:** 432.3 min, 7.21 hr
- **Note:** one late pre-clip grad_norm spike at step 402 was observed and clipped by `max_norm=1.0`; subsequent steps were normal.

## Artifacts

Large files are on the AutoDL instance for download:

- `/root/autodl-tmp/260615_0340_submission/submission.zip`
- `/root/autodl-tmp/260615_0340_submission/submission_curriculum_stage2_notie_outproj.zip`
- `/root/autodl-tmp/260615_0340_submission/checkpoint-431.zip`

Local small files saved here:

- `train_stage2.py`
- `train_log_curriculum_stage2_notie_outproj.txt`
- `requirements_curriculum_stage2.txt`
- `TRAINING_INFO.md`

## Validation

- `submission.zip`: CRC passed, 2 files, 3.6 GB.
- `submission_curriculum_stage2_notie_outproj.zip`: CRC passed, same content as `submission.zip`.
- `checkpoint-431.zip`: CRC passed, 8 files, 9.3 GB.
- Submission adapter config uses `metric/nemotron-3-nano-30b-a3b-bf16`.
- Submission adapter has 12011 tensors.
- Submission adapter has `backbone.lm_head` keys and no stale `base_model.model.lm_head` keys.
- Submission adapter includes `out_proj` LoRA tensors.
