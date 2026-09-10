# Curriculum Stage 3 SFT

- **Run folder:** `2606150434_sft_curriculum_stage3_wsd_contS2_codex_rtx6000`
- **AutoDL instance:** `ssh -p 22079 root@connect.bjb1.seetacloud.com`
- **Remote artifact folder:** `/root/autodl-tmp/260615_1601_submission`
- **Task:** Stage 3 curriculum LoRA SFT from Stage 2 adapter, fresh optimizer/scheduler/RNG.

## Inputs

- **Data:** `/root/autodl-tmp/data/260614_STAGE3_FULL.csv`
- **Data SHA256:** `ca8ba8da0ad3abb4b6a97f819552342723e492ceebb43cffa34c940edbc02c47`
- **Rows:** 19,961 raw rows, 19,985 expanded examples
- **Init LoRA:** `/root/autodl-tmp/stage3_adapter_init`
- **Base model:** `/root/autodl-tmp/Nemotron-3-Nano-30B-A3B`

## Config

- **Epochs:** 1
- **Total steps:** 625
- **Batch:** 32, micro batch 4, gradient accumulation 8
- **Sequence length:** 8192
- **LoRA:** rank 32, alpha 32, no-tie, out_proj enabled
- **Optimizer:** AdamW
- **Peak LR:** `1.5e-4`
- **Schedule:** WSD, 5% warmup then stable LR then final 30% linear decay to 0
- **Warmup steps:** 31
- **Stable until step:** 438
- **Decay steps:** 187
- **Gradient clipping:** `max_norm=1.0`
- **Fresh start:** `FRESH_START=1`

## Result

- **Final step:** 625 / 625
- **Final loss:** 0.006260
- **Final grad_norm:** 0.0071
- **Final LR:** 0
- **Elapsed time:** 658.8 min, 10.98 hr
- **Adapter saves:** 125, 375, 500, 531, 562, 594, 625
- **Final checkpoint:** `checkpoint-625_loss0.0063_lr0.00e+00`

## Artifacts

Large files are on the AutoDL instance for download:

- `/root/autodl-tmp/260615_1601_submission/submission.zip`
- `/root/autodl-tmp/260615_1601_submission/submission_curriculum_stage3_notie_outproj.zip`
- `/root/autodl-tmp/260615_1601_submission/checkpoint-625.zip`

Local small files saved here:

- `train_stage3.py`
- `train_log_curriculum_stage3_notie_outproj.txt`
- `train_stage3_nohup.log`
- `requirements_curriculum_stage3.txt`
- `TRAINING_INFO.md`

## Validation

- `submission.zip`: CRC passed, 2 files, 3.6 GB.
- `submission_curriculum_stage3_notie_outproj.zip`: CRC passed, same content as `submission.zip`.
- `checkpoint-625.zip`: CRC passed, 8 files, 9.7 GB.
- Submission adapter config uses `metric/nemotron-3-nano-30b-a3b-bf16`.
- Submission adapter has 12011 tensors.
- Submission adapter has `backbone.lm_head` keys and no stale `base_model.model.lm_head` keys.
- Submission adapter includes `out_proj` LoRA tensors.
