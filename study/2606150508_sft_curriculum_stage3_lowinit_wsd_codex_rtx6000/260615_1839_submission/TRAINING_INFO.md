# Training Info

- **Run:** curriculum stage3 low-init WSD, codex
- **Instance:** ssh -p 44311 root@connect.bjb1.seetacloud.com
- **Local study folder:** `study/2606150508_sft_curriculum_stage3_lowinit_wsd_codex_rtx6000`
- **Remote submission folder:** `/root/autodl-tmp/260615_1839_submission`
- **Training script:** `train_stage3_lowinit.py`
- **Data:** `/root/autodl-tmp/data/260614_STAGE3_FULL.csv`
- **Data SHA256:** `ca8ba8da0ad3abb4b6a97f819552342723e492ceebb43cffa34c940edbc02c47`
- **Rows / expanded samples:** 19,961 rows / 19,985 expanded
- **Init adapter:** `/root/autodl-tmp/stage3_lowinit_adapter_init` extracted from `/root/autodl-tmp/260615_0321_submission/submission_curriculum_stage2_notie_outproj.zip`
- **Base model path in adapter config:** `metric/nemotron-3-nano-30b-a3b-bf16`

## Config

- **Epochs:** 1
- **Total steps:** 625
- **Peak LR:** 1.5e-4
- **Schedule:** WSD, 5% warmup, stable to 70%, final 30% linear decay
- **Final LR:** 0
- **Batching:** same stage3 routine config, manual loop, gradient clipping max_norm=1.0
- **Checkpoint FIFO:** latest full checkpoint kept; final checkpoint force-saved
- **Soup adapter snapshots:** 20%, 60%, 80%, 85%, 90%, 95%, 100%

## Result

- **Final step:** 625/625
- **Final loss:** 0.006335
- **Final grad_norm:** 0.0058
- **Final LR:** 0.00e+00
- **Training elapsed:** 133.5 min (2.23 h) after resume from checkpoint-500

## Artifacts On AutoDL

Download these two large files from `/root/autodl-tmp/260615_1839_submission/`:

- `submission.zip` - Kaggle-ready final LoRA adapter
- `checkpoint-625.zip` - full resume checkpoint

The descriptive hardlink names are also present:

- `submission_curriculum_stage3_lowinit_notie_outproj.zip`
- `checkpoint-625_loss0.0063_lr0.00e+00.zip`

## Integrity Checks

- `submission_curriculum_stage3_lowinit_notie_outproj.zip`: CRC passed, 2 entries
- `checkpoint-625_loss0.0063_lr0.00e+00.zip`: CRC passed, 8 entries
- Adapter tensor count: 12011
- `backbone.lm_head` keys: present
- stale `base_model.model.lm_head` keys: absent
- `out_proj` LoRA keys: present

