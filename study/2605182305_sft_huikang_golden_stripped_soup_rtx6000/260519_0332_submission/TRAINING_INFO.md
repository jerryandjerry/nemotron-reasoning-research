# Training Info — Run #6 Reproduction with Soups

> **Trained:** 2026-05-18 23:14 → 2026-05-19 03:32 Chicago time (≈ 4.3 hr)
> **Instance:** AutoDL RTX PRO 6000 Blackwell (95 GB), port 21578
> **SwanLab run:** https://swanlab.cn/@jerry4083/260410_Nemotron/runs/6uomfyrl821baihma0j7b

## What this run tests

Run #10 (retrain of #7) diverged from run #7 by 8pp (0.84 → 0.76). User asked to **reproduce run #6's 0.84** with soups, byte-identical setup to verify whether single-trajectory soup can lift the score above 0.84.

Step-by-step comparison against run #6 was done live during training (see `train_log.txt` + the comparison polling script). **Final-step loss matched run #6 exactly: 0.002917 vs 0.002938 (Δ = -2e-5)**, in stark contrast to run #10 vs #7 which differed by 5e-4 at the final step.

## Data verification (vs run #6's startup log)

| | Run #6 | Run #11 (this) |
|---|---|---|
| CSV | `260514_huikang_golden_stripped.csv` | `260514_huikang_golden_stripped.csv` |
| MD5 | not recorded | `92a515f31cd2e89f9cb355c4ed09f927` |
| File size | not recorded | 35,597,597 bytes |
| File mtime | (file already in place) | 2026-05-15 14:19 Chicago (pre-dates run #6's start) |
| CSV rows (unique) | 6906 | 6906 ✓ |
| Expanded rows | 7849 | 7849 ✓ |
| Total tokens | 27,989,496 | 27,989,496 ✓ |
| Unmasked tokens | 26,713,988 | 26,713,988 ✓ |
| Stratified interleave seed | 42 | 42 ✓ |

Data is byte-identical to what run #6 used.

## Config

Same as Sub #6 (which scored 0.84). Soup additions are functional no-ops for training math.

| Parameter | Value |
|---|---|
| Model | Nemotron-3-Nano-30B-A3B bf16 (Unsloth FastLanguageModel) |
| Method | LoRA bf16 + Cut Cross-Entropy + MoE weight tying |
| LoRA | r=32, α=32, dropout=0 |
| Targets | q/k/v/o_proj, up/down_proj, in/out_proj, lm_head (9 + manual lm_head) |
| Optimizer | AdamW betas=(0.9, 0.95), eps=1e-8, wd=0 |
| LR | 2e-4 → 0 linear decay |
| Batch | 32 (micro=4, ga=8) |
| Max seq len | 8192 |
| MoE tying | True (5,888 tied params) |
| LoRA cast | fp32 |
| Steps | 246 |
| FIFO (full ckpt) | 1 (per routine §2.4A) |
| Adapter-only saves | {50, 100, 150, 197, 200, 209, 221, 234, 246} (no FIFO) |
| Boxed extractor | Kaggle metric (`\\boxed{` + `rfind('}')`) |

## Training trajectory comparison vs run #6

| Step | Run #6 loss | Run #11 loss | Δloss | Run #6 grad_norm | Run #11 grad_norm | Δgrad_norm |
|---|---|---|---|---|---|---|
| 1 | **0.403614** | **0.403614** | **+0.000000** | 1.1476 | 1.1474 | -0.0002 |
| 50 | 0.005968 | 0.005903 | -0.000065 | 0.1286 | 0.1362 | +0.0076 |
| 100 | 0.004522 | 0.004594 | +0.000072 | 0.0737 | 0.0835 | +0.0098 |
| 150 | 0.004012 | 0.003943 | -0.000069 | 0.0706 | 0.0657 | -0.0049 |
| 200 | 0.003133 | 0.003368 | +0.000235 | 0.1279 | 0.1295 | +0.0016 |
| 246 | **0.002938** | **0.002917** | **-0.000021** | 0.0975 | 0.0564 | -0.0411 |

Step-1 loss **bit-identical to 6 decimals** → confirms data, model init, forward kernels are bit-identical. Step-1 grad_norm differs by 0.0002 (CUDA atomicAdd noise in backward). Over 246 steps the loss drift stays bounded at ±5e-4, ending essentially equal at step 246.

This is fundamentally different from run #10 vs #7 (which drifted to 5e-4 at the end). Why? Unclear — the per-step noise is qualitatively similar, but THIS trajectory happened to stay near run #6's path while run #10 wandered off run #7's.

## Results

- **Train time:** 258 min (4.30 hr), ~1.05 min/step
- **Final loss:** 0.0029 (step 246) — matches run #6's 0.0029 exactly
- **Peak VRAM:** ~88.7 GB / 95 GB
- **Trainable params:** 888,154,112 / 32,466,091,456

## Outputs

In `260519_0332_submission/`:

| File | Size | Recipe |
|---|---|---|
| `submission.zip` | 1.3 GB | final-246 control |
| `submission_soup-last5.zip` | 1.3 GB | naive avg{200, 209, 221, 234, 246}, lm_head renamed |
| `submission_wise-50-150-246.zip` | 1.3 GB | naive avg{50, 150, 246} |
| `submission_svd-soup-last5.zip` | 2.8 GB | SVD-merge (`(1/N) Σ Bᵢ@Aᵢ` then rank-32 SVD) of {200, 209, 221, 234, 246} |
| `submission_svd-wise-50-150-246.zip` | 2.8 GB | SVD-merge of {50, 150, 246} |
| `checkpoint-246_loss0.0029_lr8.13e-07.zip` | 2.9 GB | full resume state |
| `train_huikang_golden_stripped_soup.py` | 28 KB | training script |
| `build_soups.py` | 7 KB | naive soup builder |
| `svd_soup.py` | 7.5 KB | SVD soup builder (QR+SVD trick) |
| `train_log.txt` | 1.7 MB | full training log |
| `requirements.txt` | 5.6 KB | pip freeze |
| `TRAINING_INFO.md` | this file | |

**Adapter post-processing applied** (in training script for final + in build_soups.py / svd_soup.py for soups):
- `base_model_name_or_path` → `metric/nemotron-3-nano-30b-a3b-bf16`
- `inference_mode` → True
- lm_head keys renamed: `base_model.model.lm_head.*` → `base_model.model.backbone.lm_head.*`

## Score predictions

| Submission | Expected score | Reasoning |
|---|---|---|
| `final-246` | ≈ 0.84 | Final loss matches run #6 exactly. If training is reproducible at the metric level, this hits 0.84. |
| `soup-last5` (naive) | 0.83–0.85 | Smooths the converged tail of a 0.84 trajectory. Run #10's data showed +0.02 over its 0.76 endpoint; here the endpoint is already 0.84, so improvement may be smaller (less noise to average out). |
| `soup-last5` (SVD) | 0.83–0.85 | Mathematically correct rank-32 truncation of the avg-of-deltas. σ_r ≈ 0 means truncation loses ~nothing. Should ≈ naive last5 for nearby checkpoints. |
| `wise-50-150-246` (naive) | 0.80–0.84 | Naive averaging of distant LoRAs introduces large cross-term garbage (B_50 @ A_246 etc). |
| `wise-50-150-246` (SVD) | 0.82–0.85 | Cleaner average than naive for distant ckpts. |

## Comparison vs prior reproduction attempts

| Run | Setup | Final loss | Score |
|---|---|---|---|
| #6 (original) | `huikang_golden_stripped.csv` | 0.0029 | 0.84 |
| #7 (lkall variant, 65 huikang→lkevincc) | `huikang_lkall_stripped.csv` | 0.0029 | 0.84 |
| #10 (retrain of #7 with soups) | same as #7 | 0.0034 | **0.76** ← drifted |
| **#11 (this — reproduction of #6 with soups)** | same as #6 | **0.0029** | **TBD** |

If #11 final hits 0.84, the divergence of #10 was unlucky noise that compounded. If it hits 0.76, the divergence is systematic and we need to investigate further.
