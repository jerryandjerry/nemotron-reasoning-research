# Training Info — new-data run (no-tie + LIVE out_proj, newsolver cryptarithm CoT)

**Run name:** `2605231427_sft_newdata_notie_outproj_r32_a32_lr2e4_seq8192_bs32_rtx6000`
**Finished:** 2026-05-23 ~18:46 Chicago (instance port 21578)
**Purpose:** Beat our 0.86 (Sub #16, out_proj run) by swapping in a new dataset. Recipe is **byte-for-byte the 0.86 recipe** — the *only* change is the training CSV. Tests whether real newsolver cryptarithm CoTs outperform the prior mixed-source data.

## Config (identical to the 0.86 out_proj run EXCEPT data)
- **Model:** Nemotron-3-Nano-30B-A3B (bf16), Unsloth FastLanguageModel, transformers 4.56.2 (base env)
- **Method:** LoRA bf16 + CCE + manual lm_head LoRA, **no MoE tie** (MOE_TIE=0)
- **out_proj fix (USE_MEM_EFF=0):** keep `is_fast_path_available=True`, set each Mamba mixer's `.training=False` so `cuda_kernels_forward` takes its own efficient unfused else-branch (`causal_conv1d_fn` + `mamba_chunk_scan_combined` + `self.out_proj(scan_output)` as a MODULE call) → out_proj LoRA receives gradients. No model-file edit. See [[project_train_outproj_lora]].
- **LoRA:** r=32, α=32, dropout=0; targets q/k/v/o/up/down/in/out_proj + lm_head
- **Optimizer/LR:** 2e-4 → 0 linear; **Batch:** 32 (micro 4, ga 8), seq 8192; **Steps:** 236
- **Data:** `260523_huikang_newsolver.csv` — 6906 unique rows, stratified-interleave shuffle (seed 42).
  - **Provenance:** 6106 huikang "golden" non-cryptarithm rows **untouched** + 800 cryptarithm rows whose CoTs were swapped to our **newsolver** solver. These are real reasoning traces (some do not reach the gt answer — intentional; true reasoning is preferred over fake answer-matched CoT). Blanked rows (>7800 tok) get oversampling=0.
  - Answer used for training = boxed answer regex-extracted from the CoT (the `answer` column is fallback only).

## Results
- **Final loss:** 0.0021 | **Time:** 4.01 hrs (240.7 min) | **Peak VRAM:** ~88.6 GB

## Adapter verification (final endpoint)
| module | this run | 0.86 out_proj run (Sub #16) |
|---|---|---|
| total tensors / expert keys | 12,011 / 11,868 | 12,011 / 11,868 |
| **out_proj.lora_B norm** | **~0.72 (TRAINED)** | ~0.70 (trained) |
| experts | untied, rank-32 | untied, rank-32 |
| key naming | backbone (Kaggle-compat) | backbone |

## Submission artifacts (on instance / for download)
| Zip | Size | Recipe | Status |
|-----|------|--------|--------|
| `submission_newdata_notie_outproj.zip` | 3.83 GB | endpoint, step-236 | **submit this (only)** |
| `checkpoint-236_loss0.0021_lr8.47e-07.zip` | 10.4 GB | full resume checkpoint | download |
| `submission_svd-soup-last5.zip` | 3.85 GB | SVD-merge {200,201,212,224,236} | **skipped — not submitting** |
| `submission_svd-wise-50-150-236.zip` | 3.85 GB | SVD-merge {50,150,236} | **skipped — not submitting** |

Adapter saves on disk: [50,100,150,189,200,201,212,224,236]. Per user, only the endpoint is being submitted this run.
