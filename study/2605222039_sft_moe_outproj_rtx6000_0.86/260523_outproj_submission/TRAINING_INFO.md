# Training Info — out_proj experiment (no-tie + LIVE out_proj)

**Run name:** `2605222039_sft_moe_notie_outproj_r32_a32_lr2e4_seq8192_bs32_rtx6000`
**Finished:** 2026-05-23 ~13:00 Chicago (instance 6000_QLORA_SFT_2, port 21578)
**Purpose:** Test the one architectural difference between our 0.84 (Subs #6/#7/#11/#15) and huikang's live-out_proj Tinker 0.86: a **trained Mamba `out_proj` LoRA**. Clean A/B vs the no-tie 0.84 baseline (Sub #15) — *the only difference is that out_proj now trains.*

## Config (identical to notie 0.84 EXCEPT out_proj)
- **Model:** Nemotron-3-Nano-30B-A3B (bf16), Unsloth FastLanguageModel, transformers 4.56.2 (base env)
- **Method:** LoRA bf16 + CCE + manual lm_head LoRA, **no MoE tie** (MOE_TIE=0)
- **out_proj fix (USE_MEM_EFF=0):** keep `is_fast_path_available=True`, set each Mamba mixer's `.training=False` so `cuda_kernels_forward` takes its own efficient unfused else-branch (`causal_conv1d_fn` + `mamba_chunk_scan_combined` + `self.out_proj(scan_output)` as a MODULE call) → out_proj LoRA receives gradients. No model-file edit; uses the model's own kernels. **Independently audited VALID** (only `self.training` use in the mixer is the fused gate; no baseline impact; inference-consistent).
- **LoRA:** r=32, α=32, dropout=0; targets q/k/v/o/up/down/in/out_proj + lm_head
- **Optimizer/LR:** 2e-4 → 0 linear; **Batch:** 32 (micro 4, ga 8), seq 8192; **Steps:** 246
- **Data:** `260514_huikang_golden_stripped.csv` (md5 92a515f31cd2e89f9cb355c4ed09f927, 6906 unique → 7849 expanded) — identical to Sub #15.
  - **Provenance (from the CSV `source` column):** 6106 huikang "golden" non-cryptarithm rows + 800 cryptarithm rows. The 800 cryptarithm CoTs are a mix: **735 lkevincc** (`source=lkevincc_golden`) **+ 65 huikang** (`source=pretok_0408_decoded`) — NOT all-lkevincc. Categories: bit_manipulation 1354, cipher 1576, gravity 975, unit_conversion 990, numeral 650, cryptarithm_deduce 639, equation_numeric_deduce 540, cryptarithm_guess 161, equation_numeric_guess 21 (unique counts).

## Results
- **Final loss:** 0.0030 | **Time:** 4.19 hrs | **Peak VRAM:** ~88.7 GB

## Adapter verification (final endpoint, byte-level)
| module | this run | notie 0.84 (Sub #15) | huikang 0.86 Tinker |
|---|---|---|---|
| total tensors / expert keys | 12,011 / 11,868 | 12,011 / 11,868 | 12,010 / 11,868 |
| **out_proj.lora_B norm** | **~0.70 (TRAINED)** | **0.0000 (dead)** | 0.70–0.74 |
| in_proj.lora_B / lm_head.lora_B | 1.48 / 3.21 (live) | live | live |
| experts | untied, rank-32 | untied, rank-32 | tied, rank-32 |
| key naming | backbone (Kaggle-compat) | backbone | model.model |

Our out_proj now trains to a magnitude matching huikang's Tinker 0.86 adapter. Everything else matches our 0.84 baseline → isolates out_proj.

## Submission artifacts (in this folder)
| Zip | Size | Recipe |
|-----|------|--------|
| `submission_moe_notie_outproj.zip` | 3654.9 MB | endpoint, step-246 |
| `submission_svd-soup-last5.zip` | 3674.8 MB | SVD-merge {200,209,221,234,246} |
| `submission_svd-wise-50-150-246.zip` | 3674.8 MB | SVD-merge {50,150,246} |
| `checkpoint-246_loss0.0030_lr8.13e-07.zip` | 10,414 MB | full resume checkpoint |

Adapter saves on disk: [50,100,150,197,200,209,221,234,246]. See [[project_train_outproj_lora]] for the mechanism.
