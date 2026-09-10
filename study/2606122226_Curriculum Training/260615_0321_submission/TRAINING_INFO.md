# Curriculum Stage 2 — Training Info

**Run name:** `260614_..._sft_curriculum_stage2_notie_outproj_r32_a32_lr2e4_seq8192_bs32_rtx6000`
**SwanLab:** project `260410_Nemotron`, run `h0xw5kok4ubgg2apjq41u`
**Date:** finished 2026-06-15 ~03:21 CDT (Chicago) — 7.43 hr run
**GPU:** NVIDIA RTX PRO 6000 Blackwell (96 GB)

## What this is
Stage 2 of the 3-stage curriculum. **Continual-FT** from the Stage 1 LoRA (`adapter-230`)
on the Stage 2 data (100% original replay + ~50% aug1). Per the user's Stage 2 config edit,
this uses the **SAME proven recipe as Stage 1 — linear 2e-4→0, NO warmup, NO WSD, NO cosine**
(NOT the original recipe-doc's WSD/1.5e-4; a stray Codex edit to that effect was reverted).
Code byte-identical to `train_stage1.py` except LOG_FILE/_DATA_TAG/CSV naming.

## Config
- **Task:** continual-SFT (LoRA), init from Stage 1 adapter (FRESH_START=1 + INIT_ADAPTER_FROM)
  - `INIT_ADAPTER_FROM=adapter_output_curriculum_stage1_notie_outproj/adapter-230_loss0.0019_lr8.70e-07`
  - Init load verified: 12,010 LoRA tensors, 0 missing; `lora_B norm = 1.467` (warm start, step-1 loss 0.014)
- **Model:** Nemotron-3-Nano-30B-A3B, bf16, Unsloth + CCE
- **LoRA:** r=32, α=32, dropout=0, no-tie experts, live out_proj, targets q/k/v/o/up/down/in/out_proj + lm_head (+ MoE experts)
- **Optimizer:** AdamW(0.9, 0.95), wd=0 (fresh — optimizer NOT restored from Stage 1)
- **LR:** 2e-4 → 0, **linear decay, NO warmup / NO WSD / NO cosine**
- **Batch:** 32 (micro=4, ga=8), seq_len=8192
- **Epochs:** 1 → **431 steps**
- **Grad clip:** max_norm=1.0
- **Shuffle:** stratified category interleave, seed=42

## Data — `260614_STAGE2_FULL.csv` (md5 95aca71b7117d27e0644a9133bb0a059)
- 19,961 CSV rows; **12,896 active** (oversampling≥1, all GT-match=True), aug gated via oversampling=0
- Expanded **13,786** (12,006×1 + 890×2)
- Active sources: pretok_0408_decoded 4191 (100% original replay), 260614_Bitm_solver 3228, 260614_NE_solver 2866, 260614_Crypt_solver 2611
- Expanded category mix: bit_manipulation 3621, equation_numeric_deduce 2618, cryptarithm_deduce 2222,
  cipher 1656, unit_conversion 1070, gravity 1055, numeral 730, cryptarithm_guess 530, equation_numeric_guess 284

## Result
- **Time:** 7.43 hrs (445.8 min), ~1.0 min/step
- **Final loss:** 0.0051 (step 431); warm continual-FT — loss stayed ~0.005–0.02 throughout
- **Grad norm:** max **0.239** across all 431 steps — **no spikes** (clip never engaged)
- **VRAM:** ~88.7 GB peak (stress confirmed 88.9 GB worst-case), fits 95 GB
- **out_proj LoRA alive:** stress grad-norm sum 0.013 > 0 ✓

## Artifacts
- `submission_curriculum_stage2_notie_outproj.zip` (3.83 GB) — final-step adapter, Kaggle-ready
  (base_model_name_or_path=`metric/nemotron-3-nano-30b-a3b-bf16`, lm_head keys renamed to backbone.lm_head ✓)
- `checkpoint-431_loss0.0051_lr4.64e-07.zip` (10.4 GB) — full resume checkpoint
- Soup adapters: trainer saved all 7 (86/259/345/366/388/409/431); user removed 86/259/345/366 to free disk
  during the run — remaining on instance: 388/409/431. (No soup build planned for Stage 2.)
- Stage 3 continual-FT can init from instance adapter `adapter-431_loss0.0051_lr4.64e-07`.
