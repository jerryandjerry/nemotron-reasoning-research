# Curriculum Stage 1 — Training Info

**Run name:** `260614_1429_sft_curriculum_stage1_notie_outproj_r32_a32_lr2e4_seq8192_bs32_rtx6000`
**SwanLab:** project `260410_Nemotron`, run `qb8vit0gnbu1titqik9fa`
**Date:** 2026-06-14 (Chicago) — started 14:29, finished ~18:22 CDT
**GPU:** NVIDIA RTX PRO 6000 Blackwell (96 GB)

## What this is
Stage 1 of the 3-stage curriculum (recipe: `../Curriculum Training Recipe.md`).
Stage 1 = the **proven 0.86/0.87 SFT recipe** run on the curriculum's **original-only** data,
to fit output format + easy categories and teach the answer template for hard categories.
Code is **byte-identical to the 0.87 run** (`2606111516_sft_crypt10k_FULL_ep2_contS3_rtx6000_0.87/train_crypt10k_FULL_ep2_contS3.py`)
except: docstring, LOG_FILE/_DATA_TAG naming → `curriculum_stage1`, CSV path → `260614_STAGE1_FULL.csv`,
and `clip_grad_norm max_norm 1e9 → 1.0` (routine-required; inert on normal steps).

## Config
- **Task:** continual-SFT (LoRA), **random init** (FRESH_START=1, INIT_ADAPTER_FROM unset)
- **Model:** Nemotron-3-Nano-30B-A3B, bf16 (NOT 4-bit), Unsloth + CCE
- **LoRA:** r=32, α=32, dropout=0, no-tie experts, **live out_proj** (mixer.training=False → fused Mamba else-branch)
- **Targets:** q/k/v/o_proj, up/down/in/out_proj, lm_head (+ MoE experts gate_up_proj/down_proj)
- **Optimizer:** AdamW(0.9, 0.95), wd=0
- **LR:** 2e-4 → 0, **linear decay, NO warmup / NO WSD / NO cosine**
- **Batch:** 32 (micro=4, grad_accum=8), seq_len=8192
- **Epochs:** 1 → **230 steps**
- **Grad clip:** max_norm=1.0
- **Shuffle:** stratified category interleave, seed=42

## Data — `260614_STAGE1_FULL.csv` (md5 cfdf8ac2327df50379cf552b2eec5bcb)
- 15,102 CSV rows; **6,450 active** (oversampling≥1, all GT-match=True), aug gated off via oversampling=0
- Expanded **7,340** (5,560×1 + 890×2 = original-only)
- Active sources: pretok_0408_decoded 4191, 260614_Bitm_solver 1497, 260614_NE_solver 621, 260614_Crypt_solver 141
- Expanded category mix: bit_manipulation 1890, cipher 1656, unit_conversion 1070, gravity 1055,
  numeral 730, equation_numeric_deduce 599, cryptarithm_deduce 264, equation_numeric_guess 58, cryptarithm_guess 18

## Result
- **Time:** 3.89 hrs (233.2 min), ~1.0 min/step
- **Final loss:** 0.0019 (step 230); loss fell 0.52 → ~0.01 by step 48, then ~0.002–0.008
- **Grad norm:** max **0.14** across all 230 steps — **no spikes** (clip never engaged)
- **VRAM:** ~88 GB peak (stress confirmed 88.6 GB worst-case), fits 95 GB
- **out_proj LoRA alive:** stress grad-norm sum 0.1395 > 0 ✓ (23 Mamba mixers forced to else-branch)

## Artifacts
- `submission_curriculum_stage1_notie_outproj.zip` (3.83 GB) — final-step adapter, Kaggle-ready
  (base_model_name_or_path=`metric/nemotron-3-nano-30b-a3b-bf16`, lm_head keys renamed to backbone.lm_head ✓)
- `checkpoint-230.zip` — full resume checkpoint (adapter + optimizer + scheduler + RNG + tokenizer + args)
- Soup adapters saved on instance (no build this stage): steps 46/138/184/196/207/218/230
- Stage 2 continual-FT can init from instance adapter `adapter-230_loss0.0019_lr8.70e-07` (instance kept up).
