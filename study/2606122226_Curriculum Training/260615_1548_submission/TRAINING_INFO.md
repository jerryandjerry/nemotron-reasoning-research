# Curriculum Stage 3 — Training Info

**Run name:** `260615_..._sft_curriculum_stage3_notie_outproj_r32_a32_lr2e4_seq8192_bs32_rtx6000`
**SwanLab:** project `260410_Nemotron`, run `sxbuaq4iwq1fu9upif03m`
**Date:** finished 2026-06-15 ~15:48 CDT (Chicago) — 11.07 hr run
**GPU:** NVIDIA RTX PRO 6000 Blackwell (96 GB), AutoDL instance port 24910 (duplicate of the Stage 1/2 box)

## What this is
Stage 3 of the 3-stage curriculum. **Continual-FT** from the Stage 2 LoRA (`adapter-431`)
on the FULL augmented data (all original + all aug1 + all aug2). Per the user's config edit,
this uses the **SAME proven recipe — linear 2e-4→0, NO warmup, NO WSD, NO cosine** — and
**1 epoch only** (the recipe doc said 2 epochs; user overrode to 1).
Code byte-identical to `train_stage1.py` except LOG_FILE/_DATA_TAG/CSV naming.

## Config
- **Task:** continual-SFT (LoRA), init from Stage 2 adapter (FRESH_START=1 + INIT_ADAPTER_FROM)
  - `INIT_ADAPTER_FROM=adapter_output_curriculum_stage2_notie_outproj/adapter-431_loss0.0051_lr4.64e-07`
  - Init load verified: 12,010 LoRA tensors, 0 missing; `lora_B norm = 2.169` (warm start, step-1 loss 0.006)
- **Model:** Nemotron-3-Nano-30B-A3B, bf16, Unsloth + CCE
- **LoRA:** r=32, α=32, dropout=0, no-tie experts, live out_proj, targets q/k/v/o/up/down/in/out_proj + lm_head (+ MoE experts)
- **Optimizer:** AdamW(0.9, 0.95), wd=0 (fresh — optimizer NOT restored from Stage 2)
- **LR:** 2e-4 → 0, **linear decay, NO warmup / NO WSD / NO cosine**
- **Batch:** 32 (micro=4, ga=8), seq_len=8192
- **Epochs:** **1** → **625 steps**
- **Grad clip:** max_norm=1.0
- **Shuffle:** stratified category interleave, seed=42

## Data — `260614_STAGE3_FULL.csv` (md5 c7e0c83978fbb7c000c951ebe7527494)
- 19,961 CSV rows; **19,095 active** (oversampling≥1, all GT-match=True), aug gated via oversampling=0
- Expanded **19,985** (18,205×1 + 890×2) — full augmented pool
- Active sources: 260614_NE_solver 5000, 260614_Bitm_solver 4958, 260614_Crypt_solver 4946, pretok_0408_decoded 4191
- Expanded category mix: bit_manipulation 5351, equation_numeric_deduce 4536, cryptarithm_deduce 4078,
  cipher 1656, unit_conversion 1070, gravity 1055, cryptarithm_guess 1009, numeral 730, equation_numeric_guess 500

## Result
- **Time:** 11.07 hrs (664.2 min), ~1.06 min/step
- **Final loss:** 0.0065 (step 625); warm continual-FT — loss stayed ~0.005–0.008 throughout
- **Grad norm:** max **0.049** across all 625 steps — **no spikes** (clip never engaged; calmest of the 3 stages)
- **VRAM:** ~89 GB peak (stress confirmed 89.0 GB worst-case), fits 95 GB
- **out_proj LoRA alive:** stress grad-norm sum 0.0045 > 0 ✓

## Artifacts
- `submission_curriculum_stage3_notie_outproj.zip` (3.84 GB) — final-step adapter, Kaggle-ready
  (base_model_name_or_path=`metric/nemotron-3-nano-30b-a3b-bf16`, lm_head keys renamed to backbone.lm_head ✓)
- `checkpoint-625_loss0.0065_lr3.20e-07.zip` (10.4 GB) — full resume checkpoint
- Soup adapters on instance (all 7 kept this time): 125/375/500/531/562/594/625. (No soup build planned.)

## Curriculum summary (3 stages, all linear 2e-4→0, no warmup/WSD)
- Stage 1: random init, original-only (7,340 exp, 230 steps), 3.89 hr
- Stage 2: ← S1 adapter, +50% aug1 (13,786 exp, 431 steps), 7.43 hr
- Stage 3: ← S2 adapter, full aug pool (19,985 exp, 625 steps, 1 epoch), 11.07 hr
