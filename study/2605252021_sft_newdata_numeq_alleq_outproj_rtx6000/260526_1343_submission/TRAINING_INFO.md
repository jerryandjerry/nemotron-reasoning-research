# allEq (NumericEq keep-all) — TRAINING_INFO

- **Run name:** 260525_2023_sft_newdata_alleq_notie_outproj_r32_a32_lr2e4_seq8192_bs32_rtx6000
- **Instance:** 28174 (RTX PRO 6000 Blackwell), SSH_LOGIN_6000_QLORA_SFT_1
- **Data:** `260525_huikang_NumericEq_allEq.csv` (md5 d2fe439d0cc64c7ac3078b89d795a886), 7077 unique → **8020 expanded**, 251 steps. = the 0.86 (#16) ID set with the numeric-equation CoTs swapped to new-solver, **keeping ALL 732 new-solver equation rows** (incl. 114 GT-mismatch + 69 over-long). vs numeq which drops the 171 lower-confidence ones.
- **Recipe (= 0.86 #16):** Unsloth + manual loop + CCE + manual lm_head LoRA, transformers 4.56.2. r32/α32/dropout0, targets q/k/v/o/up/down/in/out_proj + lm_head, **no-tie + LIVE out_proj** (Forced unfused else-branch on 23 Mamba mixers). AdamW lr 2e-4→0 linear, betas(0.9,0.95), bs32 (micro4 ga8), seq8192, stratified seed42. Env `MOE_TIE=0 USE_MEM_EFF=0 FRESH_START=1`. Endpoint-only (no soup, ADAPTER_SAVE_STEPS=[]).
- **Result:** clean convergence, no grad spikes (grad_norm always <1). Final loss **0.0024**; tracked #16's curve throughout (0.5→~0.005 by step ~50). Time **4.11 hrs**. Trainer diff vs proven gt2x trainer = only CSV_PATH + _DATA_TAG + ADAPTER_SAVE_STEPS=[].
- **Endpoint:** `submission_newdata_alleq_notie_outproj.zip` (3,831,949,291 bytes, md5 **a063c01203326353d6ee7dd22cd33804**). Checkpoint: `checkpoint-251_loss0.0024_lr7.97e-07.zip` (10.4 GB).
- **Purpose:** A/B vs numeq (drop-mismatch) and vs 0.86 — does keeping the lower-confidence new-solver equation CoTs help or hurt? Kaggle score pending.
