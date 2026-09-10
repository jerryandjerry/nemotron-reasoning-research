# Training Info — gt2x run (no-tie + LIVE out_proj)

**Run name:** `260523_2204_sft_newdata_gt2x_notie_outproj_r32_a32_lr2e4_seq8192_bs32_rtx6000`
**Finished:** 2026-05-24 ~02:10 Chicago (instance 6000_QLORA_SFT_1, port 28174)
**Purpose:** Beat 0.86 with the gt2x data variant. Code is exact-same as the proven gold-only/newsolver trainer (diff = only `CSV_PATH` + `_DATA_TAG`; stress guards are no-ops in real training). Only the data differs.

## Config (exact same code, only data differs)
- **Model:** Nemotron-3-Nano-30B-A3B (bf16), Unsloth, transformers 4.56.2
- **Launch:** `MOE_TIE=0 USE_MEM_EFF=0 FRESH_START=1` → no MoE tie + live out_proj (mode verified at launch: "Forced unfused else-branch on 23 Mamba mixers" + run name `...gt2x_notie_outproj`)
- **LoRA:** r=32, α=32, dropout=0; q/k/v/o/up/down/in/out_proj + lm_head
- **Optim/LR:** 2e-4 → 0 linear; **Batch:** 32 (micro 4, ga 8), seq 8192; **Steps:** 244
- **Data:** `260523_huikang_newsolver_gt2x.csv` — 6906 unique → **7806 expanded**.
  - Cryptarithm CoTs (source "new solver"): **285 gold-matching at 2× + 187 at 1×** kept; 328 dropped (oversampling=0).
  - Non-cryptarithm huikang-golden rows untouched (1×/2×/6×).
  - vs gold-only: gold-only used 3× on 285 crypt and dropped 515; gt2x uses 2× on 285 + keeps 187 more at 1×, drops only 328. Milder crypt emphasis, broader coverage.

## Results
- **Final loss:** 0.0057 (step 244, lr≈0) | **Time:** 4.11 hrs | **Peak VRAM:** ~88.7 GB

## Adapter (endpoint)
- Mode verified at launch (notie + out_proj-trains). **out_proj.lora_B norm to be confirmed LOCALLY on download** (expected ~0.7, trained), alongside CRC.

## Artifacts
| Zip | Size | Note |
|-----|------|------|
| `submission_newdata_gt2x_notie_outproj.zip` | 3.83 GB | endpoint — **submit this** |
| `checkpoint-244_loss0.0057_lr8.20e-07.zip` | 10.4 GB | resume checkpoint |

Adapter saves: [49, 146, 195, 207, 220, 232, 244]. **Weighted SVD soups built** (in this folder): `submission_last5-soup.zip` (0.5 on final 244 + 0.125 each on 195/207/220/232) + `submission_wise-soup.zip` (49/146/244 = 0.1/0.2/0.7).
