# Training Info — gold-only run (no-tie + LIVE out_proj)

**Run name:** `260523_2105_sft_newdata_goldonly_notie_outproj_r32_a32_lr2e4_seq8192_bs32_rtx6000`
**Finished:** 2026-05-24 ~01:18 Chicago (instance 6000_QLORA_SFT_2, port 21578)
**Purpose:** Beat 0.86 with a gold-only data variant. Recipe is byte-identical to the proven newsolver/0.86 trainer (verified via diff — only `CSV_PATH` + `_DATA_TAG` differ; the 2 stress-only guards are no-ops in real training). The ONLY experimental change is the data.

## Config (exact same code as newsolver run, only data differs)
- **Model:** Nemotron-3-Nano-30B-A3B (bf16), Unsloth, transformers 4.56.2
- **Launch:** `MOE_TIE=0 USE_MEM_EFF=0 FRESH_START=1` → no MoE tie + live out_proj (mode verified at launch: "Forced unfused else-branch on 23 Mamba mixers")
- **LoRA:** r=32, α=32, dropout=0; q/k/v/o/up/down/in/out_proj + lm_head
- **Optim/LR:** 2e-4 → 0 linear; **Batch:** 32 (micro 4, ga 8), seq 8192; **Steps:** 247
- **Data:** `260523_huikang_newsolver_goldonly.csv` — 6906 unique → **7904 expanded**.
  - Cryptarithm CoTs (source "new solver"): **285 gold-matching kept at 3× oversampling** (→855 samples); 515 dropped (346 GT-mismatch + 169 over-long, oversampling=0).
  - Non-cryptarithm huikang-golden rows untouched (1×/2×/6× → 7049 samples).
  - vs newsolver run: that one KEPT the 346 mismatched CoTs; this one DROPS them and 3×-weights the surviving gold crypt. Opposite hypothesis.

## Results
- **Final loss:** 0.0052 (step 247, lr≈0) | **Time:** 4.22 hrs | **Peak VRAM:** ~88.6 GB
- Note: floor higher than newsolver (0.0021) / 0.86 (0.0030) — expected, the 3× gold-crypt weighting emphasizes harder samples. Train loss is not the target metric.

## Adapter verification (final endpoint)
| module | this run | 0.86 (target) |
|---|---|---|
| total tensors / expert keys | 12,011 / 11,868 | 12,011 / 11,868 |
| **out_proj.lora_B norm** | **mean 0.76 / max 0.88 (TRAINED)** | ~0.70 (trained) |
| key naming | backbone (Kaggle-compat) | backbone |

## Artifacts
| Zip | Size | Note |
|-----|------|------|
| `submission_newdata_goldonly_notie_outproj.zip` | 3.83 GB | endpoint — **submit this** |
| `checkpoint-247_loss0.0052_lr8.10e-07.zip` | 10.4 GB | resume checkpoint |

Adapter saves: [49, 148, 198, 210, 222, 235, 247]. **Weighted SVD soups built** (in this folder): `submission_last5-soup.zip` (0.5 on final 247 + 0.125 each on 198/210/222/235) + `submission_wise-soup.zip` (49/148/247 = 0.1/0.2/0.7).
