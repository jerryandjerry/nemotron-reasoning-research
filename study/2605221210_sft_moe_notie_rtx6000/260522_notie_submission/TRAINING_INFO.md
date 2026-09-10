# Training Info — Exp2: NO MoE tie (NOTIE)

**Run name:** `2605221210_sft_moe_notie_r32_a32_lr2e4_seq8192_bs32_rtx6000`
**Finished:** 2026-05-23 ~01:15 Chicago (instance 6000_QLORA_SFT_1, port 28174)
**Purpose:** One of three tying-comparison runs. This one applies **no MoE tying at all** — each of the 128 routed experts gets its own independent rank-32 LoRA. Compared against Exp1 (corrected expert-direction tie) and run #7 (broken rank-direction "tie", 0.84).

## Config
- **Model:** Nemotron-3-Nano-30B-A3B (bf16), Unsloth FastLanguageModel
- **Method:** LoRA bf16 + Cut Cross-Entropy (CCE) + manual lm_head LoRA, **MoE tying OFF** (`MOE_TIE=0`)
- **LoRA:** r=32, α=32, dropout=0
- **Targets:** q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, in_proj, out_proj, lm_head
- **MoE tie:** NONE — every expert's up_proj/down_proj LoRA is independent (full rank-32 per expert)
- **Optimizer/LR:** 2e-4 → 0 linear decay
- **Batch:** 32 (micro=4, grad_accum=8), seq=8192
- **Steps:** 246 (1 epoch), Samples: 7849
- **is_fast_path_available:** True (Mamba fused kernel ON — out_proj LoRA stays dead by design, same as Exp1 / huikang)

## Data
- `/root/autodl-tmp/data/260514_huikang_golden_stripped.csv` (md5 92a515f31cd2e89f9cb355c4ed09f927) — identical to Exp1

## Results
- **Final loss:** 0.0029 (step 246)
- **Time:** 4.04 hrs (242.7 min)
- **Peak VRAM:** ~88.6 GB (smi)

## Adapter verification (byte-level, all 128 experts)
- up_proj.lora_A: **UNTIED (only 1/128 = expert 0 itself), rank=32, distinct_rows=32/32** ✅
- down_proj.lora_B: **UNTIED, rank=32, distinct_rows=2688/2688** ✅
- out_proj.lora_B: **0.0000 (dead — fast-path bypass, by design)**
- in_proj.lora_B: 1.48 (live), lm_head.lora_B: 3.26 (live)

Contrast vs Exp1: identical except experts are **independent** here (Exp1 tied them across experts). Both are full rank-32 — the only variable is the tie.

## Submission artifacts (in this folder)
| Zip | Size | Recipe |
|-----|------|--------|
| `submission_moe_notie.zip` | 3642.6 MB | endpoint, step-246 adapter |
| `submission_svd-soup-last5.zip` | 3657.3 MB | SVD-merge of steps {200, 209, 221, 234, 246} |
| `submission_svd-wise-50-150-246.zip` | 3657.3 MB | SVD-merge of steps {50, 150, 246} |
| `checkpoint-246_loss0.0029_lr8.13e-07.zip` | 7782 MB | full resume checkpoint |

Adapter saves on disk: [50, 100, 150, 197, 200, 209, 221, 234, 246]
