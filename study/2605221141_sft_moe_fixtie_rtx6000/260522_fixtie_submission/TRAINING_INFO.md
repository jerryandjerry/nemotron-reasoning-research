# Training Info — Exp1: MoE expert-direction tie (FIXTIE)

**Run name:** `2605221141_sft_moe_fixtie_r32_a32_lr2e4_seq8192_bs32_rtx6000`
**Finished:** 2026-05-23 ~00:30 Chicago (instance 6000_QLORA_SFT_2, port 21578)
**Purpose:** One of three tying-comparison runs. This one applies the **corrected MoE expert-direction tie** (share one rank-32 LoRA factor across all 128 routed experts), vs the broken rank-direction "tie" of run #7 (0.84) and the no-tie Exp2.

## Config
- **Model:** Nemotron-3-Nano-30B-A3B (bf16), Unsloth FastLanguageModel
- **Method:** LoRA bf16 + Cut Cross-Entropy (CCE) + MoE weight tying + manual lm_head LoRA
- **LoRA:** r=32, α=32, dropout=0
- **Targets:** q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, in_proj, out_proj, lm_head
- **MoE tie (CORRECTED):** up_proj.lora_A and down_proj.lora_B are tied **across the 128 experts** (one shared factor per layer, full rank-32). Grouping key collapses `.experts.{N}.` → `.experts.E.`; init = mean across experts, grads = sum across experts.
- **Optimizer/LR:** 2e-4 → 0 linear decay
- **Batch:** 32 (micro=4, grad_accum=8), seq=8192
- **Steps:** 246 (1 epoch), Samples: 7849
- **is_fast_path_available:** True (Mamba fused kernel ON — out_proj LoRA stays dead by design, same as huikang's Tinker/Unsloth)

## Data
- `/root/autodl-tmp/data/260514_huikang_golden_stripped.csv` (md5 92a515f31cd2e89f9cb355c4ed09f927)

## Results
- **Final loss:** 0.0036 (step 246)
- **Time:** 4.15 hrs (248.9 min)
- **Peak VRAM:** ~88.4 GB (smi)

## Adapter verification (byte-level, all 128 experts)
- up_proj.lora_A: **TIED (128/128 identical to expert 0), rank=32** ✅
- down_proj.lora_B: **TIED (128/128 identical to expert 0), rank=32** ✅
- out_proj.lora_B: **0.0000 (dead — fast-path bypass, by design)**
- in_proj.lora_B: 1.64 (live), lm_head.lora_B: 3.84 (live)

This confirms the fix: experts are tied at full rank-32, vs run #7's untied rank-1 collapse.

## Submission artifacts (in this folder)
| Zip | Size | Recipe |
|-----|------|--------|
| `submission_moe_fixtie.zip` | 3636.8 MB | endpoint, step-246 adapter |
| `submission_svd-soup-last5.zip` | 3666.3 MB | SVD-merge of steps {200, 209, 221, 234, 246} |
| `submission_svd-wise-50-150-246.zip` | 3665.8 MB | SVD-merge of steps {50, 150, 246} |
| `checkpoint-246_loss0.0036_lr8.13e-07.zip` | 9884.2 MB | full resume checkpoint |

Adapter saves on disk: [50, 100, 150, 197, 200, 209, 221, 234, 246]
