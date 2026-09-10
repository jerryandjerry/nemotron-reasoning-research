# DPO distillation — seed: SFT 0.85 cryptnumeq_outproj

**Run:** `260531_1319_dpo_distill_alldata_v2_fixtie` (Chicago time)
**Finished:** 2026-05-31 23:19 CDT  (training elapsed: 98.6 min = 1.64 hrs)
**GPU:** RTX PRO 6000 Blackwell (95 GB), single
**Peak VRAM:** 87.47 GB (matched stress test; 7.5 GB headroom)

## Method

GroupDPO gradient-decomposition (arXiv 2604.15602) — single-GPU DPO of a 30B-A3B MoE model with a 95 GB cap, no second model in VRAM:

- **No TRL DPOTrainer, no accelerate, no HF Trainer.** Manual loop based on SFT 0.85.
- **Phase 1 (no grad):** 2 `inference_mode` forwards on (chosen, rejected) to get current policy log-probs.
- **Phase 2 (off-graph):** `s = sigmoid(-β × Δ)` where `Δ = (logp_c − ref_c) − (logp_r − ref_r)`.
- **Phase 3 (with grad):** 2 forwards, each followed by an *immediate* backward at coefficient `±β·s`. One autograd graph live at a time.
- **Reference logp** precomputed once at startup under `bf16` autocast and cached on CPU. Both Phase 1 and Phase 3 forwards run under matching `bf16` autocast (mismatch caused a spurious step-0 gradient in earlier tests).
- **Unsloth smart-offloaded gradient checkpointing** stays wired through the entire forward chain — was the root-cause of the 46-test "memory wall" when TRL's DPOTrainer bypassed it.
- **Apple `cut_cross_entropy.linear_cross_entropy`** for the CE head (avoids materializing the (B,T,V) logits).
- **LoRA params cast to fp32**, base in bf16 (SFT 0.85 recipe verbatim).

## Config

| | |
|---|---|
| Model | Nemotron-3-Nano-30B-A3B (bf16) |
| Seed adapter | SFT 0.85 `cryptnumeq_outproj` (`/root/autodl-tmp/sft_seed_cryptnumeq`) |
| LoRA rank / α / dropout | 32 / 32 / 0.0 |
| Target modules | q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, in_proj, out_proj, lm_head, + MoE expert gate_up_proj/down_proj |
| Trainable params | 888.2 M / 32.5 B |
| MoE tying | ON (`fixtie`) |
| Max seq len | 8192 (chosen + rejected each capped at 8192) |
| Batch size / micro batch | 16 / 4 → 4 micro-steps per optimizer step |
| Optimizer | paged_adamw_8bit |
| Learning rate | 5e-7 (Tülu-3 DPO default) |
| LR schedule | cosine with 3% warmup |
| Max grad norm | 1.0 |
| β | 0.05 |
| Epochs | 1 |
| Total steps | 90 |
| Save full ckpt every | 25 steps, FIFO=∞ (all 4 kept on disk this run) |
| Adapter-only saves | [18, 54, 72, 76, 81, 86, 90] |
| Seed (shuffle) | 42 |

## Data

- **Source CSVs (on instance):**
  - PREFER: `/root/autodl-tmp/data/prefer_260527_huikang_NumericEq.csv` (chosen = `solver_cot`, 6387 rows)
  - REJECT: `/root/autodl-tmp/data/reject_numericeq.csv` (rejected = `raw_output`, 6402 rows)
- **Joined on `id`:** 6429 pairs (the script applies a slightly different join than the local 6360 — uses inner-join with all-id matches).
- **Post-filter applied at runtime:** **1439 pairs** kept. Anchor categories at 10% (`EASY_CATEGORIES_VAL_GE_98 = {numeral, gravity, unit_conversion, cipher, bit_manipulation}`); the rest (cryptarithm + equation_numeric) kept fully.
- **Tokens:** chosen 4,055,875  /  rejected 4,453,192 (sum 8.5M).

## Results

Per-batch DPO loss bounces 0.58 ↔ 0.81 (single-batch noise across 16-pair samples). Aggregates over the 90 optimizer steps:

| Window | Mean loss | Reward gap (chosen − rejected) | Chosen-wins rate |
|---|---|---|---|
| Steps 1–30   | 0.7000 | +0.017 | 60 % |
| Steps 31–60  | 0.6817 | +0.056 | 67 % |
| **Steps 61–90** | **0.6639** | **+0.101** | **80 %** |
| **Last 20**     | **0.6646** | **+0.102** | **85 %** |

- Mean loss across all 90 steps **0.6819 < ln(2) = 0.6931** → preference signal learned.
- Reward gap grew **6×** monotonically.
- Win-rate climbed **60 % → 85 %** in the back third.
- Both `chosen_r` (+0.18) and `rejected_r` (+0.08) end positive — the policy has shifted from the reference, not collapsed.

### Saved checkpoints (all four submitted to Kaggle)

| Checkpoint | Step | Single-batch loss at save |
|---|---|---|
| `checkpoint-25_loss0.6873` | 25 | 0.6873 |
| `checkpoint-50_loss0.5792` | 50 | **0.5792** (lowest) |
| `checkpoint-75_loss0.6998` | 75 | 0.6998 |
| `checkpoint-90_loss0.7173` | 90 | 0.7173 |

The loss column is one batch's reading at save time — *not* a reliable ranking. The aggregate trend says step 90 is the best policy; the four submissions test that empirically.

## Artifacts in this folder (`260531_2319_submission/`)

- `submission_ckpt{25,50,75,90}_loss{x}.zip` — Kaggle-ready adapters (adapter_config.json + adapter_model.safetensors), one per checkpoint. **Download from AutoDL file manager** at `/root/autodl-tmp/260531_2319_submission/`; the data disk persists across shutdown.
- `train_dpo_distill_v2.py` — the training script (manual loop, no TRL)
- `train_log.txt` — full training log
- `requirements.txt` — instance pip freeze (209 pkgs)
- `TRAINING_INFO.md` — this file

## Notes / lessons recorded to memory

- **TRL DPOTrainer + Unsloth smart GC is broken** — the outer-forward monkey-patch is severed when DPOTrainer reaches in. Manual loop preserves it. (`project_unsloth_smart_gc_bypass.md`)
- **bf16 autocast must match** between Phase 1 and Phase 3 — bf16/fp32 mismatch creates a spurious step-0 gradient. (`project_dpo_diagnostic_findings.md`)
- **Stress-test peak == real-training peak.** The 46-test "30 GB activation wall" was anomalous; per-token cost actually matches SFT. (`feedback_validate_load_bearing_numbers.md`)
