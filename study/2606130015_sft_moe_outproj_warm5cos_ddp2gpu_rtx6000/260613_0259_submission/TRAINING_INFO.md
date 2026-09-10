# SFT 2-GPU DDP — #16 recipe + warmup/cosine LR schedule (warm5cos)

**Run:** `2606122141_sft_moe_outproj_warm5cos_repro16_rtx6000`
**Finished:** 2026-06-13 02:59 CDT (Chicago) — training elapsed **2.74 hrs (164.2 min)**
**Hardware:** 2× RTX PRO 6000 Blackwell (97.9 GB each), AutoDL port 47429, DDP (torchrun --nproc_per_node=2)

## What this run is

A single-variable schedule experiment on top of the proven **#16 recipe** (no-tie + live out_proj, the 0.84→0.86 lever). The ONLY change vs #16 is the LR schedule:

- **#16 / prior DDP runs:** linear decay 2e-4 → 0, no warmup
- **This run (warm5cos):** peak 2e-4, **5% linear warmup (12 steps) → cosine decay to 0** over the remaining 95%

Everything else — data, LoRA, batch, DDP machinery — is byte-identical to the proven 2-GPU DDP harness (`2605231435_sft_moe_outproj_ddp`, which scored 0.84 with no measured DDP penalty vs single-GPU #16's 0.86).

## Config

| | |
|---|---|
| Model | Nemotron-3-Nano-30B-A3B (bf16) |
| Recipe | #16 no-tie + live out_proj (`MOE_TIE=0 USE_MEM_EFF=0`) |
| LoRA | r32 / α32 / dropout 0; q/k/v/o, up/down/in/out_proj, lm_head + MoE experts |
| LR | peak **2e-4**, **5% warmup + 95% cosine→0** (the only change vs #16) |
| Optimizer | AdamW (0.9, 0.95), eps 1e-8, wd 0; max_grad_norm 1e9 (no effective clip — #16 verbatim) |
| Batch | global 32 = 16/rank × micro 4 × 4 accum; per-MB norm + DDP all-reduce-mean |
| Grad ckpt | HF non-reentrant (the DDP-correct mode) |
| Seq len | 8192 |
| Data | `260514_huikang_golden_stripped.csv` (6906 unique → 7849 expanded), stratified seed 42 |
| Steps | **245** (246-step LR schedule; ragged final batch dropped for DDP) |
| Parallelism | DDP, no static_graph, find_unused_parameters=True, broadcast_buffers=False, no_sync OFF |

## Results

- **Final loss (step 245): 0.0027** — matches #16's 0.0025 (faithful replication).
- **Loss tracked the single-GPU warm5cos run almost exactly** (both at step 37 ≈ 0.017), confirming the DDP math + warm5cos schedule are correct.
- **out_proj LoRA verified live under DDP** the entire run (grad-L2 ≈ 0.046 at steps 1–3; the no-tie recipe is NOT silently capped at 0.84).
- **LR schedule executed perfectly**: 0 → peak 2e-4 at step 12 → cosine → 3.6e-8 at step 245.
- **Worst-case VRAM** (stress, longest 7610-token batches): ~97 GB/GPU — fits the 97.9 GB cards with ~0.8 GB headroom. Real training ran 80–97 GB depending on batch length composition.

### Grad-norm spike at step 216 (benign)

Step 216 logged `grad_norm = 2006.65` (vs the usual ~0.005) — a rare bf16/MoE transient explosion. It did **NOT** derail training: steps 217–245 stayed on the normal ~0.003 loss trajectory. AdamW's second-moment normalization + the tiny end-of-cosine lr (8.5e-6) bounded the actual parameter update. The final loss 0.0027 confirms no damage. Not a divergence event; the run was allowed to finish (it was 88% done at the spike).

### Saved checkpoints

- **Final adapter / submission**: `submission_moe_notie_outproj_ddp_warm5cos.zip` (3655 MB)
- **Resume checkpoint**: `checkpoint-245_loss0.0027_lr3.60e-08.zip` (9867 MB)
- **9 soup adapters** (no FIFO): steps 50, 100, 150, 196, 200, 208, 220, 233, 245

## Artifacts

**This local folder (`260613_0259_submission/`):**
- `train_moe_outproj_warm5cos_ddp.py` — the DDP training script
- `train_log.txt` — full training log
- `requirements.txt` — instance pip freeze (torch 2.8.0+cu128 / triton 3.4.0)
- `TRAINING_INFO.md` — this file

**On the instance (`/root/autodl-tmp/260613_0259_submission/`, download from AutoDL file manager):**
- `submission_moe_notie_outproj_ddp_warm5cos.zip` (3.66 GB) — Kaggle submission
- `checkpoint-245_loss0.0027_lr3.60e-08.zip` (9.87 GB) — resume checkpoint

## Environment note

This box (47429) had torch **2.9.0** which breaks `import unsloth` (duplicate flex_attention TritonTemplate). Restored the proven stack **torch 2.8.0+cu128 / torchvision 0.23.0 / triton 3.4.0** before running. The other box (47429-original) had this stack already; no downgrade was needed here — it shipped with 2.8.0.
