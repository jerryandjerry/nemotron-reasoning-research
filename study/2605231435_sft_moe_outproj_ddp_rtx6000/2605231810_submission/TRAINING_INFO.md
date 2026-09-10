# Training Info — DDP 2-GPU replication of #16 (no-tie + live out_proj)

**Study folder:** `2605231435_sft_moe_outproj_ddp_rtx6000`
**Finished:** 2026-05-23 18:10 Chicago (2026-05-24 07:10 CST instance time)
**Goal:** Replicate run #16 (0.86 — first trained adapter to clear 0.84) using 2-GPU DDP. Config + data byte-identical to #16; only parallelization changes.

## Model
- Nemotron-3-Nano-30B-A3B (bf16), hybrid Mamba2 + attention + MoE, 52 layers, 128 routed experts.
- `MODEL_PATH=/root/autodl-tmp/Nemotron-3-Nano-30B-A3B`

## LoRA config (identical to #16)
- rank=32, alpha=32, dropout=0.0
- target_modules: q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, in_proj, **out_proj**, lm_head
- **No MoE tie** (`MOE_TIE=0`) — routed experts trained as independent rank-32 LoRA
- **Live out_proj** (`USE_MEM_EFF=0`) — each Mamba mixer `.training=False` forces the unfused else-branch so the `out_proj` LoRA module trains (the 0.84→0.86 lever in #16)

## Optimizer / schedule
- AdamW, lr 2e-4 → 0 linear, betas (0.9, 0.95), weight_decay 0
- Batch 32 global (micro 4), seq 8192
- Stratified batching, seed 42
- save_steps 50; FIFO=1 checkpoints + soup adapter saves
- CCE (Cut Cross-Entropy) + manual lm_head LoRA

## DDP specifics
- 2× RTX PRO 6000 Blackwell (96 GB each), NCCL backend
- Per-rank batch 16 (contiguous split rank0=[0:16], rank1=[16:32])
- HF **non-reentrant** gradient checkpointing (`use_reentrant=False`) — the DDP-correct ckpt
- `find_unused_parameters=True`, `broadcast_buffers=False`, no static_graph
- per-MB loss normalization; `(loss/n_accum).backward()`; DDP all-reduce-mean == single-GPU /8

## Steps / data
- Data: `260514_huikang_golden_stripped.csv` (md5 92a515f31cd2e89f9cb355c4ed09f927), 6906 unique → 7849 expanded, 27,989,496 tokens (unmasked 26,713,988) — identical to #16.
- **245 steps** (dropped the ragged final 9-sample batch; `LR_DENOM=246` preserves #16's exact LR schedule). 1 epoch.

## Result
- Final loss **0.0025** (step 245). Loss trajectory matched #16: ckpt-50 0.0053 / ckpt-100 0.0044 / ckpt-200 0.0030 / step-245 0.0025.
- out_proj liveness verified under DDP: OUT_PROJ.lora_B grad-L2 = 0.046/0.034/0.022 (steps 1-3, nonzero).

## Run history (this study)
- **Attempt 1:** ran 245 steps cleanly, then **NCCL collective-timeout crash at the ragged step-246** — under DDP the 9-sample tail split left rank-1 empty → collective mismatch.
- **Fix:** drop ragged final batch (245 full-32 batches) + resume from checkpoint-200. Verified resume: step-201 loss 0.0032 (converged), LR continuous.
- **Closeout:** submission zip created + CRC-verified (`testzip()=None`). `checkpoint-245.zip` FAILED — disk hit 100% (`OSError 28`) during the optional checkpoint backup. Submission artifact unaffected.

## Artifacts
- `submission.zip` — final-246 endpoint adapter (adapter_config.json + adapter_model.safetensors, ~3.83 GB). **The 0.86 replication target.**
- `checkpoint-245.zip` — NOT created (disk full at closeout).
- Soups (svd-soup-last5 / svd-wise) — NOT built (disk full).

## Kaggle
- Pending submission. Submit `submission.zip` (fileName MUST be `submission.zip`) to nvidia-nemotron-model-reasoning-challenge.
