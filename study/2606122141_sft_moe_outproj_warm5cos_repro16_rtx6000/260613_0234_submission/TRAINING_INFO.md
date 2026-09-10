# TRAINING_INFO — warm5cos: #16 reproduction with 5% warmup + cosine decay

- **Study folder:** `2606122141_sft_moe_outproj_warm5cos_repro16_rtx6000`
- **Submission folder:** `260613_0234_submission` (training finished 2026-06-13 ~02:30 Chicago)
- **Run:** swanlab `260410_Nemotron/runs/9mo0hhs34h9e1f5o6fhm2`

## What was trained

Single-variable LR-schedule experiment: reproduce run #16 (the first *trained* 0.86: no-tie + live out_proj) with **ONE change** — the LR schedule is **5% linear warmup → cosine decay to 0 over the remaining 95%** (was: linear decay 2e-4→0, no warmup). Everything else byte-identical to `train_moe_outproj.py` (#16).

## Config

- Trainer: copy of #16's `train_moe_outproj.py`; diff = LR schedule (12-step warmup then cosine, HF `get_cosine_schedule_with_warmup` semantics num_cycles=0.5) + a STRESS-only out_proj-alive print + naming tags. Training math otherwise identical to #16. (See `train_moe_outproj_warm5cos.py`.)
- Data: `260514_huikang_golden_stripped.csv` (md5 `92a515f31cd2e89f9cb355c4ed09f927`) — **exact #16 data**, 6906 unique → 7849 expanded → 246 steps, stratified seed=42.
- LoRA r=32 α=32 dropout=0; targets q/k/v/o_proj, up/down_proj, in_proj, out_proj, lm_head; MOE_TIE=0 (no-tie), live out_proj (mixer.training=False), CCE, random init.
- LR **peak 2e-4**, warmup 12 steps (0→2e-4), cosine 234 steps (2e-4→~0). bf16, AdamW(0.9,0.95) wd=0, bs32 (micro4 ga8), seq8192.
- Env `MOE_TIE=0 USE_MEM_EFF=0 FRESH_START=1`. GPU RTX PRO 6000 96 GB (port 21578).

## Result

- **Wall:** 4.07 hr (244.3 min). Peak VRAM 71.9 GB (script-reported); per-step smi ~88.9 GB.
- **Final train loss: 0.003592** (step 246) — vs #16's 0.0030 (~1.2×, essentially the same converged loss).
- **Step-67 grad spike:** grad_norm 3936 (vs ~0.01 typical), loss stayed normal (0.0040), recovered next step (grad 0.0114). Known non-deterministic bf16/MoE transient; AdamW absorbed it; not a derail. After it the loss bumped to a ~0.009 plateau (steps ~70–100) then resolved back to #16's level as the cosine lr decayed.
- **vs #16 trajectory (byte-identical except schedule):** step-1 loss IDENTICAL (0.403614, same init+data). warm5cos then lagged ~2× through the 12-step warmup, was further bumped by the step-67 spike (~0.009 plateau), and reconverged to #16 (~1.0–1.2×) by step ~150 as the cosine decayed. Grad norms matched #16 throughout once warmed up.
- **0 derailment.** 9 soup adapters saved [50,100,150,197,200,209,221,234,246]; losses 0.0064/0.0071/0.0049/0.0038/0.0042/0.0040/0.0035/0.0047/0.0036.

## Artifacts

Instance `/root/autodl-tmp/260613_0234_submission/` (download via AutoDL file manager):
- `submission_moe_notie_outproj_warm5cos.zip` (3.83 GB) — final step-246 adapter, Kaggle-ready
- `checkpoint-246_loss0.0036_lr9.01e-09.zip` (10.38 GB) — full resume checkpoint

Local (this folder): `train_moe_outproj_warm5cos.py`, `train_log.txt`, `requirements.txt`, this file.

## Hypothesis / expectation

Per the LR-schedule literature review (2026-06-12), schedule shape (cosine vs linear, ±warmup) is within-noise once converged; final train loss here (0.0036) ≈ #16 (0.0030) supports that. So warm5cos is expected to land at ~0.86 (same as #16), not above — the schedule is not predicted to be the lever that breaks the wall. The step-67 spike is a confound on a perfectly clean schedule-only comparison, but it recovered. Awaiting LB.
