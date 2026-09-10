# SFT-step vs DPO-step Memory Diagnostic — Nemotron-3-Nano-30B-A3B

**Date:** 2026-05-31
**GPU:** RTX PRO 6000 Blackwell (95 GB cap)
**Goal:** Determine whether DPO at T=4096 actually exceeds memory, or whether the 30 GB activation cost observed in 46 failed stress tests was self-inflicted by production-code deviations from the proven SFT recipe.

## Method

Two standalone scripts run in **separate Python processes** (clean teardown between them). **Identical setup** in both, copied verbatim from the SFT 0.85 cryptnumeq_outproj script:

- `FastLanguageModel.from_pretrained(MODEL, dtype=bfloat16, attn_implementation='eager', trust_remote_code=True)`
- `FastLanguageModel.get_peft_model(model, r=32, lora_alpha=32, target_modules=[q,k,v,o,up,down,in,out,lm_head], use_gradient_checkpointing='unsloth')`
- LoRA params cast to fp32: `for n, p in model.named_parameters(): if '.lora_' in n: p.data = p.data.to(float32)`
- `FastLanguageModel.for_training(model)` — installs Unsloth's smart-offloaded GC autograd Function
- Apple `cut_cross_entropy.linear_cross_entropy` patched onto `_base.forward` with SFT's composition `lm_weight = base_w + scaling * lora_B @ lora_A`
- `torch.optim.AdamW(trainable_params, lr=2e-4)`

**The only difference:** DPO does **two forward+backward passes** (Phase 1 inference_mode prelude + Phase 3 chosen + Phase 3 rejected, the GroupDPO gradient-decomposition shape) where SFT does **one**. Same model, same LoRA, same patches, same optimizer.

**Inputs:** random `torch.long` tokens, B=1, T=2048. (T=4096 would force `get_peft_model`'s internal test forward to OOM at startup; the bucket comparison is what matters, scales linearly to T=4096 within smart-GC bounds.)

Memory marks (`torch.cuda.memory_allocated()` + `memory_reserved()`) printed after every phase. Memory snapshots (`torch.cuda.memory._dump_snapshot`) saved after forward / backward / step.

## Result — every phase, both runs

| Phase | SFT allocated | DPO allocated | Match |
|---|---|---|---|
| 0 — after base model load | 63.27 G | 63.27 G | ✓ |
| 1 — after `get_peft_model` | 66.83 G | 66.83 G | ✓ (+3.56 G LoRA adapter) |
| 2 — after LoRA fp32 cast | 66.83 G | 66.83 G | ✓ (in-place cast, no alloc) |
| 3 — after `for_training()` | 66.83 G | 66.83 G | ✓ (sets flags + installs smart-GC hook) |
| 4 — after CCE patch applied | 66.83 G | 66.83 G | ✓ (just reassigns `_base.forward`) |
| 5 — after optimizer create | 66.83 G | 66.83 G | ✓ (state lazy-allocated on first `.step()`) |
| 6 — after input build | 66.83 G | 66.83 G | ✓ (input_ids = (1, 2048) long ≈ 16 KB) |
| 7a — DPO Phase 1 chosen | n/a | 66.84 G | inference_mode forward fully releases |
| 7b — DPO Phase 1 rejected | n/a | 66.84 G | same |
| **forward** | **68.32 G** | **68.32 G** | ✓ **identical — forward activation cost = 1.49 G** |
| **backward** | **70.41 G** | **70.41 G** | ✓ **identical — .grad of 888 M fp32 = 2.09 G** |
| DPO Phase 3 rejected forward | n/a | 71.89 G | +1.48 G — same per-branch forward cost, chosen's .grad alive |
| DPO Phase 3 rejected backward | n/a | 70.43 G | graph released; chosen+rejected summed .grad alive |
| **optimizer.step** | **73.94 G** | **73.96 G** | ✓ **identical — AdamW fp32 m1+m2 state = +3.53 G** |

**SFT peak: 73.94 G. DPO peak: 73.96 G. Delta: 20 MB.**

## Per-token / per-component breakdown

| Component | SFT contribution | DPO contribution | Notes |
|---|---|---|---|
| Base model weights (bf16) | 63.27 G | 63.27 G | ~32.5 B params, mostly stored as 5888 × 10 MB blocks |
| LoRA adapter (fp32 after cast) | 3.56 G | 3.56 G | 888 M params × ~4 bytes |
| Forward activations @ T=2048 | 1.49 G | 1.49 G | ~0.73 MB/token (smart GC active) |
| `.grad` of trainable (fp32) | 2.09 G | 2.09 G | 888 M × 4 B = ~3.5 G expected; partial allocation observed |
| AdamW m1+m2 state (fp32) | 3.53 G | 3.53 G | Lazy-allocated on first `.step()` |
| **Peak (after optimizer step)** | **73.94 G** | **73.96 G** | Difference is within rounding |

## Conclusion

**The DPO two-forward gradient-decomposition shape is NOT the memory wall.** At T=2048 with smart GC active, DPO uses 20 MB more than SFT, both far under the 95 GB cap.

### Anchoring to actual production operating points

The diagnostic ran at B=1, T=2048 = 2K tokens per forward. The actual production scripts run at:

| Script | B | T | tokens/forward | Production peak (observed) | Per-token cost |
|---|---|---|---|---|---|
| SFT 0.85 (`train_cryptnumeq_outproj.py`, MAX_SEQ_LEN=8192, MICRO_BATCH_SIZE=4) | 4 | 7970 (data max, truncated from 8192) | **~32 K** | ~86 GB (smi 88105 M) | ~0.75 MB/token activations |
| DPO production (`train_dpo_distill.py`, my SequentialDPOTrainer slices to B=1) | 1 | 4096 | **~4 K** | ~92 GB at Phase 3 chosen forward (stress test) | ~7.5 MB/token claimed (anomalous) |

**SFT does 8× more tokens per forward than DPO and produces a smaller activation footprint.** The diagnostic's 0.73 MB/token at T=2048 matches SFT's ~0.75 MB/token at T=7970 — i.e., the architecture's actual cost per token is constant when smart GC is firing.

For DPO at T=4096 B=1 with smart GC firing, expected activation cost:
- forward: ~3 GB (4096 × 0.73 MB)
- step peak: ~75-77 GB
- free headroom: ~18-20 GB

**The 30 GB activation cost observed in 46 failed production stress tests is 10× the expected value and cannot come from the DPO architecture itself.** It must come from production-code differences from this SFT-verbatim diagnostic.

## Production deviations to isolate next

The production `train_dpo_distill.py` differs from this clean diagnostic in six ways. The 30 GB wall is in one or more of them:

1. **`PeftModel.from_pretrained(SEED_ADAPTER, is_trainable=True)`** — production loads the SFT 0.85 trained adapter; diagnostic builds fresh LoRA via `get_peft_model`. Production also calls `model.gradient_checkpointing_enable(...)` + `model.base_model.model.backbone.gradient_checkpointing_enable(...)` **after** `for_training()`, which may overwrite/disable Unsloth's smart-GC autograd Function with HF's standard `torch.utils.checkpoint`.
2. **`torch.addmm` CCE patch** instead of SFT's `base_w + scaling * lora_B @ lora_A`. Functionally equivalent but `addmm` requires uniform dtype (fails outside bf16 autocast).
3. **`out_proj-live` Mamba mixer pre-hook** that sets `mixer.training = False` on every forward call, forcing `cuda_kernels_forward` into the unfused else-branch where Mamba LoRA receives gradients but memory profile is different.
4. **`is_fast_path_available = True`** patched into `modeling_nemotron_h` — forces Mamba dispatch through `cuda_kernels_forward`.
5. **TRL DPOTrainer + accelerate wrapping** — `precompute_ref_log_probs` phase runs 96 forwards before training_step; accelerate wraps the optimizer/forward in autocast; HF Trainer's training_step calls model.train() / .eval() at various points.
6. **`paged_adamw_8bit`** instead of `torch.optim.AdamW` — adds bitsandbytes paging overhead, different state layout.

**Recommended next diagnostic step:** start from this SFT-verbatim DPO script and add each deviation ONE AT A TIME, snapshotting after each addition. The deviation that takes the forward activation cost from ~1.5 GB to ~30 GB is the bug. Strong prior on #1 (the extra `gradient_checkpointing_enable` calls overwriting Unsloth's smart-GC hook).

## Resolution — `train_dpo_distill_v2.py` works

Rewrote the DPO script from scratch by copying SFT 0.85 (`train_cryptnumeq_outproj.py`) verbatim and adding the minimum DPO delta:

- Dataset loader for chosen+rejected pairs (PREFER + REJECT CSVs joined on `id`)
- Reference logp precompute at startup (one pass through all pairs, cached on CPU)
- Three-forward gradient decomposition per pair: Phase 1 inference_mode forwards to get current policy logp, off-graph `s = sigmoid(-β·δ)`, Phase 3 with-grad forward + immediate backward with coefficient ±β·s
- No TRL DPOTrainer. No accelerate. No HF Trainer. Manual `for step: for pair in batch: forward → backward; opt.step()` loop, identical to SFT's structure.

**Stress test result (B=16 pairs, T=8192, 3 steps):**
```
=== Starting DPO training ===
>> step=1/3  loss=0.6931  chosen_r=-0.0000  rejected_r=-0.0000  grad_norm=72.61  lr=5.00e-07  smi_after=78809M
>> step=2/3  loss=0.6866  chosen_r=+0.0142  rejected_r=-0.0401  grad_norm=65.21  lr=3.33e-07  smi_after=78865M
>> step=3/3  loss=0.6649  chosen_r=-0.0189  rejected_r=-0.2196  grad_norm=67.14  lr=1.67e-07  smi_after=78813M
Training done. Time: 0.13 hrs (7.5 min)
Peak smi: 78813M
STRESS TEST COMPLETE
```

- **Peak GPU: 78.8 GB** (16 GB headroom under 95 GB cap)
- Loss starts at 0.6931 (textbook `ln(2)` initial value) and descends correctly: 0.6931 → 0.6866 → 0.6649
- Implicit rewards diverge the right way: chosen relatively higher than rejected each step
- Smart GC fires throughout (`Unsloth: Will smartly offload gradients to save VRAM!` printed at first forward and never resets)
- Gradient norm stable (65-72), no exploding
- Time: ~2.5 min per optimizer step at B=16, T=8192 → real run ~166 steps × 2.5 min ≈ 7 hours + 80 min precompute

**Two subtle bugs found and fixed during stress test development:**
1. Off-by-one shape mismatch in `_seq_logp`: SFT's CCE patch expects pre-shifted `(input=ids[:-1], target=ids[1:], weight=mask[1:])` tensors. My initial code passed full `ids` as both input and labels — needed to pre-shift like SFT's data prep does.
2. `_seq_logp` called `model.eval()` for the precompute pass, which set `self.training=False` on every submodule. Unsloth's smart GC autograd Function only fires when `self.training=True` — without resetting back to `model.train()`, the rest of the training-step forwards lost smart GC and forward activations ballooned to ~30 GB, OOMing at the MoE expert ReLU². Removed the `model.eval()` call; `torch.inference_mode()` alone is enough to avoid building an autograd graph, and the user-controlled out_proj-live flag still works.

**Both bugs were a microcosm of the entire 46-test campaign:** they came from deviating from SFT's exact recipe (off-by-one) and breaking smart GC by introducing state SFT doesn't have (model.eval()).

The wall is gone.

## Routine-compliant stress test (2026-05-31, bit_manipulation throttled to 10%)

Added the missing TRAINING_ROUTINE.md §2.3/§2.4 features to `train_dpo_distill_v2.py`:
- Per-micro-batch log lines: `sample_id, padded_c, padded_r, smi, chosen_tail, rejected_tail`
- `SAVE_STEPS_FULL = max(25, round(num_steps/5/25)*25)` for full-state checkpoints with FIFO=1
- `ADAPTER_SAVE_STEPS = sorted({round(num_steps*p) for p in (0.20, 0.60, 0.80, 0.85, 0.90, 0.95, 1.00)})` for soup adapters (no FIFO, always saved)
- Resume-from-latest-checkpoint at script start (skip if `STRESS_TEST=1` or `FRESH_START=1`)
- Routine §1 stress rule: sort by `(chosen+rejected)_length` desc, take top-K=`bs × max_steps × 2` (= 96), no shuffle
- Append-mode log file

Updated data filter per user 2026-05-31: added `bit_manipulation` to `EASY_CATEGORIES_VAL_GE_98` so it's also throttled to 10% (was 1354 pairs = 51% of post-filter; now ~135). Projected total: ~1062 pairs → ~66 optimizer steps → ~2.7 hr real training + ~30 min precompute.

**Stress result:**
```
Tokenized: 96 pairs  chosen_toks=516,040  rejected_toks=699,364
Precompute done in 176.0s  smi=67523M
SAVE_STEPS_FULL=25  ADAPTER_SAVE_STEPS=[1, 2, 3]
>> step=1 pair[0] sample_id=d0b1e41a padded_c=7119 padded_r=7772 smi=64475M chosen_tail: 'oxed{#{"!}<|im_end|>'  rejected_tail: 'nds in EX2: ~sub(20,'
[... 47 more micro-batch lines ...]
>> step=1/3  loss=0.6931  chosen_r=-0.0000  rejected_r=-0.0000  grad_norm=68.92  lr=5.00e-07  smi_after=78769M
>> step=2/3  loss=0.7552  chosen_r=+0.0126  rejected_r=+0.0683  grad_norm=74.56  lr=3.33e-07  smi_after=78791M
>> step=3/3  loss=0.6975  chosen_r=+0.0604  rejected_r=+0.0392  grad_norm=64.58  lr=1.67e-07  smi_after=78879M
Training done. Time: 7.7 min
Peak smi: 78879M
STRESS TEST COMPLETE
```

- **Peak GPU: 78.9 GB** (16 GB headroom)
- 48 micro-batches × 3 steps × routine §2.3 logging format ✓
- Resume / FIFO=1 / soup-save scaffolding ✓
- Loss noise at step 2 (0.6931 → 0.7552 → 0.6975) is normal for a 3-step run at bs=16; rewards by step 3 show correct DPO structure (chosen_r=+0.0604, rejected_r=+0.0392)

Ready to launch real training. Estimated time: ~30 min precompute + ~66 steps × 2.5 min/step ≈ **3.2 hours total**.

## Artifacts

- `_diag_sft_step.py` — SFT recipe verbatim, 1 forward+backward+step
- `_diag_dpo_step.py` — SFT recipe verbatim + DPO's two-forward delta only
- `_diag_run_both.py` — launches each in its own process, pulls snapshots back
- `instance_logs/train_step_v2/diag_sft.log`, `diag_dpo.log` — phase-by-phase memory marks
- `instance_logs/train_step_v2/sft/snap_after_{forward,backward,step}.pickle` — SFT snapshots
- `instance_logs/train_step_v2/dpo/snap_after_phase3_{chosen,rejected}_{forward,backward}.pickle` + `snap_after_step.pickle` — DPO snapshots
- `DIAGNOSTIC_VIZ.html` — interactive bar-chart visualization of the per-phase memory profile
