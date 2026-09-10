# VRAM Stress Test Report v2
**Date:** 2026-04-10
**GPU:** NVIDIA GeForce RTX 5090 (31.4 GB / 32607 MiB)
**Model:** Nemotron-3-Nano-30B-A3B-bnb-4bit (pre-quantized BNB NF4)
**LoRA:** rank=32, alpha=32, target=all-linear
**Optimizer tested:** adamw_torch (fp32 states)
**Gradient checkpointing:** enabled in config but NOT used in v3 manual loop (see Caveats)

## Test Setup

### v1 (flawed)
- Ran 1 step per sample at 5 different token lengths
- Measured `torch.cuda.max_memory_allocated()` instead of nvidia-smi
- **Result: Useless.** torch.allocated ≠ real GPU usage. Reported 27.68G peak but nvidia-smi showed 30+G.

### v2 (incomplete)
- Used SFTTrainer with nvidia-smi, but only checked after each step
- OOM'd at step 2, only 2 data points
- **Result: Showed OOM happens but not when or why.**

### v3 (most detailed, but no gradient checkpointing)
- Manual training loop with nvidia-smi at every phase: before_forward, after_forward, after_backward, after_del_outputs, before_optim_step, after_optim_step
- Used only 2000-2500 token samples (20 samples), sorted by length
- batch_size=1, grad_accum=4
- **CAVEAT: Manual loop does NOT use gradient checkpointing. Real SFTTrainer does. So v3 peak VRAM is HIGHER than real training.**

## Baseline Memory (all tests consistent)

| Phase | nvidia-smi | torch_alloc |
|-------|-----------|-------------|
| Model load | 18433 MiB (18.0G) | 17.50G |
| After LoRA | 24547 MiB (24.0G) | 22.12G |
| After optimizer creation | 23307 MiB (22.8G) | 22.12G |

**Overhead: nvidia-smi is ~2G higher than torch_alloc** (CUDA context, cuDNN, Triton JIT, reserved pool)

## v3 Results: Phase-by-Phase VRAM (nvidia-smi MiB)

### Clean data (sample 0 only — all subsequent samples contaminated by OOM)

**Sample 0: 2015 tokens, micro_batch 0/4**

| Phase | nvidia-smi | torch_alloc | torch_reserved |
|-------|-----------|-------------|----------------|
| after empty_cache | 23307 | 22.12G | 22.19G |
| before forward | 23307 | 22.12G | 22.19G |
| after forward | 26627 | 25.18G | 25.42G |
| **after backward** | **29723** | **26.39G** | **28.43G** |
| after del outputs | 29723 | 25.36G | 28.43G |

**Forward pass spike:** +3320 MiB (23307 → 26627)
**Backward pass spike:** +3096 MiB (26627 → 29723)
**Total peak:** 29723 MiB = **29.0G** (headroom: 2.4G)

### OOM pattern during gradient accumulation

| Sample | Tokens | Micro batch | OOM phase | nvidia-smi at OOM |
|--------|--------|-------------|-----------|-------------------|
| 0 | 2015 | 0/4 | - | 29723 (OK) |
| **1** | **2017** | **1/4** | **backward** | **31263** |
| 2 | 2056 | 2/4 | - | 30373 (OK, after OOM recovery) |
| **3** | **2070** | **3/4** | **backward** | **31257** |
| 4 | 2079 | 0/4 | - | 30121 (OK, new accum step) |
| **5** | **2087** | **1/4** | **backward** | **31351** |
| 6 | 2129 | 2/4 | **optim_step** | **32107** |
| 7+ | 2135+ | any | **forward** | 32075 (never recovers) |

### Key finding: OOM on every ODD micro-batch

- micro_batch 0: starts at 23.3G, peaks at 29.7G → **OK** (2.4G headroom)
- micro_batch 1: starts at ~26.5G (held gradients from micro 0), peaks at ~31.3G → **OOM** (exceeds 31.4G)

The ~3G difference between micro 0 and micro 1 baseline matches fp32 gradient accumulation:
- 883M LoRA trainable params × 4 bytes (fp32 gradients) = **3.5G**

## Caveats — Why This Test May Be Wrong

**CRITICAL: The manual loop does NOT use gradient checkpointing.**

The real SFTTrainer with `gradient_checkpointing=True`:
- Discards intermediate activations during forward pass
- Recomputes them during backward pass
- Uses LESS memory during backward (trades compute for memory)

The manual loop keeps ALL activations in memory during backward. So:
- v3's "after_forward" VRAM is correct (same activations either way)
- v3's "after_backward" VRAM is **HIGHER than reality** — gradient checkpointing would reduce this

**This means the OOM at micro_batch 1 might NOT happen in real training.** The real training with `adamw_torch` crashed at step 135 (not step 1), which suggests gradient checkpointing reduces backward VRAM enough for most samples, but not for the specific sample at step 135.

## What We Still Don't Know

1. What is the real peak VRAM during backward WITH gradient checkpointing?
2. What specific sample hits at step 135 in the shuffled training order?
3. Is the OOM from the backward pass or from accelerate's `convert_to_fp32` logit upcasting?
4. What is peak VRAM at 2500 tokens with gradient checkpointing?

## Comparison: adamw_torch vs adamw_8bit

| | adamw_torch | adamw_8bit |
|---|---|---|
| Optimizer states | 883M × 2 × 4 bytes = 7.1G | 883M × 2 × 1 byte = 1.8G |
| **Savings** | - | **~5.3G** |
| Baseline after LoRA+optim | ~24G | ~22G |
| Completed 741 steps? | No (OOM at 135) | **Yes** |

## The 0.72 Reference

The original 0.72 script ran on Kaggle's RTX PRO 6000 Blackwell (95GB VRAM) with bf16 full-precision model (~60GB). They had ~35GB headroom. VRAM was never a constraint for them.

## Conclusion

With 32GB VRAM on RTX 5090:
- `adamw_8bit` works (proven: 2 completed runs)
- `adamw_torch` OOMs at step 135 — likely a 2000+ token sample during gradient accumulation
- The stress test is inconclusive because it didn't use gradient checkpointing
- A proper stress test must use SFTTrainer (which enables gradient checkpointing) with nvidia-smi logging

## Next Steps

1. Add nvidia-smi logging to the real training callback (not just torch.allocated)
2. Run with `adamw_torch` + nvidia-smi logging to see the real peak at step 135
3. If it OOMs, the fix is either:
   - Use `adamw_8bit` (proven, ~5G savings from optimizer states)
   - Cap max_seq_len to ~1800 (avoid the long sample spike)
   - Reduce grad_accum from 4 to 2 (less accumulated gradients)
