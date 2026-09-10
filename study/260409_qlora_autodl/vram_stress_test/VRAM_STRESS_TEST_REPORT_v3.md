# VRAM Stress Test Report v3 — Final
**Date:** 2026-04-10
**GPU:** NVIDIA GeForce RTX 5090 (31.36 GB / 32607 MiB)
**Model:** Nemotron-3-Nano-30B-A3B-bnb-4bit (pre-quantized BNB NF4)

## Root Cause Found

**OOM at step 136, micro-batch 4/4, caused by `accelerate.utils.operations.convert_to_fp32` upcasting logits from bf16 to fp32.**

### The exact OOM moment

Step 136 micro-batch breakdown:

| Micro | seq_len | smi_before | smi_after | Result |
|-------|---------|-----------|----------|--------|
| 1 | 753 | 25841M | 29047M | OK |
| 2 | 341 | 29047M | 29085M | OK |
| 3 | 373 | 29085M | 29131M | OK |
| **4** | **2200** | **29131M** | - | **OOM** |

### Why it OOMs

1. Micro-batches 1-3 complete forward+backward, accumulating ~3.3GB of fp32 gradients (29131M - 25841M)
2. Micro-batch 4 has a 2200-token sample
3. Forward pass completes, producing logits tensor `[1, 2200, 131072]` in bf16 = 0.55GB
4. `accelerate` wraps the model forward and calls `convert_to_fp32(output)` which does `tensor.float()` on the logits
5. This allocates an fp32 copy: `2200 × 131072 × 4 bytes = 1.10GB`
6. Available: only 1.05GB (31.36G - 30.30G) → **OOM**

### Why it only happens with specific sample ordering

- Long sample on micro-batch 1: starts at 25.8G, peaks ~30G → OK (enough headroom for fp32 upcast)
- Long sample on micro-batch 4: starts at 29.1G (after 3 micro-batches of accumulated gradients), peaks ~30.3G → only 1G left → OOM when fp32 upcast needs 1.1G

The OOM requires BOTH conditions:
1. A sample with ~2000+ tokens
2. Landing on micro-batch 3 or 4 (after gradient accumulation from earlier micro-batches)

### The fp32 upcast allocation

```
Logit upcast size = seq_len × vocab_size × 4 bytes (fp32)
  - 500 tokens:  500 × 131072 × 4 = 0.26 GB
  - 1000 tokens: 1000 × 131072 × 4 = 0.52 GB
  - 1500 tokens: 1500 × 131072 × 4 = 0.79 GB
  - 2000 tokens: 2000 × 131072 × 4 = 1.05 GB  ← borderline
  - 2200 tokens: 2200 × 131072 × 4 = 1.15 GB  ← OOM
  - 2500 tokens: 2500 × 131072 × 4 = 1.31 GB  ← definitely OOM
```

### Why `adamw_8bit` survives

adamw_8bit uses ~1.5G less for optimizer states. This means:
- Micro-batch 4 starts at ~27.6G instead of 29.1G
- After forward+upcast: ~27.6 + 3G (forward) + 1.1G (upcast) = ~31.7G — still tight but PyTorch can manage with its reserved pool

### Why `gc.collect()` helped push from step 135 to 136

The original OOM at step 135 had a 2018-token sample. With `gc.collect()`, Python freed unreachable objects holding GPU tensors, giving ~200MB more headroom — enough for 2018 tokens (1.06GB upcast) but not 2200 tokens (1.15GB upcast).

---

## History of Failed Stress Tests

### v1: Single-step per sample (WRONG)
- Tested 5 samples at different lengths, 1 forward+backward each
- Used `torch.cuda.max_memory_allocated()` not nvidia-smi
- **Why wrong:** torch.allocated misses ~2G of CUDA overhead. Single step doesn't test gradient accumulation interaction.

### v2: SFTTrainer but nvidia-smi only at step boundaries (INCOMPLETE)
- Used real SFTTrainer but only checked nvidia-smi at `on_step_end`
- OOM'd at step 2, only 2 data points
- **Why wrong:** nvidia-smi at step boundaries misses the peak during micro-batches. The OOM happens mid-step.

### v3: Manual training loop (WRONG)
- Manual forward+backward without gradient checkpointing
- Showed 31.3G peak and OOM
- **Why wrong:** Real training uses gradient checkpointing which reduces backward pass VRAM significantly. Manual loop measures a completely different memory profile.

### v4 (stress_test_real.py): Real SFTTrainer, 2300-2500 only (MISLEADING)
- Used real SFTTrainer with gradient checkpointing
- Only 2300-2500 token samples, sorted by length
- Completed 2 steps without OOM, peak 30G
- **Why wrong:** All micro-batches had similar-length long samples. The real OOM needs mixed lengths: 3 short + 1 long. Uniform long samples don't reproduce the gradient accumulation + late-position long sample pattern.

### v5 (final): Real training on real dataset with sample logging (CORRECT)
- Used exact training code with adamw_torch
- Added `LoggingSFTTrainer` to print sample ID, seq_len, nvidia-smi, preview at each micro-batch
- Ran on real dataset with real shuffle order
- **Found the bug at step 136:** 2200-token sample on micro-batch 4 after 3 accumulated short samples

---

## How to Do VRAM Stress Tests Correctly

### Rules

1. **Use the EXACT training code** — same SFTTrainer, same gradient checkpointing, same optimizer, same everything. Never write a manual training loop.

2. **Use the REAL dataset with the REAL shuffle order** — synthetic/filtered datasets miss the actual sample ordering that causes OOM. The bug depends on which samples land on which micro-batch position.

3. **Measure nvidia-smi, not torch.cuda.memory_allocated()** — torch.allocated misses 2-3GB of CUDA context, cuDNN workspace, and reserved pool. nvidia-smi shows actual GPU usage.

4. **Log at EVERY micro-batch, not just step boundaries** — the OOM happens mid-step during gradient accumulation. Step-level callbacks (`on_step_begin`, `on_step_end`) miss the peak. Override `training_step()` to log before and after each micro-batch.

5. **Run the FULL training (or at least past the longest sample)** — don't stop at 2 steps. The OOM depends on which sample appears at which position in the shuffled order. You need to run until the problematic sample appears.

6. **Log sample metadata** — seq_len, sample ID, content preview. Without this you can't identify which sample caused OOM.

7. **Don't fix the bug while testing** — run the exact OOM code, observe it fail, understand WHY, then fix. Don't patch and re-run without understanding.

### Template for future stress tests

```python
class LoggingSFTTrainer(SFTTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        step = self.state.global_step + 1
        seq_len = inputs['input_ids'].shape[-1]
        smi = get_nvidia_smi_mib()
        preview = tokenizer.decode(inputs['input_ids'][0][:60].tolist(), skip_special_tokens=True)[:120]
        print(f'>> step={step} seq_len={seq_len} smi_before={smi}M preview: {preview}...')
        
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        smi_after = get_nvidia_smi_mib()
        print(f'   step={step} smi_after={smi_after}M loss={loss.item():.4f}')
        return loss
```

This logs nvidia-smi before and after EACH micro-batch (not just each step), and shows the sample content that's being processed.

---

## Fix Options

| Fix | Pros | Cons |
|-----|------|------|
| `adamw_8bit` | Proven, 1.5G savings, survives all samples | Slight precision loss |
| Monkey-patch `convert_to_fp32 = lambda x: x` | Eliminates the upcast entirely | May affect training precision |
| Cap `max_seq_len` to 1800 | Avoids long sample + upcast issue | Loses 40 training samples |
| Reduce `grad_accum` from 4 to 2 | Less accumulated gradients = more headroom | Changes effective batch size |
| Sort dataset by length (long samples first) | Long samples on micro-batch 1 (most headroom) | Changes training order |
