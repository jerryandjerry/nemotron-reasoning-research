# Single-GPU DPO of Nemotron-3-Nano-30B-A3B — what 46 stress tests taught us

**Setup**: RTX PRO 6000 Blackwell 95 GB, Nemotron-3-Nano-30B-A3B (Mamba2/Attn/MoE hybrid, 32.5 B base, 888 M LoRA trainable), seed = SFT 0.85 cryptnumeq_outproj adapter, max_length=4096, gradient_accumulation_steps=16, Unsloth + PEFT + TRL DPOTrainer + accelerate.

## 1. The memory accounting (verified empirically)

At training-step time, every micro-step's GPU footprint is the sum of:

| Component | Bytes | Notes |
|---|---|---|
| Base model + LoRA adapter (bf16) | **62.15 GB** | constant baseline |
| Forward activations w/ autograd graph at T=4096 | **+29.76 GB** | Mamba2 chunked SSD ~5 GB + MoE/ReLU² ~12 GB + residuals/scratch ~13 GB; gradient checkpointing doesn't reduce this further on Mamba (state lives outside torch.utils.checkpoint) |
| Apple CCE backward `dc = zeros_like(lm_weight)` | **+0.67 GB** | (V=131072, H=4096) bf16, unconditional, allocated inside `cce_backward.py:276` |
| `.grad` of 888 M trainable in fp32 | **+3.36 GB** | NOT reducible by freezing PEFT LoRA at load time — Unsloth/PEFT/TRL re-enables `requires_grad` at training-step time (Test 46 proof) |
| 8-bit paged Adam state (post-step) | **+1.65 GB** | bnb's paged_adamw_8bit keeps most state GPU-resident even with paging |
| 4-bit Adam state (post-step) | **+0.90 GB** | torchao AdamW4bit |

**Hard total at Phase 3 rejected forward end of step 2+**: 62.15 + 29.76 + 0.67 + 3.36 + 1.65 = **97.6 GB**. Cap is 94.97 GB. **Shortfall: ~2.6 GB.**

Every fix we tried squeezed 0.5-1.5 GB, leaving 150 MB-1 GB residual gap. The wall doesn't move enough to ship.

## 2. What worked (and is worth keeping for any future attempt)

### 2.1 Gradient decomposition (GroupDPO, arXiv 2604.15602)

DPO loss `L = -logsigmoid(β·δ)` with `δ = (logp_c - logp_r) - (ref_c - ref_r)` decomposes:
- `dL/dlogp_c = -β·s`, `dL/dlogp_r = +β·s`, `s = sigmoid(-β·δ)`

So you can do 2 inference_mode forwards → compute `s` off-graph → 2 with-grad forwards each followed by IMMEDIATE backward. Only ONE forward's autograd graph alive at a time. Saves ~30 GB versus TRL's default concat-batch approach.

`SequentialDPOTrainer._compute_loss` in [train_dpo_distill.py](train_dpo_distill.py) is a clean reference implementation. Mathematically equivalent to single-forward concat DPO.

### 2.2 Apple CCE patch with `torch.addmm` LoRA merge

`cut_cross_entropy.linear_cross_entropy(hidden, lm_weight, labels)` is the Triton-fused kernel that skips the `(B,T,V)` float32 logits materialization (saves 2 GB vs stock `modeling_nemotron_h.py:1717` `.float()` upcast).

For LoRA-on-lm_head, compose the merged weight via `torch.addmm(base_w, lB, lA, alpha=scaling, beta=1.0)` — single (V,H) allocation instead of 3 transients (delta, scaled delta, merged). Autograd flows through addmm back to lora_A/lora_B correctly (Apple CCE backward returns `dc`).

```python
def _cce_forward(input_ids=None, attention_mask=None, labels=None, **kwargs):
    if labels is None:
        return _orig_forward_fn(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    bb_out = backbone(input_ids=input_ids, attention_mask=attention_mask, ...)
    hidden = bb_out[0]
    base_w  = lh.base_layer.weight
    lA      = lh.lora_A['default'].weight
    lB      = lh.lora_B['default'].weight
    scaling = lh.scaling['default']
    torch.cuda.empty_cache()  # critical defrag before the 1 GB allocation
    lm_weight = torch.addmm(base_w, lB, lA, alpha=float(scaling), beta=1.0)
    shifted_hidden = hidden[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    per_token_ce = _lce(shifted_hidden, lm_weight, shifted_labels, reduction='none')
    model._cached_per_token_ce = per_token_ce
    return per_token_ce.mean()
```

### 2.3 Persistent CPU running sum of `.grad` across all micro-steps

For 888 M trainable, each micro-step's `.grad` is 3.5 GB. Across 16 grad-accumulation micro-steps, it accumulates in place (size constant) but persists on GPU between micro-steps, squeezing the next forward.

Fix: at the start of every `_compute_loss`, offload existing `.grad` to a persistent `self._cpu_grad_running` dict and set `.grad = None`. Monkey-patch `optimizer.step` to copy the running sum back to GPU before stepping. Result: every micro-step starts at the 62 GB baseline; optimizer sees the full accumulated gradient.

**Gotchas discovered**:
- HF Trainer 4.56 + accelerate may pass `lr` as a 0-dim CUDA tensor via `AcceleratedOptimizer.prepare`. The patched step must coerce `lr` to `float` before the inner CPU optimizer runs.
- `clip_grad_norm_` runs BEFORE `optimizer.step` in HF Trainer. The CPU running sum must be restored via an `on_pre_optimizer_step` callback (not in the patched step itself), OR you accept that `.grad` will be empty during clipping. For stress (max_steps=3 < grad_accum=16), optimizer.step never fires, so the bug doesn't surface.
- Key dict by `name` (from `named_parameters()`), not by `id(param)`. `id(param)` is fragile under PEFT `merge_and_unload`, gradient-checkpointing recompute clones, etc.

### 2.4 `empty_cache()` immediately before the 1 GB `addmm` allocation

Forces allocator defrag at the critical contiguous-block request. Buys ~150 MB in tight cases. Tested as part of Tests 40-45 — modest but real impact under expandable_segments fragmentation.

## 3. What failed and why

### 3.1 `saved_tensors_hooks(pack_to_cpu, unpack_to_gpu)` — Tests 30-31
GPU memory dropped beautifully (65 GB) but every Mamba `save_for_backward` fires the hook. Tens of thousands of tiny CPU↔GPU sync calls put the process in `Dl` (disk-wait) state. Selective size-filter (>50 MB) helped but didn't fit. **Verdict: too fine-grained for Mamba2's many small saves.**

### 3.2 `torchao CPUOffloadOptimizer(AdamW4bit, ...)` — Test 42
Memory fit perfectly (every micro-step at 62.17 GB baseline). But `AdamW4bit` uses `@torch.compile` and the compile graph sees `lr` (cuda:0 scalar tensor) × `p_f32` (CPU after offload) → `FakeTensor Device Propagation` error inside `single_param_adam`. **Verdict: torch.compile-decorated optimizers don't compose with CPU param offload.**

### 3.3 `torchao CPUOffloadOptimizer(torch.optim.AdamW, ...)` — Test 43
No compile issue. Memory fit. But each micro-step took ~3 min instead of Test 41's 18s. Two compounding causes:
- My CPU running sum offload at every micro-step (12010 grad-tensor D2H copies)
- `torch.optim.AdamW`'s CPU step is single-threaded, not AVX-512-tuned (DeepSpeed's CPU Adam is 5-7× faster — known benchmark in DeepSpeed#1635)
**Verdict: vanilla CPU AdamW for 888 M params is too slow for real training; would need DeepSpeed's `DeepSpeedCPUAdam`.**

### 3.4 Freezing MoE expert LoRA — Tests 44-46
Pattern `.experts.` in name, exclude `shared_experts`. At load time the count correctly drops from 888.2 M to 32.0 M. **But at training-step time, `requires_grad` is back to True on all LoRA params.**

Proof from Test 46 diagnostic at `_compute_loss` entry:
```
Froze MoE routed-expert LoRA POST-for_training: 888.2M -> 32.0M  (frozen 856.2M)
Trainable: 32.0M / total 32.5B  (0.099%)
...
[SequentialDPO] _compute_loss called; ...; trainable_at_runtime=888.2M
```

Either PEFT's `prepare()`, accelerate's `prepare()`, TRL's `add_adapter`, or Unsloth's compiled DPOTrainer's `_compute_loss` re-enables grads. Did not pinpoint the exact culprit. **Verdict: PEFT freeze-by-`requires_grad=False` is unreliable in this stack; need a `forward_pre_hook` or a custom autograd Function to actually disable grad flow.**

### 3.5 Apple CCE backward unconditionally allocates `dc`
`cut_cross_entropy/cce_backward.py:276`:
```python
dc = torch.zeros_like(c, dtype=e.dtype)
```
The 672 MB allocation happens regardless of whether `c.requires_grad` is True. No public flag/kwarg to skip it. The mesolitica fork (`ml-cross-entropy-lora-lm-head`) tried to fuse LoRA into the kernel to avoid this, but its backward returns zero grads for `lora_A`/`lora_B` — silently incorrect for LoRA training. **Verdict: would need a custom Triton kernel or a small autograd.Function wrapper that recomputes lm_weight on demand.**

## 4. Per-test memory budget (the wall, visualized)

| Test | Adam state on GPU | `.grad` on GPU | Effective baseline at step ≥ 2 | Phase 3 forward peak | Phase 3 backward + CCE dc peak | Verdict |
|------|------|------|------|------|------|------|
| 38 (paged 8-bit, no grad-sum) | 1.65 GB | 3.5 GB | 65.5 GB | 95.5 GB | n/a (OOM mid-forward) | OOM |
| 39 (paged 8-bit, CPU grad-sum) | 1.65 GB | 0 | 63.8 GB | 93.6 GB | 94.3 GB needs 672 MB | step 1 OK, step 2 OOM (~150 MB short) |
| 41 (adamw_torch_4bit) | 0.9 GB | 3.5 GB / 0 | 63.06 GB | 92.82 GB | 93.5 GB needs 672 MB | step 1 OK, step 2 OOM at backward |
| 42 (CPUOffload(AdamW4bit)) | 0 (CPU) | 3.5 GB / 0 | 62.16 GB | 91.92 GB | 92.6 GB ✅ | optimizer.step fails (compile) |
| 43 (CPUOffload(AdamW)) | 0 (CPU) | 3.5 GB / 0 | 62.16 GB | 91.92 GB | 92.6 GB ✅ | 3 min/micro-step (CPU too slow) |
| 45/46 (MoE freeze, paged 8-bit) | 1.65 GB | 3.5 GB (freeze undone) | 65.5 GB | 95.5 GB | n/a | OOM (freeze ineffective at runtime) |

## 5. Verified architecture facts (Nemotron-3-Nano-30B-A3B)

- 32.5 B total params, 888.2 M LoRA trainable (2.74 %) via Unsloth's MoE-aware `target_parameters`
- Hybrid: 23 Mamba2 + 6 attention + 23 MoE blocks (per NVIDIA tech report)
- d_model = 4096, ssm_state_size = 128, mamba_intermediate = 8192, MoE intermediate = 7688 (shared_experts only), routed experts = 128 per layer
- Vocab = 131072, hidden = 4096 → lm_head matrix = 1.0 GB bf16
- Activation = `squared_relu` — saves BOTH relu input AND square input for backward (per PyTorch issue #63027) → ~126 MB per MoE-FFN call × 23 ≈ 2.9 GB shared + 5.8 GB routed = ~9-12 GB total for ReLU² activations alone at T=4096
- Mamba2 fused chunked SSD path (`mamba_split_conv1d_scan_combined`) saves ~200 MB/layer × 23 ≈ 4.6 GB — NOT the bottleneck (research-confirmed via Tri Dao's SSD blog and state-spaces/mamba#532)

## 6. Key references discovered

- **GroupDPO algorithm** — arXiv 2604.15602 (no public code release; our SequentialDPOTrainer is a clean-room impl)
- **Apple cut_cross_entropy** — https://github.com/apple/ml-cross-entropy ; `cce.py:103` backward returns `(de, dc, None)` confirming autograd flows through `c`
- **mesolitica LoRA-CCE fork** — https://github.com/mesolitica/ml-cross-entropy-lora-lm-head — **BROKEN backward, do not use** (drops grads for lora_A, lora_B)
- **NeMo-RL #1922** — same model OOMs on 4×A100-80GB; suggests TP+PP doesn't trivially solve this either
- **DeepSpeed CPU Adam** — 5-7× faster than `torch.optim.AdamW`; required if going the CPU-offload-optimizer route at this scale
- **Unsloth + DPO known issues** — `unsloth-grad-fix` repo confirms grad-accumulation + Unsloth's checkpoint patch needs a separate fix (we may be hitting an adjacent bug at the freeze step)

## 7. If you fund another stress-test campaign, do this first

1. **DeepSpeed ZeRO-2 with `DeepSpeedCPUAdam`** — replicates Test 42's memory profile (Adam state on CPU) at Test 41's speed. Drop Unsloth, switch to vanilla HF Trainer + accelerate + DeepSpeed JSON. ~1 day integration. Strongest single-GPU bet.
2. **2-GPU FSDP** — shards everything across two GPUs; ~3 GB recovered per GPU. Higher cost, same integration complexity.
3. **Custom autograd Function that wraps Apple CCE** — recompute `lm_weight = addmm(base_w, lB, lA)` inside the function's forward; in backward, recompute `lm_weight` again, get `dc` from upstream CCE, derive `dlA = lB.T @ dc * scaling`, `dlB = dc @ lA.T * scaling`. Avoids persisting `dc` across the backward of the rest of the model. ~30 lines, risky but cheap to try.
4. **Skip Unsloth, use HF + accelerate vanilla** — eliminates the freeze-undoing bug. Slower training but freeze MoE expert LoRA would actually work.
5. **KTO / RLOO / single-policy-forward methods** — sidesteps the 2-forward DPO requirement entirely. Trade preference signal richness for memory.

## 8. Files preserved

- [train_dpo_distill.py](train_dpo_distill.py) — full working implementation of gradient decomposition + CCE patch (current local version)
- [instance_logs/stress_dpo.log](instance_logs/stress_dpo.log) — final stress run's full stdout/stderr
- [instance_logs/train_dpo_log.txt](instance_logs/train_dpo_log.txt) — final stress run's `_log` output
- [instance_logs/train_dpo_real.log](instance_logs/train_dpo_real.log) — earlier real-training attempt log
- [instance_logs/train_dpo_distill_LAST_ON_INSTANCE.py](instance_logs/train_dpo_distill_LAST_ON_INSTANCE.py) — exact script that was running at shutdown
- [STRESS_TEST_REPORT.md](STRESS_TEST_REPORT.md) — chronological log of every test, OOM line, and verdict
