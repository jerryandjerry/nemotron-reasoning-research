# Parallel Training Schemes — Analysis for Nemotron-3-Nano-30B-A3B QLoRA
**Date:** 2026-04-09
**Hardware:** 2x RTX 4090 (24GB each, PCIe 4.0), 1x RTX 5090 (32GB, PCIe 5.0)
**Model:** Nemotron-3-Nano-30B-A3B (30B total, 3.5B active, hybrid Mamba2/Attention/MoE, 52 layers, 128 experts)
**Training:** QLoRA with pre-quantized BNB NF4 (~17.5GB VRAM for model)

---

## Memory Components During Training

| Component | Symbol | Size (QLoRA) |
|-----------|--------|-------------|
| Model parameters (4-bit) | P | 17.5 GB |
| LoRA weights | W_lora | ~0.1 GB |
| Gradients (LoRA only) | G | ~0.1 GB |
| Optimizer states (LoRA only) | O | ~0.2 GB |
| Activations (batch_size=1, seq=2500) | A | ~6-8 GB |
| **Total** | | **~24-26 GB** |

Single RTX 4090 (24GB) = OOM. Single RTX 5090 (32GB) = fits with ~6GB headroom.

---

## Scheme-by-Scheme Analysis

### 1. DP (DataParallel)
- **How:** Full model replicated. GPU 0 collects all gradients (bottleneck).
- **Memory per GPU:** P + G + O + A (no savings)
- **Comm:** 5 sequential transfers through GPU 0
- **For us:** Useless. No memory savings, GPU 0 bottleneck.

### 2. DDP (DistributedDataParallel)
- **How:** Full model replicated. Gradients synced via ring all-reduce (no single bottleneck).
- **Memory per GPU:** P + G + O + A (no savings)
- **Comm:** 1 all-reduce per step. For LoRA gradients (~100MB) = ~3ms on PCIe (negligible).
- **For us:** Failed — each GPU needs full 17.5GB model + ~8GB activations = ~25.5GB > 24GB.
- **Note:** DDP communication itself is cheap for LoRA. The problem is purely memory.

### 3. DeepSpeed ZeRO Stage 0
- Just DDP. No optimization.

### 4. DeepSpeed ZeRO Stage 1 — Optimizer Partitioning
- **How:** Partition optimizer states across GPUs. Full model + full gradients per GPU.
- **Memory per GPU:** P + G + O/N + A
- **For us:** Optimizer states are tiny for LoRA (~0.2GB). Savings negligible. Still OOM.

### 5. DeepSpeed ZeRO Stage 2 — Optimizer + Gradient Partitioning
- **How:** Partition both optimizer states and gradients.
- **Memory per GPU:** P + G/N + O/N + A
- **For us:** Gradient savings negligible for LoRA. P (17.5GB) still not partitioned. Still OOM.

### 6. DeepSpeed ZeRO Stage 3 / FSDP — Full Partitioning
- **How:** Partition everything — parameters, gradients, optimizer states. All-gather full layer before compute, discard after.
- **Memory per GPU:** P/N + G/N + O/N + A
- **For 2x 4090:** ~8.75GB model + ~8GB activations = ~17GB. **FITS in 24GB.**
- **Comm:** 104 all-gather ops per step (52 layers × 2 for fwd+bwd). ~17GB total over PCIe.
- **On PCIe 4.0:** ~500ms overhead, but 2x throughput gain → net ~16s/sample vs 35s/sample pipeline.
- **Key requirement for BNB:** `bnb_4bit_quant_storage=torch.bfloat16` enables FSDP to shard Params4bit.
- **Risk:** Custom trust_remote_code Mamba2 architecture may not work with FSDP wrapping.

### 7. Tensor Parallel (Megatron-LM / vLLM)
- **How:** Split each layer's weight matrix column-wise across GPUs. All-reduce after each matmul.
- **Memory per GPU:** P/T + G/T + O/T + A (full activations)
- **Comm:** 2 all-reduces per layer × 52 layers = 104 all-reduces. ~500ms each on PCIe = **52 seconds pure comm.**
- **For us:** Terrible on PCIe. Also incompatible with BNB Params4bit (can't split quantized tensors column-wise). Designed for NVLink.

### 8. Sequence Parallel (DeepSpeed Ulysses, Ring Attention)
- **How:** Split sequence dimension across GPUs. All-to-all to redistribute Q,K,V for attention.
- **Memory per GPU:** P + G + O + A/S (only activation savings)
- **For us:** **Fundamentally incompatible with Mamba2.** SSM layers are recurrent — splitting the sequence breaks state propagation. Also, 92% of layers are Mamba2 (not attention), so negligible savings even if it worked. Requires transformers>=5.0 (we use 4.56.2).

### 9. Pipeline Parallel
- **How:** Split model by layer groups. GPU 0 = layers 0-25, GPU 1 = layers 26-51. Sequential flow.
- **Memory per GPU:** P/N + G/N + O/N + A_microbatches
- **For 2x 4090:** 10.7GB + 11.5GB. **Works — proven.**
- **Comm:** Only activation tensors at layer boundary. ~50MB per step (minimal).
- **Downside:** Pipeline bubble — one GPU idles while the other computes. ~25% slower than single GPU. No throughput gain (serial processing).
- **Speed:** ~35s/step (proven on our 4090 setup).

### 10. Expert Parallel (MoE-specific)
- **How:** Distribute 128 experts across GPUs. All-to-all to route tokens to correct expert.
- **Memory:** Expert params split, but dense layers (attention, shared expert) replicated.
- **Comm:** 2 all-to-all per MoE layer × 23 MoE layers = 46 all-to-all ops. ~700ms on PCIe.
- **For us:** Not supported in HF/PEFT/TRL ecosystem. Requires custom training framework (Megatron-LM, NeMo).

---

## Comparison Table

| Scheme | Fits 2x 4090? | Throughput vs Single GPU | Comm per Step | Practical? |
|--------|:---:|:---:|---|:---:|
| DP | No | - | Very high | No |
| DDP | No | 2x | ~3ms (LoRA) | No (OOM) |
| ZeRO-1 | No | 2x | ~3ms | No (OOM) |
| ZeRO-2 | No | 2x | ~3ms | No (OOM) |
| **ZeRO-3/FSDP** | **Yes** | **2x** | **~500ms** | **Try this** |
| Tensor Parallel | No | 1x | ~52 seconds | No (PCIe) |
| Sequence Parallel | No | 1x | N/A | No (Mamba2) |
| **Pipeline Parallel** | **Yes** | **1x** | **~2ms** | **Proven** |
| Expert Parallel | No | - | ~700ms | No (ecosystem) |

---

## FSDP vs Pipeline Parallel — Head-to-Head

| | Pipeline Parallel (proven) | FSDP (to test) |
|---|---|---|
| Model memory per GPU | ~10-11 GB | ~8.75 GB |
| Free for activations | ~13 GB | ~15 GB |
| Communication per step | ~50 MB (1 boundary) | ~17 GB (104 all-gathers) |
| GPU utilization | ~75% (bubble) | ~100% |
| Effective throughput | 1x | 2x (minus comm) |
| Estimated speed | 35s/step (proven) | **~16s/step (estimated)** |
| Net time for 741 steps | ~7.2 hours | **~3.3 hours (estimated)** |

---

## Unsloth — Alternative Approach

Unsloth officially supports Nemotron-3-Nano-30B-A3B with:
- **30-50% VRAM reduction** via fused Triton kernels
- **Fused Linear+CrossEntropy**: 84% less peak memory at logit layer
- **CPU activation offloading**: Only 1.9% compute overhead (vs HF's 20-30%)
- **Custom MoE Triton kernels**: 2.5x faster
- **4-bit QLoRA on single RTX 4090**: ~24GB (tight but fits)
- **Multi-GPU via DDP** or `device_map="balanced"`

Unsloth's gradient checkpointing (`"unsloth"` mode) offloads activations to CPU asynchronously, which could solve the activation memory problem on 24GB GPUs without any parallelism scheme.

### Known Issues:
- Merge errors with MoE/Mamba layers during adapter export
- vLLM export incompatibility with Mamba `mixer` layers
- Requires `mamba_ssm` and `causal_conv1d` packages

---

## Recommended Strategy

1. **5090 (32GB single GPU):** Current approach works. ~5.5 hrs per run. Good for production runs.
2. **4090 2x — FSDP + QLoRA:** Test next. If it works, ~3.3 hrs per run with 2x throughput.
3. **4090 2x — Pipeline Parallel:** Proven fallback. ~7 hrs per run.
4. **4090 — Unsloth single GPU:** Worth testing. Could fit on single 24GB with fused kernels.
5. **Both instances in parallel:** Run different data/hyperparameter experiments simultaneously.

---

## FSDP + QLoRA Setup (Answer.AI Method)

Key configuration:
```python
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,  # CRITICAL for FSDP sharding
)
```

Launch: `accelerate launch --fsdp --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP train.py`

Requirement: ~128-200GB system RAM for CPU offloading.

---

## FSDP Experiment Log (2026-04-09/10)

### Attempts on 2x RTX 4090 (24GB each)

**Attempt 1: FSDP via SFTConfig `fsdp='full_shard auto_wrap'`**
- Error: `ValueError: Must flatten tensors with uniform dtype but got torch.float32 and torch.bfloat16`
- Cause: Model has mixed fp32 (layer norms, Mamba D/A_log/dt_bias) and bf16 (quantized weights)
- FSDP requires all parameters in a wrapped module to have the same dtype

**Attempt 2: Cast all fp32 params to bf16 before FSDP wrapping**
- Model loaded successfully — 15.6GB per GPU (sharded correctly!)
- Error at training step 1: `KeyError: Parameter containing: tensor(...)` in `_exec_order_utils.py`
- Cause: PEFT/LoRA adds parameters after FSDP wrapping, breaking FSDP v1's parameter-to-name mapping

**Attempt 3: `accelerate launch` with FSDP config + `fsdp_use_orig_params: true`**
- Model loaded successfully — 15.6GB per GPU
- Hung at step 0 for 8+ minutes, no progress
- NCCL debug showed no errors — likely deadlock in all-gather during first forward pass
- Cause: Custom Mamba2/Attention hybrid model with trust_remote_code may have non-standard forward pass that breaks FSDP's all-gather/discard cycle

### Conclusion
FSDP + QLoRA + Nemotron-3-Nano-30B (custom Mamba2/Attention MoE hybrid) is NOT viable with current tooling:
1. Mixed dtype → requires manual casting
2. PEFT + FSDP v1 → KeyError on parameter registration
3. FSDP + custom trust_remote_code → hangs during forward pass
4. Answer.AI's FSDP+QLoRA was demonstrated on standard Llama architecture, NOT hybrid Mamba2/MoE models

### Next step: Try Unsloth (designed specifically for this model)

Sources:
- [Answer.AI FSDP-QLoRA](https://www.answer.ai/posts/2024-03-06-fsdp-qlora.html)
- [bitsandbytes FSDP docs](https://huggingface.co/docs/bitsandbytes/fsdp_qlora)
- [HF Accelerate SP docs](https://huggingface.co/docs/accelerate/en/concept_guides/sequence_parallelism)
- [Unsloth Nemotron docs](https://unsloth.ai/docs/models/nemotron-3)

---

## DDP for weighted SFT loss (2026-05-15)

DDP averages **gradients** across ranks: `grad = (grad_0 + grad_1 + ...) / world_size`.
This equals "gradient of mean loss" only when the loss is a uniform mean across the batch.

For SFT with completion-only masking and variable sequence lengths:

```
loss = sum(per_token_ce * mask) / sum(mask)
```

The denominator varies per batch (completion lengths differ). If rank 0 has 2000 completion
tokens and rank 1 has 6000, naive DDP gives `(grad_0 + grad_1) / 2` — biased toward the rank
with fewer tokens (each of its tokens carries more weight).

### Fix (HF PR #34191, Oct 2024)

```python
local_n = (labels != -100).sum()
global_n = local_n.clone()
dist.all_reduce(global_n, op=dist.ReduceOp.SUM)
loss = ce_sum_local / global_n * world_size   # *world_size cancels DDP's /world_size
loss.backward()
# After DDP all-reduce: grad = sum_r(ce_sum_r) / global_n   ← identical to single-GPU
```

HF Trainer plumbs this via `average_tokens_across_devices=True` (default since v4.46).
TRL inherits it. axolotl, llama-factory, unsloth converged on the same pattern. For custom
training loops, the fix must be added manually.

Sources:
- [HF blog: Fixing Gradient Accumulation](https://huggingface.co/blog/gradient_accumulation)
- [HF PR #34191](https://github.com/huggingface/transformers/pull/34191)
- [HF issue #34242](https://github.com/huggingface/transformers/issues/34242)
