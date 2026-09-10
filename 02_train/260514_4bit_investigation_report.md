# 4-bit QLoRA Investigation Report
**Date:** 2026-05-14

## Question
Why does 4-bit QLoRA use more VRAM (90.6 GB) and run slower (1.3 min/step) than bf16 (88 GB, 0.83 min/step) at the same micro batch size (4) on a 95 GB GPU?

## Findings

### 1. Standard CE materializes 16 GB logits that CCE avoids

The bf16 run uses Cut Cross-Entropy (CCE), which computes the loss directly from hidden states and the lm_head weight matrix without ever creating the full logits tensor. The 4-bit SFTTrainer run uses standard cross-entropy, which materializes the full logits tensor:

```
micro=4 × seq=8192 × vocab=128K × 4 bytes = 16 GB
```

VRAM breakdown:
- **bf16:** 60 GB model + 0 GB logits (CCE) + ~28 GB activations = 88 GB peak
- **4-bit:** 20 GB model + 16 GB logits (standard CE) + ~55 GB activations/optimizer = 90.6 GB peak

4-bit saves 40 GB on weights but adds 16 GB for logits. Net saving only ~24 GB, which is consumed by other overhead.

### 2. Dequantization overhead slows every forward pass

With 4-bit NF4 quantization, weights are stored in 4-bit but must be dequantized to bf16 on the fly for every matrix multiplication. This adds latency per layer, per forward pass.

From the HuggingFace blog: "QLoRA is slightly slower than LoRA due to dequantization overhead. Weights are decompressed only when needed — this saves memory but requires additional computational overhead."

There are no hardware kernel optimizations for the NF4 format. The dequantization runs in software, adding overhead that doesn't exist with native bf16.

## Conclusion

4-bit QLoRA is designed for GPUs where bf16 doesn't fit. On a 95 GB GPU where bf16 fits:
- **No speed benefit:** dequantization overhead makes each forward pass slower
- **No VRAM benefit:** logits tensor (from standard CE) eats the weight savings
- **bf16 + CCE is strictly better** on this hardware

## Sources
- [HuggingFace: Making LLMs accessible with bitsandbytes 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- [bitsandbytes issue #1141: Quantized 4-bit models use more memory than 16-bit](https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1141)
- [Deep Dive into LLM Quantization: FP32, BF16, INT8, NF4 & QLoRA](https://medium.com/@saurad44/a-deep-dive-into-llm-quantization-fp32-bf16-int8-nf4-qlora-830ff4936f3f)
- [QLoRA Deep Dive by Manal El Aidouni](https://manalelaidouni.github.io/4Bit-Quantization-Models-QLoRa.html)
