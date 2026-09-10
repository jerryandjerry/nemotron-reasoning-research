# Training Info — QLoRA 4-bit huikang pretokenized

## Config
- Model: Nemotron-3-Nano-30B-A3B **4-bit quantized**
- LoRA: r=32, alpha=32, dropout=0, 9 targets + lm_head
- Optimizer: AdamW, lr=2e-4 linear decay, beta2=0.95
- Batch: 32 (micro=16, 2 accum)
- Data: 7,830 pre-tokenized (huikang exact), stratified batching
- Steps: 245
- Max seq len: 8192
- Techniques: CCE, MoE weight tying, fp32 LoRA, Mamba fast path

## Results
- Final loss: 0.002
- Train time: 4.1 hrs (245.5 min)
- Peak VRAM: 86 GB / 95 GB
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition

## Files
- submission.zip + checkpoint-245.zip in /root/autodl-tmp/4bit_result/
- train_log.txt was deleted during cleanup (not saved)

## Notes
- Disk full error during initial zip creation — intermediate checkpoints deleted, zips recreated
- Same config as 0.84 bf16 run except 4-bit quantized base model
- Purpose: A/B test of 4-bit vs bf16 quantization impact on score
