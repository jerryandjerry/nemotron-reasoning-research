# Training Info — huikang 0408 text CSV (decoded from pre-tokenized)

## Config
- Model: Nemotron-3-Nano-30B-A3B bf16 (full, not 4-bit)
- LoRA: r=32, alpha=32, dropout=0, 9 targets + lm_head
- Optimizer: AdamW, lr=2e-4 linear decay, beta2=0.95
- Batch: 32 (micro=4, 8 accum)
- Data: 7,830 text CSV decoded from 04-08 pre-tokenized data, huikang's exact order from index.jsonl
- Tokenizer: raw Tokenizer from tokenizer.json for completion (NOT AutoTokenizer)
- Steps: 245
- Max seq len: 8192
- Techniques: CCE, MoE weight tying, fp32 LoRA, Mamba fast path

## Results
- Final loss: 0.002
- Train time: 3.95 hrs (236.8 min)
- Peak VRAM: 88.9 GB / 95 GB
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
- Token verification: PASSED (5/5 samples match pre-tokenized data exactly)

## Files
- submission.zip + checkpoint-245.zip in /root/autodl-tmp/0408_result/
- train_log.txt saved locally

## Purpose
- A/B test: text CSV with correct 0408 data vs pre-tokenized (0.84)
- Previous text CSV run (0413 data) scored 0.81 due to wrong text content
- This run uses decoded 0408 text, same tokenizer, same training order — should match 0.84
