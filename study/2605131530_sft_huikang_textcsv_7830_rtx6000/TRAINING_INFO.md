# Training Info — huikang 0.85 reproduction with text CSV

## Config
- Model: Nemotron-3-Nano-30B-A3B bf16 (full, not 4-bit)
- LoRA: r=32, alpha=32, dropout=0, 9 targets + lm_head
- Optimizer: AdamW, lr=2e-4 linear decay, beta2=0.95
- Batch: 32 (micro=4, 8 accum)
- Data: 7,830 samples (huikang exact match: 6,171 base + 1,659 oversampled)
- Stratified batching (categories distributed evenly across batches)
- Steps: 245 (7830/32 rounded up)
- Max seq len: 8192
- Techniques: CCE, MoE weight tying (5888 params), fp32 LoRA, Mamba fast path

## Results
- Final loss: 0.002
- Train time: 3.40 hrs (203.7 min)
- Peak VRAM: 88.9 GB / 95 GB
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition

## Checkpoints
- checkpoint-50: loss=0.0099, lr=1.60e-04
- checkpoint-100: loss=0.0023, lr=1.19e-04
- checkpoint-150: loss=0.0020, lr=7.84e-05
- checkpoint-200: loss=0.0021, lr=3.76e-05
- checkpoint-245: loss=0.0020, lr=8.16e-07

## Files
- submission.zip: 1.37 GB (user downloads from AutoDL)
- checkpoint-245_loss0.0020_lr8.16e-07.zip (user downloads from AutoDL)

## Key difference from 0.84 pre-tokenized run
- Uses text CSV tokenized on the fly (AutoTokenizer) instead of pre-tokenized data (raw Tokenizer)
- Added stratified batching matching huikang's _stratified_batches function
- Exact same 7,830 sample IDs as huikang's winning run
