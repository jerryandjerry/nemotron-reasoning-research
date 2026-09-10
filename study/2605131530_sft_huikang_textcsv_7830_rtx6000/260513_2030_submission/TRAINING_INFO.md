# Training Info — huikang 0.85 config + text CSV 7830

## Config
- **Method:** LoRA bf16 + CCE + MoE weight tying
- **Model:** Nemotron-3-Nano-30B-A3B bf16 (full, not 4-bit)
- **GPU:** NVIDIA RTX PRO 6000 Blackwell Server Edition (95 GB)
- **Instance:** SSH port 21578

## LoRA
- Rank: 32, Alpha: 32, Dropout: 0
- Targets: q/k/v/o_proj, up/down_proj, in/out_proj, lm_head (9 modules)
- LoRA params in fp32, base model bf16

## Training
- Optimizer: AdamW, beta2=0.95, weight_decay=0
- LR: 2e-4 → 0 (linear decay over 245 steps)
- Batch: 32 (micro=4, 8 accum steps)
- Epochs: 1
- Max seq len: 8192
- Stratified batching (categories distributed evenly across batches)

## Data
- 7,830 samples from huikang's winning 0.85 run (exact match)
- 6,171 unique reasoning problems + 1,659 oversampled duplicates
- 9 categories: bit_manipulation, cipher, unit_conversion, gravity, numeral, equation_numeric_deduce, cryptarithm_deduce, cryptarithm_guess, equation_numeric_guess
- Text CSV tokenized on the fly (not pre-tokenized)
- Source: `01_data/260513_huikang_cot/huikang_7830.csv`

## Results
- Steps: 245
- Final loss: 0.002
- Train time: 203.7 min (3.40 hrs)
- Peak VRAM: ~88.8 GB (smi)
- Checkpoints: 50, 100, 150, 200, 245

## Files on instance (/root/autodl-tmp/)
- submission.zip (1.3 GB)
- checkpoint-245_loss0.0020_lr8.16e-07.zip (1.3 GB)
- adapter_output_huikang_textcsv/ (adapter + all checkpoints)
