# Training Info: 260607_2057 — bitman_failcorr SFT (#16 base + per-bit ✓/✗ verification)

## Data
- CSV: 260516_bit_manipulation_failcorr.csv (md5 a9c2349c3cc88eddeebbc11709a443fa, 35.0 MB)
- Source = huikang_golden_stripped (#16's data, md5 92a515f31cd2e89f9cb355c4ed09f927) with per-bit ✓/✗
  verification block inserted before "Selected" on 1,280 of 1,354 bit_manipulation rows; 5,552 non-bit_man rows
  pass through byte-identical from huikang_golden_stripped.
- 6,905 unique rows -> 7,847 expanded -> 246 steps at bs=32

## Diff vs #16 (per direct CSV diff, verified):
- 9 categories: identical unique counts except bit_manipulation (-1 row)
- Oversampling distribution: identical except 1 fewer os=2 (dropped bit row)
- Non-bit_man rows: byte-identical CoTs (verified on 5 samples) EXCEPT 1 cryptarithm_deduce row (b1b10e83)
  has a different CoT in failcorr (README says non-bit_man pass-through; this is a minor caveat)
- bit_manipulation rows: ✓/✗ per-bit verification block inserted between rule-analysis and "Selected";
  rest of CoT identical. +~140 tokens per augmented row.

## Config (verbatim copy of cryptnumeq SFT 0.85; only CSV_PATH + _DATA_TAG changed)
- Base: nvidia/Nemotron-3-Nano-30B-A3B (bf16)
- LoRA: r=32, alpha=32, dropout=0
- Targets: q/k/v/o/up/down/in/out_proj + lm_head, no-tie
- LIVE out_proj (USE_MEM_EFF=0 + mixer.training=False)
- bf16 + CCE + AdamW(0.9, 0.95) + wd=0
- LR: 2e-4 -> 0 linear
- Batch 32 (micro 4, ga 8), seq 8192
- Steps: 246 (epoch 1, stratified seed=42)
- Soup adapter saves: [49, 148, 197, 209, 221, 234, 246]

## Training run
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (95 GB)
- Instance: AutoDL 22371
- Wall time: 4.21 hrs (252.9 min)
- Peak VRAM: 72,153 MB
- Loss: 0.450 -> 0.0045 (clean monotonic descent)
- grad_norm range: 0.004 - 0.15 (no spikes)
- Routine §1.6 out_proj check (stress step 1): grad_norm sum 0.1592 > 0 (LoRA live)

## Loss comparison at matching steps
| step | #16 (0.86) | 06-07 (0.83) | bitman_failcorr |
|---|---|---|---|
|  9 | 0.192 | 0.263 | 0.194 |
| 25 | 0.026 | 0.045 | 0.027 |
| 79 | 0.004 | 0.019 | 0.008 |
| ~end | 0.003 | 0.0015 | 0.0045 |

failcorr tracks #16 closely through step 25, then slightly elevated (2x at end) due to the ✓/✗
verification token overhead. Consistent with same data + augmentation hypothesis.

## Soup adapters (saved during training)
- adapter-49_loss0.0092   (20%)
- adapter-148_loss0.0063  (60%)
- adapter-197_loss0.0044  (80%)
- adapter-209_loss0.0052  (85%)
- adapter-221_loss0.0058  (90%)
- adapter-234_loss0.0051  (95%)
- adapter-246_loss0.0045  (100%, final)

## Submission
- ONLY submission_bitman_failcorr_notie_outproj.zip (3.83 GB) per same pattern as 06-07
- No soup build this run; no checkpoint zip in submission folder

## SwanLab
- https://swanlab.cn/@jerry4083/260410_Nemotron/runs/142f0imvnbvmdfexjx0hs
