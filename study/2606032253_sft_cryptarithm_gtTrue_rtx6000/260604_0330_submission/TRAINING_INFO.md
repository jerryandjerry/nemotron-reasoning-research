# Training Info: 260604_0330 — Cryptarithm gtTrue FRESH SFT (with weighted SVD soups)

## Data
- CSV: 260601_Cryptarithm_gtTrue.csv (sources: pretok_0408_decoded 5545 + 260601_new_solver 896)
- All GT-match=True
- 7077 raw -> 6441 unique -> 7756 expanded

## Config (copied verbatim from cryptnumeq SFT 0.85)
- Base: nvidia/Nemotron-3-Nano-30B-A3B (bf16)
- LoRA: r=32, alpha=32, dropout=0
- Targets: q/k/v/o/up/down/in/out_proj + lm_head, no-tie
- LIVE out_proj (USE_MEM_EFF=0 + mixer.training=False)
- bf16 + CCE + AdamW(0.9, 0.95) + wd=0
- LR: 2e-4 -> 0 linear
- Batch 32 (micro 4, ga 8), seq 8192
- Steps: 243 (epoch 1, stratified seed=42)
- Soup adapter saves: [49, 146, 194, 207, 219, 231, 243]

## Training run
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (95 GB)
- Instance: AutoDL 21578
- Wall time: 4.09 hrs (245.1 min)
- Peak VRAM: 72,191 MB
- Loss: 0.534 -> 0.0013 (clean monotonic descent)
- grad_norm range: 0.005 - 0.21 (no spikes, max < 1.0)
- Routine §1.6 out_proj check (stress step 1): grad_norm sum 0.1478 > 0 (LoRA live)

## Soup adapters (saved during training)
- adapter-49_loss0.0201   (20%)
- adapter-146_loss0.0051  (60%)
- adapter-194_loss0.0085  (80%)
- adapter-207_loss0.0141  (85%)
- adapter-219_loss0.0077  (90%)
- adapter-231_loss0.0149  (95%)
- adapter-243_loss0.0013  (100%, final)

## Soup builds (weighted SVD, svd_soup_weighted.py md5 51c8045e)
- **last5-soup** (3.85 GB): steps [194, 207, 219, 231, 243], weights [0.125, 0.125, 0.125, 0.125, 0.5]
- **wise-soup**  (3.85 GB): steps [49, 146, 243], weights [0.1, 0.2, 0.7]
- Both CRC OK

## Files (this folder)
- submission_cryptarithm_gtTrue_notie_outproj.zip — final-step adapter (also on instance)
- submission_last5-soup.zip (3.85 GB) — Kaggle-ready weighted soup
- submission_wise-soup.zip  (3.85 GB) — Kaggle-ready weighted soup
- train_log.txt
- soup_build_log.txt
- requirements.txt
- train_cryptarithm_gtTrue.py
- test_imports.py
- svd_soup_weighted.py
- (checkpoint-243*.zip kept on instance — user downloads from file manager if needed)

## SwanLab
- https://swanlab.cn/@jerry4083/260410_Nemotron/runs/gk6jnjikvq3sra11cs8yp
