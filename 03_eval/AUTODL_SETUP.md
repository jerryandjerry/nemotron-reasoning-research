# AutoDL Inference Instance Setup

## Instance Info

- Provider: AutoDL (seetacloud.com)
- SSH credentials: see `kaggle.env` (SSH_LOGIN_*_INF entries)
- SSH password: see `kaggle.env` (SSH_PASSWORD)
- Connect via paramiko (sshpass not available on Windows)

## Setup from Scratch

```bash
# 1. Install vllm (pulls torch==2.10.0 automatically)
/root/miniconda3/bin/pip install vllm pandas

# 2. Upload val CSV
# (via paramiko sftp from local)
# Local: 01_data/train_10-90_split/260407_val.csv
# Remote: /root/autodl-tmp/260407_val.csv

# 3. Upload eval script
# Local: 03_eval/eval_autodl_vllm.py
# Remote: /root/autodl-tmp/eval_autodl_vllm.py
```

## Setup from Cloned Instance

If the instance is cloned from another that had older torch/vllm:

```bash
# MUST do clean install — in-place upgrades cause torch inductor conflicts
/root/miniconda3/bin/pip uninstall -y torch torchvision torchaudio vllm
rm -rf /root/miniconda3/lib/python3.12/site-packages/torch*
rm -rf /root/miniconda3/lib/python3.12/site-packages/vllm*
rm -rf /root/.cache/torch /root/.cache/vllm /root/.triton
/root/miniconda3/bin/pip install vllm pandas
```

## Known Issues

### "duplicate template name" / "Model architectures failed to be inspected"
- Cause: Stale torch artifacts from a previous torch version (e.g. 2.8) left behind after upgrade to 2.10
- Fix: Full clean uninstall + reinstall (see above)

### BNB 4-bit inference speed
- BNB dequantizes weights on every forward pass → slower than native fp16/bf16
- ~3 tok/s per sample on RTX 5090 (32GB), similar on RTX PRO 6000 (98GB)
- VRAM doesn't help speed — compute is the bottleneck
- 950 samples with max_tokens=7680 can take several hours

## Running Eval

```bash
# Test run (6 samples, 1 per category) — always do this first
TEST_MODE=1 /root/miniconda3/bin/python /root/autodl-tmp/eval_autodl_vllm.py

# Full run (950 samples, saves every 10)
/root/miniconda3/bin/python /root/autodl-tmp/eval_autodl_vllm.py

# Background run
nohup /root/miniconda3/bin/python /root/autodl-tmp/eval_autodl_vllm.py > /root/autodl-tmp/eval.log 2>&1 &
```

## Config in eval_autodl_vllm.py

Edit these at the top of the script per run:
```python
MODEL_PATH = "/root/autodl-tmp/<model-name>"
MODEL_NAME = "<model-name>"
OUTPUT_DIR = "/root/autodl-tmp/YYMMDD_<name>_950val"
```

## File Locations

```
/root/autodl-tmp/
├── 260407_val.csv              # Val set (950 samples)
├── eval_autodl_vllm.py         # Eval script
├── <model-name>/               # Model weights
└── YYMMDD_<name>_950val/       # Output dir
    ├── val_eval_results.csv    # Full results (or partial checkpoint)
    ├── test_run_results.csv    # Test run results (6 samples)
    └── eval_summary.json      # Summary stats
```
