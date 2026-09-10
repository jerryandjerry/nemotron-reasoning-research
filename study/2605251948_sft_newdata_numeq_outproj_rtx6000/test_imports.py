#!/usr/bin/env python3
"""Pre-flight: import every package the trainer uses + print versions. Catches missing deps / ABI mismatch fast."""
import sys
print('python', sys.version.split()[0])
import unsloth  # must import first (patches transformers)
print('unsloth', unsloth.__version__)
for m in ['torch', 'transformers', 'mamba_ssm', 'cut_cross_entropy', 'peft', 'safetensors', 'tokenizers', 'swanlab']:
    try:
        mod = __import__(m)
        print(f'OK  {m}', getattr(mod, '__version__', '?'))
    except Exception as e:
        print(f'FAIL {m}: {type(e).__name__}: {e}')
import torch
print('CUDA', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')
print('transformers MUST be 4.56.2 for the proven recipe')
