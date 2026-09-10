#!/usr/bin/env python3
"""Pre-flight import check."""
import sys
print(f'Python: {sys.version}')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
import mamba_ssm; print(f'mamba_ssm: {mamba_ssm.__version__}')
import peft; print(f'peft: {peft.__version__}')
from peft.tuners.lora import Linear as LoraLinear; print('LoraLinear: OK')
from safetensors.torch import load_file, save_file; print('safetensors: OK')
import swanlab; print(f'swanlab: {swanlab.__version__}')
from trl import SFTTrainer, SFTConfig; print('trl SFTTrainer: OK')
from datasets import Dataset; print('datasets: OK')
print('\nAll imports OK')
