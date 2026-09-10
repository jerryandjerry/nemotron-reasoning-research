"""Pre-flight import check.
Imports every package the training script uses and prints versions.
Catches missing deps / ABI mismatches / arch incompatibility before stress test."""
import sys, types

# Stub mamba3 modules just like the training script does
for _mod_name in [
    'mamba_ssm.modules.mamba3',
    'mamba_ssm.ops.cute',
    'mamba_ssm.ops.cute.mamba3',
    'mamba_ssm.ops.cute.mamba3.mamba3_step_fn',
]:
    _m = types.ModuleType(_mod_name); _m.__path__ = []; _m.__package__ = _mod_name
    sys.modules[_mod_name] = _m
sys.modules['mamba_ssm.modules.mamba3'].Mamba3 = None

import torch;                 print(f'torch:            {torch.__version__}')
import unsloth;               print(f'unsloth:          {unsloth.__version__}')
import mamba_ssm;             print(f'mamba_ssm:        {mamba_ssm.__version__}')
from cut_cross_entropy import linear_cross_entropy
print(f'cut_cross_entropy: OK')
from peft import LoraConfig
from peft.tuners.lora import Linear as LoraLinear
print(f'peft:             OK')
from safetensors.torch import load_file, save_file
print(f'safetensors:      OK')
from tokenizers import Tokenizer as RawTokenizer
print(f'tokenizers:       OK')
import swanlab;               print(f'swanlab:          {swanlab.__version__}')
import csv, re, json, logging, zipfile, gc, subprocess, math
print(f'stdlib:           OK')

print(f'\nGPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print('\nALL IMPORTS OK')
