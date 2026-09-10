#!/usr/bin/env python3
"""Pre-flight: import every package train_resume_086.py uses, print versions.
Per routine §1 — catches missing deps / ABI mismatches before the stress test."""
import os, sys
os.environ['TRANSFORMERS_NO_TF']    = '1'
os.environ['TRANSFORMERS_NO_FLAX']  = '1'
os.environ['HF_HUB_OFFLINE']        = '1'
os.environ['TRANSFORMERS_OFFLINE']  = '1'

print(f'python: {sys.version.split()[0]}')
print(f'platform: {sys.platform}')

import unsloth
print(f'unsloth: {unsloth.__version__}')
from unsloth import FastLanguageModel
print('  unsloth.FastLanguageModel: OK')

# Stub mamba3 modules (matches the trainer's setup)
import types
for _mod_name in [
    'mamba_ssm.modules.mamba3',
    'mamba_ssm.ops.cute',
    'mamba_ssm.ops.cute.mamba3',
    'mamba_ssm.ops.cute.mamba3.mamba3_step_fn',
]:
    _m = types.ModuleType(_mod_name); _m.__path__ = []; _m.__package__ = _mod_name
    sys.modules[_mod_name] = _m
sys.modules['mamba_ssm.modules.mamba3'].Mamba3 = None
import mamba_ssm
print(f'mamba_ssm: {mamba_ssm.__version__}')

import torch
print(f'torch: {torch.__version__}')
print(f'  cuda available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  cuda devices: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'    {i}: {torch.cuda.get_device_name(i)}  '
              f'{torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')

from cut_cross_entropy import linear_cross_entropy
print('cut_cross_entropy.linear_cross_entropy: OK')

import peft
print(f'peft: {peft.__version__}')
from peft import LoraConfig
from peft.tuners.lora import Linear as LoraLinear
print('  peft.LoraConfig: OK')
print('  peft.tuners.lora.Linear: OK')

import safetensors
print(f'safetensors: {safetensors.__version__}')
from safetensors.torch import load_file, save_file
print('  safetensors.torch.{load_file,save_file}: OK')

import tokenizers
print(f'tokenizers: {tokenizers.__version__}')
from tokenizers import Tokenizer as RawTokenizer
print('  tokenizers.Tokenizer: OK')

try:
    import swanlab
    print(f'swanlab: {swanlab.__version__}')
except Exception as e:
    print(f'swanlab: NOT AVAILABLE ({type(e).__name__}: {e})  -- non-fatal, trainer falls back')

import transformers
print(f'transformers: {transformers.__version__}')

print('\nALL IMPORTS OK')
