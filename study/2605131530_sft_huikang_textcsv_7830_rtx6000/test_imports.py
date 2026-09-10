#!/usr/bin/env python3
"""Pre-flight import check — catches missing deps before training starts."""
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

import unsloth
print(f'unsloth: {unsloth.__version__}')

import mamba_ssm
print(f'mamba_ssm: {mamba_ssm.__version__}')

from cut_cross_entropy import linear_cross_entropy
print(f'cut_cross_entropy: OK')

import peft
print(f'peft: {peft.__version__}')

from safetensors.torch import load_file, save_file
print(f'safetensors: OK')

import swanlab
print(f'swanlab: {swanlab.__version__}')

import csv, re, json, zipfile, math, gc, subprocess, types, logging
print(f'stdlib: OK')

print('\nAll imports OK')
