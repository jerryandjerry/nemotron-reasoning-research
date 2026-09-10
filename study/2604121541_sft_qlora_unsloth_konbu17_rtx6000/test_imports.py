"""Quick sanity check: every package the Unsloth QLoRA training needs imports cleanly.

Unsloth must be imported before transformers — do it first before anything else.
"""
import unsloth
print(f'unsloth: {unsloth.__version__}')
from unsloth import FastLanguageModel
print('FastLanguageModel OK')

import torch
print(f'torch: {torch.__version__}')
print(f'cuda : {torch.version.cuda}')
print(f'archs: {torch.cuda.get_arch_list()}')
print(f'cap  : {torch.cuda.get_device_capability(0)}')

import mamba_ssm
print(f'mamba_ssm: {mamba_ssm.__version__}')
import causal_conv1d
print(f'causal_conv1d: {causal_conv1d.__version__}')
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
print('mamba-ssm kernel OK')

import transformers, peft, trl, accelerate, bitsandbytes, datasets, swanlab
print(f'transformers: {transformers.__version__}')
print(f'peft        : {peft.__version__}')
print(f'trl         : {trl.__version__}')
print(f'accelerate  : {accelerate.__version__}')
print(f'bnb         : {bitsandbytes.__version__}')
print(f'datasets    : {datasets.__version__}')
print(f'swanlab     : {swanlab.__version__}')
print('ALL IMPORTS OK')
