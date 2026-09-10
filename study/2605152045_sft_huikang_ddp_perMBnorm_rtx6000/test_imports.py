#!/usr/bin/env python3
"""Pre-flight import check for DDP 2-GPU training."""
import sys, os
print(f'Python: {sys.version}')
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'Visible GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} '
          f'({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)')
import torch.distributed as dist
print(f'NCCL available: {dist.is_nccl_available()}')
import unsloth; print(f'unsloth: {unsloth.__version__}')
import mamba_ssm; print(f'mamba_ssm: {mamba_ssm.__version__}')
from cut_cross_entropy import linear_cross_entropy; print('cut_cross_entropy: OK')
import peft; print(f'peft: {peft.__version__}')
from safetensors.torch import load_file, save_file; print('safetensors: OK')
import swanlab; print(f'swanlab: {swanlab.__version__}')
from tokenizers import Tokenizer; print('tokenizers: OK')
print('\nAll imports OK')
print('\nLaunch command: torchrun --nproc_per_node=2 train_huikang_textcsv_0408_ddp.py')
