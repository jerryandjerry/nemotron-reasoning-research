#!/usr/bin/env python3
"""Pre-flight: import every package train_stage1.py uses, print versions."""
import os, sys, types
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
os.environ['TRANSFORMERS_NO_TF']   = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['HF_HUB_OFFLINE']       = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import unsloth
from unsloth import FastLanguageModel
print('unsloth:', unsloth.__version__)

# Stub mamba3 modules exactly as the trainer does
for _mod_name in ['mamba_ssm.modules.mamba3','mamba_ssm.ops.cute',
                  'mamba_ssm.ops.cute.mamba3','mamba_ssm.ops.cute.mamba3.mamba3_step_fn']:
    _m = types.ModuleType(_mod_name); _m.__path__ = []; _m.__package__ = _mod_name
    sys.modules[_mod_name] = _m
sys.modules['mamba_ssm.modules.mamba3'].Mamba3 = None

import mamba_ssm;                     print('mamba_ssm:', mamba_ssm.__version__)
import torch;                         print('torch:', torch.__version__)
import transformers;                  print('transformers:', transformers.__version__)
import peft;                          print('peft:', peft.__version__)
from peft import LoraConfig
from peft.tuners.lora import Linear as LoraLinear
from cut_cross_entropy import linear_cross_entropy
print('cut_cross_entropy: OK')
from tokenizers import Tokenizer as RawTokenizer
import tokenizers;                    print('tokenizers:', tokenizers.__version__)
import swanlab;                       print('swanlab:', swanlab.__version__)
import csv, re, json, zipfile, gc, subprocess, math, logging, shutil
print('stdlib: OK')

print('GPU:', torch.cuda.get_device_name(0))
print('VRAM total: %.1f GB' % (torch.cuda.get_device_properties(0).total_memory/1024**3))
print('CUDA avail:', torch.cuda.is_available())
print('\nALL IMPORTS OK')
