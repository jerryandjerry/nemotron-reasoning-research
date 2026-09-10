#!/usr/bin/env python3
"""Pre-flight for GRPO: import every package train_grpo_v1.py uses + verify the
GRPOConfig knobs we set actually exist in this TRL version."""
import sys, types

# mamba3 stubs (same as training script)
for _mod_name in [
    'mamba_ssm.modules.mamba3', 'mamba_ssm.ops.cute',
    'mamba_ssm.ops.cute.mamba3', 'mamba_ssm.ops.cute.mamba3.mamba3_step_fn',
]:
    _m = types.ModuleType(_mod_name); _m.__path__ = []; _m.__package__ = _mod_name
    sys.modules[_mod_name] = _m
sys.modules['mamba_ssm.modules.mamba3'].Mamba3 = None

import unsloth;       print('unsloth      :', unsloth.__version__)
import unsloth_zoo;   print('unsloth_zoo  :', unsloth_zoo.__version__)
import mamba_ssm;     print('mamba_ssm    :', mamba_ssm.__version__)
import torch;         print('torch        :', torch.__version__, '| cuda', torch.version.cuda,
                            '| gpu', torch.cuda.get_device_name(0))
import transformers;  print('transformers :', transformers.__version__)
import peft;          print('peft         :', peft.__version__)
import trl;           print('trl          :', trl.__version__)
import vllm;          print('vllm         :', vllm.__version__)
import datasets;      print('datasets     :', datasets.__version__)
from safetensors.torch import load_file;  print('safetensors  : ok')

from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
print('GRPOConfig + GRPOTrainer + SamplingParams import: ok')

# Verify every non-default GRPOConfig knob train_grpo_v1.py sets is accepted
import inspect, dataclasses
fields = {f.name for f in dataclasses.fields(GRPOConfig)}
NEEDED = ['beta', 'loss_type', 'importance_sampling_level', 'epsilon_high',
          'temperature', 'top_p', 'top_k', 'min_p', 'num_generations',
          'max_prompt_length', 'max_completion_length', 'vllm_sampling_params',
          'save_strategy', 'save_steps', 'save_total_limit', 'max_grad_norm']
missing = [k for k in NEEDED if k not in fields]
print('GRPOConfig knobs:', 'ALL PRESENT' if not missing else f'MISSING: {missing}')
assert not missing, f'TRL {trl.__version__} lacks: {missing}'

# Instantiate with our exact algorithm values (catches enum/validation errors)
cfg = GRPOConfig(
    output_dir='/tmp/_grpo_cfg_test', beta=0.02, loss_type='dapo',
    importance_sampling_level='sequence', epsilon_high=0.28,
    temperature=1.0, top_p=1.0, top_k=-1, min_p=0.1,
    num_generations=4, max_prompt_length=512, max_completion_length=3584,
    save_strategy='no', report_to='none',
)
print('GRPOConfig instantiation with our values: ok')
print('\nALL IMPORTS + CONFIG CHECKS PASS')
