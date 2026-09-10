"""Pre-flight imports check for DPO stack."""
import sys, types
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Stub mamba3 (matches SFT seed env)
for _mod_name in [
    'mamba_ssm.modules.mamba3',
    'mamba_ssm.ops.cute',
    'mamba_ssm.ops.cute.mamba3',
    'mamba_ssm.ops.cute.mamba3.mamba3_step_fn',
]:
    _m = types.ModuleType(_mod_name); _m.__path__ = []; _m.__package__ = _mod_name
    sys.modules[_mod_name] = _m
sys.modules['mamba_ssm.modules.mamba3'].Mamba3 = None

def v(modname):
    try:
        mod = __import__(modname)
        print(f'  OK  {modname:20s} {getattr(mod, "__version__", "?")}')
        return mod
    except Exception as e:
        print(f'  ERR {modname:20s} {type(e).__name__}: {str(e)[:140]}')
        return None

print('=== DPO stack imports ===')
v('torch')
v('unsloth')
v('mamba_ssm')
v('transformers')
v('peft')
v('trl')
v('datasets')
v('pandas')
v('safetensors')
v('tokenizers')
v('accelerate')
v('swanlab')

# Specific submodule imports we actually use
print('\n=== DPO-specific symbols ===')
ok = True
try:
    from unsloth import FastLanguageModel; print('  OK  unsloth.FastLanguageModel')
except Exception as e:
    print(f'  ERR unsloth.FastLanguageModel: {e}'); ok = False
try:
    from peft import PeftModel, LoraConfig; print('  OK  peft.PeftModel, peft.LoraConfig')
except Exception as e:
    print(f'  ERR peft.PeftModel: {e}'); ok = False
try:
    from trl import DPOTrainer, DPOConfig; print('  OK  trl.DPOTrainer, trl.DPOConfig')
except Exception as e:
    print(f'  ERR trl.DPOTrainer: {e}'); ok = False
try:
    from datasets import Dataset; print('  OK  datasets.Dataset')
except Exception as e:
    print(f'  ERR datasets.Dataset: {e}'); ok = False
try:
    from safetensors.torch import load_file, save_file; print('  OK  safetensors.torch.load_file/save_file')
except Exception as e:
    print(f'  ERR safetensors.torch: {e}'); ok = False
try:
    from transformers import TrainerCallback; print('  OK  transformers.TrainerCallback')
except Exception as e:
    print(f'  ERR transformers.TrainerCallback: {e}'); ok = False

print('\n=== GPU check ===')
import torch
print(f'  cuda available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {p.name}  {p.total_memory/1024**3:.1f} GB  sm_{p.major}{p.minor}')

print(f'\n=== Result: {"ALL OK" if ok else "FAILURES"} ===')
