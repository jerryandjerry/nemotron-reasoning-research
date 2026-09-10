#!/usr/bin/env python3
"""
Build 3 soup submissions from the saved DDP checkpoints:

  1. final-244           = checkpoint-244 (control, no averaging)
  2. soup-last6          = uniform avg of {220, 225, 229, 234, 239, 244}
  3. wise-30+120+244     = uniform avg of {30, 120, 244}

Each soup:
  - load each adapter_model.safetensors
  - element-wise average tensors (in fp32, cast back to original dtype)
  - rename lm_head keys: base_model.model.lm_head.* -> base_model.model.backbone.lm_head.*
  - patch adapter_config.json: base_model_name_or_path, inference_mode=True
  - zip as submission_<name>.zip with the standard 2-file layout

Run on instance: /root/miniconda3/bin/python build_soups.py
"""
import os, sys, json, zipfile, glob, shutil, time
import torch
from safetensors.torch import load_file, save_file

OUTPUT_DIR  = '/root/autodl-tmp/adapter_output_huikang_ddp_perMBnorm'
SUBMIT_DIR  = '/root/autodl-tmp/soups'
WORKING_DIR = '/root/autodl-tmp/soups/_work'

os.makedirs(SUBMIT_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

# ---- Index saved checkpoints by step ----------------------------------
step_to_dir = {}
for d in sorted(os.listdir(OUTPUT_DIR)):
    full = os.path.join(OUTPUT_DIR, d)
    if not os.path.isdir(full):
        continue
    # Names: 'adapter-30_loss...' or 'checkpoint-244_loss...'
    if d.startswith(('adapter-', 'checkpoint-')):
        try:
            step = int(d.split('-')[1].split('_')[0])
            step_to_dir[step] = full
        except (IndexError, ValueError):
            print(f'WARN: skipping unparseable dir {d}')
print(f'Found {len(step_to_dir)} saved checkpoints: steps = {sorted(step_to_dir.keys())}')

# ---- Soup recipes -----------------------------------------------------
SOUPS = [
    ('final-244',           [244],                              [1.0]),
    ('soup-last6',          [220, 225, 229, 234, 239, 244],     [1/6]*6),
    ('wise-30+120+244',     [30, 120, 244],                     [1/3]*3),
]

# ---- Helpers ----------------------------------------------------------
def average_adapters(adapter_paths, weights):
    """Element-wise weighted average. Cast each tensor to fp32 to avoid
    bf16 precision loss when summing, then cast back to original dtype."""
    assert len(adapter_paths) == len(weights), 'len mismatch'
    assert abs(sum(weights) - 1.0) < 1e-6, f'weights must sum to 1, got {sum(weights)}'

    print(f'  loading {len(adapter_paths)} adapter(s)...')
    tensors_list = []
    for p in adapter_paths:
        t0 = time.time()
        td = load_file(p)
        tensors_list.append(td)
        print(f'    {p}: {len(td)} tensors, {time.time()-t0:.1f}s')

    keys = set(tensors_list[0].keys())
    for i, td in enumerate(tensors_list[1:], 1):
        if set(td.keys()) != keys:
            extra = set(td.keys()) - keys
            missing = keys - set(td.keys())
            raise RuntimeError(f'adapter[{i}] key mismatch — extra={list(extra)[:3]} missing={list(missing)[:3]}')

    print(f'  averaging {len(keys)} tensors...')
    out = {}
    for k in keys:
        original_dtype = tensors_list[0][k].dtype
        acc = torch.zeros_like(tensors_list[0][k], dtype=torch.float32)
        for w, td in zip(weights, tensors_list):
            acc += w * td[k].to(torch.float32)
        out[k] = acc.to(original_dtype)
    return out

def rename_lm_head(tensors):
    """base_model.model.lm_head.* -> base_model.model.backbone.lm_head.*"""
    src_prefix = 'base_model.model.lm_head.'
    dst_prefix = 'base_model.model.backbone.lm_head.'
    out = {}
    renamed_count = 0
    for k, v in tensors.items():
        if k.startswith(src_prefix):
            new_k = dst_prefix + k[len(src_prefix):]
            out[new_k] = v
            renamed_count += 1
        else:
            out[k] = v
    print(f'  renamed {renamed_count} lm_head keys')
    return out

def patch_config(src_config_path, dst_config_path):
    with open(src_config_path) as f:
        cfg = json.load(f)
    cfg['base_model_name_or_path'] = 'metric/nemotron-3-nano-30b-a3b-bf16'
    cfg['inference_mode'] = True
    with open(dst_config_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'  patched config -> {dst_config_path}')

def build_zip(adapter_safetensors_path, config_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(adapter_safetensors_path, arcname='adapter_model.safetensors')
        zf.write(config_path,             arcname='adapter_config.json')
    size_mb = os.path.getsize(zip_path) / 1024 / 1024
    print(f'  -> {zip_path}  ({size_mb:.1f} MB)')

def crc_check(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
    if bad is None:
        print(f'  CRC OK')
    else:
        print(f'  CRC FAIL: {bad}')

# ---- Build each soup --------------------------------------------------
for name, steps, weights in SOUPS:
    print(f'\n=== Building {name} ===')
    print(f'  steps: {steps}  weights: {weights}')

    # Sanity check all steps present
    missing = [s for s in steps if s not in step_to_dir]
    if missing:
        print(f'  SKIP: missing checkpoint(s) {missing}')
        continue

    adapter_paths = [os.path.join(step_to_dir[s], 'adapter_model.safetensors') for s in steps]

    # Average
    averaged = average_adapters(adapter_paths, weights)

    # Rename lm_head keys
    renamed = rename_lm_head(averaged)

    # Save + patch config to a working dir
    work = os.path.join(WORKING_DIR, name)
    os.makedirs(work, exist_ok=True)
    safetensors_path = os.path.join(work, 'adapter_model.safetensors')
    save_file(renamed, safetensors_path)
    print(f'  wrote averaged adapter ({os.path.getsize(safetensors_path)/1024/1024:.1f} MB)')

    # Pick a config to patch (use the final-step config; all should be equivalent)
    src_config = os.path.join(step_to_dir[steps[-1]], 'adapter_config.json')
    config_path = os.path.join(work, 'adapter_config.json')
    patch_config(src_config, config_path)

    # Zip
    zip_path = os.path.join(SUBMIT_DIR, f'submission_{name}.zip')
    build_zip(safetensors_path, config_path, zip_path)
    crc_check(zip_path)

    # Cleanup: delete the intermediate ~4 GB adapter file to free disk for next soup
    os.remove(safetensors_path)
    os.remove(config_path)
    os.rmdir(work)
    print(f'  cleaned up {work}')

print('\n=== Summary ===')
for f in sorted(glob.glob(os.path.join(SUBMIT_DIR, '*.zip'))):
    print(f'  {f}  ({os.path.getsize(f)/1024/1024:.1f} MB)')
