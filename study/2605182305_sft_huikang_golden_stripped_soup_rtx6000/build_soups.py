#!/usr/bin/env python3
"""Build the 2 default soup zips per TRAINING_ROUTINE.md §2.4B + §3.

Soups (user defaults, 2026-05-18):
  1. soup-last5         = uniform avg of the 5 highest-step adapters on disk
  2. wise-50-150-246    = uniform avg of {50, 150, 246}

For each recipe:
  - load each adapter_model.safetensors
  - element-wise average in fp32, cast back to original dtype
  - rename lm_head keys: base_model.model.lm_head.* -> base_model.model.backbone.lm_head.*
  - patch adapter_config.json: base_model_name_or_path + inference_mode=True
  - zip as submission_<recipe>.zip + CRC verify
  - drop into the submission folder

Run on instance: cd /root/autodl-tmp && /root/miniconda3/bin/python build_soups.py
"""
import os, sys, json, zipfile, glob, time
import torch
from safetensors.torch import load_file, save_file

OUTPUT_DIR  = '/root/autodl-tmp/adapter_output_huikang_golden_stripped_soup'
# Submission folder will be created with the training-finish timestamp; we read
# the env var SUBMIT_DIR set by the wrapper, or default to ./_soup_zips here.
SUBMIT_DIR  = os.environ.get('SUBMIT_DIR', '/root/autodl-tmp/_soup_zips')
WORKING_DIR = '/root/autodl-tmp/_soup_work'

os.makedirs(SUBMIT_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

# ── Index adapter-* dirs by step ─────────────────────────────────────
step_to_dir = {}
for d in sorted(os.listdir(OUTPUT_DIR)):
    full = os.path.join(OUTPUT_DIR, d)
    if not os.path.isdir(full): continue
    if d.startswith('adapter-'):
        try:
            step = int(d.split('-')[1].split('_')[0])
            step_to_dir[step] = full
        except (IndexError, ValueError):
            print(f'WARN: skipping unparseable dir {d}')
all_steps = sorted(step_to_dir.keys())
print(f'Found {len(all_steps)} adapter-only saves: steps = {all_steps}')

# ── Soup recipes (compute concretely from what's on disk) ──────────────
# 1. soup-last5: last 5 highest-step adapters
last5_steps = all_steps[-5:] if len(all_steps) >= 5 else all_steps
# 2. wise-50-150-246: explicit per user spec
wise_steps = [50, 150, 246]

SOUPS = [
    ('soup-last5',       last5_steps, [1.0/len(last5_steps)]*len(last5_steps)),
    ('wise-50-150-246',  wise_steps,  [1.0/3]*3),
]

# ── Helpers ──────────────────────────────────────────────────────────

def average_adapters(adapter_paths, weights):
    assert len(adapter_paths) == len(weights), 'len mismatch'
    assert abs(sum(weights) - 1.0) < 1e-6, f'weights must sum to 1, got {sum(weights)}'
    print(f'  loading {len(adapter_paths)} adapter(s)...')
    tensors_list = []
    for p in adapter_paths:
        t0 = time.time()
        td = load_file(p)
        tensors_list.append(td)
        print(f'    {os.path.basename(os.path.dirname(p))}: {len(td)} tensors, {time.time()-t0:.1f}s')
    keys = set(tensors_list[0].keys())
    for i, td in enumerate(tensors_list[1:], 1):
        if set(td.keys()) != keys:
            extra = set(td.keys()) - keys
            missing = keys - set(td.keys())
            raise RuntimeError(f'adapter[{i}] key mismatch — extra={list(extra)[:3]} missing={list(missing)[:3]}')
    print(f'  averaging {len(keys)} tensors in fp32...')
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
    n = 0
    for k, v in tensors.items():
        if k.startswith(src_prefix):
            out[dst_prefix + k[len(src_prefix):]] = v
            n += 1
        else:
            out[k] = v
    print(f'  renamed {n} lm_head keys')
    return out


def patch_config(src_config_path, dst_config_path):
    with open(src_config_path) as f:
        cfg = json.load(f)
    cfg['base_model_name_or_path'] = 'metric/nemotron-3-nano-30b-a3b-bf16'
    cfg['inference_mode'] = True
    with open(dst_config_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'  patched config -> {dst_config_path}')


def build_zip(safetensors_path, config_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(safetensors_path, arcname='adapter_model.safetensors')
        zf.write(config_path,      arcname='adapter_config.json')
    size_mb = os.path.getsize(zip_path) / 1024 / 1024
    print(f'  -> {zip_path}  ({size_mb:.1f} MB)')


def crc_check(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
    print(f'  CRC {"OK" if bad is None else "FAIL: "+bad}')


# ── Build each soup ──────────────────────────────────────────────────
for name, steps, weights in SOUPS:
    print(f'\n=== Building {name} ===')
    print(f'  steps: {steps}  weights: {[round(w,3) for w in weights]}')

    missing = [s for s in steps if s not in step_to_dir]
    if missing:
        print(f'  SKIP: missing checkpoint(s) {missing}')
        continue

    adapter_paths = [os.path.join(step_to_dir[s], 'adapter_model.safetensors') for s in steps]

    bad = [p for p in adapter_paths if not os.path.isfile(p)]
    if bad:
        print(f'  SKIP: missing adapter_model.safetensors in {bad}')
        continue

    averaged = average_adapters(adapter_paths, weights)
    renamed = rename_lm_head(averaged)

    work = os.path.join(WORKING_DIR, name)
    os.makedirs(work, exist_ok=True)
    safetensors_path = os.path.join(work, 'adapter_model.safetensors')
    save_file(renamed, safetensors_path)
    print(f'  wrote averaged adapter ({os.path.getsize(safetensors_path)/1024/1024:.1f} MB)')

    src_config = os.path.join(step_to_dir[steps[-1]], 'adapter_config.json')
    config_path = os.path.join(work, 'adapter_config.json')
    patch_config(src_config, config_path)

    zip_path = os.path.join(SUBMIT_DIR, f'submission_{name}.zip')
    build_zip(safetensors_path, config_path, zip_path)
    crc_check(zip_path)

    os.remove(safetensors_path)
    os.remove(config_path)
    os.rmdir(work)

# ── Summary ──────────────────────────────────────────────────────────
print('\n=== Summary — soup submission zips ===')
for f in sorted(glob.glob(os.path.join(SUBMIT_DIR, 'submission_*.zip'))):
    print(f'  {f}  ({os.path.getsize(f)/1024/1024:.1f} MB)')
