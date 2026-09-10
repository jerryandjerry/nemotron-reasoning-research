#!/usr/bin/env python3
"""
Build LoRA soup submissions from the 7 saved adapter checkpoints + final adapter.

Saved adapters from run #8 (steps 150-282) live in the adapter_output dir as raw
adapter_model.safetensors (lm_head keys NOT renamed). The final step-297 adapter
is inside submission.zip (lm_head ALREADY renamed) AND checkpoint-297.zip (full
state, raw adapter inside). We use the raw version from checkpoint-297 dir for
averaging, then rename lm_head ONCE on the final soup output.

Recipes built (skipped if any required adapter is missing):

  1. final-297        = checkpoint-297 (control — same as submission.zip)
  2. soup-last6       = uniform avg of {238, 250, 252, 267, 282, 297}
                        captures the converged-LR-decay tail
  3. soup-last4       = uniform avg of {267, 282, 297}
                        tighter end average
  4. soup-all7        = uniform avg of {150, 200, 238, 250, 252, 267, 282}
                        without 297 (in case 297 is unavailable)
  5. wise-200-297     = 0.5 * adapter-200 + 0.5 * checkpoint-297
                        early × late WiSE-FT interpolation

Run on instance: cd /root/autodl-tmp && /root/miniconda3/bin/python build_soups.py
"""
import os, sys, json, zipfile, glob, time
import torch
from safetensors.torch import load_file, save_file

OUTPUT_DIR  = '/root/autodl-tmp/adapter_output_9500_catprefix'
SUBMIT_DIR  = '/root/autodl-tmp/soups'
WORKING_DIR = '/root/autodl-tmp/soups/_work'

os.makedirs(SUBMIT_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

# ── Index saved checkpoints by step ──────────────────────────────────
step_to_dir = {}
for d in sorted(os.listdir(OUTPUT_DIR)):
    full = os.path.join(OUTPUT_DIR, d)
    if not os.path.isdir(full): continue
    if d.startswith(('adapter-', 'checkpoint-')):
        try:
            step = int(d.split('-')[1].split('_')[0])
            step_to_dir[step] = full
        except (IndexError, ValueError):
            print(f'WARN: skipping unparseable dir {d}')
print(f'Found {len(step_to_dir)} saved checkpoints: steps = {sorted(step_to_dir.keys())}')

# ── Soup recipes ─────────────────────────────────────────────────────
SOUPS = [
    ('soup-last5',    [250, 252, 267, 282, 297],   [1/5]*5),
    ('wise-150-297',  [150, 297],                  [0.5, 0.5]),
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
    # All adapters must have the same keys
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

    # Sanity check files exist
    bad = [p for p in adapter_paths if not os.path.isfile(p)]
    if bad:
        print(f'  SKIP: missing adapter_model.safetensors in {bad}')
        continue

    # Average
    averaged = average_adapters(adapter_paths, weights)

    # Rename lm_head keys for Kaggle compatibility
    renamed = rename_lm_head(averaged)

    # Write to working dir
    work = os.path.join(WORKING_DIR, name)
    os.makedirs(work, exist_ok=True)
    safetensors_path = os.path.join(work, 'adapter_model.safetensors')
    save_file(renamed, safetensors_path)
    print(f'  wrote averaged adapter ({os.path.getsize(safetensors_path)/1024/1024:.1f} MB)')

    # Patch config (use config from the final-step adapter in the recipe)
    src_config = os.path.join(step_to_dir[steps[-1]], 'adapter_config.json')
    config_path = os.path.join(work, 'adapter_config.json')
    patch_config(src_config, config_path)

    # Build the submission zip
    zip_path = os.path.join(SUBMIT_DIR, f'submission_{name}.zip')
    build_zip(safetensors_path, config_path, zip_path)
    crc_check(zip_path)

    # Cleanup the intermediate ~4 GB safetensors to free disk for next soup
    os.remove(safetensors_path)
    os.remove(config_path)
    os.rmdir(work)
    print(f'  cleaned up {work}')

# ── Summary ──────────────────────────────────────────────────────────
print('\n=== Summary — soup submission zips ===')
for f in sorted(glob.glob(os.path.join(SUBMIT_DIR, '*.zip'))):
    print(f'  {f}  ({os.path.getsize(f)/1024/1024:.1f} MB)')
print('\nDownload via AutoDL file manager from /root/autodl-tmp/soups/')
