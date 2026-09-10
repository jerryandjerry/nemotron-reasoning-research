#!/usr/bin/env python3
"""SVD LoRA soup for the MoE-tie experiments. Same QR+SVD math as the run #11 reference;
OUTPUT_DIR + SUBMIT_DIR are env-controlled so one script serves both experiments.
Recipes match ADAPTER_SAVE_STEPS for num_steps=246 (last5 + wise-50-150-246)."""
import os, sys, json, zipfile, glob, time
import torch
from safetensors.torch import load_file, save_file

print = lambda *a, **kw: __builtins__.print(*a, **{**kw, 'flush': True})

OUTPUT_DIR  = os.environ.get('OUTPUT_DIR', '/root/autodl-tmp/adapter_output_moe_fixtie')
SUBMIT_DIR  = os.environ.get('SUBMIT_DIR', '/root/autodl-tmp/_moe_submit')
WORKING_DIR = '/root/autodl-tmp/_svd_work_moe'

os.makedirs(SUBMIT_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

step_to_dir = {}
for d in sorted(os.listdir(OUTPUT_DIR)):
    full = os.path.join(OUTPUT_DIR, d)
    if not os.path.isdir(full):
        continue
    if d.startswith('adapter-'):
        try:
            step = int(d.split('-')[1].split('_')[0])
            step_to_dir[step] = full
        except (IndexError, ValueError):
            pass
print(f'OUTPUT_DIR={OUTPUT_DIR}')
print(f'Found {len(step_to_dir)} adapters: {sorted(step_to_dir.keys())}')

first_dir = step_to_dir[list(step_to_dir.keys())[0]]
with open(os.path.join(first_dir, 'adapter_config.json')) as f:
    cfg = json.load(f)
RANK = cfg['r']
ALPHA = cfg['lora_alpha']
print(f'rank={RANK}, alpha={ALPHA}')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {DEVICE}')

SOUPS = [
    ('svd-soup-last5',       [200, 201, 212, 224, 236]),
    ('svd-wise-50-150-236',  [50, 150, 236]),
]


def svd_merge_qr(Bs, As, rank):
    N = len(Bs)
    Bs = [B.to(DEVICE, dtype=torch.float32) for B in Bs]
    As = [A.to(DEVICE, dtype=torch.float32) for A in As]
    B_stack = torch.cat(Bs, dim=1)
    A_stack = torch.cat(As, dim=0)
    Q, R = torch.linalg.qr(B_stack, mode='reduced')
    M_small = (R @ A_stack) / N
    U_small, S, Vt = torch.linalg.svd(M_small, full_matrices=False)
    rank_eff = min(rank, S.numel())
    sqrtS = torch.sqrt(S[:rank_eff].clamp_min(0))
    U_top = Q @ U_small[:, :rank_eff]
    new_B = U_top * sqrtS
    new_A = sqrtS.unsqueeze(-1) * Vt[:rank_eff, :]
    return new_B.cpu(), new_A.cpu(), float(S[0].item()), float(S[rank_eff - 1].item())


def rename_lm_head(tensors):
    src = 'base_model.model.lm_head.'
    dst = 'base_model.model.backbone.lm_head.'
    out, n = {}, 0
    for k, v in tensors.items():
        if k.startswith(src):
            out[dst + k[len(src):]] = v
            n += 1
        else:
            out[k] = v
    print(f'  renamed {n} lm_head keys')
    return out


def patch_config(src, dst):
    with open(src) as f:
        c = json.load(f)
    c['base_model_name_or_path'] = 'metric/nemotron-3-nano-30b-a3b-bf16'
    c['inference_mode'] = True
    with open(dst, 'w') as f:
        json.dump(c, f, indent=2)


for name, steps in SOUPS:
    print(f'\n=== Building {name} ===  steps: {steps}')
    missing = [s for s in steps if s not in step_to_dir]
    if missing:
        print(f'  SKIP: missing adapter steps {missing}')
        continue
    t0 = time.time()
    all_t = [load_file(os.path.join(step_to_dir[s], 'adapter_model.safetensors')) for s in steps]
    print(f'  load time: {time.time()-t0:.1f}s')

    base_keys = set(all_t[0].keys())
    lora_a_keys = sorted([k for k in base_keys if '.lora_A.' in k])
    print(f'  LoRA modules to SVD-merge: {len(lora_a_keys)}')

    new_tensors = {}
    for k in base_keys:
        if '.lora_A.' in k or '.lora_B.' in k:
            continue
        new_tensors[k] = all_t[0][k]

    t_svd = 0; n_done = 0; s_max_samples = []; s_min_samples = []
    for a_key in lora_a_keys:
        b_key = a_key.replace('.lora_A.', '.lora_B.')
        if b_key not in base_keys:
            print(f'  WARN: no B for {a_key}'); continue
        As = [td[a_key] for td in all_t]
        Bs = [td[b_key] for td in all_t]
        orig_dtype = As[0].dtype
        t1 = time.time()
        new_B, new_A, s_max, s_min = svd_merge_qr(Bs, As, rank=RANK)
        t_svd += time.time() - t1
        new_tensors[a_key] = new_A.to(orig_dtype)
        new_tensors[b_key] = new_B.to(orig_dtype)
        s_max_samples.append(s_max); s_min_samples.append(s_min)
        n_done += 1
    print(f'  SVD-merge done: {n_done} modules in {t_svd:.1f}s')

    renamed = rename_lm_head(new_tensors)
    work = os.path.join(WORKING_DIR, name)
    os.makedirs(work, exist_ok=True)
    sf_path = os.path.join(work, 'adapter_model.safetensors')
    save_file(renamed, sf_path)
    cfg_path = os.path.join(work, 'adapter_config.json')
    patch_config(os.path.join(step_to_dir[steps[-1]], 'adapter_config.json'), cfg_path)
    zip_path = os.path.join(SUBMIT_DIR, f'submission_{name}.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(sf_path, arcname='adapter_model.safetensors')
        zf.write(cfg_path, arcname='adapter_config.json')
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
    print(f'  -> {zip_path} ({os.path.getsize(zip_path)/1024/1024:.1f} MB)  CRC {"OK" if bad is None else "FAIL"}')
    os.remove(sf_path); os.remove(cfg_path); os.rmdir(work)
    del all_t, new_tensors, renamed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print('\n=== Summary — SVD soup zips ===')
for f in sorted(glob.glob(os.path.join(SUBMIT_DIR, 'submission_svd-*.zip'))):
    print(f'  {f}  ({os.path.getsize(f)/1024/1024:.1f} MB)')
