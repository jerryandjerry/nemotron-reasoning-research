#!/usr/bin/env python3
"""WEIGHTED SVD LoRA soup. Best rank-r approx of  Σ wᵢ·(Bᵢ@Aᵢ)  via the QR+SVD trick:
scale each Aᵢ block by wᵢ (weights sum to 1) and drop the /N. OUTPUT_DIR/SUBMIT_DIR env-controlled.
Recipes (from the 0.86 out_proj run's adapter saves):
  1. wsoup-last5-f0.5 : final step 0.5, the other 4 = 0.5/4 = 0.125 each
  2. wise-0.1-0.2-0.7 : first 0.1, middle 0.2, last 0.7"""
import os, sys, json, zipfile, glob, time
import torch
from safetensors.torch import load_file, save_file

print = lambda *a, **kw: __builtins__.print(*a, **{**kw, 'flush': True})

OUTPUT_DIR  = os.environ.get('OUTPUT_DIR', '/root/autodl-tmp/adapter_output_moe_notie_outproj')
SUBMIT_DIR  = os.environ.get('SUBMIT_DIR', '/root/autodl-tmp/260523_wsoup_submission')
WORKING_DIR = '/root/autodl-tmp/_svd_work_wsoup'

os.makedirs(SUBMIT_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

step_to_dir = {}
for d in sorted(os.listdir(OUTPUT_DIR)):
    full = os.path.join(OUTPUT_DIR, d)
    if os.path.isdir(full) and d.startswith('adapter-'):
        try:
            step_to_dir[int(d.split('-')[1].split('_')[0])] = full
        except (IndexError, ValueError):
            pass
print(f'OUTPUT_DIR={OUTPUT_DIR}')
print(f'Found {len(step_to_dir)} adapters: {sorted(step_to_dir.keys())}')

first_dir = step_to_dir[sorted(step_to_dir.keys())[0]]
with open(os.path.join(first_dir, 'adapter_config.json')) as f:
    cfg = json.load(f)
RANK = cfg['r']; ALPHA = cfg['lora_alpha']
print(f'rank={RANK}, alpha={ALPHA}')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'device: {DEVICE}')

# name -> {step: weight}  (weights must sum to 1.0)
SOUPS = [
    ('wsoup-last5-f0.5',  {200: 0.125, 209: 0.125, 221: 0.125, 234: 0.125, 246: 0.5}),
    ('wise-0.1-0.2-0.7',  {50: 0.1, 150: 0.2, 246: 0.7}),
]


def svd_merge_qr_weighted(Bs, As, weights, rank):
    """best rank-r approx of  Σ wᵢ Bᵢ@Aᵢ ."""
    Bs = [B.to(DEVICE, dtype=torch.float32) for B in Bs]
    As = [w * A.to(DEVICE, dtype=torch.float32) for w, A in zip(weights, As)]  # scale Aᵢ by wᵢ
    B_stack = torch.cat(Bs, dim=1)
    A_stack = torch.cat(As, dim=0)
    Q, R = torch.linalg.qr(B_stack, mode='reduced')
    M_small = R @ A_stack                      # = B_stack @ A_stack = Σ wᵢ Bᵢ@Aᵢ  (NO /N)
    U_small, S, Vt = torch.linalg.svd(M_small, full_matrices=False)
    rank_eff = min(rank, S.numel())
    sqrtS = torch.sqrt(S[:rank_eff].clamp_min(0))
    new_B = (Q @ U_small[:, :rank_eff]) * sqrtS
    new_A = sqrtS.unsqueeze(-1) * Vt[:rank_eff, :]
    return new_B.cpu(), new_A.cpu()


def rename_lm_head(tensors):
    src = 'base_model.model.lm_head.'; dst = 'base_model.model.backbone.lm_head.'
    out, n = {}, 0
    for k, v in tensors.items():
        if k.startswith(src):
            out[dst + k[len(src):]] = v; n += 1
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


for name, wmap in SOUPS:
    steps = sorted(wmap.keys())
    ws = [wmap[s] for s in steps]
    assert abs(sum(ws) - 1.0) < 1e-6, f'{name} weights sum {sum(ws)} != 1'
    print(f'\n=== Building {name} ===  steps: {steps}  weights: {ws}')
    missing = [s for s in steps if s not in step_to_dir]
    if missing:
        print(f'  SKIP: missing adapter steps {missing}'); continue
    t0 = time.time()
    all_t = [load_file(os.path.join(step_to_dir[s], 'adapter_model.safetensors')) for s in steps]
    print(f'  load time: {time.time()-t0:.1f}s')

    base_keys = set(all_t[0].keys())
    lora_a_keys = sorted([k for k in base_keys if '.lora_A.' in k])
    print(f'  LoRA modules to SVD-merge: {len(lora_a_keys)}')

    new_tensors = {k: all_t[0][k] for k in base_keys if '.lora_A.' not in k and '.lora_B.' not in k}
    t_svd = 0.0; n_done = 0
    for a_key in lora_a_keys:
        b_key = a_key.replace('.lora_A.', '.lora_B.')
        if b_key not in base_keys:
            print(f'  WARN: no B for {a_key}'); continue
        As = [td[a_key] for td in all_t]
        Bs = [td[b_key] for td in all_t]
        orig_dtype = As[0].dtype
        t1 = time.time()
        new_B, new_A = svd_merge_qr_weighted(Bs, As, ws, rank=RANK)
        t_svd += time.time() - t1
        new_tensors[a_key] = new_A.to(orig_dtype)
        new_tensors[b_key] = new_B.to(orig_dtype)
        n_done += 1
    print(f'  SVD-merge done: {n_done} modules in {t_svd:.1f}s')

    renamed = rename_lm_head(new_tensors)
    work = os.path.join(WORKING_DIR, name); os.makedirs(work, exist_ok=True)
    sf_path = os.path.join(work, 'adapter_model.safetensors'); save_file(renamed, sf_path)
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

print('\n=== Summary — weighted SVD soup zips ===')
for f in sorted(glob.glob(os.path.join(SUBMIT_DIR, 'submission_*.zip'))):
    print(f'  {f}  ({os.path.getsize(f)/1024/1024:.1f} MB)')
