#!/usr/bin/env python3
"""Merge-then-average LoRA soup with SVD re-extraction (FAST QR+SVD variant).

Math: for each LoRA module with N adapters (Bᵢ, Aᵢ) where Bᵢ∈ℝ^(out×r), Aᵢ∈ℝ^(r×in),
the soup delta we want is Δ̄ = (1/N) Σᵢ Bᵢ@Aᵢ. This has rank ≤ N·r.

Naive: form Δ̄ explicitly (out×in, can be huge for lm_head) → SVD it. Slow.

QR+SVD trick (much faster):
  1. Stack B's column-wise:  B_stack = [B₁ | … | Bₙ] ∈ ℝ^(out × N·r)
  2. Stack A's row-wise:     A_stack = [A₁; …; Aₙ] ∈ ℝ^(N·r × in)
  3. B_stack @ A_stack = Σᵢ Bᵢ @ Aᵢ (block-product identity)
  4. QR: B_stack = Q · R   (Q orthonormal, R square N·r × N·r)
  5. Δ̄ = Q · (R · A_stack / N)
  6. SVD the SMALL matrix M' = R · A_stack / N  ∈ ℝ^(N·r × in)
     M' = U' · S · Vᵀ
  7. Full SVD: Δ̄ = (Q · U') · S · Vᵀ → truncate to top r
  8. new_B = (Q · U')[:, :r] · sqrt(S[:r])
     new_A = sqrt(S[:r]) · Vᵀ[:r, :]
  9. new_B @ new_A = best rank-r approximation of Δ̄.

For lm_head with N=5: avoids materializing (131072, 2688) and replaces the big SVD
with QR on (131072, 160) + SVD on (160, 2688). ~100× faster.

Output is unbuffered (flush=True on every print).
"""
import os, sys, json, zipfile, glob, time
import torch
from safetensors.torch import load_file, save_file

# Force unbuffered output
print = lambda *a, **kw: __builtins__.print(*a, **{**kw, 'flush': True})

OUTPUT_DIR  = '/root/autodl-tmp/adapter_output_huikang_golden_stripped_soup'
SUBMIT_DIR  = os.environ.get('SUBMIT_DIR', '/root/autodl-tmp/260518_1507_submission')
WORKING_DIR = '/root/autodl-tmp/_svd_work'

os.makedirs(SUBMIT_DIR, exist_ok=True)
os.makedirs(WORKING_DIR, exist_ok=True)

step_to_dir = {}
for d in sorted(os.listdir(OUTPUT_DIR)):
    full = os.path.join(OUTPUT_DIR, d)
    if not os.path.isdir(full): continue
    if d.startswith('adapter-'):
        try:
            step = int(d.split('-')[1].split('_')[0])
            step_to_dir[step] = full
        except (IndexError, ValueError):
            pass
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
    ('svd-soup-last5',       [200, 209, 221, 234, 246]),
    ('svd-wise-50-150-246',  [50, 150, 246]),
]


def svd_merge_qr(Bs, As, rank):
    """Returns (new_B, new_A): best rank-r approximation of (1/N) Σ Bᵢ@Aᵢ
    using the QR+SVD trick (no (out, in) materialization)."""
    N = len(Bs)
    # All to GPU as fp32
    Bs = [B.to(DEVICE, dtype=torch.float32) for B in Bs]
    As = [A.to(DEVICE, dtype=torch.float32) for A in As]
    out_dim = Bs[0].shape[0]
    in_dim = As[0].shape[1]
    # Stack: B_stack ∈ (out, N·r),  A_stack ∈ (N·r, in)
    B_stack = torch.cat(Bs, dim=1)             # column-stack
    A_stack = torch.cat(As, dim=0)             # row-stack
    # QR on (out, N·r): Q (out, N·r), R (N·r, N·r)
    Q, R = torch.linalg.qr(B_stack, mode='reduced')
    # Small matrix M' = R @ A_stack / N, shape (N·r, in)
    M_small = (R @ A_stack) / N
    # SVD on the small matrix
    U_small, S, Vt = torch.linalg.svd(M_small, full_matrices=False)
    # Truncate to top rank
    rank_eff = min(rank, S.numel())
    sqrtS = torch.sqrt(S[:rank_eff].clamp_min(0))
    # Full U for top-r:  U_full = Q @ U_small[:, :rank_eff]
    U_top = Q @ U_small[:, :rank_eff]          # (out, r)
    new_B = U_top * sqrtS                       # (out, r)
    new_A = sqrtS.unsqueeze(-1) * Vt[:rank_eff, :]  # (r, in)
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
    with open(src) as f: c = json.load(f)
    c['base_model_name_or_path'] = 'metric/nemotron-3-nano-30b-a3b-bf16'
    c['inference_mode'] = True
    with open(dst, 'w') as f: json.dump(c, f, indent=2)


for name, steps in SOUPS:
    print(f'\n=== Building {name} ===')
    print(f'  steps: {steps}')

    t0 = time.time()
    all_t = []
    for s in steps:
        td = load_file(os.path.join(step_to_dir[s], 'adapter_model.safetensors'))
        all_t.append(td)
        print(f'  loaded step {s}: {len(td)} tensors')
    print(f'  load time: {time.time()-t0:.1f}s')

    base_keys = set(all_t[0].keys())
    lora_a_keys = sorted([k for k in base_keys if '.lora_A.' in k])
    print(f'  LoRA modules to SVD-merge: {len(lora_a_keys)}')

    new_tensors = {}
    for k in base_keys:
        if '.lora_A.' in k or '.lora_B.' in k: continue
        new_tensors[k] = all_t[0][k]
    print(f'  copied {len(new_tensors)} non-LoRA tensor(s) from first adapter')

    t_svd = 0
    n_done = 0
    s_max_samples = []
    s_min_samples = []
    for a_key in lora_a_keys:
        b_key = a_key.replace('.lora_A.', '.lora_B.')
        if b_key not in base_keys:
            print(f'  WARN: no B for {a_key}')
            continue
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
        if n_done % 500 == 0:
            print(f'    {n_done}/{len(lora_a_keys)} modules, t_svd={t_svd:.1f}s, '
                  f'last s_max={s_max:.4f} s_r={s_min:.6f}')
    print(f'  SVD-merge done: {n_done} modules in {t_svd:.1f}s')
    s_max_samples.sort(); s_min_samples.sort()
    print(f'  σ_1   median: {s_max_samples[len(s_max_samples)//2]:.4f}')
    print(f'  σ_r   median: {s_min_samples[len(s_min_samples)//2]:.6f}')

    renamed = rename_lm_head(new_tensors)

    work = os.path.join(WORKING_DIR, name)
    os.makedirs(work, exist_ok=True)
    sf_path = os.path.join(work, 'adapter_model.safetensors')
    save_file(renamed, sf_path)
    print(f'  wrote {sf_path} ({os.path.getsize(sf_path)/1024/1024:.1f} MB)')

    cfg_path = os.path.join(work, 'adapter_config.json')
    patch_config(os.path.join(step_to_dir[steps[-1]], 'adapter_config.json'), cfg_path)

    zip_path = os.path.join(SUBMIT_DIR, f'submission_{name}.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(sf_path, arcname='adapter_model.safetensors')
        zf.write(cfg_path, arcname='adapter_config.json')
    print(f'  -> {zip_path}  ({os.path.getsize(zip_path)/1024/1024:.1f} MB)')
    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
    print(f'  CRC {"OK" if bad is None else "FAIL: "+bad}')

    os.remove(sf_path); os.remove(cfg_path); os.rmdir(work)
    del all_t, new_tensors, renamed
    if torch.cuda.is_available(): torch.cuda.empty_cache()

print('\n=== Summary — SVD soup zips ===')
for f in sorted(glob.glob(os.path.join(SUBMIT_DIR, 'submission_svd-*.zip'))):
    print(f'  {f}  ({os.path.getsize(f)/1024/1024:.1f} MB)')
