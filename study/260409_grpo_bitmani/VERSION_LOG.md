# 260409 GRPO Bit Manipulation — Version Log

Kaggle kernel: `jerry4083/260409-grpo-bitmani`
Goal: GRPO training on bit_manipulation puzzles, starting from 0.72 SFT adapter
GPU: NvidiaRtxPro6000 (Blackwell, sm_120, 96GB)

---

## Core Problem: vLLM ABI Incompatibility

The pre-built vllm-0.18.0 wheel (`_C.abi3.so`) was compiled against a different PyTorch.
Kaggle docker has `torch 2.12.0.dev20260324+cu128`. The vLLM wheel expects a symbol
`_ZN3c1013MessageLoggerC1EPKciib` (c10::MessageLogger constructor) that doesn't exist
in this torch dev build.

**What works:**
- `import vllm` works IF we preload libtorch libs with `ctypes.CDLL(RTLD_GLOBAL)` first
- BUT TRL's `from trl import GRPOTrainer` triggers `vllm/platforms/cuda.py` → `import vllm._C` which still crashes

**What doesn't work:**
- Pre-built vllm-0.18.0 wheel (ABI mismatch)
- TRL 0.29.1 supports vllm 0.10.2-0.17.1 (0.18.0 too new)
- metric/nvidia-metric-utility-script as kernel_source (downgrades torch to 2.10.0, breaks mamba_ssm)
- Building from source on Kaggle (dataset upload issues, build fails at metadata generation)

**The metric notebook's approach:** Uninstalls system torch, extracts `/kaggle/usr/lib/notebooks/metric/nvidia_metric_utility_script` to `/tmp`, uses that torch+vllm bundle. But that path only exists for the metric kernel — we can't access `metric/nvidia-metric-utility-script` kernel source (private).

---

## Version History

### v1-v3: Setup issues
- v1: mamba_ssm not found (missing mayukh18/nemotron-packages dataset)
- v2: Added datasets, mamba_ssm works, but train.csv path wrong
- v3: Fixed CSV path, all setup works. BUT model loaded, training started WITHOUT vLLM (~26 min/step)

### v4-v6: First GRPO training (no vLLM, HF generate)
- v4: 1602 prompts × 4 gens × 2048 max_completion — WAY too slow (26 min/step × 3204 steps)
- v5: 200 prompts with vLLM flag — but vLLM not installed, fell back to HF generate
- v6: 10 prompts, vLLM flag — "vLLM not available", fell back to HF generate

**v6 GRPO actually ran (slowly):**
- Step 1: 26 min, Step 2: 26 min
- Step 5/20: loss=0.0023, elapsed=78.6min
- Step 10/20: loss=0.0677, elapsed=131.1min
- Model generates 4000+ tokens without \boxed{} — SFT adapter doesn't produce format for bit_manipulation

### v7-v9: Attempting vLLM install
- v7: `pip install vllm` from PyPI — DNS fails (no internet with competition_sources)
- v8: Added metric/nvidia-metric-utility-script — broke torch (downgraded to 2.10.0), mamba_ssm ABI crash
- v9: Diagnostic — vLLM wheels found in offline packages but not pre-installed

### v10: Install vLLM from offline wheel
- Installed vllm-0.18.0 from dennisfong wheel with --no-deps
- `import vllm` succeeds! But `from trl import GRPOTrainer` crashes:
  `_C.abi3.so: undefined symbol: _ZN3c1013MessageLoggerC1EPKciib`

### v11: Fallback to HF generate (512 completion)
- 10 prompts, max_completion=512 — cancelled by user

### v12-v14: Trying metric notebook approach
- v12: Uninstall system torch + extract metric utility — path doesn't exist
- v13: Diagnostic — /kaggle/usr/lib/notebooks/ is empty
- v14: Full diagnostic — utility script is at `/kaggle/usr/lib/nvidia-utility-script/` (no vLLM there)

### v15-v18: vLLM build from source attempts
- v15-v16: vllm-source-018 dataset failed (filenames with = and , characters)
- v17: Preload libtorch with ctypes — `import vllm` works! But TRL import still crashes at _C
- v18: Import TRL before installing vllm (delayed import) — still crashes when GRPOTrainer accessed

### v19-v24: More build attempts
- v19-v22: vllm-src-clean/v2 datasets fail to create (Kaggle processing issues)
- v23-v24: Uploaded vllm_source.tar.gz to bitsandbytes-offline dataset — tarball not found (version not propagated)

---

## What Works (proven in v6)

Without vLLM, GRPO training works but is extremely slow:
- HF generate: ~2 tok/s (NemotronH cache issue)
- 10 prompts × 4 gens × 2048 tokens ≈ 5+ hours
- With max_completion=512: ~2 hours for 10 prompts

## What Needs to Happen Next

1. **Build vLLM wheel on a machine with torch 2.12.0.dev+cu128** — the Kaggle docker itself
   - Upload vLLM source as a Kaggle dataset (fix the tarball propagation issue)
   - OR build on AutoDL after installing matching torch
2. **Alternative: Use unsloth** — discussion thread says it works with 2 hrs/epoch for 3000-token sequences
3. **Alternative: Run GRPO on AutoDL** — different torch, standard vllm works

## What Was Achieved

1. **vLLM import works** with ctypes preload trick (v17)
2. **vLLM source builds past metadata** with SETUPTOOLS_SCM_PRETEND_VERSION (v31)
3. **cmake runs** with VLLM_CUTLASS_SRC_DIR + TRITON_KERNELS_SRC_DIR (v38)
4. **cmake needs 3 more deps**: FlashMLA, qutlass, flash-attention — all have env var overrides
5. All 6 deps cloned locally at `study/260409_vllm_build/`: vllm_source, cutlass_src, triton_kernels_src, flashmla_src, qutlass_src, flash_attn_src
6. **Dataset versioning doesn't work** — Kaggle ignores version updates. Must create NEW datasets.
7. **New dataset creation takes 10+ min** and sometimes fails silently with "invalid character" errors

## Clear Path Forward

1. Create a NEW Kaggle dataset with ALL 6 source dirs (use `dir_mode='zip'`)
2. WAIT until dataset appears in `api.dataset_list()` before pushing kernel
3. Set these env vars in the build:
   - `SETUPTOOLS_SCM_PRETEND_VERSION=0.18.0`
   - `VLLM_CUTLASS_SRC_DIR` → cutlass_src
   - `TRITON_KERNELS_SRC_DIR` → triton_kernels_src
   - `FLASH_MLA_SRC_DIR` → flashmla_src
   - `QUTLASS_SRC_DIR` → qutlass_src
   - `VLLM_FLASH_ATTN_SRC_DIR` → flash_attn_src
4. Build with `pip install <dir> --no-build-isolation --no-deps`
5. C++ compilation will take 5-10+ min on Blackwell GPU
6. After vLLM builds, use `use_vllm=True, vllm_mode='colocate'` in GRPOConfig

## Local Source Files (already cloned, cleaned)

All at `D:\SynologyDrive\00_Kaggle\2026_Nemotron\study\260409_vllm_build\`:
- `vllm_source/` — vLLM v0.18.0 (36MB, no .git, configs with bad chars removed)
- `cutlass_src/` — NVIDIA CUTLASS v4.2.1 (46MB, docs/test/media removed)
- `triton_kernels_src/` — triton v3.6.0 triton_kernels (small)
- `flashmla_src/` — vllm-project/FlashMLA (1.6MB)
- `qutlass_src/` — IST-DASLab/qutlass (1.5MB)
- `flash_attn_src/` — vllm-project/flash-attention (24MB)

## v43-v46: cmake errors during C++ compilation

- v43: All 6 deps found. cmake configures (16s). Fails: `CUDA::cuda_driver` target not found
- v44: Added CUDA_HOME + CUDAToolkit_ROOT. Still fails — log only 16 lines (missing -v)
- v45-v46: With -v flag. cmake fails exit code 1 in 0.6 min. 1982 log lines but actual error in middle

**Next step:** Print lines 1700-1850 of build log to find actual cmake error. Likely still CUDA::cuda_driver or similar missing cmake target. May need `find_package(CUDAToolkit)` or the CUDA stubs library.

## Key Kaggle Environment Info

```
PyTorch: 2.12.0.dev20260324+cu128
CUDA: 12.8
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition
Python: 3.12
TRL: 0.29.1
mamba_ssm: 2.3.1
vllm wheel available: 0.18.0 (ABI incompatible)
torch location: /kaggle/usr/lib/nvidia-utility-script/torch
```

## Files

- `kernel-metadata.json` — kernel config
- `grpo-bitmani.ipynb` — GRPO training notebook (11 cells)
- Based on nemotron-ultimate-sft-grpo-v3 (0.70 score) reference notebook
