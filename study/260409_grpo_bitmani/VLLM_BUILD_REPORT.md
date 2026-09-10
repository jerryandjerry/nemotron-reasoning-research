# vLLM Build Report for Kaggle RTX PRO 6000 (Blackwell sm_120)

> Date: 2026-04-10
> Kernel: jerry4083/260409-grpo-bitmani (v1-v55)
> Goal: Get vLLM working with TRL GRPOTrainer on Kaggle's Blackwell docker

---

## The Problem

Kaggle's docker has `torch 2.12.0.dev20260324+cu128` — a **custom NVIDIA/Kaggle nightly build**. The pre-built vLLM 0.18.0 wheel (from offline packages) has `_C.abi3.so` compiled against a different PyTorch, causing:

```
ImportError: vllm/_C.abi3.so: undefined symbol: _ZN3c1013MessageLoggerC1EPKciib
```

This is `c10::MessageLogger` constructor — ABI changed between PyTorch versions.

---

## What Was Tried (55 versions)

### Phase 1: Pre-built wheel (v6-v18)
| Version | Approach | Result |
|---------|----------|--------|
| v6-v7 | `pip install vllm` from PyPI | DNS fails (no internet with competition_sources) |
| v8 | metric/nvidia-metric-utility-script kernel_source | Downgrades torch to 2.10, breaks mamba_ssm |
| v9 | Diagnostic — found vllm wheel in offline packages | vllm-0.18.0 wheel exists but not installed |
| v10 | Install wheel with `--no-deps` | `import vllm` succeeds but TRL import crashes at `_C.abi3.so` |
| v17 | Preload libtorch with ctypes RTLD_GLOBAL | `import vllm` works directly but TRL's deeper import chain still hits `_C` |
| v18 | Import TRL before installing vllm (avoid eager import) | Still crashes when `from trl import GRPOTrainer` triggers vllm submodule |

**Conclusion:** The pre-built wheel is ABI-incompatible. No workaround can fix the `_C.abi3.so` undefined symbol.

### Phase 2: Build from source on Kaggle (v19-v52)
| Version | Approach | Result |
|---------|----------|--------|
| v19-v22 | Upload vllm source as Kaggle dataset | Dataset creation issues — filenames with `=`,`,`,`[`,`]` rejected by Kaggle |
| v29 | Clean source dataset `vllm-source-clean` | Source found, but `setup.py` missing (uses `pyproject.toml`) |
| v31 | Set `SETUPTOOLS_SCM_PRETEND_VERSION=0.18.0` | Metadata passed! But cmake fails — git clone cutlass (no internet) |
| v33-v38 | Bundle cutlass_src + triton_kernels + flashmla + qutlass + flash_attn | Progressively fixed: cutlass ✓, triton ✓, flashmla ✓, all deps found |
| v43 | All 6 deps found, cmake configures (16s) | `CUDA::cuda_driver` target not found |
| v44-v48 | Set CUDA_HOME, CUDAToolkit_ROOT | Still fails — `libcuda.so` not on filesystem (injected at runtime) |
| v49-v50 | Symlink `/usr/local/cuda/compat/libcuda.so` to stubs | Read-only filesystem → create in `/tmp/cuda_stubs` |
| v51 | `/tmp/cuda_stubs` symlink works! cmake passes! CUDA compilation starts! | **ptxas error: f16 requires .target sm_53** — wrong arch |
| v52 | Set `TORCH_CUDA_ARCH_LIST=12.0` | Same ptxas error — env var not respected by cmake |

**Conclusion:** Building on Kaggle gets very close but the CUDA architecture flag isn't propagating to nvcc. The cmake configuration uses `-- Autodetected CUDA architecture(s): 12.0` but ptxas still compiles for a lower arch.

### Phase 3: Build on AutoDL 5090 (v53-v55)
| Step | Result |
|------|--------|
| Connect to 5090 instance | ✓ RTX 5090, sm_120, Python 3.12 |
| Install torch 2.12.0.dev20260324+cu128 | ✓ Exact same version as Kaggle |
| Clone vLLM v0.18.0 via GitHub mirror | ✓ (direct GitHub fails from China) |
| Install cmake 4.3.1, setuptools_scm, packaging | ✓ |
| Build vLLM with `TORCH_CUDA_ARCH_LIST=12.0` | ✓ **BUILD SUCCESSFUL** after ~85 min |
| Save wheel: `vllm-0.18.0+cu128-cp312-cp312-linux_x86_64.whl` (578MB) | ✓ |
| Upload wheel to Kaggle dataset `jerry4083/vllm-wheel-sm120` | ✓ |
| v53: Install wheel on Kaggle | Kaggle strips `+` from filename → pip rejects version |
| v54: Fix filename (copy with `+` restored) | ✓ vLLM 0.18.0 installed and imported! |
| v55: Install msgspec deps + wheel | All packages pass! But... |
| v55: `from trl import GRPOTrainer` | **SAME `_C.abi3.so` undefined symbol** |

**Conclusion:** Even with the exact same torch version string, the Kaggle docker's torch is a **custom build** with different ABI than the public PyPI nightly. Building vLLM externally against the public nightly produces an incompatible `_C.abi3.so`.

---

## Root Cause

Kaggle's `torch 2.12.0.dev20260324+cu128` is NOT the same binary as `pip install torch==2.12.0.dev20260324+cu128 --index-url .../nightly/cu128`. It's a **custom NVIDIA/Kaggle build** bundled in the docker image at `/kaggle/usr/lib/nvidia-utility-script/torch/`. The C++ ABI differs from the public nightly.

The competition metric notebook solves this by using `metric/nvidia-metric-utility-script` which has its own matching torch+vllm bundle — but that kernel_source is private and unavailable to participants.

---

## What DOES Work

1. **vLLM Python package imports** (with ctypes preload or `--no-deps` install)
2. **TRL without vLLM** — GRPO runs with HF generate at ~2 tok/s
3. **v6 proved GRPO training works** — 10 steps completed, loss decreased, just extremely slow

---

## Remaining Options

1. **Build vLLM directly on Kaggle docker** — the v51 approach was closest. Need to fix the `TORCH_CUDA_ARCH_LIST` propagation to nvcc. The cmake log shows `-- Autodetected CUDA architecture(s): 12.0` but ptxas gets the wrong arch. May need to patch vllm's `CMakeLists.txt` to force `set(CMAKE_CUDA_ARCHITECTURES 120)`.

2. **Extract Kaggle's torch .so files and build against them** — copy `/kaggle/usr/lib/nvidia-utility-script/torch/lib/*.so` from Kaggle, replicate the exact environment on AutoDL, build vLLM there.

3. **Use Unsloth** — discussion says it works with ~2 hrs/epoch. Avoids vLLM entirely.

4. **Run GRPO on AutoDL** — the 5090 has a working vLLM build. But the user said the 5090 is only for building, not training.

---

## Key Environment Info

```
Kaggle Docker:
  PyTorch: 2.12.0.dev20260324+cu128 (CUSTOM BUILD at /kaggle/usr/lib/nvidia-utility-script/torch/)
  CUDA: 12.8
  Python: 3.12
  GPU: NVIDIA RTX PRO 6000 Blackwell (sm_120, 96GB)
  TRL: 0.29.1
  mamba_ssm: 2.3.1
  vllm wheel available: 0.18.0 (ABI INCOMPATIBLE)
  libcuda.so: /usr/local/cuda/compat/libcuda.so (runtime injected)
  /usr/local/cuda: READ-ONLY filesystem
  Internet: BLOCKED with competition_sources
  
AutoDL 5090:
  PyTorch: 2.12.0.dev20260324+cu128 (PUBLIC NIGHTLY — different ABI)
  CUDA: 12.8
  Python: 3.12
  GPU: RTX 5090 (sm_120, 32GB)
  vLLM: 0.18.0 (built from source, works locally)
```

## Files

- `vllm-0.18.0+cu128-cp312-cp312-linux_x86_64.whl` — built on 5090, 578MB (ABI mismatch with Kaggle)
- Kaggle dataset `jerry4083/vllm-wheel-sm120` — uploaded wheel
- Kaggle dataset `jerry4083/vllm-full-build` — vllm source + cutlass + triton_kernels + flashmla + qutlass + flash_attn
- All source at `study/260409_vllm_build/`
