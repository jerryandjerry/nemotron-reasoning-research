# 260408 QLoRA 4-bit Training — Version Log

Kaggle kernel: `jerry4083/260408-qlora-4bit-test`
GPU: NvidiaRtxPro6000 (Blackwell, sm_120)
Goal: QLoRA 4-bit training of nemotron-3-nano-30b-a3b using same data/recipe as the 0.72 LoRA scheme.
Philosophy: fast iteration with 4-bit QLoRA, find best config/data, then switch to bf16 for final training.

---

## Core Problems Encountered

1. **Kaggle API auth**: `KGAT_` prefixed tokens are access tokens, not legacy API keys. Must use `KAGGLE_API_TOKEN` env var (Bearer auth), not `KAGGLE_KEY`.
2. **Blackwell sm_120 + bitsandbytes**: bitsandbytes CUDA kernels do NOT support Blackwell sm_120 — not even the latest preview wheel (1.33.7). All BNB approaches fail with "CUDA error: no kernel image is available for execution on the device."
3. **mamba_ssm import**: The original model's `modeling_nemotron_h.py` (line 66) does `raise ImportError("mamba-ssm is required...")` if mamba_ssm is not installed. mamba_ssm is NOT on the Kaggle docker image.
4. **openNemo weight compatibility**: empero-ai/openNemo has pure-PyTorch modeling code (no mamba_ssm). Weights are binary-compatible with original Nemotron checkpoint — can overlay openNemo .py files + config.json on original weights.
5. **Disk space**: Cannot copy ~60GB model weights — must use symlinks.

---

## Version History

### v1–v13: Early iterations (API auth, model mounting)
- Solved Kaggle API authentication (KGAT_ = access token, use KAGGLE_API_TOKEN)
- Tried various model_sources configurations
- v14: openNemo HF model reference not mountable via model_sources — doesn't work
- Switched to original model (`metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1`)

### v14–v16: Model path fixes
- v14: Model not mounting (openNemo HF reference doesn't work in model_sources)
- v15: Switched to original metric/nemotron model
- v16: Wrong model variation path — `bf16/1` should be `default/1`. Fixed.

### v17: Disk space exceeded
- Used `shutil.copytree` to copy ~60GB model — disk full
- **Fix**: Switch to symlinks

### v18: First bitsandbytes CUDA failure
- Symlinks working, model loads, but bitsandbytes 4-bit quantization fails
- Error: `CUDA error: no kernel image is available for execution on the device`
- Root cause: BNB CUDA kernels don't have sm_120 (Blackwell) support
- Tried `BNB_BACKEND=pytorch` — didn't help

### v19: Same BNB CUDA error
- Further BNB_BACKEND attempts, same failure

### v20: BNB_CUDA_VERSION=128
- Set `BNB_CUDA_VERSION=128` to force loading CUDA 12.8 library
- Library loaded but still lacks sm_120 kernel support (bnb 0.49.2 was released before sm_120 existed)
- Same CUDA error

### v21: bitsandbytes 1.33.7 preview wheel + openNemo overlay
- Downloaded and uploaded `bitsandbytes-1.33.7.preview` wheel (latest at time)
- Also tried overlaying openNemo config.json on original weights
- BNB preview wheel STILL fails on Blackwell sm_120
- Additionally, openNemo config.json weight structure mismatch with original checkpoint caused missing weight errors
- **Conclusion**: bitsandbytes is a dead end on Blackwell. User said "try other quantize method."

### v22: HQQ (Half-Quadratic Quantization) — first attempt
- Switched from bitsandbytes to HQQ for 4-bit quantization
- HQQ uses pure PyTorch operations (no custom CUDA kernels) via `HQQBackend.PYTORCH`
- Built and uploaded `hqq-0.2.8.post1` wheel to `jerry4083/bitsandbytes-offline` dataset
- Used original model directly (no openNemo overlay)
- **Result: ERROR** — `ImportError: mamba-ssm is required by the Mamba model but cannot be imported`
- The original model's `modeling_nemotron_h.py` imports mamba_ssm at module level
- HQQ loads the model in bf16 first before quantizing, so the model code still needs to import

### v23: HQQ + openNemo modeling code overlay (PENDING)
- Same HQQ approach as v22
- Added openNemo modeling code overlay: symlink original weights + copy openNemo .py/.json files
- openNemo code has no mamba_ssm dependency (pure PyTorch reimplementation)
- Weights are binary-compatible (confirmed by openNemo docs: "All weights are binary-compatible — load original checkpoints directly")
- **Status: RUNNING** — awaiting result

---

## Current Notebook Structure (v23)

- **Cell 0**: Markdown header (HQQ description)
- **Cell 1**: Setup — install hqq wheel, datasets, trl, peft (no bitsandbytes)
- **Cell 2**: Imports — HqqConfig, HQQBackend.PYTORCH, triton ptxas fix
- **Cell 3**: Config — symlink original weights + overlay openNemo code to `/kaggle/working/model_merged`
- **Cell 4**: Dataset — same filtering as 0.72 recipe (RELEVANT_KEYWORDS)
- **Cell 5**: Tokenizer & prompt formatting, token length filter
- **Cell 6**: Load model with HQQ 4-bit (HqqConfig nbits=4, group_size=64, axis=1), prepare_model_for_kbit_training, LoRA (rank=32, all-linear)
- **Cell 7**: LiveProgressCallback
- **Cell 8**: SFTTrainer training loop
- **Cell 9**: Save adapter, fix base_model_name_or_path, verify weight norms
- **Cell 10**: Create submission.zip
- **Cell 11**: Quick inference sanity check (3 samples)
- **Cell 12**: Training summary printout

## Kaggle Datasets Used

- `kienngx/nemotron-30b-competition-trainingdata-cot-labels` — training data CSV
- `dennisfong/nvidia-nemotron-offline-packages` — offline pip packages
- `jerry4083/bitsandbytes-offline` — contains hqq-0.2.8.post1 wheel (+ old bnb wheels)
- `jerry4083/opennemo-modeling-code` — openNemo .py and .json files

## Kaggle Model Sources

- `metric/nemotron-3-nano-30b-a3b-bf16/transformers/default/1` — original model weights (~60GB)

## Key Files

- `kernel-metadata.json` — Kaggle kernel config (GPU: NvidiaRtxPro6000, docker image pinned)
- `qlora-4bit-train.ipynb` — main training notebook
- `opennemo_code/` — local copy of openNemo modeling files
- `bnb_wheel/` — local copy of wheels uploaded to jerry4083/bitsandbytes-offline

### v23: HQQ + openNemo overlay — mamba_ssm fix
- Added openNemo modeling code overlay to avoid mamba_ssm import
- Symlink original weights + copy openNemo .py/.json to `/kaggle/working/model_merged`
- **Result: ERROR** — `NotImplementedError: QuantizationMethod.HQQ is not available yet and will be supported soon.`
- The Kaggle transformers version has a new loading pipeline (`get_quantize_ops()`) but the HQQ quantizer doesn't implement it
- Root cause: transformers ~4.51+ added `get_quantize_ops` to `core_model_loading.py` but `HqqHfQuantizer` never overrode the base class method

### v24: Quanto (optimum-quanto) int4 — PENDING
- Switched from HQQ to **Quanto** (`QuantoConfig(weights="int4")`)
- Quanto's quantizer DOES implement `get_quantize_ops()` — confirmed in transformers source
- Quanto is also pure PyTorch (no custom CUDA kernels) — should work on Blackwell
- Still using openNemo code overlay (no mamba_ssm)
- `optimum-quanto` may or may not be pre-installed on Kaggle docker; cell 1 tries to import/install it
- **Status: RUNNING** — awaiting result

---

## Quantization Methods Tried

| Method | Why it failed | CUDA kernels needed? |
|--------|--------------|---------------------|
| bitsandbytes (0.49.2) | sm_120 not in CUDA kernels | YES |
| bitsandbytes (1.33.7 preview) | Same — still no sm_120 | YES |
| HQQ (0.2.8) via HqqConfig | `get_quantize_ops` not implemented in transformers' HqqHfQuantizer | No (pure PyTorch) |
| Quanto via QuantoConfig | **Testing** — `get_quantize_ops` IS implemented | No (pure PyTorch) |

## If v24 Fails — Next Steps

1. If `optimum-quanto` not installed: build and upload the wheel to bitsandbytes-offline dataset
2. If Quanto has other issues: monkey-patch `HqqHfQuantizer.get_quantize_ops` or use HQQ's `AutoHQQHFModel.quantize_model()` directly
3. Fallback: train on AutoDL with Ampere GPU (A100) where bitsandbytes works fine
   - AutoDL credentials in `D:\SynologyDrive\00_Kaggle\2026_Nemotron\kaggle.env`
   - Recommended: PyTorch 2.4.x, CUDA 12.1/12.4, Python 3.10/3.11
4. Nuclear option: skip 4-bit, do bf16 LoRA on Kaggle (proven to work — the 0.72 scheme used this)
