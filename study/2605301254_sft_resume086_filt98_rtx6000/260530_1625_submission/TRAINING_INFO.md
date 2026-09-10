# TRAINING_INFO — resume086_filt98

**Study folder:** `study/2605301254_sft_resume086_filt98_rtx6000/`
**Training finished:** 2026-05-30 16:25 CDT (instance: 2026-05-31 05:25 CST+0800)
**Public LB:** *pending submission*

---

## What was trained

**Resume training from Sub #16's 0.86 LoRA adapter** on a filtered subset of the cryptnumeq data.

The intent: refine #16's weights on the **new-trace cryptarithm + numeric-equation CoTs** (which #16 has never seen) while sharply downsampling the 5 categories where #16 already trained on byte-identical CoTs (to avoid overfitting / forgetting).

---

## Config

| field | value |
|---|---|
| Base model | Nemotron-3-Nano-30B-A3B (bf16, hybrid Mamba2 + attention + 128-routed-MoE) |
| Starting LoRA | **Sub #16's 0.86 adapter** (from `submission_moe_notie_outproj.zip`, unzipped to `/root/autodl-tmp/adapter_086_init/`) |
| Framework | Unsloth FastLanguageModel + manual training loop |
| LoRA arch | r = 32, α = 32, dropout = 0; targets q/k/v/o/up/down/in/`out_proj` + lm_head |
| MoE tying | **off** (`MOE_TIE=0`) — independent rank-32 LoRA per expert |
| Mamba `out_proj` | **LIVE** — `is_fast_path_available=True` + `mixer.training=False`. Verified: `out_proj.lora_B norms (first 5): [0.7059, 0.6435, 0.6189, 0.6648, 0.6742]` (matches #16's converged signature ~0.7) |
| lm_head | manual LoRA + post-train rename `base_model.model.lm_head.` → `base_model.model.backbone.lm_head.` |
| Loss | CCE (cut_cross_entropy) |
| Optimizer | **fresh AdamW** (no optimizer state from #16 — only adapter weights), β = (0.9, 0.95), wd = 0 |
| LR schedule | **5e-5 → 0 linear over 69 steps** (gentle resume vs #16's 2e-4 fresh-train peak) |
| Batch | 32 (micro_batch = 4, grad_accum = 8) |
| Sequence length | 8192 |
| Stratified shuffle seed | 42 |
| Env flags | `MOE_TIE=0 USE_MEM_EFF=0 ADAPTER_INIT_PATH=/root/autodl-tmp/adapter_086_init` (no `FRESH_START` — that would short-circuit the adapter-load branch) |
| Transformers | 4.56.2 (see `requirements.txt`) |
| GPU | RTX PRO 6000 Blackwell (96 GB), single GPU |

---

## Data

| field | value |
|---|---|
| CSV | `260530_huikang_NumericEq_5cat15_crg3x.csv` |
| md5 | `5551b00711fae8e8d21177d77e9be52c` |
| Unique rows | **1,715** → expanded **2,201** → **69 steps** |
| GT-match | **100 %** (all rows; verified pre-training) |
| **5 "avoid-forget" cats** (`bit_manipulation, cipher, gravity, unit_conversion, numeral`) | sampled at **15 % of source rows, oversampling forced to 1**. These are byte-identical CoTs to what #16 already trained on, so heavy re-exposure was avoided. |
| **4 "new-trace" cats** (`equation_numeric_deduce/guess, cryptarithm_deduce/guess`) | kept **100 %** of source rows (new-solver CoTs #16 never saw). `cryptarithm_guess` oversampling bumped from source 2× → **3×** to lift its per-batch coverage from 57 % → 94 % of batches. |

### Per-batch distribution (replayed from saved CSV)

| category | total | avg/batch | min | max | present |
|---|--:|--:|--:|--:|--:|
| equation_numeric_deduce | 676 | 9.8 | 9 | 11 | 100 % |
| cryptarithm_deduce | 486 | 7.0 | 6 | 8 | 100 % |
| cipher (15 %) | 236 | 3.4 | 3 | 4 | 100 % |
| bit_manipulation (15 %) | 203 | 2.9 | 2 | 4 | 100 % |
| unit_conversion (15 %) | 148 | 2.1 | 2 | 3 | 100 % |
| gravity (15 %) | 146 | 2.1 | 1 | 3 | 100 % |
| equation_numeric_guess | 142 | 2.1 | 1 | 3 | 100 % |
| numeral (15 %) | 98 | 1.4 | 1 | 2 | 100 % |
| cryptarithm_guess (3×) | 66 | 1.0 | 0 | 2 | 94 % |

---

## Results

| metric | value |
|---|---|
| Steps | **69** (1 epoch on filtered data) |
| Final loss (step 69) | **0.0716** |
| Loss trajectory | step 1: 0.672 → step 11: 0.385 → step 18: 0.231 → step 33: 0.123 → step 50: 0.100 → step 69: 0.072 (clean monotonic descent — known cats already near-zero, new traces being learned) |
| Total time | **1.07 hrs (64.1 min)** |
| Peak VRAM | 71,895 MB (within 95 GB) |
| Grad spikes | **none** (max grad_norm observed: 0.79 at step 1, decayed to ~0.05 by step 69) |
| Soup save steps | `[14, 41, 55, 59, 62, 66, 69]` = 20/60/80/85/90/95/100 % of 69 (routine §2.4B compliant) |
| FIFO checkpoint saves | step 25, 50, 69 (SAVE_STEPS=25 computed per routine formula; FIFO=1 kept only the latest) |

---

## Adapter-init verification (the load-bearing checks)

| check | result |
|---|---|
| `External adapter init: /root/autodl-tmp/adapter_086_init` | ✓ branch fired |
| `Loaded external adapter: 12011 tensors (missing=6242, unexpected=0)` | ✓ all LoRA tensors loaded; 6242 missing are frozen base params (correct) |
| `out_proj.lora_B norms (first 5): [0.7059, 0.6435, 0.6189, 0.6648, 0.6742]` | ✓ matches #16's converged ~0.7 signature |
| `Forced unfused else-branch on 23 Mamba mixers (out_proj LoRA will train)` | ✓ live-out_proj trick fires |
| Stress test step 2 loss = 0.039 on bit_manipulation worst-case batch | ✓ impossible without #16 weights (fresh-init equivalent run gave 0.460) |

---

## Output artifacts on instance (21578, `/root/autodl-tmp/`)

Grouped on instance per routine §3 in `260530_1625_submission/`:

| artifact | path / size |
|---|---|
| Submission zip | `260530_1625_submission/submission_resume086_filt98_notie_outproj.zip` (3.83 GB; `adapter_config.json` + `adapter_model.safetensors` only) |
| Checkpoint zip | `260530_1625_submission/checkpoint-69_loss0.0716_lr7.25e-07.zip` (10.4 GB; adapter + optimizer + scheduler + RNG + tokenizer + training_args) |
| 7 soup-source adapters | `adapter_output_resume086_filt98_notie_outproj/adapter-{14,41,55,59,62,66,69}_loss{x}_lr{x}/` — each 4.26 GB safetensors |

---

## Local files in this folder

| file | purpose |
|---|---|
| `train_resume_086.py` | the trainer (copied from run folder) |
| `test_imports.py` | pre-flight import check (copied from run folder) |
| `train_log.txt` | full training log pulled from instance |
| `requirements.txt` | `pip freeze` from instance |
| `TRAINING_INFO.md` | this file |

---

## Cross-references

- `02_train/TRAINING_ROUTINE.md` — Step 3/4 spec this folder follows
- `02_train/260522_moe_expert_rank1_bug_report.md` — `out_proj`-live mechanism (the load-bearing #16 trait this resume preserves)
- `study/SCORE_TRACKER.md` — Sub #16 (0.86 baseline being resumed from), Sub #22-24 (cryptnumeq data lineage)
- `01_data/260530_resume_data/260530_huikang_NumericEq_5cat20_noover.csv` — wait, the actual data is `260530_huikang_NumericEq_5cat15_crg3x.csv` — see Data section above

---

## Notes (routine compliance + lessons from prior runs)

- **First attempt's bug:** initial stress test was launched with `FRESH_START=1` which short-circuited the `ADAPTER_INIT_PATH` branch. Fixed by removing `FRESH_START` from both stress and real launches — verified by `Loaded external adapter: 12011 tensors` + `out_proj.lora_B norms ~0.7` in the actual training log.
- **Step 3 → Step 4 order followed properly this time** (unlike alleqgt/cryptnumeq where the instance was shut down before small-file save). This folder was populated before any shutdown command was issued.
