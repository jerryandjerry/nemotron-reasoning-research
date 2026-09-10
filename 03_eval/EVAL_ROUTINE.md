# Evaluation Routine

## Prerequisites

- Read `kaggle.env` fresh before every run for credentials
- Val set: `01_data/archive/train_10-90_split/260407_val.csv` (950 samples). Has both `category` (official 6) and `huikang_category` (9 finer). **Report by `huikang_category`.** Kaggle dataset: `jerry4083/nemotron-val-huikang-950-260407`.
- Eval params (confirmed by Kaggle staff): `temperature=0.0`, `max_tokens=7680`, `max_model_len=8192`, `max_num_seqs=64`, `top_p=1.0`
- Official metric: `nvidia-nemotron-metric.ipynb` — never modify `extract_final_answer()` or `verify()`

## Eval scripts (top-level in 03_eval/)

- method 1: `eval_kaggle_nemotron.ipynb` + `kernel-metadata.json` — Kaggle kernel eval for Nemotron + LoRA
- method 2: `eval_api.py` — API eval (Gemini, DeepSeek, etc.)
- method 3: `eval_autodl_vllm.py` — AutoDL remote GPU eval via vLLM

---

## METHOD 1(DEFAULT): Kaggle Kernel Eval (Nemotron + LoRA)

### Step 0: Look up submission on Kaggle

```bash
kaggle competitions submissions nvidia-nemotron-model-reasoning-challenge
```
Find the matching submission by description/date to get the public LB score.

### Step 1: Create eval folder

Folder name = current `YYMMDDHHMM` (run `date +"%y%m%d%H%M"`), NOT the study folder's timestamp.

```
03_eval/<YYMMDDHHMM>_<short-name>/
├── kernel-metadata.json
├── eval_kaggle_nemotron.ipynb   (cp from 03_eval/)
```

### Step 2: Provide adapter (one of two ways)

**Option A: From Kaggle kernel output**
- Set `"kernel_sources": ["jerry4083/<kernel-slug>"]`

**Option 0 (CHECK FIRST): reuse an already-uploaded adapter.** If this adapter's submission.zip was uploaded as a dataset in a prior eval (`kaggle datasets list --mine --search "<name>"`), just reference that existing `jerry4083/nemo-...` dataset — skip the upload entirely.

**Option B: From local submission.zip**
- **ONLY upload the exact submission.zip. Other files are CONFIDENTIAL.**
- Upload from a **LOCAL temp dir on C:\**, NOT the Synology folder — uploading directly from Synology fails with "Permission denied" because Synology re-syncs (locks) the new file mid-upload.
1. Copy submission.zip + a `dataset-metadata.json` into `C:\Users\...\AppData\Local\Temp\<name>_upload\` (use PowerShell `Copy-Item`).
   - **Synology cloud-eviction:** the source may be a cloud-only placeholder (3.8GB shown but bytes not local). If `cp`/`Copy-Item` fails with **"network unavailable"**, the file must hydrate first — pin it in Synology Drive Client ("Always keep on this device") and wait, or open its folder. A copy that *takes time* = it's reading real bytes (good); an instant 0-byte result = failed.
   - **Always foreground the copy and VERIFY `dest size == source size` before proceeding.** Never background a Synology copy (silent 0-byte fail).
2. `dataset-metadata.json`: `{"title": "nemo-<name>", "id": "jerry4083/nemo-<name>", "licenses": [{"name": "CC0-1.0"}]}`
3. `cd <temp_dir> && kaggle datasets create -p . -r zip`
4. Wait for dataset to register: `until kaggle datasets list --mine --search "<name>" | grep -q "<name>"; do sleep 20; done`
5. **Delete the temp dir** once the dataset has registered: `rm -rf <temp_dir>` (these are 3.8GB each — don't leave them in C:\Temp).
6. Add dataset to `"dataset_sources"` in kernel-metadata.json

### Step 3: Write kernel-metadata.json

For Option A, add the kernel slug to `kernel_sources`.
For Option B, add the uploaded dataset to `dataset_sources`.
Always include `competition_sources: ["nvidia-nemotron-model-reasoning-challenge"]`.

```json
{
  "id": "jerry4083/<eval-folder-name>",
  "title": "<eval-folder-name>",
  "code_file": "eval_kaggle_nemotron.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": false,
  "keywords": ["gpu"],
  "dataset_sources": [
    "jerry4083/nemotron-val-huikang-950-260407",
    "dennisfong/nvidia-nemotron-offline-packages",
    "jerry4083/<uploaded-dataset-name>"
  ],
  "kernel_sources": [],
  "competition_sources": ["nvidia-nemotron-model-reasoning-challenge"],
  "model_sources": ["metric/nemotron-3-nano-30b-a3b-bf16/Transformers/default/1"],
  "docker_image": "gcr.io/kaggle-private-byod/python@sha256:9fa0da194fad2241d3f01a80581cbecbd3a258b4d1b695e2cbbbc62a0fd205ac",
  "machine_shape": "NvidiaRtxPro6000"
}
```

### Step 4: Update notebook title cell

Change cell-0 markdown to describe this eval run.

### Step 5: Push and monitor

Only push after dataset finishing uploading, otherwise push will fail.

```bash
kaggle kernels push -p <eval_folder>
# If dataset source rejected: wait 3 min, re-push
kaggle kernels status jerry4083/<kernel-id>   # every 5 min (sleep 300) while in flight
# CRITICAL: after pushing, confirm the push returned a kernel URL, then poll status
# ONCE to confirm the kernel exists and is RUNNING before claiming it's running.
# Never chain copy->upload->push behind a size-poll; verify each step's result first.
# If status API returns empty/403/500 (Kaggle outages happen), detect completion by
# trying `kaggle kernels output <id>` — the CSV only downloads once the kernel ends.
```

### Step 6: Save snapshot

```bash
# Download CSV
kaggle kernels output jerry4083/<kernel-id> -p <eval_folder>
# Download notebook + metadata
kaggle kernels pull jerry4083/<kernel-id> -p <eval_folder> -m
# Download log (CLI fails with charmap error, use Python)
python -c "
import requests
r = requests.get('https://www.kaggle.com/api/v1/kernels/output?userName=jerry4083&kernelSlug=<kernel-id>',
                 headers={'Authorization': 'Bearer <token>'})
with open('<eval_folder>/<kernel-id>.log', 'w', encoding='utf-8') as f:
    f.write(r.json()['log'])
"
```

### Step 7: Check results and update leaderboard

```bash
python -c "
import pandas as pd
df = pd.read_csv('<eval_folder>/val_eval_results.csv')
print(f'Overall: {df[\"correct\"].mean():.4f}')
for cat in sorted(df['huikang_category'].unique()):
    c = df[df['huikang_category']==cat]
    print(f'  {cat}: {c[\"correct\"].sum()}/{len(c)} = {c[\"correct\"].mean():.4f}')
"
```

Show per-category results to user, then append to `03_eval/leaderboard.csv`:
```
date,kernel,model,author,setup,bit_manipulation,equation_transformation,gravitational_constant,number_conversion,text_encryption,unit_conversion,overall,public_lb
```
- author: "Jerry" for locally trained adapters, check kernel source for others (e.g. konbu17, huikang)

---

## METHOD 1B: Full-set token+prob eval (golden_stripped)

Same kernel mechanics as Method 1, but a bigger run that captures per-token data. Used for analysing model confidence, not for the leaderboard.

- **Dataset:** `jerry4083/huikang-golden-stripped-4col-6906-260514` (6,906 rows; the SAME full set every time — see memory `feedback_consistent_full_set`). Cols: id, prompt, answer, category.
- **Template:** copy from the most recent `*_on_huikang_golden_stripped/eval_kaggle_nemotron.ipynb` (has chunked save + resume + token/prob capture).
- **Output CSV** `<name>_with_probs.csv` columns: `id, prompt, answer, category, model_answer, accuracy, tokens (JSON list of token strings), probs (JSON list of per-token probs = exp(logprob)), raw_output (TRUE raw via tokenizer.decode(token_ids, skip_special_tokens=False) — keeps </think>, <|im_end|>), output_token_len`.
- **Chunked save every 64 rows** to `/kaggle/working/<name>_with_probs.csv` + resume support — survives the ~5-6h runtime under Kaggle's 12h limit.
- **Runtime ~5-6h.** Poll status every 5 min; if status API is flaky, detect completion by trying `kaggle kernels output`.
- **Verify after download:** `len(tokens)==len(probs)==output_token_len` for all rows; `raw_output` ends with `<|im_end|>`.
- Report per-`category` (the golden_stripped 9 categories ARE the huikang categories). Do NOT add to `leaderboard.csv` (that's 950val + LB only).

---

## METHOD 2: API Eval (Gemini, DeepSeek, etc.)

Script: `03_eval/eval_api.py` — edit config at top (API key, model name, pricing).
Requires user approval before running. Run 1 sample per category first.
Results saved to `03_eval/YYMMDD_<model>_950val/`.

---

## METHOD 3: AutoDL Remote GPU Eval (non-Nemotron models)

Script: `03_eval/eval_autodl_vllm.py`
Setup guide: `03_eval/AUTODL_SETUP.md`
Connect via paramiko (see `kaggle.env` for SSH creds). Always use `PYTHONIOENCODING=utf-8`.
Requires user approval before running. Run `TEST_MODE=1` (1 sample per category) first.

```bash
# Test run
TEST_MODE=1 python eval_autodl_vllm.py
# Full run
nohup python eval_autodl_vllm.py > eval.log 2>&1 &
```

---

## Rules

- **Never upload anything beyond the exact file specified** — other files are confidential
- **Never change model choice** — use exactly what user specifies, never substitute
- **Never run without approval** — always get explicit user approval before GPU jobs
- **Never modify official metric functions** — `extract_final_answer()` and `verify()` are verbatim from official metric
