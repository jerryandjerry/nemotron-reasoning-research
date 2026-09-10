# Training Routine — Agent Guidelines
**Last updated:** 2026-05-16

This is a template for any agent running training (SFT, GRPO, DPO, etc.) on AutoDL instances.

---

## Step 0: Before Training

### 0a. Create Study Folder

Create folder under `study/` with naming format:
```
{yymmddhhmm}_{task_type}_{notes}_{gpu}
```

- **Timezone:** Chicago time (CDT = UTC-5, CST = UTC-6 winter)
- **task_type:** sft, grpo, dpo, etc.
- **notes:** anything distinctive about this experiment
- **gpu:** rtx5090, rtx6000, etc.

Examples:
```
2604102138_sft_qlora_rtx6000
2604111430_grpo_bitmanip_rtx5090
2604120900_sft_filtered_cot_rtx6000
```

### 0b. Connect to AutoDL instance

Default to use the last instance port you used. Read SSH credentials from `kaggle.env` at repo root if don't remember, then connect via paramiko:

```python
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('connect.bjb1.seetacloud.com', port=<PORT>, username='root', password=<PASSWORD>, timeout=15)
```

Verify GPU and disk before proceeding: `nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader` and `df -h /root/autodl-tmp`.


### 0c. How to upload file to AutoDL instance

- **Small files (<5 MB, e.g. train script):** put paths as **literals in a python script** and run it via the **PowerShell tool** (the Bash tool is Git-Bash and mangles `/root/...` args into `C:/Program Files/Git/root/...`). Upload with `ssh.open_sftp().put(local, '/root/autodl-tmp/<name>')` (works on the mount; `paramiko.Transport`-based SFTP and base64-over-exec do not).
- **Run python on the instance:** non-login SSH exec has no `python` on PATH — use `bash -lc '...'` (activates conda) or `/root/miniconda3/bin/python`.

---

## Step 1: Stress Test

**Before uploading the new script** — clean the instance, but be precise about what to delete vs keep:

**SAFE TO DELETE** (after the user has downloaded them locally): 
- prior run outputs (`adapter_output_*`, `*.zip`, `*_submission/`), logs (`*log*.txt`, `requirements.txt`), and old scripts (`train_*.py`, `test_imports.py`, `_*.py`).

**MUST KEEP** (do NOT touch):
- `/root/autodl-tmp/Nemotron-*` or any base-model directory
- `/root/autodl-tmp/data/` — all training CSVs, tokenizer, pretok dumps
- `/root/autodl-tmp/swanlog/` — past run logs
- `/root/autodl-tmp/unsloth_compiled_cache/` — speeds up next compile
- `/root/autodl-tmp/model/` — use for continue-training sometimes
- Anything you don't explicitly recognize as a prior run artifact — if in doubt, ask the user.

Confirm with `df -h /root/autodl-tmp` that there's at least 30 GB free after cleanup.

**Pre-flight**: run a `test_imports.py` that imports every package the training script uses and prints versions — catches missing deps / ABI mismatches / arch incompatibility in seconds. Then run the stress test to verify VRAM fits.

### Rules:
1. Use the **exact same training code** — same training loop (or `Trainer`/`SFTTrainer` class), gradient checkpointing, optimizer, etc.
2. Only for stress test(not real training), **Sort dataset by token length descending and take top-K** (K ≥ `bs × max_steps × 2`) so every batch is worst-case. No shuffle of the data so we can inspect which sample cause the problem. 
3. Run **2–4 steps** with save disabled (manual loop: `STRESS_TEST=1` env caps `num_steps`; SFTTrainer: `max_steps` + `save_strategy='no'`). Match real `batch_size` + `grad_accum`.
4. Measure VRAM with **nvidia-smi** (not `torch.cuda.memory_allocated`). Log `sample_id, padded_len, content[-20:]` per micro-batch.
5. If OOM, reduce batch_size and retry.
6. **Verify `out_proj` LoRA is alive** after stress step 1, else fused Mamba kernel folded it and you're silently capped at 0.84: `assert sum(p.grad.norm().item() for n,p in model.named_parameters() if 'out_proj.lora_B' in n and p.grad is not None) > 1e-6`

### Reference (for context, not prescription):
- **Current recipe** (Nemotron-3-Nano-30B-A3B bf16 + Unsloth + manual loop + CCE + MoE tying + lm_head LoRA, r32/α32, seq=8192):
  - RTX PRO 6000 (96GB): bs=32 micro=4 ga=8 peaks at ~88.8 GB on worst-case (longest) samples. ~1.0 min/step real average.
- **Older SFTTrainer + standard CE** (no CCE): bs=8 peaked at 71GB on RTX PRO 6000; bs=12 OOM'd. Don't extrapolate to the current recipe — CCE saves ~16GB by not materializing the 128K-vocab logits tensor.

---

## Step 2: Start Training

### Before starting, confirm with user:

**1. Train name for SwanLab:**
```
{yymmddhhmm}_{task}_{method}_{key_config_summary}_{gpu}
```
Example: `260410_2138_sft_qlora_3000samp_r32_lr2e4_seq2500_adamw_bs8ga2_rtx6000`

Include the most important config items that distinguish this run from others.

**2. Training config** — show all key parameters to user before starting:
- Task type (SFT, GRPO, DPO, etc.)
- Model and quantization
- LoRA config (rank, alpha, target modules, dropout)
- Optimizer and learning rate
- Batch size and gradient accumulation
- Number of epochs
- Max sequence length
- Dataset (size, filtering, seed)
- Checkpoint strategy

**3. Per-step logging format:**
```
>> step=1 batch[0] sample_id=xxx padded_len=2482 smi=25165M content_tail: \boxed{71.22}
>> step=1 batch[1] sample_id=xxx padded_len=358 smi=25165M content_tail: \boxed{XLII}
   step=1 smi_after=70935M loss=4.2401
```

Each micro-batch must log: **sample_id, padded_len, nvidia-smi, content[-20:]** (with padding stripped).

**4. Required code features:**
- `gc.collect()` + `torch.cuda.empty_cache()` at step_begin and step_end
- Gradient clipping `clip_grad_norm_(max_norm=1.0)`
- Log VRAM (nvidia-smi) per step / micro-batch (SFTTrainer: in progress bar)
- Checkpoint folder named with metrics: `checkpoint-{step}_loss{x}_lr{x}` (extend with `_ep{x}_acc{x}` if available)
- Log file uses **append mode** (`>>`) — never overwrite
- SwanLab logging with Chicago-time run name
- `token_to_sid` lookup for mapping tokenized input back to CSV sample ID
- **Resume-from-latest-checkpoint** at script start: scan `OUTPUT_DIR` for highest-step `checkpoint-*` dir, restore adapter weights + optimizer + RNG + step counter. PEFT `save_pretrained` strips `.default.` from LoRA keys — rename them back before `load_state_dict`. Skip resume if `FRESH_START=1`.
- **`save_steps`** = `max(25, round(total_steps / 5 / 25) * 25)`. Examples: 244 → 50, 728 → 150, 3000 → 600.
- **Two save purposes — handle independently:**

  **A. Resume-training checkpoint** (always on):
  - Save full state (adapter + optimizer + scheduler + RNG + tokenizer + training args) every `save_steps`.
  - **FIFO = 1**: keep only the latest checkpoint on disk; delete previous on each save.
  - On clean training finish, force-save the final-step checkpoint (manual loop: `if step == num_steps: save()`; SFTTrainer: `FinalStepSaveCallback`). This final one is the resume-ready artifact.

  **B. LoRA soup adapters** (ALWAYS save — cheap insurance, on EVERY run):
  - Save adapter-only (no optimizer state) at **20, 60, 80, 85, 90, 95, 100%** of `total_steps` (`step = round(total_steps * pct)`); keep all on disk (no FIFO). These are exactly the 7 adapters the §3 soups consume.
  - **Save these on every run regardless of whether a soup will be built.** Training is expensive (~4 hrs); the 7 snapshots are cheap (~28 GB, deletable later) and preserve all post-hoc soup options. "No soup" / endpoint-only means skip only the post-training §3 BUILD — **NOT** these saves. **Never set `ADAPTER_SAVE_STEPS=[]`.**


**5. Training Time:** 
- **Estimated time** — calculate and show user before starting:
```
steps = num_samples / (batch_size × grad_accum)
total_time = steps × time_per_step (from stress test)
```
- **Time tracker** — record the total time spent:
```
Before the training loop: t0 = time.time()
After the loop: elapsed = time.time() - t0
log(f'Training done. Time: {elapsed/3600:.2f} hrs ({elapsed/60:.1f} min)')`
```
---

## Step 3: After Training

### Save to local study folder:

**Small files (download via SSH/SFTP):**
- Training script (e.g. `train_qlora.py`)
- `train_log.txt`
- `requirements.txt`
- `TRAINING_INFO.md` — summary of what was trained, config, results, time spent

**Large files (user downloads from AutoDL file manager):**
- `submission.zip` — for Kaggle submission (adapter_config.json + adapter_model.safetensors)
- `checkpoint-{final_step}.zip` — full checkpoint for resuming (includes adapter, optimizer, scheduler, tokenizer, rng state, training args)
- **Group both zips into a folder on the instance** named `{yymmdd_hhmm}_submission/` 

### Adapter post-processing (in training script, before zipping):
1. **Fix `base_model_name_or_path`** in `adapter_config.json` → `metric/nemotron-3-nano-30b-a3b-bf16`
2. **Rename lm_head keys** if manual lm_head LoRA was used: `base_model.model.lm_head.` → `base_model.model.backbone.lm_head.`

### Create zips on instance:
```bash
# submission.zip — auto-created by training script

# Zip final checkpoint:
cd /root/autodl-tmp/adapter_output
zip -r /root/autodl-tmp/checkpoint-{step}.zip checkpoint-{step}_{metrics}/
```

### Soup build (only if soups requested — on instance, BEFORE shutdown):

Run `svd_soup_weighted.py` (reference impl: `study/2605222039_sft_moe_outproj_rtx6000/260523_wsoup_submission/`) in **GPU mode** (the merge OOMs in AutoDL's 2 GB no-GPU cap). It builds **weighted SVD soups** from the §2.4B adapters — the best rank-`r` SVD approximation of the weighted delta sum `Σ wᵢ·Bᵢ@Aᵢ` (weights sum to 1) — into Kaggle-ready `submission_<recipe>.zip` files and drops them in `{yymmdd_hhmm}_submission/`. SFTP the script down too. Recipes:
1. **last5 soup** — the 5 highest adapters (80–100%): **0.5** on the final, **0.125** on each of the other four.
2. **wise soup** — adapters nearest **20 / 60 / 100%**: **0.1 / 0.2 / 0.7**.

### Local folder structure:
```
study/{yymmddhhmm}_{task}_{gpu}/
├── {yymmdd_hhmm}_submission/         ← timestamp = when training finished
│   ├── submission.zip                       ← user downloads (final-step adapter)
│   ├── submission_last5-soup.zip           ← user downloads (only if soups requested, ~2.8 GB)
│   ├── submission_wise-soup.zip            ← user downloads (only if soups requested, ~2.8 GB)
│   ├── checkpoint-{final_step}.zip          ← user downloads
│   ├── train_script.py                      ← via SSH
│   ├── svd_soup.py                          ← via SSH (only if soups requested)
│   ├── train_log.txt                        ← via SSH
│   ├── requirements.txt                     ← via SSH
│   └── TRAINING_INFO.md                     ← via SSH
├── test_imports.py
├── stress_test.py
└── stress_test_step_log.txt
```

---

## Step 4: Shutdown Instance

**After saving all small files,**

fire an **instance-side deadman shutdown (15 min)** so it powers off even when your shutdown command fail, then issue the immediate shutdown:

```bash
# On the instance, via SSH:
nohup sh -c 'sleep 900 && shutdown -h now' > /dev/null 2>&1 &
shutdown -h now
```

User will download large files from AutoDL file manager later. Do NOT wait for user's large files downloads — shutdown immediately to save cost.

---

## Step 5: Submit submission.zip to Kaggle

**Kaggle submission names** must include date + key config so they're identifiable
Submit to **nvidia-nemotron-model-reasoning-challenge** competiton

**Before uploading**, run a CRC check on the local submission.zip:
```python
import zipfile
with zipfile.ZipFile(path) as zf:
    assert zf.testzip() is None, "CRC check failed"
```
If CRC fails, ask user to re-download from AutoDL.
Don't CRC check on the autodl instance, local check before uploading is enough.

Use the Kaggle MCP server (`claude mcp add --transport http --scope user kaggle https://www.kaggle.com/mcp --header "Authorization: Bearer $KAGGLE_API_TOKEN"`, one-time + restart Claude Code). Three calls:

1. `mcp__kaggle__start_competition_submission_upload` — returns `token` + `create_url`. **The `fileName` arg MUST be `"submission.zip"`** — this competition rejects any other registered blob name ("Submission files must be named submission.zip"). The local file can be named anything (e.g. `submission_last5-soup.zip`); only the `fileName` you register in this call must be `submission.zip`. Use a descriptive name in `submissionDescription` (step 3) to tell submissions apart.
2. Upload via kaggle SDK `api.upload_complete(FILE, create_url, quiet=False)` (**not** `requests.put` with a generator — GCS throttles chunked CN uploads). Run in background, poll every **2 min**.
3. The moment upload returns `COMPLETE`, call `mcp__kaggle__submit_to_competition` with the blob token and submission description (date + key config per rule 9). Don't wait — the blob token can expire.

**After uploading**, remove `_tmp_*.py` polling scripts from the local study folder.

**Soup runs:** submit `submission.zip` (final-step control) first, then each `submission_<recipe>.zip` separately — one submission per zip, each with its own descriptive name (`..._final{N}`, `..._last5-soup`, `..._wise-soup`). Mind Kaggle's **daily 5-submission cap** (resets at UTC 00:00 = Chicago 19:00); a full soup run is 1 final + 2 soups = 3 submissions, so it fits in one day. If the cap is hit mid-batch, the GCS upload still succeeds — the blob token stays valid ~24h, so just retry `submit_to_competition` after reset (no re-upload).

---

## Step 6: Write to Score Tracker

**Only trigger when user reports an actual Kaggle score.** Never pre-fill from training loss or pending submissions.

**Soup runs = ONE experiment entry.** A training run that produced several submissions (endpoint + soups) is a single experiment — record it as one `## Submission #N` entry with a results table (one row per zip: recipe / method / Kaggle ref / score / Δ vs endpoint), not N separate entries.

Append one entry at the **end** of `study/SCORE_TRACKER.md` . Use this exact format:

```markdown
## Submission #N — YYYY-MM-DD — Score X.XX

- **Study folder:** `{yymmddhhmm}_{task}_{notes}_{gpu}`
- **Data:** CSV path + row count + expanded count + key composition notes
- **Method:** key differences from prior runs (e.g. "same as #6 + Kaggle regex + ✓ stripped")
- **Why:** one paragraph explaining what changed vs prior runs and what the score implies. If score moved meaningfully (≥0.02), state the hypothesis confirmed/falsified. If the score dropped unexpectedly, investigate the reason deeply, don't find excuse or lie.
- **Detail:** link to `TRAINING_INFO.md` and/or `EVALUATION.md` in the submission folder.
```
Detailed analysis writes to EVALUATION.md in the submission folder.

`N` = next sequential number (read the file to find the last `Submission #N` and add 1).

---

## Rules

1. **NEVER kill a running training** to make "improvements" without explicit user approval — wait for it to finish or checkpoint
2. **NEVER delete checkpoints** without explicit user approval
3. **NEVER overwrite log files** — always append or use new filenames
4. **NEVER repeat the same training config**, check swanlab for previous train experiment
5. **NEVER fallback, patch to hide the bug** — observe the failure first, understand why, report to user
6. **Report problems to user** before attempting fixes — don't silently restart runs
7. **Check training every 5 minutes, check installation every 2 minutes** — when you have multiple instances running at the same time, offset the polling windows by 1min. See "n-Minute Check Mechanism" below.
8. **NEVER patch or modify base model files** (modeling_nemotron_h.py, config.json, etc.) without explicit user approval. Use proper dependencies (mamba-ssm, causal-conv1d) instead of pure-PyTorch workarounds. Patching model code causes subtle numerical differences that ruin training.
9. **NEVER touch output files while post-training steps are running** — after training finishes, the script saves adapter, creates submission.zip, and zips the checkpoint. These can take minutes for large files (3+ GB). Wait for the training process to fully exit (pgrep shows NO_TRAINING) before running ANY command that reads or writes output files. Patience costs nothing; corrupting a zip costs a full re-run.
10. **Cleanup only what you know** — for any "cleanup" step in this routine, only remove files you created or explicitly recognize. Other agents and the user may have files in the same folders; if you don't recognize it, leave it alone or ask.
11. **SFTP upload of file larger than 5mb is prohibitted.** You should always request the user to upload it. User usually upload files to /root/autodl-tmp/ , you should move it to where it should be.
12. when user say use the **same code**, you need to literaly read the code file instead of refering to your memory. **Your memory is always out of date.**

---

## n-Minute Check/Pulling Mechanism


Use the `Bash` tool with `run_in_background: true` to launch a single command that sleeps then runs the check script:

```
sleep <n * 60 seconds> && python <study_folder>/_tmp_check_<tra