#!/usr/bin/env python3
"""
VRAM Stress Test v3: Fine-grained nvidia-smi timing.
Hooks into forward, backward, optimizer step, and between micro-batches.
"""

import os, sys, time, json, shutil, subprocess

os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import polars as pl
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_PATH = '/root/autodl-tmp/Nemotron-3-Nano-30B-A3B-bnb-4bit'
DATA_PATH = '/root/autodl-tmp/data/final_Nemotron_training_data.csv'
OUTPUT_DIR = '/root/autodl-tmp/stress_test_output'
LOG_FILE = '/root/autodl-tmp/stress_test_v3_log.csv'

os.makedirs(OUTPUT_DIR, exist_ok=True)

hf_cache_dir = os.path.expanduser('~/.cache/huggingface/modules/transformers_modules')
if os.path.exists(hf_cache_dir):
    shutil.rmtree(hf_cache_dir)

def nvidia_smi_mib():
    r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True)
    return int(r.stdout.strip())

def log(msg):
    smi = nvidia_smi_mib()
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    line = f'[{smi}MiB | alloc={alloc:.2f}G | rsv={reserved:.2f}G] {msg}'
    print(line, flush=True)
    return smi

print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

# ── Prepare dataset ─────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_df = pl.read_csv(DATA_PATH)
RELEVANT_KEYWORDS = [
    'cipher', 'string', 'transformation', 'bitwise', 'binary', 'unit',
    'measurement', 'encoding', 'decoding', 'pattern', 'sequence',
    'conversion', 'arithmetic', 'logical', 'symbolic',
]
label_col = train_df['label'].str.to_lowercase()
mask = pl.lit(False)
for kw in RELEVANT_KEYWORDS:
    mask = mask | label_col.str.contains(kw)
filtered_df = train_df.filter(mask).sample(n=3000, seed=99)

samples = []
for idx, row in enumerate(filtered_df.iter_rows(named=True)):
    user_msg = row['prompt'] + '\nPut your final answer inside \\boxed{}.'
    assistant_msg = f"{row['generated_cot']}\n\n\\boxed{{{row['answer']}}}"
    try:
        messages = [{'role': 'user', 'content': user_msg}, {'role': 'assistant', 'content': assistant_msg}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except:
        text = f'<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>'
    ids = tokenizer(text, truncation=False, return_attention_mask=False)['input_ids']
    tlen = len(ids)
    if 2000 <= tlen <= 2500:
        samples.append((idx, tlen, text))

samples.sort(key=lambda x: x[1])
print(f'Stress samples: {len(samples)}')
for i, (idx, tlen, _) in enumerate(samples):
    print(f'  [{i}] orig_idx={idx}, tokens={tlen}')

# ── Load model ──────────────────────────────────────────────────────
log('Before model load')
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map={'': 0}, trust_remote_code=True, torch_dtype=torch.bfloat16,
)
log('After model load')

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
log('After prepare_model_for_kbit_training')

lora_config = LoraConfig(
    r=32, lora_alpha=32, target_modules='all-linear',
    lora_dropout=0.05, bias='none', task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
log('After LoRA')

# ── Manual training loop for fine-grained VRAM monitoring ───────────
print('\n=== Manual training loop (adamw_torch) ===')

# Tokenize all samples
tokenized = []
for i, (orig_idx, tlen, text) in enumerate(samples):
    enc = tokenizer(text, truncation=True, max_length=2500, padding=False, return_tensors='pt')
    input_ids = enc['input_ids'].to('cuda')
    attention_mask = enc['attention_mask'].to('cuda')
    labels = input_ids.clone()
    tokenized.append((i, orig_idx, tlen, input_ids, attention_mask, labels))

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4,
)

log('After optimizer creation')

# Write CSV header
with open(LOG_FILE, 'w') as f:
    f.write('sample_idx,orig_idx,token_len,phase,nvidia_smi_mib,torch_alloc_gb,torch_reserved_gb\n')

def log_csv(sample_idx, orig_idx, token_len, phase):
    smi = nvidia_smi_mib()
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    with open(LOG_FILE, 'a') as f:
        f.write(f'{sample_idx},{orig_idx},{token_len},{phase},{smi},{alloc:.3f},{reserved:.3f}\n')
    return smi

model.train()
grad_accum = 4
micro_step = 0

for i, (sample_idx, orig_idx, tlen, input_ids, attention_mask, labels) in enumerate(tokenized):
    micro = micro_step % grad_accum
    print(f'\n--- Sample {i}/{len(tokenized)}: orig_idx={orig_idx}, tokens={tlen}, micro_batch={micro}/{grad_accum} ---')

    torch.cuda.empty_cache()
    smi = log_csv(sample_idx, orig_idx, tlen, 'before_empty_cache')
    print(f'  after empty_cache: {smi} MiB')

    try:
        # Forward
        smi = log_csv(sample_idx, orig_idx, tlen, 'before_forward')
        print(f'  before forward: {smi} MiB')

        outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
        loss = outputs.loss / grad_accum

        smi = log_csv(sample_idx, orig_idx, tlen, 'after_forward')
        print(f'  after forward: {smi} MiB')

        # Backward
        loss.backward()

        smi = log_csv(sample_idx, orig_idx, tlen, 'after_backward')
        print(f'  after backward: {smi} MiB')

        loss_val = loss.item() * grad_accum

        # Delete loss and outputs to free logits
        del outputs, loss

        smi = log_csv(sample_idx, orig_idx, tlen, 'after_del_outputs')
        print(f'  after del outputs: {smi} MiB')
        print(f'  loss: {loss_val:.4f}')

        micro_step += 1

        # Optimizer step every grad_accum micro-steps
        if micro_step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            smi = log_csv(sample_idx, orig_idx, tlen, 'before_optim_step')
            print(f'  before optimizer step: {smi} MiB')

            optimizer.step()

            smi = log_csv(sample_idx, orig_idx, tlen, 'after_optim_step')
            print(f'  after optimizer step: {smi} MiB')

            optimizer.zero_grad()

            smi = log_csv(sample_idx, orig_idx, tlen, 'after_zero_grad')
            print(f'  after zero_grad: {smi} MiB')

    except torch.cuda.OutOfMemoryError as e:
        smi = nvidia_smi_mib()
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f'  *** OOM! nvidia-smi={smi}MiB, alloc={alloc:.2f}G, reserved={reserved:.2f}G')
        print(f'  Error: {e}')
        log_csv(sample_idx, orig_idx, tlen, f'OOM_{smi}MiB')
        # Try to recover
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        continue

print('\n=== Summary ===')
import csv
with open(LOG_FILE) as f:
    rows = list(csv.DictReader(f))
if rows:
    max_row = max(rows, key=lambda r: int(r['nvidia_smi_mib']))
    print(f'Peak: {max_row["nvidia_smi_mib"]} MiB at sample {max_row["sample_idx"]} (tokens={max_row["token_len"]}, phase={max_row["phase"]})')
    print(f'\nFull log:')
    for r in rows:
        print(f'  sample={r["sample_idx"]}, tokens={r["token_len"]}, phase={r["phase"]}, smi={r["nvidia_smi_mib"]}MiB, alloc={r["torch_alloc_gb"]}G')

print('\n=== Done ===')
