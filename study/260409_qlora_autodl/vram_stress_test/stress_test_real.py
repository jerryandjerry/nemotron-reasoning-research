#!/usr/bin/env python3
"""
VRAM Stress Test — REAL training code with SFTTrainer + gradient checkpointing.
Uses only 2300-2500 token samples, no shuffle, adamw_torch.
Logs nvidia-smi at on_step_begin, on_step_end, and hooks into forward/backward.
"""

import os, sys, time, json, shutil, subprocess

os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import polars as pl
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_PATH = '/root/autodl-tmp/Nemotron-3-Nano-30B-A3B-bnb-4bit'
DATA_PATH = '/root/autodl-tmp/data/final_Nemotron_training_data.csv'
OUTPUT_DIR = '/root/autodl-tmp/stress_test_output'
LOG_FILE = '/root/autodl-tmp/stress_test_real_vram.csv'

os.makedirs(OUTPUT_DIR, exist_ok=True)

hf_cache_dir = os.path.expanduser('~/.cache/huggingface/modules/transformers_modules')
if os.path.exists(hf_cache_dir):
    shutil.rmtree(hf_cache_dir)

def smi():
    r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True)
    return int(r.stdout.strip())

def log_vram(phase, extra=''):
    s = smi()
    a = torch.cuda.memory_allocated() / 1024**3
    r = torch.cuda.memory_reserved() / 1024**3
    line = f'{phase},{s},{a:.3f},{r:.3f},{extra}'
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')
    print(f'  [{phase}] nvidia-smi={s}MiB alloc={a:.2f}G rsv={r:.2f}G {extra}', flush=True)
    return s

# CSV header
with open(LOG_FILE, 'w') as f:
    f.write('phase,nvidia_smi_mib,torch_alloc_gb,torch_reserved_gb,extra\n')

print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

# ── Dataset: only 2300-2500 token samples ───────────────────────────
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

def build_text(row):
    user_msg = row['prompt'] + '\nPut your final answer inside \\boxed{}.'
    assistant_msg = f"{row['generated_cot']}\n\n\\boxed{{{row['answer']}}}"
    try:
        messages = [{'role': 'user', 'content': user_msg}, {'role': 'assistant', 'content': assistant_msg}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except:
        return f'<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>'

# Build and filter to 2300-2500 tokens only
samples = []
for idx, row in enumerate(filtered_df.iter_rows(named=True)):
    text = build_text(row)
    ids = tokenizer(text, truncation=False, return_attention_mask=False)['input_ids']
    tlen = len(ids)
    if 2300 <= tlen <= 2500:
        samples.append((idx, tlen, text))

samples.sort(key=lambda x: x[1])
print(f'\nStress test samples (2300-2500 tokens): {len(samples)}')
for i, (idx, tlen, _) in enumerate(samples):
    print(f'  [{i}] orig_idx={idx}, tokens={tlen}')

# Need at least 4 samples for 1 full grad_accum step
if len(samples) < 4:
    # Duplicate samples to get at least 8
    while len(samples) < 8:
        samples = samples + samples
    samples = samples[:8]
    print(f'  Duplicated to {len(samples)} samples')

hf_dataset = Dataset.from_dict({'text': [s[2] for s in samples]})

# ── Load model ──────────────────────────────────────────────────────
log_vram('before_model_load')

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map={'': 0}, trust_remote_code=True, torch_dtype=torch.bfloat16,
)
log_vram('after_model_load')

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
log_vram('after_grad_checkpoint_enable')

lora_config = LoraConfig(
    r=32, lora_alpha=32, target_modules='all-linear',
    lora_dropout=0.05, bias='none', task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
log_vram('after_lora')

# ── Hook into model forward to log VRAM during forward pass ────────
original_forward = model.forward.__wrapped__ if hasattr(model.forward, '__wrapped__') else None

_micro_step = [0]  # mutable counter for hooks

def forward_hook(module, input, output):
    _micro_step[0] += 1
    log_vram(f'forward_hook_micro{_micro_step[0]}')

# Register hook on the base model's last layer
try:
    last_layer = model.base_model.model.backbone.layers[-1]
    last_layer.register_forward_hook(forward_hook)
    print('Forward hook registered on last layer')
except Exception as e:
    print(f'Could not register forward hook: {e}')

# ── Callback for step-level nvidia-smi ──────────────────────────────
class VRAMCallback(TrainerCallback):
    def __init__(self):
        self.step_count = 0

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()
        self.step_count += 1
        _micro_step[0] = 0  # reset micro step counter
        log_vram(f'step{self.step_count}_begin')

    def on_step_end(self, args, state, control, **kwargs):
        loss_val = ''
        if state.log_history and 'loss' in state.log_history[-1]:
            loss_val = f'loss={state.log_history[-1]["loss"]:.4f}'
        log_vram(f'step{self.step_count}_end', loss_val)
        torch.cuda.empty_cache()
        log_vram(f'step{self.step_count}_after_empty_cache', loss_val)

# ── Training ────────────────────────────────────────────────────────
print(f'\n=== Stress test: adamw_torch, {len(hf_dataset)} samples, no shuffle ===')

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=1e-4,
    logging_steps=1,
    bf16=True,
    bf16_full_eval=True,
    max_grad_norm=1.0,
    optim='adamw_torch',
    lr_scheduler_type='cosine',
    warmup_ratio=0.1,
    save_strategy='no',
    report_to='none',
    dataset_text_field='text',
    max_length=2500,
    packing=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
)

trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
    args=training_args,
    callbacks=[VRAMCallback()],
)

log_vram('before_training')

t0 = time.time()
try:
    trainer.train()
    elapsed = time.time() - t0
    print(f'\nStress test completed in {elapsed/60:.1f} min')
    log_vram('training_complete')
except torch.cuda.OutOfMemoryError as e:
    log_vram(f'OOM')
    print(f'\n*** OOM: {e}')
except Exception as e:
    log_vram(f'ERROR')
    print(f'\n*** Error: {e}')

# ── Print full log ──────────────────────────────────────────────────
print('\n=== Full VRAM Log ===')
with open(LOG_FILE) as f:
    print(f.read())

print('=== Done ===')
