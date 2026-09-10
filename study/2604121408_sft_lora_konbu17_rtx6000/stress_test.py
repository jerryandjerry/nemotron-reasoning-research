#!/usr/bin/env python3
"""
Stress test for konbu17 recipe on RTX PRO 6000 (97 GB).

Verifies VRAM fits with max_length=4096, bs=1, ga=8, LoRA on Mamba+MoE-FFN.
Filters dataset to top-length samples (>=3500 tokens), runs max_steps=4
(= 32 micro-batches of worst-case samples), uses the exact same trainer
class, LoraConfig, gradient checkpointing, optimizer, CoT format as
train_lora.py.
"""

import os, sys, time, gc, subprocess, types, re

os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Stub mamba3 modules (same as real run)
for _mod_name in [
    'mamba_ssm.modules.mamba3',
    'mamba_ssm.ops.cute',
    'mamba_ssm.ops.cute.mamba3',
    'mamba_ssm.ops.cute.mamba3.mamba3_step_fn',
]:
    _m = types.ModuleType(_mod_name)
    _m.__path__ = []
    _m.__package__ = _mod_name
    sys.modules[_mod_name] = _m
sys.modules['mamba_ssm.modules.mamba3'].Mamba3 = None

import mamba_ssm
print(f'mamba_ssm: {mamba_ssm.__version__}')

# ── Config (must match real training) ───────────────────────────────
SEED         = 123
LORA_RANK    = 32
LORA_ALPHA   = 32
MAX_SEQ_LEN  = 2000
BATCH_SIZE   = 4
GRAD_ACCUM   = 2
LR           = 1e-4
MAX_STEPS    = 4   # 4 optimizer steps × 2 grad_accum × 4 bs = 32 worst-case micro-batches

MODEL_PATH = '/root/autodl-tmp/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16'
DATA_PATH  = '/root/autodl-tmp/data/konbu17_verified_cot_6558rows.csv'
OUTPUT_DIR = '/root/autodl-tmp/stress_out'
os.makedirs(OUTPUT_DIR, exist_ok=True)

import torch
import polars as pl
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from tqdm.auto import tqdm

print(f'PyTorch : {torch.__version__}')
print(f'GPU     : {torch.cuda.get_device_name(0)}')
print(f'VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

# ── Dataset ─────────────────────────────────────────────────────────
print('\n=== Loading dataset ===')
df = pl.read_csv(DATA_PATH)
print(f'Full dataset: {len(df)} rows')

# For stress test: take all 6558 rows, drop empty CoT, then keep the LONGEST
# samples so every micro-batch hits worst-case VRAM
PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def build_training_text(example):
    prompt = str(example['prompt'])
    answer = str(example['answer'])
    cot = str(example['generated_cot'])
    cot_cleaned = re.sub(r'\\boxed\{[^}]*\}', '', cot).rstrip()
    user_msg = prompt + PROMPT_SUFFIX
    assistant_msg = cot_cleaned + f'\n</think>\n\\boxed{{{answer}}}'
    text = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': user_msg},
         {'role': 'assistant', 'content': assistant_msg}],
        tokenize=False, add_generation_prompt=False,
    )
    return {'text': text, 'sample_id': example['id']}

hf_dataset = Dataset.from_pandas(df.to_pandas())
hf_dataset = hf_dataset.filter(
    lambda x: x['generated_cot'] and str(x['generated_cot']) != 'nan' and len(str(x['generated_cot']).strip()) >= 5,
    desc='Dropping empty CoT',
)
hf_dataset = hf_dataset.map(
    build_training_text,
    remove_columns=[c for c in hf_dataset.column_names if c not in ['id']],
    desc='Applying chat template',
)

# Build sid lookup before dropping sample_id
token_to_sid = {}
for row in hf_dataset:
    ids = tokenizer(row['text'], truncation=False, return_attention_mask=False)['input_ids']
    token_to_sid[tuple(ids[:50])] = row['sample_id']
hf_dataset = hf_dataset.remove_columns(['sample_id'])

def get_token_length(example):
    ids = tokenizer(example['text'], truncation=False, return_attention_mask=False)['input_ids']
    return {'token_len': len(ids)}

hf_dataset = hf_dataset.map(get_token_length, desc='Counting tokens')
# Keep samples in [1500, 2000] — worst case within the real training budget
before = len(hf_dataset)
hf_dataset = hf_dataset.filter(lambda x: 1500 <= x['token_len'] <= MAX_SEQ_LEN, desc='Stress filter 1500-2000')
hf_dataset = hf_dataset.remove_columns(['token_len'])
print(f'Long samples kept (1500-2000): {len(hf_dataset)} / {before}')

# With bs=2 × ga=8 × 4 steps = 64 micro-batch draws we need the dataset
# to be at least ~16 rows so the dataloader keeps feeding batches.
# Duplicate the filtered rows as many times as needed.
need = BATCH_SIZE * GRAD_ACCUM * MAX_STEPS
if 0 < len(hf_dataset) < need:
    reps = (need // len(hf_dataset)) + 1
    print(f'Duplicating {len(hf_dataset)} rows {reps}x to get enough micro-batches')
    from datasets import concatenate_datasets
    hf_dataset = concatenate_datasets([hf_dataset] * reps)
    print(f'Stress dataset size: {len(hf_dataset)}')

# ── Load bf16 model ─────────────────────────────────────────────────
print('\n=== Loading bf16 model ===')
t_load = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={'': 0},
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
print(f'Model loaded in {time.time() - t_load:.1f}s')
print(f'VRAM after load: {torch.cuda.memory_allocated()/1024**3:.1f} GB')

model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=r'.*\.(in_proj|out_proj|up_proj|down_proj)$',
    lora_dropout=0.05,
    bias='none',
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print(f'VRAM after LoRA: {torch.cuda.memory_allocated()/1024**3:.1f} GB')

# ── Training (max_steps=4) ──────────────────────────────────────────
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    max_steps=MAX_STEPS,
    learning_rate=LR,
    logging_steps=1,
    bf16=True,
    max_grad_norm=1.0,
    optim='adamw_torch',
    lr_scheduler_type='cosine',
    warmup_ratio=0.05,
    save_strategy='no',
    report_to='none',
    dataset_text_field='text',
    max_length=MAX_SEQ_LEN,
    packing=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    dataloader_num_workers=0,
    seed=SEED,
)

class LoggingSFTTrainer(SFTTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        step = self.state.global_step + 1
        smi_val = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True).stdout.strip())
        if 'input_ids' in inputs:
            pad_id = tokenizer.pad_token_id
            for b in range(inputs['input_ids'].shape[0]):
                ids = inputs['input_ids'][b].tolist()
                ids_nopad = [t for t in ids if t != pad_id]
                content = tokenizer.decode(ids_nopad, skip_special_tokens=True)
                sid = token_to_sid.get(tuple(ids[:50]), '?')
                print(f'>> step={step} batch[{b}] sample_id={sid} padded_len={len(ids)} smi={smi_val}M content_tail: {content[-20:]}', flush=True)
        loss = super().training_step(model, inputs, num_items_in_batch)
        smi_after = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                       capture_output=True, text=True).stdout.strip())
        print(f'   step={step} smi_after={smi_after}M loss={loss.item():.4f}', flush=True)
        return loss

class StepGCCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        gc.collect(); torch.cuda.empty_cache()
    def on_step_end(self, args, state, control, **kwargs):
        gc.collect(); torch.cuda.empty_cache()

trainer = LoggingSFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
    args=training_args,
    callbacks=[StepGCCallback()],
)

print('\n=== Running stress test ===')
t0 = time.time()
trainer.train()
elapsed = time.time() - t0
print(f'\nStress test done in {elapsed:.1f}s ({elapsed/MAX_STEPS:.1f}s/step)')

# Final VRAM peak
smi_final = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True).stdout.strip())
print(f'Final nvidia-smi used: {smi_final} MiB / 97887 MiB')
print(f'torch allocated:       {torch.cuda.memory_allocated()/1024**2:.0f} MiB')
print(f'torch max reserved:    {torch.cuda.max_memory_reserved()/1024**2:.0f} MiB')
print(f'torch max allocated:   {torch.cuda.max_memory_allocated()/1024**2:.0f} MiB')

# Clean up stress output dir
import shutil
try: shutil.rmtree(OUTPUT_DIR)
except: pass

print('\nOK')
