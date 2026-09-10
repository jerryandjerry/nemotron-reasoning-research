#!/usr/bin/env python3
"""Stress test for v3 CoT training — uses the real training pipeline.

Sorts dataset by token length descending, takes top-K longest samples,
runs 3 steps with no saving to measure peak VRAM.
"""

import os, sys, time, json, gc, subprocess, types, re

os.environ['TRANSFORMERS_NO_TF']     = '1'
os.environ['TRANSFORMERS_NO_FLAX']   = '1'
os.environ['CUDA_VISIBLE_DEVICES']   = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HF_HUB_OFFLINE']         = '1'
os.environ['TRANSFORMERS_OFFLINE']   = '1'
os.environ['WANDB_DISABLED']         = '1'

import unsloth
from unsloth import FastLanguageModel

for _mod_name in [
    'mamba_ssm.modules.mamba3',
    'mamba_ssm.ops.cute',
    'mamba_ssm.ops.cute.mamba3',
    'mamba_ssm.ops.cute.mamba3.mamba3_step_fn',
]:
    _m = types.ModuleType(_mod_name); _m.__path__ = []; _m.__package__ = _mod_name
    sys.modules[_mod_name] = _m
sys.modules['mamba_ssm.modules.mamba3'].Mamba3 = None

import torch
import polars as pl
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# ── Config ──────────────────────────────────────────────────────────
SEED         = 123
LORA_RANK    = 32
LORA_ALPHA   = 16
MAX_SEQ_LEN  = 3500
BATCH_SIZE   = int(sys.argv[1]) if len(sys.argv) > 1 else 8
MAX_STEPS    = 3

PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

DATA_FILES = [
    '/root/autodl-tmp/data/v3cot/train_cot_bit_manipulation.csv',
    '/root/autodl-tmp/data/v3cot/train_cot_equation_cryptarithm.csv',
    '/root/autodl-tmp/data/v3cot/train_cot_equation_numeric.csv',
]

MODEL_PATH = '/root/autodl-tmp/Nemotron-3-Nano-30B-A3B'

print(f'Stress test: bs={BATCH_SIZE}, max_seq={MAX_SEQ_LEN}, steps={MAX_STEPS}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

# ── Load model ──────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH, max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=True, load_in_8bit=False, full_finetuning=False,
    trust_remote_code=True, unsloth_force_compile=True,
    attn_implementation='eager', dtype=torch.bfloat16,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = FastLanguageModel.get_peft_model(
    model, r=LORA_RANK,
    target_modules=['in_proj', 'out_proj', 'up_proj', 'down_proj'],
    lora_alpha=LORA_ALPHA, lora_dropout=0.05,
    bias='none', use_gradient_checkpointing='unsloth',
    random_state=3407, use_rslora=False, loftq_config=None,
)

# ── Load data, sort by length descending ────────────────────────────
dfs = [pl.read_csv(f) for f in DATA_FILES]
df = pl.concat(dfs)
print(f'Total samples: {len(df)}')

hf_dataset = Dataset.from_pandas(df.to_pandas())

def has_valid_cot(ex):
    cot = str(ex['solver_cot']) if ex['solver_cot'] is not None else ''
    return cot and cot != 'nan' and len(cot.strip()) >= 5

hf_dataset = hf_dataset.filter(has_valid_cot, desc='Dropping empty CoT')

def build_training_text(example):
    prompt = str(example['prompt'])
    answer = str(example['answer'])
    cot = str(example['solver_cot'])
    cot_cleaned = re.sub(r'\\boxed\{[^}]*\}', '', cot).rstrip()
    user_msg = prompt + PROMPT_SUFFIX
    assistant_msg = cot_cleaned + f'\n</think>\n\\boxed{{{answer}}}'
    try:
        text = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': user_msg},
             {'role': 'assistant', 'content': assistant_msg}],
            tokenize=False, add_generation_prompt=False,
        )
    except Exception:
        text = (f'<|im_start|>user\n{user_msg}<|im_end|>\n'
                f'<|im_start|>assistant\n{assistant_msg}<|im_end|>')
    return {'text': text, 'sample_id': example['id']}

hf_dataset = hf_dataset.map(build_training_text,
    remove_columns=[c for c in hf_dataset.column_names if c != 'id'],
    desc='Applying chat template')

# Token lengths
def get_token_length(example):
    ids = tokenizer(example['text'], truncation=False, return_attention_mask=False)['input_ids']
    return {'token_len': len(ids)}

hf_dataset = hf_dataset.map(get_token_length, desc='Counting tokens')

# Print distribution
lens = sorted(hf_dataset['token_len'], reverse=True)
print(f'\nToken length distribution:')
print(f'  max={lens[0]}, p99={lens[int(len(lens)*0.01)]}, p95={lens[int(len(lens)*0.05)]}, p90={lens[int(len(lens)*0.1)]}, median={lens[len(lens)//2]}, min={lens[-1]}')
print(f'  > {MAX_SEQ_LEN}: {sum(1 for l in lens if l > MAX_SEQ_LEN)} samples would be dropped')

# Filter and sort by length descending (worst case)
hf_dataset = hf_dataset.filter(lambda x: x['token_len'] <= MAX_SEQ_LEN, desc='Filter')
hf_dataset = hf_dataset.sort('token_len', reverse=True)

# Take top-K for worst-case stress
K = BATCH_SIZE * MAX_STEPS * 2
hf_dataset_stress = hf_dataset.select(range(min(K, len(hf_dataset))))
hf_dataset_stress = hf_dataset_stress.remove_columns(['sample_id', 'token_len'])
print(f'\nStress test: {len(hf_dataset_stress)} longest samples, bs={BATCH_SIZE}')
print(f'Longest sample: {lens[0]} tokens')

# ── Run stress test ─────────────────────────────────────────────────
training_args = SFTConfig(
    output_dir='/root/autodl-tmp/stress_test_output',
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1,
    max_steps=MAX_STEPS,
    learning_rate=1e-4,
    bf16=True, max_grad_norm=1.0,
    optim='adamw_torch',
    save_strategy='no',
    report_to='none',
    dataset_text_field='text',
    packing=False, padding_free=False,
    seed=SEED,
)

trainer = SFTTrainer(
    model=model, train_dataset=hf_dataset_stress,
    processing_class=tokenizer, args=training_args,
)

gc.collect(); torch.cuda.empty_cache()
print(f'\nVRAM before training: {torch.cuda.memory_allocated()/1024**3:.1f} GB')

for step in range(MAX_STEPS):
    gc.collect(); torch.cuda.empty_cache()
    smi = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout.strip()
    print(f'Step {step+1}/{MAX_STEPS} starting, smi={smi}M')

t0 = time.time()
trainer.train()
elapsed = time.time() - t0

smi_final = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True).stdout.strip()
print(f'\nStress test done in {elapsed:.1f}s')
print(f'Final smi={smi_final}M')
print(f'Peak VRAM (torch): {torch.cuda.max_memory_allocated()/1024**3:.1f} GB')
print(f'Peak VRAM (smi): check nvidia-smi above')
print(f'\nResult: bs={BATCH_SIZE} with seq={MAX_SEQ_LEN} — {"OK" if True else "OOM"}')
