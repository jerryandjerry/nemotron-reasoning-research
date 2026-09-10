#!/usr/bin/env python3
"""
VRAM stress test: find peak VRAM during training with adamw_torch.
Tests with shortest, median, and longest samples to understand
how sequence length affects peak VRAM.
"""

import os, sys, time, json, shutil

os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import polars as pl
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Clear HF cache
hf_cache_dir = os.path.expanduser('~/.cache/huggingface/modules/transformers_modules')
if os.path.exists(hf_cache_dir):
    shutil.rmtree(hf_cache_dir)

MODEL_PATH = '/root/autodl-tmp/Nemotron-3-Nano-30B-A3B-bnb-4bit'
DATA_PATH = '/root/autodl-tmp/data/final_Nemotron_training_data.csv'

print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

# ── Load tokenizer ──────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Prepare test samples at different lengths ───────────────────────
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
        messages = [
            {'role': 'user', 'content': user_msg},
            {'role': 'assistant', 'content': assistant_msg},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except:
        return f'<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>'

# Compute all token lengths and find target samples
print('\nComputing token lengths...')
samples = []
for row in filtered_df.iter_rows(named=True):
    text = build_text(row)
    ids = tokenizer(text, truncation=False, return_attention_mask=False)['input_ids']
    tlen = len(ids)
    if tlen <= 2500:
        samples.append((tlen, text))

samples.sort(key=lambda x: x[0])
print(f'Total samples <= 2500: {len(samples)}')
print(f'Length range: {samples[0][0]} - {samples[-1][0]}')

# Pick test cases
test_cases = [
    ('shortest', samples[0]),
    ('median', samples[len(samples)//2]),
    ('p90', samples[int(len(samples)*0.9)]),
    ('p99', samples[int(len(samples)*0.99)]),
    ('longest', samples[-1]),
]

print('\nTest cases:')
for name, (tlen, _) in test_cases:
    print(f'  {name}: {tlen} tokens')

# ── Load model ──────────────────────────────────────────────────────
print('\n=== Loading model ===')
torch.cuda.reset_peak_memory_stats()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={'': 0},
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

vram_model = torch.cuda.memory_allocated() / 1024**3
peak_model = torch.cuda.max_memory_allocated() / 1024**3
print(f'After model load: allocated={vram_model:.2f}G, peak={peak_model:.2f}G')

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=32, lora_alpha=32,
    target_modules='all-linear',
    lora_dropout=0.05, bias='none',
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

vram_lora = torch.cuda.memory_allocated() / 1024**3
peak_lora = torch.cuda.max_memory_allocated() / 1024**3
print(f'After LoRA: allocated={vram_lora:.2f}G, peak={peak_lora:.2f}G')

# ── Test each sample length ─────────────────────────────────────────
for optim_name in ['adamw_torch', 'adamw_8bit']:
    print(f'\n{"="*60}')
    print(f'OPTIMIZER: {optim_name}')
    print(f'{"="*60}')

    for case_name, (tlen, text) in test_cases:
        # Create single-sample dataset
        ds = Dataset.from_dict({'text': [text] * 4})  # 4 copies for 1 step with grad_accum=4

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated() / 1024**3

        try:
            training_args = SFTConfig(
                output_dir='/root/autodl-tmp/vram_test_output',
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                num_train_epochs=1,
                learning_rate=1e-4,
                bf16=True,
                max_grad_norm=1.0,
                optim=optim_name,
                save_strategy='no',
                report_to='none',
                dataset_text_field='text',
                max_length=2500,
                packing=False,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={'use_reentrant': False},
                logging_steps=1,
            )

            trainer = SFTTrainer(
                model=model,
                train_dataset=ds,
                processing_class=tokenizer,
                args=training_args,
            )

            trainer.train()

            peak = torch.cuda.max_memory_allocated() / 1024**3
            after = torch.cuda.memory_allocated() / 1024**3
            torch.cuda.empty_cache()
            after_cache = torch.cuda.memory_allocated() / 1024**3

            print(f'  {case_name} ({tlen} tokens): before={before:.2f}G, peak={peak:.2f}G, after={after:.2f}G, after_cache={after_cache:.2f}G, headroom={31.4-peak:.2f}G')

        except torch.cuda.OutOfMemoryError as e:
            peak = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.empty_cache()
            print(f'  {case_name} ({tlen} tokens): OOM! peak={peak:.2f}G, needed={31.4-peak:.2f}G more')

        except Exception as e:
            print(f'  {case_name} ({tlen} tokens): ERROR: {e}')

        # Clean up trainer
        del trainer
        torch.cuda.empty_cache()

print('\n=== Done ===')
