#!/usr/bin/env python3
"""
VRAM Stress Test v2: Train on longest samples (2000-2500 tokens) in fixed order.
Uses nvidia-smi for real GPU memory measurement.
No shuffle — so we know exactly which sample index causes OOM.
Tests adamw_torch to find the real peak VRAM.
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
LOG_FILE = '/root/autodl-tmp/stress_test_vram_log.csv'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clear HF cache
hf_cache_dir = os.path.expanduser('~/.cache/huggingface/modules/transformers_modules')
if os.path.exists(hf_cache_dir):
    shutil.rmtree(hf_cache_dir)

def get_nvidia_smi_mib():
    """Get actual GPU memory usage from nvidia-smi in MiB."""
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True
    )
    return int(result.stdout.strip())

print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

# ── Prepare stress test dataset: only 2000-2500 token samples ───────
print('\n=== Preparing stress test dataset ===')
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

# Build texts and compute token lengths
samples = []
for idx, row in enumerate(filtered_df.iter_rows(named=True)):
    user_msg = row['prompt'] + '\nPut your final answer inside \\boxed{}.'
    assistant_msg = f"{row['generated_cot']}\n\n\\boxed{{{row['answer']}}}"
    try:
        messages = [
            {'role': 'user', 'content': user_msg},
            {'role': 'assistant', 'content': assistant_msg},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except:
        text = f'<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>'

    ids = tokenizer(text, truncation=False, return_attention_mask=False)['input_ids']
    tlen = len(ids)
    if 2000 <= tlen <= 2500:
        samples.append((idx, tlen, text))

# Sort by token length (ascending) — fixed order, no shuffle
samples.sort(key=lambda x: x[1])

print(f'Samples with 2000-2500 tokens: {len(samples)}')
if samples:
    print(f'Length range: {samples[0][1]} - {samples[-1][1]}')
    for i, (idx, tlen, _) in enumerate(samples):
        print(f'  [{i}] orig_idx={idx}, tokens={tlen}')

# Create HF dataset in fixed order (sorted by length, ascending)
hf_dataset = Dataset.from_dict({
    'text': [s[2] for s in samples],
    'orig_idx': [s[0] for s in samples],
    'token_len': [s[1] for s in samples],
})
print(f'\nStress test dataset: {len(hf_dataset)} samples, sorted by length')

# ── Load model ──────────────────────────────────────────────────────
print('\n=== Loading model ===')
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={'': 0},
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

print(f'Model loaded. nvidia-smi: {get_nvidia_smi_mib()} MiB')

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
print(f'After LoRA. nvidia-smi: {get_nvidia_smi_mib()} MiB')

# ── VRAM logging callback ──────────────────────────────────────────
class VRAMLogCallback(TrainerCallback):
    def __init__(self, log_file, dataset_info):
        self.log_file = log_file
        self.dataset_info = dataset_info  # list of (orig_idx, token_len)
        self.step_in_dataset = 0
        # Write CSV header
        with open(log_file, 'w') as f:
            f.write('step,sample_idx_in_stress,orig_idx,token_len,nvidia_smi_mib,torch_allocated_gb,torch_reserved_gb,torch_peak_gb\n')

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        nvidia_mib = get_nvidia_smi_mib()
        torch_alloc = torch.cuda.memory_allocated() / 1024**3
        torch_reserved = torch.cuda.memory_reserved() / 1024**3
        torch_peak = torch.cuda.max_memory_allocated() / 1024**3

        # Figure out which sample we're on
        # With batch_size=1, grad_accum=4, each "step" processes 4 samples
        # The samples in this step are indices: (step-1)*4 to (step-1)*4+3
        start_idx = (step - 1) * 4
        end_idx = min(start_idx + 4, len(self.dataset_info))

        for i in range(start_idx, end_idx):
            if i < len(self.dataset_info):
                orig_idx, tlen = self.dataset_info[i]
            else:
                orig_idx, tlen = -1, -1

        # Log the step info (use the last sample in the batch as representative)
        if start_idx < len(self.dataset_info):
            # Find the longest sample in this batch (likely the one causing peak)
            batch_samples = self.dataset_info[start_idx:end_idx]
            max_sample = max(batch_samples, key=lambda x: x[1])
            orig_idx, tlen = max_sample
        else:
            orig_idx, tlen = -1, -1

        with open(self.log_file, 'a') as f:
            f.write(f'{step},{start_idx},{orig_idx},{tlen},{nvidia_mib},{torch_alloc:.3f},{torch_reserved:.3f},{torch_peak:.3f}\n')

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        print(f'  Step {step}: nvidia-smi={nvidia_mib}MiB, alloc={torch_alloc:.2f}G, reserved={torch_reserved:.2f}G, peak={torch_peak:.2f}G, max_token_len={tlen}')
        sys.stdout.flush()

# ── Training (stress test) ──────────────────────────────────────────
print('\n=== Starting stress test training (adamw_torch) ===')

dataset_info = [(s[0], s[1]) for s in samples]  # (orig_idx, token_len)

# Remove token_len and orig_idx columns before passing to trainer
hf_dataset_clean = hf_dataset.remove_columns(['orig_idx', 'token_len'])

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
    train_dataset=hf_dataset_clean,
    processing_class=tokenizer,
    args=training_args,
    callbacks=[VRAMLogCallback(LOG_FILE, dataset_info)],
)

t0 = time.time()
try:
    trainer.train()
    print(f'\nStress test completed in {(time.time()-t0)/60:.1f} min')
except torch.cuda.OutOfMemoryError as e:
    elapsed = time.time() - t0
    print(f'\n*** OOM at step {trainer.state.global_step} after {elapsed/60:.1f} min ***')
    print(f'nvidia-smi: {get_nvidia_smi_mib()} MiB')
    print(f'Error: {e}')
except Exception as e:
    print(f'\nError: {e}')

# ── Print results ───────────────────────────────────────────────────
print('\n=== VRAM Log Summary ===')
try:
    import csv
    with open(LOG_FILE) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if rows:
        max_row = max(rows, key=lambda r: int(r['nvidia_smi_mib']))
        print(f'Peak nvidia-smi: {max_row["nvidia_smi_mib"]} MiB at step {max_row["step"]} (token_len={max_row["token_len"]})')
        print(f'Total steps logged: {len(rows)}')
        print(f'\nAll steps:')
        for r in rows:
            print(f'  step={r["step"]}, nvidia_smi={r["nvidia_smi_mib"]}MiB, token_len={r["token_len"]}, peak={r["torch_peak_gb"]}G')
except Exception as e:
    print(f'Could not read log: {e}')

print('\n=== Done ===')
