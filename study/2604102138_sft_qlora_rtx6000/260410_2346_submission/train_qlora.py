#!/usr/bin/env python3
"""
QLoRA 4-bit Training on AutoDL (RTX 5090, 32GB VRAM)
Same recipe as 0.72 LoRA but with pre-quantized BNB NF4 model.
Fast iteration: test data quality hypotheses, then final bf16 run.
"""

import os, sys, time, json, zipfile, stat, shutil, gc, subprocess

os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['SWANLAB_API_KEY'] = dict(l.strip().split('=', 1) for l in open('kaggle.env', encoding='utf-8-sig') if '=' in l and not l.startswith('#'))['SWANLAB_API_KEY']
os.environ['SWANLAB_PROJECT'] = '260410_Nemotron'

# ── Config ──────────────────────────────────────────────────────────
SAMPLE_SIZE  = 3000
LORA_RANK    = 32
MAX_SEQ_LEN  = 2500
NUM_EPOCHS   = 1
BATCH_SIZE   = 8
GRAD_ACCUM   = 2
LR           = 2e-4
# Generate run name with CST timestamp
from datetime import datetime, timezone, timedelta
_cst = timezone(timedelta(hours=-5))  # Chicago CDT
_ts = datetime.now(_cst).strftime('%y%m%d_%H%M')
RUN_NAME     = f'{_ts}_sft_qlora_3000samp_r32_lr2e4_seq2500_adamw_bs8ga2_rtx6000'

MODEL_PATH   = '/root/autodl-tmp/Nemotron-3-Nano-30B-A3B-bnb-4bit'
DATA_PATH    = '/root/autodl-tmp/data/final_Nemotron_training_data.csv'
OUTPUT_DIR   = '/root/autodl-tmp/adapter_output'
ZIP_PATH     = '/root/autodl-tmp/submission.zip'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clear HF cache to ensure fresh model code is loaded
hf_cache_dir = os.path.expanduser('~/.cache/huggingface/modules/transformers_modules')
if os.path.exists(hf_cache_dir):
    shutil.rmtree(hf_cache_dir)
    print(f'Cleared HF cache: {hf_cache_dir}')

# ── Imports ─────────────────────────────────────────────────────────
import torch
import torch.nn.functional as F
import polars as pl
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from tqdm.auto import tqdm

print(f'PyTorch : {torch.__version__}')
print(f'GPU     : {torch.cuda.get_device_name(0)}')
print(f'VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'CUDA cap: {torch.cuda.get_device_capability()}')

import bitsandbytes as bnb
print(f'BNB     : {bnb.__version__}')

# ── Dataset: same filtering as 0.72 recipe ──────────────────────────
print('\n=== Loading dataset ===')
train_df = pl.read_csv(DATA_PATH)
print(f'Total samples: {len(train_df)}')

RELEVANT_KEYWORDS = [
    'cipher', 'string', 'transformation', 'bitwise', 'binary', 'unit',
    'measurement', 'encoding', 'decoding', 'pattern', 'sequence',
    'conversion', 'arithmetic', 'logical', 'symbolic',
]

label_col = train_df['label'].str.to_lowercase()
mask = pl.lit(False)
for kw in RELEVANT_KEYWORDS:
    mask = mask | label_col.str.contains(kw)

filtered_df = train_df.filter(mask)
print(f'After keyword filter: {len(filtered_df)}')
print(filtered_df['label'].value_counts().sort('count', descending=True))

if len(filtered_df) < SAMPLE_SIZE:
    sampled_df = filtered_df
else:
    sampled_df = filtered_df.sample(n=SAMPLE_SIZE, seed=99)

hf_dataset = Dataset.from_pandas(sampled_df.to_pandas())
print(f'Sampled: {len(hf_dataset)}')

# ── Tokenizer & prompt formatting ───────────────────────────────────
print('\n=== Loading tokenizer ===')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def build_training_text(example):
    user_msg = example['prompt'] + '\nPut your final answer inside \\boxed{}.'
    assistant_msg = f"{example['generated_cot']}\n\n\\boxed{{{example['answer']}}}"
    try:
        messages = [
            {'role': 'user', 'content': user_msg},
            {'role': 'assistant', 'content': assistant_msg},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        text = (
            f'<|im_start|>user\n{user_msg}<|im_end|>\n'
            f'<|im_start|>assistant\n{assistant_msg}<|im_end|>'
        )
    return {'text': text, 'sample_id': example['id']}

hf_dataset = hf_dataset.map(
    build_training_text,
    remove_columns=[c for c in hf_dataset.column_names if c not in ['id']],
    desc='Applying chat template',
)

# Build lookup: first 50 tokens -> sample_id
token_to_sid = {}
for row in hf_dataset:
    ids = tokenizer(row['text'], truncation=False, return_attention_mask=False)['input_ids']
    token_to_sid[tuple(ids[:50])] = row['sample_id']

# Remove sample_id column (trainer doesn't need it)
hf_dataset = hf_dataset.remove_columns(['sample_id'])

# Drop samples > MAX_SEQ_LEN tokens (no truncated CoT)
print(f'Filtering samples > {MAX_SEQ_LEN} tokens...')
before = len(hf_dataset)

def get_token_length(example):
    ids = tokenizer(example['text'], truncation=False,
                    return_attention_mask=False)['input_ids']
    return {'token_len': len(ids)}

hf_dataset = hf_dataset.map(get_token_length, desc='Counting tokens')
hf_dataset = hf_dataset.filter(lambda x: x['token_len'] <= MAX_SEQ_LEN, desc='Dropping oversized')
hf_dataset = hf_dataset.remove_columns(['token_len'])
print(f'Kept {len(hf_dataset)} / {before} ({before - len(hf_dataset)} dropped)')

steps_estimate = len(hf_dataset) // (BATCH_SIZE * GRAD_ACCUM) * NUM_EPOCHS
print(f'Estimated training steps: {steps_estimate}')

# ── Load pre-quantized model ────────────────────────────────────────
print('\n=== Loading pre-quantized NF4 model ===')
t_load = time.time()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map={'': 0},
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

print(f'Model loaded in {time.time() - t_load:.1f}s')
print(f'VRAM after load: {torch.cuda.memory_allocated()/1024**3:.1f} GB')

# Prepare for QLoRA training
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

# Apply LoRA
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=32,
    target_modules='all-linear',
    lora_dropout=0.05,
    bias='none',
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print(f'VRAM after LoRA: {torch.cuda.memory_allocated()/1024**3:.1f} GB')

# ── Progress callback ───────────────────────────────────────────────
class LiveProgressCallback(TrainerCallback):
    def __init__(self):
        self.pbar = None
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.pbar = tqdm(total=state.max_steps, desc='Training', unit='step',
                         dynamic_ncols=True, file=sys.stdout)
        self.start_time = time.time()

    def on_step_begin(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()

    def on_step_end(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()

        if self.pbar is None:
            return
        elapsed = time.time() - self.start_time
        step = state.global_step
        eta = (elapsed / step) * (state.max_steps - step) if step > 0 else 0
        loss_str = (
            f"loss={state.log_history[-1]['loss']:.4f}"
            if state.log_history and 'loss' in state.log_history[-1]
            else 'loss=...'
        )
        vram = torch.cuda.memory_allocated() / 1024**3
        try:
            smi = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True).stdout.strip())
            smi_str = f'smi={smi}M'
        except:
            smi_str = ''
        self.pbar.set_postfix_str(f'{loss_str}  vram={vram:.1f}G  {smi_str}  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m')
        self.pbar.update(1)
        sys.stdout.flush()

    def on_save(self, args, state, control, **kwargs):
        # Rename checkpoint folder to include key metrics
        if state.log_history:
            last = {}
            for entry in reversed(state.log_history):
                if 'loss' in entry and 'loss' not in last:
                    last['loss'] = entry['loss']
                    last['lr'] = entry.get('learning_rate', 0)
                    last['epoch'] = entry.get('epoch', 0)
                    last['acc'] = entry.get('mean_token_accuracy', 0)
                    break
            if 'loss' in last:
                step = state.global_step
                old_dir = os.path.join(args.output_dir, f'checkpoint-{step}')
                parts = [f'checkpoint-{step}',
                         f'loss{last["loss"]:.4f}',
                         f'lr{last["lr"]:.2e}',
                         f'ep{last["epoch"]:.2f}']
                if last['acc'] > 0:
                    parts.append(f'acc{last["acc"]:.4f}')
                new_dir = os.path.join(args.output_dir, '_'.join(parts))
                if os.path.exists(old_dir) and not os.path.exists(new_dir):
                    os.rename(old_dir, new_dir)
                    print(f'\nCheckpoint saved: {os.path.basename(new_dir)}')

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.close()

# ── Training ────────────────────────────────────────────────────────
print('\n=== Starting training ===')
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    logging_steps=10,
    bf16=True,
    bf16_full_eval=True,
    max_grad_norm=1.0,
    optim='adamw_torch',
    lr_scheduler_type='cosine',
    warmup_ratio=0.1,
    save_strategy='steps',
    save_steps=100,
    save_total_limit=6,
    report_to='swanlab',
    run_name=RUN_NAME,
    dataset_text_field='text',
    max_length=MAX_SEQ_LEN,
    packing=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
)

# ── Subclass SFTTrainer to log sample info at each training step ────
class LoggingSFTTrainer(SFTTrainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        step = self.state.global_step + 1
        smi_val = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True).stdout.strip())
        # Log each sample in the batch
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

trainer = LoggingSFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
    args=training_args,
    callbacks=[LiveProgressCallback()],
)

# Check for existing checkpoint to resume from
# Skip resume if FRESH_START env var is set
import glob as globmod
if os.environ.get('FRESH_START'):
    resume_from = None
    print('FRESH_START: ignoring existing checkpoints')
else:
    checkpoints = sorted(globmod.glob(os.path.join(OUTPUT_DIR, 'checkpoint-*')))
    resume_from = checkpoints[-1] if checkpoints else None
if resume_from:
    print(f'Resuming from {resume_from}')
else:
    print('Starting from scratch')

t0 = time.time()
trainer.train(resume_from_checkpoint=resume_from)
elapsed = time.time() - t0
print(f'\nTraining done. Time: {elapsed / 3600:.2f} hrs ({elapsed / 60:.1f} min)')

# ── Save adapter ────────────────────────────────────────────────────
print('\n=== Saving adapter ===')
trainer.model.save_pretrained(OUTPUT_DIR)

# Fix base_model_name_or_path to competition model
config_path = os.path.join(OUTPUT_DIR, 'adapter_config.json')
with open(config_path) as f:
    adapter_config = json.load(f)

adapter_config['base_model_name_or_path'] = 'metric/nemotron-3-nano-30b-a3b-bf16'

with open(config_path, 'w') as f:
    json.dump(adapter_config, f, indent=2)

print(f"base_model_name_or_path -> {adapter_config['base_model_name_or_path']}")

# Verify weights
try:
    from safetensors import safe_open
    with safe_open(os.path.join(OUTPUT_DIR, 'adapter_model.safetensors'),
                   framework='pt') as f:
        keys = list(f.keys())
        norms = [f.get_tensor(k).norm().item() for k in keys[:5]]
    print(f'Adapter keys: {len(keys)}')
    print(f'Weight norms (first 5): {[f"{n:.4f}" for n in norms]}')
    if all(n < 0.001 for n in norms):
        print('WARNING: Norms near 0 -- adapter may not have trained.')
    else:
        print('Adapter looks healthy.')
except Exception as e:
    print(f'Could not verify: {e}')

# ── Create submission zip ───────────────────────────────────────────
print('\n=== Creating submission zip ===')
REQUIRED = {'adapter_config.json', 'adapter_model.safetensors'}

with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(OUTPUT_DIR):
        if fname in REQUIRED:
            zf.write(os.path.join(OUTPUT_DIR, fname), arcname=fname)

with zipfile.ZipFile(ZIP_PATH) as zf:
    contents = zf.namelist()

print(f'ZIP contents: {contents}')
print(f'ZIP size: {os.path.getsize(ZIP_PATH)/1024/1024:.1f} MB')
assert set(contents) == REQUIRED, f'Wrong files in zip: {contents}'

# ── Quick inference test ────────────────────────────────────────────
print('\n=== Quick inference test ===')
model.eval()

test_prompts = [
    "In Alice's Wonderland, numbers are written in a special numeral system. Convert: 42",
    "In Alice's Wonderland, a secret unit conversion rule transforms measurements.\n10.00 m -> 13.81\n20.00 m -> 27.62\nConvert: 30.00 m",
    "In Alice's Wonderland, the gravitational constant has been changed.\nt=1.00s -> 5.00m\nt=2.00s -> 20.00m\nCalculate distance for t=3.00s",
]

for i, prompt in enumerate(test_prompts):
    full_prompt = prompt + '\nPut your final answer inside \\boxed{}.'
    inputs = tokenizer(full_prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.0, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f'\n[{i}] {prompt[:60]}...')
    print(f'    -> {response[:200]}')

# ── Summary ─────────────────────────────────────────────────────────
print('\n' + '='*60)
print('TRAINING SUMMARY')
print('='*60)
print(f'Method:      QLoRA 4-bit (BNB NF4, pre-quantized)')
print(f'GPU:         {torch.cuda.get_device_name(0)}')
print(f'Samples:     {len(hf_dataset)}')
print(f'Epochs:      {NUM_EPOCHS}')
print(f'LoRA rank:   {LORA_RANK}')
print(f'LR:          {LR}')
print(f'Max seq len: {MAX_SEQ_LEN}')
print(f'Train time:  {elapsed/60:.1f} min ({elapsed/3600:.2f} hrs)')
print(f'Adapter:     {OUTPUT_DIR}')
print(f'Submission:  {ZIP_PATH}')
print('='*60)
