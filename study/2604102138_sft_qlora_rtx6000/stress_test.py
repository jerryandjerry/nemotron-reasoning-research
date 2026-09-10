#!/usr/bin/env python3
"""
Stress test: same training code, batch_size=4, grad_accum=1, 2300-2500 token samples only.
Logs sample ID, length, content at each step. Appends to log file.
"""

import os, sys, time, json, shutil, gc, subprocess

os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ── Config ──────────────────────────────────────────────────────────
LORA_RANK    = 32
MAX_SEQ_LEN  = 2500
NUM_EPOCHS   = 1
BATCH_SIZE   = 8
GRAD_ACCUM   = 2
LR           = 2e-4

MODEL_PATH   = '/root/autodl-tmp/Nemotron-3-Nano-30B-A3B-bnb-4bit'
DATA_PATH    = '/root/autodl-tmp/data/final_Nemotron_training_data.csv'
OUTPUT_DIR   = '/root/autodl-tmp/stress_test_output'
STEP_LOG     = '/root/autodl-tmp/stress_test_step_log.txt'

os.makedirs(OUTPUT_DIR, exist_ok=True)

hf_cache_dir = os.path.expanduser('~/.cache/huggingface/modules/transformers_modules')
if os.path.exists(hf_cache_dir):
    shutil.rmtree(hf_cache_dir)

def smi_mib():
    r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True)
    return int(r.stdout.strip())

# ── Imports ─────────────────────────────────────────────────────────
import torch
import polars as pl
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

# ── Dataset: only 2300-2500 token samples ───────────────────────────
print('\n=== Loading dataset ===')
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

# Build texts, keep sample IDs, filter to 2300-2500 tokens
samples = []
for idx, row in enumerate(filtered_df.iter_rows(named=True)):
    sample_id = row['id']
    prompt = row['prompt']
    cot = str(row['generated_cot'])
    answer = str(row['answer'])
    label = row['label']

    user_msg = prompt + '\nPut your final answer inside \\boxed{}.'
    assistant_msg = f"{cot}\n\n\\boxed{{{answer}}}"
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
    if 2000 <= tlen <= 3000:
        samples.append({
            'text': text,
            'sample_id': sample_id,
            'token_len': tlen,
            'label': label,
            'prompt_preview': prompt[:200],
            'cot_preview': cot[:200],
        })

# Sort by length (no shuffle)
samples.sort(key=lambda x: x['token_len'])

print(f'Stress test samples (2300-2500 tokens): {len(samples)}')
for i, s in enumerate(samples):
    print(f'  [{i}] id={s["sample_id"]} tokens={s["token_len"]} label={s["label"][:50]}')

# Pad to multiple of BATCH_SIZE if needed
while len(samples) % BATCH_SIZE != 0:
    samples.append(samples[-1])
    print(f'  Duplicated last sample to pad to batch_size={BATCH_SIZE}')

hf_dataset = Dataset.from_dict({
    'text': [s['text'] for s in samples],
    'sample_id': [s['sample_id'] for s in samples],
    'token_len': [s['token_len'] for s in samples],
    'label': [s['label'] for s in samples],
    'prompt_preview': [s['prompt_preview'] for s in samples],
    'cot_preview': [s['cot_preview'] for s in samples],
})

# Build lookup: first 50 tokens -> sample_id, for matching during training
token_to_id = {}
for s in samples:
    ids = tokenizer(s['text'], truncation=False, return_attention_mask=False)['input_ids']
    key = tuple(ids[:50])
    token_to_id[key] = (s['sample_id'], s['token_len'], s['label'])

# Remove metadata columns, keep only 'text'
hf_dataset = hf_dataset.remove_columns(['sample_id', 'token_len', 'label', 'prompt_preview', 'cot_preview'])

print(f'Dataset ready: {len(hf_dataset)} samples')

# ── Load model ──────────────────────────────────────────────────────
print('\n=== Loading model ===')
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, device_map={'': 0}, trust_remote_code=True, torch_dtype=torch.bfloat16,
)
print(f'Model loaded. smi={smi_mib()}M')

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=LORA_RANK, lora_alpha=32, target_modules='all-linear',
    lora_dropout=0.05, bias='none', task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
print(f'After LoRA. smi={smi_mib()}M')

# ── Logging trainer ─────────────────────────────────────────────────
class LoggingSFTTrainer(SFTTrainer):
    def __init__(self, *args, token_to_id=None, step_log_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_to_id = token_to_id or {}
        self.step_log_file = step_log_file
        self._micro_count = 0

    def training_step(self, model, inputs, num_items_in_batch=None):
        self._micro_count += 1
        step = self.state.global_step + 1
        batch_size = inputs['input_ids'].shape[0] if 'input_ids' in inputs else '?'
        seq_len = inputs['input_ids'].shape[-1] if 'input_ids' in inputs else '?'
        smi_before = smi_mib()

        # Decode full content for each sample in the batch
        log_lines = []
        if 'input_ids' in inputs:
            for b in range(inputs['input_ids'].shape[0]):
                ids = inputs['input_ids'][b].tolist()
                # Strip padding tokens
                pad_id = tokenizer.pad_token_id
                ids_nopad = [t for t in ids if t != pad_id]
                content = tokenizer.decode(ids_nopad, skip_special_tokens=True)
                actual_len = len(ids)
                # Look up sample ID
                key = tuple(ids[:50])
                sid, slen, slabel = token_to_id.get(key, ('?', actual_len, '?'))
                log_lines.append(
                    f'  batch[{b}] sample_id={sid} padded_len={actual_len} content: {content[-20:]}'
                )

        header = f'>> step={step} micro={self._micro_count} batch_size={batch_size} seq_len={seq_len} smi_before={smi_before}M'
        print(f'\n{header}', flush=True)
        for l in log_lines:
            print(l, flush=True)

        # Append to log file
        with open(self.step_log_file, 'a') as f:
            f.write(f'{header}\n')
            for l in log_lines:
                f.write(f'{l}\n')

        loss = super().training_step(model, inputs, num_items_in_batch)

        smi_after = smi_mib()
        loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
        result_line = f'   step={step} smi_after={smi_after}M loss={loss_val:.4f}'
        print(result_line, flush=True)
        with open(self.step_log_file, 'a') as f:
            f.write(f'{result_line}\n')

        return loss

# ── Callback ────────────────────────────────────────────────────────
class StressCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()

    def on_step_end(self, args, state, control, **kwargs):
        gc.collect()
        torch.cuda.empty_cache()

# ── Training ────────────────────────────────────────────────────────
print(f'\n=== Stress test: adamw_torch, batch_size={BATCH_SIZE}, grad_accum={GRAD_ACCUM} ===')

# Write log header (append mode)
with open(STEP_LOG, 'a') as f:
    f.write(f'\n=== Stress test started at {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')
    f.write(f'GPU: {torch.cuda.get_device_name(0)}\n')
    f.write(f'batch_size={BATCH_SIZE}, grad_accum={GRAD_ACCUM}, optim=adamw_torch\n')
    f.write(f'Samples: {len(hf_dataset)} (2300-2500 tokens)\n\n')

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
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
    max_length=MAX_SEQ_LEN,
    packing=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
)

trainer = LoggingSFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
    args=training_args,
    callbacks=[StressCallback()],
    token_to_id=token_to_id,
    step_log_file=STEP_LOG,
)

t0 = time.time()
try:
    trainer.train()
    elapsed = time.time() - t0
    msg = f'\nStress test completed in {elapsed/60:.1f} min'
    print(msg)
    with open(STEP_LOG, 'a') as f:
        f.write(msg + '\n')
except torch.cuda.OutOfMemoryError as e:
    msg = f'\n*** OOM at step {trainer.state.global_step} smi={smi_mib()}M: {e}'
    print(msg)
    with open(STEP_LOG, 'a') as f:
        f.write(msg + '\n')
except Exception as e:
    msg = f'\n*** Error: {e}'
    print(msg)
    with open(STEP_LOG, 'a') as f:
        f.write(msg + '\n')

print('\n=== Done ===')
