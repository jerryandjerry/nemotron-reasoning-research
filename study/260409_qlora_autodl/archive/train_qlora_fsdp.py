#!/usr/bin/env python3
"""
FSDP + QLoRA Training on 2x RTX 4090 (24GB each).
Model sharded across GPUs — each holds ~8.75GB instead of 17.5GB.
Uses Answer.AI's bnb_4bit_quant_storage trick for FSDP compatibility.
"""

import os, sys, time, json, zipfile, shutil

os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['TRANSFORMERS_NO_FLAX'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# ── Config (same as 0.72 scheme) ────────────────────────────────────
SAMPLE_SIZE  = 3000
LORA_RANK    = 32
MAX_SEQ_LEN  = 2048
NUM_EPOCHS   = 1
BATCH_SIZE   = 1
GRAD_ACCUM   = 2       # 2 GPUs × 1 batch × 2 accum = effective batch 4
LR           = 1e-4

MODEL_PATH   = '/root/autodl-tmp/Nemotron-3-Nano-30B-A3B-bnb-4bit'
DATA_PATH    = '/root/autodl-tmp/data/final_Nemotron_training_data.csv'
OUTPUT_DIR   = '/root/autodl-tmp/adapter_output_fsdp'
ZIP_PATH     = '/root/autodl-tmp/submission_fsdp.zip'

os.makedirs(OUTPUT_DIR, exist_ok=True)

local_rank = int(os.environ.get('LOCAL_RANK', 0))

# Clear HF cache (main process only)
if local_rank == 0:
    hf_cache_dir = os.path.expanduser('~/.cache/huggingface/modules/transformers_modules')
    if os.path.exists(hf_cache_dir):
        shutil.rmtree(hf_cache_dir)
        print('Cleared HF cache')

# ── Imports ─────────────────────────────────────────────────────────
import torch
import polars as pl
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from tqdm.auto import tqdm
from torch.distributed.fsdp import MixedPrecision

if local_rank == 0:
    print(f'PyTorch : {torch.__version__}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}   : {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)')
    import bitsandbytes as bnb
    print(f'BNB     : {bnb.__version__}')
    print(f'FSDP + QLoRA mode: model sharded across {torch.cuda.device_count()} GPUs')

# ── Dataset ─────────────────────────────────────────────────────────
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

filtered_df = train_df.filter(mask)
if len(filtered_df) < SAMPLE_SIZE:
    sampled_df = filtered_df
else:
    sampled_df = filtered_df.sample(n=SAMPLE_SIZE, seed=99)

hf_dataset = Dataset.from_pandas(sampled_df.to_pandas())
if local_rank == 0:
    print(f'Total: {len(train_df)}, Filtered: {len(filtered_df)}, Sampled: {len(hf_dataset)}')

# ── Tokenizer ───────────────────────────────────────────────────────
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
    return {'text': text}

hf_dataset = hf_dataset.map(build_training_text, remove_columns=hf_dataset.column_names, desc='Chat template')

before = len(hf_dataset)
def get_token_length(example):
    ids = tokenizer(example['text'], truncation=False, return_attention_mask=False)['input_ids']
    return {'token_len': len(ids)}

hf_dataset = hf_dataset.map(get_token_length, desc='Counting tokens')
hf_dataset = hf_dataset.filter(lambda x: x['token_len'] <= MAX_SEQ_LEN, desc='Filter')
hf_dataset = hf_dataset.remove_columns(['token_len'])
if local_rank == 0:
    print(f'Kept {len(hf_dataset)} / {before} ({before - len(hf_dataset)} dropped)')

# ── Load model with FSDP-compatible BNB config ──────────────────────
if local_rank == 0:
    print('\n=== Loading model with FSDP-compatible quantization ===')
t_load = time.time()

# Key: bnb_4bit_quant_storage=torch.bfloat16 enables FSDP sharding of Params4bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_storage=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map={'': local_rank},
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

if local_rank == 0:
    print(f'Model loaded in {time.time() - t_load:.1f}s')
    print(f'GPU {local_rank} VRAM: {torch.cuda.memory_allocated(local_rank)/1024**3:.1f} GB')

model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()

# Cast all non-quantized parameters to bf16 so FSDP sees uniform dtype
for name, param in model.named_parameters():
    if param.dtype == torch.float32:
        param.data = param.data.to(torch.bfloat16)

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=32,
    target_modules='all-linear',
    lora_dropout=0.05,
    bias='none',
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
if local_rank == 0:
    model.print_trainable_parameters()

# ── Progress callback ───────────────────────────────────────────────
class LiveProgressCallback(TrainerCallback):
    def __init__(self):
        self.pbar = None
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        if args.local_rank in (-1, 0):
            self.pbar = tqdm(total=state.max_steps, desc='Training', unit='step',
                             dynamic_ncols=True, file=sys.stdout)
            self.start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
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
        self.pbar.set_postfix_str(f'{loss_str}  vram={vram:.1f}G  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m')
        self.pbar.update(1)
        sys.stdout.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.close()

# ── Training with FSDP ──────────────────────────────────────────────
if local_rank == 0:
    print(f'\n=== Starting FSDP training ===')

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    logging_steps=10,
    bf16=True,
    max_grad_norm=1.0,
    optim='adamw_torch',
    lr_scheduler_type='cosine',
    warmup_ratio=0.1,
    save_strategy='steps',
    save_steps=100,
    save_total_limit=2,
    report_to='none',
    dataset_text_field='text',
    max_length=MAX_SEQ_LEN,
    packing=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},
    ddp_find_unused_parameters=False,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=hf_dataset,
    processing_class=tokenizer,
    args=training_args,
    callbacks=[LiveProgressCallback()],
)

t0 = time.time()
trainer.train()
elapsed = time.time() - t0

if local_rank == 0:
    print(f'\nTraining done. Time: {elapsed / 3600:.2f} hrs ({elapsed / 60:.1f} min)')

    # Save adapter
    trainer.model.save_pretrained(OUTPUT_DIR)
    config_path = os.path.join(OUTPUT_DIR, 'adapter_config.json')
    with open(config_path) as f:
        adapter_config = json.load(f)
    adapter_config['base_model_name_or_path'] = 'metric/nemotron-3-nano-30b-a3b-bf16'
    with open(config_path, 'w') as f:
        json.dump(adapter_config, f, indent=2)

    try:
        from safetensors import safe_open
        with safe_open(os.path.join(OUTPUT_DIR, 'adapter_model.safetensors'), framework='pt') as f:
            keys = list(f.keys())
            norms = [f.get_tensor(k).norm().item() for k in keys[:5]]
        print(f'Adapter keys: {len(keys)}, norms: {[f"{n:.4f}" for n in norms]}')
    except Exception as e:
        print(f'Could not verify: {e}')

    REQUIRED = {'adapter_config.json', 'adapter_model.safetensors'}
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(OUTPUT_DIR):
            if fname in REQUIRED:
                zf.write(os.path.join(OUTPUT_DIR, fname), arcname=fname)
    print(f'ZIP size: {os.path.getsize(ZIP_PATH)/1024/1024:.1f} MB')

    print('\n' + '='*60)
    print('TRAINING SUMMARY')
    print('='*60)
    print(f'Method:      FSDP + QLoRA (2x GPU sharded)')
    print(f'Samples:     {len(hf_dataset)}')
    print(f'Epochs:      {NUM_EPOCHS}')
    print(f'LoRA rank:   {LORA_RANK}')
    print(f'LR:          {LR}')
    print(f'Max seq len: {MAX_SEQ_LEN}')
    print(f'Train time:  {elapsed/60:.1f} min ({elapsed/3600:.2f} hrs)')
    print('='*60)
