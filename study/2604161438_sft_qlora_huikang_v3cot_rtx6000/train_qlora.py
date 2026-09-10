#!/usr/bin/env python3
"""
QLoRA training on AutoDL RTX PRO 6000 — continue from huikang's adapter + v3 CoT.

Start from huikang's 0.877 adapter, train on v3 CoT data for the 3 weak categories:
  - bit_manipulation (1602 rows)
  - equation_cryptarithm (823 rows)
  - equation_numeric (732 rows)
Config: r=32, alpha=16 (scale 0.5), dropout=0.05, lr=1e-4, 1 epoch
Seq len determined by stress test (data is 2000-3200 tokens).
"""

import os, sys, time, json, zipfile, shutil, gc, subprocess, types, re

os.environ['TRANSFORMERS_NO_TF']     = '1'
os.environ['TRANSFORMERS_NO_FLAX']   = '1'
os.environ['CUDA_VISIBLE_DEVICES']   = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HF_HUB_OFFLINE']         = '1'
os.environ['TRANSFORMERS_OFFLINE']   = '1'
os.environ['SWANLAB_API_KEY'] = dict(l.strip().split('=', 1) for l in open('kaggle.env', encoding='utf-8-sig') if '=' in l and not l.startswith('#'))['SWANLAB_API_KEY']
os.environ['SWANLAB_PROJECT']        = '260410_Nemotron'

# ── Unsloth MUST be imported before transformers so its patches land ──
import unsloth
from unsloth import FastLanguageModel
print(f'unsloth: {unsloth.__version__}')

# Stub mamba3 modules (needs cutlass which isn't available)
for _mod_name in [
    'mamba_ssm.modules.mamba3',
    'mamba_ssm.ops.cute',
    'mamba_ssm.ops.cute.mamba3',
    'mamba_ssm.ops.cute.mamba3.mamba3_step_fn',
]:
    _m = types.ModuleType(_mod_name); _m.__path__ = []; _m.__package__ = _mod_name
    sys.modules[_mod_name] = _m
sys.modules['mamba_ssm.modules.mamba3'].Mamba3 = None

import mamba_ssm
print(f'mamba_ssm: {mamba_ssm.__version__}')

# ── Config ──────────────────────────────────────────────────────────
SEED         = 123
LORA_RANK    = 32
LORA_ALPHA   = 16    # huikang's scale 0.5
MAX_SEQ_LEN  = 8000  # v3 CoT: median ~6400, max ~8000 tokens
NUM_EPOCHS   = 1
BATCH_SIZE   = int(os.environ.get('STRESS_BS', 4))  # default 4, overridable for stress test
GRAD_ACCUM   = 1
LR           = 1e-4  # huikang's recommendation
WARMUP_RATIO = 0.05

# Pretrained adapter to load (huikang's 0.877 submission)
PRETRAINED_ADAPTER = '/root/autodl-tmp/model/huikang_adapter'

# v3 CoT data — only weak categories
DATA_FILES = [
    '/root/autodl-tmp/data/260416_train_cot_v3/train_cot_bit_manipulation.csv',
    '/root/autodl-tmp/data/260416_train_cot_v3/train_cot_equation_cryptarithm.csv',
    '/root/autodl-tmp/data/260416_train_cot_v3/train_cot_equation_numeric.csv',
]

PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

from datetime import datetime, timezone, timedelta
_cdt = timezone(timedelta(hours=-5))
_ts = datetime.now(_cdt).strftime('%y%m%d_%H%M')
RUN_NAME = f'{_ts}_sft_qlora_huikang_v3cot_r32_a16_lr1e4_seq8000_ep1_rtx6000'

MODEL_PATH = '/root/autodl-tmp/Nemotron-3-Nano-30B-A3B'
OUTPUT_DIR = '/root/autodl-tmp/adapter_output_huikang_v3cot'
ZIP_PATH   = '/root/autodl-tmp/submission.zip'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Imports ─────────────────────────────────────────────────────────
import torch
import polars as pl
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig
from tqdm.auto import tqdm

print(f'PyTorch : {torch.__version__}')
print(f'GPU     : {torch.cuda.get_device_name(0)}')
print(f'VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
print(f'CUDA cap: {torch.cuda.get_device_capability()}')

# ── Load model + tokenizer via Unsloth FastLanguageModel ───────────
# Local bf16 path; load_in_4bit quantizes attention/FFN on the fly while
# Unsloth keeps Mamba in_proj/out_proj in bf16 so the real mamba-ssm kernel
# path works (pre-quantized BNB storage is [N,1] flat and breaks F.linear).
print('\n=== Loading bf16 model via Unsloth (quantize on the fly) ===')
t_load = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name            = MODEL_PATH,
    max_seq_length        = MAX_SEQ_LEN,
    load_in_4bit          = True,
    load_in_8bit          = False,
    full_finetuning       = False,
    trust_remote_code     = True,
    unsloth_force_compile = True,
    attn_implementation   = 'eager',
    dtype                 = torch.bfloat16,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f'Model loaded in {time.time() - t_load:.1f}s')
print(f'VRAM after load: {torch.cuda.memory_allocated()/1024**3:.1f} GB')

# ── Apply LoRA via Unsloth — konbu17 targets ────────────────────────
# konbu17 uses PEFT regex r'.*\.(in_proj|out_proj|up_proj|down_proj)$' which
# matches Mamba in_proj/out_proj + non-expert FFN up_proj/down_proj +
# MoE expert down_proj (suffix match). Unsloth's get_peft_regex accepts a
# list of bare module names and builds the same suffix regex internally, so
# passing the list below is equivalent.
model = FastLanguageModel.get_peft_model(
    model,
    r              = LORA_RANK,
    target_modules = ['in_proj', 'out_proj', 'up_proj', 'down_proj'],
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = 0.05,  # huikang's config; triggers unsloth slow fallback
    bias           = 'none',
    use_gradient_checkpointing = 'unsloth',
    random_state   = 3407,
    use_rslora     = False,
    loftq_config   = None,
)
print(f'VRAM after LoRA: {torch.cuda.memory_allocated()/1024**3:.1f} GB')

# ── Load pretrained adapter (huikang's 0.877) ──────────────────────
print(f'\n=== Loading pretrained adapter from {PRETRAINED_ADAPTER} ===')
from peft import set_peft_model_state_dict
from safetensors import safe_open
adapter_path = os.path.join(PRETRAINED_ADAPTER, 'adapter_model.safetensors')
with safe_open(adapter_path, framework='pt', device='cpu') as f:
    pretrained_state = {k: f.get_tensor(k) for k in f.keys()}
# Filter to only keys that match our LoRA target modules
model_keys = set(dict(model.named_parameters()).keys())
loaded, skipped = 0, 0
for k, v in pretrained_state.items():
    if k in model_keys:
        dict(model.named_parameters())[k].data.copy_(v.to(dict(model.named_parameters())[k].device))
        loaded += 1
    else:
        skipped += 1
print(f'Loaded {loaded} tensors, skipped {skipped} (not in our target modules)')
print(f'VRAM after adapter load: {torch.cuda.memory_allocated()/1024**3:.1f} GB')

# ── Dataset: v3 CoT for weak categories ─────────────────────────────
print('\n=== Loading dataset ===')
dfs = []
for fpath in DATA_FILES:
    df = pl.read_csv(fpath).with_columns(pl.col('answer').cast(pl.Utf8))
    print(f'  {os.path.basename(fpath)}: {len(df)} rows')
    dfs.append(df)
train_df = pl.concat(dfs).sample(fraction=1.0, seed=SEED, shuffle=True)
print(f'\nTraining samples: {len(train_df)}')
print(train_df['category'].value_counts().sort('count', descending=True))
hf_dataset = Dataset.from_pandas(train_df.to_pandas())

# ── Prompt formatting (konbu17 CoT with </think> close) ────────────
def build_training_text(example):
    prompt = str(example['prompt'])
    answer = str(example['answer'])
    cot = str(example['solver_cot'])
    # Strip any pre-existing \boxed{} from the CoT so the assistant
    # only has one \boxed{} at the end
    cot_cleaned = re.sub(r'\\boxed\{[^}]*\}', '', cot).rstrip()
    user_msg = prompt + PROMPT_SUFFIX
    # Nemotron chat template auto-injects `<think>\n` at the start of the
    # assistant turn; assistant content begins with the CoT body and must
    # close with `</think>` before the final \boxed{} answer
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

# Drop rows with empty / broken CoT
def has_valid_cot(ex):
    cot = str(ex['solver_cot']) if ex['solver_cot'] is not None else ''
    return cot and cot != 'nan' and len(cot.strip()) >= 5

before_filter = len(hf_dataset)
hf_dataset = hf_dataset.filter(has_valid_cot, desc='Dropping empty CoT')
print(f'Dropped {before_filter - len(hf_dataset)} rows with empty CoT')

hf_dataset = hf_dataset.map(
    build_training_text,
    remove_columns=[c for c in hf_dataset.column_names if c != 'id'],
    desc='Applying chat template',
)

# Build lookup: first 50 tokens -> sample_id for per-microbatch logging
token_to_sid = {}
for row in hf_dataset:
    ids = tokenizer(row['text'], truncation=False, return_attention_mask=False)['input_ids']
    token_to_sid[tuple(ids[:50])] = row['sample_id']
hf_dataset = hf_dataset.remove_columns(['sample_id'])

# Drop samples > MAX_SEQ_LEN tokens (pre-filter so SFTConfig.max_length is
# not needed — follows unsloth pattern)
print(f'\nFiltering samples > {MAX_SEQ_LEN} tokens...')
before = len(hf_dataset)
def get_token_length(example):
    ids = tokenizer(example['text'], truncation=False, return_attention_mask=False)['input_ids']
    return {'token_len': len(ids)}
hf_dataset = hf_dataset.map(get_token_length, desc='Counting tokens')
hf_dataset = hf_dataset.filter(lambda x: x['token_len'] <= MAX_SEQ_LEN, desc='Dropping oversized')
hf_dataset = hf_dataset.remove_columns(['token_len'])
print(f'Kept {len(hf_dataset)} / {before} ({before - len(hf_dataset)} dropped)')

STRESS_TEST = bool(os.environ.get('STRESS_TEST'))
if STRESS_TEST:
    # Sort by token length descending for worst-case stress test
    print('\n=== STRESS TEST MODE ===')
    hf_dataset = hf_dataset.map(get_token_length, desc='Re-counting tokens for sort')
    hf_dataset = hf_dataset.sort('token_len', reverse=True)
    K = BATCH_SIZE * 3 * 2  # bs × max_steps × 2
    hf_dataset = hf_dataset.select(range(min(K, len(hf_dataset))))
    print(f'Stress test: {len(hf_dataset)} longest samples, bs={BATCH_SIZE}')
    hf_dataset = hf_dataset.remove_columns(['token_len'])

steps_estimate = len(hf_dataset) // (BATCH_SIZE * GRAD_ACCUM) * NUM_EPOCHS
print(f'Estimated training steps: {steps_estimate}')

# ── Callbacks ───────────────────────────────────────────────────────
class LiveProgressCallback(TrainerCallback):
    def __init__(self):
        self.pbar = None
        self.start_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.pbar = tqdm(total=state.max_steps, desc='Training', unit='step',
                         dynamic_ncols=True, file=sys.stdout)
        self.start_time = time.time()

    def on_step_begin(self, args, state, control, **kwargs):
        gc.collect(); torch.cuda.empty_cache()

    def on_step_end(self, args, state, control, **kwargs):
        gc.collect(); torch.cuda.empty_cache()
        if self.pbar is None:
            return
        elapsed = time.time() - self.start_time
        step = state.global_step
        eta = (elapsed / step) * (state.max_steps - step) if step > 0 else 0
        loss_str = (f"loss={state.log_history[-1]['loss']:.4f}"
                    if state.log_history and 'loss' in state.log_history[-1]
                    else 'loss=...')
        vram = torch.cuda.memory_allocated() / 1024**3
        try:
            smi = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True).stdout.strip())
            smi_str = f'smi={smi}M'
        except Exception:
            smi_str = ''
        self.pbar.set_postfix_str(f'{loss_str}  vram={vram:.1f}G  {smi_str}  elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m')
        self.pbar.update(1)
        sys.stdout.flush()

    def on_save(self, args, state, control, **kwargs):
        if state.log_history:
            last = {}
            for entry in reversed(state.log_history):
                if 'loss' in entry and 'loss' not in last:
                    last['loss']  = entry['loss']
                    last['lr']    = entry.get('learning_rate', 0)
                    last['epoch'] = entry.get('epoch', 0)
                    last['acc']   = entry.get('mean_token_accuracy', 0)
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

# Force save at the final step
class FinalStepSaveCallback(TrainerCallback):
    def on_train_end(self, args, state, control, **kwargs):
        control.should_save = True

# ── Training args (konbu17 config, no max_length per unsloth pattern) ─
print('\n=== Starting training ===')
training_args = SFTConfig(
    output_dir                  = OUTPUT_DIR,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    num_train_epochs            = NUM_EPOCHS,
    learning_rate               = LR,
    logging_steps               = 10,
    bf16                        = True,
    bf16_full_eval              = True,
    max_grad_norm               = 1.0,
    optim                       = 'adamw_torch',
    lr_scheduler_type           = 'cosine',
    warmup_ratio                = WARMUP_RATIO,
    max_steps                   = 3 if STRESS_TEST else -1,
    save_strategy               = 'no' if STRESS_TEST else 'steps',
    save_steps                  = 150,  # ~789 total steps / 5 ≈ 158, rounded to 150
    save_total_limit            = 2,
    report_to                   = 'none' if STRESS_TEST else 'swanlab',
    run_name                    = RUN_NAME,
    dataset_text_field          = 'text',
    packing                     = False,
    padding_free                = False,  # bypass unsloth SFTTrainer check (max_length in SFTConfig defaults non-None even when omitted; our dataset is already pre-filtered)
    dataloader_num_workers      = 2,
    seed                        = SEED,
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

trainer = LoggingSFTTrainer(
    model            = model,
    train_dataset    = hf_dataset,
    processing_class = tokenizer,
    args             = training_args,
    callbacks        = [LiveProgressCallback(), FinalStepSaveCallback()],
)

# Resume from latest checkpoint unless FRESH_START=1
import glob as globmod
if os.environ.get('FRESH_START'):
    resume_from = None
    print('FRESH_START: ignoring existing checkpoints')
else:
    checkpoints = sorted(globmod.glob(os.path.join(OUTPUT_DIR, 'checkpoint-*')))
    resume_from = checkpoints[-1] if checkpoints else None
print(f'Resuming from {resume_from}' if resume_from else 'Starting from scratch')

t0 = time.time()
trainer.train(resume_from_checkpoint=resume_from)
elapsed = time.time() - t0
print(f'\nTraining done. Time: {elapsed / 3600:.2f} hrs ({elapsed / 60:.1f} min)')

# ── Save adapter (no tokenizer — breaks vLLM eval) ─────────────────
print('\n=== Saving adapter ===')
trainer.model.save_pretrained(OUTPUT_DIR)

config_path = os.path.join(OUTPUT_DIR, 'adapter_config.json')
with open(config_path) as f:
    adapter_config = json.load(f)
adapter_config['base_model_name_or_path'] = 'metric/nemotron-3-nano-30b-a3b-bf16'
adapter_config['inference_mode'] = True
adapter_config['lora_dropout'] = 0.05
with open(config_path, 'w') as f:
    json.dump(adapter_config, f, indent=2)
print(f"base_model_name_or_path -> {adapter_config['base_model_name_or_path']}")

try:
    from safetensors import safe_open
    with safe_open(os.path.join(OUTPUT_DIR, 'adapter_model.safetensors'), framework='pt') as f:
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

# ── Zip final checkpoint folder ───────────────────────────────────────
print('\n=== Zipping final checkpoint ===')
ckpt_dirs = sorted([d for d in os.listdir(OUTPUT_DIR) if d.startswith('checkpoint-')],
                   key=lambda x: int(x.split('-')[1].split('_')[0]))
if ckpt_dirs:
    final_ckpt = ckpt_dirs[-1]
    ckpt_zip_path = os.path.join(os.path.dirname(OUTPUT_DIR), f'{final_ckpt}.zip')
    ckpt_full = os.path.join(OUTPUT_DIR, final_ckpt)
    with zipfile.ZipFile(ckpt_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(ckpt_full):
            for f in files:
                fp = os.path.join(root, f)
                zf.write(fp, arcname=os.path.relpath(fp, ckpt_full))
    print(f'Checkpoint zip: {ckpt_zip_path} ({os.path.getsize(ckpt_zip_path)/1024/1024:.1f} MB)')
else:
    print('WARNING: no checkpoint dirs found')

# ── Summary ─────────────────────────────────────────────────────────
print('\n' + '='*60)
print('TRAINING SUMMARY — QLoRA Unsloth + huikang adapter + v3 CoT')
print('='*60)
print(f'Method:       QLoRA via Unsloth (load_in_4bit=True on bf16 source)')
print(f'GPU:          {torch.cuda.get_device_name(0)}')
print(f'Samples:      {len(hf_dataset)}')
print(f'Epochs:       {NUM_EPOCHS}')
print(f'LoRA rank:    {LORA_RANK}  alpha: {LORA_ALPHA}')
print(f'LoRA targets: in_proj, out_proj, up_proj, down_proj')
print(f'LR:           {LR}  warmup: {WARMUP_RATIO}')
print(f'Max seq len:  {MAX_SEQ_LEN}')
print(f'Eff batch:    {BATCH_SIZE * GRAD_ACCUM}')
print(f'mamba_ssm:    {mamba_ssm.__version__}')
print(f'unsloth:      {unsloth.__version__}')
print(f'Train time:   {elapsed/60:.1f} min ({elapsed/3600:.2f} hrs)')
print(f'Adapter:      {OUTPUT_DIR}')
print(f'Submission:   {ZIP_PATH}')
print('='*60)
