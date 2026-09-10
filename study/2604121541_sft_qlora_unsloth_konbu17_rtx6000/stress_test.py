#!/usr/bin/env python3
"""
Stress test — same QLoRA Unsloth + konbu17 code as train_qlora.py with 3 tweaks:
  1. Dataset filtered to LONG samples (1800-2000 tokens) for worst-case VRAM.
  2. training_args.max_steps = 4, save_strategy = 'no'.
  3. output_dir = stress_output/ so real adapter_output isn't touched.
Exits with STRESS_TEST_OK and peak nvidia-smi VRAM after 4 steps.
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

# ── Config (konbu17 0.72 recipe) ────────────────────────────────────
SEED         = 123
LORA_RANK    = 32
LORA_ALPHA   = 32
MAX_SEQ_LEN  = 2000  # drops 2 outliers; rest of konbu17 dataset ≤ 1960
NUM_EPOCHS   = 2
BATCH_SIZE   = 22    # STRESS TEST: try bs=22 ga=1 (not multiple of 8)
GRAD_ACCUM   = 1
LR           = 1e-4
WARMUP_RATIO = 0.05

# Type-balanced sampling from konbu17's notebook (~2907 total)
TYPE_SAMPLES = {
    'Numeral Conversion':       300,
    'Gravitational Constant':   400,
    'Unit Conversion':          700,
    'Text Encryption':          700,
    'Bit Manipulation':         607,  # all available
    'Equation Transformation':  200,  # all available
}

PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

from datetime import datetime, timezone, timedelta
_cdt = timezone(timedelta(hours=-5))
_ts = datetime.now(_cdt).strftime('%y%m%d_%H%M')
RUN_NAME = f'{_ts}_sft_qlora_unsloth_konbu17_r32_lr1e4_seq2000_ep2_bs4ga2_rtx6000'

MODEL_PATH = '/root/autodl-tmp/Nemotron-3-Nano-30B-A3B'
DATA_PATH  = '/root/autodl-tmp/data/konbu17_verified_cot_6558rows.csv'
OUTPUT_DIR = '/root/autodl-tmp/stress_output'
ZIP_PATH   = '/root/autodl-tmp/stress_submission.zip'

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
    lora_dropout   = 0,  # unsloth fast-patch requires 0; we save 0.0 to adapter_config
    bias           = 'none',
    use_gradient_checkpointing = 'unsloth',
    random_state   = 3407,
    use_rslora     = False,
    loftq_config   = None,
)
print(f'VRAM after LoRA: {torch.cuda.memory_allocated()/1024**3:.1f} GB')

# ── Dataset: konbu17 type-balanced sampling ─────────────────────────
print('\n=== Loading dataset ===')
df = pl.read_csv(DATA_PATH)
print(f'Full dataset: {len(df)} rows')
print(df['type'].value_counts().sort('count', descending=True))

sampled_parts = []
for ptype, n_samples in TYPE_SAMPLES.items():
    subset = df.filter(pl.col('type') == ptype)
    sampled = subset if n_samples >= len(subset) else subset.sample(n=n_samples, seed=SEED)
    print(f'  {ptype}: {len(subset)} -> {len(sampled)}')
    sampled_parts.append(sampled)

train_df = pl.concat(sampled_parts).sample(fraction=1.0, seed=SEED, shuffle=True)
print(f'\nTraining samples: {len(train_df)}')
hf_dataset = Dataset.from_pandas(train_df.to_pandas())

# ── Prompt formatting (konbu17 CoT with </think> close) ────────────
def build_training_text(example):
    prompt = str(example['prompt'])
    answer = str(example['answer'])
    cot = str(example['generated_cot'])
    # konbu17: strip any pre-existing \boxed{} from the CoT so the assistant
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
    cot = str(ex['generated_cot']) if ex['generated_cot'] is not None else ''
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
# STRESS TEST: (1) drop samples > MAX_SEQ_LEN like real training does,
# (2) sort descending by token_len, (3) take top-K so every batch is heavy.
# Combined with SequentialSampler (see LoggingSFTTrainer below), this means
# batches go in strict length-desc order — step 1 has the longest samples
# so OOMs are reproducible and attributable to specific samples.
hf_dataset = hf_dataset.filter(lambda x: x['token_len'] <= MAX_SEQ_LEN, desc='<= MAX_SEQ_LEN')
K = max(BATCH_SIZE * 4, 128)
hf_dataset = hf_dataset.sort('token_len', reverse=True).select(range(min(K, len(hf_dataset))))
_lens = hf_dataset['token_len']
print(f'Kept top-{len(hf_dataset)} longest samples. max={_lens[0]}  min={_lens[-1]}')
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
    save_strategy               = 'no',   # STRESS TEST: don't save
    save_steps                  = 100,
    save_total_limit            = 2,
    max_steps                   = 2,      # STRESS TEST: only 2 steps
    report_to                   = 'swanlab',
    run_name                    = RUN_NAME,
    dataset_text_field          = 'text',
    packing                     = False,
    padding_free                = False,  # bypass unsloth SFTTrainer check (max_length in SFTConfig defaults non-None even when omitted; our dataset is already pre-filtered)
    dataloader_num_workers      = 2,
    seed                        = SEED,
)

class LoggingSFTTrainer(SFTTrainer):
    def _get_train_sampler(self, *args, **kwargs):
        # STRESS TEST: no shuffle — iterate dataset in order (longest first
        # since we pre-sorted) so OOMs can be traced to specific samples.
        from torch.utils.data import SequentialSampler
        return SequentialSampler(self.train_dataset)

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
trainer.train(resume_from_checkpoint=None)  # STRESS TEST: never resume
elapsed = time.time() - t0
print(f'\n=== STRESS TEST DONE ===')
print(f'Time for {training_args.max_steps} steps: {elapsed:.1f}s  ({elapsed/training_args.max_steps:.1f}s/step)')
_peak = subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
                       capture_output=True, text=True).stdout.strip()
print(f'Peak smi VRAM: {_peak} MiB')
print('STRESS_TEST_OK')
sys.exit(0)

# ── Save adapter (no tokenizer — breaks vLLM eval) ─────────────────
print('\n=== Saving adapter ===')
trainer.model.save_pretrained(OUTPUT_DIR)

config_path = os.path.join(OUTPUT_DIR, 'adapter_config.json')
with open(config_path) as f:
    adapter_config = json.load(f)
adapter_config['base_model_name_or_path'] = 'metric/nemotron-3-nano-30b-a3b-bf16'
adapter_config['inference_mode'] = True
adapter_config['lora_dropout'] = 0.0
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

# ── Summary ─────────────────────────────────────────────────────────
print('\n' + '='*60)
print('TRAINING SUMMARY — QLoRA Unsloth + konbu17 0.72 recipe')
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
