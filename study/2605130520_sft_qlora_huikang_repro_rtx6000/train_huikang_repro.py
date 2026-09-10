#!/usr/bin/env python3
"""
Reproduce huikang's 0.85 Kaggle training on AutoDL RTX PRO 6000.

Exact reproduction of huikang's end-to-end-finetuning-for-lb-0-85.ipynb:
  - load_in_4bit=False (full bf16 base model)
  - Cut Cross-Entropy (no logit materialization)
  - MoE weight tying (Tinker-style)
  - Manual lm_head LoRA
  - LoRA params in fp32, base in bf16
  - Response-only masking (prompt tokens weight=0)
  - LR 2e-4, linear decay to 0
  - AdamW beta2=0.95, no weight decay
  - 9 target modules: q/k/v/o_proj, up/down_proj, in/out_proj, lm_head
  - BATCH_SIZE=32, MICRO_BATCH_SIZE=4
  - Replays training order from huikang's winning run
  - RESET_WEIGHTS=True (fresh LoRA init)
"""

import os, sys, time, json, zipfile, gc, subprocess, types, math

os.environ['TRANSFORMERS_NO_TF']     = '1'
os.environ['TRANSFORMERS_NO_FLAX']   = '1'
os.environ['CUDA_VISIBLE_DEVICES']   = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HF_HUB_OFFLINE']         = '1'
os.environ['TRANSFORMERS_OFFLINE']   = '1'
os.environ['SWANLAB_API_KEY'] = dict(l.strip().split('=', 1) for l in open('kaggle.env', encoding='utf-8-sig') if '=' in l and not l.startswith('#'))['SWANLAB_API_KEY']
os.environ['SWANLAB_PROJECT']        = '260410_Nemotron'

# ── Config (exact huikang reproduction) ─────────────────────────────
LORA_RANK    = 32
LORA_ALPHA   = 32
LORA_DROPOUT = 0.0
MAX_SEQ_LEN  = 8192
BATCH_SIZE   = 32
MICRO_BATCH_SIZE = 4
LEARNING_RATE = 2e-4
NUM_STEPS    = 1000  # will be clamped to max_steps
MOE_TIE_WEIGHTS = True
SHUFFLE_DATASET = False
RESET_WEIGHTS = True

TARGET_MODULES = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'up_proj', 'down_proj', 'in_proj', 'out_proj',
    'lm_head',
]

# ── Paths ───────────────────────────────────────────────────────────
MODEL_PATH = '/root/autodl-tmp/Nemotron-3-Nano-30B-A3B'
CORPUS_PATH = '/root/autodl-tmp/data/huikang_corpus/04-08-16-14/tokens'
TRAIN_ORDER_PATH = '/root/autodl-tmp/data/huikang_corpus/04-08-16-14/logprobs/index.jsonl'
OUTPUT_DIR = '/root/autodl-tmp/adapter_output_huikang_repro'
ZIP_PATH   = '/root/autodl-tmp/submission.zip'

STRESS_TEST = bool(os.environ.get('STRESS_TEST'))

os.makedirs(OUTPUT_DIR, exist_ok=True)

from datetime import datetime, timezone, timedelta
_cdt = timezone(timedelta(hours=-5))
_ts = datetime.now(_cdt).strftime('%y%m%d_%H%M')
RUN_NAME = f'{_ts}_sft_qlora_huikang_repro_r32_a32_lr2e4_seq8192_bs32_rtx6000'

# ── Imports ─────────────────────────────────────────────────────────
# Unsloth MUST be imported before transformers
import unsloth
from unsloth import FastLanguageModel
print(f'unsloth: {unsloth.__version__}')

# Stub mamba3 modules
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
import torch
from cut_cross_entropy import linear_cross_entropy
from peft import LoraConfig
from peft.tuners.lora import Linear as LoraLinear

print(f'mamba_ssm: {mamba_ssm.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

# ── Load corpus in training order ───────────────────────────────────
print('\n=== Loading corpus ===')
ordered_ids = []
seen = set()
with open(TRAIN_ORDER_PATH) as f:
    for line in f:
        rec = json.loads(line)
        if rec.get('epoch', 0) != 0:
            continue
        pid = rec['problem_id']
        if pid in seen:
            continue
        seen.add(pid)
        ordered_ids.append(pid)
print(f'Loaded {len(ordered_ids)} problem_ids in training order')

examples = []
for sid in ordered_ids:
    seg_path = os.path.join(CORPUS_PATH, sid, 'synthetic.json')
    if not os.path.isfile(seg_path):
        print(f'WARNING: missing {seg_path}, skipping')
        continue
    with open(seg_path) as f:
        rec = json.load(f)
    tokens = rec['tokens']
    mask = rec['mask']
    if not tokens:
        continue
    if len(tokens) > MAX_SEQ_LEN:
        tokens = tokens[:MAX_SEQ_LEN]
        mask = mask[:MAX_SEQ_LEN]
    if not any(mask):
        continue
    examples.append({
        'problem_id': sid,
        'tokens': tokens[:-1],
        'targets': tokens[1:],
        'weights': [float(m) for m in mask[1:]],
    })

total_unmasked = sum(sum(e['weights']) for e in examples)
total_tokens = sum(len(e['tokens']) for e in examples)
print(f'Loaded {len(examples)} examples, {total_tokens:,} tokens (unmasked={total_unmasked:,.0f})')

if STRESS_TEST:
    print('\n=== STRESS TEST MODE ===')
    # Sort by length descending, take top-K
    examples.sort(key=lambda e: len(e['tokens']), reverse=True)
    K = BATCH_SIZE * 2
    examples = examples[:min(K, len(examples))]
    print(f'Stress test: {len(examples)} longest examples')

# ── Load base model (full bf16, NOT 4-bit) ──────────────────────────
print('\n=== Loading bf16 model via Unsloth ===')
gc.collect(); torch.cuda.empty_cache()
t_load = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
    trust_remote_code=True,
    unsloth_force_compile=True,
    attn_implementation='eager',
    dtype=torch.bfloat16,
)
print(f'Model loaded in {time.time() - t_load:.1f}s')

def _smi():
    try:
        return int(subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True).stdout.strip())
    except:
        return 0

print(f'VRAM after load: smi={_smi()}M')

# ── Wrap in LoRA ────────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias='none',
    use_gradient_checkpointing='unsloth',
    random_state=42,
)
FastLanguageModel.for_training(model)

# ── Patch Mamba CUDA fast path ──────────────────────────────────────
nemotron_mod = None
for _name, _m in sys.modules.items():
    if 'modeling_nemotron_h' in _name and hasattr(_m, 'is_fast_path_available'):
        nemotron_mod = _m
        break
if nemotron_mod is not None:
    print(f'is_fast_path_available was: {nemotron_mod.is_fast_path_available}')
    nemotron_mod.is_fast_path_available = True
    print('Patched is_fast_path_available = True')

# ── Manually add lm_head LoRA ───────────────────────────────────────
_causal_lm = model
while hasattr(_causal_lm, 'model'):
    _causal_lm = _causal_lm.model
_lm_head = _causal_lm.lm_head
if not isinstance(_lm_head, LoraLinear):
    _cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)
    model.base_model._create_and_replace(
        _cfg, 'default', target=_lm_head, target_name='lm_head', parent=_causal_lm,
    )
    print('Manually added LoRA to lm_head')
else:
    print('lm_head already has LoRA')

# ── Cast LoRA params to fp32 ────────────────────────────────────────
for name, param in model.named_parameters():
    if '.lora_' in name:
        param.data = param.data.to(torch.float32)
print('Cast LoRA params to fp32')

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f'Model: {trainable:,} trainable / {total:,} total')

# ── Patch forward with Cut Cross-Entropy ────────────────────────────
_base = model
while hasattr(_base, 'model'):
    _base = _base.model

def _patched_causal_forward(input_ids=None, attention_mask=None, labels=None, **kwargs):
    backbone_out = _base.backbone(
        input_ids=input_ids, attention_mask=attention_mask,
        **{k: v for k, v in kwargs.items() if k in ('position_ids', 'past_key_values', 'use_cache')},
    )
    hidden_states = backbone_out[0]
    lm_head = _base.lm_head
    base_w = lm_head.base_layer.weight
    lora_A = lm_head.lora_A['default'].weight
    lora_B = lm_head.lora_B['default'].weight
    scaling = lm_head.scaling['default']
    lm_weight = base_w + scaling * lora_B @ lora_A
    if labels is not None:
        per_token_ce = linear_cross_entropy(hidden_states, lm_weight, labels, reduction='none')
        loss = per_token_ce.mean()
    else:
        per_token_ce = None
        loss = None
    model._cached_per_token_ce = per_token_ce
    return loss

_base.forward = _patched_causal_forward
print('Patched CausalLM.forward with CCE')

# ── MoE weight tying ────────────────────────────────────────────────
moe_tied_params = []
if MOE_TIE_WEIGHTS:
    w1_proj_names = ('gate_up_proj', 'up_proj', 'gate_proj', '.w1.')
    w2_proj_names = ('down_proj', '.w2.')
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if '.experts.' not in name or '.lora_' not in name:
            continue
        is_w1 = any(p in name for p in w1_proj_names)
        is_w2 = any(p in name for p in w2_proj_names)
        is_A = '.lora_A.' in name
        is_B = '.lora_B.' in name
        should_tie = (is_w1 and is_A) or (is_w2 and is_B)
        if not should_tie:
            continue
        if param.dim() < 2 or param.shape[0] <= 1:
            continue
        moe_tied_params.append(param)

    def _tie_param_init():
        with torch.no_grad():
            for p in moe_tied_params:
                mean = p.data.mean(dim=0, keepdim=True)
                p.data.copy_(mean.expand_as(p.data))

    def _tie_grads():
        with torch.no_grad():
            for p in moe_tied_params:
                if p.grad is None:
                    continue
                grad_sum = p.grad.sum(dim=0, keepdim=True)
                p.grad.copy_(grad_sum.expand_as(p.grad))

    print(f'MoE weight tying: {len(moe_tied_params)} params')
    _tie_param_init()
else:
    def _tie_grads():
        pass

# ── Training loop ───────────────────────────────────────────────────
gc.collect(); torch.cuda.empty_cache()
device = next(model.parameters()).device
optimizer = None

indices = list(range(len(examples)))
if SHUFFLE_DATASET:
    import random
    rng = random.Random(0)
    rng.shuffle(indices)

max_steps = len(examples) // BATCH_SIZE
num_steps = min(NUM_STEPS, max_steps)
if STRESS_TEST:
    num_steps = 3

print(f'\n=== Starting training ===')
print(f'Steps: {num_steps}, batch_size={BATCH_SIZE}, micro_batch={MICRO_BATCH_SIZE}, lr={LEARNING_RATE}')
print(f'VRAM before training: smi={_smi()}M')

t0 = time.time()
step = 0
for batch_start in range(0, len(indices), BATCH_SIZE):
    if step >= num_steps:
        break
    gc.collect(); torch.cuda.empty_cache()

    batch_indices = indices[batch_start:batch_start + BATCH_SIZE]
    batch = [examples[i] for i in batch_indices]
    batch_tokens = [e['tokens'] for e in batch]
    batch_targets = [e['targets'] for e in batch]
    batch_weights = [e['weights'] for e in batch]

    n = len(batch)
    n_accum = math.ceil(n / MICRO_BATCH_SIZE)
    total_loss_sum = 0.0
    total_weight_sum = 0.0

    for mb_start in range(0, n, MICRO_BATCH_SIZE):
        mb_end = min(mb_start + MICRO_BATCH_SIZE, n)
        mb_toks = batch_tokens[mb_start:mb_end]
        mb_tgts = batch_targets[mb_start:mb_end]
        mb_wts = batch_weights[mb_start:mb_end]
        mb_pids = [batch[i]['problem_id'] for i in range(mb_start, mb_end)]

        n_micro = len(mb_toks)
        max_len = max(len(t) for t in mb_toks)
        total_len = sum(len(t) for t in mb_toks)

        padded_input = torch.zeros(n_micro, max_len, dtype=torch.long, device=device)
        padded_targets = torch.zeros(n_micro, max_len, dtype=torch.long, device=device)
        padded_weights = torch.zeros(n_micro, max_len, dtype=torch.float32, device=device)
        attention_mask = torch.zeros(n_micro, max_len, dtype=torch.long, device=device)
        for i in range(n_micro):
            seq_len = len(mb_toks[i])
            padded_input[i, :seq_len] = torch.tensor(mb_toks[i], dtype=torch.long)
            padded_targets[i, :seq_len] = torch.tensor(mb_tgts[i], dtype=torch.long)
            padded_weights[i, :seq_len] = torch.tensor(mb_wts[i], dtype=torch.float32)
            attention_mask[i, :seq_len] = 1

        smi_before = _smi()
        # Per micro-batch logging
        for b in range(n_micro):
            seq_len = len(mb_toks[b])
            print(f'>> step={step+1} batch[{mb_start+b}] sample_id={mb_pids[b]} padded_len={max_len} smi={smi_before}M', flush=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            model(input_ids=padded_input, attention_mask=attention_mask, labels=padded_targets, use_cache=False)
            per_token_ce = model._cached_per_token_ce
            weighted_loss = per_token_ce * padded_weights
            weight_sum_t = padded_weights.sum()
            loss_sum_t = weighted_loss.sum()
            loss = loss_sum_t / weight_sum_t if weight_sum_t > 0 else loss_sum_t * 0.0

        (loss / n_accum).backward()
        total_loss_sum += loss_sum_t.item()
        total_weight_sum += weight_sum_t.item()
        del loss, per_token_ce, weighted_loss

    # Optimizer step
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
        )
    lr = LEARNING_RATE * (1 - step / num_steps)
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    _tie_grads()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1e9)
    optimizer.step()
    optimizer.zero_grad()

    loss_mean = total_loss_sum / total_weight_sum if total_weight_sum > 0 else 0
    smi_after = _smi()
    step += 1
    elapsed = time.time() - t0
    eta = (elapsed / step) * (num_steps - step) if step > 0 else 0
    print(f'   step={step} smi_after={smi_after}M loss={loss_mean:.6f} grad_norm={grad_norm:.4f} lr={lr:.2e} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m', flush=True)

    gc.collect(); torch.cuda.empty_cache()

elapsed = time.time() - t0
print(f'\nTraining done. Time: {elapsed/3600:.2f} hrs ({elapsed/60:.1f} min)')
print(f'Peak VRAM: smi={_smi()}M')

if STRESS_TEST:
    print('STRESS TEST COMPLETE')
    sys.exit(0)

# ── Save adapter + rename lm_head keys ──────────────────────────────
print('\n=== Saving adapter ===')
from safetensors.torch import load_file, save_file

model.save_pretrained(OUTPUT_DIR)
st_path = os.path.join(OUTPUT_DIR, 'adapter_model.safetensors')
tensors = load_file(st_path)
renamed = {
    k.replace('base_model.model.lm_head.', 'base_model.model.backbone.lm_head.'): v
    for k, v in tensors.items()
}
save_file(renamed, st_path)
print(f'Saved and renamed lm_head keys')

# Fix adapter_config.json
config_path = os.path.join(OUTPUT_DIR, 'adapter_config.json')
with open(config_path) as f:
    adapter_config = json.load(f)
adapter_config['base_model_name_or_path'] = 'metric/nemotron-3-nano-30b-a3b-bf16'
adapter_config['inference_mode'] = True
with open(config_path, 'w') as f:
    json.dump(adapter_config, f, indent=2)

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

# ── Summary ─────────────────────────────────────────────────────────
print('\n' + '='*60)
print('TRAINING SUMMARY — huikang 0.85 reproduction')
print('='*60)
print(f'Method:       LoRA bf16 + CCE + MoE tying')
print(f'GPU:          {torch.cuda.get_device_name(0)}')
print(f'Samples:      {len(examples)}')
print(f'Steps:        {num_steps}')
print(f'LoRA rank:    {LORA_RANK}  alpha: {LORA_ALPHA}')
print(f'LoRA targets: {TARGET_MODULES}')
print(f'LR:           {LEARNING_RATE} -> 0 (linear decay)')
print(f'Batch:        {BATCH_SIZE} (micro={MICRO_BATCH_SIZE})')
print(f'Max seq len:  {MAX_SEQ_LEN}')
print(f'MoE tying:    {MOE_TIE_WEIGHTS}')
print(f'Train time:   {elapsed/60:.1f} min ({elapsed/3600:.2f} hrs)')
print(f'Adapter:      {OUTPUT_DIR}')
print(f'Submission:   {ZIP_PATH}')
print('='*60)
