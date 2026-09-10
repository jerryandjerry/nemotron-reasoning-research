#!/usr/bin/env python3
"""
Plain-transformers (NO Unsloth) + PEFT trainer for the out_proj experiment.
Uses transformers 5.5.4 BUILT-IN nemotron_h (trust_remote_code=False), which exposes
`use_mem_eff_path` on each Mamba mixer. USE_MEM_EFF=0 routes training through the unfused
CUDA path (mamba_chunk_scan_combined + self.out_proj as a module) so out_proj LoRA TRAINS,
without the 126GB pure-torch OOM and without patching model code.

Everything else (data, batching, CCE, manual lm_head LoRA, manual loop, soup saves, FIFO)
mirrors the Unsloth 0.84 trainer. Built-in backbone is `causal_lm.model` (not `.backbone`),
so the CCE patch + save-time key rename are adjusted; final keys are renamed to the proven
`base_model.model.backbone.*` Kaggle format.

Env: MOE_TIE (0/1), USE_MEM_EFF (0/1), STRESS_TEST, FRESH_START.
"""

import os, sys, time, json, zipfile, gc, subprocess, types, math, logging

os.environ['TRANSFORMERS_NO_TF']     = '1'
os.environ['TRANSFORMERS_NO_FLAX']   = '1'
os.environ['CUDA_VISIBLE_DEVICES']   = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HF_HUB_OFFLINE']         = '1'
os.environ['TRANSFORMERS_OFFLINE']   = '1'
os.environ['SWANLAB_API_KEY'] = dict(l.strip().split('=', 1) for l in open('kaggle.env', encoding='utf-8-sig') if '=' in l and not l.startswith('#'))['SWANLAB_API_KEY']
os.environ['SWANLAB_PROJECT']        = '260410_Nemotron'

# ── Logging setup (single writer, no duplicates) ───────────────────
LOG_FILE = f"/root/autodl-tmp/train_log_plain_{'fixtie' if os.environ.get('MOE_TIE', '1') == '1' else 'notie'}{'' if os.environ.get('USE_MEM_EFF', '1') == '1' else '_outproj'}.txt"
_log_fh = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
_log_sh = logging.StreamHandler(sys.stdout)
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[_log_fh, _log_sh])
_logger = logging.getLogger('train')
def _log(msg):
    _logger.info(msg)

# ── Config (matches Unsloth 0.84 trainer) ──────────────────────────
LORA_RANK    = 32
LORA_ALPHA   = 32
LORA_DROPOUT = 0.0
MAX_SEQ_LEN  = 8192
BATCH_SIZE   = 32
MICRO_BATCH_SIZE = 4
LEARNING_RATE = 2e-4
NUM_STEPS    = 1000  # clamped to actual max
MOE_TIE_WEIGHTS = os.environ.get('MOE_TIE', '1') == '1'
USE_MEM_EFF_PATH = os.environ.get('USE_MEM_EFF', '1') == '1'  # 0 => unfused CUDA path => out_proj LoRA trains
SHUFFLE_DATASET = True
SHUFFLE_SEED = 42

# lm_head added manually; PEFT targets the 8 proj families (incl. 128 experts' up/down_proj)
TARGET_MODULES_PROJ = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                       'up_proj', 'down_proj', 'in_proj', 'out_proj']
TARGET_MODULES = TARGET_MODULES_PROJ + ['lm_head']

# ── Paths ───────────────────────────────────────────────────────────
MODEL_PATH = '/root/autodl-tmp/Nemotron-3-Nano-30B-A3B'
CSV_PATH   = '/root/autodl-tmp/data/260514_huikang_golden_stripped.csv'
TOKENIZER_JSON = '/root/autodl-tmp/data/tokenizer.json'
_MOE_MODE  = 'fixtie' if MOE_TIE_WEIGHTS else 'notie'
_MEM_SUFFIX = '' if USE_MEM_EFF_PATH else '_outproj'
OUTPUT_DIR = f'/root/autodl-tmp/adapter_output_plain_{_MOE_MODE}{_MEM_SUFFIX}'
ZIP_PATH   = f'/root/autodl-tmp/submission_plain_{_MOE_MODE}{_MEM_SUFFIX}.zip'

PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
SAVE_STEPS = 50

STRESS_TEST = bool(os.environ.get('STRESS_TEST'))
FRESH_START = bool(os.environ.get('FRESH_START'))

os.makedirs(OUTPUT_DIR, exist_ok=True)

from datetime import datetime, timezone, timedelta
_cdt = timezone(timedelta(hours=-5))
_ts = datetime.now(_cdt).strftime('%y%m%d_%H%M')
RUN_NAME = f'{_ts}_sft_plain_{_MOE_MODE}{_MEM_SUFFIX}_r32_a32_lr2e4_seq8192_bs32_rtx6000'

# ── mamba3 stubs (mamba_ssm 2.3.1 may import mamba3 in __init__) ─────
for _mod_name in [
    'mamba_ssm.modules.mamba3', 'mamba_ssm.ops.cute',
    'mamba_ssm.ops.cute.mamba3', 'mamba_ssm.ops.cute.mamba3.mamba3_step_fn',
]:
    _m = types.ModuleType(_mod_name); _m.__path__ = []; _m.__package__ = _mod_name
    sys.modules[_mod_name] = _m
sys.modules['mamba_ssm.modules.mamba3'].Mamba3 = None

# ── Imports ─────────────────────────────────────────────────────────
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from cut_cross_entropy import linear_cross_entropy
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import Linear as LoraLinear

# transformers 5.5.4 _can_use_grouped_mm only checks hasattr(torch,'_grouped_mm'), NOT GPU
# capability. On Blackwell (sm_120) torch._grouped_mm crashes ("compute capability = 9.0").
# Force the official differentiable for-loop fallback. The fast path casts input.to(weight.dtype)
# but the fallback doesn't, so do the cast here to avoid fp32/bf16 mm mismatch.
import transformers.integrations.moe as _tf_moe
def _patched_grouped_mm(_input, _weight, offs):
    return torch.ops.transformers.grouped_mm_fallback(_input.to(_weight.dtype), _weight, offs=offs)
_tf_moe._can_use_grouped_mm = lambda *a, **k: False
_tf_moe._grouped_mm = _patched_grouped_mm
print('Patched moe._grouped_mm -> dtype-safe for-loop fallback (Blackwell sm_120)')

print(f'transformers: {transformers.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

def _smi():
    try:
        return int(subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True).stdout.strip())
    except Exception:
        return 0

# ── Load base model (full bf16, built-in nemotron_h) ────────────────
print('\n=== Loading bf16 model (plain transformers, built-in nemotron_h) ===')
gc.collect(); torch.cuda.empty_cache()
t_load = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=False,
    low_cpu_mem_usage=True, attn_implementation='sdpa',
).cuda()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False)
print(f'Model loaded in {time.time()-t_load:.1f}s | class={type(model).__name__} | module={type(model).__module__}')
print(f'VRAM after load: smi={_smi()}M')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Load raw Tokenizer (huikang's, for completion encoding) ────────
from tokenizers import Tokenizer as RawTokenizer
import csv, re

raw_tokenizer = RawTokenizer.from_file(TOKENIZER_JSON)
_boxed_start_pat = re.compile(r'\\boxed\{')

def _extract_boxed(text):
    starts = list(_boxed_start_pat.finditer(text))
    out = []
    for i, m in enumerate(starts):
        seg_end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        seg = text[m.end():seg_end]
        lb = seg.rfind('}')
        out.append(seg[:lb] if lb != -1 else seg)
    return out

# ── Load and expand data ─────────────────────────────────────────────
print('\n=== Loading and tokenizing data ===')
with open(CSV_PATH, encoding='utf-8') as f:
    raw_rows = list(csv.DictReader(f))
print(f'CSV rows (unique): {len(raw_rows)}')

expanded_rows = []
for row in raw_rows:
    n_repeat = int(row.get('oversampling', 1))
    for _ in range(n_repeat):
        expanded_rows.append(row)
print(f'Expanded rows: {len(expanded_rows)}')

examples = []
for row in expanded_rows:
    prompt_text = row['prompt']
    cot = row['solver_cot']
    answer = row.get('answer', '')
    pid = row['id']
    if not cot or cot == 'nan' or len(cot.strip()) < 5:
        print(f'SKIP {pid}: empty cot')
        continue
    user_msg = prompt_text + PROMPT_SUFFIX
    _enc = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': user_msg}],
        tokenize=True, add_generation_prompt=True, enable_thinking=True,
    )
    # transformers 5.x apply_chat_template(tokenize=True) returns a BatchEncoding; older returns a list
    prompt_ids = _enc['input_ids'] if hasattr(_enc, 'keys') else _enc
    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    boxed_matches = _extract_boxed(cot)
    reasoning_answer = boxed_matches[-1] if boxed_matches else answer
    completion_text = cot + '\n</think>\n' + chr(92) + 'boxed{' + reasoning_answer + '}<|im_end|>'
    completion_ids = raw_tokenizer.encode(completion_text, add_special_tokens=False).ids
    all_ids = prompt_ids + completion_ids
    if len(all_ids) > MAX_SEQ_LEN:
        all_ids = all_ids[:MAX_SEQ_LEN]
    prompt_len = len(prompt_ids)
    mask = [0] * min(prompt_len, len(all_ids)) + [1] * max(0, len(all_ids) - prompt_len)
    if not any(mask):
        print(f'SKIP {pid}: no unmasked tokens')
        continue
    content_tail = tokenizer.decode(all_ids[-20:], skip_special_tokens=False)[-20:]
    examples.append({
        'problem_id': pid, 'category': row['category'],
        'tokens': all_ids[:-1], 'targets': all_ids[1:],
        'weights': [float(m) for m in mask[1:]], 'content_tail': content_tail,
    })

token_to_sid = {}
for e in examples:
    token_to_sid[tuple(e['tokens'][:8])] = e['problem_id']

total_unmasked = sum(sum(e['weights']) for e in examples)
total_tokens = sum(len(e['tokens']) for e in examples)
_log(f'Loaded {len(examples)} examples, {total_tokens:,} tokens (unmasked={total_unmasked:,.0f})')

from collections import Counter
cat_counts = Counter(e['category'] for e in examples)
_log('Category breakdown (expanded):')
for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
    _log(f'  {cat}: {cnt}')

if STRESS_TEST:
    print('\n=== STRESS TEST MODE ===')
    examples.sort(key=lambda e: len(e['tokens']), reverse=True)
    examples = examples[:min(BATCH_SIZE * 2, len(examples))]
    print(f'Stress test: {len(examples)} longest examples')

# ── Gradient checkpointing + input grads (needed for PEFT + grad ckpt) ──
# Built-in NemotronH blocks are GradientCheckpointingLayer, but the model doesn't set
# _supports_gradient_checkpointing=True (oversight). Flip it; fall back to manual per-block.
model._supports_gradient_checkpointing = True
if hasattr(model, 'model'):
    model.model._supports_gradient_checkpointing = True
try:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    print('gradient checkpointing enabled (standard)')
except Exception as e:
    print(f'gradient_checkpointing_enable failed ({e}) -> manual fallback')
    import torch.utils.checkpoint as _ckpt_mod
    from functools import partial as _partial
    _gc_func = _partial(_ckpt_mod.checkpoint, use_reentrant=False)
    _n_gc = 0
    for _m in model.modules():
        if _m.__class__.__name__ == 'NemotronHBlock':
            _m.gradient_checkpointing = True
            _m._gradient_checkpointing_func = _gc_func
            _n_gc += 1
    print(f'manual grad ckpt on {_n_gc} NemotronHBlock layers')
model.enable_input_require_grads()

# ── Wrap in LoRA (plain PEFT) ───────────────────────────────────────
lora_config = LoraConfig(
    r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES, bias='none', task_type='CAUSAL_LM',
)
model = get_peft_model(model, lora_config)
print(f'Applied PEFT LoRA to {TARGET_MODULES}')

# ── Set use_mem_eff_path on Mamba mixers ────────────────────────────
# USE_MEM_EFF=1 (baseline): fused kernel folds out_proj via raw weight -> out_proj LoRA dead.
# USE_MEM_EFF=0: unfused CUDA path (mamba_chunk_scan_combined + self.out_proj module call)
# -> out_proj LoRA trains. is_fast_path_available stays True (CUDA kept).
_n_mix = 0
for _mod in model.modules():
    if hasattr(_mod, 'use_mem_eff_path'):
        _mod.use_mem_eff_path = USE_MEM_EFF_PATH
        _n_mix += 1
print(f'Set use_mem_eff_path={USE_MEM_EFF_PATH} on {_n_mix} Mamba mixers'
      + ('' if USE_MEM_EFF_PATH else '  -> out_proj LoRA will train'))

# ── Manually add lm_head LoRA ───────────────────────────────────────
_causal_lm = model.base_model.model           # NemotronHForCausalLM
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
_causal_lm = model.base_model.model           # NemotronHForCausalLM
_backbone = _causal_lm.model                  # NemotronHModel (built-in: .model, not .backbone)

def _patched_causal_forward(input_ids=None, attention_mask=None, labels=None, **kwargs):
    out = _backbone(
        input_ids=input_ids, attention_mask=attention_mask,
        **{k: v for k, v in kwargs.items()
           if k in ('position_ids', 'past_key_values', 'use_cache', 'cache_position')},
    )
    hidden_states = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
    lm_head = _causal_lm.lm_head
    base_w = lm_head.base_layer.weight
    lora_A = lm_head.lora_A['default'].weight
    lora_B = lm_head.lora_B['default'].weight
    scaling = lm_head.scaling['default']
    lm_weight = base_w + scaling * lora_B @ lora_A
    if labels is not None:
        per_token_ce = linear_cross_entropy(hidden_states, lm_weight, labels, reduction='none')
        loss = per_token_ce.mean()
    else:
        per_token_ce = None; loss = None
    model._cached_per_token_ce = per_token_ce
    return loss

_causal_lm.forward = _patched_causal_forward
print('Patched NemotronHForCausalLM.forward with CCE')

# ── Training mode (REQUIRED: GradientCheckpointingLayer checkpoints only if self.training) ──
model.train()
print(f'model.train() -> training={model.training}')

# ── MoE weight tying (across the 128 experts) — only if MOE_TIE=1 ────
moe_tied_groups = []
if MOE_TIE_WEIGHTS:
    w1_proj_names = ('gate_up_proj', 'up_proj', 'gate_proj', '.w1.')
    w2_proj_names = ('down_proj', '.w2.')
    _group_map = {}
    for name, param in model.named_parameters():
        if not param.requires_grad or '.experts.' not in name or '.lora_' not in name:
            continue
        is_w1 = any(p in name for p in w1_proj_names)
        is_w2 = any(p in name for p in w2_proj_names)
        is_A = '.lora_A.' in name
        is_B = '.lora_B.' in name
        if not ((is_w1 and is_A) or (is_w2 and is_B)):
            continue
        gkey = re.sub(r'\.experts\.\d+\.', '.experts.E.', name)
        _group_map.setdefault(gkey, []).append(param)
    moe_tied_groups = [g for g in _group_map.values() if len(g) > 1]

    def _tie_param_init():
        with torch.no_grad():
            for group in moe_tied_groups:
                mean = torch.stack([p.data for p in group], dim=0).mean(dim=0)
                for p in group:
                    p.data.copy_(mean)

    def _tie_grads():
        with torch.no_grad():
            for group in moe_tied_groups:
                if any(p.grad is None for p in group):
                    continue
                gsum = torch.stack([p.grad for p in group], dim=0).sum(dim=0)
                for p in group:
                    p.grad.copy_(gsum)

    _n_tied = sum(len(g) for g in moe_tied_groups)
    print(f'MoE weight tying (across experts): {len(moe_tied_groups)} groups, {_n_tied} params')
    _tie_param_init()
else:
    def _tie_grads():
        pass

# ── Build batches ─────────────────────────────────────────────────────
import random as _random
gc.collect(); torch.cuda.empty_cache()
device = next(model.parameters()).device
optimizer = None

if STRESS_TEST:
    batches = None
    max_steps = len(examples) // BATCH_SIZE
    num_steps = min(3, max_steps)
else:
    from collections import defaultdict
    cat_indices = defaultdict(list)
    for i, e in enumerate(examples):
        cat_indices[e['category']].append(i)
    rng = _random.Random(SHUFFLE_SEED)
    interleaved = []
    for cat, indices in cat_indices.items():
        rng.shuffle(indices)
        n = len(indices)
        for rank, idx in enumerate(indices):
            interleaved.append((rank / n + rng.random() * 1e-6, idx))
    interleaved.sort(key=lambda x: x[0])
    all_indices = [idx for _, idx in interleaved]
    _log(f'Stratified interleave: {len(all_indices)} samples, seed={SHUFFLE_SEED}')
    batches = [all_indices[i:i + BATCH_SIZE] for i in range(0, len(all_indices), BATCH_SIZE)]
    num_steps = len(batches)

_log(f'Batching: stratified, {num_steps} steps')

ADAPTER_SAVE_STEPS = sorted({
    *range(SAVE_STEPS, num_steps, SAVE_STEPS),
    round(num_steps * 0.80), round(num_steps * 0.85),
    round(num_steps * 0.90), round(num_steps * 0.95), round(num_steps * 1.00),
})
ADAPTER_SAVE_STEPS = [s for s in ADAPTER_SAVE_STEPS if s > 0]
_log(f'Adapter-only soup save steps: {ADAPTER_SAVE_STEPS}')

import shutil
_ckpt_queue = []
def _fifo_cleanup():
    while len(_ckpt_queue) > 1:
        old = _ckpt_queue.pop(0)
        if os.path.isdir(old):
            shutil.rmtree(old)
            _log(f'  FIFO: deleted old checkpoint {os.path.basename(old)}')

# ── SwanLab init ────────────────────────────────────────────────────
try:
    import swanlab
    swanlab.init(project='260410_Nemotron', experiment_name=RUN_NAME, config={
        'lora_rank': LORA_RANK, 'lora_alpha': LORA_ALPHA, 'lr': LEARNING_RATE,
        'batch_size': BATCH_SIZE, 'micro_batch': MICRO_BATCH_SIZE, 'max_seq_len': MAX_SEQ_LEN,
        'num_examples': len(examples), 'num_steps': num_steps,
        'moe_tie_weights': MOE_TIE_WEIGHTS, 'use_mem_eff_path': USE_MEM_EFF_PATH,
        'stack': 'plain_transformers_5.5.4', 'targets': TARGET_MODULES,
        'soup_save_steps': ADAPTER_SAVE_STEPS,
    })
    _swanlab_ok = True
    _log(f'SwanLab init: {RUN_NAME}')
except Exception as e:
    _swanlab_ok = False
    _log(f'SwanLab init failed: {e}')

_log('\n=== Starting training ===')
_log(f'Steps: {num_steps}, batch_size={BATCH_SIZE}, micro_batch={MICRO_BATCH_SIZE}, lr={LEARNING_RATE}')
_log(f'VRAM before training: smi={_smi()}M')

# ── Resume from latest checkpoint (skip if FRESH_START=1) ────────────
_resume_step = 0
_resume_optimizer_state = None
from safetensors.torch import load_file as _load_file
if not FRESH_START:
    _ckpt_dirs = sorted(
        [d for d in os.listdir(OUTPUT_DIR)
         if d.startswith('checkpoint-') and os.path.isdir(os.path.join(OUTPUT_DIR, d))],
        key=lambda x: int(x.split('-')[1].split('_')[0]))
    if _ckpt_dirs:
        _latest = _ckpt_dirs[-1]
        _latest_dir = os.path.join(OUTPUT_DIR, _latest)
        _resume_step = int(_latest.split('-')[1].split('_')[0])
        _log(f'Resuming from checkpoint: {_latest} (step {_resume_step})')
        _adapter_state = _load_file(os.path.join(_latest_dir, 'adapter_model.safetensors'))
        _renamed_state = {}
        for _k, _v in _adapter_state.items():
            _nk = _k
            for _sfx in ('lora_A.weight', 'lora_B.weight', 'lora_embedding_A.weight', 'lora_embedding_B.weight'):
                if _nk.endswith('.' + _sfx):
                    _nk = _nk[:-len(_sfx)] + _sfx.replace('.weight', '.default.weight')
                    break
            _renamed_state[_nk] = _v
        _missing, _unexpected = model.load_state_dict(_renamed_state, strict=False)
        _log(f'  Loaded adapter weights: {len(_renamed_state)} tensors (unexpected={len(_unexpected)})')
        _ts_path = os.path.join(_latest_dir, 'training_state.pt')
        if os.path.exists(_ts_path):
            _tsd = torch.load(_ts_path, map_location='cpu', weights_only=False)
            _resume_optimizer_state = _tsd['optimizer_state_dict']
            _log('  Loaded optimizer state')
    else:
        _log('No checkpoint to resume from, starting fresh')
else:
    _log('FRESH_START=1, starting from scratch')

t0 = time.time()
step = _resume_step
for step_idx in range(_resume_step, num_steps):
    gc.collect(); torch.cuda.empty_cache()
    if batches:
        batch_indices = batches[step_idx]
    else:
        batch_indices = list(range(step_idx * BATCH_SIZE, min((step_idx + 1) * BATCH_SIZE, len(examples))))
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
        mb_tails = [batch[i]['content_tail'] for i in range(mb_start, mb_end)]
        n_micro = len(mb_toks)
        max_len = max(len(t) for t in mb_toks)

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
        for b in range(n_micro):
            _log(f'>> step={step+1} batch[{mb_start+b}] sample_id={mb_pids[b]} padded_len={max_len} smi={smi_before}M content_tail: {mb_tails[b]}')

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

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
        if _resume_optimizer_state is not None:
            optimizer.load_state_dict(_resume_optimizer_state)
            _log(f'  Optimizer state loaded, resuming at step {_resume_step + 1}')
            _resume_optimizer_state = None
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
    _log(f'   step={step} smi_after={smi_after}M loss={loss_mean:.6f} grad_norm={grad_norm:.4f} lr={lr:.2e} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m')

    if _swanlab_ok:
        try:
            swanlab.log({'loss': loss_mean, 'grad_norm': float(grad_norm), 'lr': lr, 'vram_mb': smi_after}, step=step)
        except Exception:
            pass

    if step % SAVE_STEPS == 0 or step == num_steps:
        ckpt_name = f'checkpoint-{step}_loss{loss_mean:.4f}_lr{lr:.2e}'
        ckpt_dir = os.path.join(OUTPUT_DIR, ckpt_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        torch.save({
            'optimizer_state_dict': optimizer.state_dict(), 'step': step, 'lr': lr, 'loss': loss_mean,
            'rng_state': torch.cuda.get_rng_state(), 'python_rng_state': _random.getstate(),
        }, os.path.join(ckpt_dir, 'training_state.pt'))
        _log(f'  Checkpoint saved: {ckpt_name}')
        _ckpt_queue.append(ckpt_dir)
        _fifo_cleanup()

    if step in ADAPTER_SAVE_STEPS:
        adapter_name = f'adapter-{step}_loss{loss_mean:.4f}_lr{lr:.2e}'
        adapter_dir = os.path.join(OUTPUT_DIR, adapter_name)
        os.makedirs(adapter_dir, exist_ok=True)
        model.save_pretrained(adapter_dir)
        _log(f'  Adapter-only saved (soup): {adapter_name}')

    gc.collect(); torch.cuda.empty_cache()

elapsed = time.time() - t0
_log(f'\nTraining done. Time: {elapsed/3600:.2f} hrs ({elapsed/60:.1f} min)')
_log(f'Peak VRAM: smi={_smi()}M')

if STRESS_TEST:
    _log('STRESS TEST COMPLETE')
    if _swanlab_ok:
        try: swanlab.finish()
        except Exception: pass
    sys.exit(0)

# ── Save final adapter + rename keys to proven backbone format ──────
_log('\n=== Saving adapter ===')
from safetensors.torch import load_file, save_file
model.save_pretrained(OUTPUT_DIR)
st_path = os.path.join(OUTPUT_DIR, 'adapter_model.safetensors')
tensors = load_file(st_path)
renamed = {}
for k, v in tensors.items():
    nk = k.replace('base_model.model.model.', 'base_model.model.backbone.')
    nk = nk.replace('base_model.model.lm_head.', 'base_model.model.backbone.lm_head.')
    renamed[nk] = v
save_file(renamed, st_path)
_log(f'Saved + renamed keys to backbone format ({len(renamed)} tensors)')

config_path = os.path.join(OUTPUT_DIR, 'adapter_config.json')
with open(config_path) as f:
    adapter_config = json.load(f)
adapter_config['base_model_name_or_path'] = 'metric/nemotron-3-nano-30b-a3b-bf16'
adapter_config['inference_mode'] = True
with open(config_path, 'w') as f:
    json.dump(adapter_config, f, indent=2)

# ── Create submission zip ───────────────────────────────────────────
_log('\n=== Creating submission zip ===')
REQUIRED = {'adapter_config.json', 'adapter_model.safetensors'}
with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(OUTPUT_DIR):
        if fname in REQUIRED:
            zf.write(os.path.join(OUTPUT_DIR, fname), arcname=fname)
with zipfile.ZipFile(ZIP_PATH) as zf:
    _log(f'ZIP contents: {zf.namelist()}')
_log(f'ZIP size: {os.path.getsize(ZIP_PATH)/1024/1024:.1f} MB')

# ── Zip final checkpoint ─────────────────────────────────────────────
_log('\n=== Zipping final checkpoint ===')
ckpt_dirs = sorted([d for d in os.listdir(OUTPUT_DIR) if d.startswith('checkpoint-')],
                   key=lambda x: int(x.split('-')[1].split('_')[0]))
if ckpt_dirs:
    final_ckpt = ckpt_dirs[-1]
    ckpt_zip_path = os.path.join(os.path.dirname(OUTPUT_DIR), f'{final_ckpt}_plain.zip')
    ckpt_full = os.path.join(OUTPUT_DIR, final_ckpt)
    with zipfile.ZipFile(ckpt_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(ckpt_full):
            for f in files:
                fp = os.path.join(root, f)
                zf.write(fp, arcname=os.path.relpath(fp, ckpt_full))
    _log(f'Checkpoint zip: {ckpt_zip_path} ({os.path.getsize(ckpt_zip_path)/1024/1024:.1f} MB)')

_log('\n' + '=' * 60)
_log(f'PLAIN-TRANSFORMERS RUN — moe_mode={_MOE_MODE} use_mem_eff_path={USE_MEM_EFF_PATH}')
_log(f'Steps: {num_steps} | Submission: {ZIP_PATH}')
if _swanlab_ok:
    try: swanlab.finish()
    except Exception: pass
