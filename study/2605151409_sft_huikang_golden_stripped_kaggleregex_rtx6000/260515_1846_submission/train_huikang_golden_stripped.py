#!/usr/bin/env python3
"""
Huikang 0.84 config + lkevincc golden cryptarithm data + KAGGLE-COMPATIBLE
\\boxed extraction + tick-stripped lkevincc CoTs.

Changes vs the 0.68 run (2605141458_sft_huikang_golden_rtx6000):
1. Kaggle-metric extractor for the post-</think> boxed line (handles answers
   containing `}` correctly).
2. Data: 260514_huikang_golden_stripped.csv — same as the golden CSV but with
   the 2,537 ✓ characters stripped from 704 lkevincc rows. Digit mapping,
   Reading order, digits: verification lines, and operator names are all
   PRESERVED (reasoning logic untouched).
3. id=45076dc9 row fixed (`\\boxed{6}}` → `\\boxed{6}`).

Same training setup as 0.84 run: bf16 + CCE + MoE tying, r32 a32,
lr 2e-4 linear decay, AdamW beta2=0.95, bs=32 micro=4,
9 targets + lm_head, seq=8192, stratified batching.
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
LOG_FILE = '/root/autodl-tmp/train_log.txt'
_log_fh = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
_log_sh = logging.StreamHandler(sys.stdout)
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[_log_fh, _log_sh])
_logger = logging.getLogger('train')
def _log(msg):
    _logger.info(msg)

# ── Config (exact huikang reproduction + golden cryptarithm) ───────
LORA_RANK    = 32
LORA_ALPHA   = 32
LORA_DROPOUT = 0.0
MAX_SEQ_LEN  = 8192
BATCH_SIZE   = 32
MICRO_BATCH_SIZE = 4
LEARNING_RATE = 2e-4
NUM_STEPS    = 1000  # will be clamped to actual max
MOE_TIE_WEIGHTS = True
SHUFFLE_DATASET = True
SHUFFLE_SEED = 42
RESET_WEIGHTS = True

TARGET_MODULES = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'up_proj', 'down_proj', 'in_proj', 'out_proj',
    'lm_head',
]

# ── Paths ───────────────────────────────────────────────────────────
MODEL_PATH = '/root/autodl-tmp/Nemotron-3-Nano-30B-A3B'
CSV_PATH   = '/root/autodl-tmp/data/260514_huikang_golden_stripped.csv'
TOKENIZER_JSON = '/root/autodl-tmp/data/tokenizer.json'  # huikang's raw tokenizer
OUTPUT_DIR = '/root/autodl-tmp/adapter_output_huikang_golden_stripped_kaggleregex'
ZIP_PATH   = '/root/autodl-tmp/submission.zip'

PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
SAVE_STEPS = 50  # ~245 steps / 5 ≈ 49, rounded to 50

STRESS_TEST = bool(os.environ.get('STRESS_TEST'))
FRESH_START = bool(os.environ.get('FRESH_START'))

os.makedirs(OUTPUT_DIR, exist_ok=True)

from datetime import datetime, timezone, timedelta
_cdt = timezone(timedelta(hours=-5))
_ts = datetime.now(_cdt).strftime('%y%m%d_%H%M')
RUN_NAME = f'{_ts}_sft_huikang_golden_stripped_kaggleregex_r32_a32_lr2e4_seq8192_bs32_rtx6000'

# ── Imports ─────────────────────────────────────────────────────────
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

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Load raw Tokenizer (huikang's, for completion encoding) ────────
from tokenizers import Tokenizer as RawTokenizer
import csv, re

raw_tokenizer = RawTokenizer.from_file(TOKENIZER_JSON)
# Kaggle metric's \boxed{} extractor: for each \boxed{, take everything up to
# the LAST } before the next \boxed{ or end of text. Correctly handles answers
# containing } (e.g. lkevincc cryptarithm answers like '+}', '%}|').
# https://www.kaggle.com/code/metric/nvidia-nemotron-metric
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

# Expand by oversampling column
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

    # ── Prompt tokenization (matches huikang's tokenize_prompt) ──
    user_msg = prompt_text + PROMPT_SUFFIX
    prompt_ids = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': user_msg}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    # ── Completion tokenization (raw Tokenizer, matches huikang exactly) ──
    boxed_matches = _extract_boxed(cot)
    reasoning_answer = boxed_matches[-1] if boxed_matches else answer

    # chr(92) = backslash, avoids Python \b escape
    completion_text = cot + '\n</think>\n' + chr(92) + 'boxed{' + reasoning_answer + '}<|im_end|>'
    completion_ids = raw_tokenizer.encode(completion_text, add_special_tokens=False).ids

    all_ids = prompt_ids + completion_ids
    if len(all_ids) > MAX_SEQ_LEN:
        all_ids = all_ids[:MAX_SEQ_LEN]

    # Mask: 0 for prompt, 1 for completion
    prompt_len = len(prompt_ids)
    mask = [0] * min(prompt_len, len(all_ids)) + [1] * max(0, len(all_ids) - prompt_len)

    if not any(mask):
        print(f'SKIP {pid}: no unmasked tokens')
        continue

    # Decode last 20 chars for content_tail logging
    content_tail = tokenizer.decode(all_ids[-20:], skip_special_tokens=False)[-20:]

    examples.append({
        'problem_id': pid,
        'category': row['category'],
        'tokens': all_ids[:-1],
        'targets': all_ids[1:],
        'weights': [float(m) for m in mask[1:]],
        'content_tail': content_tail,
    })

# Build token_to_sid lookup (first few tokens -> sample ID)
token_to_sid = {}
for e in examples:
    key = tuple(e['tokens'][:8])
    token_to_sid[key] = e['problem_id']

total_unmasked = sum(sum(e['weights']) for e in examples)
total_tokens = sum(len(e['tokens']) for e in examples)
_log(f'Loaded {len(examples)} examples, {total_tokens:,} tokens (unmasked={total_unmasked:,.0f})')
_log(f'token_to_sid: {len(token_to_sid)} entries')

# ── Category breakdown ─────────────────────────────────────────────
from collections import Counter
cat_counts = Counter(e['category'] for e in examples)
_log(f'Category breakdown (expanded):')
for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
    _log(f'  {cat}: {cnt}')

if STRESS_TEST:
    print('\n=== STRESS TEST MODE ===')
    examples.sort(key=lambda e: len(e['tokens']), reverse=True)
    K = BATCH_SIZE * 2
    examples = examples[:min(K, len(examples))]
    print(f'Stress test: {len(examples)} longest examples')

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
    # Stratified batching: interleave categories so each batch is proportional
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
            # rank/n spreads each category evenly across [0,1)
            # small jitter breaks ties between categories
            pos = rank / n + rng.random() * 1e-6
            interleaved.append((pos, idx))
    interleaved.sort(key=lambda x: x[0])
    all_indices = [idx for _, idx in interleaved]

    _log(f'Stratified interleave: {len(all_indices)} samples, seed={SHUFFLE_SEED}')
    for cat in sorted(cat_indices, key=lambda c: -len(cat_indices[c])):
        _log(f'  {cat}: {len(cat_indices[cat])}')

    batches = []
    for i in range(0, len(all_indices), BATCH_SIZE):
        batches.append(all_indices[i:min(i + BATCH_SIZE, len(all_indices))])
    num_steps = len(batches)

_log(f'Batching: stratified, {num_steps} steps')

# ── FIFO checkpoint queue — keep only 2 ────────────────────────────
import shutil

_ckpt_queue = []

def _fifo_cleanup():
    while len(_ckpt_queue) > 2:
        old = _ckpt_queue.pop(0)
        if os.path.isdir(old):
            shutil.rmtree(old)
            _log(f'  FIFO: deleted old checkpoint {os.path.basename(old)}')

# ── SwanLab init ────────────────────────────────────────────────────
try:
    import swanlab
    swanlab.init(
        project='260410_Nemotron',
        experiment_name=RUN_NAME,
        config={
            'lora_rank': LORA_RANK, 'lora_alpha': LORA_ALPHA,
            'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE,
            'micro_batch': MICRO_BATCH_SIZE, 'max_seq_len': MAX_SEQ_LEN,
            'num_examples': len(examples), 'num_steps': num_steps,
            'moe_tie_weights': MOE_TIE_WEIGHTS,
            'targets': TARGET_MODULES,
            'data': 'huikang_0408 + lkevincc_golden_cryptarithm (45076dc9 fixed, ✓ stripped)',
            'extractor': 'kaggle_metric_rfind',
        },
    )
    _swanlab_ok = True
    _log(f'SwanLab init: {RUN_NAME}')
except Exception as e:
    _swanlab_ok = False
    _log(f'SwanLab init failed: {e}')

_log(f'\n=== Starting training ===')
_log(f'Steps: {num_steps}, batch_size={BATCH_SIZE}, micro_batch={MICRO_BATCH_SIZE}, lr={LEARNING_RATE}')
_log(f'VRAM before training: smi={_smi()}M')

t0 = time.time()
step = 0
for step_idx in range(num_steps):
    gc.collect(); torch.cuda.empty_cache()

    if batches:
        batch_indices = batches[step_idx]
    else:
        # Stress test: sequential
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
    _log(f'   step={step} smi_after={smi_after}M loss={loss_mean:.6f} grad_norm={grad_norm:.4f} lr={lr:.2e} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m')

    if _swanlab_ok:
        try:
            swanlab.log({'loss': loss_mean, 'grad_norm': float(grad_norm), 'lr': lr, 'vram_mb': smi_after}, step=step)
        except Exception:
            pass

    # Checkpoint saving (full: adapter + optimizer + scheduler + rng + tokenizer + args)
    if step % SAVE_STEPS == 0 or step == num_steps:
        ckpt_name = f'checkpoint-{step}_loss{loss_mean:.4f}_lr{lr:.2e}'
        ckpt_dir = os.path.join(OUTPUT_DIR, ckpt_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        torch.save({
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step,
            'lr': lr,
            'loss': loss_mean,
            'rng_state': torch.cuda.get_rng_state(),
            'python_rng_state': _random.getstate(),
            'training_args': {
                'lora_rank': LORA_RANK, 'lora_alpha': LORA_ALPHA,
                'learning_rate': LEARNING_RATE, 'batch_size': BATCH_SIZE,
                'micro_batch_size': MICRO_BATCH_SIZE, 'max_seq_len': MAX_SEQ_LEN,
                'num_steps': num_steps, 'num_examples': len(examples),
                'moe_tie_weights': MOE_TIE_WEIGHTS,
            },
        }, os.path.join(ckpt_dir, 'training_state.pt'))
        _log(f'  Checkpoint saved: {ckpt_name}')
        _ckpt_queue.append(ckpt_dir)
        _fifo_cleanup()

    gc.collect(); torch.cuda.empty_cache()

elapsed = time.time() - t0
_log(f'\nTraining done. Time: {elapsed/3600:.2f} hrs ({elapsed/60:.1f} min)')
_log(f'Peak VRAM: smi={_smi()}M')

if STRESS_TEST:
    _log('STRESS TEST COMPLETE')
    if _swanlab_ok:
        try: swanlab.finish()
        except: pass
    sys.exit(0)

# ── Save adapter + rename lm_head keys ──────────────────────────────
_log('\n=== Saving adapter ===')
from safetensors.torch import load_file, save_file

model.save_pretrained(OUTPUT_DIR)
st_path = os.path.join(OUTPUT_DIR, 'adapter_model.safetensors')
tensors = load_file(st_path)
renamed = {
    k.replace('base_model.model.lm_head.', 'base_model.model.backbone.lm_head.'): v
    for k, v in tensors.items()
}
save_file(renamed, st_path)
_log('Saved and renamed lm_head keys')

# Fix adapter_config.json
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
    contents = zf.namelist()
_log(f'ZIP contents: {contents}')
_log(f'ZIP size: {os.path.getsize(ZIP_PATH)/1024/1024:.1f} MB')

# ── Zip final checkpoint ─────────────────────────────────────────────
_log('\n=== Zipping final checkpoint ===')
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
    _log(f'Checkpoint zip: {ckpt_zip_path} ({os.path.getsize(ckpt_zip_path)/1024/1024:.1f} MB)')
else:
    _log('WARNING: no checkpoint dirs found')

# ── Summary ─────────────────────────────────────────────────────────
_log('\n' + '='*60)
_log('TRAINING SUMMARY — huikang golden_stripped + Kaggle regex + 45076dc9 fixed')
_log('='*60)
_log(f'Method:       LoRA bf16 + CCE + MoE tying')
_log(f'GPU:          {torch.cuda.get_device_name(0)}')
_log(f'Samples:      {len(examples)}')
_log(f'Steps:        {num_steps}')
_log(f'LoRA rank:    {LORA_RANK}  alpha: {LORA_ALPHA}')
_log(f'LoRA targets: {TARGET_MODULES}')
_log(f'LR:           {LEARNING_RATE} -> 0 (linear decay)')
_log(f'Batch:        {BATCH_SIZE} (micro={MICRO_BATCH_SIZE})')
_log(f'Max seq len:  {MAX_SEQ_LEN}')
_log(f'MoE tying:    {MOE_TIE_WEIGHTS}')
_log(f'Shuffle:      {SHUFFLE_DATASET} (seed={SHUFFLE_SEED})')
_log(f'Train time:   {elapsed/60:.1f} min ({elapsed/3600:.2f} hrs)')
_log(f'Adapter:      {OUTPUT_DIR}')
_log(f'Submission:   {ZIP_PATH}')
_log('='*60)
if _swanlab_ok:
    try: swanlab.finish()
    except: pass
