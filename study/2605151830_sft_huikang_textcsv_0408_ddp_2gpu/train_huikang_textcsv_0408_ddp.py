#!/usr/bin/env python3
"""
Huikang 0.85 config — DDP 2-GPU version of train_huikang_textcsv_0408.py

Effective batch stays 32 (same as single-GPU). Each rank processes 16 samples
per step (micro=4, accum=4). Time should ~halve from 3.95 hrs to ~2 hrs.

Launch: torchrun --nproc_per_node=2 train_huikang_textcsv_0408_ddp.py

Loss normalization for DDP:
- Each rank computes local CE sum (no division)
- All-reduce to get global completion-token count across both ranks
- loss = local_ce_sum / global_n * world_size
- (*world_size cancels DDP's automatic /world_size gradient averaging)
- Result: gradient identical to single-GPU on the same 32-sample batch
"""

import os, sys, time, json, zipfile, gc, subprocess, types, math, logging

os.environ['TRANSFORMERS_NO_TF']     = '1'
os.environ['TRANSFORMERS_NO_FLAX']   = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HF_HUB_OFFLINE']         = '1'
os.environ['TRANSFORMERS_OFFLINE']   = '1'
os.environ['SWANLAB_API_KEY'] = dict(l.strip().split('=', 1) for l in open('kaggle.env', encoding='utf-8-sig') if '=' in l and not l.startswith('#'))['SWANLAB_API_KEY']
os.environ['SWANLAB_PROJECT']        = '260410_Nemotron'

# ── DDP setup ──────────────────────────────────────────────────────
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Bind each rank to its own GPU BEFORE NCCL init — otherwise NCCL may
# fail with "no GPUs found" if both workers race to grab device 0.
LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
torch.cuda.set_device(LOCAL_RANK)
dist.init_process_group(backend='nccl')
WORLD_SIZE = dist.get_world_size()
RANK = dist.get_rank()
IS_MAIN = (RANK == 0)

# Null context manager for conditional DDP no_sync
class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *a): return False

# ── Logging (main rank only writes to file; all ranks print to stdout) ──
LOG_FILE = '/root/autodl-tmp/train_log.txt'
if IS_MAIN:
    _log_fh = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
    _log_sh = logging.StreamHandler(sys.stdout)
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[_log_fh, _log_sh])
else:
    logging.basicConfig(level=logging.WARNING, format=f'[r{RANK}] %(message)s')
_logger = logging.getLogger('train')
def _log(msg):
    if IS_MAIN:
        _logger.info(msg)

# ── Config (matches single-GPU recipe) ──────────────────────────────
LORA_RANK    = 32
LORA_ALPHA   = 32
LORA_DROPOUT = 0.0
MAX_SEQ_LEN  = 8192
GLOBAL_BATCH_SIZE = 32          # effective batch across both ranks
MICRO_BATCH_SIZE = 4
LEARNING_RATE = 2e-4
MOE_TIE_WEIGHTS = True

# Each rank handles half the batch: 16 samples = 4 accumulation steps of 4
PER_RANK_BATCH_SIZE = GLOBAL_BATCH_SIZE // WORLD_SIZE  # 16

TARGET_MODULES = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'up_proj', 'down_proj', 'in_proj', 'out_proj',
    'lm_head',
]

# ── Paths ───────────────────────────────────────────────────────────
MODEL_PATH = '/root/autodl-tmp/Nemotron-3-Nano-30B-A3B'
CSV_PATH   = '/root/autodl-tmp/data/260514_huikang_update/huikang_7830.csv'
TOKENIZER_JSON = '/root/autodl-tmp/data/tokenizer.json'
OUTPUT_DIR = '/root/autodl-tmp/adapter_output_huikang_textcsv_0408_ddp'
ZIP_PATH   = '/root/autodl-tmp/submission.zip'

PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
SAVE_STEPS = 50

STRESS_TEST = bool(os.environ.get('STRESS_TEST'))
FRESH_START = bool(os.environ.get('FRESH_START'))
STRESS_MAX_STEPS = 3

if IS_MAIN:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
dist.barrier()

from datetime import datetime, timezone, timedelta
_cdt = timezone(timedelta(hours=-5))
_ts = datetime.now(_cdt).strftime('%y%m%d_%H%M')
RUN_NAME = f'{_ts}_sft_huikang_textcsv0408_ddp2gpu_r32_a32_lr2e4_seq8192_bs32_rtx6000'

# ── Imports ─────────────────────────────────────────────────────────
import unsloth
from unsloth import FastLanguageModel
if IS_MAIN:
    print(f'unsloth: {unsloth.__version__}')

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
from cut_cross_entropy import linear_cross_entropy
from peft import LoraConfig
from peft.tuners.lora import Linear as LoraLinear

if IS_MAIN:
    print(f'mamba_ssm: {mamba_ssm.__version__}')
    print(f'PyTorch: {torch.__version__}')
    print(f'GPU: {torch.cuda.get_device_name(LOCAL_RANK)}')
    print(f'World size: {WORLD_SIZE}, this rank: {RANK} (local {LOCAL_RANK})')
    print(f'Global batch: {GLOBAL_BATCH_SIZE}, per-rank batch: {PER_RANK_BATCH_SIZE}')

# ── Load base model ─────────────────────────────────────────────────
if IS_MAIN:
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
    device_map={'': LOCAL_RANK},  # Each rank loads on its own GPU
)
if IS_MAIN:
    print(f'Model loaded in {time.time() - t_load:.1f}s')

def _smi():
    try:
        return int(subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits',
             f'--id={LOCAL_RANK}'],
            capture_output=True, text=True).stdout.strip())
    except:
        return 0

if IS_MAIN:
    print(f'VRAM after load: smi={_smi()}M')

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Load raw Tokenizer ─────────────────────────────────────────────
from tokenizers import Tokenizer as RawTokenizer
import csv, re

raw_tokenizer = RawTokenizer.from_file(TOKENIZER_JSON)
# Kaggle metric's \boxed{} extractor: for each \boxed{, take everything up to
# the LAST } before the next \boxed{ or end of text. Correctly handles answers
# containing } (e.g. cryptarithm answers like '+}', '%}|').
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

# ── Load and tokenize data (all ranks do this independently) ────────
PRETOK_DIR = '/root/autodl-tmp/data/huikang_corpus/04-08-16-14/tokens'

if IS_MAIN:
    print('\n=== Loading and tokenizing data ===')

with open(CSV_PATH, encoding='utf-8') as f:
    raw_rows = list(csv.DictReader(f))
if IS_MAIN:
    print(f'CSV rows: {len(raw_rows)}')

examples = []
for row in raw_rows:
    prompt_text = row['prompt']
    cot = row['solver_cot']
    answer = row.get('answer', '')
    pid = row['id']

    if not cot or cot == 'nan' or len(cot.strip()) < 5:
        if IS_MAIN:
            print(f'SKIP {pid}: empty cot')
        continue

    user_msg = prompt_text + PROMPT_SUFFIX
    prompt_ids = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': user_msg}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )

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
        continue

    content_tail = tokenizer.decode(all_ids[-20:], skip_special_tokens=False)[-20:]

    examples.append({
        'problem_id': pid,
        'category': row['category'],
        'tokens': all_ids[:-1],
        'targets': all_ids[1:],
        'weights': [float(m) for m in mask[1:]],
        'content_tail': content_tail,
    })

token_to_sid = {}
for e in examples:
    key = tuple(e['tokens'][:8])
    token_to_sid[key] = e['problem_id']

total_unmasked = sum(sum(e['weights']) for e in examples)
total_tokens = sum(len(e['tokens']) for e in examples)
_log(f'Loaded {len(examples)} examples, {total_tokens:,} tokens (unmasked={total_unmasked:,.0f})')
_log(f'token_to_sid: {len(token_to_sid)} entries')

# ── Token verification (main rank only, warn-only) ─────────────────
# Note: pretokenized files were produced with the OLD buggy [^}]* regex;
# the new Kaggle-style extractor will differ on answers containing '}'.
# This check is informational; mismatches do NOT fail training.
_suf_pat_v = re.compile(r'-(?:d|p)\d+$')
if IS_MAIN:
    _pid_map = {e['problem_id']: e for e in examples}
    _verify_ids = ['00066667', '00189f6a', '001c63cb', '0040ff76', '0133bcec-d0']
    _verify_ok = True
    for _vid in _verify_ids:
        _base_vid = _suf_pat_v.sub('', _vid)
        _pretok_path = os.path.join(PRETOK_DIR, _vid, 'synthetic.json')
        if not os.path.exists(_pretok_path) or _base_vid not in _pid_map:
            continue
        with open(_pretok_path) as _f:
            _rec = json.load(_f)
        _pretok_all = _rec['tokens']
        _ours = _pid_map[_base_vid]['tokens']
        _expected = _pretok_all[:-1]
        if _ours == _expected:
            _log(f'Token verify {_vid} (base={_base_vid}): MATCH ({len(_ours)} tokens)')
        else:
            _log(f'Token verify {_vid} (base={_base_vid}): MISMATCH '
                 f'(ours={len(_ours)} pretok={len(_expected)}) — expected if answer contains }}')
            _verify_ok = False
    if _verify_ok:
        _log('Token verification PASSED')
    else:
        _log('Token verification: some mismatches (warn-only; regex fix changed tokenization)')

if STRESS_TEST:
    if IS_MAIN:
        print('\n=== STRESS TEST MODE ===')
    examples.sort(key=lambda e: len(e['tokens']), reverse=True)
    # Routine: K >= bs * max_steps * 2 = 32 * 3 * 2 = 192 (longest samples only)
    K = GLOBAL_BATCH_SIZE * STRESS_MAX_STEPS * 2
    examples = examples[:min(K, len(examples))]
    if IS_MAIN:
        print(f'Stress test: {len(examples)} longest examples (K={K})')

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
    if IS_MAIN:
        print(f'is_fast_path_available was: {nemotron_mod.is_fast_path_available}')
    nemotron_mod.is_fast_path_available = True
    if IS_MAIN:
        print('Patched is_fast_path_available = True')

# ── Manually add lm_head LoRA (BEFORE DDP wrap) ──────────────────────
_causal_lm = model
while hasattr(_causal_lm, 'model'):
    _causal_lm = _causal_lm.model
_lm_head = _causal_lm.lm_head
if not isinstance(_lm_head, LoraLinear):
    _cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)
    model.base_model._create_and_replace(
        _cfg, 'default', target=_lm_head, target_name='lm_head', parent=_causal_lm,
    )
    if IS_MAIN:
        print('Manually added LoRA to lm_head')
else:
    if IS_MAIN:
        print('lm_head already has LoRA')

# ── Cast LoRA params to fp32 ────────────────────────────────────────
for name, param in model.named_parameters():
    if '.lora_' in name:
        param.data = param.data.to(torch.float32)
if IS_MAIN:
    print('Cast LoRA params to fp32')

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
if IS_MAIN:
    print(f'Model: {trainable:,} trainable / {total:,} total')

# ── Patch forward with CCE (BEFORE DDP wrap) ────────────────────────
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
    _base._cached_per_token_ce = per_token_ce  # stash on _base (survives DDP wrap)
    return loss

_base.forward = _patched_causal_forward
if IS_MAIN:
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

    if IS_MAIN:
        print(f'MoE weight tying: {len(moe_tied_params)} params')
    _tie_param_init()
else:
    def _tie_grads():
        pass

# ── Wrap in DDP (after ALL param modifications) ─────────────────────
# static_graph=True is REQUIRED because Unsloth's gradient checkpointing
# does reentrant backward — without static_graph, DDP fires the reducer
# hook twice for the same param ("marked as ready twice" error).
# It is incompatible with find_unused_parameters; rely on every routed
# expert + shared_experts + lm_head being hit each iter (bs*seq*top6 routes).
model = DDP(
    model,
    device_ids=[LOCAL_RANK],
    output_device=LOCAL_RANK,
    static_graph=True,
    gradient_as_bucket_view=True,
    broadcast_buffers=False,
)
if IS_MAIN:
    print('Wrapped model in DDP (static_graph=True)')

# ── Build training order (from huikang's index.jsonl) ───────────────
# The new CSV stores BASE pids + oversampling count. index.jsonl stores
# 7,830 SUFFIXED pids (e.g. '524cb5c6-d3', 'a3055572-p0') — one entry per
# training-step occurrence. We strip the suffix and look up the base row,
# reusing the same example for each suffixed occurrence (=oversampling).
import random as _random
TRAIN_ORDER_PATH = '/root/autodl-tmp/data/huikang_corpus/04-08-16-14/logprobs/index.jsonl'
_pid_to_idx = {e['problem_id']: i for i, e in enumerate(examples)}
_suf_pat = re.compile(r'-(?:d|p)\d+$')

if STRESS_TEST:
    batches = None
    num_steps = min(3, len(examples) // GLOBAL_BATCH_SIZE)
else:
    ordered_indices = []
    miss = 0
    with open(TRAIN_ORDER_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get('epoch', 0) != 0:
                continue
            base_pid = _suf_pat.sub('', rec['problem_id'])
            if base_pid in _pid_to_idx:
                ordered_indices.append(_pid_to_idx[base_pid])
            else:
                miss += 1
    _log(f'Training order: {len(ordered_indices)} samples from index.jsonl '
         f'(expect 7830, miss={miss}, unique_examples={len(set(ordered_indices))})')
    batches = []
    for i in range(0, len(ordered_indices), GLOBAL_BATCH_SIZE):
        if len(ordered_indices[i:i + GLOBAL_BATCH_SIZE]) == GLOBAL_BATCH_SIZE:
            batches.append(ordered_indices[i:i + GLOBAL_BATCH_SIZE])
    num_steps = len(batches)

_log(f'Batching: {num_steps} steps, global_batch={GLOBAL_BATCH_SIZE}, per_rank={PER_RANK_BATCH_SIZE}')

# ── SwanLab (main rank only) ────────────────────────────────────────
_swanlab_ok = False
if IS_MAIN:
    try:
        import swanlab
        swanlab.init(
            project='260410_Nemotron',
            experiment_name=RUN_NAME,
            config={
                'lora_rank': LORA_RANK, 'lora_alpha': LORA_ALPHA,
                'lr': LEARNING_RATE,
                'global_batch_size': GLOBAL_BATCH_SIZE,
                'per_rank_batch_size': PER_RANK_BATCH_SIZE,
                'world_size': WORLD_SIZE,
                'micro_batch': MICRO_BATCH_SIZE,
                'max_seq_len': MAX_SEQ_LEN,
                'num_examples': len(examples), 'num_steps': num_steps,
                'moe_tie_weights': MOE_TIE_WEIGHTS,
                'targets': TARGET_MODULES,
            },
        )
        _swanlab_ok = True
        _log(f'SwanLab init: {RUN_NAME}')
    except Exception as e:
        _log(f'SwanLab init failed: {e}')

_log(f'\n=== Starting training ===')
_log(f'Steps: {num_steps}, global_batch={GLOBAL_BATCH_SIZE} (per_rank={PER_RANK_BATCH_SIZE}), '
     f'micro_batch={MICRO_BATCH_SIZE}, lr={LEARNING_RATE}')
_log(f'VRAM before training: smi={_smi()}M')

# ── Training loop ───────────────────────────────────────────────────
gc.collect(); torch.cuda.empty_cache()
device = torch.device(f'cuda:{LOCAL_RANK}')
optimizer = None

t0 = time.time()
step = 0
for step_idx in range(num_steps):
    gc.collect(); torch.cuda.empty_cache()

    if batches:
        global_batch_indices = batches[step_idx]
    else:
        global_batch_indices = list(range(step_idx * GLOBAL_BATCH_SIZE,
                                          min((step_idx + 1) * GLOBAL_BATCH_SIZE, len(examples))))

    # Split global batch across ranks: rank r gets samples [r::WORLD_SIZE]
    rank_batch_indices = global_batch_indices[RANK::WORLD_SIZE]
    rank_batch = [examples[i] for i in rank_batch_indices]

    # ── Compute GLOBAL weight (sum of completion tokens across both ranks) ──
    rank_weight = sum(sum(e['weights']) for e in rank_batch)
    rank_weight_t = torch.tensor(rank_weight, dtype=torch.float32, device=device)
    global_weight_t = rank_weight_t.clone()
    dist.all_reduce(global_weight_t, op=dist.ReduceOp.SUM)
    global_weight = global_weight_t.item()
    if global_weight <= 0:
        _log(f'WARN step {step+1}: global_weight=0, skipping')
        continue

    n_rank = len(rank_batch)
    n_accum = math.ceil(n_rank / MICRO_BATCH_SIZE)

    step_loss_sum_local = torch.zeros(1, dtype=torch.float32, device=device)

    for mb_idx, mb_start in enumerate(range(0, n_rank, MICRO_BATCH_SIZE)):
        mb_end = min(mb_start + MICRO_BATCH_SIZE, n_rank)
        mb_toks = [e['tokens'] for e in rank_batch[mb_start:mb_end]]
        mb_tgts = [e['targets'] for e in rank_batch[mb_start:mb_end]]
        mb_wts = [e['weights'] for e in rank_batch[mb_start:mb_end]]
        mb_pids = [rank_batch[i]['problem_id'] for i in range(mb_start, mb_end)]
        mb_tails = [rank_batch[i]['content_tail'] for i in range(mb_start, mb_end)]

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
            _log(f'>> step={step+1} r{RANK} mb{mb_idx} batch[{mb_start+b}] '
                 f'sample_id={mb_pids[b]} padded_len={max_len} smi={smi_before}M '
                 f'content_tail: {mb_tails[b]}')

        # ── For all but the LAST micro-batch, suppress DDP gradient sync ──
        # DDP all-reduces gradients on every backward by default. With grad
        # accumulation we only need the sync once, on the final micro-batch.
        is_last_mb = (mb_idx == n_accum - 1)
        sync_ctx = model.no_sync() if not is_last_mb else _NullContext()

        with sync_ctx:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                model(input_ids=padded_input, attention_mask=attention_mask,
                      labels=padded_targets, use_cache=False)
                per_token_ce = _base._cached_per_token_ce
                weighted_loss = per_token_ce * padded_weights
                loss_sum_t = weighted_loss.sum()
                # Global normalization, *WORLD_SIZE cancels DDP's gradient averaging
                loss = loss_sum_t / global_weight * WORLD_SIZE
            loss.backward()
            step_loss_sum_local += loss_sum_t.detach().float()

        del loss, per_token_ce, weighted_loss

    # ── Optimizer step (after all micro-batches synced) ──
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
        )
    lr = LEARNING_RATE * (1 - step / num_steps)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # MoE tying: gradients are now globally averaged (DDP all-reduce already done)
    _tie_grads()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1e9)
    optimizer.step()
    optimizer.zero_grad()

    # Global mean loss: sum CE across both ranks, divide by global token count.
    global_loss_sum_t = step_loss_sum_local.clone()
    dist.all_reduce(global_loss_sum_t, op=dist.ReduceOp.SUM)
    global_mean_loss = (global_loss_sum_t / global_weight).item()

    smi_after = _smi()
    step += 1
    elapsed = time.time() - t0
    eta = (elapsed / step) * (num_steps - step) if step > 0 else 0
    _log(f'   step={step} smi_after={smi_after}M loss={global_mean_loss:.4f} '
         f'grad_norm={grad_norm:.4f} lr={lr:.2e} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m')

    if IS_MAIN and _swanlab_ok:
        try:
            swanlab.log({'loss': global_mean_loss, 'grad_norm': float(grad_norm),
                         'lr': lr, 'vram_mb': smi_after}, step=step)
        except Exception:
            pass

    # ── Checkpoint (main rank only; FinalStepSaveCallback: force-save on last step) ──
    if IS_MAIN and (step % SAVE_STEPS == 0 or step == num_steps):
        ckpt_name = f'checkpoint-{step}_loss{global_mean_loss:.4f}_lr{lr:.2e}'
        ckpt_dir = os.path.join(OUTPUT_DIR, ckpt_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        # Save the inner (unwrapped) model
        model.module.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        torch.save({
            'optimizer_state_dict': optimizer.state_dict(),
            'step': step, 'lr': lr, 'loss': global_mean_loss,
            'rng_state': torch.cuda.get_rng_state(),
            'python_rng_state': _random.getstate(),
            'training_args': {
                'lora_rank': LORA_RANK, 'lora_alpha': LORA_ALPHA,
                'learning_rate': LEARNING_RATE,
                'global_batch_size': GLOBAL_BATCH_SIZE,
                'per_rank_batch_size': PER_RANK_BATCH_SIZE,
                'world_size': WORLD_SIZE,
                'micro_batch_size': MICRO_BATCH_SIZE,
                'max_seq_len': MAX_SEQ_LEN,
                'num_steps': num_steps, 'num_examples': len(examples),
                'moe_tie_weights': MOE_TIE_WEIGHTS,
            },
        }, os.path.join(ckpt_dir, 'training_state.pt'))
        _log(f'  Checkpoint saved: {ckpt_name}')

        # FIFO: keep only 2 checkpoints
        ckpts = sorted([d for d in os.listdir(OUTPUT_DIR) if d.startswith('checkpoint-')],
                       key=lambda x: int(x.split('-')[1].split('_')[0]))
        while len(ckpts) > 2:
            import shutil
            shutil.rmtree(os.path.join(OUTPUT_DIR, ckpts[0]))
            _log(f'  Removed old checkpoint: {ckpts[0]}')
            ckpts = ckpts[1:]

    gc.collect(); torch.cuda.empty_cache()
    dist.barrier()  # all ranks wait before next step

elapsed = time.time() - t0
_log(f'\nTraining done. Time: {elapsed/3600:.2f} hrs ({elapsed/60:.1f} min)')
_log(f'Peak VRAM: smi={_smi()}M')

if STRESS_TEST:
    _log('STRESS TEST COMPLETE')
    if IS_MAIN and _swanlab_ok:
        try: swanlab.finish()
        except: pass
    dist.destroy_process_group()
    sys.exit(0)

# ── Save adapter (main rank only) ──────────────────────────────────
if IS_MAIN:
    _log('\n=== Saving adapter ===')
    from safetensors.torch import load_file, save_file

    model.module.save_pretrained(OUTPUT_DIR)
    st_path = os.path.join(OUTPUT_DIR, 'adapter_model.safetensors')
    tensors = load_file(st_path)
    renamed = {
        k.replace('base_model.model.lm_head.', 'base_model.model.backbone.lm_head.'): v
        for k, v in tensors.items()
    }
    save_file(renamed, st_path)
    _log('Saved and renamed lm_head keys')

    config_path = os.path.join(OUTPUT_DIR, 'adapter_config.json')
    with open(config_path) as f:
        adapter_config = json.load(f)
    adapter_config['base_model_name_or_path'] = 'metric/nemotron-3-nano-30b-a3b-bf16'
    adapter_config['inference_mode'] = True
    with open(config_path, 'w') as f:
        json.dump(adapter_config, f, indent=2)

    _log('\n=== Creating submission zip ===')
    REQUIRED = {'adapter_config.json', 'adapter_model.safetensors'}
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(OUTPUT_DIR):
            if fname in REQUIRED:
                zf.write(os.path.join(OUTPUT_DIR, fname), arcname=fname)
    _log(f'ZIP size: {os.path.getsize(ZIP_PATH)/1024/1024:.1f} MB')

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

    _log('\n' + '='*60)
    _log('TRAINING SUMMARY — huikang DDP 2-GPU + text CSV 0408 decoded')
    _log('='*60)
    _log(f'Method:       LoRA bf16 + CCE + MoE tying + DDP')
    _log(f'GPU:          {torch.cuda.get_device_name(LOCAL_RANK)} x {WORLD_SIZE}')
    _log(f'World size:   {WORLD_SIZE}')
    _log(f'Samples:      {len(examples)}')
    _log(f'Steps:        {num_steps}')
    _log(f'LoRA rank:    {LORA_RANK}  alpha: {LORA_ALPHA}')
    _log(f'LoRA targets: {TARGET_MODULES}')
    _log(f'LR:           {LEARNING_RATE} -> 0 (linear decay)')
    _log(f'Global batch: {GLOBAL_BATCH_SIZE} (per_rank={PER_RANK_BATCH_SIZE}, micro={MICRO_BATCH_SIZE})')
    _log(f'Max seq len:  {MAX_SEQ_LEN}')
    _log(f'MoE tying:    {MOE_TIE_WEIGHTS}')
    _log(f'Train time:   {elapsed/60:.1f} min ({elapsed/3600:.2f} hrs)')
    _log(f'Adapter:      {OUTPUT_DIR}')
    _log(f'Submission:   {ZIP_PATH}')
    _log('='*60)
    if _swanlab_ok:
        try: swanlab.finish()
        except: pass

dist.destroy_process_group()
