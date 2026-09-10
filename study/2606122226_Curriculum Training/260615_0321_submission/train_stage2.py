#!/usr/bin/env python3
"""
Curriculum Stage 2 — continual-FT from Stage 1 LoRA on the Stage 2 data
(100% original replay + ~50% aug1). SAME proven 0.86/0.87 recipe as Stage 1.
Code byte-identical to Stage 1 (train_stage1.py) EXCEPT docstring + CSV/naming:
  - LOG_FILE/_DATA_TAG naming -> curriculum_stage2
  - CSV_PATH       -> 260614_STAGE2_FULL.csv
(max_norm=1.0 and the linear 2e-4->0 schedule are unchanged from Stage 1.)

USER CONFIG (Stage 2): NO warmup, NO WSD, NO cosine — keep the favorite
linear decay 2e-4 -> 0 with the original peak LR. (i.e. identical schedule to Stage 1.)

Run with: MOE_TIE=0 USE_MEM_EFF=0 FRESH_START=1 and
  INIT_ADAPTER_FROM=/root/autodl-tmp/adapter_output_curriculum_stage1_notie_outproj/adapter-230_loss0.0019_lr8.70e-07
  -> Stage 2 loads Stage 1 trained LoRA as init (continual-FT), fresh optimizer/scheduler,
     no-tie, live out_proj. (FRESH_START=1 => no resume-from-checkpoint; INIT block applies the Stage 1 weights.)

Data note: 260614_STAGE2_FULL.csv marks dropped aug rows with oversampling=0, so the
n_repeat=int(oversampling) expansion keeps only the curriculum Stage 2 pool.
Expected at startup:
  CSV rows: 19961; rows with oversampling>=1: 12896 (all GT-match=True)
  Expanded ~13786 -> ~431 steps at bs=32 (~7.2 hrs at ~1 min/step)

Soup adapter saves kept verbatim (cheap insurance, [[feedback_always_save_soup_adapters]]);
resumable checkpoint required. No soup BUILD planned for Stage 2 (recipe).
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
LOG_FILE = f"/root/autodl-tmp/train_log_curriculum_stage2_{'fixtie' if os.environ.get('MOE_TIE', '1') == '1' else 'notie'}{'' if os.environ.get('USE_MEM_EFF', '1') == '1' else '_outproj'}.txt"
_log_fh = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
_log_sh = logging.StreamHandler(sys.stdout)
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[_log_fh, _log_sh])
_logger = logging.getLogger('train')
def _log(msg):
    _logger.info(msg)

# ── Config (matches run #7 exactly + soup-save additions) ──────────
LORA_RANK    = 32
LORA_ALPHA   = 32
LORA_DROPOUT = 0.0
MAX_SEQ_LEN  = 8192
BATCH_SIZE   = 32
MICRO_BATCH_SIZE = 4
LEARNING_RATE = 2e-4
NUM_STEPS    = 1000  # will be clamped to actual max
MOE_TIE_WEIGHTS = os.environ.get('MOE_TIE', '1') == '1'  # 1=fixed expert-direction tie, 0=no tie
USE_MEM_EFF_PATH = os.environ.get('USE_MEM_EFF', '1') == '1'  # 0=disable fused Mamba kernel so out_proj LoRA trains
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
CSV_PATH   = '/root/autodl-tmp/data/260614_STAGE2_FULL.csv'  # curriculum Stage 2: 100% original replay + ~50% aug1 -> 12896 active -> ~13786 expanded -> ~431 steps
TOKENIZER_JSON = '/root/autodl-tmp/data/tokenizer.json'
_MOE_MODE  = 'fixtie' if MOE_TIE_WEIGHTS else 'notie'
_MEM_SUFFIX = '' if USE_MEM_EFF_PATH else '_outproj'
_DATA_TAG  = 'curriculum_stage2'  # Stage 2: 100% original replay + ~50% aug1, continual-FT from Stage 1 adapter
OUTPUT_DIR = f'/root/autodl-tmp/adapter_output_{_DATA_TAG}_{_MOE_MODE}{_MEM_SUFFIX}'
ZIP_PATH   = f'/root/autodl-tmp/submission_{_DATA_TAG}_{_MOE_MODE}{_MEM_SUFFIX}.zip'

PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
SAVE_STEPS = 50  # per routine: max(25, round(246/5/25)*25) = 50

STRESS_TEST = bool(os.environ.get('STRESS_TEST'))
FRESH_START = bool(os.environ.get('FRESH_START'))

os.makedirs(OUTPUT_DIR, exist_ok=True)

from datetime import datetime, timezone, timedelta
_cdt = timezone(timedelta(hours=-5))
_ts = datetime.now(_cdt).strftime('%y%m%d_%H%M')
RUN_NAME = f'{_ts}_sft_{_DATA_TAG}_{_MOE_MODE}{_MEM_SUFFIX}_r32_a32_lr2e4_seq8192_bs32_rtx6000'

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
csv.field_size_limit(2**31 - 1)  # this data has 0x-dropped runaway crypt CoTs up to 6.8M chars; DictReader parses all rows before the 0x drop, so raise the field limit to read past them (training-neutral: those rows are dropped, max TRAINED CoT ~18K chars)

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
        print(f'SKIP {pid}: no unmasked tokens')
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
    # ALWAYS keep is_fast_path_available=True so dispatch uses cuda_kernels_forward (CUDA),
    # never the naive torch_forward (which materializes a 126GB SSM tensor at seq 8192).
    nemotron_mod.is_fast_path_available = True
    print('Patched is_fast_path_available = True')

# out_proj training: inside cuda_kernels_forward the fused mamba_split_conv1d_scan_combined
# (which folds out_proj via raw weight -> out_proj LoRA dead) is gated on
#   `if self.training and cache_params is None:`
# The ELSE branch is the model's own efficient path: causal_conv1d_fn + mamba_chunk_scan_combined
# + `out = self.out_proj(scan_output)` (a MODULE call -> out_proj LoRA TRAINS).
# USE_MEM_EFF=0 -> set each Mamba mixer's .training=False so the fused check fails and the mixer
# takes that else-branch. The mixer has no dropout and autograd ignores the training flag, so
# gradients still flow; grad-checkpoint recompute is consistent (flag persists). No model-file
# edit, no reimplementation -- uses the model's own kernels (same category as the flag flip above).
if not USE_MEM_EFF_PATH:
    _n_mix = 0
    for _mod in model.modules():
        if hasattr(_mod, 'cuda_kernels_forward'):  # the Mamba2 mixer
            _mod.training = False
            _n_mix += 1
    print(f'Forced unfused else-branch on {_n_mix} Mamba mixers (out_proj LoRA will train)')

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

# ── Stage 2: load Stage 1 adapter as init (fresh optimizer + scheduler stay) ───────
# Loads LoRA weights from INIT_ADAPTER_FROM into the freshly created LoRA layers.
# Mirrors the §2.4A resume block's `.default.` key rename. Optimizer/scheduler NOT
# restored — Stage 2 trains a clean 1-epoch cycle on the FULL data.
_INIT_ADAPTER_FROM = os.environ.get('INIT_ADAPTER_FROM', '').strip()
if _INIT_ADAPTER_FROM:
    from safetensors.torch import load_file as _init_load_file
    _init_path = os.path.join(_INIT_ADAPTER_FROM, 'adapter_model.safetensors')
    assert os.path.exists(_init_path), f'INIT_ADAPTER_FROM/adapter_model.safetensors not found: {_init_path}'
    _init_state = _init_load_file(_init_path)
    _init_renamed = {}
    for _k, _v in _init_state.items():
        _nk = _k
        # Reverse the §3 Kaggle-compat rename: backbone.lm_head -> lm_head
        # (Stage 1's submission.zip stores lm_head LoRA under backbone.lm_head;
        # Stage 2's fresh PEFT model exposes it at the original lm_head path.)
        if 'base_model.model.backbone.lm_head.' in _nk:
            _nk = _nk.replace('base_model.model.backbone.lm_head.', 'base_model.model.lm_head.', 1)
        # PEFT save_pretrained strips '.default.' from adapter keys; restore for state_dict load
        for _sfx in ('lora_A.weight', 'lora_B.weight',
                     'lora_embedding_A.weight', 'lora_embedding_B.weight'):
            if _nk.endswith('.' + _sfx):
                _nk = _nk[: -len(_sfx)] + _sfx.replace('.weight', '.default.weight')
                break
        _init_renamed[_nk] = _v
    # Cast to fp32 to match the LoRA fp32 cast above
    for _k in list(_init_renamed):
        if 'lora_' in _k:
            _init_renamed[_k] = _init_renamed[_k].to(torch.float32)
    _missing, _unexpected = model.load_state_dict(_init_renamed, strict=False)
    _lora_loaded = sum(1 for _k in _init_renamed if 'lora_' in _k)
    _lora_missing = [m for m in _missing if 'lora_' in m]
    print(f'INIT_ADAPTER_FROM={_INIT_ADAPTER_FROM}: loaded {len(_init_renamed)} tensors '
          f'({_lora_loaded} LoRA), missing={len(_missing)} ({len(_lora_missing)} LoRA), '
          f'unexpected={len(_unexpected)}')
    if _unexpected:
        print(f'  Unexpected (first 3): {_unexpected[:3]}')
    if _lora_missing:
        print(f'  LoRA missing (first 3): {_lora_missing[:3]}')
    # Sanity: a trained adapter has non-zero lora_B (random init is B=0).
    # Sample one to verify the load actually applied trained weights.
    for _n, _p in model.named_parameters():
        if 'lora_B' in _n and _p.numel() > 0:
            _norm = _p.detach().float().norm().item()
            print(f'  Sanity lora_B norm ({_n.split(".")[-3]}): {_norm:.6f} (random init = 0.0)')
            break
    assert len(_lora_missing) == 0, f'INIT_ADAPTER_FROM failed: {len(_lora_missing)} LoRA keys missing'
else:
    print('INIT_ADAPTER_FROM unset; Stage 2 LoRA weights left at random init')

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

# ── MoE weight tying (FIXED: tie ACROSS the 128 experts, not within rank) ──
# Our Unsloth exposes each expert as a SEPARATE [rank, in] param, so the old
# mean(dim=0) collapsed each expert's own rank to 1. Correct expert-direction
# tie = group the per-expert params (same layer/proj/factor) and average ACROSS
# experts, keeping each [rank, in] full rank-32 and all 128 experts identical.
moe_tied_groups = []
if MOE_TIE_WEIGHTS:
    w1_proj_names = ('gate_up_proj', 'up_proj', 'gate_proj', '.w1.')
    w2_proj_names = ('down_proj', '.w2.')
    _group_map = {}
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
        gkey = re.sub(r'\.experts\.\d+\.', '.experts.E.', name)  # group across expert index
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
    print(f'MoE weight tying (across experts): {len(moe_tied_groups)} groups, {_n_tied} params, '
          f'group_sizes={sorted(set(len(g) for g in moe_tied_groups))}, '
          f'factor_shape={tuple(moe_tied_groups[0][0].shape) if moe_tied_groups else None}')
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

# ── Adapter-only soup save schedule (no FIFO, all on disk) ───────
# Per routine §2.4B: only the 7 steps the two soups consume, as % of num_steps
# (no per-SAVE_STEPS spam). wise=20/60/100%, last5=80/85/90/95/100%.
ADAPTER_SAVE_STEPS = sorted({
    round(num_steps * p) for p in (0.20, 0.60, 0.80, 0.85, 0.90, 0.95, 1.00)
})
ADAPTER_SAVE_STEPS = [s for s in ADAPTER_SAVE_STEPS if s > 0]
_log(f'Adapter-only soup save steps: {ADAPTER_SAVE_STEPS}')

# ── FIFO checkpoint queue — keep only 1 per routine §2.4A ──────────
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
            'data': 'run #7 byte-identical: huikang_non_cryptarithm (6106) + lkevincc_all_cryptarithm (800)',
            'extractor': 'kaggle_metric_rfind',
            'soup_save_steps': ADAPTER_SAVE_STEPS,
            'fifo_keep': 1,
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

# ── Resume from latest checkpoint (skip if FRESH_START=1) ────────────
_resume_step = 0
_resume_optimizer_state = None
if not FRESH_START:
    from safetensors.torch import load_file as _load_file
    _ckpt_dirs = sorted(
        [d for d in os.listdir(OUTPUT_DIR)
         if d.startswith('checkpoint-') and os.path.isdir(os.path.join(OUTPUT_DIR, d))],
        key=lambda x: int(x.split('-')[1].split('_')[0])
    )
    if _ckpt_dirs:
        _latest = _ckpt_dirs[-1]
        _latest_dir = os.path.join(OUTPUT_DIR, _latest)
        _resume_step = int(_latest.split('-')[1].split('_')[0])
        _log(f'Resuming from checkpoint: {_latest} (step {_resume_step})')
        _adapter_path = os.path.join(_latest_dir, 'adapter_model.safetensors')
        _adapter_state = _load_file(_adapter_path)
        # PEFT save_pretrained strips '.default.' from adapter keys; restore for state_dict load
        _renamed_state = {}
        for _k, _v in _adapter_state.items():
            _nk = _k
            for _sfx in ('lora_A.weight', 'lora_B.weight',
                         'lora_embedding_A.weight', 'lora_embedding_B.weight'):
                if _nk.endswith('.' + _sfx):
                    _nk = _nk[: -len(_sfx)] + _sfx.replace('.weight', '.default.weight')
                    break
            _renamed_state[_nk] = _v
        _missing, _unexpected = model.load_state_dict(_renamed_state, strict=False)
        _log(f'  Loaded adapter weights: {len(_renamed_state)} tensors '
             f'(missing={len(_missing)}, unexpected={len(_unexpected)})')
        if _unexpected:
            _log(f'  Unexpected keys (first 3): {_unexpected[:3]}')
        if _missing:
            _lora_missing = [m for m in _missing if 'lora_' in m]
            _log(f'  Missing keys: {len(_missing)} total, {len(_lora_missing)} LoRA-related '
                 f'(rest are frozen base params, expected). LoRA missing sample: {_lora_missing[:3]}')
        _ts_path = os.path.join(_latest_dir, 'training_state.pt')
        if os.path.exists(_ts_path):
            _ts = torch.load(_ts_path, map_location='cpu', weights_only=False)
            _resume_optimizer_state = _ts['optimizer_state_dict']
            _log(f'  Loaded optimizer state from {os.path.basename(_ts_path)}')
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
            lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
        )
        if _resume_optimizer_state is not None:
            optimizer.load_state_dict(_resume_optimizer_state)
            _log(f'  Optimizer state loaded, resuming at step {_resume_step + 1}')
            _resume_optimizer_state = None
    lr = LEARNING_RATE * (1 - step / num_steps)
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    _tie_grads()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
    # Routine §1 rule 6: verify out_proj LoRA is alive after stress step 1 (live-out_proj trick fired)
    if STRESS_TEST and step_idx == 0:
        _opn = sum(p.grad.norm().item() for n,p in model.named_parameters() if 'out_proj.lora_B' in n and p.grad is not None)
        assert _opn > 1e-6, f'out_proj.lora_B grads ~0 ({_opn:.2e}) — fused Mamba kernel folded it; check USE_MEM_EFF=0 + is_fast_path_available=True + mixer.training=False'
        _log(f'out_proj.lora_B grad norm sum (stress step 1): {_opn:.4f} > 0 ✓')
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

    # Full checkpoint saving (adapter + optimizer + scheduler + rng + tokenizer + args, FIFO=1)
    if not STRESS_TEST and (step % SAVE_STEPS == 0 or step == num_steps):
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

    # Adapter-only save for LoRA soup (no optimizer state, no FIFO)
    if not STRESS_TEST and step in ADAPTER_SAVE_STEPS:
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
        except: pass
    sys.exit(0)

# ── Save final adapter + rename lm_head keys ────────────────────────
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
_log('TRAINING SUMMARY — retrain of run #7 with soup-ready adapter saves')
_log('='*60)
_log(f'Method:       LoRA bf16 + CCE + MoE tying')
_log(f'GPU:          {torch.cuda.get_device_name(0)}')
_log(f'Samples:      {len(examples)}')
_log(f'Steps:        {num_steps}')
_log(f'LoRA rank:    {LORA_RANK}  alpha: {LORA_ALPHA}')
_log(f'LoRA targets: {TARGET_MODULES}')
_log(f'LR:           {LEARNING_RATE} -> 0 (linear decay)')
_log(f'Batch:        {BATCH_SIZE} (micro={MICRO_BATCH_SIZE})')
_log(f'Adapter saves: {ADAPTER_SAVE_STEPS}')
_log(f'Submission:   {ZIP_PATH}')

if _swanlab_ok:
    try: swanlab.finish()
    except: pass
