#!/usr/bin/env python3
"""
DDP 2-GPU version of the warm5cos run (#16 + warmup/cosine schedule).
Base = the proven DDP harness from run #8/#17 (2605231435_sft_moe_outproj_ddp, scored 0.84
with NO measured DDP penalty); the ONLY change vs that harness is the LR schedule
(linear-decay -> 5% warmup + cosine-to-0), matching the single-GPU train_moe_outproj_warm5cos.py.

Run #16 recipe (kept exactly):
- no MoE tie (MOE_TIE=0): experts independent rank-32
- out_proj LIVE: is_fast_path_available=True + every Mamba mixer .training=False
  -> mixer takes its unfused else-branch (mamba_chunk_scan_combined + self.out_proj
  as a MODULE call) -> out_proj LoRA trains.
- data 260514_huikang_golden_stripped.csv (6906 unique -> 7849 expanded)
- r32/a32/dropout0, PEAK lr 2e-4 with 5% warmup + 95% cosine-to-0, bs 32 (micro 4),
  seq 8192, 246-step schedule (245 full-batch DDP steps), stratified batching seed 42,
  CCE + manual lm_head LoRA, Kaggle rfind extractor.

DDP machinery (from run #8 2605161900, the proven DDP fix):
- contiguous rank split: rank0=global[0:16], rank1=[16:32] -> per-rank 16, micro 4, 4 accum
- per-MB normalization + (loss/n_accum).backward(); DDP all-reduce-mean == single-GPU /8
- HF NON-REENTRANT gradient checkpointing (NOT Unsloth reentrant, which is the DDP
  0.78 bug). The out_proj-live trick is orthogonal to ckpt mode.
- DDP wrap: no static_graph, find_unused_parameters=True, broadcast_buffers=False,
  gradient_as_bucket_view=False (no-tie has no in-place grad writes; removes run #8's
  prime residual-0.02 suspect).
- no_sync OFF (every micro-batch all-reduces).

Launch: MOE_TIE=0 USE_MEM_EFF=0 torchrun --nproc_per_node=2 train_moe_outproj_warm5cos_ddp.py
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

LOCAL_RANK = int(os.environ.get('LOCAL_RANK', 0))
torch.cuda.set_device(LOCAL_RANK)
dist.init_process_group(backend='nccl')
WORLD_SIZE = dist.get_world_size()
RANK = dist.get_rank()
IS_MAIN = (RANK == 0)

class _NullContext:
    def __enter__(self): return self
    def __exit__(self, *a): return False

# ── Config (IDENTICAL to run #16) ──────────────────────────────────
LORA_RANK    = 32
LORA_ALPHA   = 32
LORA_DROPOUT = 0.0
MAX_SEQ_LEN  = 8192
BATCH_SIZE   = 32                 # global effective batch
MICRO_BATCH_SIZE = 4
LEARNING_RATE = 2e-4
MOE_TIE_WEIGHTS  = os.environ.get('MOE_TIE', '0') == '1'      # default 0 (#16 = no tie)
USE_MEM_EFF_PATH = os.environ.get('USE_MEM_EFF', '0') == '1'  # default 0 (#16 = out_proj live)
SHUFFLE_SEED = 42
PER_RANK_BATCH_SIZE = BATCH_SIZE // WORLD_SIZE   # 16

TARGET_MODULES = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'up_proj', 'down_proj', 'in_proj', 'out_proj',
    'lm_head',
]

MODEL_PATH = '/root/autodl-tmp/Nemotron-3-Nano-30B-A3B'
CSV_PATH   = '/root/autodl-tmp/data/260514_huikang_golden_stripped.csv'
TOKENIZER_JSON = '/root/autodl-tmp/data/tokenizer.json'
_MOE_MODE  = 'fixtie' if MOE_TIE_WEIGHTS else 'notie'
_MEM_SUFFIX = '' if USE_MEM_EFF_PATH else '_outproj'
OUTPUT_DIR = f'/root/autodl-tmp/adapter_output_moe_{_MOE_MODE}{_MEM_SUFFIX}_ddp_warm5cos'
ZIP_PATH   = f'/root/autodl-tmp/submission_moe_{_MOE_MODE}{_MEM_SUFFIX}_ddp_warm5cos.zip'

PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
SAVE_STEPS = 50

STRESS_TEST = bool(os.environ.get('STRESS_TEST'))
FRESH_START = bool(os.environ.get('FRESH_START'))
DIAGNOSTIC_GRAD_STEPS = {1, 2, 3}

# ── Logging (main rank writes file; all ranks print to stdout) ──────
LOG_FILE = '/root/autodl-tmp/train_log_warm5cos_ddp.txt'
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

if IS_MAIN:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
dist.barrier()

from datetime import datetime, timezone, timedelta
_cdt = timezone(timedelta(hours=-5))
_ts = datetime.now(_cdt).strftime('%y%m%d_%H%M')
RUN_NAME = f'{_ts}_sft_moe_{_MOE_MODE}{_MEM_SUFFIX}_ddp2gpu_r32_a32_lr2e4_warm5cos_seq8192_bs32_rtx6000'

# ── Imports ─────────────────────────────────────────────────────────
import unsloth
from unsloth import FastLanguageModel
if IS_MAIN:
    print(f'unsloth: {unsloth.__version__}')

for _mod_name in ['mamba_ssm.modules.mamba3', 'mamba_ssm.ops.cute',
                  'mamba_ssm.ops.cute.mamba3', 'mamba_ssm.ops.cute.mamba3.mamba3_step_fn']:
    _m = types.ModuleType(_mod_name); _m.__path__ = []; _m.__package__ = _mod_name
    sys.modules[_mod_name] = _m
sys.modules['mamba_ssm.modules.mamba3'].Mamba3 = None

import mamba_ssm
from cut_cross_entropy import linear_cross_entropy
from peft import LoraConfig
from peft.tuners.lora import Linear as LoraLinear

if IS_MAIN:
    print(f'mamba_ssm: {mamba_ssm.__version__}  PyTorch: {torch.__version__}')
    print(f'World size: {WORLD_SIZE}, rank {RANK} (local {LOCAL_RANK}), per-rank batch {PER_RANK_BATCH_SIZE}')
    print(f'MOE_TIE={MOE_TIE_WEIGHTS}  USE_MEM_EFF={USE_MEM_EFF_PATH} (out_proj live={not USE_MEM_EFF_PATH})')

# ── Load base model ─────────────────────────────────────────────────
if IS_MAIN:
    print('\n=== Loading bf16 model via Unsloth ===')
gc.collect(); torch.cuda.empty_cache()
t_load = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH, max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False, load_in_8bit=False, full_finetuning=False,
    trust_remote_code=True, unsloth_force_compile=True,
    attn_implementation='eager', dtype=torch.bfloat16,
    device_map={'': LOCAL_RANK},
)
if IS_MAIN:
    print(f'Model loaded in {time.time() - t_load:.1f}s')

def _smi():
    try:
        return int(subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits',
             f'--id={LOCAL_RANK}'], capture_output=True, text=True).stdout.strip())
    except:
        return 0

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Tokenizer + boxed extractor (IDENTICAL to #16) ──────────────────
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

# ── Load + expand data (IDENTICAL to #16) ───────────────────────────
if IS_MAIN:
    print('\n=== Loading and tokenizing data ===')
with open(CSV_PATH, encoding='utf-8') as f:
    raw_rows = list(csv.DictReader(f))
if IS_MAIN:
    print(f'CSV rows (unique): {len(raw_rows)}')

expanded_rows = []
for row in raw_rows:
    n_repeat = int(row.get('oversampling', 1))
    for _ in range(n_repeat):
        expanded_rows.append(row)

examples = []
for row in expanded_rows:
    prompt_text = row['prompt']; cot = row['solver_cot']
    answer = row.get('answer', ''); pid = row['id']
    if not cot or cot == 'nan' or len(cot.strip()) < 5:
        continue
    user_msg = prompt_text + PROMPT_SUFFIX
    prompt_ids = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': user_msg}],
        tokenize=True, add_generation_prompt=True, enable_thinking=True)
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
        'problem_id': pid, 'category': row['category'],
        'tokens': all_ids[:-1], 'targets': all_ids[1:],
        'weights': [float(m) for m in mask[1:]], 'content_tail': content_tail,
    })

total_unmasked = sum(sum(e['weights']) for e in examples)
total_tokens = sum(len(e['tokens']) for e in examples)
_log(f'Loaded {len(examples)} examples, {total_tokens:,} tokens (unmasked={total_unmasked:,.0f})')
_log('(run #16 expected: 7849 examples, 27,989,496 tokens, unmasked=26,713,988)')

if STRESS_TEST:
    if IS_MAIN:
        print('\n=== STRESS TEST MODE ===')
    examples.sort(key=lambda e: len(e['tokens']), reverse=True)
    K = BATCH_SIZE * 3 * 2   # >= bs*max_steps*2
    examples = examples[:min(K, len(examples))]

# ── Wrap in LoRA (use_gradient_checkpointing=False; HF non-reentrant enabled below) ──
model = FastLanguageModel.get_peft_model(
    model, r=LORA_RANK, target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, bias='none',
    use_gradient_checkpointing=False, random_state=42,
)
FastLanguageModel.for_training(model)

# Patch Mamba CUDA fast path ON (dispatch -> cuda_kernels_forward)
nemotron_mod = None
for _name, _m in sys.modules.items():
    if 'modeling_nemotron_h' in _name and hasattr(_m, 'is_fast_path_available'):
        nemotron_mod = _m; break
if nemotron_mod is not None:
    nemotron_mod.is_fast_path_available = True
    if IS_MAIN:
        print('Patched is_fast_path_available = True')

# out_proj LIVE: force every Mamba mixer to .training=False -> unfused else-branch
# (mamba_chunk_scan_combined + self.out_proj(...) module call -> out_proj LoRA trains).
def _force_outproj_live(root):
    n = 0
    for _mod in root.modules():
        if hasattr(_mod, 'cuda_kernels_forward'):
            _mod.training = False; n += 1
    return n
if not USE_MEM_EFF_PATH:
    _n_mix = _force_outproj_live(model)
    if IS_MAIN:
        print(f'Forced unfused else-branch on {_n_mix} Mamba mixers (out_proj LoRA will train)')

# Manually add lm_head LoRA
_causal_lm = model
while hasattr(_causal_lm, 'model'):
    _causal_lm = _causal_lm.model
_lm_head = _causal_lm.lm_head
if not isinstance(_lm_head, LoraLinear):
    _cfg = LoraConfig(r=LORA_RANK, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT)
    model.base_model._create_and_replace(_cfg, 'default', target=_lm_head,
                                         target_name='lm_head', parent=_causal_lm)
    if IS_MAIN: print('Manually added LoRA to lm_head')

for name, param in model.named_parameters():
    if '.lora_' in name:
        param.data = param.data.to(torch.float32)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
if IS_MAIN:
    print(f'Model: {trainable:,} trainable / {total:,} total')

# ── CCE forward patch (stash on _base so it survives DDP wrap) ──────
_base = model
while hasattr(_base, 'model'):
    _base = _base.model

def _patched_causal_forward(input_ids=None, attention_mask=None, labels=None, **kwargs):
    backbone_out = _base.backbone(
        input_ids=input_ids, attention_mask=attention_mask,
        **{k: v for k, v in kwargs.items() if k in ('position_ids', 'past_key_values', 'use_cache')})
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
        per_token_ce = None; loss = None
    _base._cached_per_token_ce = per_token_ce
    return loss

_base.forward = _patched_causal_forward
if IS_MAIN:
    print('Patched CausalLM.forward with CCE')

# ── HF non-reentrant gradient checkpointing (the DDP-correct ckpt) ──
try:
    _base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if IS_MAIN: print('Enabled HF non-reentrant gradient checkpointing')
except Exception as e:
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if IS_MAIN: print(f'Enabled HF non-reentrant ckpt on PeftModel ({e})')

# ── MoE tie: no-tie (#16) -> _tie_grads is a no-op ──────────────────
if MOE_TIE_WEIGHTS:
    raise SystemExit('This DDP replication targets the no-tie #16 recipe; run with MOE_TIE=0.')
def _tie_grads():
    pass

# ── Wrap in DDP ─────────────────────────────────────────────────────
model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK,
            gradient_as_bucket_view=False, broadcast_buffers=False,
            find_unused_parameters=True)
# re-assert out_proj-live on the wrapped module (DDP wrap doesn't call .train(),
# but be defensive — the trick must hold through every forward)
if not USE_MEM_EFF_PATH:
    _force_outproj_live(model.module)
if IS_MAIN:
    print('Wrapped in DDP (no static_graph, find_unused_parameters=True, bucket_view=False, non-reentrant ckpt)')

# ── Stratified batching (IDENTICAL to #16: both ranks compute same batches) ──
import random as _random
if STRESS_TEST:
    # Routine §1 Rule 2: worst-case stress = sort by token length DESC, take top-K
    # (K >= bs*steps*2 = 32*3*2), no shuffle -> every micro-batch holds the longest
    # sequences -> true peak VRAM (this recipe peaks ~96 GB on a ~96 GB card, so worst-case matters).
    examples = sorted(examples, key=lambda e: len(e['tokens']), reverse=True)[:BATCH_SIZE * 3 * 2]
    batches = None
    num_steps = min(3, len(examples) // BATCH_SIZE)
    LR_DENOM = num_steps
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
        for rk, idx in enumerate(indices):
            pos = rk / n + rng.random() * 1e-6
            interleaved.append((pos, idx))
    interleaved.sort(key=lambda x: x[0])
    all_indices = [idx for _, idx in interleaved]
    # LR schedule denominator = #16's num_steps (includes ragged final batch) so
    # steps 1..245 have IDENTICAL LR to single-GPU #16.
    LR_DENOM = math.ceil(len(all_indices) / BATCH_SIZE)   # = 246
    # DROP the ragged final batch: under DDP a <BATCH_SIZE batch leaves rank 1 empty
    # -> uneven grad all-reduce count -> NCCL collective deadlock. Standard drop_last.
    batches = []
    for i in range(0, len(all_indices), BATCH_SIZE):
        b = all_indices[i:i + BATCH_SIZE]
        if len(b) == BATCH_SIZE:
            batches.append(b)
    num_steps = len(batches)   # = 245 (full batches only)
_log(f'Batching: stratified seed={SHUFFLE_SEED}, {num_steps} full-32 steps (LR_DENOM={LR_DENOM}, ragged dropped)')

# Soup adapter save schedule (no FIFO) + FIFO=1 full checkpoint
ADAPTER_SAVE_STEPS = sorted({*range(SAVE_STEPS, num_steps, SAVE_STEPS),
                             round(num_steps*0.80), round(num_steps*0.85),
                             round(num_steps*0.90), round(num_steps*0.95),
                             round(num_steps*1.00)})
ADAPTER_SAVE_STEPS = [s for s in ADAPTER_SAVE_STEPS if s > 0]
_log(f'Adapter soup save steps: {ADAPTER_SAVE_STEPS}')
import shutil
_ckpt_queue = []
def _fifo_cleanup():
    while len(_ckpt_queue) > 1:
        old = _ckpt_queue.pop(0)
        if os.path.isdir(old):
            shutil.rmtree(old, ignore_errors=True)
            _log(f'  FIFO: deleted {os.path.basename(old)}')

# ── SwanLab (main rank) ─────────────────────────────────────────────
_swanlab_ok = False
if IS_MAIN:
    try:
        import swanlab
        swanlab.init(project='260410_Nemotron', experiment_name=RUN_NAME, config={
            'lora_rank': LORA_RANK, 'lora_alpha': LORA_ALPHA, 'lr': LEARNING_RATE,
            'global_batch': BATCH_SIZE, 'per_rank_batch': PER_RANK_BATCH_SIZE,
            'world_size': WORLD_SIZE, 'micro_batch': MICRO_BATCH_SIZE,
            'max_seq_len': MAX_SEQ_LEN, 'num_examples': len(examples), 'num_steps': num_steps,
            'moe_tie': MOE_TIE_WEIGHTS, 'out_proj_live': not USE_MEM_EFF_PATH,
            'ckpt': 'hf_non_reentrant', 'targets': TARGET_MODULES,
        })
        _swanlab_ok = True; _log(f'SwanLab init: {RUN_NAME}')
    except Exception as e:
        _log(f'SwanLab init failed: {e}')

# ── Diagnostic: per-group grad-L2 + out_proj liveness (steps 1-3) ───
def _classify(name):
    if '.lm_head.' in name: return 'lm_head'
    if '.experts.' in name and '.up_proj.' in name: return 'moe_up'
    if '.experts.' in name and '.down_proj.' in name: return 'moe_down'
    if any(p in name for p in ('.q_proj.', '.k_proj.', '.v_proj.', '.o_proj.')): return 'attn'
    if '.out_proj.' in name: return 'mixer_out'
    if '.in_proj.' in name: return 'mixer_in'
    return 'other'
def _log_grad_groups(prefix, mod):
    g = {}
    op_b = 0.0
    for name, p in mod.named_parameters():
        if p.grad is None or not p.requires_grad: continue
        n2 = float((p.grad.detach().float()**2).sum().item())
        g[_classify(name)] = g.get(_classify(name), 0.0) + n2
        if '.out_proj.' in name and '.lora_B.' in name:
            op_b += n2
    total = sum(g.values())**0.5
    parts = ' '.join(f'{k}={v**0.5:.4f}' for k, v in sorted(g.items()))
    _log(f'  {prefix} total_grad={total:.4f} | {parts}')
    _log(f'  {prefix} OUT_PROJ.lora_B grad-L2 = {op_b**0.5:.6f}  (MUST be > 0 -> out_proj trains under DDP)')

# ── Resume from latest checkpoint (all ranks load identically) ──────
_resume_step = 0
_resume_optimizer_state = None
if not FRESH_START and not STRESS_TEST:
    from safetensors.torch import load_file as _load_file
    _ckpt_dirs = sorted([d for d in os.listdir(OUTPUT_DIR)
                         if d.startswith('checkpoint-') and os.path.isdir(os.path.join(OUTPUT_DIR, d))],
                        key=lambda x: int(x.split('-')[1].split('_')[0]))
    if _ckpt_dirs:
        _latest = _ckpt_dirs[-1]; _latest_dir = os.path.join(OUTPUT_DIR, _latest)
        _resume_step = int(_latest.split('-')[1].split('_')[0])
        _log(f'Resuming from {_latest} (step {_resume_step})')
        _astate = _load_file(os.path.join(_latest_dir, 'adapter_model.safetensors'))
        _renamed = {}
        for _k, _v in _astate.items():
            _nk = _k
            for _sfx in ('lora_A.weight', 'lora_B.weight', 'lora_embedding_A.weight', 'lora_embedding_B.weight'):
                if _nk.endswith('.' + _sfx):
                    _nk = _nk[:-len(_sfx)] + _sfx.replace('.weight', '.default.weight'); break
            _renamed[_nk] = _v
        _miss, _unexp = model.module.load_state_dict(_renamed, strict=False)
        _lora_miss = [m for m in _miss if 'lora_' in m]
        _log(f'  Loaded {len(_renamed)} adapter tensors (lora_missing={len(_lora_miss)}, unexpected={len(_unexp)})')
        _tsp = os.path.join(_latest_dir, 'training_state.pt')
        if os.path.exists(_tsp):
            _tsd = torch.load(_tsp, map_location='cpu', weights_only=False)
            _resume_optimizer_state = _tsd['optimizer_state_dict']
            _log('  Loaded optimizer state (will apply on optimizer init)')
    else:
        _log('No checkpoint to resume from; starting fresh')

_log('\n=== Starting training (DDP no-tie + live out_proj) ===')
_log(f'VRAM before training: smi={_smi()}M  resume_step={_resume_step}')

# ── Training loop ───────────────────────────────────────────────────
gc.collect(); torch.cuda.empty_cache()
device = torch.device(f'cuda:{LOCAL_RANK}')
optimizer = None
t0 = time.time()
step = _resume_step
for step_idx in range(_resume_step, num_steps):
    gc.collect(); torch.cuda.empty_cache()
    if batches:
        global_batch_indices = batches[step_idx]
    else:
        global_batch_indices = list(range(step_idx*BATCH_SIZE, min((step_idx+1)*BATCH_SIZE, len(examples))))
    rank_batch_indices = global_batch_indices[RANK*PER_RANK_BATCH_SIZE:(RANK+1)*PER_RANK_BATCH_SIZE]
    rank_batch = [examples[i] for i in rank_batch_indices]

    n_rank = len(rank_batch)
    if n_rank == 0:
        # ragged final batch: nothing for this rank; still must participate in collectives
        n_accum = 1
    else:
        n_accum = math.ceil(n_rank / MICRO_BATCH_SIZE)
    step_loss_sum_local = torch.zeros(1, dtype=torch.float32, device=device)
    step_weight_sum_local = torch.zeros(1, dtype=torch.float32, device=device)

    for mb_idx, mb_start in enumerate(range(0, max(n_rank, 0), MICRO_BATCH_SIZE)):
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
            sl = len(mb_toks[i])
            padded_input[i, :sl] = torch.tensor(mb_toks[i], dtype=torch.long)
            padded_targets[i, :sl] = torch.tensor(mb_tgts[i], dtype=torch.long)
            padded_weights[i, :sl] = torch.tensor(mb_wts[i], dtype=torch.float32)
            attention_mask[i, :sl] = 1

        smi_before = _smi()
        for b in range(n_micro):
            _log(f'>> step={step+1} r{RANK} mb{mb_idx} batch[{mb_start+b}] '
                 f'sample_id={mb_pids[b]} padded_len={max_len} smi={smi_before}M content_tail: {mb_tails[b]}')

        # NO no_sync() — every micro-batch all-reduces
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            model(input_ids=padded_input, attention_mask=attention_mask,
                  labels=padded_targets, use_cache=False)
            per_token_ce = _base._cached_per_token_ce
            weighted_loss = per_token_ce * padded_weights
            weight_sum_t = padded_weights.sum()
            loss_sum_t = weighted_loss.sum()
            loss = loss_sum_t / weight_sum_t if weight_sum_t > 0 else loss_sum_t * 0.0
        (loss / n_accum).backward()
        step_loss_sum_local += loss_sum_t.detach().float()
        step_weight_sum_local += weight_sum_t.detach().float()
        del loss, per_token_ce, weighted_loss, weight_sum_t, loss_sum_t

    if optimizer is None:
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
        if _resume_optimizer_state is not None:
            optimizer.load_state_dict(_resume_optimizer_state)
            _log(f'  Optimizer state restored; resuming at step {_resume_step + 1}')
            _resume_optimizer_state = None
    # LR: 5% linear WARMUP then COSINE decay to 0 over the remaining 95% (the ONLY change vs
    # run #16 / the linear-decay DDP run). Schedule length = LR_DENOM (#16's 246-step count incl.
    # the dropped ragged batch) so per-step LR is byte-identical to the single-GPU warm5cos run.
    # `step` is 0-indexed here (set =_resume_step at loop start, += 1 only after optimizer.step).
    _warmup_steps = max(1, round(LR_DENOM * 0.05))
    if step < _warmup_steps:
        lr = LEARNING_RATE * float(step) / float(_warmup_steps)
    else:
        _cos_progress = float(step - _warmup_steps) / float(max(1, LR_DENOM - _warmup_steps))
        lr = LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * _cos_progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    if (step + 1) in DIAGNOSTIC_GRAD_STEPS:
        _log_grad_groups(f'step={step+1}', model.module)

    _tie_grads()  # no-op (no tie)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1e9)
    optimizer.step()
    optimizer.zero_grad()

    # global mean loss across ranks
    gl = step_loss_sum_local.clone(); gw = step_weight_sum_local.clone()
    dist.all_reduce(gl, op=dist.ReduceOp.SUM); dist.all_reduce(gw, op=dist.ReduceOp.SUM)
    loss_mean = (gl / gw).item() if gw.item() > 0 else 0.0

    smi_after = _smi(); step += 1
    elapsed = time.time() - t0
    eta = (elapsed / step) * (num_steps - step) if step > 0 else 0
    _log(f'   step={step} smi_after={smi_after}M loss={loss_mean:.6f} grad_norm={grad_norm:.4f} '
         f'lr={lr:.2e} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m')
    if IS_MAIN and _swanlab_ok:
        try: swanlab.log({'loss': loss_mean, 'grad_norm': float(grad_norm), 'lr': lr, 'vram_mb': smi_after}, step=step)
        except Exception: pass

    # ── Saves (main rank only; disabled during stress per routine §1 Rule 3) ──
    if IS_MAIN and not STRESS_TEST and (step % SAVE_STEPS == 0 or step == num_steps):
        ckpt_name = f'checkpoint-{step}_loss{loss_mean:.4f}_lr{lr:.2e}'
        ckpt_dir = os.path.join(OUTPUT_DIR, ckpt_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        model.module.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        torch.save({'optimizer_state_dict': optimizer.state_dict(), 'step': step, 'lr': lr,
                    'loss': loss_mean, 'rng_state': torch.cuda.get_rng_state(),
                    'python_rng_state': _random.getstate(),
                    'training_args': {'lora_rank': LORA_RANK, 'lora_alpha': LORA_ALPHA,
                        'learning_rate': LEARNING_RATE, 'global_batch': BATCH_SIZE,
                        'per_rank_batch': PER_RANK_BATCH_SIZE, 'world_size': WORLD_SIZE,
                        'micro_batch': MICRO_BATCH_SIZE, 'max_seq_len': MAX_SEQ_LEN,
                        'num_steps': num_steps, 'num_examples': len(examples),
                        'moe_tie': MOE_TIE_WEIGHTS, 'out_proj_live': not USE_MEM_EFF_PATH}},
                   os.path.join(ckpt_dir, 'training_state.pt'))
        _log(f'  Checkpoint saved: {ckpt_name}')
        _ckpt_queue.append(ckpt_dir); _fifo_cleanup()

    if IS_MAIN and not STRESS_TEST and step in ADAPTER_SAVE_STEPS:
        adir = os.path.join(OUTPUT_DIR, f'adapter-{step}_loss{loss_mean:.4f}_lr{lr:.2e}')
        os.makedirs(adir, exist_ok=True)
        model.module.save_pretrained(adir)
        _log(f'  Adapter-only saved (soup): {os.path.basename(adir)}')

    gc.collect(); torch.cuda.empty_cache()
    dist.barrier()

elapsed = time.time() - t0
_log(f'\nTraining done. Time: {elapsed/3600:.2f} hrs ({elapsed/60:.1f} min)')
_log(f'Peak VRAM: smi={_smi()}M')

if STRESS_TEST:
    _log('STRESS TEST COMPLETE')
    if IS_MAIN and _swanlab_ok:
        try: swanlab.finish()
        except: pass
    dist.destroy_process_group(); sys.exit(0)

# ── Save final adapter (main rank) ──────────────────────────────────
if IS_MAIN:
    _log('\n=== Saving adapter ===')
    from safetensors.torch import load_file, save_file
    model.module.save_pretrained(OUTPUT_DIR)
    st_path = os.path.join(OUTPUT_DIR, 'adapter_model.safetensors')
    tensors = load_file(st_path)
    renamed = {k.replace('base_model.model.lm_head.', 'base_model.model.backbone.lm_head.'): v
               for k, v in tensors.items()}
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
        czip = os.path.join(os.path.dirname(OUTPUT_DIR), f'{final_ckpt}.zip')
        cfull = os.path.join(OUTPUT_DIR, final_ckpt)
        with zipfile.ZipFile(czip, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(cfull):
                for f in files:
                    fp = os.path.join(root, f)
                    zf.write(fp, arcname=os.path.relpath(fp, cfull))
        _log(f'Checkpoint zip: {czip} ({os.path.getsize(czip)/1024/1024:.1f} MB)')

    _log('\n' + '='*60)
    _log('TRAINING SUMMARY — DDP 2-GPU no-tie + live out_proj (repl. of #16)')
    _log(f'GPU: {torch.cuda.get_device_name(LOCAL_RANK)} x {WORLD_SIZE}')
    _log(f'Steps: {num_steps}  Train time: {elapsed/60:.1f} min')
    _log('='*60)
    if _swanlab_ok:
        try: swanlab.finish()
        except: pass

dist.destroy_process_group()
