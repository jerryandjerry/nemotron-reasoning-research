#!/usr/bin/env python3
"""DPO distillation from SFT 0.85 cryptnumeq_outproj seed — TRL-free rewrite.

Built by copying study/2605270226_sft_newdata_cryptnumeq_outproj_rtx6000/train_cryptnumeq_outproj.py
verbatim and adding the minimum DPO delta:
  - Load chosen + rejected pairs (PREFER_CSV / REJECT_CSV joined on `id`)
  - Precompute reference logp at startup (one eval pass, cached on CPU)
  - Each step does 2 inference_mode forwards (Phase 1: get current policy logp_c/logp_r),
    compute s = sigmoid(-β·δ) off-graph, then 2 with-grad forwards each followed by
    immediate .backward() with coefficient ±β·s (GroupDPO gradient decomposition,
    arXiv 2604.15602)
  - Then optimizer.step() / zero_grad() — SFT-style manual loop, NO TRL DPOTrainer,
    NO accelerate, NO HF Trainer wrapping

Memory profile target (verified by diagnostic at T=8192 B=1): peak ~74 GB, headroom ~20 GB.
"""

import os, sys, time, json, zipfile, gc, subprocess, types, math, logging, re

os.environ['TRANSFORMERS_NO_TF']     = '1'
os.environ['TRANSFORMERS_NO_FLAX']   = '1'
os.environ['CUDA_VISIBLE_DEVICES']   = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HF_HUB_OFFLINE']         = '1'
os.environ['TRANSFORMERS_OFFLINE']   = '1'
os.environ['SWANLAB_API_KEY'] = dict(l.strip().split('=', 1) for l in open('kaggle.env', encoding='utf-8-sig') if '=' in l and not l.startswith('#'))['SWANLAB_API_KEY']
os.environ['SWANLAB_PROJECT']        = 'nemotron-dpo'

# ── Logging (single writer) ────────────────────────────────────────
LOG_FILE = '/root/autodl-tmp/train_dpo_log.txt'
_log_fh = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
_log_sh = logging.StreamHandler(sys.stdout)
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[_log_fh, _log_sh])
_logger = logging.getLogger('dpo')
def _log(msg): _logger.info(msg)

# ── Config (DPO values; rest mirrors SFT 0.85) ─────────────────────
LORA_RANK         = 32
LORA_ALPHA        = 32
LORA_DROPOUT      = 0.0
MAX_SEQ_LEN       = 8192          # SFT's value — proven to fit with smart GC
MAX_PROMPT_LEN    = 1024
BATCH_SIZE        = 16            # effective batch (16 pairs per optimizer step)
MICRO_BATCH_SIZE  = 4             # pairs per micro-step. Each Phase 3 forward processes
                                  # B=MICRO_BATCH pairs at once (same B×T as SFT's forward).
                                  # 16 / 4 = 4 micro-steps per optimizer step.
LEARNING_RATE     = 5e-7          # Tülu-3 DPO default
BETA              = 0.05          # low: rejected often correct → soft distillation
NUM_EPOCHS        = 1

# MoE tying (same as SFT)
MOE_TIE_WEIGHTS   = os.environ.get('MOE_TIE', '1') == '1'
USE_MEM_EFF_PATH  = os.environ.get('USE_MEM_EFF', '1') == '1'   # 0=out_proj LoRA trains
SHUFFLE_SEED      = 42

TARGET_MODULES = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'up_proj', 'down_proj', 'in_proj', 'out_proj',
    'lm_head',
]

EASY_CATEGORIES_VAL_GE_98 = {'numeral', 'gravity', 'unit_conversion', 'cipher', 'bit_manipulation'}
ANCHOR_FRACTION = 0.10
# 'bit_manipulation' added 2026-05-31 per user: dominant at 1354/2658 = 51% of post-filter.
# Throttling to 10% gives more relative weight to cryptarithm + equation_numeric.

# ── Paths ─────────────────────────────────────────────────────────
MODEL_PATH        = '/root/autodl-tmp/Nemotron-3-Nano-30B-A3B'
SEED_ADAPTER_PATH = '/root/autodl-tmp/sft_seed_cryptnumeq'
PREFER_CSV        = '/root/autodl-tmp/data/prefer_260527_huikang_NumericEq.csv'
REJECT_CSV        = '/root/autodl-tmp/data/reject_numericeq.csv'
TOKENIZER_JSON    = '/root/autodl-tmp/data/tokenizer.json'

_MOE_MODE  = 'fixtie' if MOE_TIE_WEIGHTS else 'notie'
_MEM_SUFFIX = '' if USE_MEM_EFF_PATH else '_outproj'
_DATA_TAG  = 'dpo_distill_alldata_v2'
OUTPUT_DIR = f'/root/autodl-tmp/adapter_output_{_DATA_TAG}_{_MOE_MODE}{_MEM_SUFFIX}'

PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'
SAVE_STEPS = 25

STRESS_TEST = bool(os.environ.get('STRESS_TEST'))
FRESH_START = bool(os.environ.get('FRESH_START'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

from datetime import datetime, timezone, timedelta
_cdt = timezone(timedelta(hours=-5))
_ts  = datetime.now(_cdt).strftime('%y%m%d_%H%M')
RUN_NAME = f'{_ts}_dpo_distill_{_DATA_TAG}_b{BETA}_lr{LEARNING_RATE}_bs{BATCH_SIZE}_seq{MAX_SEQ_LEN}_rtx6000'

# ── Stub mamba3 modules (same as SFT) ──────────────────────────────
for _mod_name in [
    'mamba_ssm.modules.mamba3',
    'mamba_ssm.ops.cute',
    'mamba_ssm.ops.cute.mamba3',
    'mamba_ssm.ops.cute.mamba3.mamba3_step_fn',
]:
    _m = types.ModuleType(_mod_name); _m.__path__ = []; _m.__package__ = _mod_name
    sys.modules[_mod_name] = _m
sys.modules['mamba_ssm.modules.mamba3'].Mamba3 = None

import unsloth
from unsloth import FastLanguageModel
print(f'unsloth: {unsloth.__version__}')

import mamba_ssm
import torch
import torch.nn.functional as F
from cut_cross_entropy import linear_cross_entropy
from peft import PeftModel, LoraConfig
from peft.tuners.lora import Linear as LoraLinear
import pandas as pd

print(f'mamba_ssm: {mamba_ssm.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'GPU: {torch.cuda.get_device_name(0)}')

def _smi():
    try:
        return int(subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True).stdout.strip())
    except: return 0

# ── Load base bf16 model (SFT-verbatim) ────────────────────────────
print('\n=== Loading bf16 model via Unsloth ===')
gc.collect(); torch.cuda.empty_cache()
from transformers.dynamic_module_utils import get_class_from_dynamic_module as _get_cls
_clm = _get_cls('modeling_nemotron_h.NemotronHForCausalLM', MODEL_PATH)
_clm._supports_sdpa = True
for _b in _clm.__mro__:
    if _b.__name__ == 'NemotronHPreTrainedModel':
        _b._supports_sdpa = True
        break

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
print(f'Model loaded in {time.time() - t_load:.1f}s, smi={_smi()}M')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Load SFT seed adapter via get_peft_model + load adapter state ──
# We use get_peft_model (NOT PeftModel.from_pretrained) so Unsloth's smart-GC
# autograd Function is installed on the outer forward path.
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=TARGET_MODULES,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias='none',
    use_gradient_checkpointing='unsloth',
    random_state=SHUFFLE_SEED,
)
FastLanguageModel.for_training(model)

# Load seed adapter weights from the SFT 0.85 checkpoint into the fresh LoRA layout.
# PEFT save_pretrained strips '.default.' from keys; restore for state_dict load.
from safetensors.torch import load_file as _load_file
_seed_state = _load_file(os.path.join(SEED_ADAPTER_PATH, 'adapter_model.safetensors'))
_renamed = {}
for _k, _v in _seed_state.items():
    _nk = _k
    for _sfx in ('lora_A.weight', 'lora_B.weight'):
        if _nk.endswith('.' + _sfx):
            _nk = _nk[:-len(_sfx)] + _sfx.replace('.weight', '.default.weight')
            break
    _renamed[_nk] = _v
_missing, _unexpected = model.load_state_dict(_renamed, strict=False)
_log(f'Seed adapter loaded: {len(_renamed)} tensors  '
     f'(missing-lora={len([m for m in _missing if "lora_" in m])}, unexpected={len(_unexpected)})')

# ── Mamba fast path patch (SFT-verbatim) ───────────────────────────
nemotron_mod = None
for _name, _m in sys.modules.items():
    if 'modeling_nemotron_h' in _name and hasattr(_m, 'is_fast_path_available'):
        nemotron_mod = _m
        break
if nemotron_mod is not None:
    nemotron_mod.is_fast_path_available = True
    print('Patched is_fast_path_available = True')

# ── out_proj-live trick (SFT-verbatim, only if USE_MEM_EFF=0) ──────
if not USE_MEM_EFF_PATH:
    _n_mix = 0
    for _mod in model.modules():
        if hasattr(_mod, 'cuda_kernels_forward'):
            _mod.training = False
            _n_mix += 1
    print(f'Forced unfused else-branch on {_n_mix} Mamba mixers')

# ── Manually add lm_head LoRA if not already wrapped (SFT-verbatim) ──
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

# ── Cast LoRA params to fp32 (SFT-verbatim) ────────────────────────
for name, p in model.named_parameters():
    if '.lora_' in name:
        p.data = p.data.to(torch.float32)
print('Cast LoRA params to fp32')

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'Trainable: {trainable/1e6:.1f}M / total {total/1e9:.1f}B')

# ── CCE forward patch (SFT-verbatim: base_w + scaling * lora_B @ lora_A) ──
_base = model
while hasattr(_base, 'model'):
    _base = _base.model

def _patched_causal_forward(input_ids=None, attention_mask=None, labels=None, **kwargs):
    backbone_out = _base.backbone(
        input_ids=input_ids, attention_mask=attention_mask,
        **{k: v for k, v in kwargs.items() if k in ('position_ids', 'past_key_values', 'use_cache')},
    )
    hidden = backbone_out[0]
    lh = _base.lm_head
    base_w  = lh.base_layer.weight
    lora_A  = lh.lora_A['default'].weight
    lora_B  = lh.lora_B['default'].weight
    scaling = lh.scaling['default']
    lm_weight = base_w + scaling * lora_B @ lora_A
    if labels is not None:
        per_token_ce = linear_cross_entropy(hidden, lm_weight, labels, reduction='none')
        loss = per_token_ce.mean()
    else:
        per_token_ce = None
        loss = None
    model._cached_per_token_ce = per_token_ce
    return loss

_base.forward = _patched_causal_forward
print('Patched CausalLM.forward with CCE')

# ── MoE weight tying (SFT-verbatim) ────────────────────────────────
moe_tied_groups = []
if MOE_TIE_WEIGHTS:
    w1_proj_names = ('gate_up_proj', 'up_proj', 'gate_proj', '.w1.')
    w2_proj_names = ('down_proj', '.w2.')
    _group_map = {}
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if '.experts.' not in name or '.lora_' not in name: continue
        is_w1 = any(p in name for p in w1_proj_names)
        is_w2 = any(p in name for p in w2_proj_names)
        is_A  = '.lora_A.' in name
        is_B  = '.lora_B.' in name
        if not ((is_w1 and is_A) or (is_w2 and is_B)): continue
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
                if any(p.grad is None for p in group): continue
                gsum = torch.stack([p.grad for p in group], dim=0).sum(dim=0)
                for p in group:
                    p.grad.copy_(gsum)
    print(f'MoE tying: {len(moe_tied_groups)} groups, {sum(len(g) for g in moe_tied_groups)} params')
    _tie_param_init()
else:
    def _tie_grads(): pass

# ── Build DPO dataset (PREFER + REJECT joined on id) ───────────────
print('\n=== Loading DPO data (chosen + rejected pairs) ===')
dp = pd.read_csv(PREFER_CSV)
dr = pd.read_csv(REJECT_CSV)

def _last_tok(s):
    try: return json.loads(s)[-1]
    except: return None
dr['_eos_emitted'] = dr['tokens'].apply(_last_tok) == '<|im_end|>'
joined = dp[['id','prompt','answer','solver_cot']].merge(
    dr[['id','raw_output','category','_eos_emitted']], on='id'
)
_log(f'Joined: {len(joined)} pairs')

def _filter_easy(df):
    parts = []
    for cat, grp in df.groupby('category'):
        if cat in EASY_CATEGORIES_VAL_GE_98:
            n = max(1, int(round(len(grp) * ANCHOR_FRACTION)))
            parts.append(grp.sample(n=n, random_state=42))
        else:
            parts.append(grp)
    return pd.concat(parts, ignore_index=True)

joined = _filter_easy(joined)
_log(f'Post-filter: {len(joined)} pairs')

if STRESS_TEST:
    # Routine §1.2: sort by length DESC, take top-K = bs × max_steps × 2 (no shuffle)
    STRESS_STEPS = 3
    K = BATCH_SIZE * STRESS_STEPS * 2  # 16 × 3 × 2 = 96
    joined = joined.assign(_tot=joined.solver_cot.str.len() + joined.raw_output.str.len())
    joined = joined.sort_values('_tot', ascending=False).head(K).reset_index(drop=True)
    print(f'STRESS_TEST: {len(joined)} longest pairs, {STRESS_STEPS} steps (no shuffle)')

_boxed_pat = re.compile(r'\\boxed\{')
def _extract_last_boxed(text):
    starts = list(_boxed_pat.finditer(text))
    if not starts: return None
    last = starts[-1]; seg = text[last.end():]; lb = seg.rfind('}')
    return seg[:lb] if lb != -1 else seg

def _tokenize_pair(row):
    user_msg = row['prompt'] + PROMPT_SUFFIX
    prompt_ids = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': user_msg}],
        tokenize=True, add_generation_prompt=True, enable_thinking=True,
    )
    reasoning_answer = _extract_last_boxed(row['solver_cot']) or str(row['answer'])
    chosen_text = row['solver_cot'] + '\n</think>\n' + chr(92) + 'boxed{' + reasoning_answer + '}<|im_end|>'
    rejected_text = row['raw_output'] + ('<|im_end|>' if row['_eos_emitted'] else '')
    chosen_ids   = tokenizer(chosen_text,   add_special_tokens=False)['input_ids']
    rejected_ids = tokenizer(rejected_text, add_special_tokens=False)['input_ids']
    chosen_full   = (prompt_ids + chosen_ids)[:MAX_SEQ_LEN]
    rejected_full = (prompt_ids + rejected_ids)[:MAX_SEQ_LEN]
    pL = len(prompt_ids)
    chosen_mask   = [0] * min(pL, len(chosen_full))   + [1] * max(0, len(chosen_full)   - pL)
    rejected_mask = [0] * min(pL, len(rejected_full)) + [1] * max(0, len(rejected_full) - pL)
    return {
        'problem_id'   : row['id'],
        'category'     : row['category'],
        'chosen_ids'   : chosen_full,
        'chosen_mask'  : chosen_mask,
        'rejected_ids' : rejected_full,
        'rejected_mask': rejected_mask,
    }

print('Tokenizing pairs …')
examples = [_tokenize_pair(r) for _, r in joined.iterrows()]
total_chosen_toks   = sum(len(e['chosen_ids'])   for e in examples)
total_rejected_toks = sum(len(e['rejected_ids']) for e in examples)
_log(f'Tokenized: {len(examples)} pairs  chosen_toks={total_chosen_toks:,}  rejected_toks={total_rejected_toks:,}')

# token_to_sid: map first-8-tokens prefix → problem_id for inspectability
token_to_sid = {}
for e in examples:
    token_to_sid[tuple(e['chosen_ids'][:8])]   = e['problem_id']
    token_to_sid[tuple(e['rejected_ids'][:8])] = e['problem_id']
_log(f'token_to_sid: {len(token_to_sid)} entries')

# content_tail (last 20 chars of decoded chosen response, for per-micro-batch logging)
for e in examples:
    try:
        e['chosen_tail']   = tokenizer.decode(e['chosen_ids'][-20:],   skip_special_tokens=False)[-20:]
        e['rejected_tail'] = tokenizer.decode(e['rejected_ids'][-20:], skip_special_tokens=False)[-20:]
    except Exception:
        e['chosen_tail'] = ''; e['rejected_tail'] = ''

# ── Precompute reference logp (one eval pass with adapter weights frozen at SFT seed) ──
# We DON'T disable the adapter here — the seed-adapter IS our reference policy.
# We compute logp on chosen and rejected sequences once, cache on CPU.
print('\n=== Precomputing reference logp (one pass over chosen+rejected) ===')
def _make_padded(ids_list, mask_list):
    """Returns pre-shifted (input, target, target_weight, attn_mask) tensors.
    input = ids[:-1] (pad to maxL-1). target = ids[1:]. weight = mask[1:]."""
    n = len(ids_list)
    maxL = max(len(x) for x in ids_list)
    Lm1 = maxL - 1
    inp     = torch.zeros(n, Lm1, dtype=torch.long,    device='cuda')
    tgt     = torch.zeros(n, Lm1, dtype=torch.long,    device='cuda')
    tgt_w   = torch.zeros(n, Lm1, dtype=torch.float32, device='cuda')
    attn    = torch.zeros(n, Lm1, dtype=torch.long,    device='cuda')
    for i, (ids, msk) in enumerate(zip(ids_list, mask_list)):
        L = len(ids)
        inp[i,   :L-1] = torch.tensor(ids[:-1], dtype=torch.long)
        tgt[i,   :L-1] = torch.tensor(ids[1:],  dtype=torch.long)
        tgt_w[i, :L-1] = torch.tensor(msk[1:],  dtype=torch.float32)
        attn[i,  :L-1] = 1
    return inp, tgt, tgt_w, attn

@torch.no_grad()
def _seq_logp(input_ids_list, mask_list, batch_size=MICRO_BATCH_SIZE):
    """Return per-example sum-of-completion-logp. Pre-shifts (input, target, weight)
    like SFT's data prep: input=ids[:-1], target=ids[1:], weight=msk[1:].
    Processes inputs in batches of `batch_size` for speed.

    NOTE 1: do NOT call model.eval() — that would set self.training=False on every submodule
    and disable Unsloth's smart GC autograd Function for the remainder of the run.

    NOTE 2: precompute MUST use the SAME bf16 autocast context that Phase 3 with-grad forwards
    use. Otherwise ref_logp ≠ policy_logp at step 0, producing a spurious step-0 gradient."""
    out = []
    for i in range(0, len(input_ids_list), batch_size):
        batch_ids  = input_ids_list[i:i+batch_size]
        batch_msks = mask_list[i:i+batch_size]
        inp, tgt, tgt_w, attn = _make_padded(batch_ids, batch_msks)
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _ = model(input_ids=inp, attention_mask=attn, labels=tgt, use_cache=False)
            ptc = model._cached_per_token_ce  # (B, T-1)
            logp = -(ptc * tgt_w).sum(dim=-1).detach().cpu()  # (B,)
        for j in range(logp.shape[0]):
            out.append(logp[j])
        del inp, tgt, tgt_w, attn, ptc
        torch.cuda.empty_cache()
    return torch.stack(out)

t_pre = time.time()
ref_chosen_logps   = _seq_logp([e['chosen_ids']   for e in examples], [e['chosen_mask']   for e in examples])
ref_rejected_logps = _seq_logp([e['rejected_ids'] for e in examples], [e['rejected_mask'] for e in examples])
_log(f'Precompute done in {time.time()-t_pre:.1f}s  smi={_smi()}M')

# ── Build batches ──────────────────────────────────────────────────
import random as _random
rng = _random.Random(SHUFFLE_SEED)
indices = list(range(len(examples)))
if STRESS_TEST:
    # Routine §1.2: no shuffle for stress (so user can inspect which sample caused issues)
    pass
else:
    rng.shuffle(indices)
batches = []
for i in range(0, len(indices), BATCH_SIZE):
    batches.append(indices[i:i + BATCH_SIZE])
num_steps = min(3, len(batches)) if STRESS_TEST else len(batches)
_log(f'Steps: {num_steps}  batch_size={BATCH_SIZE}')

# Save-step schedule per routine §2.4A/B
# A: full-checkpoint save_steps = max(25, round(total_steps/5/25)*25)
SAVE_STEPS_FULL = max(25, round(num_steps / 5 / 25) * 25)
# B: adapter-only soup saves at 20/60/80/85/90/95/100%
ADAPTER_SAVE_STEPS = sorted({round(num_steps * p) for p in (0.20, 0.60, 0.80, 0.85, 0.90, 0.95, 1.00)})
ADAPTER_SAVE_STEPS = [s for s in ADAPTER_SAVE_STEPS if s > 0]
_log(f'SAVE_STEPS_FULL={SAVE_STEPS_FULL}  ADAPTER_SAVE_STEPS={ADAPTER_SAVE_STEPS}')

# FIFO=1 queue for full checkpoints
import shutil
_ckpt_queue = []
def _fifo_cleanup():
    while len(_ckpt_queue) > 1:
        old = _ckpt_queue.pop(0)
        if os.path.isdir(old):
            shutil.rmtree(old)
            _log(f'  FIFO: deleted {os.path.basename(old)}')

# Resume from latest checkpoint (skip if STRESS_TEST or FRESH_START)
_resume_step = 0
_resume_opt_state = None
if not STRESS_TEST and not FRESH_START:
    from safetensors.torch import load_file as _load_ckpt
    _ckpts = sorted(
        [d for d in os.listdir(OUTPUT_DIR)
         if d.startswith('checkpoint-') and os.path.isdir(os.path.join(OUTPUT_DIR, d))],
        key=lambda x: int(x.split('-')[1].split('_')[0])
    )
    if _ckpts:
        _latest = _ckpts[-1]
        _latest_dir = os.path.join(OUTPUT_DIR, _latest)
        _resume_step = int(_latest.split('-')[1].split('_')[0])
        _log(f'Resuming from checkpoint: {_latest} (step {_resume_step})')
        _adapter_state = _load_ckpt(os.path.join(_latest_dir, 'adapter_model.safetensors'))
        _renamed = {}
        for _k, _v in _adapter_state.items():
            _nk = _k
            for _sfx in ('lora_A.weight', 'lora_B.weight'):
                if _nk.endswith('.' + _sfx):
                    _nk = _nk[:-len(_sfx)] + _sfx.replace('.weight', '.default.weight')
                    break
            _renamed[_nk] = _v
        model.load_state_dict(_renamed, strict=False)
        _ts_path = os.path.join(_latest_dir, 'training_state.pt')
        if os.path.exists(_ts_path):
            _ts = torch.load(_ts_path, map_location='cpu', weights_only=False)
            _resume_opt_state = _ts.get('optimizer_state_dict')
            _log(f'  Loaded optimizer state from {os.path.basename(_ts_path)}')
        _ckpt_queue.append(_latest_dir)

# ── Manual training loop (SFT-style, NO TRL) ───────────────────────
optimizer = None
device = next(model.parameters()).device
# _make_padded defined above (before _seq_logp uses it)

def _policy_logp_no_grad(inp, tgt, tgt_w, attn):
    """Single inference_mode forward; return per-example sum of completion logp.
    Wrapped in bf16 autocast to match Phase 3's with-grad forward (and the ref precompute)."""
    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        _ = model(input_ids=inp, attention_mask=attn, labels=tgt, use_cache=False)
        ptc = model._cached_per_token_ce  # (B, T-1), same shape as tgt
        return -(ptc * tgt_w).sum(dim=-1).detach().clone()

_log(f'\n=== Starting DPO training ===')
t0 = time.time()
step = _resume_step
torch.cuda.reset_peak_memory_stats()
for step_idx in range(_resume_step, num_steps):
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()   # so per-step peak reflects only this step's activity
    batch_idx = batches[step_idx]
    n_pairs = len(batch_idx)

    # Init optimizer lazily on first step (no Adam state allocated until needed)
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0,
        )
        if _resume_opt_state is not None:
            optimizer.load_state_dict(_resume_opt_state)
            _log(f'  Optimizer state loaded, resuming at step {_resume_step + 1}')
            _resume_opt_state = None

    # Linear LR decay
    lr = LEARNING_RATE * (1 - step_idx / num_steps)
    for pg in optimizer.param_groups: pg['lr'] = lr

    # Accumulate over each pair (micro-batch = 1 pair = 1 chosen forward + 1 rejected forward)
    smi_before = _smi()
    step_metrics = {'loss': 0.0, 'chosen_reward': 0.0, 'rejected_reward': 0.0, 'n': 0}

    # Process the batch in micro-batches of MICRO_BATCH_SIZE pairs each
    for mb_start in range(0, n_pairs, MICRO_BATCH_SIZE):
        mb_end = min(mb_start + MICRO_BATCH_SIZE, n_pairs)
        mb_ex_idx = batch_idx[mb_start:mb_end]
        mb_exs    = [examples[i] for i in mb_ex_idx]
        mb_n      = len(mb_exs)

        c_inp, c_tgt, c_tgt_w, c_attn = _make_padded(
            [e['chosen_ids']   for e in mb_exs], [e['chosen_mask']   for e in mb_exs])
        r_inp, r_tgt, r_tgt_w, r_attn = _make_padded(
            [e['rejected_ids'] for e in mb_exs], [e['rejected_mask'] for e in mb_exs])

        # Routine §2.3: per-micro-batch logging
        for i, ex in enumerate(mb_exs):
            _log(f'>> step={step+1} mb[{mb_start+i}] sample_id={ex["problem_id"]} '
                 f'padded_c={c_inp.shape[1]} padded_r={r_inp.shape[1]} smi={_smi()}M '
                 f'chosen_tail: {ex["chosen_tail"]!r}  rejected_tail: {ex["rejected_tail"]!r}')

        # ── Phase 1: per-sample current policy logp (no autograd graph) ──
        logp_c = _policy_logp_no_grad(c_inp, c_tgt, c_tgt_w, c_attn)  # (mb_n,)
        logp_r = _policy_logp_no_grad(r_inp, r_tgt, r_tgt_w, r_attn)  # (mb_n,)

        ref_c = torch.stack([ref_chosen_logps[i]   for i in mb_ex_idx]).to(device)  # (mb_n,)
        ref_r = torch.stack([ref_rejected_logps[i] for i in mb_ex_idx]).to(device)
        delta = (logp_c - ref_c) - (logp_r - ref_r)            # (mb_n,)
        s     = torch.sigmoid(-BETA * delta).detach()           # (mb_n,)

        for j in range(mb_n):
            step_metrics['loss']            += -F.logsigmoid(BETA * delta[j]).item()
            step_metrics['chosen_reward']   += float((BETA * (logp_c[j] - ref_c[j])).item())
            step_metrics['rejected_reward'] += float((BETA * (logp_r[j] - ref_r[j])).item())
            step_metrics['n']               += 1

        # ── Phase 3: with-grad B=mb_n forward + immediate backward, per branch ──
        # Per-sample loss: L_i = -log σ(β·δ_i). dL_i/dlogp_c_i = -β·s_i.
        # branch_loss = Σ_i (coef_i · logp_i) / n_pairs.
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            model(input_ids=c_inp, attention_mask=c_attn, labels=c_tgt, use_cache=False)
            ptc_c = model._cached_per_token_ce               # (mb_n, T-1) WITH GRAD
            logp_c_grad = -(ptc_c * c_tgt_w).sum(dim=-1)      # (mb_n,) per-sample logp WITH GRAD
            branch_loss_c = (-BETA * s * logp_c_grad).sum() / n_pairs
        branch_loss_c.backward()
        del ptc_c, logp_c_grad, branch_loss_c
        torch.cuda.empty_cache()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            model(input_ids=r_inp, attention_mask=r_attn, labels=r_tgt, use_cache=False)
            ptc_r = model._cached_per_token_ce
            logp_r_grad = -(ptc_r * r_tgt_w).sum(dim=-1)
            branch_loss_r = (BETA * s * logp_r_grad).sum() / n_pairs
        branch_loss_r.backward()
        del ptc_r, logp_r_grad, branch_loss_r
        torch.cuda.empty_cache()

    _tie_grads()
    # TRL DPOTrainer default is max_grad_norm=1.0; standard DPO recipes use 1.0-10.0.
    # 1e9 = effectively disabled, was wrong.
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    step += 1
    loss_mean   = step_metrics['loss']            / max(1, step_metrics['n'])
    chosen_r    = step_metrics['chosen_reward']   / max(1, step_metrics['n'])
    rejected_r  = step_metrics['rejected_reward'] / max(1, step_metrics['n'])
    smi_after = _smi()
    # True peak this step (across all forwards/backwards/opt.step) from PyTorch's allocator
    peak_alloc_gb    = torch.cuda.max_memory_allocated()    / 1e9
    peak_reserved_gb = torch.cuda.max_memory_reserved()     / 1e9
    elapsed = time.time() - t0
    eta     = (elapsed / step) * (num_steps - step) if step > 0 else 0
    _log(f'>> step={step}/{num_steps}  loss={loss_mean:.4f}  '
         f'chosen_r={chosen_r:+.4f}  rejected_r={rejected_r:+.4f}  '
         f'grad_norm={float(grad_norm):.4f}  lr={lr:.2e}  '
         f'smi_before={smi_before}M  smi_after={smi_after}M  '
         f'peak_alloc={peak_alloc_gb:.2f}G  peak_reserved={peak_reserved_gb:.2f}G  '
         f'elapsed={elapsed/60:.1f}m  eta={eta/60:.1f}m')

    if not STRESS_TEST and (step % SAVE_STEPS == 0 or step == num_steps):
        ckpt = os.path.join(OUTPUT_DIR, f'checkpoint-{step}_loss{loss_mean:.4f}')
        os.makedirs(ckpt, exist_ok=True)
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        torch.save({'optimizer_state_dict': optimizer.state_dict(), 'step': step},
                   os.path.join(ckpt, 'training_state.pt'))
        _log(f'  Checkpoint saved: {os.path.basename(ckpt)}')

elapsed = time.time() - t0
_log(f'\nTraining done. Time: {elapsed/3600:.2f} hrs ({elapsed/60:.1f} min)')
_log(f'Peak smi: {_smi()}M')

if STRESS_TEST:
    _log('STRESS TEST COMPLETE')
    sys.exit(0)

# ── Final save ─────────────────────────────────────────────────────
final_dir = os.path.join(OUTPUT_DIR, f'final_{_ts}')
os.makedirs(final_dir, exist_ok=True)
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
_log(f'Final adapter saved: {final_dir}')
