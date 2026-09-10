#!/usr/bin/env python3
"""GRPO on Nemotron-3-Nano-30B-A3B, seeded from SFT 0.86.

Built by copying study/2605270226_sft_newdata_cryptnumeq_outproj_rtx6000/train_cryptnumeq_outproj.py
verbatim through "Cast LoRA params to fp32", then replacing the manual SFT loop with
TRL GRPOTrainer + Unsloth fast_inference (vLLM) for rollouts.

Same SFT levers preserved (NON-NEGOTIABLE — these gave 0.84 → 0.86):
  - bf16 base, no quantization
  - USE_MEM_EFF=0  →  mixer.training=False  →  out_proj LoRA actively trains
  - LoRA r=32 α=32, target_modules: q/k/v/o + up/down/in/out + lm_head
  - LoRA params cast to fp32, base bf16
  - Manual lm_head LoRA injection

GRPO additions:
  - fast_inference=True               (vLLM rollout)
  - GRPOConfig: beta=0.02             (small KL anchor — research shows β=0 catastrophically forgets;
                                       see arxiv 2509.07430. Reverse-KL by default; forward-KL is better
                                       per the divergence paper but not a TRL toggle.)
  - GRPOConfig: loss_type='dapo'      (length-bias fix, NVIDIA's reference value)
  - GRPOConfig: importance_sampling_level='sequence'  (GSPO)
  - GRPOConfig: epsilon_high=0.28     (DAPO clip-higher)
  - 4-stack reward functions using \\boxed{} extractor + verify() — VERBATIM from 03_eval

Run via:
  STRESS_TEST=1 python train_grpo_v1.py       # 32 prompts, sanity check
  python train_grpo_v1.py                     # real run

Env overrides:
  SEED_ADAPTER=/root/autodl-tmp/sft_seed_086_moe_outproj
  DATA_CSV=/root/autodl-tmp/data/grpo_prompts.csv
  MOE_TIE=0           (default 0 = no-tie, matches SFT 0.86)
  USE_MEM_EFF=0       (default 0 = out_proj LoRA trains; DO NOT CHANGE without reason)
"""

import os, sys, time, gc, subprocess, types, math, logging, re
os.environ['TRANSFORMERS_NO_TF']      = '1'
os.environ['TRANSFORMERS_NO_FLAX']    = '1'
os.environ['CUDA_VISIBLE_DEVICES']    = '0'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['HF_HUB_OFFLINE']          = '1'
os.environ['TRANSFORMERS_OFFLINE']    = '1'
os.environ['UNSLOTH_VLLM_STANDBY']    = '1'   # +30% context length (Unsloth official trick)
os.environ['SWANLAB_API_KEY'] = dict(l.strip().split('=', 1) for l in open('kaggle.env', encoding='utf-8-sig') if '=' in l and not l.startswith('#'))['SWANLAB_API_KEY']
os.environ['SWANLAB_PROJECT']         = 'nemotron-grpo'

# ── Logging (single writer) ────────────────────────────────────────
LOG_FILE = '/root/autodl-tmp/train_grpo_log.txt'
_log_fh = logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8')
_log_sh = logging.StreamHandler(sys.stdout)
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[_log_fh, _log_sh])
_logger = logging.getLogger('grpo')
def _log(msg): _logger.info(msg)

# ── Config ─────────────────────────────────────────────────────────
LORA_RANK          = 32
LORA_ALPHA         = 32
LORA_DROPOUT       = 0.0

# Sequence budget — RL needs completion room for CoT + \boxed{}
MAX_PROMPT_LEN     = 512
MAX_COMPLETION_LEN = 3584
MAX_SEQ_LEN        = MAX_PROMPT_LEN + MAX_COMPLETION_LEN   # 4096

# Batching (Unsloth auto-bumps per_device_train_batch_size to match num_generations)
PER_DEVICE_BSZ     = 1
GRAD_ACCUM         = 4
NUM_GENERATIONS    = 4
TOTAL_OPTIM_STEPS  = 300

# Optimizer
LEARNING_RATE      = 5e-6     # Unsloth GRPO consensus (4 of 6 notebooks use this)
WARMUP_RATIO       = 0.1
LR_SCHEDULER_TYPE  = 'cosine'
WEIGHT_DECAY       = 0.001
MAX_GRAD_NORM      = 1.0

# GRPO algorithm (Unsloth notebooks all leave these at defaults; we set them explicitly
# based on: deep-research workflow + 6-notebook code study + corrections from forgetting research)
#
# WHY beta > 0 (NOT 0): research is unambiguous that β=0 causes catastrophic forgetting on
# out-of-domain prompts (Choice of Divergence paper, arxiv 2509.07430; observed Pass@k drop to
# ~85% of starting capability). NVIDIA's official DAPO recipe uses β=0 but trains on a single
# domain (DAPO-Math-17k); we have 9 categories to preserve. β=0.02 is mid-Tülu 3 range —
# small enough to allow learning, big enough to actually anchor.
#
# TRL uses reverse-KL by default. Forward-KL / JS would be better per arxiv 2509.07430
# (mode-covering vs mode-seeking) but isn't a TRL toggle. If we see drift, this is the
# next lever to swap.
#
# Memory: PEFT + Unsloth handles the reference policy by disable_adapter() on the base
# model, NOT a separate 60 GB copy. The cost is one extra forward pass per step, not
# +60 GB VRAM. If we OOM we trim max_completion_length.
GRPO_BETA               = 0.02         # KL coefficient. 0 = no anchor (causes forgetting); 0.02 = small but real
GRPO_LOSS_TYPE          = 'dapo'       # DAPO: uniform per-token aggregation (length-bias fix)
GRPO_IS_LEVEL           = 'sequence'   # GSPO: sequence-level importance sampling (vs GRPO token-level)
GRPO_EPSILON_HIGH       = 0.28         # DAPO clip-higher (recommended by DAPO paper)
GRPO_TEMPERATURE        = 1.0          # diversity for exploration (Theorem 3 requirement)
GRPO_TOP_P              = 1.0
GRPO_TOP_K              = -1
GRPO_MIN_P              = 0.1          # match Qwen3 FP8 notebook

# Architecture switches (defaults match SFT 0.86)
MOE_TIE_WEIGHTS    = os.environ.get('MOE_TIE',     '0') == '1'   # 0 = no-tie (matches SFT 0.86)
USE_MEM_EFF_PATH   = os.environ.get('USE_MEM_EFF', '0') == '1'   # 0 = out_proj LoRA TRAINS

TARGET_MODULES = [
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'up_proj', 'down_proj', 'in_proj', 'out_proj',
    'lm_head',
]

# ── Paths ──────────────────────────────────────────────────────────
MODEL_PATH        = '/root/autodl-tmp/Nemotron-3-Nano-30B-A3B'
# Seed = SFT crypt10k_FULL_ep2_contS3 (submission zip unzipped on instance; lm_head keys
# carry the submission-time backbone.lm_head rename which we reverse at load below).
# Local source: study/2606111516_sft_crypt10k_FULL_ep2_contS3_rtx6000/260612_0130_submission/
#               submission_crypt10k_FULL_ep2_contS3_notie_outproj.zip  (3.58 GB — user uploads)
SEED_ADAPTER_PATH = os.environ.get('SEED_ADAPTER', '/root/autodl-tmp/sft_seed_crypt10k_FULL_ep2_contS3')
# Dataset built by _build_dataset.py: 928 rows, 4 categories, oversampling>=1 (user: ov=0 rows
# are the too-long AUG puzzles, excluded), GT-match=True.
# Columns: id, category, prompt, answer, gold_cot, oversampling, cot_token_len
DATA_CSV          = os.environ.get('DATA_CSV',     '/root/autodl-tmp/data/grpo_dataset_cryptEq.csv')

# CATEGORIES env: 'both' (default) | 'crypt' | 'eq' — allows staged runs without code change
_CAT_MODE = os.environ.get('CATEGORIES', 'both')
KEEP_CATEGORIES = {
    'crypt': {'cryptarithm_deduce', 'cryptarithm_guess'},
    'eq':    {'equation_numeric_deduce', 'equation_numeric_guess'},
    'both':  {'cryptarithm_deduce', 'cryptarithm_guess',
              'equation_numeric_deduce', 'equation_numeric_guess'},
}[_CAT_MODE]

# Launch-time length filter: drop puzzles whose GOLD CoT wouldn't fit the completion budget
# (the model can't be expected to solve in fewer tokens than the deterministic solver needs;
# keeping over-budget puzzles produces truncated all-negative groups = wasted/noisy steps).
# Fit table for this dataset (928 rows): 3584→79.3%, 4096→84.4%, 5120→91.4%, 7680→100%
COT_LEN_MARGIN = 256   # gold CoT must fit MAX_COMPLETION_LEN - margin

_MOE_MODE   = 'fixtie' if MOE_TIE_WEIGHTS else 'notie'
_MEM_SUFFIX = '' if USE_MEM_EFF_PATH else '_outproj'
OUTPUT_DIR  = f'/root/autodl-tmp/adapter_output_grpo_{_MOE_MODE}{_MEM_SUFFIX}'
os.makedirs(OUTPUT_DIR, exist_ok=True)
FINAL_LORA_PATH = os.path.join(OUTPUT_DIR, f'final_grpo_{_MOE_MODE}{_MEM_SUFFIX}_lora')

PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

STRESS_TEST = bool(os.environ.get('STRESS_TEST'))

from datetime import datetime, timezone, timedelta
_cdt = timezone(timedelta(hours=-5))
_ts  = datetime.now(_cdt).strftime('%y%m%d_%H%M')
RUN_NAME = f'{_ts}_grpo_{_MOE_MODE}{_MEM_SUFFIX}_b{GRPO_BETA}_lr{LEARNING_RATE}_g{NUM_GENERATIONS}x{GRAD_ACCUM}_seq{MAX_SEQ_LEN}_rtx6000'

# ── Stub mamba3 (verbatim from SFT — required for our mamba_ssm version) ──
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
from peft import LoraConfig
from peft.tuners.lora import Linear as LoraLinear
import pandas as pd

print(f'mamba_ssm: {mamba_ssm.__version__}')
print(f'PyTorch:   {torch.__version__}')
print(f'GPU:       {torch.cuda.get_device_name(0)}')
print(f'VRAM:      {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

def _smi():
    try:
        return int(subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True).stdout.strip())
    except Exception:
        return 0

# ── Load base model (bf16 + vLLM fast_inference) ─────────────────
print('\n=== Loading bf16 model via Unsloth (fast_inference=True) ===')
gc.collect(); torch.cuda.empty_cache()
t_load = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name           = MODEL_PATH,
    max_seq_length       = MAX_SEQ_LEN,
    load_in_4bit         = False,
    load_in_8bit         = False,
    full_finetuning      = False,
    trust_remote_code    = True,
    attn_implementation  = 'eager',
    dtype                = torch.bfloat16,
    fast_inference       = True,             # vLLM rollout engine
    max_lora_rank        = LORA_RANK,
    gpu_memory_utilization = 0.7,            # leave 30% for KV cache + training activations
)
print(f'Model loaded in {time.time() - t_load:.1f}s  smi={_smi()}M')

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Wrap in LoRA (verbatim from SFT) ─────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_RANK,
    target_modules = TARGET_MODULES,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = LORA_DROPOUT,
    bias           = 'none',
    use_gradient_checkpointing = 'unsloth',
    random_state   = 42,
)
FastLanguageModel.for_training(model)

# ── Patch Mamba CUDA fast path (verbatim) ────────────────────────
nemotron_mod = None
for _name, _m in sys.modules.items():
    if 'modeling_nemotron_h' in _name and hasattr(_m, 'is_fast_path_available'):
        nemotron_mod = _m; break
if nemotron_mod is not None:
    nemotron_mod.is_fast_path_available = True
    print('Patched is_fast_path_available = True')

# out_proj training (verbatim — same lever that gave 0.84 → 0.86)
if not USE_MEM_EFF_PATH:
    _n_mix = 0
    for _mod in model.modules():
        if hasattr(_mod, 'cuda_kernels_forward'):   # Mamba2 mixer
            _mod.training = False
            _n_mix += 1
    print(f'Forced unfused else-branch on {_n_mix} Mamba mixers (out_proj LoRA will train)')

# ── Manually add lm_head LoRA (verbatim) ─────────────────────────
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

# ── Load SFT 0.86 seed adapter weights into the LoRA wrappers ────
from safetensors.torch import load_file
print(f'\n=== Loading seed adapter from {SEED_ADAPTER_PATH} ===')
_seed_st = os.path.join(SEED_ADAPTER_PATH, 'adapter_model.safetensors')
if not os.path.exists(_seed_st):
    raise FileNotFoundError(f'Seed adapter not found: {_seed_st}. Set SEED_ADAPTER env var.')
seed_tensors = load_file(_seed_st)

# Reverse the lm_head key rename done at submission build time:
#   base_model.model.backbone.lm_head.*   →   base_model.model.lm_head.*
seed_tensors = {
    k.replace('base_model.model.backbone.lm_head.', 'base_model.model.lm_head.'): v
    for k, v in seed_tensors.items()
}
# PEFT save strips '.default.' from LoRA keys; load_state_dict needs them back
seed_tensors_v2 = {}
for k, v in seed_tensors.items():
    if   '.lora_A.weight' in k and '.lora_A.default.weight' not in k:
        k = k.replace('.lora_A.weight', '.lora_A.default.weight')
    elif '.lora_B.weight' in k and '.lora_B.default.weight' not in k:
        k = k.replace('.lora_B.weight', '.lora_B.default.weight')
    seed_tensors_v2[k] = v
missing, unexpected = model.load_state_dict(seed_tensors_v2, strict=False)
n_loaded = len(seed_tensors_v2) - len(unexpected)
_missing_lora = [m for m in missing if '.lora_' in m]
print(f'Seed adapter loaded: {n_loaded} tensors  '
      f'(missing-lora={len(_missing_lora)}, unexpected={len(unexpected)})')
if _missing_lora[:3]:
    print(f'  sample missing-lora: {_missing_lora[:3]}')
if unexpected[:3]:
    print(f'  sample unexpected:   {unexpected[:3]}')

# ── Cast LoRA params to fp32 (verbatim from SFT) ─────────────────
for name, param in model.named_parameters():
    if '.lora_' in name:
        param.data = param.data.to(torch.float32)
print('Cast LoRA params to fp32')

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f'Trainable: {trainable/1e6:.1f}M / total {total/1e9:.1f}B  smi={_smi()}M')

# ── Reward functions ─────────────────────────────────────────────
# Per-step distillation reward stack (designed via deep-read workflow on
# solver code + SFT training CSV + model eval traces). See:
#   2605312353_grpo_unsloth_codestudy_rtx6000/WORKFLOW_REPORT.md
# 7 reward functions targeting the diagnosed SFT failure modes:
#   PRIMARY    : reward_final_answer (Kaggle metric verbatim; +2.5 / −1.5 on correctness)
#   PROCESS    : reward_conclusion_semantic (C2 'unknown'-in-Summary mode = 71% of crypt failures)
#   PROCESS    : reward_solve_query_remap (D2 boxed-letters mode)
#   SCAFFOLD   : reward_skeleton_alignment, reward_closed_vocab_discipline
#   GUARDRAIL  : reward_termination_quality (truncation + duplicate-box)
#   RESERVED   : reward_solver_alignment (OFF by default, gold_cot SequenceMatcher bonus)
# Total range: ~[-10.0, +10.0]; primary outcome ~6× scaffolding.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # grpo_rewards.py ships next to this script
from grpo_rewards import REWARD_FUNCS, extract_final_answer, verify  # noqa: E402

# ── Dataset ─────────────────────────────────────────────────────
print(f'\n=== Loading prompts from {DATA_CSV} ===')
if not os.path.exists(DATA_CSV):
    raise FileNotFoundError(f'Prompts CSV not found: {DATA_CSV}. Set DATA_CSV env var.')

import csv as _csv
_csv.field_size_limit(2**31 - 1)
with open(DATA_CSV, encoding='utf-8', newline='') as f:
    rows = list(_csv.DictReader(f))
print(f'CSV rows: {len(rows)}')
print(f'CSV columns: {list(rows[0].keys()) if rows else "<empty>"}')

from datasets import Dataset

# Helpers to extract per-prompt features the reward funcs need (forwarded via **kwargs by TRL)
_EX_RE = re.compile(r'\bEX(\d+)\b')

def _parse_ex_count(prompt_text: str) -> int:
    """Count of EX{i} examples referenced in the prompt."""
    ix = [int(m.group(1)) for m in _EX_RE.finditer(prompt_text)]
    return max(ix) if ix else 0

def _parse_symbol_pool(prompt_text: str) -> str:
    """Set of unique non-whitespace puzzle symbols appearing on EX/QUERY lines.
    Used by reward_final_answer D2 to detect 'boxed raw letters' failure mode in cryptarithm.
    """
    pool = set()
    for line in prompt_text.splitlines():
        # crude but effective: collect single-char tokens from EX*/QUERY lines
        if 'EX' in line or 'QUERY' in line or '=' in line:
            for ch in line:
                if not ch.isspace() and ch not in '=,;:.()[]{}':
                    pool.add(ch)
    return ''.join(sorted(pool))

# Pre-render each prompt with the chat template + suffix.
# Dataset is pre-filtered by _build_dataset.py (4 cats, GT-match=True, gold_cot present);
# here we apply the launch-time filters: prompt length + gold-CoT-fits-completion-budget.
COT_LEN_BUDGET   = MAX_COMPLETION_LEN - COT_LEN_MARGIN
formatted        = []
n_dropped_long   = 0
n_dropped_cat    = 0
n_dropped_cotlen = 0
for row in rows:
    if not row.get('answer') or not row.get('prompt'):
        continue
    cat = row.get('category', '')
    if cat not in KEEP_CATEGORIES:
        n_dropped_cat += 1; continue
    cot_len = int(float(row.get('cot_token_len', 0) or 0))
    if cot_len > COT_LEN_BUDGET:
        n_dropped_cotlen += 1; continue
    user_content = row['prompt'] + PROMPT_SUFFIX
    prompt_text  = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': user_content}],
        tokenize             = False,
        add_generation_prompt= True,
    )
    n_tokens = len(tokenizer.encode(prompt_text))
    if n_tokens > MAX_PROMPT_LEN:
        n_dropped_long += 1; continue
    formatted.append({
        'id'          : row.get('id', ''),
        'category'    : cat,
        'prompt'      : prompt_text,
        'answer'      : str(row['answer']),
        'gold_cot'    : row['gold_cot'],
        'gold_answer' : str(row['answer']),                          # alias used by reward fns
        'oversampling': int(float(row.get('oversampling', 1) or 1)),
        'cot_token_len': cot_len,
        'ex_count'    : _parse_ex_count(row['prompt']),
        'symbol_pool' : _parse_symbol_pool(row['prompt']),
    })
print(f'\nFilter summary:')
print(f'  rows in CSV                : {len(rows)}')
print(f'  dropped (not in cat)       : {n_dropped_cat}')
print(f'  dropped (cot>{COT_LEN_BUDGET}tok): {n_dropped_cotlen}')
print(f'  dropped (prompt long)      : {n_dropped_long}')
print(f'  kept                       : {len(formatted)}')

from collections import Counter
_cat_counts = Counter(r['category'] for r in formatted)
print('Category breakdown:')
for cat, n in sorted(_cat_counts.items(), key=lambda x: -x[1]):
    print(f'  {cat:<24} {n}')

if STRESS_TEST:
    # Routine §1 rules: sort WORST-CASE FIRST (longest gold CoT = the prompts most likely
    # to drive rollouts to the full completion budget), take top-K, no shuffle.
    formatted.sort(key=lambda r: -r['cot_token_len'])
    formatted = formatted[:32]
    TOTAL_OPTIM_STEPS = 3          # routine: 2-4 steps
    print(f'STRESS_TEST: {len(formatted)} worst-case rows, {TOTAL_OPTIM_STEPS} steps, saves disabled')
    print(f'  cot_token_len of selected: {[r["cot_token_len"] for r in formatted[:8]]} ...')

dataset = Dataset.from_list(formatted)
print(f'Dataset: {len(dataset)} rows, columns={dataset.column_names}')

# ── GRPOConfig ──────────────────────────────────────────────────
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

vllm_sampling_params = SamplingParams(
    temperature                = GRPO_TEMPERATURE,
    top_p                      = GRPO_TOP_P,
    top_k                      = GRPO_TOP_K,
    min_p                      = GRPO_MIN_P,
    max_tokens                 = MAX_COMPLETION_LEN,
    seed                       = 42,
    stop                       = [tokenizer.eos_token],
    include_stop_str_in_output = True,
)

training_args = GRPOConfig(
    output_dir  = OUTPUT_DIR,
    run_name    = RUN_NAME,

    # Optimizer
    learning_rate     = LEARNING_RATE,
    weight_decay      = WEIGHT_DECAY,
    warmup_ratio      = WARMUP_RATIO,
    lr_scheduler_type = LR_SCHEDULER_TYPE,
    optim             = 'paged_adamw_8bit',
    max_grad_norm     = MAX_GRAD_NORM,

    # Batching
    per_device_train_batch_size = PER_DEVICE_BSZ,
    gradient_accumulation_steps = GRAD_ACCUM,
    num_generations             = NUM_GENERATIONS,
    max_prompt_length           = MAX_PROMPT_LEN,
    max_completion_length       = MAX_COMPLETION_LEN,
    max_steps                   = TOTAL_OPTIM_STEPS,

    # GRPO algorithm (the four knobs that move us from vanilla → GSPO+DAPO+no-KL)
    beta                       = GRPO_BETA,           # 0 = no KL → no ref model loaded
    loss_type                  = GRPO_LOSS_TYPE,      # 'dapo'
    importance_sampling_level  = GRPO_IS_LEVEL,      # 'sequence' = GSPO
    epsilon_high               = GRPO_EPSILON_HIGH,  # DAPO clip-higher
    temperature                = GRPO_TEMPERATURE,
    top_p                      = GRPO_TOP_P,
    top_k                      = GRPO_TOP_K,
    min_p                      = GRPO_MIN_P,

    # vLLM rollout
    vllm_sampling_params = vllm_sampling_params,

    # Saves + logging (stress: saves disabled per routine §1 rule 3)
    save_strategy    = 'no' if STRESS_TEST else 'steps',
    save_steps       = max(50, TOTAL_OPTIM_STEPS // 5),
    save_total_limit = 3,
    logging_steps    = 1,
    report_to        = 'none',   # swap to 'swanlab' once verified
)

# ── Trainer ─────────────────────────────────────────────────────
print('\n=== Building GRPOTrainer ===')
trainer = GRPOTrainer(
    model            = model,
    processing_class = tokenizer,
    reward_funcs     = REWARD_FUNCS,
    args             = training_args,
    train_dataset    = dataset,
)

# Routine §1 rule 6: snapshot out_proj LoRA-B before training so we can verify
# the unfused-path patch kept out_proj alive (else we're silently capped at 0.84)
_outproj_b_before = {
    n: p.detach().float().norm().item()
    for n, p in model.named_parameters()
    if 'out_proj' in n and 'lora_B' in n
}
print(f'out_proj lora_B params tracked: {len(_outproj_b_before)}  '
      f'norm[0..2]={[round(v,4) for v in list(_outproj_b_before.values())[:3]]}')

# ── Train ───────────────────────────────────────────────────────
print(f'\n=== Starting GRPO training ===')
print(f'  steps:       {TOTAL_OPTIM_STEPS}')
print(f'  prompts:     {len(dataset)}')
print(f'  batch shape: {NUM_GENERATIONS} generations × {GRAD_ACCUM} grad-accum = {NUM_GENERATIONS*GRAD_ACCUM} samples/optim step')
print(f'  GRPO knobs:  beta={GRPO_BETA}  loss={GRPO_LOSS_TYPE}  IS={GRPO_IS_LEVEL}  eps_hi={GRPO_EPSILON_HIGH}')
print(f'  Run name:    {RUN_NAME}')
print(f'  Output dir:  {OUTPUT_DIR}')
print(f'  smi_before:  {_smi()}M')

t_train = time.time()
trainer.train()
elapsed = time.time() - t_train
print(f'\nTraining done. Time: {elapsed/3600:.2f} hrs ({elapsed/60:.1f} min)')
print(f'Peak smi: {_smi()}M')
print(f'Peak torch alloc: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB  '
      f'reserved: {torch.cuda.max_memory_reserved()/1024**3:.2f} GB')

# Routine §1 rule 6: out_proj LoRA must have MOVED (gradient flowed through the
# unfused path).  If every lora_B norm is unchanged, the fused kernel folded it
# and we're silently capped at 0.84 — abort loudly.
_changed = 0
for n, p in model.named_parameters():
    if n in _outproj_b_before:
        if abs(p.detach().float().norm().item() - _outproj_b_before[n]) > 1e-9:
            _changed += 1
print(f'out_proj lora_B params changed by training: {_changed}/{len(_outproj_b_before)}')
if _outproj_b_before and _changed == 0:
    print('FATAL: out_proj LoRA did NOT train — fused Mamba kernel folded it. '
          'Check the mixer.training=False patch survived GRPOTrainer.')
    sys.exit(2)

# ── Save final LoRA ─────────────────────────────────────────────
print(f'\n=== Saving final LoRA to {FINAL_LORA_PATH} ===')
model.save_lora(FINAL_LORA_PATH)

# Llama 3.2 Advanced notebook safety check: assert no LoRA tensor is all-zero
from safetensors import safe_open
_final_st = os.path.join(FINAL_LORA_PATH, 'adapter_model.safetensors')
if os.path.exists(_final_st):
    with safe_open(_final_st, framework='pt') as f:
        all_zero = []
        for k in f.keys():
            t = f.get_tensor(k)
            if (t == 0).all():
                all_zero.append(k)
        if all_zero:
            print(f'WARNING: {len(all_zero)} LoRA tensors are all-zero (training silently failed?):')
            for k in all_zero[:5]: print(f'  {k}')
        else:
            print(f'All {len(f.keys())} LoRA tensors non-zero ✓')

print('\n=== DONE ===')
