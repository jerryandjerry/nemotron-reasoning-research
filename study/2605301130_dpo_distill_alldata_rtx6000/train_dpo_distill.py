import os as _os_early
# expandable_segments + garbage_collection_threshold defragments allocator at high seq_len;
# essential for 30B-MoE training that hits 93+ GB allocated with shared_experts + per-expert LoRA.
_os_early.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'

"""DPO distillation from cryptnumeq 0.85 SFT seed (Sub #24).

Goal: train the LoRA to prefer gold `solver_cot` over the seed model's own `raw_output`.
beta=0.05 (low) because the data audit showed 92.6% of "rejected" outputs are actually correct
- DPO collapses to soft distillation toward gold-CoT style rather than a correctness fix.

DATA FILTER (mitigates 3D-Properties dispersion-on-unseen failure mode, arXiv:2406.07327):
  - Drop categories with cryptnumeq val accuracy >= 98% (numeral/gravity/unit_conversion/cipher),
    EXCEPT keep 10% sampled (seed=42) as 'regularization anchors' to prevent drift.
  - Keep all rows from categories with real headroom (cryptarithm/bit_manipulation/equations).
  - Result: 2658 pairs (down from 6429); ~84% of budget concentrated on high-headroom rows.

Single-GPU, no DDP. TRL DPOTrainer with ref_model=None → PEFT disable_adapter (no second 30B).

Routine compliance (see 02_train/TRAINING_ROUTINE.md):
  - SwanLab logging, append-mode train log, gc+empty_cache at step boundaries
  - Resume-from-checkpoint (FRESH_START=1 to disable)  — REQUIRED for DPO per user
  - FIFO=1 resume checkpoints
  - Force-save final checkpoint + group artifacts into {yymmdd_hhmm}_submission/ on closeout
  - NO soup adapters (per user: not needed for DPO)
"""
import os, sys, time, gc, math, json, subprocess, types, re, datetime, zipfile, shutil
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Stub mamba3 (matches SFT seed env) ───────────────────────────────
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
print(f'unsloth: {unsloth.__version__}', flush=True)

import mamba_ssm, torch, pandas as pd
from peft import PeftModel
from safetensors.torch import load_file, save_file
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from transformers import TrainerCallback
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt
from trl.trainer.dpo_trainer import selective_log_softmax, disable_gradient_checkpointing, is_peft_model, use_adapter
import json as _json


# ── Chunked CCE: compute log p(label) per token WITHOUT materializing (B, T, V) logits ──
# vocab V=256000 makes the lm_head output 2 GB bf16 + 4 GB float32 for a single (1, 4096) seq.
# We chunk along the sequence dim and use torch.utils.checkpoint so backward recomputes the
# per-chunk logits — peak ~ chunk × V × 6 bytes (bf16 + float32 transient) instead of T × V.
# At chunk=512: peak ≈ 768 MB instead of 6 GB. Math identical to selective_log_softmax.
def _cce_chunk(h, w, l):
    logits = F.linear(h, w)                       # (B, chunk, V) bf16
    logp   = F.log_softmax(logits.float(), dim=-1)  # (B, chunk, V) float32 — transient
    safe_l = l.clamp(min=0)
    return logp.gather(-1, safe_l.unsqueeze(-1)).squeeze(-1)  # (B, chunk)

def chunked_token_logp(hidden, lm_head_weight, labels, chunk_size=512):
    B, T, H = hidden.shape
    chunks = []
    for i in range(0, T, chunk_size):
        h = hidden[:, i:i+chunk_size, :].contiguous()
        l = labels[:, i:i+chunk_size].contiguous()
        if h.requires_grad:
            gathered = _ckpt.checkpoint(_cce_chunk, h, lm_head_weight, l, use_reentrant=False)
        else:
            gathered = _cce_chunk(h, lm_head_weight, l)
        chunks.append(gathered)
    return torch.cat(chunks, dim=-1)  # (B, T) float32


# ── Custom DPOTrainer that does chosen/rejected as TWO sequential forwards ──
# Default TRL DPOTrainer concatenates chosen+rejected into one (2, seq) batch and forwards
# the model ONCE — this stores activations for BOTH sequences simultaneously, giving us the
# OOM at ~95 GB peak on single 96 GB GPU. We override _compute_loss to do two sequential
# forwards: chosen first, release its activations, then rejected. Peak activation footprint
# halves. Wall-clock per step roughly doubles. DPO loss math is unchanged.
# Pre-condition: precompute_ref_log_probs=True (so the ref forward branch is skipped here).
class SequentialDPOTrainer(DPOTrainer):
    _seq_dpo_log_counter = 0   # so we can confirm in the log that this override is firing

    def _compute_loss(self, model, inputs, return_outputs):
        # FLUSH GPU allocator at start of each compute_loss — defrags between micro-batches
        # AND between precompute end + first training step. precompute runs 96 no_grad forwards
        # and may leave allocator blocks; first training step needs maximum headroom.
        import gc as _gc
        _gc.collect()
        torch.cuda.empty_cache()

        if SequentialDPOTrainer._seq_dpo_log_counter < 3:
            SequentialDPOTrainer._seq_dpo_log_counter += 1
            try:
                _alloc = torch.cuda.memory_allocated() / (1024**3)
                _reserved = torch.cuda.memory_reserved() / (1024**3)
                # RUNTIME diagnostic: confirm freeze still holds at training_step time.
                _n_train_rt = sum(_p.numel() for _p in model.parameters() if _p.requires_grad) / 1e6
                _log(f"  [SequentialDPO] _compute_loss called; input_ids shape = {tuple(inputs['input_ids'].shape)}; VRAM allocated={_alloc:.2f}G reserved={_reserved:.2f}G; trainable_at_runtime={_n_train_rt:.1f}M")
            except Exception: pass
        mode = "train" if self.model.training else "eval"
        device = self.accelerator.device

        input_ids       = inputs["input_ids"]
        attention_mask  = inputs["attention_mask"]
        completion_mask = inputs["completion_mask"]
        B = input_ids.shape[0]
        assert B % 2 == 0, f"Expected batch [chosen, rejected]; got odd B={B}"
        half = B // 2

        shift_labels          = input_ids[..., 1:].contiguous()
        shift_completion_mask = completion_mask[..., 1:].contiguous()

        # Pass-through optional fields (vision etc); will be sliced per-half.
        _extra_keys = ("token_type_ids","mm_token_type_ids","pixel_values",
                       "pixel_attention_mask","image_grid_thw","image_sizes","pixel_position_ids")
        def _half_kwargs(lo, hi):
            kw = {"input_ids": input_ids[lo:hi], "attention_mask": attention_mask[lo:hi], "use_cache": False}
            for k in _extra_keys:
                if k in inputs:
                    kw[k] = inputs[k][lo:hi]
            return kw

        # CCE patch via cut_cross_entropy.linear_cross_entropy — bypasses model.forward's
        # `.float()` upcast at modeling_nemotron_h.py:1717 (the 2 GB allocation that OOMs in
        # Phase 2 with_grad). Use torch.addmm for ONE-tensor merge instead of 3× transients.
        _peft = model
        _inner_clm = getattr(getattr(_peft, 'base_model', _peft), 'model', None) or _peft
        while not hasattr(_inner_clm, 'backbone') and hasattr(_inner_clm, 'model'):
            _inner_clm = _inner_clm.model
        if not getattr(_inner_clm, '_cce_forward_patched', False):
            from cut_cross_entropy import linear_cross_entropy as _lce
            _inner_clm_ref = _inner_clm
            _model_ref     = model
            _orig_forward_fn = _inner_clm.forward
            def _cce_forward(input_ids=None, attention_mask=None, labels=None, **kwargs):
                # No labels → defer to ORIGINAL forward (Test 35 proved Phase 1 no_grad fits
                # the .float() upcast in inference_mode). Avoid CCE/addmm overhead in Phase 1.
                if labels is None:
                    return _orig_forward_fn(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
                _bb_kwargs = {k: v for k, v in kwargs.items()
                              if k in ('position_ids', 'past_key_values', 'use_cache')}
                bb_out = _inner_clm_ref.backbone(
                    input_ids=input_ids, attention_mask=attention_mask, **_bb_kwargs)
                hidden = bb_out[0]
                lh = _inner_clm_ref.lm_head
                if hasattr(lh, 'base_layer'):
                    base_w  = lh.base_layer.weight
                    lA      = lh.lora_A['default'].weight
                    lB      = lh.lora_B['default'].weight
                    scaling = lh.scaling['default']
                    # Force allocator defrag just before the 1 GB (V, H) allocation —
                    # critical headroom in step 2+ where ~1 GB of paged_adamw_8bit state
                    # squats on GPU and the addmm has razor-thin margin.
                    torch.cuda.empty_cache()
                    lm_weight = torch.addmm(base_w, lB, lA, alpha=float(scaling), beta=1.0)
                else:
                    lm_weight = lh.weight
                shifted_hidden = hidden[:, :-1, :].contiguous()
                shifted_labels = labels[:, 1:].contiguous()
                per_token_ce = _lce(shifted_hidden, lm_weight, shifted_labels, reduction='none')
                loss = per_token_ce.mean()
                _model_ref._cached_per_token_ce = per_token_ce
                return loss
            _inner_clm._orig_forward_before_cce = _inner_clm.forward
            _inner_clm.forward = _cce_forward
            _inner_clm._cce_forward_patched = True
            _log('  [PATCH] NemotronHForCausalLM.forward -> CCE (linear_cross_entropy + addmm merge)')

        # ── GRADIENT DECOMPOSITION (truly sequential, SFT-equivalent memory) ──
        # DPO loss: L = -logsigmoid(β·δ), δ = (logp_c - logp_r) - (ref_c - ref_r)
        # dL/dlogp_c = -β·s, dL/dlogp_r = +β·s, where s = sigmoid(-β·δ).
        # So we can backward each branch independently with coefficient ±β·s,
        # keeping only ONE forward's autograd graph alive at a time.
        # Cost: 4 forwards/step (2 no_grad for values + 2 grad+backward), ~2x slower.

        ref_chosen_logps   = inputs["ref_chosen_logps"]    # (half,)
        ref_rejected_logps = inputs["ref_rejected_logps"]  # (half,)
        grad_accum = self.args.gradient_accumulation_steps

        # Phase 1: INFERENCE_MODE forwards → get logp values, no autograd graph.
        # Use ORIGINAL forward (no labels → patched forward delegates). Stock path's
        # 2 GB float32 logits upcast is fine in inference_mode (transient, no graph).
        from trl.trainer.dpo_trainer import selective_log_softmax as _sls
        def _logp_no_grad(lo, hi):
            with torch.inference_mode():
                kw  = _half_kwargs(lo, hi)
                out = model(**kw)
                shift_logits = out.logits[..., :-1, :].contiguous()
                shift_labels = kw['input_ids'][..., 1:].contiguous()
                ptlp = _sls(shift_logits, shift_labels)
                ptlp = ptlp * shift_completion_mask[lo:hi].to(ptlp.dtype)
                logp = ptlp.sum(dim=1).detach().clone()
                del out, shift_logits, ptlp
                return logp

        def _vram():
            import torch as _t
            return f"{_t.cuda.memory_allocated()/1024**3:.2f}G/{_t.cuda.memory_reserved()/1024**3:.2f}G"
        # Diagnostic: log micro-step 1 of each optimizer step (so we see how step 2+ baseline changes
        # after the optimizer adds Adam state). Use modulo grad_accum to detect "first micro-step of step N".
        _ga = self.args.gradient_accumulation_steps
        _is_first_microstep = ((SequentialDPOTrainer._seq_dpo_log_counter - 1) % max(1, _ga) == 0)
        _diag = STRESS_TEST and (_is_first_microstep or SequentialDPOTrainer._seq_dpo_log_counter <= 3)
        if _diag: _log(f"  [DPO-MEM] before Phase 1 chosen: {_vram()}")

        chosen_logp_val   = _logp_no_grad(0, half)
        torch.cuda.empty_cache()
        if _diag: _log(f"  [DPO-MEM] after Phase 1 chosen + empty_cache: {_vram()}")
        rejected_logp_val = _logp_no_grad(half, B)
        torch.cuda.empty_cache()
        if _diag: _log(f"  [DPO-MEM] after Phase 1 rejected + empty_cache: {_vram()}")

        # Phase 2: compute s coefficient off-graph
        chosen_logratios_v   = chosen_logp_val   - ref_chosen_logps
        rejected_logratios_v = rejected_logp_val - ref_rejected_logps
        delta_score_v = chosen_logratios_v - rejected_logratios_v
        s_val = torch.sigmoid(-self.beta * delta_score_v).detach()    # (half,)
        total_loss_val = -F.logsigmoid(self.beta * delta_score_v).mean().detach()

        # Phase 3: WITH-grad forwards, each followed by immediate backward.
        # branch_loss = coef * logp; HF's training_step will divide our returned loss by
        # grad_accum, so we pre-divide here ourselves and return a no-op dummy.
        def _grad_forward_backward(lo, hi, coef, tag):
            kw  = _half_kwargs(lo, hi)
            kw['labels'] = kw['input_ids']
            if _diag: _log(f"  [DPO-MEM] before Phase 3 {tag} forward: {_vram()}")
            _ = model(**kw)
            if _diag: _log(f"  [DPO-MEM] after  Phase 3 {tag} forward: {_vram()}")
            per_token_ce = model._cached_per_token_ce      # (B, T-1) WITH GRAD
            ptlp = -per_token_ce
            ptlp = ptlp * shift_completion_mask[lo:hi].to(ptlp.dtype)
            logp = ptlp.sum(dim=1)                          # (half,) with grad
            branch_loss = (coef.detach() * logp).mean() / grad_accum
            self.accelerator.backward(branch_loss)
            model._cached_per_token_ce = None
            if _diag: _log(f"  [DPO-MEM] after  Phase 3 {tag} backward: {_vram()}")
            return logp.detach()

        chosen_coef   = -self.beta * s_val   # dL/dlogp_c = -β·s
        chosen_logps  = _grad_forward_backward(0, half, chosen_coef, 'chosen')
        torch.cuda.empty_cache()
        # MoE expert LoRA frozen → trainable ~50 M, .grad ~200 MB. Rejected forward fits
        # at baseline + .grad (~62.3 GB) + 30 GB activations = 92.3 GB. No CPU offload
        # of chosen grads needed; they just accumulate naturally into rejected's via
        # the second accelerator.backward call.
        rejected_coef = self.beta * s_val    # dL/dlogp_r = +β·s
        rejected_logps = _grad_forward_backward(half, B, rejected_coef, 'rejected')
        torch.cuda.empty_cache()
        if _diag: _log(f"  [DPO-MEM] after Phase 3 rejected backward + sum: {_vram()}")

        # Metrics (use detached values from phase 3 logps which match the actual forwards)
        chosen_logratios   = chosen_logps   - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        self._metrics[mode]["logps/chosen"  ].append(self.accelerator.gather(chosen_logps  ).mean().item())
        self._metrics[mode]["logps/rejected"].append(self.accelerator.gather(rejected_logps).mean().item())
        self._metrics[mode]["rewards/chosen"  ].append(self.accelerator.gather(self.beta * chosen_logratios  ).mean().item())
        self._metrics[mode]["rewards/rejected"].append(self.accelerator.gather(self.beta * rejected_logratios).mean().item())
        self._metrics[mode]["rewards/margins" ].append(self.accelerator.gather(self.beta * (chosen_logratios - rejected_logratios)).mean().item())
        self._metrics[mode]["rewards/accuracies"].append(self.accelerator.gather((chosen_logratios > rejected_logratios).float()).mean().item())

        # Return value for HF Trainer: gradients are already accumulated above.
        # We return a tensor with the right value (total_loss) but whose .backward() is a no-op
        # (gradient of 0·param = 0 added to param.grad, no effect on accumulation).
        _first_grad_param = next(p for p in model.parameters() if p.requires_grad)
        dummy_loss = total_loss_val + 0.0 * _first_grad_param.sum()
        outputs = {'chosen_logps': chosen_logps, 'rejected_logps': rejected_logps}
        return (dummy_loss, outputs) if return_outputs else dummy_loss

print(f'mamba_ssm: {mamba_ssm.__version__}  PyTorch: {torch.__version__}  GPU: {torch.cuda.get_device_name(0)}', flush=True)

# ── Config (single source of truth) ──────────────────────────────────
MODEL_PATH        = '/root/autodl-tmp/Nemotron-3-Nano-30B-A3B'
SEED_ADAPTER_PATH = '/root/autodl-tmp/sft_seed_cryptnumeq/'
PREFER_CSV        = '/root/autodl-tmp/data/prefer_260527_huikang_NumericEq.csv'
REJECT_CSV        = '/root/autodl-tmp/data/reject_numericeq.csv'
OUTPUT_DIR        = '/root/autodl-tmp/dpo_distill_output'
LOG_FILE          = '/root/autodl-tmp/train_dpo_log.txt'

MAX_SEQ_LEN       = 4096   # User's minimum floor. saved_tensors_hooks offloads ALL saved-for-backward tensors (including Mamba's cuda_kernels_forward state) to CPU between forwards. Both forwards' state cached in RAM; GPU peak = one forward's footprint.
MAX_PROMPT_LEN    = 1024
BETA              = 0.05      # low: 92.6% of rejected was actually correct → distillation
LEARNING_RATE     = 5e-7      # Tülu-3 DPO default
PER_DEVICE_BATCH  = 1
GRAD_ACCUM        = 16
NUM_EPOCHS        = 1

# Per-routine save_steps formula: max(25, round(total_steps/5/25)*25). For 167 steps → 25.
SAVE_STEPS        = 25

PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

# Categories where cryptnumeq val acc >= 98% (no headroom to lift)
EASY_CATEGORIES_VAL_GE_98 = {'numeral', 'gravity', 'unit_conversion', 'cipher'}
ANCHOR_FRACTION = 0.10   # keep 10% of easy categories as regularization anchors

STRESS_TEST  = os.environ.get('STRESS_TEST', '0') == '1'
FRESH_START  = os.environ.get('FRESH_START',  '0') == '1'

SWANLAB_PROJECT  = 'nemotron-dpo'
SWANLAB_RUN_NAME = '260530_1130_dpo_distill_seed-cryptnumeq_alldata-filtered_b0.05_lr5e-7_bs16_seq8192_rtx6000'

# ── Logging helpers ──────────────────────────────────────────────────
def _log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, 'a') as f:
        f.write(str(msg) + '\n')

def _smi():
    try:
        return int(subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True).stdout.strip())
    except Exception:
        return 0

_log(f'=== train_dpo_distill.py START  STRESS_TEST={STRESS_TEST}  FRESH_START={FRESH_START} ===')

# ── 1. SwanLab init (skip on stress) ─────────────────────────────────
swanlab_cb = None
if not STRESS_TEST:
    try:
        import swanlab
        from swanlab.integration.transformers import SwanLabCallback
        sw_key = os.environ.get('SWANLAB_API_KEY')
        if sw_key:
            swanlab.login(api_key=sw_key, save=False)
        swanlab_cb = SwanLabCallback(
            project        = SWANLAB_PROJECT,
            experiment_name= SWANLAB_RUN_NAME,
            config = dict(
                model_path=MODEL_PATH, seed_adapter=SEED_ADAPTER_PATH,
                beta=BETA, lr=LEARNING_RATE,
                per_device_batch=PER_DEVICE_BATCH, grad_accum=GRAD_ACCUM,
                num_epochs=NUM_EPOCHS, max_seq_len=MAX_SEQ_LEN,
                max_prompt_len=MAX_PROMPT_LEN, save_steps=SAVE_STEPS,
                anchor_fraction=ANCHOR_FRACTION,
                easy_categories=sorted(EASY_CATEGORIES_VAL_GE_98),
            ),
        )
        _log(f'SwanLab init OK: project={SWANLAB_PROJECT} run={SWANLAB_RUN_NAME}')
    except Exception as e:
        _log(f'SwanLab init FAILED ({type(e).__name__}: {str(e)[:100]}); continuing without it')
        swanlab_cb = None

# ── 2. Load base bf16 model via Unsloth (matches SFT seed loader exactly) ──
_log('Loading bf16 base model via Unsloth …')
# Pre-trigger Nemotron-H dynamic module import (AutoConfig does NOT load modeling, only config)
# so we can flip _supports_sdpa to True. The modeling file ships NemotronHSdpaAttention class +
# 'sdpa' key in NEMOTRONH_ATTENTION_CLASSES — only the class-level flag gates HF dispatch.
from transformers.dynamic_module_utils import get_class_from_dynamic_module as _get_cls_dyn
_clm_cls = _get_cls_dyn('modeling_nemotron_h.NemotronHForCausalLM', MODEL_PATH)
_clm_cls._supports_sdpa = True
# Walk MRO to find NemotronHPreTrainedModel (HF checks the attribute on the base class)
for _base in _clm_cls.__mro__:
    if _base.__name__ == 'NemotronHPreTrainedModel':
        _base._supports_sdpa = True
        break
_log(f'  [PATCH] NemotronHForCausalLM._supports_sdpa = True (cls + MRO base)')
t_load = time.time()
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name              = MODEL_PATH,
    max_seq_length          = MAX_SEQ_LEN,
    load_in_4bit            = False,
    load_in_8bit            = False,
    full_finetuning         = False,
    trust_remote_code       = True,
    unsloth_force_compile   = True,
    attn_implementation     = 'eager',    # NemotronHSdpaAttention has a shape bug (attn_output.view with wrong hidden_size at line 1193); FA2 not installed. Eager is the only working path. We trim max_length to compensate for the 80-MB shortfall #18 showed at T=7785.
    dtype                   = torch.bfloat16,
)
_log(f'  loaded in {time.time()-t_load:.1f}s, VRAM={_smi()}M')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── 3. Load LoRA adapter directly (no temp copy; seed already has original lm_head keys) ──
# Our seed comes from the SFT soup-save adapter-248 dir, which has the original
# `base_model.model.lm_head.*` key naming (NOT the Kaggle-submission `backbone.lm_head` form).
# So no rename is needed. Loading direct from SEED_ADAPTER_PATH avoids the temp-copy +
# save_file dance, which previously left a corrupt half-written file when killed mid-execution.
# Only patch adapter_config.json in place if it's not already in training mode.
_log('Loading SFT seed adapter (direct, no temp copy) …')
_cfg_path = os.path.join(SEED_ADAPTER_PATH, 'adapter_config.json')
with open(_cfg_path) as _f:
    _cfg = json.load(_f)
if _cfg.get('inference_mode') or _cfg.get('base_model_name_or_path') != MODEL_PATH:
    _log(f'  adjusting adapter_config.json in place (inference_mode={_cfg.get("inference_mode")}, '
         f'base={_cfg.get("base_model_name_or_path")} → {MODEL_PATH})')
    _cfg['inference_mode'] = False
    _cfg['base_model_name_or_path'] = MODEL_PATH
    with open(_cfg_path, 'w') as _f:
        json.dump(_cfg, _f, indent=2)
model = PeftModel.from_pretrained(model, SEED_ADAPTER_PATH, is_trainable=True)
_log(f'  loaded, VRAM={_smi()}M')

# ── 5. Force out_proj live (PERSISTENT across trainer.train() calls) ──
# `mixer.training=False` makes cuda_kernels_forward take the unfused else-branch where
# `self.out_proj()` is a real module call → out_proj LoRA receives gradients (the SFT
# 0.84 → 0.86 lever). BUT TRL's DPOTrainer calls model.train() at the start of training,
# which recursively sets .training=True on every submodule, undoing this. So we ALSO
# register a forward_pre_hook that re-asserts .training=False on every forward call —
# even after trainer.train() has flipped it, the next forward sets it back.
def _force_eval_pre_hook(module, _input):
    module.training = False
forced = 0
for _mod in model.modules():
    if hasattr(_mod, 'cuda_kernels_forward'):
        _mod.training = False
        _mod.register_forward_pre_hook(_force_eval_pre_hook)
        forced += 1
_log(f'Forced {forced} Mamba mixers to .training=False + registered pre-hook (out_proj-live)')

# ── 6. Unsloth SMART offloaded gradient checkpointing — MATCH SFT PATH EXACTLY ──
# SFT 0.85 uses FastLanguageModel.get_peft_model(use_gradient_checkpointing='unsloth'), which
# internally calls patch_unsloth_SMART_gradient_checkpointing(dtype=...) and FastLanguageModel
# .for_training(model). The non-SMART variant I called earlier does NOT route HF's
# _gradient_checkpointing_func through Unsloth's offloader — that's why backbone forward at
# T=7680 grew 32 GB instead of being offload-bounded.
from unsloth_zoo.gradient_checkpointing import patch_unsloth_smart_gradient_checkpointing
patch_unsloth_smart_gradient_checkpointing(dtype=torch.bfloat16)
# CRITICAL: gradient_checkpointing_enable sets _gradient_checkpointing_func via functools.partial
# capturing the patched checkpoint. for_training only sets the flag, not the func. Need both.
model.base_model.model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={'use_reentrant': False}
)
if hasattr(model.base_model.model, 'backbone'):
    model.base_model.model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False}
    )
FastLanguageModel.for_training(model)
# Mamba CUDA fast path — the comment in SFT says torch_forward materializes a 126 GB SSM tensor
# at seq 8192. Force is_fast_path_available=True so dispatch uses cuda_kernels_forward.
for _name, _m in list(sys.modules.items()):
    if 'modeling_nemotron_h' in _name and hasattr(_m, 'is_fast_path_available'):
        _m.is_fast_path_available = True
        _log(f'  [PATCH] {_name}.is_fast_path_available = True')
        break
_log('Enabled Unsloth SMART offloaded GC + for_training + Mamba fast path (SFT-matched setup)')

# Freeze MoE routed expert LoRA AFTER for_training (in case for_training re-enables grads).
def _is_moe_routed_expert_lora(name):
    nm = name.lower()
    if 'shared_expert' in nm or 'lora_' not in nm:
        return False
    return any(tok in nm for tok in ['.experts.', 'routed_experts', 'expert_weights', '.expert.', 'moe.experts'])
_n_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
_n_frozen = 0
for _n, _p in model.named_parameters():
    if _p.requires_grad and _is_moe_routed_expert_lora(_n):
        _p.requires_grad = False
        _n_frozen += _p.numel()
_log(f'  Froze MoE routed-expert LoRA POST-for_training: {_n_before/1e6:.1f}M -> {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M  (frozen {_n_frozen/1e6:.1f}M)')

n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
_log(f'Trainable: {n_train/1e6:.1f}M / total {n_total/1e9:.1f}B  ({100*n_train/n_total:.3f}%)')

# ── 7. Build + filter dataset ────────────────────────────────────────
_log('Loading + joining prefer + reject CSVs …')
dp = pd.read_csv(PREFER_CSV)
dr = pd.read_csv(REJECT_CSV)

# Parse the `tokens` column (JSON list of token strings) to detect whether the model emitted
# <|im_end|> as its final token. The detokenized raw_output column STRIPS this EOS, so we
# need the per-row flag to reconstruct the model's actual emission for DPO.
def _last_tok(s):
    try:    return _json.loads(s)[-1]
    except: return None
dr['_eos_emitted'] = dr['tokens'].apply(_last_tok) == '<|im_end|>'
_log(f'  rejected: model emitted <|im_end|>: {dr["_eos_emitted"].sum()} / {len(dr)} = {100*dr["_eos_emitted"].mean():.1f}%')

m = dp[['id', 'prompt', 'answer', 'solver_cot']].merge(
    dr[['id', 'raw_output', 'category', '_eos_emitted']], on='id'
)
_log(f'  joined: {len(m)} pairs (pre-filter)')

def _filter_easy(df):
    parts = []
    for cat, grp in df.groupby('category'):
        if cat in EASY_CATEGORIES_VAL_GE_98:
            n = max(1, int(round(len(grp) * ANCHOR_FRACTION)))
            parts.append(grp.sample(n=n, random_state=42))
        else:
            parts.append(grp)
    return pd.concat(parts, ignore_index=True)

m = _filter_easy(m)
_log(f'  post-filter: {len(m)} pairs')
_log('  per-category counts:')
for cat, n in m.groupby('category').size().sort_index().items():
    flag = ' (anchor 10%)' if cat in EASY_CATEGORIES_VAL_GE_98 else ' (full keep)'
    _log(f'    {cat:30s} {n:>5d}{flag}')

# STRESS TEST: top-K longest pairs; K = bs × max_steps × 2 per routine rule (1 × 3 × 2 × ga(16) ≈ 96)
STRESS_K = 96
STRESS_STEPS = 3
if STRESS_TEST:
    m = m.assign(_tot=m.solver_cot.str.len() + m.raw_output.str.len())
    m = m.sort_values('_tot', ascending=False).head(STRESS_K).reset_index(drop=True)
    _log(f'STRESS_TEST: using {len(m)} longest pairs, {STRESS_STEPS} steps')

_boxed_pat = re.compile(r'\\boxed\{')
def _extract_last_boxed(text):
    starts = list(_boxed_pat.finditer(text))
    if not starts: return None
    last = starts[-1]; seg = text[last.end():]; lb = seg.rfind('}')
    return seg[:lb] if lb != -1 else seg

def fmt_example(row):
    user_msg = row['prompt'] + PROMPT_SUFFIX
    prompt_text = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': user_msg}],
        tokenize               = False,
        add_generation_prompt  = True,
        enable_thinking        = True,
    )
    reasoning_answer = _extract_last_boxed(row['solver_cot']) or str(row['answer'])
    # Chosen: apply SFT's processing byte-for-byte (train_moe_outproj.py L191) including
    # <|im_end|>, because the model emits <|im_end|> as its final token (audit: 98.4% of
    # rejected generations end with it; the rest were truncated by max_tokens).
    chosen   = row['solver_cot'] + '\n</think>\n' + chr(92) + 'boxed{' + reasoning_answer + '}<|im_end|>'
    # Rejected: raw_output is the detokenized form, which strips the EOS. Reconstruct what
    # the model actually emitted by appending <|im_end|> when the tokens column shows the
    # model produced it. For the 1.6% truncated cases, leave rejected without EOS — those
    # are genuinely incomplete generations.
    rejected = row['raw_output'] + ('<|im_end|>' if row['_eos_emitted'] else '')
    return {'prompt': prompt_text, 'chosen': chosen, 'rejected': rejected, 'problem_id': row['id']}

examples = [fmt_example(r) for _, r in m.iterrows()]
train_ds = Dataset.from_list(examples)
_log(f'Dataset: {len(train_ds)} examples')

# Token length sanity
def _toklen(text): return len(tokenizer(text, add_special_tokens=False)['input_ids'])
sample_ids = list(range(min(50, len(examples))))
import statistics as stt
for name, key in [('prompt', 'prompt'), ('chosen', 'chosen'), ('rejected', 'rejected')]:
    lens = [_toklen(examples[i][key]) for i in sample_ids]
    _log(f'tokens {name:8s}: med={stt.median(lens):.0f} p90={sorted(lens)[int(0.9*len(lens))]:.0f} max={max(lens)}')

# ── 8. Compute total steps + soup save steps + resume target ─────────
n_examples       = len(train_ds)
eff_batch        = PER_DEVICE_BATCH * GRAD_ACCUM
total_steps_full = max(1, math.ceil(n_examples / eff_batch) * NUM_EPOCHS)
if STRESS_TEST:
    planned_steps = STRESS_STEPS
else:
    planned_steps = total_steps_full
_log(f'Planned optimizer steps: {planned_steps}  (n={n_examples}, eff_batch={eff_batch}, epochs={NUM_EPOCHS})')

# Resume-from-checkpoint
RESUME_FROM = None
if not STRESS_TEST and not FRESH_START and os.path.exists(OUTPUT_DIR):
    ckpts = [d for d in os.listdir(OUTPUT_DIR) if d.startswith('checkpoint-')]
    if ckpts:
        ckpts.sort(key=lambda x: int(x.split('-')[1].split('_')[0]))
        RESUME_FROM = os.path.join(OUTPUT_DIR, ckpts[-1])
        _log(f'Resume from: {RESUME_FROM}')

# ── 9. Callbacks ─────────────────────────────────────────────────────
class GcCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **k):
        gc.collect(); torch.cuda.empty_cache()
    def on_step_end(self, args, state, control, **k):
        gc.collect(); torch.cuda.empty_cache()

class StepInstrumentation(TrainerCallback):
    def __init__(self, out_proj_recorder):
        self.rec = out_proj_recorder
        self.t0  = None
    def on_step_begin(self, args, state, control, **k):
        self.t0 = time.time()
    def on_step_end(self, args, state, control, **k):
        dt = time.time() - self.t0 if self.t0 else 0.0
        if state.global_step <= 5 or state.global_step % 5 == 0:
            recs = list(self.rec.items())[:2]
            tag = ' '.join(f'L{n.split(".layers.")[-1][:3]}:{v:.4f}' for n, v in recs)
            _log(f'>> step={state.global_step} smi={_smi()}M dt={dt:.1f}s out_proj_gL2[{tag}]')

# grad hooks on out_proj.lora_B
out_proj_l2 = {}
def make_hook(name):
    def hook(grad):
        out_proj_l2[name] = grad.detach().float().norm().item()
    return hook
n_hooked = 0
for n, p in model.named_parameters():
    if 'out_proj' in n and 'lora_B' in n and p.requires_grad:
        p.register_hook(make_hook(n)); n_hooked += 1
_log(f'Registered grad hooks on {n_hooked} out_proj.lora_B params')

# ── 10. DPOConfig + DPOTrainer ───────────────────────────────────────
dpo_config = DPOConfig(
    output_dir                       = OUTPUT_DIR,
    beta                             = BETA,
    max_length                       = MAX_SEQ_LEN,
    # TRL 1.0.0 dropped `max_prompt_length` as a separate arg; total `max_length` is sufficient
    # since our longest pair (prompt 115 + rejected 7682) is well under 8192.
    # Precompute ref logp once upfront so training step does ONE policy forward (no ref re-run).
    # Combined with our SequentialDPOTrainer (batch=1 forward) + chunked CCE (no full lm_head logits),
    # this is the configuration that actually fits at max_length=8192.
    precompute_ref_log_probs         = True,
    activation_offloading            = False,  # Agent research: on no_grad forwards the offload context's hooks + pinned-host churn FRAGMENT the allocator across Mamba/attention layer sizes — actually HURTS, not helps. Per-layer Mamba_chunk_scan_combined allocations need to reuse blocks freely.
    # CCE is now implemented in pure pytorch inside SequentialDPOTrainer (chunked logp), so we
    # don't need the Liger fused kernel. Standard _compute_loss path (our subclass override).
    use_liger_kernel                 = False,
    per_device_train_batch_size      = PER_DEVICE_BATCH,
    gradient_accumulation_steps      = GRAD_ACCUM,
    learning_rate                    = LEARNING_RATE,
    num_train_epochs                 = NUM_EPOCHS,
    max_steps                        = (STRESS_STEPS if STRESS_TEST else -1),
    bf16                             = True,
    gradient_checkpointing           = False,    # already enabled manually
    logging_steps                    = 1,
    save_strategy                    = ('no' if STRESS_TEST else 'steps'),
    save_steps                       = SAVE_STEPS,
    save_total_limit                 = 1,        # FIFO=1 per routine
    remove_unused_columns            = False,
    seed                             = 42,
    # paged_adamw_8bit: only ~50 M trainable after MoE expert freeze, so Adam state
    # ~0.05 GB on GPU. Fits comfortably alongside Phase 3 backward's CCE dc allocation.
    optim                            = 'paged_adamw_8bit',
    lr_scheduler_type                = 'linear',
    warmup_ratio                     = 0.0,
    report_to                        = 'none',   # SwanLab via callback
    dataloader_drop_last             = False,
    dataloader_pin_memory            = False,
)

callbacks = [GcCallback(), StepInstrumentation(out_proj_l2)]
if swanlab_cb is not None:
    callbacks.append(swanlab_cb)

# ── Monkey-patch model.add_adapter + model.get_parameter to bypass TRL #5222 ──
# UnslothDPOTrainer's __init__ unconditionally calls model.add_adapter("ref", peft_config["default"])
# to create a frozen "ref" copy. This fails on Nemotron-3-Nano-30B-A3B because the MoE experts use
# stacked tensors → PEFT requires target_parameters → PEFT hardcodes "only one adapter per model
# with target_parameters allowed". The downstream use_adapter(...) calls (UnslothDPOTrainer.py:1388,
# 1544) already fall back to adapter_name=None → disable_adapter() when "ref" isn't in peft_config.
# We make add_adapter("ref") a no-op + make get_parameter(".ref.…") return a dummy so the copy
# loop completes silently. Net effect: ref forward = base model (no LoRA). For low-beta=0.05
# distillation DPO this is an acceptable trade-off.
_orig_add_adapter   = model.add_adapter
_orig_get_parameter = model.get_parameter
class _DummyData:
    @staticmethod
    def copy_(*a, **k): return None
class _DummyParam:
    data = _DummyData()
def _patched_add_adapter(name, *a, **k):
    if name == 'ref':
        _log("  [PATCH] skipping add_adapter('ref') — TRL #5222 PEFT target_parameters workaround")
        return None
    return _orig_add_adapter(name, *a, **k)
def _patched_get_parameter(name):
    if '.ref.' in name:
        return _DummyParam()
    return _orig_get_parameter(name)
model.add_adapter   = _patched_add_adapter
model.get_parameter = _patched_get_parameter

# ── Patch NemotronHForCausalLM.get_decoder ──
# Liger DPO's _compute_loss_liger calls model.get_decoder() to forward through the backbone
# without the lm_head, then runs the fused CCE op on hidden_states + lm_head.weight.
# Nemotron-H's modeling file (line 1605) returns `self.model`, but its __init__ stores the
# backbone as `self.backbone` — so get_decoder() raises AttributeError. We can't edit the
# modeling file. Find the NemotronHForCausalLM instance and override its bound method.
def _find_causal_lm(m, depth=0):
    if type(m).__name__ == 'NemotronHForCausalLM':
        return m
    if depth > 6:
        return None
    for child in m.children():
        r = _find_causal_lm(child, depth + 1)
        if r is not None:
            return r
    return None
_inner_clm = _find_causal_lm(model)
if _inner_clm is not None and hasattr(_inner_clm, 'backbone'):
    _inner_clm.get_decoder = lambda: _inner_clm.backbone
    _log(f"  [PATCH] NemotronHForCausalLM.get_decoder -> backbone (Liger CCE compatibility)")
else:
    _log(f"  [WARN] could not locate NemotronHForCausalLM under PEFT wrapper; Liger may fail")

try:
    trainer = SequentialDPOTrainer(
        model           = model,
        ref_model       = None,    # → PEFT disable_adapter for ref forward (precompute only)
        args            = dpo_config,
        train_dataset   = train_ds,
        processing_class= tokenizer,
        callbacks       = callbacks,
    )
finally:
    # Restore real methods so anything downstream that legitimately calls them works
    model.add_adapter   = _orig_add_adapter
    model.get_parameter = _orig_get_parameter

# ── 11. Train ────────────────────────────────────────────────────────
_log(f'\n=== Training start  VRAM={_smi()}M  resume={RESUME_FROM} ===')
t0 = time.time()
trainer.train(resume_from_checkpoint=RESUME_FROM)
elapsed = time.time() - t0
_log(f'\n=== Training done. Time: {elapsed/3600:.2f} hrs ({elapsed/60:.1f} min) ===')

# Out_proj liveness final report
if out_proj_l2:
    items = sorted(out_proj_l2.items())[:8]
    _log('OUT_PROJ.lora_B grad-L2 final samples (MUST be > 0):')
    for n, v in items:
        _log(f'  {n}  =  {v:.6f}')
else:
    _log('WARNING: no out_proj.lora_B grad hooks fired (out_proj may be DEAD under DPO forward)')

# ── 12. STRESS exit ──────────────────────────────────────────────────
if STRESS_TEST:
    _log('STRESS_TEST: skipping save + closeout. STRESS TEST COMPLETE.')
    sys.exit(0)

# ── 13. Force-save final adapter + Kaggle-compat rename ──────────────
final_dir = OUTPUT_DIR + '_final'
_log(f'Saving final adapter to {final_dir}')
trainer.save_model(final_dir)              # full Trainer save (checkpoint format)
model.save_pretrained(final_dir)           # ensure PEFT adapter files written

st_path = os.path.join(final_dir, 'adapter_model.safetensors')
if os.path.exists(st_path):
    tensors = load_file(st_path)
    renamed = {k.replace('base_model.model.lm_head.', 'base_model.model.backbone.lm_head.'): v
               for k, v in tensors.items()}
    save_file(renamed, st_path)
    cfg_path = os.path.join(final_dir, 'adapter_config.json')
    with open(cfg_path) as f: cfg = json.load(f)
    cfg['base_model_name_or_path'] = 'metric/nemotron-3-nano-30b-a3b-bf16'
    cfg['inference_mode'] = True
    with open(cfg_path, 'w') as f: json.dump(cfg, f, indent=2)
    _log('Saved + renamed lm_head keys for Kaggle compat')

# ── 14. Group artifacts into {yymmdd_hhmm}_submission/ (Chicago time) ──
chi_now = datetime.datetime.utcnow() - datetime.timedelta(hours=5)   # CDT
ts = chi_now.strftime('%y%m%d_%H%M')
sub_dir = f'/root/autodl-tmp/{ts}_submission/'
os.makedirs(sub_dir, exist_ok=True)
_log(f'Submission group folder: {sub_dir}')

# submission.zip: only adapter_config.json + adapter_model.safetensors (Kaggle-renamed)
REQUIRED = {'adapter_config.json', 'adapter_model.safetensors'}
sub_zip = os.path.join(sub_dir, 'submission.zip')
with zipfile.ZipFile(sub_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir(final_dir):
        if fname in REQUIRED:
            zf.write(os.path.join(final_dir, fname), arcname=fname)
_log(f'  ZIP: submission.zip ({os.path.getsize(sub_zip)/1024/1024:.1f} MB)')

# checkpoint-{final}.zip: latest FIFO checkpoint dir
ckpt_dirs = sorted([d for d in os.listdir(OUTPUT_DIR) if d.startswith('checkpoint-')],
                   key=lambda x: int(x.split('-')[1].split('_')[0]))
if ckpt_dirs:
    final_ckpt = ckpt_dirs[-1]
    czip = os.path.join(sub_dir, f'{final_ckpt}.zip')
    cfull = os.path.join(OUTPUT_DIR, final_ckpt)
    with zipfile.ZipFile(czip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(cfull):
            for f in files:
                fp = os.path.join(root, f)
                zf.write(fp, arcname=os.path.relpath(fp, cfull))
    _log(f'  ZIP: {final_ckpt}.zip ({os.path.getsize(czip)/1024/1024:.1f} MB)')

_log(f'\n=== TRAINING SUMMARY — DPO distillation from cryptnumeq 0.85 seed ===')
_log(f'  GPU: {torch.cuda.get_device_name(0)}')
_log(f'  Examples: {n_examples}  Steps: {planned_steps}  Time: {elapsed/60:.1f} min')
_log(f'  Submission folder: {sub_dir}')
_log('=== DPO RUN COMPLETE ===')

if swanlab_cb is not None:
    try:
        import swanlab; swanlab.finish()
    except: pass
