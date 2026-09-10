"""
Nemotron-3-Nano-30B-A3B — Activation Analysis
==============================================
Hooks into model.generate() with use_cache=False to capture per-token
expert routing and layer contribution ratios from the last forward pass.

Each decode step reprocesses the full sequence (no KV cache).
The last step's data covers the complete generated sequence.
Hooks overwrite each step — when generation ends, storage has the final data.

Usage:
  python analyze_activations.py                                    # 6 task types
  python analyze_activations.py --types "Numeral Conversion" --samples-per-type 2
  python analyze_activations.py --prompt "some prompt"
"""

import argparse
import json
import os
import torch
import numpy as np
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/root/autodl-tmp/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
TYPE_MAP = {'M': 'mamba', 'E': 'moe', '*': 'attn'}
LAYER_TYPES = [TYPE_MAP[ch] for ch in PATTERN]
NUM_LAYERS = len(LAYER_TYPES)

# ===== HOOK STORAGE =====
# Overwrite each step — last forward pass = complete sequence
layer_contrib_last = {}       # layer_idx -> {ratio, delta_norm, residual_norm, per_token}
expert_routing_last = {}      # layer_idx -> [[e1..e6], ...] per token
residual_before = {}
hooks = []


def install_hooks(model):
    global hooks
    hooks = []
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
        layers = model.backbone.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        raise AttributeError(f"Cannot find layers in model: {type(model)}")

    for idx, layer in enumerate(layers):
        ltype = LAYER_TYPES[idx]

        def make_pre_hook(layer_idx):
            def pre_hook(module, args):
                x = args[0] if isinstance(args[0], torch.Tensor) else args[0][0]
                residual_before[layer_idx] = x.detach().clone()
            return pre_hook
        h = layer.register_forward_pre_hook(make_pre_hook(idx))
        hooks.append(h)

        def make_post_hook(layer_idx):
            def post_hook(module, args, output):
                x_after = output[0] if isinstance(output, tuple) else output
                if isinstance(x_after, torch.Tensor) and layer_idx in residual_before:
                    x_before = residual_before[layer_idx]
                    delta = x_after - x_before
                    delta_norm = delta.float().norm().item()
                    before_norm = x_before.float().norm().item()
                    ratio = delta_norm / (before_norm + 1e-8)
                    # Per-token ratio
                    delta_per_tok = delta.float().norm(dim=-1).squeeze(0)
                    before_per_tok = x_before.float().norm(dim=-1).squeeze(0)
                    ratio_per_tok = (delta_per_tok / (before_per_tok + 1e-8)).tolist()
                    layer_contrib_last[layer_idx] = {
                        'ratio': round(ratio, 6),
                        'delta_norm': round(delta_norm, 2),
                        'residual_norm': round(before_norm, 2),
                        'per_token': [round(r, 6) for r in ratio_per_tok],
                    }
                    # Free GPU memory
                    del residual_before[layer_idx]
            return post_hook
        h = layer.register_forward_hook(make_post_hook(idx))
        hooks.append(h)

        if ltype == 'moe' and hasattr(layer, 'mixer') and hasattr(layer.mixer, 'gate'):
            router = layer.mixer.gate
            def make_router_hook(layer_idx):
                def router_hook(module, args, output):
                    if isinstance(output, tuple) and len(output) >= 2:
                        expert_ids = output[0]  # [num_tokens, 6]
                        expert_routing_last[layer_idx] = expert_ids.detach().cpu().tolist()
                return router_hook
            h = router.register_forward_hook(make_router_hook(idx))
            hooks.append(h)


def remove_hooks():
    for h in hooks:
        h.remove()
    hooks.clear()


def reset_storage():
    layer_contrib_last.clear()
    expert_routing_last.clear()
    residual_before.clear()


def run_inference(model, tokenizer, prompt, max_new_tokens=7680):
    """
    Generate with use_cache=False so each step reprocesses the full sequence.
    Hooks overwrite each step. After generation, storage has the last step's data
    which covers all tokens in the complete sequence.
    """
    reset_storage()

    user_msg = prompt + PROMPT_SUFFIX
    messages = [{"role": "user", "content": user_msg}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs['input_ids'].shape[1]

    print(f"  Generating (input={input_len}, max_new={max_new_tokens})...")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            use_cache=False,
        )

    gen_len = outputs.shape[1] - input_len
    total_len = outputs.shape[1]
    generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    gen_token_ids = outputs[0][input_len:].tolist()
    gen_tokens = [tokenizer.decode([tid]) for tid in gen_token_ids]
    print(f"  Generated {gen_len} tokens (total seq: {total_len})")

    # Collect from storage
    layer_contrib = {}
    for idx in range(NUM_LAYERS):
        if idx in layer_contrib_last:
            layer_contrib[idx] = layer_contrib_last[idx]

    expert_routing = {}
    for idx in range(NUM_LAYERS):
        if idx in expert_routing_last:
            entries = expert_routing_last[idx]
            freq = defaultdict(int)
            for token_experts in entries:
                for eid in token_experts:
                    freq[eid] += 1
            expert_routing[idx] = {
                'expert_freq': dict(freq),
                'num_tokens': len(entries),
                'per_token': entries,
            }
            print(f"  L{idx}: {len(entries)} tokens")

    return {
        'prompt': prompt,
        'input_tokens': input_len,
        'generated_token_count': gen_len,
        'total_tokens': total_len,
        'generated_text': generated,
        'generated_tokens': gen_tokens,
        'layer_contrib': layer_contrib,
        'expert_routing': expert_routing,
    }


def summarize_results(result):
    print(f"\n{'='*70}")
    print(f"Prompt: {result['prompt'][:80]}")
    print(f"Input: {result['input_tokens']} | Generated: {result['generated_token_count']} | Total: {result['total_tokens']}")
    print(f"{'='*70}")

    contribs = sorted(result['layer_contrib'].items(), key=lambda x: x[1]['ratio'])
    print(f"\n--- Layer Contribution (top 10) ---")
    for idx, data in contribs[-10:]:
        idx = int(idx)
        print(f"  L{idx:>3}   {LAYER_TYPES[idx]:>8}   ratio={data['ratio']:>8.4f}  delta={data['delta_norm']}  resid={data['residual_norm']}")

    print(f"\n--- MoE Expert Routing ---")
    for layer_idx in sorted(int(k) for k in result['expert_routing']):
        routing = result['expert_routing'].get(layer_idx, result['expert_routing'].get(str(layer_idx)))
        freq = routing['expert_freq']
        nt = routing['num_tokens']
        top3 = sorted(freq.items(), key=lambda x: -x[1])[:3]
        top3_str = ", ".join(f"E{eid}({c/nt:.2f})" for eid, c in top3)
        print(f"  L{layer_idx:>3}: {len(freq):>3}/128 | {nt} tok | {top3_str}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str)
    parser.add_argument('--max-new-tokens', type=int, default=7680)
    parser.add_argument('--num-types', type=int, default=6)
    parser.add_argument('--types', type=str)
    parser.add_argument('--samples-per-type', type=int, default=1)
    parser.add_argument('--output-dir', type=str, default='/root/autodl-tmp')
    args = parser.parse_args()

    if args.prompt:
        prompts = [args.prompt]
    else:
        import pandas as pd
        DATA_PATH = "/root/autodl-tmp/data/konbu17_verified_cot_6558rows.csv"
        df = pd.read_csv(DATA_PATH)
        prompts = []
        if args.types:
            type_list = [t.strip() for t in args.types.split(',')]
        else:
            type_list = sorted(df['type'].unique())[:args.num_types]
        for task_type in type_list:
            subset = df[df['type'] == task_type].head(args.samples_per_type)
            for _, row in subset.iterrows():
                prompts.append(row['prompt'])
                print(f"  [{task_type}] {row['prompt'][:60]}...")

    print(f"Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    import transformers
    print(f"transformers {transformers.__version__}")
    print(f"Model class: {type(model).__name__}")

    print("Installing hooks...")
    install_hooks(model)

    json_path = os.path.join(args.output_dir, 'activation_analysis.json')
    results = []

    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] {prompt[:60]}...")
        result = run_inference(model, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
        summarize_results(result)
        results.append(result)

        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  Saved to {json_path} ({os.path.getsize(json_path)/1024:.0f} KB)")


if __name__ == '__main__':
    main()
