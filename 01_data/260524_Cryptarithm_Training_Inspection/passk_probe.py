#!/usr/bin/env python3
"""pass@k probe for cryptarithm — RUN ON THE EVAL GPU (vLLM), not locally.

Decides whether RL could help: RL can only amplify correct answers the base model already
produces *sometimes when sampled*. We deploy greedy, but RL trains by sampling, so the ceiling for
RL-then-greedy accuracy is ~ base pass@k. If pass@64 ~ 0 on cryptarithm, RL is futile.

Drop this in place of the generation cell in eval_kaggle_nemotron.ipynb (it reuses MODEL_PATH,
LORA_PATH, VAL_PATH, extract_final_answer, verify, build_prompt). Key change: temperature>0 and n>1.
"""
import collections
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

K = 64                      # samples per puzzle
TEMP = 0.8
lora_request = LoRARequest('adapter', 1, LORA_PATH)

# cryptarithm = symbol operands (no 0-9 digits in the example lines); eqnum has real digits
def is_cryptarithm(prompt):
    eqs = [l for l in prompt.splitlines() if '=' in l]
    return not any(c.isdigit() for l in eqs for c in l)

crypt = val_df[val_df['prompt'].apply(is_cryptarithm)].reset_index(drop=True)
print(f"cryptarithm val puzzles: {len(crypt)}")

prompts = [build_prompt(r['prompt']) for _, r in crypt.iterrows()]
sp = SamplingParams(n=K, temperature=TEMP, top_p=0.95, max_tokens=7680, skip_special_tokens=False)
outputs = llm.generate(prompts, sp, lora_request=lora_request)

# subtype split (rough): concat if any example RHS equals operands concatenated is hard to detect
# without the mapping; instead split by output length behaviour later. Here: overall + per gt-len.
rows = []
for i, out in enumerate(outputs):
    gt = str(crypt.iloc[i]['answer'])
    correct = [verify(gt, extract_final_answer(o.text)) for o in out.outputs]
    toks = [len(o.token_ids) for o in out.outputs]
    rows.append({
        'id': crypt.iloc[i]['id'],
        'gt': gt,
        'n_correct': sum(correct),
        'pass_at_k': any(correct),
        'frac_correct': sum(correct) / K,          # ~ pass@1 (sampling-efficiency proxy)
        'median_tok': sorted(toks)[K // 2],
        'frac_truncated': sum(t >= 7670 for t in toks) / K,
    })

import pandas as pd
df = pd.DataFrame(rows)
print(f"\n=== cryptarithm pass@{K} (temp {TEMP}) ===")
print(f"pass@{K} (>=1 of {K} correct): {df['pass_at_k'].sum()}/{len(df)} = {100*df['pass_at_k'].mean():.1f}%")
print(f"pass@1 estimate (mean per-sample correct): {100*df['frac_correct'].mean():.2f}%")
print(f"median generated tokens: {df['median_tok'].median():.0f} | mean frac truncated: {df['frac_truncated'].mean():.2f}")
print("\npuzzles with >=1 correct sample (these are what RL could amplify):")
for _, r in df[df['pass_at_k']].sort_values('n_correct', ascending=False).iterrows():
    print(f"   {r['id']} gt={r['gt']!r}  {r['n_correct']}/{K} correct")
df.to_csv('/kaggle/working/crypt_passk.csv', index=False)
print("\nDECISION: pass@%d ~0 -> RL futile (no reward to amplify). pass@%d>0 -> RL can lift that subset." % (K, K))
