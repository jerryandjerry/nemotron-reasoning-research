import csv, os, re, sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/Nemotron-3-Nano-30B-A3B", trust_remote_code=True)
PROMPT_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

path = "/root/autodl-tmp/data/260416_train_cot_v1"
for fname in sorted(os.listdir(path)):
    if not fname.endswith(".csv"):
        continue
    lens = []
    with open(os.path.join(path, fname), encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cot = str(row["solver_cot"])
            cot_cleaned = re.sub(r'\\boxed\{[^}]*\}', '', cot).rstrip()
            user_msg = row["prompt"] + PROMPT_SUFFIX
            assistant_msg = cot_cleaned + f'\n</think>\n\\boxed{{{row["answer"]}}}'
            text = f'<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>'
            ids = tokenizer(text, truncation=False, return_attention_mask=False)["input_ids"]
            lens.append(len(ids))
    lens.sort()
    n = len(lens)
    if n == 0:
        continue
    print(f'{fname} ({n} rows): min={lens[0]} med={lens[n//2]} p90={lens[int(n*0.9)]} p95={lens[int(n*0.95)]} p99={lens[int(n*0.99)]} max={lens[-1]}')
