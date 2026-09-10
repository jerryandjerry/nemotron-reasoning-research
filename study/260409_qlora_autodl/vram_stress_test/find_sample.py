import polars as pl
from transformers import AutoTokenizer

train_df = pl.read_csv("/root/autodl-tmp/data/final_Nemotron_training_data.csv")
KEYWORDS = ["cipher","string","transformation","bitwise","binary","unit","measurement","encoding","decoding","pattern","sequence","conversion","arithmetic","logical","symbolic"]
label_col = train_df["label"].str.to_lowercase()
mask = pl.lit(False)
for kw in KEYWORDS:
    mask = mask | label_col.str.contains(kw)
filtered_df = train_df.filter(mask).sample(n=3000, seed=99)

tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/Nemotron-3-Nano-30B-A3B-bnb-4bit", trust_remote_code=True)

for idx, row in enumerate(filtered_df.iter_rows(named=True)):
    prompt = row["prompt"]
    cot = str(row["generated_cot"])
    answer = str(row["answer"])
    user_msg = prompt + "\nPut your final answer inside \\boxed{}."
    assistant_msg = cot + "\n\n\\boxed{" + answer + "}"
    messages = [{"role": "user", "content": user_msg}, {"role": "assistant", "content": assistant_msg}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    except:
        continue
    ids = tokenizer(text, truncation=False, return_attention_mask=False)["input_ids"]
    tlen = len(ids)
    if 2190 <= tlen <= 2210:
        print(f"FOUND: idx={idx} tokens={tlen}")
        print(f"LABEL: {row['label']}")
        print(f"PROMPT: {prompt[:500]}")
        print(f"ANSWER: {answer[:200]}")
        print(f"COT: {cot[:500]}")
        print("---END---")
