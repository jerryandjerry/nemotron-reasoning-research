"""
260408 DeepSeek R1 (deepseek-reasoner) Eval on 950 val set
Uses OpenAI-compatible API with reasoning_content separate from content.
max_tokens=32768 (matching mainstream LLM benchmark for fair comparison).
50 concurrent workers to maximize throughput.
Tracks token usage and cost per sample.
"""
import os, sys, time, math, re, json
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# === Config ===
API_KEY = dict(l.strip().split('=', 1) for l in open('kaggle.env', encoding='utf-8-sig') if '=' in l and not l.startswith('#'))['DEEPSEEK_API_KEY']
MODEL = "deepseek-reasoner"
VAL_CSV = "D:/SynologyDrive/00_Kaggle/2026_Nemotron/01_data/10-90_split/260407_val.csv"
OUTPUT_DIR = "D:/SynologyDrive/00_Kaggle/2026_Nemotron/03_eval/260408_deepseek_r1_950val"
CONCURRENCY = 50  # parallel requests — 50 samples at a time

# Pricing (DeepSeek R1 reasoner)
INPUT_PRICE_PER_M = 0.55    # $/1M input tokens (cache miss)
OUTPUT_PRICE_PER_M = 2.19   # $/1M output tokens (including reasoning)

# Use 32k to match mainstream LLM benchmark comparison
MAX_TOKENS = 32768

METRIC_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

# === Metric functions (same as competition scorer) ===
def extract_final_answer(text):
    if text is None:
        return 'NOT_FOUND'
    matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()
    patterns = [
        r'The final answer is:\s*([^\n]+)',
        r'Final answer is:\s*([^\n]+)',
        r'Final answer\s*[::]\s*([^\n]+)',
        r'final answer\s*[::]\s*([^\n]+)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()
    matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    if matches:
        return matches[-1]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else 'NOT_FOUND'

def verify(stored_answer, predicted):
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()
    if re.fullmatch(r'[01]+', stored_answer):
        return predicted.lower() == stored_answer.lower()
    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        return predicted.lower() == stored_answer.lower()

# === Single sample inference with retry ===
MAX_RETRIES = 5

def process_sample(client, idx, row):
    prompt = row['prompt'] + METRIC_SUFFIX
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=MAX_TOKENS,
            )
            choice = response.choices[0].message
            reasoning_text = getattr(choice, 'reasoning_content', '') or ''
            answer_text = choice.content or ''
            raw_text = reasoning_text + "\n" + answer_text if reasoning_text else answer_text

            usage = response.usage
            input_tokens = usage.prompt_tokens or 0
            output_tokens = usage.completion_tokens or 0
            reasoning_tokens = getattr(usage, 'completion_tokens_details', None)
            if reasoning_tokens and hasattr(reasoning_tokens, 'reasoning_tokens'):
                reasoning_tokens = reasoning_tokens.reasoning_tokens or 0
            else:
                reasoning_tokens = 0
            break  # success
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)  # exponential backoff: 1, 2, 4, 8, 16s
                continue
            raw_text = f"ERROR: {str(e)}"
            answer_text = ""
            input_tokens = 0
            output_tokens = 0
            reasoning_tokens = 0

    predicted = extract_final_answer(answer_text if answer_text else raw_text)
    ground_truth = str(row['answer'])
    correct = verify(ground_truth, predicted)

    return {
        'idx': idx,
        'id': row['id'],
        'category': row['category'],
        'prompt': row['prompt'],
        'ground_truth': ground_truth,
        'predicted': predicted,
        'correct': correct,
        'raw_output': raw_text,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'reasoning_tokens': reasoning_tokens,
    }

# === Main ===
def main():
    if not API_KEY:
        print("ERROR: Set DEEPSEEK_API_KEY environment variable or edit the script.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load val set
    val_df = pd.read_csv(VAL_CSV)
    print(f"Val set: {len(val_df)} rows")
    print(val_df['category'].value_counts())

    # Init DeepSeek client (OpenAI-compatible)
    client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

    # Resume from partial results if they exist
    results = []
    partial_csv = os.path.join(OUTPUT_DIR, 'val_eval_results.csv')
    start_idx = 0
    if os.path.exists(partial_csv):
        partial_df = pd.read_csv(partial_csv)
        start_idx = len(partial_df)
        results = partial_df.to_dict('records')
        print(f"Resuming from sample {start_idx} (loaded {start_idx} partial results)")

    # Build work items
    work_items = []
    for i, row in val_df.iterrows():
        if i < start_idx:
            continue
        work_items.append((i, row))

    print(f"\nRunning {len(work_items)} samples with {CONCURRENCY} concurrent workers...")
    print(f"max_tokens={MAX_TOKENS} (same as competition)")
    t0 = time.time()

    completed = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {
            executor.submit(process_sample, client, idx, row): idx
            for idx, row in work_items
        }

        for future in as_completed(futures):
            result = future.result()
            with lock:
                results.append(result)
                completed += 1

                status = "OK" if result['correct'] else "WRONG"
                gt_safe = result['ground_truth'].encode('ascii', 'replace').decode()
                pred_safe = result['predicted'].encode('ascii', 'replace').decode()
                print(f"  [{completed}/{len(work_items)}] {status} cat={result['category']} "
                      f"gt={gt_safe} pred={pred_safe} "
                      f"tokens=in:{result['input_tokens']}/out:{result['output_tokens']}/reason:{result['reasoning_tokens']}")

                # Checkpoint every 50 completions
                if completed % 50 == 0:
                    # Sort by idx before saving
                    sorted_results = sorted(results, key=lambda x: x.get('idx', 0))
                    save_df = pd.DataFrame(sorted_results)
                    if 'idx' in save_df.columns:
                        save_df = save_df.drop(columns=['idx'])
                    save_df.to_csv(partial_csv, index=False)
                    print(f"  [checkpoint saved: {len(sorted_results)} results]")

    elapsed = time.time() - t0

    # Sort results by original index and save
    results = sorted(results, key=lambda x: x.get('idx', 0))
    # Remove idx column
    for r in results:
        r.pop('idx', None)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'val_eval_results.csv'), index=False)

    # Token totals
    total_input_tokens = int(results_df['input_tokens'].sum())
    total_output_tokens = int(results_df['output_tokens'].sum())
    total_reasoning_tokens = int(results_df['reasoning_tokens'].sum())

    # === Score ===
    overall_acc = results_df['correct'].mean()
    total_cost = (total_input_tokens * INPUT_PRICE_PER_M + total_output_tokens * OUTPUT_PRICE_PER_M) / 1_000_000

    print(f"\n{'='*60}")
    print(f"MODEL: {MODEL} (max_tokens={MAX_TOKENS})")
    print(f"OVERALL ACCURACY: {overall_acc:.4f} ({results_df['correct'].sum()}/{len(results_df)})")
    print(f"{'='*60}")
    print(f"\nPer-category breakdown:")
    print(f"{'Category':<30} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 60)
    for cat in sorted(results_df['category'].unique()):
        cat_df = results_df[results_df['category'] == cat]
        print(f"{cat:<30} {cat_df['correct'].sum():>8} {len(cat_df):>8} {cat_df['correct'].mean():>10.4f}")

    print(f"\n{'='*60}")
    print(f"TOKEN USAGE & COST")
    print(f"{'='*60}")
    print(f"  Input tokens:     {total_input_tokens:>10,}")
    print(f"  Output tokens:    {total_output_tokens:>10,}")
    print(f"  Reasoning tokens: {total_reasoning_tokens:>10,}")
    print(f"  Total cost:       ${total_cost:.4f}")
    print(f"  Time:             {elapsed:.1f}s ({len(work_items)/elapsed:.2f} samples/sec)")

    # Save cost summary
    cost_summary = {
        'model': MODEL,
        'max_tokens': MAX_TOKENS,
        'concurrency': CONCURRENCY,
        'total_samples': len(results_df),
        'overall_accuracy': round(overall_acc, 4),
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'reasoning_tokens': total_reasoning_tokens,
        'total_cost_usd': round(total_cost, 4),
        'elapsed_seconds': round(elapsed, 1),
    }
    with open(os.path.join(OUTPUT_DIR, 'cost_summary.json'), 'w') as f:
        json.dump(cost_summary, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()
