"""
Retry only the ERROR samples from val_eval_results.csv.
Uses lower concurrency (10) to avoid rate limits.
"""
import os, sys, time, math, re, json
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

API_KEY = dict(l.strip().split('=', 1) for l in open('kaggle.env', encoding='utf-8-sig') if '=' in l and not l.startswith('#'))['DEEPSEEK_API_KEY']
MODEL = "deepseek-reasoner"
OUTPUT_DIR = "D:/SynologyDrive/00_Kaggle/2026_Nemotron/03_eval/260408_deepseek_r1_950val"
VAL_CSV = "D:/SynologyDrive/00_Kaggle/2026_Nemotron/01_data/10-90_split/260407_val.csv"
CONCURRENCY = 10  # lower concurrency to avoid rate limits
MAX_TOKENS = 32768
MAX_RETRIES = 5
METRIC_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

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
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                print(f"    [retry {attempt+1}/{MAX_RETRIES}] idx={idx} waiting {wait}s: {str(e)[:80]}")
                time.sleep(wait)
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

def main():
    results_csv = os.path.join(OUTPUT_DIR, 'val_eval_results.csv')
    results_df = pd.read_csv(results_csv)
    val_df = pd.read_csv(VAL_CSV)

    # Find error rows
    error_mask = results_df['predicted'].str.startswith('ERROR', na=False)
    error_indices = results_df[error_mask].index.tolist()
    print(f"Found {len(error_indices)} error rows to retry")

    if not error_indices:
        print("No errors to retry!")
        return

    client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

    work_items = [(idx, val_df.iloc[idx]) for idx in error_indices]
    print(f"Retrying with {CONCURRENCY} concurrent workers...")
    t0 = time.time()

    completed = 0
    lock = threading.Lock()
    fixed = []

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {
            executor.submit(process_sample, client, idx, row): idx
            for idx, row in work_items
        }
        for future in as_completed(futures):
            result = future.result()
            with lock:
                completed += 1
                status = "OK" if result['correct'] else "WRONG"
                is_error = result['predicted'].startswith('ERROR')
                print(f"  [{completed}/{len(work_items)}] {status} cat={result['category']} "
                      f"gt={result['ground_truth']} pred={result['predicted'][:60]} "
                      f"{'(STILL ERROR)' if is_error else ''}")
                fixed.append(result)

    elapsed = time.time() - t0
    print(f"\nRetry done in {elapsed:.1f}s")

    # Patch results back into the dataframe
    for r in fixed:
        idx = r.pop('idx')
        for col, val in r.items():
            results_df.at[idx, col] = val

    results_df.to_csv(results_csv, index=False)

    # Report
    still_errors = results_df['predicted'].str.startswith('ERROR', na=False).sum()
    acc = results_df['correct'].mean()
    print(f"\nUpdated results saved. Still {still_errors} errors remaining.")
    print(f"Overall accuracy: {acc:.4f} ({int(results_df['correct'].sum())}/{len(results_df)})")

if __name__ == '__main__':
    main()
