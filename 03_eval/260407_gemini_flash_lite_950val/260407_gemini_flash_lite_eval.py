"""
260407 Gemini 3.1 Flash-Lite Eval on 950 val set
Thinking level: HIGH
Tracks token usage and cost per sample.
"""
import os, sys, time, math, re, json
import pandas as pd
from google import genai
from google.genai import types

# === Config ===
API_KEY = dict(l.strip().split('=', 1) for l in open('kaggle.env', encoding='utf-8-sig') if '=' in l and not l.startswith('#'))['GEMINI_API_KEY']
MODEL = "gemini-3.1-flash-lite-preview"
THINKING_LEVEL = "HIGH"
VAL_CSV = "D:/SynologyDrive/00_Kaggle/2026_Nemotron/01_data/10-90_split/260407_val.csv"
OUTPUT_DIR = "D:/SynologyDrive/00_Kaggle/2026_Nemotron/03_eval/260407_gemini_flash_lite_950val"

# Pricing (paid tier, prompts <= 200k)
INPUT_PRICE_PER_M = 0.25   # $/1M input tokens
OUTPUT_PRICE_PER_M = 1.50  # $/1M output tokens (including thinking)

METRIC_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

# === Metric functions ===
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

# === Main ===
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load val set
    val_df = pd.read_csv(VAL_CSV)
    print(f"Val set: {len(val_df)} rows")
    print(val_df['category'].value_counts())

    # Init Gemini client
    client = genai.Client(api_key=API_KEY)

    # Token tracking
    total_input_tokens = 0
    total_output_tokens = 0
    total_thinking_tokens = 0

    # Resume from partial results if they exist
    results = []
    partial_csv = os.path.join(OUTPUT_DIR, 'val_eval_results.csv')
    start_idx = 0
    if os.path.exists(partial_csv):
        partial_df = pd.read_csv(partial_csv)
        start_idx = len(partial_df)
        results = partial_df.to_dict('records')
        total_input_tokens = int(partial_df['input_tokens'].sum())
        total_output_tokens = int(partial_df['output_tokens'].sum())
        total_thinking_tokens = int(partial_df['thinking_tokens'].sum())
        print(f"Resuming from sample {start_idx} (loaded {start_idx} partial results)")

    t0 = time.time()

    for i, row in val_df.iterrows():
        if i < start_idx:
            continue
        prompt = row['prompt'] + METRIC_SUFFIX

        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=-1,  # HIGH = unlimited thinking
                    ),
                    temperature=1.0,
                    max_output_tokens=8192,
                ),
            )

            # Extract text (skip thinking parts)
            raw_text = ""
            thinking_text = ""
            for part in response.candidates[0].content.parts:
                if part.thought:
                    thinking_text += part.text
                else:
                    raw_text += part.text

            # Token usage
            usage = response.usage_metadata
            input_tokens = usage.prompt_token_count or 0
            output_tokens = usage.candidates_token_count or 0
            thinking_tokens = getattr(usage, 'thoughts_token_count', 0) or 0

            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            total_thinking_tokens += thinking_tokens

        except Exception as e:
            raw_text = f"ERROR: {str(e)}"
            thinking_text = ""
            input_tokens = 0
            output_tokens = 0
            thinking_tokens = 0

        predicted = extract_final_answer(raw_text)
        ground_truth = str(row['answer'])
        correct = verify(ground_truth, predicted)

        results.append({
            'id': row['id'],
            'category': row['category'],
            'ground_truth': ground_truth,
            'predicted': predicted,
            'correct': correct,
            'raw_output': raw_text,
            'thinking_output': thinking_text,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'thinking_tokens': thinking_tokens,
        })

        status = "OK" if correct else "WRONG"
        cost_so_far = (total_input_tokens * INPUT_PRICE_PER_M + total_output_tokens * OUTPUT_PRICE_PER_M) / 1_000_000
        gt_safe = ground_truth.encode('ascii', 'replace').decode()
        pred_safe = predicted.encode('ascii', 'replace').decode()
        print(f"  [{i+1}/{len(val_df)}] {status} cat={row['category']} gt={gt_safe} pred={pred_safe} "
              f"tokens=in:{input_tokens}/out:{output_tokens}/think:{thinking_tokens} "
              f"running_cost=${cost_so_far:.4f}")

        # Save partial results every 50 samples
        if (i + 1) % 50 == 0 or (i + 1) == len(val_df):
            pd.DataFrame(results).to_csv(partial_csv, index=False)
            print(f"  [checkpoint saved: {len(results)} results]")

    elapsed = time.time() - t0

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'val_eval_results.csv'), index=False)

    # === Score ===
    overall_acc = results_df['correct'].mean()
    total_cost = (total_input_tokens * INPUT_PRICE_PER_M + total_output_tokens * OUTPUT_PRICE_PER_M) / 1_000_000

    print(f"\n{'='*60}")
    print(f"MODEL: {MODEL} (thinking={THINKING_LEVEL})")
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
    print(f"  Input tokens:    {total_input_tokens:>10,}")
    print(f"  Output tokens:   {total_output_tokens:>10,}")
    print(f"  Thinking tokens: {total_thinking_tokens:>10,}")
    print(f"  Total cost:      ${total_cost:.4f}")
    print(f"  Time:            {elapsed:.1f}s ({len(val_df)/elapsed:.2f} samples/sec)")

    # Save cost summary
    cost_summary = {
        'model': MODEL,
        'thinking_level': THINKING_LEVEL,
        'total_samples': len(val_df),
        'overall_accuracy': round(overall_acc, 4),
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'thinking_tokens': total_thinking_tokens,
        'total_cost_usd': round(total_cost, 4),
        'elapsed_seconds': round(elapsed, 1),
    }
    with open(os.path.join(OUTPUT_DIR, 'cost_summary.json'), 'w') as f:
        json.dump(cost_summary, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")

if __name__ == '__main__':
    main()
