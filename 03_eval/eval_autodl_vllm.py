"""
AutoDL vLLM Eval Script for non-Nemotron models on 950 val set.
Metric functions (extract_final_answer, verify) copied verbatim from official:
  nvidia-nemotron-metric.ipynb (metric/nvidia-nemotron-metric)

Only additions vs official metric:
  - vLLM config for local model (BNB quantization, no LoRA)
  - Save raw output per sample to CSV (thinking_output + raw_output columns)
  - Test mode (1 sample per category) via TEST_MODE=1 env var
  - Batch inference with checkpoint every 10 samples

Usage:
  TEST_MODE=1 python autodl_vllm_eval.py   # test run (6 samples)
  python autodl_vllm_eval.py               # full run (950 samples)
"""
import os, sys, time, math, re, json
import pandas as pd
from vllm import LLM, SamplingParams

# === Config (edit these per run) ===
MODEL_PATH = "/root/autodl-tmp/DeepSeek-R1-Distill-Qwen-32B-bnb-4bit"
MODEL_NAME = "DeepSeek-R1-Distill-Qwen-32B-bnb-4bit"
VAL_CSV = "/root/autodl-tmp/260407_val.csv"
OUTPUT_DIR = "/root/autodl-tmp/260411_deepseek_distill_950val"

# Eval parameters (match Kaggle eval page)
MAX_TOKENS = 7680
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_MODEL_LEN = 8192

METRIC_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'


# ============================================================
# Official metric functions — DO NOT MODIFY
# Source: nvidia-nemotron-metric.ipynb (metric/nvidia-nemotron-metric)
# ============================================================

def extract_final_answer(text: str | None) -> str:
    r"""Extracts the final answer from the model response.

    Prioritizes extracting answers inside `\boxed{}`.
    If no `\boxed{}` format is found, attempts to extract numbers from other formats.

    Examples:
        >>> extract_final_answer(r"The answer is \boxed{42}")
        '42'
        >>> extract_final_answer("The final answer is: 3.14")
        '3.14'
        >>> extract_final_answer("Just a number 100 in text")
        '100'
        >>> extract_final_answer(None)
        'NOT_FOUND'
    """
    if text is None:
        return 'NOT_FOUND'

    # Search for boxed answer
    # Match all instances of \boxed{...} or unclosed \boxed{ at the end
    matches = re.findall(r'\\boxed\{([^}]*)(?:\}|$)', text)
    if matches:
        non_empty = [m.strip() for m in matches if m.strip()]
        if non_empty:
            return non_empty[-1]
        return matches[-1].strip()

    # Other common formats if \boxed{} is not found
    patterns = [
        r'The final answer is:\s*([^\n]+)',
        r'Final answer is:\s*([^\n]+)',
        r'Final answer\s*[:：]\s*([^\n]+)',
        r'final answer\s*[:：]\s*([^\n]+)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    # If no structured format is found, extract the last valid number in the text
    matches = re.findall(r'-?\d+(?:\.\d+)?', text)
    if matches:
        return matches[-1]

    # If no numeric answer is found, return the last line of text as a fallback
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else 'NOT_FOUND'


def verify(stored_answer: str, predicted: str) -> bool:
    """Verify if the answer matches.

    For numerical answers, allow them to be judged as equal within a certain relative tolerance (1e-2);
    otherwise, compare strictly as strings (case-insensitive).

    Examples:
        >>> verify("10011000", "10011000")
        True
        >>> verify("10011000", "10011001")
        False
        >>> verify("24.64", "24.6401")
        True
        >>> verify("XLVII", "xlvii")
        True
        >>> verify("11011", "00011011")
        False
    """
    # Clean up strings
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    # If the answer is a binary string, compare strictly as strings
    if re.fullmatch(r'[01]+', stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        # Try to convert the answers to floating point numbers
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        # Use a small absolute tolerance for numbers near zero
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except Exception:
        # Fallback to case-insensitive string comparison
        return predicted.lower() == stored_answer.lower()


# ============================================================
# End of official metric functions
# ============================================================


def split_thinking_answer(raw_text):
    """Split raw output into thinking_output and raw_output (answer) based on </think> tag."""
    if '</think>' in raw_text:
        parts = raw_text.split('</think>', 1)
        thinking = parts[0]
        if thinking.startswith('<think>'):
            thinking = thinking[len('<think>'):]
        return thinking.strip(), parts[1].strip()
    return '', raw_text.strip()


def process_output(output, ground_truth, sample_id, category):
    """Process a single vLLM output into a result dict."""
    raw_text = output.outputs[0].text
    thinking, answer_text = split_thinking_answer(raw_text)
    predicted = extract_final_answer(raw_text)
    correct = verify(ground_truth, predicted)
    return {
        'id': sample_id,
        'category': category,
        'ground_truth': ground_truth,
        'predicted': predicted,
        'correct': correct,
        'thinking_output': thinking,
        'raw_output': answer_text,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load val set
    val_df = pd.read_csv(VAL_CSV)
    print(f"Val set: {len(val_df)} rows")
    print(val_df['category'].value_counts())

    # Load vLLM
    print(f"Loading model from {MODEL_PATH}...")
    t_load = time.time()
    llm = LLM(
        model=MODEL_PATH,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
        enforce_eager=True,
    )
    print(f"Model loaded in {time.time() - t_load:.1f}s")

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        skip_special_tokens=False,
    )

    # Build prompts
    tokenizer = llm.get_tokenizer()
    prompts = []
    for _, row in val_df.iterrows():
        user_content = row['prompt'] + METRIC_SUFFIX
        try:
            prompt = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = user_content
        prompts.append(prompt)
    print(f"Built {len(prompts)} prompts")

    # === TEST MODE ===
    test_mode = os.environ.get('TEST_MODE', '0') == '1'
    if test_mode:
        test_indices = val_df.groupby('category').apply(lambda x: x.index[0]).tolist()
        test_prompts = [prompts[i] for i in test_indices]
        test_df = val_df.iloc[test_indices].reset_index(drop=True)
        print(f"\n=== TEST MODE: {len(test_prompts)} samples (1 per category) ===")

        t0 = time.time()
        outputs = llm.generate(test_prompts, sampling_params)
        elapsed = time.time() - t0

        results = []
        for j, output in enumerate(outputs):
            r = process_output(output, str(test_df.iloc[j]['answer']),
                             test_df.iloc[j]['id'], test_df.iloc[j]['category'])
            results.append(r)
            status = "OK" if r['correct'] else "WRONG"
            print(f"  [{j+1}/{len(test_prompts)}] {status} cat={r['category']} "
                  f"gt={r['ground_truth'][:50]} pred={r['predicted'][:50]} "
                  f"think_len={len(r['thinking_output'])} answer_len={len(r['raw_output'])}")

        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(OUTPUT_DIR, 'test_run_results.csv'), index=False)
        print(f"\nTest run done in {elapsed:.1f}s")
        print(f"Results: {sum(r['correct'] for r in results)}/{len(results)} correct")
        print(f"Saved to {OUTPUT_DIR}/test_run_results.csv")
        return

    # === FULL MODE ===
    results = []
    partial_csv = os.path.join(OUTPUT_DIR, 'val_eval_results.csv')
    start_idx = 0
    if os.path.exists(partial_csv):
        partial_df = pd.read_csv(partial_csv)
        start_idx = len(partial_df)
        results = partial_df.to_dict('records')
        print(f"Resuming from sample {start_idx}")

    BATCH_SIZE = 10
    t0 = time.time()
    print(f"Running inference on {len(prompts) - start_idx} remaining samples (batch_size={BATCH_SIZE})...")

    for batch_start in range(start_idx, len(prompts), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]

        bt0 = time.time()
        outputs = llm.generate(batch_prompts, sampling_params)
        bt_elapsed = time.time() - bt0

        for j, output in enumerate(outputs):
            i = batch_start + j
            r = process_output(output, str(val_df.iloc[i]['answer']),
                             val_df.iloc[i]['id'], val_df.iloc[i]['category'])
            results.append(r)
            status = "OK" if r['correct'] else "WRONG"
            print(f"  [{i+1}/{len(prompts)}] {status} cat={r['category']} "
                  f"gt={r['ground_truth'][:30]} pred={r['predicted'][:30]}")

        # Checkpoint
        pd.DataFrame(results).to_csv(partial_csv, index=False)
        elapsed = time.time() - t0
        rate = (len(results) - start_idx) / elapsed if elapsed > 0 else 0
        eta = (len(prompts) - len(results)) / rate / 60 if rate > 0 else 0
        print(f"  [batch {batch_start}-{batch_end}] {bt_elapsed:.1f}s | "
              f"total {len(results)}/{len(prompts)} | {rate:.2f} samples/s | ETA {eta:.0f}min")
        sys.stdout.flush()

    elapsed = time.time() - t0

    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'val_eval_results.csv'), index=False)

    # Score
    overall_acc = results_df['correct'].mean()
    print(f"\n{'='*60}")
    print(f"MODEL: {MODEL_NAME} via vLLM")
    print(f"OVERALL ACCURACY: {overall_acc:.4f} ({results_df['correct'].sum()}/{len(results_df)})")
    print(f"{'='*60}")
    print(f"\nPer-category breakdown:")
    for cat in sorted(results_df['category'].unique()):
        cat_df = results_df[results_df['category'] == cat]
        print(f"  {cat:<30} {cat_df['correct'].sum():>4}/{len(cat_df):<4} = {cat_df['correct'].mean():.4f}")
    print(f"\nTime: {elapsed:.1f}s ({len(val_df)/elapsed:.2f} samples/sec)")

    # Save summary
    summary = {
        'model': MODEL_NAME,
        'inference': f'vLLM + bitsandbytes',
        'max_tokens': MAX_TOKENS,
        'temperature': TEMPERATURE,
        'max_model_len': MAX_MODEL_LEN,
        'total_samples': len(val_df),
        'overall_accuracy': round(overall_acc, 4),
        'elapsed_seconds': round(elapsed, 1),
    }
    with open(os.path.join(OUTPUT_DIR, 'eval_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
