# Evaluation Parameters — Verified from Kaggle

> Source: https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/overview/evaluation
> Scraped: 2026-04-07 via Playwright

## Official Evaluation Parameters (from Evaluation page)

| Parameter | Value |
|-----------|-------|
| max_lora_rank | 32 |
| **max_tokens** | **7680** |
| top_p | 1.0 |
| **temperature** | **0.0** |
| **max_num_seqs** | **64** |
| gpu_memory_utilization | 0.85 |
| **max_model_len** | **8192** |

## vs Metric Code Defaults (WRONG — do NOT use)

| Parameter | Metric Code Default | Actual Eval |
|-----------|-------------------|-------------|
| max_tokens | 3584 | **7680** |
| temperature | 1.0 | **0.0** |
| max_num_seqs | 128 | **64** |
| max_model_len | 4096 | **8192** |

## Confirmation

**Ryan Holbrook (Kaggle Staff)** in [Thread #682561](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/682561):
> "The parameters on the Evaluation page override the default parameters."

## Other Key Facts

- Input prompt capped at **~512 tokens**
- `'\nPlease put your final answer inside \boxed{}. For example: \boxed{your answer}'` is **automatically appended**
- **No system prompt** during inference — just the question + LoRA adapter
- You **cannot modify** the inference pipeline — only submit adapter weights
- Thinking tokens share the same max_tokens=7680 pool (no separate thinking budget)
- temperature=0.0 means **deterministic** output (greedy decoding)

## Implications for Our Eval

Our local eval notebook should use:
```python
llm = LLM(
    max_num_seqs=64,
    max_model_len=8192,
    ...
)
sampling_params = SamplingParams(
    temperature=0.0,   # NOT 1.0!
    top_p=1.0,
    max_tokens=7680,
)
```
