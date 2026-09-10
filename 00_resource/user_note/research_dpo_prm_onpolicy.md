# Research: DPO, Process Supervision, On-Policy Training for Greedy CoT

> Generated: 2026-05-14
> Context: Nemotron Kaggle competition, LoRA rank 32, RTX PRO 6000 (96GB)

## 1. Hard Negatives / DPO for Reasoning

**Key papers:**
- **IRPO** (NeurIPS 2024): Iterative DPO + NLL. Sample multiple CoTs, pair correct vs wrong. Llama-2-70B: 55.6% → 81.6% on GSM8K. https://arxiv.org/abs/2404.19733
- **Step-DPO** (NeurIPS 2024): Step-level preference pairs — find first diverging step between correct/incorrect traces. https://openreview.net/pdf?id=H5FUVj0vMd
- **DPO-ST** (ACL 2024): Self-training + DPO for CoT. https://github.com/TianduoWang/DPO-ST

**Practical for us:**
1. Run greedy on training puzzles
2. Collect wrong answers (verify with solver)
3. Pair: chosen=solver_correct_trace, rejected=model_greedy_wrong_trace
4. Train DPO+NLL with TRL's DPOTrainer
5. LR ~5e-7 (10-100x smaller than SFT)
6. LoRA rank 32 + DPO fits in 96GB with QLoRA 4-bit (~30-40GB)

## 2. Process Supervision (PRM)

**Key papers:**
- **Let's Verify Step by Step** (OpenAI, ICLR 2024): PRM800K. https://arxiv.org/abs/2305.20050
- **Math-Shepherd**: Automated step labels via MC estimation — sample N completions per step, fraction correct = step score. No human annotation.
- **PRIME** (2025): Train ORM on response-level labels, extract implicit per-step rewards. No step annotations needed. https://github.com/PRIME-RL/PRIME
- **OmegaPRM** (DeepMind): Divide-and-conquer MCTS to find first error. https://github.com/sanowl/OmegaPRM

**For our puzzles:**
- Math-Shepherd approach: at each step, sample N completions, verify final answer, fraction correct = step quality
- PRIME implicit PRM: train on full-solution correctness, get step rewards for free
- Binary search for first error: given wrong greedy trace, binary-search which step first diverges

## 3. On-Policy / Iterative Self-Improvement

**Key papers:**
- **STaR** (2022): Generate rationales → keep correct → rationalize wrong ones → fine-tune → repeat. https://arxiv.org/abs/2203.14465
- **V-STaR** (COLM 2024): Adds DPO verifier trained on both correct/incorrect. 4-17% improvement. https://arxiv.org/abs/2402.06457

**Practical:**
- 2-3 iterations is typically optimal
- Each iteration: fresh greedy rollouts → find errors → construct pairs → DPO train
- Stop when improvement < 1% per iteration
- Must use on-policy data (stale data degrades)

## 4. Search + Verify + Distill

**Key frameworks:**
- **ReST-MCTS*** (NeurIPS 2024): MCTS + PRM → distill via self-training. https://github.com/THUDM/ReST-MCTS
- **AlphaLLM-CPL**: Step-level pairs from MCTS tree nodes. 150% improvement on GSM8K.

**Simplest effective version for us:**
1. Sample N=16-64 solutions per puzzle (temperature=0.7)
2. Verify with puzzle checker
3. Keep correct ones
4. Fine-tune on them (rejection sampling + SFT)
5. Iterate

## 5. Feasibility on RTX PRO 6000

| Task | VRAM | Feasible? |
|------|------|-----------|
| SFT LoRA BF16 | ~60GB | Yes |
| SFT QLoRA 4-bit | ~25GB | Easy |
| DPO QLoRA 4-bit | ~35GB | Yes |
| GRPO QLoRA 4-bit | ~50-60GB | Tight |
| MCTS search (vLLM) | ~30GB | Yes |

## Recommended Strategy

1. **Stage 1 — SFT** on canonical CoT traces (current approach)
2. **Stage 2 — Iterative DPO** (highest ROI next step):
   - Greedy inference → collect failures → pair with solver traces → DPO+NLL
   - 2-3 iterations
3. **Stage 3 — GRPO** (if time permits): puzzle verifier as reward
