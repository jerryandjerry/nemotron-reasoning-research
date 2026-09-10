# RLVR Research Report — Nemotron-3-Nano-30B-A3B → 0.87+

**Date:** 2026-05-30 (Chicago) • **Author:** Claude • **Status:** Pre-implementation; no run yet
**Goal:** Push Kaggle (NVIDIA Nemotron Model Reasoning Challenge) score from 0.86 → 0.87+ via RLVR starting from a SFT'd LoRA on `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
**Compute envelope:** 2× RTX PRO 6000 Blackwell, 96 GB each, on AutoDL (single box, no plan to scale).
**Reward source:** HuggingFace `math-verify` + our existing `text.rfind('}')` last-`\boxed{}` extractor.

This document consolidates the deep-research workflow run on 2026-05-27 (28 verified sources, 22/25 claims confirmed; agent_count=111) **plus** targeted follow-up searches on Unsloth, the "16-GPU floor" claim, and the open-source Kaggle landscape.

---

## TL;DR

| Decision | Pick | Why |
|---|---|---|
| **Framework — primary** | **Unsloth GRPO** (single-GPU + colocated/external vLLM) | SFT for this model already works in Unsloth (~60 GB VRAM, fits 96 GB); GRPO trainer is model-agnostic at API level; cheapest path to a working loop on 2× 96 GB. |
| **Framework — fallback** | **NeMo-RL v0.6.0** (DTensor/FSDP2) | First-party, shipped `grpo-nanov3-30BA3B-2n8g-fsdp2-lora.yaml`. *But* its lowest-scale shipped config is **2 nodes × 8 GPUs = 16 GPUs** — adaptation to 2 GPUs is unattested. |
| **Algorithm core** | **GSPO** (Qwen, arXiv 2507.18071) | Only published algorithm that empirically stabilizes MoE RL at 30B-A3B (token-level importance ratios collapse under ~10%/step expert-routing drift). |
| **Hyperparameter recipe** | **DAPO's four tricks** layered on top (arXiv 2503.14476) | No KL (removes reference-model memory cost), Clip-Higher (ε_low=0.2/ε_high=0.28), Dynamic Sampling (filter zero-advantage groups), Token-Level Loss. |
| **Reward** | **`math-verify`** + last-`\boxed{}` rfind extractor | Direct match to Kaggle eval; both libraries in our stack already. |
| **Strategic caveat** | **None of the top public submissions used RL** — the Progress Prize was won with pure SFT. RLVR on this competition is genuinely uncharted territory. | See §2.4 — re-evaluate whether better SFT data is a cheaper path to 0.87+ before committing GPU time to RL. |

---

## 1. Why RLVR at all? (and why the SFT alternative matters)

Our public LB sits at **0.86**, matching huikang's Tinker-trained adapter (Sub #13 = his weights submitted byte-for-byte) and our own first trained 0.86 (Sub #16 — no-tie + live `out_proj`). Multiple SFT runs at this recipe land at 0.84 (#4/#6/#7/#11/#15) with #16 = the favorable tail. Two intuitions argue RLVR is the right next lever:

1. **Reward-shaped post-training closes the gap between "produces correct-looking CoT" and "produces the right answer."** SFT can teach format and reasoning style; RLVR with `math-verify` directly optimizes for terminal correctness.
2. **MoE + long-CoT models have published RLVR lifts** on math benchmarks of comparable size: DeepSeek-R1, Qwen3-30B-A3B (per the GSPO paper), Tülu-3.

**Counter-argument worth taking seriously (§2.4 below):** the open-source Progress-Prize winning solution for this exact competition is **pure SFT, no RL**. The 0.86 wall has not been pierced by RLVR in public yet. Spending weeks on RL when nobody has demonstrated +0.01 from it on this challenge is a real risk.

---

## 2. Open-source landscape (Kaggle community)

What's actually published for this competition, ranked by relevance:

| Repo / Notebook | Approach | RLVR? | Notes |
|---|---|---|---|
| **[tonghuikang/nemotron](https://github.com/tonghuikang/nemotron)** — **Progress Prize winner** | EDA → augmentation → corpus → **SFT** → adapter upload | ❌ | `train_sft.py` is the only training script. No GRPO/RL code at all. Linked Kaggle writeup: [discussion/689915](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915). |
| **[SebAustin/NVIDIA-Nemotron-Model-Reasoning-Challenge](https://github.com/SebAustin/NVIDIA-Nemotron-Model-Reasoning-Challenge)** | `01_eda → 02_prepare_data → 03_train_lora → 04_evaluate → 05_package_submission` (QLoRA + vLLM eval) | ❌ | SFT only. |
| **[yunior123/nvidia-nemotron-reasoning](https://github.com/yunior123/nvidia-nemotron-reasoning)** | Competition pipeline | ❌ | Same SFT pattern. |
| Kaggle community notebooks (sarcasmos, etc.) | Inference / submission packaging | ❌ | No RL surfaced. |

### 2.4 The headline (uncomfortable)

**The Progress-Prize-winning approach was pure SFT.** No GRPO. No RLVR. The 0.86 wall everyone hits when resubmitting "the public adapter" is *that same SFT-trained adapter* (huikang's), byte-for-byte. The cheapest documented path to 0.86 is therefore: do better SFT on better data. If we want 0.87+, the most-attested lever is **SFT data curation**, not policy gradient.

**If we still want RLVR**, we are first publicly attempting it on this competition. That's a real shot at a novel angle (high-upside) but unambiguously high-friction — no shipped notebook, no proven recipe, and the open-feature-request signal ([unsloth#4711](https://github.com/unslothai/unsloth/issues/4711)) is what it looks like before something is supported.

---

## 3. Framework comparison (verified late 2025 / early 2026)

| Framework | LoRA-RL on 30B-A3B? | Multi-GPU strategy | Math-verify reward | Status / risk |
|---|---|---|---|---|
| **Unsloth + NeMo Gym** | ⚠️ SFT yes, GRPO trainer yes (model-agnostic API), **no shipped notebook for this 30B-A3B** — issue [#4711](https://github.com/unslothai/unsloth/issues/4711) open since 2026-03-31 | Single-GPU training; vLLM colocated or external endpoint. **No FSDP/TP for the training pass.** | Custom Python reward via NeMo Gym env or direct `reward_funcs` kwarg | Fewest moving parts. SFT-on-this-model already validated. Highest probability of "just working" once we port a Qwen3-4B-GRPO or Gemma3-4B-GRPO notebook upward. |
| **NeMo-RL v0.6.0** | ✓ Shipped configs `grpo-nanov3-30BA3B-2n8g-fsdp2-lora.yaml`, `…megatron-lora.yaml`; v0.7 roadmap = "Improved Large MoE Performance" | DTensor (FSDP2) or Megatron-Core; vLLM + SGLang rollouts | First-party; supports GRPO/GSPO/DAPO/GDPO with feature flags | **Lowest shipped config = 2 nodes × 8 GPUs = 16 GPUs.** No source attests successful 30B-A3B LoRA RL on 2 GPUs. v0.6.0 released 2026-04-30 — bleeding edge; pin specifically. |
| **verl** | ✓ LoRA+GRPO+FSDP/FSDP2 documented; rank ≥ 32 (rank 128 used at 32B-scale); `target_modules='all-linear'`; ships `examples/tuning/lora/run_qwen3_30b_a3b_megatron.sh` | FSDP/FSDP2 or Megatron-Core; vLLM ≥ 0.8.2 / SGLang | Function-based math-verify built in (v0.3.0.post0) | Qwen3-30B MoE LoRA Megatron path has documented bugs (issues #4303, #4370, #4990, #3906). Backup, not first. |
| **OpenRLHF** | ⚠️ LoRA-on-MoE was *refuted* in the deep-research verification (1-2 vote); the "Mixtral example" claim didn't survive | Ray + vLLM ≥ 0.19.1 with AutoTP/PP | Clean `reward_func(queries, prompts, labels) → {rewards, scores, extra_logs}` callback | **Best A/B harness** if you want to switch PPO/GRPO/RLOO/Dr.GRPO/REINFORCE++/DAPO/GSPO via flags — but verify LoRA support empirically before relying on it. |
| TRL `GRPOTrainer`, Unsloth GRPO ≤ 8B, AReaL, SkyRL, prime-rl | Viable for smaller models | n/a | n/a | Not where the 30B-MoE RLVR action is in 2025–26. |

### Status snapshot (verify-checked)

- **NeMo-RL** was the framework NVIDIA used to train Nemotron-3-Nano-30B-A3B itself ([NeMo-RL README](https://github.com/NVIDIA-NeMo/RL), [Nemotron-3-Nano Technical Report arXiv:2512.20848](https://arxiv.org/abs/2512.20848)). LoRA-GRPO is supported on both DTensor and Megatron-Core backends (verified 3-0).
- **verl** migrated from `volcengine/verl` → `verl-project/verl` in Jan 2026; first meetup hosted by Volcengine + NVIDIA Jan 10, 2026; actively maintained.
- **Refuted** in verification: verl's "supports 671B" claim (1-2 vote — don't quote), OpenRLHF "LoRA-on-MoE Mixtral example" (1-2 — verify empirically), GSPO "eliminates Routing Replay" (0-3 — MoE stabilization stands but "eliminates" is too strong).

---

## 4. Algorithm: GSPO + DAPO tricks

### 4.1 GSPO (Group Sequence Policy Optimization) — [arXiv:2507.18071](https://arxiv.org/abs/2507.18071), [Qwen blog](https://qwenlm.github.io/blog/gspo/)

GRPO's importance ratio is **per token**: `r_t = π_θ(y_t | y_<t, x) / π_old(y_t | y_<t, x)`. For MoE this breaks. Qwen *measured* on the architecturally-identical Qwen3-30B-A3B: **after each GRPO update, ~10% of activated experts differ between old and new policy for the same rollout**. The per-token ratio is then comparing apples to oranges, and the loss eventually collapses.

GSPO defines the ratio at the **sequence level**:
```
s_i(θ) = ( π_θ(y_i|x) / π_old(y_i|x) ) ^ (1/|y_i|)
```
Then clips, rewards, and optimizes at the sequence level. Two bonus engineering wins:
- **Precision-drift tolerance.** vLLM rollout log-probs and training-engine recomputed log-probs differ at the bf16-noise level. Sequence-level ratios are robust to that. **You can use vLLM's rollout-time log-probs directly** — no recomputation pass.
- **No Routing Replay hack** needed (which GRPO sometimes uses on MoE to replay rollout-time routing decisions during the gradient step).

**Honest caveat:** GSPO's MoE stabilization is established for Qwen3-30B-A3B (dense-attention + MoE). Nemotron-3-Nano-30B-A3B adds **Mamba2** layers. GSPO's interaction with state-space layers under policy gradient is not directly tested in public.

### 4.2 DAPO's four tricks (Decoupled clip + Dynamic sAmpling) — [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)

ByteDance Seed, March 2025. Their Qwen2.5-32B math reasoner recipe. All four tricks are independent and layer onto any importance-ratio scheme — including GSPO's.

1. **No KL penalty.** Removed entirely. Long-CoT distribution legitimately drifts away from the SFT anchor; KL drag fights that. **This is also the term that demands a frozen reference model in memory** — removing it is what makes 2-GPU 30B-MoE feasible at all.
2. **Clip-Higher.** Asymmetric PPO clip: `ε_low = 0.2`, `ε_high = 0.28`. The looser upper bound allows low-probability exploration tokens to be uplifted; the tighter lower bound still clips aggressive exploits. Prevents entropy collapse.
3. **Dynamic Sampling.** Filter prompt groups where every rollout scored 0 or every rollout scored 1. Under group-normalized advantage, those produce zero gradient — pure compute waste. Resample to refill.
4. **Token-Level Loss.** Average over all tokens, not sequences-then-batches. Sequence-level averaging under-weights long CoT and lets gibberish-tail samples blow up entropy.

Reference config (Qwen2.5-32B on 32 H100s): G=16 rollouts/prompt, prompt batch 512, LR 1e-6 constant AdamW, max gen 20480, soft-overlong buffer 4096. Scale down for 2 GPUs (see §8).

### 4.3 Why the combination, not either alone

| Failure mode | Addressed by |
|---|---|
| MoE token-routing drift collapses token-level ratios | GSPO (sequence-level ratio) |
| Reference model doesn't fit in 2× 96 GB | DAPO (no KL → no ref model in memory) |
| Entropy collapse on aggressive uplift / collapse on conservative clip | DAPO (Clip-Higher asymmetric ε) |
| Zero-gradient prompts wasting compute | DAPO (Dynamic Sampling) |
| Long CoT under-trained, tail gibberish blowing entropy | DAPO (Token-Level Loss) |
| Precision drift between rollout and training engines | GSPO (sequence-level ratio tolerates it) |

GSPO defines *what* to optimize (sequence-level), DAPO defines *how* (no-KL, asymmetric clip, dynamic sampling, token-level normalization). Orthogonal and composable. All three frameworks (NeMo-RL, verl, OpenRLHF) expose both as independent flags.

---

## 5. MoE / Mamba2 / out_proj gotchas

### 5.1 The `out_proj` "live vs dead" trick — **untested under RL**

In our SFT runs the `0.84 → 0.86` lever was forcing `mixer.training = False` on each Mamba mixer so the unfused else-branch runs and the `out_proj` LoRA module receives gradients (see [`260522_moe_expert_rank1_bug_report.md`](260522_moe_expert_rank1_bug_report.md)). Under fused Mamba kernels, `out_proj`'s LoRA is dead.

**No source I verified addresses whether this trick survives RL training.** It depends on whether the chosen framework's forward pass goes through `cuda_kernels_forward` at all under policy-gradient training. **The first instrumentation under RL must be:**
- Steps 1, 2, 3: log `OUT_PROJ.lora_B grad-L2`. Must be `> 0`. If it's 0, port the `mixer.training = False` patch.

### 5.2 MoE expert sparsity under small rollout batches

Small batch + 128 routed experts = many experts un-activated per step → noisy/zero gradient updates for those experts. Mitigations the literature points to:
- Bigger effective batch via gradient accumulation (already in our plan: 32× accum).
- Aux load-balancing loss kept on during RL — but the field hasn't fully settled. Start with whatever the Nemotron-3-Nano guide sets.
- See [langcopilot's MoE-post-training writeup](https://langcopilot.com/posts/2026-03-25-moe-post-training-load-balancing-routing-replay-expert-parallelism) for the routing-replay + EP + aux-loss state of the art as of March 2026.

### 5.3 Mamba2 under policy gradient

Genuinely unknown territory. No published reports of SSM-layer instability under RL specifically. Watch `grad_norm` on the SSM blocks for blowups.

---

## 6. LoRA-on-policy details

- **Rank 32 may be undersized for RL on 30B.** verl docs explicitly state rank 128 was needed at 32B-scale for "training convergence speed and final performance almost identical to non-LoRA." Plan: start at rank 32 (our SFT adapter — keeps the SFT lift) and bump to 64 → 128 if RL stalls.
- **No reference model** (no-KL DAPO) means only one base copy + active LoRA + a tiny snapshot of pre-rollout LoRA weights (or just cached rollout log-probs) as `π_old`. **Fits 2× 96 GB.**
- **LoRA collapse during RL** is documented ([thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/), [arXiv:2510.26788](https://arxiv.org/html/2510.26788v1), [TRL issue #3108](https://github.com/huggingface/trl/issues/3108), [kalomaze blog](https://kalomaze.bearblog.dev/rl-lora-ddd/)). Typical mitigations: lower LR + entropy bonus + early-stop on collapse.
- **Target modules.** Our SFT used `q/k/v/o + up/down/in/out_proj + lm_head`. verl recommends `target_modules='all-linear'` for RL — slightly broader. We keep our SFT targets to preserve the SFT lift.

---

## 7. Verifier: `math-verify`

[HF Math-Verify repo](https://github.com/huggingface/Math-Verify) implements symbolic numeric/string/latex verification with a sympy backend. Composes cleanly with our existing extractor:

```python
def reward_fn(prompts, completions, gold_answers):
    rewards = []
    for completion, gold in zip(completions, gold_answers):
        # 1. Last \boxed{} via rfind
        idx = completion.rfind('}')
        if idx == -1: rewards.append(0.0); continue
        start = completion.rfind('\\boxed{', 0, idx)
        if start == -1: rewards.append(0.0); continue
        pred = completion[start + 7 : idx]
        # 2. math-verify
        r = math_verify.parse_and_verify(pred, gold, …)
        rewards.append(1.0 if r else 0.0)
    return rewards
```

**Open: equation-form coverage, sympy backend gotchas, partial-credit shaping.** The deep-research run did not close this — needs a targeted follow-up. Bench `math-verify` against our gold column on the existing train set before relying on it as the sole reward.

**Reward shaping starting point:** pure 0/1 correctness. Add a format reward (`+ 0.1` if output contains a `\boxed{}`) only if the model starts breaking format. Skip length penalty; the soft-overlong DAPO mechanism handles it.

---

## 8. Recommended layout for 2× RTX PRO 6000 (96 GB each)

Two viable placements; we plan to start with **B**:

**A — Colocated on GPU 0** (Unsloth's default GRPO pattern). Unsloth shares the base weights between training and vLLM:
- GPU 0: base (~60 GB) + LoRA + grads + AdamW (~1 GB) + vLLM KV cache (~5–15 GB) ≈ **75–85 GB total**.
- GPU 1: idle.
- Pros: simplest. Cons: tight KV-cache budget — limits group size × generation length.

**B — Split: Unsloth on GPU 0, external vLLM server on GPU 1** ← **recommended**:
- GPU 0: Unsloth + base + LoRA + grads + AdamW (~62 GB used; ~34 GB headroom).
- GPU 1: standalone vLLM serving the same base for rollouts (single 96 GB card fits bf16 model + large KV cache).
- Pros: clean separation, much larger KV cache headroom → bigger G or longer gens. Cons: needs Unsloth GRPO's "external vLLM endpoint" mode (recent Unsloth feature) to be working.

**No tensor parallelism for the training pass.** Unsloth doesn't do FSDP/TP. (If we fall back to NeMo-RL, FSDP2 shards the trainable model — but that's the backup plan.)

---

## 9. Phase-1 config (initial; iterate from this)

| Knob | Starting value | Source |
|---|---|---|
| Framework | Unsloth GRPO (Layout B) | §3 + §8 |
| Algorithm | GSPO loss + DAPO 4 tricks | §4 |
| Base model | `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` (Unsloth-supported) | [unsloth docs](https://unsloth.ai/docs/models/nemotron-3) |
| Starting LoRA | our best SFT adapter (#16 or huikang's Tinker) | retain SFT lift |
| LoRA rank | 32 (bump → 64/128 if RL stalls) | verl recommendation |
| LoRA targets | `q/k/v/o + up/down/in/out_proj + lm_head` (SFT match) | preserve SFT lift |
| Group size G | **8** (DAPO 16 → halved for VRAM) | adapt |
| Prompt batch / step | **16** (DAPO 512 → infeasible) | adapt |
| Grad accumulation | **32** → effective ~4096 rollouts/step (DAPO 8192) | adapt |
| Max generation | **4096** (DAPO 20480 → infeasible) | adapt |
| Temperature / top_p | 1.0 / 1.0 | GRPO/DAPO default |
| LR | **1e-6 constant AdamW** | DAPO |
| KL | **none** | DAPO |
| Clip | **ε_low=0.2, ε_high=0.28** | DAPO Clip-Higher |
| Dynamic sampling | **on** | DAPO |
| Token-level loss | **on** | DAPO |
| Reward | `math-verify(rfind('\\boxed{')) → 0/1` | §7 |
| Gradient checkpointing | HF non-reentrant (DDP-safe; see SFT writeup) | SFT lessons |

### Step-1 instrumentation (don't skip)

- VRAM via `nvidia-smi`, per micro-batch, both GPUs.
- `OUT_PROJ.lora_B grad-L2 > 0` for steps 1, 2, 3 (or apply `mixer.training=False`).
- Per-group `grad_norm`; watch SSM blocks for blowups.
- Rollout-time log-probs vs training-engine log-probs at sample tokens (sanity: small diff, GSPO tolerates it).
- Reward distribution (% of rollouts hitting reward 1 in the first batch) — if all-0 or all-1, dynamic sampling will kill the run; adjust prompt mix.

### Convergence expectations

- DAPO's published Qwen2.5-32B → AIME'24 50 → 55+ in ~10k rollouts/prompt = a few-thousand prompt-steps. On our ~7k prompt set with G=8, expect to see meaningful score movement after **2–4k prompt-steps** if it works at all.

---

## 10. Watch-outs (collected from the verification)

1. **Reward hacking.** Math RLVR's classic failure: model learns to output `\boxed{42}` or similar default that scores nonzero on a tiny fraction. Monitor reward distribution per prompt category.
2. **KL blow-up / collapse.** No KL term in DAPO. The guardrails are entropy bonus + format reward + early-stop on reward plateau or entropy floor.
3. **Off-policy correction.** Rollout engine (vLLM bf16) and training engine drift in numerics. GSPO tolerates this natively. If we ever switch to GRPO/PPO, reintroduce log-prob recomputation pass.
4. **Version-pin minefield.** Pin `unsloth==2026.1.4`, `unsloth_zoo==2026.1.4`, vLLM ≥ 0.8.2 (verl-published floor) but consider 0.19.1+ if going OpenRLHF. PEFT and Transformers versions matter — Unsloth bundles its own; respect their pin.
5. **Reentrant vs non-reentrant gradient checkpointing.** Our SFT writeup (DDP investigation, EVALUATION.md for Sub #17) found these produce different weights. NeMo-RL's DTensor backend uses non-reentrant; Unsloth's uses reentrant. **The trained-weight signature WILL differ across frameworks** — don't expect a verl-trained adapter to match an Unsloth-trained one.
6. **Checkpointing during RL.** Save adapter only every N prompt-steps; full optimizer state is small (LoRA only). The disk-full failure mode from SFT closeout (see Sub #17 EVALUATION.md) applies again: budget output footprint before launch.
7. **MoE expert sparsity at small batch.** First steps may show many experts with zero or noisy LoRA gradient. Bump grad accumulation before bumping LoRA rank.

---

## 11. Reading priority

1. **[NeMo-RL Nemotron-3-Nano guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/nemotron-3-nano.md)** — only first-party recipe for this exact model.
2. **GSPO paper** — [arXiv:2507.18071](https://arxiv.org/abs/2507.18071). Algorithm core + the MoE motivation that decided this choice.
3. **DAPO paper** — [arXiv:2503.14476](https://arxiv.org/abs/2503.14476). The four hyperparameter tricks with rationale.
4. **[Unsloth GRPO tutorial](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/tutorial-train-your-own-reasoning-model-with-grpo)** — the actual code path we'll port from.
5. **[huikang's Kaggle writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915)** + [tonghuikang/nemotron](https://github.com/tonghuikang/nemotron) — read BEFORE committing to RLVR; understand what SFT achieved and what data they used.
6. **[verl LoRA-PPO docs](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html)** — backup framework rank/targets guidance.
7. **[Thinking Machines LoRA blog](https://thinkingmachines.ai/blog/lora/)** — LoRA collapse modes + mitigations.

---

## 12. Refuted (don't quote)

- **verl "supports 671B MoE"** — refuted 1-2 in verification.
- **OpenRLHF "LoRA-on-policy via Mixtral example is turnkey"** — refuted 1-2. Verify empirically before relying on it for LoRA-on-MoE.
- **GSPO "eliminates Routing Replay"** — refuted 0-3. The MoE stabilization claim stands; "eliminates" is too strong.

---

## 13. Open questions (follow-up needed before commit)

1. **`math-verify` coverage on our actual prompt distribution.** Bench on existing train CSV — % of gold answers it parses correctly, % it rejects valid forms.
2. **Does the `out_proj`-live trick survive Unsloth's RL forward pass?** Empirical only — step-1 diagnostic.
3. **Does Unsloth GRPO actually work on Nemotron-3-Nano-30B-A3B?** Issue #4711 is open. Adapt a Qwen3-4B-GRPO or Gemma3-4B-GRPO notebook and run a 10-step dry run before committing a real RL job.
4. **Is the SFT path cheaper to +0.01?** Read huikang's writeup; if his data strategy is reproducible and the lever is bit_manipulation or another data category, SFT may be the faster lever than RL.

---

*Companion documents: [TRAINING_ROUTINE.md](TRAINING_ROUTINE.md) (execution checklist), [260522_moe_expert_rank1_bug_report.md](260522_moe_expert_rank1_bug_report.md) (out_proj/MoE-tie mechanism), [study/SCORE_TRACKER.md](../study/SCORE_TRACKER.md) (run log).*
