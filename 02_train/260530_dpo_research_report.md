# DPO Research Report — Nemotron-3-Nano-30B-A3B → 0.87+

**Date:** 2026-05-30 (Chicago) • **Author:** Claude • **Status:** Pre-implementation; no run yet
**Goal:** Push Kaggle (NVIDIA Nemotron Model Reasoning Challenge) score from 0.86 → 0.87+ via offline preference optimization (DPO and variants) starting from a SFT'd LoRA on `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.
**Compute envelope:** 2× RTX PRO 6000 Blackwell, 96 GB each, on AutoDL (single box, no plan to scale).
**Verifier:** HuggingFace `math-verify` + our existing `text.rfind('}')` last-`\boxed{}` extractor.
**Companion document:** [`260530_rlvr_research_report.md`](260530_rlvr_research_report.md) — read alongside; the two are complementary phases of the same post-training pipeline, not alternatives.

Based on deep-research workflow run 2026-05-30 (24 verified sources, 19/25 claims confirmed; agent_count=106; 6 claims refuted in vetting).

---

## TL;DR

**The headline reframe: DPO and RLVR are not alternatives — they are sequential phases.** Tülu-3 ([arXiv:2411.15124](https://arxiv.org/abs/2411.15124)) — the canonical 2024 recipe with verifiable rewards — uses **SFT → DPO → RLVR**, and the AllenAI blog reports RLVR delivers **+1.7 on MATH, +3.3 on GSM8K, +1.3 on IFEval *on top of* the DPO checkpoint**. DPO is not redundant; it's the cheap intermediate stage that seeds RL with a lower-variance starting policy.

| Decision | Pick | Why |
|---|---|---|
| **Framework — primary** | **Axolotl** (paired with Transformers 4.x stack) | Only framework with documented, default `rl_adapter_ref_model: false` → routes through TRL's `disable_adapter` → no second 30B in VRAM. Single config switch between DPO / IPO / KTO / ORPO / SimPO / GRPO / GDPO. |
| **Framework — avoid (for DPO)** | Unsloth | SFT for this model works (~60 GB VRAM, fits 96 GB) but MoE doc page does NOT list Nemotron-3-Nano-30B-A3B, does NOT mention DPO, and disables router LoRA by default. Great SFT engine, wrong DPO engine. |
| **Framework — risk** | TRL `DPOTrainer` (Transformers 5.x) | Open bug [#5222](https://github.com/huggingface/trl/issues/5222) (2026-03-05): crashes when adding "ref" adapter on PEFT `target_parameters` — the only LoRA path for fused-expert MoE on Transformers 5.x. PR #5292 proposed but unmerged. Axolotl's default sidesteps it; pinning to Transformers 4.x sidesteps it too. |
| **Algorithm** | **Iterative / Online DPO with rule-based verifiable rewards** (Online-DPO-R1 / DPO-VP pattern) | Vanilla one-shot DPO documented to **struggle on long-chain math** (Full-Step-DPO, ACL 2025). Iterative outcome-DPO with `math-verify`-labeled best-of-N vs worst-of-N from on-policy rollouts is the simplest variant with published math results. Step-DPO / Full-Step-DPO only as fallback if iterative outcome-DPO plateaus. |
| **Reference model handling** | `model.disable_adapter()` context manager (PEFT) — *one base, two forwards* | No second 30B in memory. Documented in TRL since 2023 ([#624](https://github.com/huggingface/trl/issues/624)); default in TRL `DPOTrainer` when `ref_model=None`; default in Axolotl. |
| **Preference data** | Per prompt: 4–8 rollouts at T=0.7–1.0 → `math-verify` label → chosen=longest correct, rejected=random incorrect; resample each round | DPO-VP recipe. Scoring: +1 correct & formatted, 0 incorrect & formatted, −1 malformed. |
| **Strategy verdict** | **DPO first, then RLVR** | DPO is the better *first* bet (offline determinism, fits comfortably with `disable_adapter` ref, Tülu-3 proves independent gains). RLVR (GSPO+DAPO from the RL report) is the better *second* bet if 3–4 iterative DPO rounds don't deliver 0.86→0.87. |

---

## 1. The strategic landscape: DPO is not the lever, it's the pipeline stage before the lever

Going into this research I was treating DPO and RLVR as competing options. Two pieces of evidence flipped that:

1. **Tülu-3** ([arXiv:2411.15124](https://arxiv.org/abs/2411.15124)) abstract verbatim: *"The training algorithms for our models include supervised finetuning (SFT), Direct Preference Optimization (DPO), and a novel method we call Reinforcement Learning with Verifiable Rewards (RLVR)."* RLVR is defined as *"we replace the reward model in traditional RLHF with a scoring function that outputs a positive reward if the answer to the prompt is correct"* — i.e. **exactly** the `math-verify` + last-`\boxed{}` setup of this competition. AllenAI's blog confirms RLVR was *"implemented on top of DPO models"* and reports **+1.7/+3.3/+1.3** on MATH/GSM8K/IFEval over the DPO checkpoint.
2. ["It Takes Two: Your GRPO Is Secretly DPO"](https://arxiv.org/abs/2510.00977) (Oct 2025) argues GRPO's efficacy comes from an implicit contrastive (DPO-like) objective acting as a control variate; **a minimal 2-rollout GRPO retains 97.6% of 16-GRPO** at 12.5% of rollouts and 21% of training time on Qwen-1.5B/7B MATH. This is the theoretical bridge: the *pairwise contrast* is the dominant learning signal, not the on-policy nature. (Caveat: tested only at ≤7B, not 30B-MoE — confidence MEDIUM.)

**Honest read:** the cheapest order is **iterative DPO from our 0.86 SFT adapter** (cheap; offline; fits 2× 96 GB easily), then **optional GRPO/GSPO** if a 0.86→0.87 lift doesn't materialize after 3–4 DPO rounds.

**Caveat that doesn't go away:** As surfaced in the RLVR report, [huikang's Progress-Prize-winning code](https://github.com/tonghuikang/nemotron) is **pure SFT** — no DPO either. There is **no published DPO or RLVR submission** to this competition. The "DPO is cheap to try" argument assumes it works at all on this specific model. We are first publicly on both fronts.

---

## 2. Open-source landscape (Kaggle community, again)

| Repo / Notebook | Approach | DPO / RL? |
|---|---|---|
| **[tonghuikang/nemotron](https://github.com/tonghuikang/nemotron)** — Progress Prize winner | **SFT only.** No DPO, no RL. | ❌ |
| **[SebAustin/NVIDIA-Nemotron-Model-Reasoning-Challenge](https://github.com/SebAustin/NVIDIA-Nemotron-Model-Reasoning-Challenge)** | QLoRA SFT pipeline | ❌ |
| **[Hxrob/nemotron3-nano-finetune](https://github.com/Hxrob/nemotron3-nano-finetune)** | Generic Nemotron-3-Nano fine-tuning experiments | (forum-quality, low signal) |
| **[unsloth/notebooks/blob/main/DPO_Zephyr_Unsloth_Example.ipynb](https://huggingface.co/datasets/unsloth/notebooks/blob/main/DPO_Zephyr_Unsloth_Example.ipynb)** | DPO with Unsloth — **Zephyr 7B**, not Nemotron-3-Nano-30B-A3B | Closest existing template; would need porting (and Unsloth isn't the recommended DPO engine for our model — see §3). |
| **[NeMo-RL issue #1922](https://github.com/NVIDIA-NeMo/RL/issues/1922)** | NeMo-RL DPO support status | Relevant infrastructure issue, not a recipe. |
| **[docs.nvidia.com/nemo/rl/latest/guides/nemotron-3-nano.html](https://docs.nvidia.com/nemo/rl/latest/guides/nemotron-3-nano.html)** | First-party Nemotron-3-Nano guide | Focuses on GRPO, not DPO. |

**Same conclusion as the RL report:** there is no DPO recipe shipped for Nemotron-3-Nano-30B-A3B. The closest porting target is **Unsloth's `DPO_Zephyr` notebook** (smaller model, but the Unsloth DPO API is the same shape), and the closest *framework* recipe is **Axolotl's documented LoRA-DPO config** ([docs.axolotl.ai/docs/rlhf.html](https://docs.axolotl.ai/docs/rlhf.html)).

---

## 3. Framework comparison

| Framework | LoRA-DPO on 30B-A3B? | Reference handling | VRAM at 30B | Math-verify reward | Status / risk |
|---|---|---|---|---|---|
| **Axolotl** | ✓ Documented `rl: dpo/ipo/kto/orpo/simpo/grpo/gdpo/ebft` config switch | `rl_adapter_ref_model: false` (DEFAULT) → routes through TRL `disable_adapter` → **no second 30B in memory** | base ~60 GB + LoRA + grads + 2× forward activations — fits on 2× 96 GB DDP, tight on single 96 GB | Python preference data via `dataset.path` config | Best documented LoRA-DPO path for this scale; refuted that it *explicitly lists* Nemotron in supported models (verified 1-2) — supported via generic Qwen3/Llama 4/Mistral family configs. |
| **TRL `DPOTrainer`** | ⚠ Open bug **[#5222](https://github.com/huggingface/trl/issues/5222)** (2026-03-05): crashes when adding "ref" adapter on PEFT `target_parameters` — required for fused-expert MoE on Transformers 5.x. PR #5292 unmerged. | With `ref_model=None` + PEFT model: auto `disable_adapter` (working) | Same as Axolotl footprint if `ref_model=None` path is taken; +60 GB if a second LoRA adapter is added (the broken path) | Custom `compute_metrics` or formatter | Use only via Axolotl wrapper, or pin to Transformers 4.x where experts were still `nn.Linear` and `target_modules` works. Refuted that #5222 *explicitly lists* Qwen3-30B-A3B (verified 1-2). |
| **Unsloth DPO** | ⚠ SFT works for Nemotron-3-Nano-30B-A3B (60 GB; verified on RTX PRO 6000 Blackwell) but MoE doc page doesn't list this model under faster-MoE; **DPO not mentioned** on the MoE page; **router LoRA disabled by default**. | Unclear for 30B-MoE — Unsloth's DPO is well-tested on smaller models (Zephyr 7B). | 63 GB baseline → 80–85 GB at 1k–16k ctx for Qwen3-30B-A3B 16-bit LoRA (proxy) | Custom prompt/chosen/rejected | **Right engine for SFT, wrong engine for DPO** at this scale. |
| **verl** | ✗ verl is **on-policy RL only** (PPO/GRPO/RLOO/DAPO/GSPO/etc). No DPO. | n/a | n/a | n/a | Not a DPO tool. |
| **NeMo-RL / NeMo-Aligner** | ⚠ DPO supported in NeMo-Aligner (older codebase). NeMo-RL focuses on GRPO and GSPO/DAPO. Issue [#1922](https://github.com/NVIDIA-NeMo/RL/issues/1922) discusses DPO support. | NeMo-Aligner uses a separate frozen reference policy (heavyweight). | High — separate frozen ref = +60 GB. | First-party for Nemotron, but DPO-Aligner not adapted to 2-GPU. | Backup; not first-line. |
| **OpenRLHF** | Has a `DPOTrainer` and `KTOTrainer`; integrates Ray + vLLM for online flavors. | Configurable; can use `disable_adapter` pattern | Similar to Axolotl with PEFT | Custom callback | Viable A/B harness if Axolotl breaks. |

**Verified status snapshot:**
- Axolotl docs verbatim ([docs.axolotl.ai/docs/rlhf.html](https://docs.axolotl.ai/docs/rlhf.html)): *"TRL supports auto-unwrapping PEFT models … reference model log-probabilities can be obtained by disabling PEFT adapters. This is enabled by default. To turn it off, pass `rl_adapter_ref_model: true`"* and lists `rl: dpo/ipo/kto/orpo/simpo/grpo/gdpo/ebft` as switchable values.
- TRL maintainers (lvwerra, kashif) confirmed the `disable_adapter` pattern in 2023 ([#624](https://github.com/huggingface/trl/issues/624)). Now the documented `DPOTrainer` default: *"If `ref_model` is None, the trainer will automatically use the initial policy corresponding to model."*
- PEFT source (`peft/tuners/lora/model.py:277-283`) hardcodes *"only one LoRA adapter per model with target_parameters is allowed"* — the root of the #5222 crash. Workaround: don't add a second adapter; use `disable_adapter`.

---

## 4. Algorithm choice — Iterative / Online DPO with verifiable labels

The variant space breaks down as follows. Picking a winner per ([row priority for math RLVR scenarios](https://arxiv.org/abs/2502.14356) + verified comparisons):

| Variant | Reference model? | Length-normalized? | Math/reasoning evidence | Verdict for our setting |
|---|---|---|---|---|
| **Vanilla DPO** (Rafailov et al., 2023, [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)) | Yes | No | Full-Step-DPO (ACL 2025) explicitly: *"DPO often struggles with long-chain mathematical reasoning"* | Skip as a one-shot baseline. |
| **IPO** | Yes | No | Variant with explicit identity preference — robust to label noise, but no specific math superiority | Fallback if vanilla DPO over-fits the margin |
| **KTO** | Yes | No | Operates on individual examples (not pairs) — useful if you only have correctness labels per rollout without explicit pairs | Backup if pairing rollouts loses too many examples to filtering |
| **ORPO** | **No** | No | Reference-free; integrates SFT loss + odds-ratio term. Strong for low-resource | Backup if `disable_adapter` ref path breaks on MoE |
| **SimPO** | **No** | **Yes** | Reference-free + length-normalized — mitigates DPO's length bias directly | Strongest reference-free fallback; explicit length-bias guard |
| **Step-DPO / Full-Step-DPO** ([arXiv:2406.18629](https://arxiv.org/abs/2406.18629), [arXiv:2502.14356](https://arxiv.org/abs/2502.14356)) | Yes | Step-wise | Designed for long-chain math; requires step-level rewards | Only if iterative outcome-DPO plateaus and we can get step-level labels |
| **Iterative DPO with verifiable rewards** (Online-DPO-R1, DPO-VP) | Yes (`disable_adapter`) | Optional | DPO-VP on Qwen2.5-Math-7B reaches **48.2 average** across MATH500/Minerva/Olympiad/AMC23/AIME24 from 8K base prompts over 6 rounds — comparable to Qwen2.5-Math-7B-Instruct (47.5, with 2.5M SFT) and SimpleRL-Zero (48.8) | **Pick this.** |

### 4.1 The recommended core: Online-DPO-R1 / DPO-VP pattern

Quoting [RLHFlow/Online-DPO-R1](https://github.com/RLHFlow/Online-DPO-R1) verbatim: *"in every iteration, we sample responses … label rewards using the rule-based method … use the response with the highest reward and lowest reward as a preference pair."*

[DPO-VP](https://github.com/TU2021/DPO-VP) scoring: **+1 correct & formatted, 0 incorrect & formatted, −1 malformed.** Chosen = longest correct (or longest 0-score if none correct); rejected = random from lower-scoring; **no separate reward model**. Each round: regenerate, relabel, retrain. 6 rounds reached the 48.2 average.

**Hyperparameters (Tülu-3 / DPO-VP defaults, scaled to our setting):**

| Knob | Starting value | Note |
|---|---|---|
| `beta` (KL strength) | **0.1** | DPO-VP and Tülu-3 default; lower if reward margin saturates |
| Label smoothing | 0 | Add 0.1 only if chosen/rejected are very similar (mode collapse signal) |
| `max_prompt_length` | 1024 | Length-bias is a known DPO failure mode |
| `max_length` | 4096 (start) → 8192 (if needed) | Same caveat |
| LR | **5e-7** (Tülu-3 used this for DPO on 8B-class) | Scale down further if reward margin collapses |
| Per-device batch | 1 | 30B + 2 forwards (policy + ref) is heavy |
| Grad accumulation | 8–16 | Effective batch ≈ 16–32 pairs |
| Rollouts per prompt | 4–8 | DPO-VP used similar |
| Sampling T / top-p | 0.7–1.0 / 0.95 | T=0.7 starts conservative; raise if pairs are too similar |
| Iterative rounds | **3–6** | DPO-VP reached saturation around 6 |
| LoRA targets | same as SFT (q/k/v/o + up/down/in/out_proj + lm_head) | preserves SFT lift; out_proj-live trick presumed (must verify) |
| LoRA rank | 32 (bump to 64 if reward margin stalls) | Same as SFT |

### 4.2 Refuted claims I won't repeat

- **"PPO outperforms DPO/RAFT on math reasoning"** — refuted 0-3 in vetting. Don't quote this framing; the evidence base does not support a clean PPO-beats-DPO claim on math, especially with verifiable rewards.
- **"A single round of DPO with coarse filtering is enough to meaningfully boost mathematical reasoning"** — refuted 0-3. **Iterative is required**, not single-round.
- "Iterative DPO with verifiable rewards reaches 51.8% average on Qwen2.5-MATH-7B" — refuted 1-2. The actual reported number from Online-DPO-R1 is closer to 48.2; the higher quote was an inflation.
- "DPO-VP achieves RL-level performance significantly less compute" — refuted 1-2. SimpleRL-Zero scored 48.8 vs DPO-VP's 48.2 — comparable, but RL is *slightly* higher, not dominated.
- "TRL #5222 explicitly lists Qwen3-30B-A3B as affected" — refuted 1-2. The bug class is plausible-on-Nemotron but not directly confirmed there.

---

## 5. MoE / Mamba2 gotchas under DPO

### 5.1 Does DPO suffer the "routing drift" that GRPO does?

**Probably less, but not zero.** GRPO's per-token importance ratio collapses because router-induced expert sets drift between rollout-time and gradient-time (Qwen's ~10%/step measurement on Qwen3-30B-A3B). DPO is *offline* — preferences are computed against frozen rollouts. The ratio is between policy and reference (via `disable_adapter`), not between policy and old-policy-of-the-same-LoRA. **Routing drift between policy and reference is bounded by the LoRA delta itself, not by gradient updates.** So the GSPO-vs-GRPO MoE pathology is much smaller for DPO.

That said: the DPO loss does *two* forward passes (policy + reference) per pair, and on a fused-expert MoE both forwards must run through the same router. If the policy's LoRA-perturbed router routes differently than the reference's no-LoRA router, the log-prob ratio still has expert-mismatch noise. The implications:
- LoRA usually does NOT include the router (Unsloth disables router LoRA by default; verl excludes routers unless `router` is explicitly added to `target_modules`).
- So the router is the same module for policy and reference → routing decisions match → no expert-mismatch.
- Conclusion: **don't put LoRA on the router** for DPO. Match SFT targets.

### 5.2 The `out_proj` "live vs dead" trick under DPO

This is the same critical unknown as the RL report. Our SFT 0.84→0.86 lever was forcing `mixer.training = False` so each Mamba mixer takes the unfused else-branch and `out_proj` LoRA receives gradients (see [`260522_moe_expert_rank1_bug_report.md`](260522_moe_expert_rank1_bug_report.md)).

DPO's forward pass is **closer to SFT than to RL** (it's just a log-prob computation through the same model, twice). **So the trick should survive — but no source verifies this.** First instrumentation under DPO must be:
- Steps 1, 2, 3: log `OUT_PROJ.lora_B grad-L2` for both policy and reference forwards. Must be `> 0` on the policy. Apply `mixer.training=False` if not.

### 5.3 Aux load-balancing loss during DPO

Under DPO the gradient signal comes from the preference loss, not from token-level CE. Aux load-balancing loss should still be **kept on** (default) to maintain expert utilization — the field hasn't settled on disabling it for offline alignment, and switching it off mid-pipeline risks expert collapse.

---

## 6. Preference data generation — the actual question

For a verifiable-reward domain the right preference signal is **on-policy verifier-labeled best-of-N vs worst-of-N**, not human or judge preferences. The DPO objective then minimizes:

```
L_DPO = -log σ( β · [ log π_θ(y_chosen|x) - log π_ref(y_chosen|x) ]
                  -β · [ log π_θ(y_rej   |x) - log π_ref(y_rej   |x) ] )
```

with `chosen`/`rejected` from the verifier. This is **not equivalent to RLOO/REINFORCE** — DPO contrasts pairs in log-prob space against a frozen reference, while RLOO directly maximizes expected reward minus a leave-one-out baseline. ([It Takes Two](https://arxiv.org/abs/2510.00977) argues GRPO's effective loss includes an *implicit* DPO-like contrast — that's the bridge.)

### 6.1 Recipe

For each of our **7849 prompts**:
1. Sample **N=4–8 rollouts** with the current policy at T=0.7–1.0, top-p=0.95, max_new_tokens=4096.
2. Run `math-verify` on each rollout → reward ∈ {-1, 0, +1} (DPO-VP scoring).
3. **Chosen** = longest correct rollout (longest because length proxies CoT richness; if no correct, longest 0-score). **Rejected** = random incorrect.
4. Drop prompts where all rollouts have the same label (zero-margin → no DPO gradient → wasted compute, identical to DAPO's dynamic-sampling filter).
5. Form ~5k–7k pairs per round.
6. Train one DPO epoch on these pairs.
7. Repeat for **3–6 iterative rounds**.

### 6.2 Length-bias mitigation

DPO's well-documented failure mode: the policy learns "longer = better" because longer sequences tend to have higher log-probability mass to redistribute. Mitigations:
- **SimPO** (length-normalized loss) — fallback if vanilla DPO loss shows length blowup.
- Cap `max_length` at 4096 initially.
- Track chosen vs rejected lengths per round; if chosen avg length grows monotonically while rejected shrinks, switch to SimPO.

### 6.3 Mode collapse mitigation

If chosen and rejected become too similar, the loss saturates. Mitigations:
- Add 0.1 label smoothing.
- Increase rollout temperature to widen the distribution.
- Add a small entropy bonus.

### 6.4 What about off-policy / huikang's adapter as preferences?

A tempting shortcut: use *huikang's Tinker 0.86 adapter* outputs vs *our base bf16* outputs as chosen/rejected pairs. **The literature says don't.** Off-policy pairs from a stronger model are useful for *distillation* but DPO's KL constraint pulls the policy toward the chosen distribution; if chosen is a different model's distribution, the policy drifts off its own manifold. Stick to **on-policy** rollouts.

---

## 7. Memory layout on 2× 96 GB

Recommended layout (Axolotl + DDP):

| Component | GPU 0 | GPU 1 |
|---|---|---|
| Base bf16 backbone | ~60 GB (sharded ½ via DDP gather, or full per rank) | ~60 GB |
| Active LoRA (trainable) | ~MB | ~MB |
| AdamW state for LoRA params | ~MB | ~MB |
| Reference forward via `disable_adapter` | **Same backbone; zero extra cost** | Same |
| Per-rank batch=1, 2 forwards (chosen+rejected) | ~10–15 GB activations | ~10–15 GB |
| Gradient checkpointing (HF non-reentrant — DDP-safe per our SFT findings) | enabled | enabled |
| **Estimated peak** | **~75–85 GB** | **~75–85 GB** |

**No vLLM rollouts during training** — rollouts are generated *between* rounds, not concurrent with the gradient pass. That removes the colocation tension that complicated the RLVR layout.

**Generation phase** (between rounds): use vLLM with `tensor_parallel_size=1` on a single 96 GB card (bf16 base + LoRA hot-swap), sample 4–8 rollouts per prompt with `math-verify` labeling. Estimated time per round: 7849 prompts × 6 rollouts × ~2k avg tokens / ~80 tokens/sec ≈ **30–60 minutes** per generation phase + 1–2 hrs for DPO update = **~2–3 hrs per iterative round** = **8–18 hrs total** for 4–6 rounds.

---

## 8. Phase-1 config

| Knob | Value | Source |
|---|---|---|
| Framework | **Axolotl** (latest pinned) + Transformers **4.x** | §3 + §3 footnote on #5222 |
| Algorithm | **Iterative DPO** with verifier labels | §4 |
| Reference model | `disable_adapter` (default; `rl_adapter_ref_model: false`) | §3 + §6 |
| Base | `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` | model card |
| Starting LoRA | our best 0.86 SFT adapter (#16 or huikang Tinker) | retains SFT lift |
| LoRA rank | 32 → bump to 64 if reward margin stalls | matches SFT |
| LoRA targets | `q/k/v/o + up/down/in/out_proj + lm_head` (NOT router) | §5 + matches SFT |
| beta | **0.1** | DPO-VP / Tülu-3 default |
| Label smoothing | 0 (raise to 0.1 if mode collapse) | guard |
| LR | **5e-7** constant | Tülu-3 DPO setting on 8B |
| Per-device batch | 1 | 2 forwards on 30B is heavy |
| Grad accumulation | **8–16** | effective ~16–32 pairs/step |
| Max prompt | 1024 | length-bias guard |
| Max length | 4096 (raise to 8192 if needed) | length-bias guard |
| Rollouts per prompt | **4–8** | DPO-VP |
| Sampling T / top-p | 0.7–1.0 / 0.95 | DPO-VP |
| Iterative rounds | **3–6** | DPO-VP plateau |
| Gradient checkpointing | HF non-reentrant (DDP-safe; matches SFT writeup) | SFT lessons |
| Reward fn | `math-verify(last \boxed{...} via rfind)` → +1/0/−1 | matches Kaggle eval |
| Chosen / rejected | longest correct vs random incorrect | DPO-VP |
| Filter | drop prompts with all-same-label (zero margin) | matches DAPO dynamic sampling |

### Step-1 instrumentation

- `OUT_PROJ.lora_B grad-L2 > 0` for steps 1–3 (else apply `mixer.training=False`)
- VRAM via `nvidia-smi` per micro-batch on both GPUs
- Chosen vs rejected log-probabilities — both should be in a sane range, ratio should be > 1 immediately
- Length distribution of chosen vs rejected — flag if chosen length grows monotonically (length bias)
- Reward distribution per category (bit_manipulation, equation_numeric_deduce, cryptarithm, etc.) — drift signal

### Convergence expectations

- DPO-VP saturated at 48.2 on Qwen2.5-Math-7B after **6 rounds** over 8K prompts (closely matching our 7849 prompt set).
- Per-round score lift in our regime is unknown; if no score improvement appears after 3 rounds, the gain may be ceiling-bound and we should move to the RLVR phase.

---

## 9. Watch-outs

1. **Length bias.** DPO's classic failure. Monitor chosen-length vs rejected-length per round. Switch to SimPO if chosen length grows monotonically.
2. **Mode collapse.** If chosen and rejected start looking similar (small log-prob margin), add label smoothing 0.1 or raise rollout temperature.
3. **TRL #5222 lurking.** If we end up on Transformers 5.x with fused MoE, ensure Axolotl is using the `disable_adapter` path (default), not a second-adapter path. Test on a 10-step dry run.
4. **Reference-model VRAM blow-up.** If something forces a separate frozen-ref-model copy, the run won't fit — we'd OOM on the second 60 GB. Verify via `nvidia-smi` step-1 that only one base is in memory.
5. **LoRA collapse during DPO.** Same failure mode the RL report flagged. Lower LR + entropy bonus + early-stop are the standard mitigations.
6. **Reentrant vs non-reentrant gradient checkpointing.** Our SFT DDP investigation (see Sub #17 [EVALUATION.md](../study/2605231435_sft_moe_outproj_ddp_rtx6000/2605231810_submission/EVALUATION.md)) found these produce different weights at convergence. Use **HF non-reentrant** (DDP-safe). Don't mix engines mid-pipeline.
7. **Disk budget.** Each round generates ~7849 × 6 rollouts ≈ 50k completions × ~2k tokens average + LoRA snapshots. Pre-compute the output footprint; see [`feedback_budget_disk_soup_runs.md`](../../../.claude/projects/c--Users-YongsanHuang-SynologyDrive-00-Kaggle-2026-Nemotron/memory/feedback_budget_disk_soup_runs.md) — same lesson applies.
8. **`math-verify` coverage** is still an open question (unresolved by both research runs). Bench it against the gold column on the existing train set **before** committing a DPO round, otherwise we'll be training against a flawed signal.

---

## 10. Reading priority

1. **Tülu-3 paper** ([arXiv:2411.15124](https://arxiv.org/abs/2411.15124)) — the canonical SFT → DPO → RLVR recipe; defines RLVR for this exact setting.
2. **Online-DPO-R1** ([RLHFlow/Online-DPO-R1](https://github.com/RLHFlow/Online-DPO-R1)) — iterative DPO with rule-based verifiable rewards; closest published recipe.
3. **DPO-VP** ([TU2021/DPO-VP](https://github.com/TU2021/DPO-VP)) — math-specific iterative DPO, runnable codebase, +1/0/−1 scoring matches what we'll do.
4. **Axolotl RLHF docs** ([docs.axolotl.ai/docs/rlhf.html](https://docs.axolotl.ai/docs/rlhf.html)) — the actual framework config we'll use.
5. **Full-Step-DPO** ([arXiv:2502.14356](https://arxiv.org/abs/2502.14356), ACL 2025) — fallback if outcome-DPO plateaus on long-chain math.
6. **"It Takes Two: Your GRPO Is Secretly DPO"** ([arXiv:2510.00977](https://arxiv.org/abs/2510.00977)) — the theoretical bridge; informs how much marginal GRPO is worth after iterative DPO.

---

## 11. Refuted (don't quote)

- **"PPO outperforms DPO/RAFT on math reasoning"** — 0-3 vote.
- **"A single round of DPO with coarse filtering is enough to boost math"** — 0-3 vote.
- **"Iterative DPO reaches 51.8% on Qwen2.5-MATH-7B"** — 1-2; the verified DPO-VP number is 48.2.
- **"DPO-VP achieves RL-level performance with significantly less compute"** — 1-2; comparable but SimpleRL-Zero (48.8) is slightly higher.
- **"TRL #5222 explicitly lists Qwen3-30B-A3B as affected"** — 1-2; bug class is plausible-on-Nemotron, not directly confirmed there.
- **"Axolotl explicitly lists Nemotron-3-Nano-30B-A3B as supported under DPO"** — 1-2; supported under generic family configs, not by name.

---

## 12. Open questions (follow-up needed before commit)

1. **Does the `out_proj`-live `mixer.training=False` trick survive DPO's two forwards through Axolotl + DDP?** Empirical; step-1 grad-L2 instrumentation.
2. **Does TRL PR #5292 (fix for #5222) land before kickoff** — or does Axolotl's `disable_adapter` default fully sidestep it on Transformers 5.x fused experts? Re-check before starting.
3. **`math-verify` coverage on our actual prompt distribution.** Bench on existing train CSV; % gold answers parsed correctly, % valid forms rejected. Same open question as the RL report.
4. **Does 2-GRPO's "97.6% of 16-GRPO" finding** ([arXiv:2510.00977](https://arxiv.org/abs/2510.00977)) transfer to Nemotron-3-Nano-30B-A3B's hybrid Mamba2 + 128-expert MoE? Tested only at Qwen-1.5B/7B. This decides whether RLVR phase-2 is worth running after iterative DPO.
5. **Will Unsloth land DPO support for Nemotron-3-Nano-30B-A3B** (analogue to the open GRPO issue [#4711](https://github.com/unslothai/unsloth/issues/4711))? Worth a one-line check before committing to Axolotl.

---

## 13. Honest verdict — DPO vs RLVR for our specific situation

For **2× 96 GB + 7849 prompts + 0.86 SFT adapter + `math-verify` + 30B-A3B hybrid MoE**:

**DPO is the better first bet** because:
1. **Offline-deterministic** — no rollout-engine vs training-engine numerical drift, no MoE expert-routing drift across gradient steps; reproducibility is much easier.
2. **Fits comfortably on 2× 96 GB** via `disable_adapter` — no second 30B in memory, no co-location tension with vLLM.
3. **Tülu-3 confirms DPO provides independent gains** before RLVR — it's part of the pipeline, not a competing alternative.
4. **2-GRPO theory** ([arXiv:2510.00977](https://arxiv.org/abs/2510.00977)) argues the pairwise contrast itself is the dominant signal — meaning iterative DPO captures most of what RL would deliver, at lower cost.

**RLVR (GSPO + DAPO from the RL report) is the better second bet** if 3–4 iterative DPO rounds don't deliver 0.86 → 0.87.

**DPO is strictly preferred when**:
- We need offline determinism (reproducibility critical).
- We can't afford the on-policy rollout cost during training (we can amortize generation between rounds instead).
- We want a stable, low-variance intermediate checkpoint to seed RL later.

**RLVR is strictly preferred when**:
- DPO plateaus and we have headroom for more compute.
- We can confirm out_proj-live works under our chosen framework's RL forward pass.
- We accept the higher operational risk for higher upside.

---

*Companion documents: [260530_rlvr_research_report.md](260530_rlvr_research_report.md) (online RL phase), [TRAINING_ROUTINE.md](TRAINING_ROUTINE.md) (execution checklist), [260522_moe_expert_rank1_bug_report.md](260522_moe_expert_rank1_bug_report.md) (out_proj/MoE-tie mechanism), [study/SCORE_TRACKER.md](../study/SCORE_TRACKER.md) (run log).*
