# DPO Foundations — what we have to understand before another preference run

**Written:** 2026-05-31 (after run #26 regressed SFT 0.85 → 0.51–0.56 on all 4 ckpts)
**Author:** triage of failure, not a fresh literature review — every claim here cites a paper that was actually opened (not just abstracted) by parallel research agents.

This document is the foundation. The specific failed run is analyzed separately in [`study/2605301130_dpo_distill_alldata_rtx6000/260531_2319_submission/EVALUATION.md`](../study/2605301130_dpo_distill_alldata_rtx6000/260531_2319_submission/EVALUATION.md). Read this one first if you want to understand DPO well enough to design the next experiment.

---

## 1. Where DPO comes from

DPO is the closed-form solution to RLHF with a KL-constrained reward maximization objective:

$$
\max_{\pi_\theta}\ \mathbb{E}_{x \sim D,\ y \sim \pi_\theta}\big[r(x,y)\big] \;-\; \beta\, D_{KL}\big[\pi_\theta(y|x)\,\|\,\pi_{\text{ref}}(y|x)\big]
$$

The optimal policy is $\pi^*(y|x) \propto \pi_{\text{ref}}(y|x)\, \exp(\frac{1}{\beta} r(x,y))$. Inverting that gives $r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$. Plugging this into Bradley–Terry $P(y_w \succ y_l) = \sigma(r(y_w) - r(y_l))$ and taking negative log yields the DPO loss [1, §4]:

$$
\mathcal{L}_{\text{DPO}} \;=\; -\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} \;-\; \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)
$$

Two things to internalize:

- **The "reward" is implicit.** $\hat r(x,y) = \beta \log \pi_\theta(y|x)/\pi_{\text{ref}}(y|x)$ — DPO never trains a reward model. The policy IS the reward model.
- **The loss only sees the difference $\hat r(y_w) - \hat r(y_l)$.** Not the absolute log-probs. That single fact drives every failure mode in §3.

## 2. What the gradient actually does

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} \;=\; -\beta\, \sigma(\hat r(y_l) - \hat r(y_w))\, \big[\nabla_\theta \log\pi_\theta(y_w|x) \;-\; \nabla_\theta \log\pi_\theta(y_l|x)\big]
$$

[1, §4.1]. Two terms multiplied:

**(a) The sigmoid weight $\sigma(\hat r(y_l) - \hat r(y_w))$.**
- Model already ranks the pair correctly ($\hat r_w \gg \hat r_l$) ⇒ $\sigma \to 0$ ⇒ gradient vanishes.
- Model ranks them wrong ⇒ $\sigma \to 1$ ⇒ full gradient.
- **Chosen ≈ rejected ⇒ $\sigma \approx 0.5$ ⇒ half-strength gradient.**

**(b) The difference of token-level log-prob gradients.** Chain rule: $\nabla \log\pi_\theta(y|x) = \sum_t \nabla \log\pi_\theta(y_t|x, y_{<t})$. For any token shared between $y_w$ and $y_l$, the per-token gradient roughly cancels (same token, same context, same parameters). **The gradient concentrates on the tokens that differ.**

## 3. The fundamental failure mode — likelihood displacement

This is the load-bearing insight. The DPO loss decreases under any of three movements:

| What moves | Direction | DPO loss |
|---|---|---|
| $\log \pi_\theta(y_w)$ up, $\log \pi_\theta(y_l)$ down | intended | ↓ |
| Both up, $y_w$ more | benign | ↓ |
| **Both DOWN, $y_l$ more** | **likelihood displacement** | **↓** |

The third row is the failure mode [2, Theorem 1; 3, §3]. DPO can satisfy its objective by pushing the chosen probability *down*, as long as it pushes rejected *down more*. The KL term doesn't prevent this — KL penalizes ratio drift, not absolute drift.

**Why this happens specifically on near-identical pairs:** when $y_w$ and $y_l$ share most tokens, gradient concentrates on the few differing positions. Those gradient updates flow through *shared parameters* (lm_head, MoE experts, attention). The parameter updates pull the shared representation in a direction that helps the differing tokens — and that same direction often hurts the shared tokens, because shared tokens were already at the optimum the SFT seed found. Net effect: $\log \pi_\theta$ of the WHOLE chosen sequence collapses while $\log \pi_\theta(y_l)$ collapses faster.

**Empirical confirmation:**

| Paper | Pair distribution | Vanilla DPO outcome |
|---|---|---|
| Smaug [3, §3, Fig 4] | MetaMath, mean normalized edit distance = **6.5%** | Chosen log-prob went from $-0.37 \to -1.82$; rejected fell further; pair ratio improved but absolute capability collapsed. |
| Razin [2, Table 1] | Llama-3-8B, semantically similar chosen/rejected tokens | Chosen token probability dropped **0.99 → 0.03**. |
| Razin [2, §4.2.1] | — | The **CHES score** (centered hidden-embedding similarity) predicts displacement *better than edit distance*. Pairs that share semantic structure but differ in surface text still trigger displacement. |

This is what hit run #26. 72% of our pairs had ≤10% character-diff. The SFT seed had already learned to produce text near both chosen and rejected (both were near the gold). Gradient updates concentrated on minor differing tokens, flowed through lm_head + MoE experts, and pulled the entire policy off the SFT minimum.

## 4. What β actually controls

β appears in two places:

- Inside $\hat r = \beta \log(\pi/\pi_{\text{ref}})$ — scales the implicit reward magnitude.
- Inside the sigmoid $\sigma(\beta(\hat r_w - \hat r_l))$ — temperature of the preference decision.

Higher β = (1) bigger implicit rewards per unit log-prob movement, so smaller policy moves trigger the sigmoid → faster saturation → shorter effective learning window; AND (2) stronger anchoring because moving away from $\pi_{\text{ref}}$ inflates the reward magnitude proportionally to β, which the loss then tries to balance.

In the limit β → ∞ DPO collapses to pure SFT on the chosen response (no policy movement at all). In the limit β → 0 the KL anchor disappears entirely.

**Documented β values across the literature, with LR all at 5e-7:**

| Source | β | Extra | Notes |
|---|---|---|---|
| Original DPO [1, §6.1] | 0.1 | — | "default" |
| Zephyr [4, §4.4] | 0.1 | — | 64K pairs, length-norm, global batch 32 |
| Smaug [3, §5] | **0.3** | **λ=50** for DPOP penalty | Mistral-7B, on top of MetaMath |
| DA-DPO [5, §5.1] | **0.2** | per-sample upweighting on hard examples | LLaVA-1.5/OV |
| β-DPO [6] | adaptive per-batch | — | Explicit finding: **on low-margin pairs, lower β reduces win rate.** |
| **Our run #26** | **0.05** | — | **half the lowest documented working value** |

β-DPO [6] is the explicit citation that closes this question: **on low-gap (similar) pairs, higher β reduces win rate; for high-gap pairs, higher β improves it.** We picked β=0.05 thinking "soft distillation, gentle." We actually picked "weak anchoring, free drift." That's the second independent error in run #26 after the path mismatch.

## 5. What the alternatives change, mechanistically

| Method | Math change | What failure mode it addresses |
|---|---|---|
| **DPOP / Smaug** [3] | $\mathcal{L}_{\text{DPOP}} = -\log\sigma\big(\beta(\hat r_w - \hat r_l) - \lambda\cdot\max(0,\,\log\frac{\pi_{\text{ref}}(y_w)}{\pi_\theta(y_w)})\big)$ | Explicitly penalizes the chosen log-prob falling below reference. Direct attack on likelihood displacement. β=0.3, λ=50 documented working. |
| **IPO** [7] | $\mathcal{L}_{\text{IPO}} = (\hat r_w - \hat r_l - \tfrac{1}{2\tau})^2$ | Bounded squared loss with target margin $\tfrac{1}{2\tau}$. Once at target, loss is zero — can't keep pushing rejected probability to 0. Theoretically strongest fit for near-deterministic preferences. Empirical reproducibility mixed [HF blog, 8]. |
| **KTO** [9] | Per-sample, no pairs: $v(x,y) = \lambda_D\,\sigma(\beta(\hat r - z_0))$ for desirable, $\lambda_U\,\sigma(z_0 - \hat r)$ for undesirable; $z_0 = \text{KL}(\pi_\theta\|\pi_{\text{ref}})$ as moving anchor | Prospect-theoretic asymmetry. When model is already near reference on this sample, σ saturates → gradient → 0 → not punished for being already correct. **The only method documented to match SFT-then-method without SFT** [9, §6]. |
| **SimPO** [10] | Reference-free: $-\log\sigma\big(\tfrac{\beta}{\|y_w\|}\log\pi_\theta(y_w) - \tfrac{\beta}{\|y_l\|}\log\pi_\theta(y_l) - \gamma\big)$ | Length-normalized, no ref anchor at all. β values 2.0–2.5, γ 0.5–1.5. Authors warn this risks reward hacking [10, §6]; safety relies on **diverse data + LLM robustness**. Wrong choice when SFT seed is precious. |
| **ORPO** [11] | $\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}} + \lambda\cdot \mathcal{L}_{\text{OR}}$ where $\mathcal{L}_{\text{OR}} = -\log\sigma(\log\frac{\text{odds}_\theta(y_w)}{\text{odds}_\theta(y_l)})$ | NLL on $y_w$ explicitly fights displacement. λ ≈ 0.05–0.25, smaller for larger models. Designed to subsume SFT but applicable on top of an SFT'd model with small λ. **Does not need a separate SFT dataset** — uses $y_w$ from preference pair as the NLL target. |
| **R-DPO** [12] | DPO loss + $-(\alpha\|y_w\| - \alpha\|y_l\|)$ inside the σ | Orthogonal length regularizer. Stackable with anything. α ≈ 0.01 in original (HH/TL;DR scale); much smaller for long CoTs. |

**The pattern:** every successful variant either bounds the loss (IPO), explicitly anchors the chosen log-prob (DPOP, ORPO), avoids pairwise modeling (KTO), or removes a known bias (R-DPO length, SimPO length). Vanilla DPO has none of these.

## 6. Data construction — what the literature actually recommends

Three converging recipes from production-scale runs:

**Tülu 3** [13]:
- Pool of model completers, GPT-4o as judge, ~300K+ unique prompts (≈30% of SFT size).
- **Length-normalized DPO** as default.
- Three ablations that matter for us:
  1. *"Duplicating prompts with different responses does not significantly improve DPO performance, and may even degrade it."* → invest in unique prompts.
  2. *"Including on-policy data (text generations from the SFT model) improves downstream DPO performance compared to using only off-policy data."* → on-policy beats off-policy.
  3. New prompts beat reusing SFT prompts.

**NVIDIA HelpSteer2-Preference** [14]:
- Annotators rate on $[-3, +3]$.
- **Explicit drops:**
  - 22% of pairs with average preference ∈ near-zero band — *"a near-zero average indicates low-confidence preferences."*
  - 10% with inter-annotator spread > 2 — noisy margin.
- Final size: **7,118 pairs.** Reward model hits 94.1% RewardBench.
- Lesson: small + heavily filtered beats large + noisy. Drop the zero-margin pairs explicitly.

**Less-is-More / BeeS** [15]:
- Keep **top 10%** of UltraFeedback by Bayesian-aggregated margin (RM + DPO-implicit) → 3–8% AlpacaEval gain.
- **High-negative-margin pairs (chosen actually worse than rejected per RM) are actively harmful** — closest documented analogue to our chosen-≈-rejected case.

**Zephyr** [4, §3]:
- 64K pairs from UltraFeedback. Highest-score response as $y_w$, **random** lower-score as $y_l$ — *"to encourage diversity and make the DPO objective more challenging."* They actively avoided largest-margin pairs to add hardness. **Our problem is the opposite** — we got near-zero-margin pairs by accident.

**Step-DPO** [16] — most analogous to our reasoning-CoT setup:
- *"DPO has shown limited benefits for long-chain mathematical reasoning."* Their fix: localize the wrong step (single-step delta) rather than whole-CoT-vs-CoT pairs.
- 10K step-wise pairs, < 500 training steps, ~3% gain on MATH at 70B+.
- Implication: for long reasoning chains, meaningful signal requires localized differences, not a near-identical full chain.

**Synthesis — what a documented best-practice DPO pair pool looks like:**
- Filter on judged margin, not character diff (NVIDIA: drop the ≈-zero middle).
- Drop categories that produce no real disagreement (Tülu 3: unique prompts only).
- Generate $y_l$ on-policy via rejection sampling against the SFT seed (Tülu 3 ablation).
- For long CoTs, build pairs with localized step-level differences (Step-DPO).
- Total scale: small + curated > large + noisy (HelpSteer2 7K, BeeS 6K).

## 7. The principled diagnosis of run #26

Three independent failure modes stacked:

1. **Likelihood displacement regime [2, 3].** 72% of pairs ≤10% char-diff put us deep inside the Smaug/Razin failure mode. The seed's log-prob on chosen sequences was driven down despite training loss decreasing.
2. **Weak β anchoring [6].** β=0.05 was below every documented successful DPO run. The implicit-reward KL term existed but was too soft to prevent OOD drift on categories with no preference signal.
3. **Train/inference path mismatch.** `USE_MEM_EFF=1` (default) in DPO vs `USE_MEM_EFF=0` (SFT 0.85 seed) — out_proj LoRA was inert during gradient computation but active at Kaggle inference. Non-out_proj LoRAs were optimized for a forward path the inference scaffold doesn't use.

These compound. Path mismatch made every gradient point slightly off; weak β let those gradients move the policy far from the seed; data composition meant most of those gradients were pushing on directions with no real preference signal.

**RL's Razor** [17, §5.1] gives the macro-level confirmation: *offline preference optimization (DPO, SimPO) behaves similarly to SFT rather than on-policy methods in terms of catastrophic forgetting.* Our SFT 0.85 → 0.51 collapse is the textbook curve.

## 8. What "make it work" actually requires

Not "tweak β and rerun." The literature is pointing at a **design problem**, not a hyperparam problem.

**Minimum non-negotiable changes for the next experiment:**

1. **Use a method with explicit chosen-log-prob protection.** Either:
   - **DPOP** [3]: λ ≈ 50, β = 0.3. Minimal code change on top of vanilla DPO — one extra penalty term. Directly targets the Smaug failure mode that broke run #26.
   - **KTO** [9]: strongest documented seed preservation; the only method that matches "SFT + method" without SFT pretraining. Bigger code change, fundamental method shift.

2. **Fix the path mismatch.** `USE_MEM_EFF=0`. Free.

3. **Filter on margin.** NVIDIA HelpSteer2 recipe: drop near-zero-margin pairs. For us, the cheap proxy is `diff_pct ≥ 25`; the principled proxy is judging the surviving pairs with an LLM to confirm real preference. Result: 308 pairs (down from 1431).

4. **Don't expect anchor data from preference pairs.** Anchoring comes from (a) high enough β, (b) explicit NLL on chosen (ORPO), or (c) on-policy diversity (Tülu 3). Filling the pair set with near-identical "anchors" provides none of these — it's just noise.

**Strongly supported additions:**

5. **β ≥ 0.1** — match documented values.
6. **Length normalization** — cryptarithm CoTs span 2K–7K tokens; vanilla DPO is length-biased at this scale [12, 13].
7. **On-policy $y_l$** — generate from SFT seed; pair against gold $y_w$ [13].

**Honest expectation.** Even with all of the above, the literature does **not** promise a gain over a strong SFT seed. DA-DPO Table 1 [5] shows their method *recovers* general capability from vanilla DPO's regression but doesn't always *surpass* the SFT baseline. With our 308-pair filtered set (cryptarithm-dominated), the realistic best case is "match SFT 0.85 on easy categories, gain on cryptarithm." Setting that as the target — not "break 0.86" — is the principled framing.

---

## Sources

Primary papers (read in depth by parallel research agents):

1. **DPO** — Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, [arxiv 2305.18290](https://arxiv.org/abs/2305.18290)
2. **Likelihood Displacement** — Razin et al., *Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization*, [arxiv 2410.08847](https://arxiv.org/abs/2410.08847)
3. **Smaug / DPOP** — Pal et al., *Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive*, [arxiv 2402.13228](https://arxiv.org/abs/2402.13228)
4. **Zephyr / dDPO** — Tunstall et al., *Zephyr: Direct Distillation of LM Alignment*, [arxiv 2310.16944](https://arxiv.org/abs/2310.16944)
5. **DA-DPO** — *Cost-efficient Difficulty-aware Preference Optimization for Reducing MLLM Hallucinations*, [arxiv 2601.00623](https://arxiv.org/abs/2601.00623)
6. **β-DPO** — *β-DPO: Direct Preference Optimization with Dynamic β*, [arxiv 2407.08639](https://arxiv.org/abs/2407.08639)
7. **IPO** — Azar et al., *A General Theoretical Paradigm to Understand Learning from Human Preferences*, [arxiv 2310.12036](https://arxiv.org/abs/2310.12036)
8. **HuggingFace Preference Tuning blog** — DPO/IPO/KTO empirical comparison, [hf.co/blog/pref-tuning](https://huggingface.co/blog/pref-tuning)
9. **KTO** — Ethayarajh et al., *KTO: Model Alignment as Prospect Theoretic Optimization*, [arxiv 2402.01306](https://arxiv.org/abs/2402.01306)
10. **SimPO** — Meng et al., *SimPO: Simple Preference Optimization with a Reference-Free Reward*, [arxiv 2405.14734](https://arxiv.org/abs/2405.14734)
11. **ORPO** — Hong et al., *ORPO: Monolithic Preference Optimization without Reference Model*, [arxiv 2403.07691](https://arxiv.org/abs/2403.07691)
12. **R-DPO** — Park et al., *Disentangling Length from Quality in Direct Preference Optimization*, [arxiv 2403.19159](https://arxiv.org/abs/2403.19159)
13. **Tülu 3** — Lambert et al. (AI2), *Tülu 3: Pushing Frontiers in Open Language Model Post-Training*, [arxiv 2411.15124](https://arxiv.org/abs/2411.15124)
14. **HelpSteer2-Preference** — Wang et al. (NVIDIA), [arxiv 2410.01257](https://arxiv.org/abs/2410.01257)
15. **Less-is-More / BeeS** — Deng et al., *Less is More: Improving LLM Alignment via Preference Data Selection*, [arxiv 2502.14560](https://arxiv.org/abs/2502.14560)
16. **Step-DPO** — Lai et al., [arxiv 2406.18629](https://arxiv.org/abs/2406.18629)
17. **RL's Razor** — *RL's Razor: Why Online Methods Forget Less than Offline*, [arxiv 2509.04259](https://arxiv.org/abs/2509.04259)

Secondary sources cited in agent reports:

18. **DPO Survey** — *A Survey of Direct Preference Optimization*, [arxiv 2503.11701](https://arxiv.org/abs/2503.11701)
19. **MetaGDPO** — *Alleviating Catastrophic Forgetting with Metacognitive Knowledge through Group DPO*, [arxiv 2511.12113](https://arxiv.org/abs/2511.12113) (light on mechanism; surface-level citations only)
20. **Iterative DPO empirical** — *Enhancing LLM Reasoning with Iterative DPO: A Comprehensive Empirical Investigation*, [arxiv 2503.12854](https://arxiv.org/html/2503.12854v1)
21. **InCo-DPO** — [arxiv 2503.15880](https://arxiv.org/html/2503.15880v1)
22. **TRL KTO Trainer docs** — [github.com/huggingface/trl/blob/main/docs/source/kto_trainer.md](https://github.com/huggingface/trl/blob/main/docs/source/kto_trainer.md)
23. **Argilla / Mantis IPO writeup** — [argilla.io/blog/mantisnlp-rlhf-part-6](https://argilla.io/blog/mantisnlp-rlhf-part-6/)
24. **Queirozf summary of Azar et al.** — [queirozf.com/entries/paper-summary-...](https://queirozf.com/entries/paper-summary-a-general-theoretical-paradigm-to-understand-learning-from-human-preferences)
25. **Ritvik19 Tülu 3 explainer** — [medium.com/papers-explained-183-tulu-v3](https://ritvik19.medium.com/papers-explained-183-tulu-v3-fc7758b18724)
26. **Nathan Lambert Tülu 3 deep dive** — [interconnects.ai/p/tulu-3](https://www.interconnects.ai/p/tulu-3)

## Related local docs

- [`02_train/TRAINING_ROUTINE.md`](TRAINING_ROUTINE.md) — generic training routine (no DPO-specific guidance yet)
- [`study/2605301130_dpo_distill_alldata_rtx6000/260531_2319_submission/EVALUATION.md`](../study/2605301130_dpo_distill_alldata_rtx6000/260531_2319_submission/EVALUATION.md) — run #26 specific failure analysis
- [`study/SCORE_TRACKER.md`](../study/SCORE_TRACKER.md) — submission #26 entry
