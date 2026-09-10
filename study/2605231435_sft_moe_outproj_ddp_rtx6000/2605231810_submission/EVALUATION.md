# EVALUATION — DDP 2-GPU replication of #16: why 0.84, not 0.86

**Submission:** ref 52968815 — `2026-05-23 DDP-2GPU repl #16 | no-tie + live out_proj | r32a32 seq8192 bs32 | 245steps final-246 endpoint | loss0.0025`
**Result:** Kaggle **0.84** (single-GPU #16 endpoint = 0.86). A real 0.02 gap (NOT inference noise — noise is ±0.01).

## Question
Why did the DDP replica score 0.84 when the byte-identical-recipe single-GPU run (#16) scored 0.86?

## Method
Three independent analyses (two via clean agents, one local script), plus a decisive baseline:
1. **Code-path audit** of `train_moe_outproj.py` (#16) vs `train_moe_outproj_ddp.py` (DDP).
2. **Training-log dynamics** comparison (#16 246-step log vs DDP attempt-1 245-step log).
3. **Per-module adapter-diff**: effective ΔW=B@A cosine + norm, DDP vs #16, computed exactly via r×r inner products (`<ΔW1,ΔW2>=Σ(B1ᵀB2)*(A1A2ᵀ)`), no out×in materialization.
4. **Baseline**: same adapter-diff between **two single-GPU runs of the same recipe** (#6 `2605151409` vs #11 `2605182305`, both 0.84, #11 a documented reproduction of #6) → measures *normal single-GPU run-to-run weight drift*.

## Findings

### The gradient math is equivalent (audit)
Per-rank `(loss/n_accum).backward()` with n_accum=4 + DDP all-reduce-MEAN across 2 ranks = single-GPU `/8` token-mean. The contiguous rank split (rank0=global[0:16], rank1=[16:32]) preserves the **identical 8 micro-batches** as single-GPU. No normalization/weighting bug. Real differences: reentrant (Unsloth, #16) → non-reentrant (HF, DDP) checkpointing; NCCL reduction order; dropped ragged final batch (9 samples, lr≈8e-7 — negligible); resume from ckpt-200 (DDP→DDP internal, since ckpt-200 was made by DDP attempt-1).

### The training dynamics match (logs)
**Step 1 bit-identical**: loss 0.403614 / grad_norm 0.1480 in BOTH. Loss stays within 0.0009 of #16 across all 245 steps (mean |diff| 0.0001), grad_norm ratio mean 1.05 / median 1.01 with only scattered minibatch noise, identical LR schedule. No compounding divergence; the diff actually *shrinks* late in training. out_proj is alive under DDP (grad-L2 0.046/0.034/0.022 at steps 1-3).

### The final weights: norm-preserving directional drift (adapter-diff)
| group | cos | relL2 | ‖dW16‖ | ‖dWddp‖ | norm ratio |
|---|---|---|---|---|---|
| attn | 0.956 | 0.293 | 2.287 | 2.243 | 0.981 |
| lm_head | 0.998 | 0.061 | 3.505 | 3.489 | 0.995 |
| mamba_in_proj | 0.964 | 0.266 | 5.623 | 5.588 | 0.994 |
| mamba_out_proj | 0.968 | 0.253 | 2.573 | 2.558 | 0.994 |
| moe_down | 0.970 | 0.247 | 25.68 | 25.63 | 0.998 |
| moe_up | 0.941 | 0.344 | 22.92 | 22.92 | 1.000 |
| **TOTAL** | **0.957** | **0.292** | 35.22 | 35.17 | **0.999** |

Every group's **magnitude is preserved** (ratio 0.98–1.00); only the **direction** drifts (~16°). **out_proj ‖B‖ per layer matches #16 almost exactly** (0.706 vs 0.700, 0.855 vs 0.852, … ratio ~0.994) — the 0.84→0.86 lever is fully intact, NOT under-trained. So the drift is diffuse, not a localized defect.

### DECISIVE: the drift equals single-GPU run-to-run drift
| comparison | TOTAL cos | relL2 |
|---|---|---|
| two single-GPU runs, same recipe (#6 vs #11) | **0.9555** | 0.2983 |
| DDP vs single-GPU #16 | **0.9575** | 0.2915 |

The DDP adapter is **no further from #16 than two independent single-GPU runs are from each other** — marginally *closer*, in fact. Per-group drifts match (moe_up 0.9421 vs 0.9409; lm_head 0.9986 vs 0.9982).

## Conclusion
**There is no DDP penalty.** DDP introduces no extra weight drift beyond ordinary single-GPU run-to-run training non-determinism. The 0.84 is **training trajectory variance**: this recipe converges into a flat loss basin (loss 0.0025 either way); different numerical paths — a second single-GPU seed *or* DDP — land on genuinely different weights (cos ~0.955 apart) that score differently on the eval.

**0.84 is the modal score for this recipe** (single-GPU #4/#6/#7/#11/#15 all = 0.84). #16's 0.86 was the favorable tail of the run-to-run distribution (and #10 drew 0.76 — the unfavorable tail). The DDP run drew 0.84 — the mode, not a degradation.

This is a *real weight difference* (cos 0.957, measured), **not inference noise** (±0.01) — but it is **not DDP-attributable**. It revises Sub #8's "the full 0.02 is DDP-attributable," which assumed single-GPU variance ≈ 0 from score-clustering but never measured single-vs-single *weights*; measured here, single-GPU weight variance is large and equal to DDP's.

**Caveat:** the baseline pair (#6/#11) is the older dead-out_proj recipe, and we have only N=1 live-out_proj single-GPU run (#16). But the drift mechanism is architectural (flat basin + bf16/atomicAdd non-determinism) and per-group drifts match across both recipes, so the baseline is representative.

## Implications
- **DDP is validated for this model**: same expected result as single-GPU at 2× speed. Use it.
- **To reach 0.86 reliably is NOT a parallelization problem.** 0.86 = the favorable tail of run-to-run variance (and = huikang's published-adapter wall). Beating the 0.84 mode needs a genuinely better recipe/data (e.g. the bit_manipulation lever per `STRATEGY_0.84_to_0.86.md`), not a DDP fix.
- A cheap way to *raise the floor* toward the better tail would be reducing trajectory variance (multi-seed soup), but #16's own SVD-soup scored 0.84 — soup hasn't beaten the mode here.
