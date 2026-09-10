# DPO distillation #26 — failure analysis

> Submission: ckpt 25/50/75/90 = **0.51 / 0.56 / 0.54 / 0.56** (SFT seed = 0.85). All four checkpoints regressed ≥0.29.

## TL;DR

Three things went wrong simultaneously:
1. **Data**: 72.3% of training pairs were near-identical chosen-vs-rejected — DPO had no preference to learn for them, only noise to propagate.
2. **Train/inference path mismatch**: `USE_MEM_EFF=1` during DPO ⇒ out_proj LoRA was inert in the forward path used to compute every gradient. But the SFT seed adapter (loaded verbatim into the saved adapter) was trained with `USE_MEM_EFF=0`, so the served adapter's out_proj LoRA is non-trivial. Other LoRAs were optimized against a forward path that the inference server doesn't use.
3. **Speed-of-damage**: ckpt-25 (28% trained) was already at 0.51. Damage is fast and structural; not over-training drift.

I had #1 in my hands (the diff% histogram) before launch and trained anyway. That's the operator error.

## Data composition

The combined chosen+rejected CSV (6360 joined rows; tokenized with the actual Nemotron-3-Nano-30B-A3B tokenizer) revealed:

| diff% bucket | 6360 rows | 1431 training-input rows |
|---|---|---|
| `[0, 10)` | 92.4% | **72.3%** (1034 pairs) |
| `[10, 25)` | 2.0% | 6.2% (89) |
| `[25, 50)` | 1.6% | 4.6% (66) |
| `[50, 75)` | 2.8% | 11.7% (168) |
| `[75, 90)` | 1.2% | 5.2% (74) |
| `[90, 100]` | 0.0% | 0.0% (0) |

**Where the real signal lives** (training-input rows with `diff_pct ≥ 25`):

| Category | Training pairs | ≥25% diff | Share with signal |
|---|---|---|---|
| cryptarithm_deduce | 243 | **242** | 99.6% |
| cryptarithm_guess | 22 | **22** | 100% |
| equation_numeric_deduce | 562 | 33 | 5.9% |
| equation_numeric_guess | 57 | 7 | 12.3% |
| bit_manipulation | 134 | 2 | 1.5% |
| gravity | 97 | 2 | 2.1% |
| unit_conversion | 100 | 0 | 0% |
| cipher | 151 | 0 | 0% |
| numeral | 65 | 0 | 0% |
| **TOTAL** | **1431** | **308 (21.5%)** | — |

**Reading**: only ~265 cryptarithm pairs carried real preference signal; the rest were either noise or near-zero-gradient samples that consume optimizer steps without teaching the model anything new about preference.

## Out_proj path mismatch

| | DPO run #26 | SFT 0.85 seed |
|---|---|---|
| `USE_MEM_EFF_PATH` | **1** (default) | 0 (set explicitly) |
| Fused Mamba kernel | **active** | bypassed |
| out_proj LoRA during forward | **inert** | active |
| `for_each(mixer).training=False` | **NOT applied** (no log line printed) | applied |

Saved adapter:
- out_proj LoRA values: **unchanged** from seed (it was inert, no gradient flowed)
- All other LoRAs (q/k/v/o_proj attn, up/down/in_proj Mamba, lm_head, MoE experts): **DPO-updated**

At Kaggle inference, the served model is `base + out_proj_seed + non_out_proj_DPO_updated`. The non-out_proj LoRAs were optimized assuming `out_proj_seed` was not in the forward — so they're tuned to compensate for whatever out_proj-shaped representation was missing. Add out_proj back at inference and the layers fight.

This is a real second-order effect on top of the data problem. The data problem is dominant — but this is fixable for free by setting `USE_MEM_EFF=0` next time.

## Training-time metrics looked fine, and yet

| Window | Mean loss | Reward gap | Chosen-wins |
|---|---|---|---|
| Steps 1–30 | 0.7000 | +0.017 | 60% |
| Steps 31–60 | 0.6817 | +0.056 | 67% |
| Steps 61–90 | 0.6639 | +0.101 | 80% |

Loss below `ln(2) = 0.6931`, monotone reward-gap growth, win-rate climb 60→85%. These say "DPO is learning the preference signal from the data you gave it." They cannot say "the preference signal is the right thing to learn."

When 78.5% of pairs have no real preference to learn, "preference learned correctly" still means the model has moved toward whatever weak bias exists in those near-identical pairs. The training stats are necessary, not sufficient.

## Why ckpt-25 was already at 0.51

Per-sequence reward magnitudes in the final third of training: chosen_r ≈ +0.18, rejected_r ≈ +0.08. With β=0.05, that means `logp_policy(chosen) − logp_ref(chosen) ≈ +3.6` and `…(rejected) ≈ +1.6` over the whole sequence. Both go UP. The policy makes both chosen AND rejected sequences ~5–36× more likely than the reference does.

That's the failure mode in plain language: the model has become more confident in everything that looks like the (chosen ≈ rejected) text it was trained on. For easy categories where the seed was already correct, "more confident" doesn't help (and can hurt by making rare-edge-case generations less diverse). For hard categories where the seed was wrong, "more confident in the seed's wrong answer" actively hurts.

25 optimizer steps × 16 pairs/step × bs=16 is enough movement at this β to overshoot for non-meaningful pairs. The score curve `ckpt-25=0.51 → ckpt-50=0.56 → ckpt-75=0.54 → ckpt-90=0.56` is mostly inference noise around a damaged equilibrium. The model never recovers to the seed within 90 steps.

## What I missed in pre-flight

I computed the diff% distribution earlier in the day, printed it (the user saw the table: "92.4% of pairs are <10% different"), and proceeded to train. The right pre-flight gate would have been:

> "If `meaningful_pair_count / training_input_count < 0.5`, do not launch; redesign the data."

With 21.5% I should have stopped.

## Recommended next experiment (not launched — for review)

1. **Re-filter PREFER∩REJECT to diff_pct ≥ 25** → ~308 pairs, mostly cryptarithm
2. Add **explicit oversampling of cryptarithm** (e.g. 3×) → ~700-900 training pairs
3. Set `USE_MEM_EFF=0` to match the SFT seed's path and the Kaggle inference path
4. Keep β=0.05, but consider lowering to 0.02 because cryptarithm gold is genuinely much better than model output and we want soft distillation, not aggressive preference enforcement
5. Train 1–2 epochs over the smaller filtered set
6. Save every 10 steps and submit a sweep similar to this run so we see the damage/recovery curve

I want explicit go/no-go from the user before launching any of the above. The SFT seed is at 0.85; any further DPO experiment is a risk of regression unless the data is clearly different from this run.
