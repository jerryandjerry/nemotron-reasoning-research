# Why our Unsloth adapter (0.84) differs from huikang's — his Unsloth notebook (0.85) and his Tinker adapter (0.86)

**Date:** 2026-05-22 (Chicago). Updated after reading huikang's actual Unsloth notebook.
**Status:** Our adapter and his **Tinker** adapter verified at the byte level. Claims about his **Unsloth-notebook** adapter are *inferred from his code* — that adapter isn't in our files (only his Tinker one is).

> **✅ RESOLVED 2026-05-23 — see §7.** Training the Mamba `out_proj` LoRA live takes our **own** trained model from 0.84 → **0.86** (Score Tracker Sub #16), matching the wall without resubmitting his weights. `out_proj` (Issue A) was the missing piece. Two of §5's original guesses were wrong: **(a)** the tie is NOT wanted — tying experts rank-32 *hurts* (Sub #14 = 0.71); **(b)** the way to train `out_proj` is NOT `is_fast_path_available=False` (that OOMs) but `mixer.training=False` (the model's own unfused else-branch).

---

## TL;DR

Our training pipeline is a **port of huikang's own free Unsloth notebook** (`end-to-end-finetuning-for-lb-0-85.ipynb`, scores 0.85). The MoE-tying code, the fast-path patch, manual lm_head, and CCE are all **copied from his notebook**. There are **three** adapters to keep straight:

| Adapter | Route | `out_proj` | routed experts | LB |
|---|---|---|---|---|
| huikang **Tinker** (the 0.86 wall; `kienngx/…/tinker-adapter`) | Tinker API | **live, rank-32** | **tied, rank-32** | **0.86** (verified) |
| huikang **Unsloth notebook** | Unsloth | dead (inferred) | tied rank-32 (inferred) | 0.85 (his README) |
| **ours** | Unsloth | **dead** (verified) | **untied, rank-1** (verified) | 0.84 |

Two things separate our 0.84 from those:

1. **Dead `out_proj` is NOT our unique bug — it's in his Unsloth notebook too.** His notebook patches `is_fast_path_available = True` exactly like ours; that fused Mamba kernel uses the base `out_proj` weight, so the `out_proj` LoRA never trains. It's a property of the **Unsloth+Mamba route** (his and ours), and part of why Unsloth tops ~0.85 while his **Tinker** adapter (live `out_proj`) reaches 0.86.
2. **The expert rank-1 collapse is *his* tying code breaking under *our* Unsloth version.** His code assumes Unsloth exposes experts as a batched `[128, rank, in]` tensor (so `mean(dim=0)` ties across experts → rank-32). Our Unsloth version exposes **separate per-expert `[rank, in]`** tensors, so the same line collapses the rank instead → our rank-1, untied result. Same code, different environment.

**So "why didn't we use his exact code?" — we did.** We just ran it on a **different Unsloth/transformers version**. His notebook ships pinned wheels (`rtx-wheels/`, `nemotron-packages/`) = his exact environment; running it as-is on Kaggle reproduces his 0.85. Our AutoDL port on a newer Unsloth silently flipped the expert layout and broke his tying.

---

## 1. Background — the model and the three adapters

**Model:** `NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` — hybrid **Mamba-2 + attention + 128-expert MoE**, ~30B params / ~3B active. Dimensions (from the LoRA shapes): `d_model=2688`, vocab `131072`, routed expert MLP `2688→1856→2688` ×128, shared expert `2688→3712→2688`, attention GQA (`q→4096`, `k/v→256`), Mamba `in_proj→10304` / `out_proj 4096→2688`. The adapter spans 23 Mamba blocks, 6 attention blocks, 23 MoE blocks; the routed experts are the bulk of the weights.

**huikang has two training routes** (his README/REPORT):
- **Tinker** (`train_sft.py`, managed API) → train solve-rate 0.877; the served adapter (`huikang/nemotron-adapter` v20 → asalhi convert → `kienngx/…/tinker-adapter`) is the **0.86 wall**. Tinker handles LoRA correctly server-side.
- **Unsloth notebook** (`end-to-end-finetuning-for-lb-0-85.ipynb`, free, Kaggle RTX PRO 6000) → **0.85**. Manual MoE tying, manual lm_head LoRA, CCE — the techniques our pipeline copied.

**LoRA refresher (why the indicators matter):** `ΔW = B·A`, `A` random-init (full rank), `B` zero-init. A module is dead iff `‖B‖=0` (ΔW=0); an update is rank-1 if either factor is rank-1. Checking `lora_A` rank alone is blind to a dead `B` — that hole is why an earlier version of this report missed the dead `out_proj`.

---

## 2. Issue A — Mamba `out_proj` never trains (Unsloth route, his and ours)

**Verified (ours):** `out_proj.lora_B` is exactly 0.0 across all 23 Mamba blocks → ΔW=0, never adapted.
**Cause:** the Mamba CUDA fast path is on (`is_fast_path_available = True`). Its fused kernel applies `out_proj` from the **base** weight, so the PEFT `out_proj` LoRA is never in the autograd graph → no gradient → `lora_B` stays zero. `in_proj` (a plain Linear *before* the kernel) trains normally — that's the in_proj-live / out_proj-dead asymmetry.

**This is not unique to us:** huikang's Unsloth notebook contains the identical patch:
```python
nemotron_mod.is_fast_path_available = True
print("Patched is_fast_path_available = True")
```
So his **Unsloth** adapter almost certainly has a dead `out_proj` too (inferred — we don't have that adapter to check). His **Tinker** adapter has it live because Tinker doesn't use this fused path. The dead `out_proj` is therefore part of the **Unsloth-route ~0.85 ceiling vs Tinker's 0.86**, not a defect we introduced.

(The exact line in `modeling_nemotron_h.py` where the fast path consumes `out_proj.weight` is not personally read; the symptom + the in/out asymmetry + the shared patch make the attribution solid.)

---

## 3. Issue B — routed-expert LoRA collapses to rank-1 (our environment only)

**Verified (ours):** each routed expert's `up_proj.lora_A` and `down_proj.lora_B` are rank-1, and the 128 experts are **untied** (all different). Net: every routed-expert ΔW is rank-1 (vs huikang's rank-32).

**The tying code is huikang's, byte-for-byte** (we copied his notebook):
```python
should_tie = (is_w1 and is_A) or (is_w2 and is_B)   # tie up_A and down_B across experts
...
p.data.copy_(p.data.mean(dim=0, keepdim=True).expand_as(p.data))   # init
p.grad.copy_(p.grad.sum(dim=0, keepdim=True).expand_as(p.grad))     # each step
```
His own comment states the assumption:
> *"We keep Unsloth's batched `[num_experts, ...]` tensor layout; 'tying' means all 128 expert slices are kept identical."*

If experts are a **batched `[128, rank, in]`** tensor, `dim=0` is the expert axis → `mean(dim=0)` ties across experts and each `[rank, in]` slice stays **rank-32** (his intended, 0.85 result). In **our** Unsloth version, experts are **separate per-expert `[rank, in]`** params → `dim=0` is the rank axis → the same line **collapses each expert's factor to rank-1** and never ties them. Identical code, opposite outcome, driven entirely by the Unsloth expert-tensor layout.

So this is **not a bug we authored** — it's huikang's code running on a different Unsloth version than the one it was written for.

---

## 4. Evidence (byte-level — ours vs his Tinker adapter)

`‖lora_B‖` by module type (zero ⇒ dead):

```
module (lora_B)        HIS Tinker ‖B‖     OURS ‖B‖
Mamba out_proj         0.64 – 0.89        0.0000 – 0.0000   ← dead in ours (issue A)
Mamba in_proj          1.91 – 2.27        1.53 – 1.99
attention q/k/v/o      0.20 – 1.08        0.22 – 1.15
routed experts up/dn   0.52 – 0.81        0.38 – 1.69       (live, but rank-1 — issue B)
shared_experts up/dn   0.64 – 1.10        0.69 – 1.16
lm_head                11.30              3.82
```

Cross-expert tying + within-expert rank (all 128 experts, one MoE layer):

```
                       HIS Tinker         OURS
up_proj.lora_A         128/128 tied,r32   1/128 untied, rank-1
down_proj.lora_B       128/128 tied,r32   1/128 untied, rank-1
```

Both adapters store the identical 888,154,112 LoRA parameters; only the values differ. **Note:** this compares ours to his **Tinker** adapter (verified). His **Unsloth** adapter (the apt same-route comparison) isn't in our files — by his code it should match ours on `out_proj` (dead) but match his Tinker on experts (tied rank-32).

Scripts (in `study/`): `_tmp_lora_b.py` (‖B‖ per module — found issue A), `_tmp_verify_all.py` / `_tmp_rank.py` (tying + rank — issue B), `_tmp_keys.py` / `_tmp_compare.py` / `_tmp_scope.py`, `_tmp_read_nb.py` (parsed his notebook).

---

## 5. Why we didn't "use his exact code" — and the paths forward

We **did** use his code; we ran it in a **different environment**. His notebook bundles his exact wheels (`rtx-wheels/` 7.1 GB, `nemotron-packages/` 5.5 GB). Running his notebook **as-is on Kaggle with those wheels** reproduces his Unsloth 0.85 (correct tying, though still dead `out_proj`). Our AutoDL port on a newer Unsloth changed the expert layout and broke the tying.

- **Reproduce his 0.85 (legitimate own-model baseline):** run his notebook with his pinned wheels, unmodified.
- **Match the wall (0.86) without training:** that's only his Tinker adapter resubmitted — *not our work*, not a real result.
- **Beat 0.86 (the real goal):** ~~need both `out_proj` **live** and experts **tied rank-32**~~ **[SUPERSEDED — see §7]**. Original guess: (a) `is_fast_path_available = False` during training, and (b) tie experts rank-32. **Both wrong.** (a) `is_fast_path_available=False` → pure-torch `torch_forward` → OOM (126 GB at seq 8192); the right switch is `mixer.training=False` (model's own unfused else-branch). (b) tying experts rank-32 *hurts* (Sub #14 = 0.71); use **no tie** (independent rank-32 experts). What actually worked: **no tie + live `out_proj`** → 0.86 (Sub #16). Acceptance test before trusting any run still holds: `out_proj.lora_B ≠ 0` and expert rank = 32.

---

## 6. Verified vs inferred

- **Verified (weights):** our adapter — dead `out_proj`, untied rank-1 experts. His **Tinker** adapter — live `out_proj`, tied rank-32 experts.
- **Verified (code):** his Unsloth notebook patches `is_fast_path_available=True`; its MoE-tying code is byte-identical to ours; its comment assumes a batched expert layout; targets include `out_proj`; manual lm_head + CCE.
- **Inferred (not on weights — we lack his Unsloth adapter):** his Unsloth 0.85 adapter has dead `out_proj` (same patch) and tied rank-32 experts (his comment + his Unsloth version batches experts).
- **Not read:** the exact `modeling_nemotron_h.py` fast-path line; his data-gen solvers (`reasoners/`, `augmenters/`); `notebook_tinker.py`.
- **Score lineage:** 0.877 = his Tinker *train solve-rate* (his writeup); 0.86 = his Tinker *adapter* on the LB (we verified by submitting it); 0.85 = his Unsloth notebook (his README); 0.84 = our Unsloth runs **with dead `out_proj`** (0.86 once `out_proj` is trained — §7).

---

## 7. RESOLUTION (2026-05-23) — `out_proj` was the missing piece; 0.84 → 0.86 with our OWN model

We ran the three-way comparison §5 called for, on the Unsloth/4.56.2 stack, identical data, one variable at a time:

| Sub | routed experts | Mamba `out_proj` | LB |
|---|---|---|---|
| #14 | expert-direction tie, rank-32 (corrected) | dead | **0.71** |
| #15 | untied, rank-32 (no tie) | dead | **0.84** (endpoint & soup) |
| #16 | untied, rank-32 (no tie) | **LIVE, ‖B‖≈0.70** | **0.86** (endpoint) / 0.84 (soup) |

**Headline:** Sub #16's endpoint = **0.86** — the first time any of our *trained* adapters cleared 0.84 (every prior run #1/#4/#6/#7/#11/#15 topped at 0.84 across many endpoint/soup/resubmit draws). Our `out_proj.lora_B` trained to ‖B‖≈0.70, matching huikang's Tinker 0.86 magnitude (§4 table). **So Issue A (dead `out_proj`) was indeed the bulk of the Unsloth-0.84 vs Tinker-0.86 gap, and we close it with our own training — not by resubmitting his weights (Sub #13).** Honest caveat: the soup draw stayed 0.84 (within ±0.01 noise), so +0.02 is ~2 noise notches — but dead-`out_proj` never produced a 0.86 across many tries, so the weight of evidence is clear.

**Correction 1 — tying experts HURTS (don't do it).** §5 assumed we needed experts *tied* rank-32. The corrected expert-direction tie (Sub #14) scored **0.71** (−0.13 vs no-tie). Independent rank-32 experts (no tie, Issue B's "just don't tie" option) is correct. huikang's Tinker adapter happens to tie *and* score 0.86, but for our pipeline the tie is harmful — his 0.86 is driven by `out_proj` (+ Tinker training), not the tie.

**Correction 2 — the right way to train `out_proj`.** Not `is_fast_path_available=False` (→ pure-PyTorch `torch_forward`, materializes a ~126 GB SSM tensor at seq 8192 → instant OOM). The model already has an efficient unfused path: in `cuda_kernels_forward`, the fused `mamba_split_conv1d_scan_combined` (which folds `out_proj` via the raw weight → LoRA bypassed) is gated by `if self.training and cache_params is None:`; the **else-branch** does `causal_conv1d_fn` + `mamba_chunk_scan_combined` + `self.norm` + `out = self.out_proj(scan_output)` **as a module call** → LoRA applies. **Fix:** keep `is_fast_path_available=True`, set each Mamba mixer's `.training=False` after `for_training` → the mixer takes that else-branch → `out_proj.lora_B` gets gradients via the model's own CUDA kernels. No model-file patching; same ~88.7 GB VRAM and ~1 min/step as baseline. (The mixer's only `self.training` use is that gate; autograd ignores the flag; grad-checkpointing is block-level → unaffected. Independently audited VALID.) At inference Kaggle runs with a cache → `out_proj` is always a module call → the trained LoRA is consumed exactly as learned.

**Dead end (for the record):** transformers 5.x built-in `nemotron_h` exposes a clean `use_mem_eff_path` knob, BUT stores the 128 experts as packed `nn.Parameter` tensors that PEFT cannot LoRA-target (→ no per-expert LoRA, 235 keys vs ~12k) and uses `torch._grouped_mm` (Hopper sm_90 only; crashes on Blackwell sm_120). So we stayed on Unsloth/4.56.2 (individual `nn.Linear` experts, Kaggle-compatible) and used the `mixer.training=False` route there.

**Recipe to reproduce 0.86 (our own model):** no MoE tie + **live `out_proj`** (`mixer.training=False`); everything else identical to the 0.84 config — r32/α32/dropout0, targets q/k/v/o/up/down/in/out_proj + lm_head, AdamW lr 2e-4→0, seq 8192, bs 32 (micro 4, ga 8), 246 steps, `260514_huikang_golden_stripped.csv`. Script: `study/2605222039_sft_moe_outproj_rtx6000/train_moe_outproj.py` (env `MOE_TIE=0 USE_MEM_EFF=0`). Acceptance test: `out_proj.lora_B ≠ 0`, expert rank = 32, backbone key naming.

**Next, to go beyond 0.86:** `out_proj` closes the Unsloth↔Tinker architectural gap; further gains now need better/more-diverse data or recipe on top of live `out_proj` (per `STRATEGY_0.84_to_0.86.md`).
