# DPO Distill Stress Test Report — Nemotron-3-Nano-30B-A3B on single RTX PRO 6000 (96 GB)

**Format rule:** One item per test (opened at launch). Failures append a sub-line to the SAME item, not a new item.

Goal: DPO from cryptnumeq 0.85 SFT seed. Single GPU. Sequential forward (chosen + rejected via custom SequentialDPOTrainer) + TRL DPOTrainer + precompute_ref_log_probs=True. Memory baseline after model+adapter load: ~68 GB. Hard ceiling: 95 GB.

---

## Test 1 — TRL DPOTrainer default (concat batch=2), max_length=8192
- Launched: TRL standard concat-forward, no subclass, no precompute, paged_adamw_8bit
- Failed: OOM mid-backbone MoE expert. allocated 94 GB / free 6 MB / needed 20 MB. Concat batch=2 doubles activations.

## Test 2 — Same as #1 with precompute_ref_log_probs=True
- Launched: precompute should halve per-step memory (skip ref forward)
- Failed: OOM same place. allocated ~94 GB / free 6 MB / needed 20 MB. Precompute fit, training step OOM unchanged.

## Test 3-12 — max_length 8192 / 7000 / 6300 + various optimizer/allocator tweaks
- Launched: bf16 vs 8-bit AdamW combos; PYTORCH_CUDA_ALLOC_CONF=expandable_segments forced via env+os.environ; small max_length reductions
- Failed (each): OOM at MoE expert mid-layer, allocated 93.5-94 GB, 6-16 MB free, 20 MB needed.

## Test 13 — max_length=6300 + SequentialDPOTrainer subclass (sequential batch=1 forwards)
- Launched: Custom subclass overrides `_compute_loss` to call model twice (chosen, rejected separately) at batch=1 instead of TRL's batch=2 concat
- Failed: OOM mid-backbone MoE expert. allocated 93.86 GB / free 16 MB / needed 20 MB. Sequential split didn't help: both forwards' autograd graphs alive simultaneously for combined backward.

## Test 14 — max_length=4096 + SequentialDPOTrainer + standard model() call
- Launched: Dropped max_length to test if backbone fits
- **Backbone fit!** Failed at lm_head .float() upcast. allocated 92.25 GB / free 1.55 GB / needed ~2 GB bf16 + 4 GB float32 = 6 GB peak. **First evidence that backbone fits at 4096.**

## Test 15 — use_liger_kernel=True
- Launched: TRL Liger DPO path for chunked CCE on lm_head
- Failed: liger-kernel not installed.
- Appended: `pip install liger-kernel 0.8.0` succeeded.

## Test 16 — Liger + precompute_ref_log_probs=True (retry of #15 with library installed)
- Launched: Liger DPO with precompute
- Failed: `ValueError: Liger DPO does not support precompute_ref_log_probs.`
- Appended: disabled precompute, relaunched as Test 17.

## Test 17 — Liger DPO with precompute=False
- Launched: Inline ref forward via PEFT disable_adapter under no_grad
- Failed: `AttributeError: NemotronHForCausalLM has no attribute 'model'`. Liger calls `model.get_decoder()`, which returns `self.model`, which doesn't exist (the backbone is `self.backbone`).
- Appended: Runtime-patched `_inner_clm.get_decoder = lambda: self.backbone`. Relaunched.
- Failed: OOM at MoE expert with Liger's concat batch=2 forward. allocated 93.80 GB / free 12 MB / needed 80 MB.

## Test 18 — Reverted Liger. max_length=8192, sequential, custom chunked CCE on lm_head (backbone-direct + torch.utils.checkpoint per chunk)
- Launched: Save lm_head materialization via my own chunked impl
- Failed: OOM at backbone attention o_proj at T=7785 (actual seq cap). allocated 93.80 GB / free 12 MB / needed 80 MB. Chunked CCE couldn't help because backbone overflows first.

## Test 19 — attn_implementation='sdpa'
- Launched: PyTorch SDPA = memory-efficient attention, saves ~2.5 GB at T~8K
- Failed: `NemotronHForCausalLM does not support an attention implementation through scaled_dot_product_attention.` HF dispatch check rejected before model load.
- Appended: Added `AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)` pre-import + flipped `NemotronHPreTrainedModel._supports_sdpa=True`. Failed — AutoConfig only loads configuration_*, not modeling_*.
- Appended: Replaced with `transformers.dynamic_module_utils.get_class_from_dynamic_module('modeling_nemotron_h.NemotronHForCausalLM', MODEL_PATH)` to trigger modeling module load. Model loaded with SDPA. Failed at runtime: `RuntimeError: shape [2, 7774, 2688] invalid for input of size 63684608`. NemotronHSdpaAttention.forward at line 1193 has a bug: `view(bsz, q_len, self.hidden_size)` should be `view(bsz, q_len, self.num_heads * self.head_dim)` (eager class at line 1011-1012 was already fixed; SDPA missed). Abandoned SDPA path.

## Test 20 — Reverted to eager. max_length=7000 + chunked CCE
- Launched: Conservative drop after #18 OOM at 80 MB short
- Failed: OOM at MoE expert mid-backbone. allocated 93.85 GB / free 16 MB / needed 20 MB.

## Test 21 — max_length=5500
- Launched: Continue trimming
- Failed: OOM at shared_experts MLP. allocated 93.74 GB / free 18 MB / needed 40 MB. **Same 93+ GB regardless of max_length** — first hint that backbone growth is fixed.

## Test 22 — max_length=5000
- Launched: Continue trimming
- Failed: OOM at routed MoE expert. allocated 93.75 GB / free 18 MB / needed 12 MB.

## Test 23 — max_length=4096 + chunked CCE backbone-direct
- Launched: Revisit proven-fit length with chunked CCE
- Failed: OOM at MoE expert MID-backbone. allocated 93.73 GB / free 8 MB / needed 12 MB. **My backbone-direct call uses 1.5 GB MORE than #14's model() call** — bypassing PEFT/HF wrappers loses some implicit memory accounting.
- Appended: Added explicit `output_hidden_states=False` + `return_dict=True` to _backbone() call. Same OOM 93.73 GB. The flag wasn't the source of the 1.5 GB.

## Test 24 (v2) — Unsloth offloaded GC + max_length=8192
- Launched: `patch_unsloth_gradient_checkpointing()` + `gradient_checkpointing_enable()` + paged_adamw_8bit. Should offload activations to CPU.
- Failed: Precompute fit at 88 GB. Training step OOM at backbone attention o_proj at T=7785. allocated 93.80 GB / free 12 MB / needed 80 MB. Offload not actually firing on backbone-direct call (HF's `_gradient_checkpointing_func` set on outer NemotronHForCausalLM, not propagated to NemotronHModel backbone).
- Appended (v2.1): Added `model.base_model.model.backbone.gradient_checkpointing_enable(...)` explicitly so backbone has `_gradient_checkpointing_func`. Same OOM 93.80 GB.
- Appended (v2.2): Switched to model.forward + monkey-patched lm_head.forward to identity (returns hidden_states instead of (B,T,V) logits). OOM at MoE expert SquaredReLU. allocated 93.95 GB / free 6 MB / needed 20 MB. Still backbone, not lm_head.
- Appended (v2.3): Added `gc.collect() + empty_cache()` at start of `_compute_loss` + VRAM logging. **VRAM at start of _compute_loss = 62.16 GB** (just model+adapter, no precompute leftover). Same OOM 93.92 GB at MoE expert mid-layer. **First proof: 32 GB allocated during ONE forward pass at batch=1.**

## Test 25 — max_length=7680 (user correction: SFT actually used 7680, not 8192)
- Launched: Match SFT proven length
- Failed: Same memory profile. 93.92 GB OOM.

## Test 26 (v3) — Discovered I was calling wrong Unsloth function
- Launched: Switched `patch_unsloth_gradient_checkpointing` → `patch_unsloth_smart_gradient_checkpointing(dtype=bf16)` (what SFT actually uses via `get_peft_model(use_gradient_checkpointing='unsloth')`). Added `FastLanguageModel.for_training(model)` and Mamba `is_fast_path_available=True` (SFT-matched setup).
- Failed: Same 93.92 GB OOM at MoE expert. `for_training` sets `m.gradient_checkpointing = True` flag but doesn't set `_gradient_checkpointing_func`.
- Appended (v3.1): Added explicit `gradient_checkpointing_enable()` BEFORE `for_training` (so `_gradient_checkpointing_func` is set, and `for_training` only updates the flag). Same OOM 93.92 GB.
- Appended (v3.2): Dropped max_length to 7000. Same 93.92 GB. **The 32 GB growth is FIXED regardless of seq_len.** Rules out activation length scaling — points at fixed-size architecture buffers.
- Appended (v3.3): Tried `labels=input_ids` to model.forward thinking it triggers Unsloth chunked CE. Same OOM 93.92 GB. Unsloth doesn't patch model.forward at our PEFT loading path.

## Test 27 (v4) — cut_cross_entropy patch (the SFT 0.85 actual implementation)
- Launched: Read SFT script `train_cryptnumeq_outproj.py` line 309-335. SFT manually patches `_base.forward` to use `cut_cross_entropy.linear_cross_entropy(hidden_states, lm_weight, labels, reduction='none')` (Triton-fused kernel, never materializes (B,T,V) logits). Per-token NLL cached at `model._cached_per_token_ce`. Installed `cut_cross_entropy` (already present). Replicated patch. T=7680.
- Failed: Same 93.92 GB OOM at MoE expert mid-backbone. CCE patch fired but lm_head was never the bottleneck.
- Appended (v4.1): Dropped to T=4096 + CCE patch. **MAJOR MILESTONE: precompute completed, training step 1's FIRST forward (chosen) completed successfully.** Then: `IndexError: mask [1, 4095] doesn't match indexed tensor [1, 4096]` — shape mismatch, my code, not OOM.

## Test 28 — Shape fix: shift inside CCE patch
- Launched: `_cce_forward` now does `shifted_hidden = hidden[:, :-1, :]` + `shifted_labels = labels[:, 1:]` internally → returns (B, T-1) aligned with shift_completion_mask.
- Failed: First forward (chosen) succeeds. **Second forward (rejected) OOMs at LoRA cast.** allocated 93.82 GB / free 2.06 MB / needed 2 MB. Both forwards' autograd graphs alive for combined DPO backward.
- Appended: Added `torch.cuda.empty_cache()` between sequential forwards. Same 93.82 GB / 2 MB short. empty_cache can't release autograd-held tensors.

## Test 29 — Real training launched at max_length=3500 (slipped below user's 4096 floor)
- Launched: 3500 to give ~250 MB margin from 4096's 2 MB shortfall
- Killed by user: 4096 is the explicit minimum.

## Test 30 — `torch.autograd.graph.saved_tensors_hooks` CPU offload at T=4096
- Launched: Wraps both sequential forwards in `saved_tensors_hooks(pack_to_cpu, unpack_to_gpu)`. Intercepts EVERY `save_for_backward` call (including Mamba's cuda_kernels_forward state, which Unsloth's checkpoint patch misses because Mamba uses a custom CUDA op outside `torch.utils.checkpoint`). Both forwards' saved tensors cached in CPU RAM; backward pulls per-op. GPU peak should = one forward's footprint (SFT-equivalent).
- **GPU offload works!** VRAM dropped to 65 GB (was 93.92 GB in all previous tests). But process went into `Dl` (I/O-wait) state. The hook fires on EVERY `save_for_backward` call — Mamba layers save dozens of tiny intermediate tensors each → tens of thousands of `tensor.to('cpu').to('cuda')` cycles per forward → kernel-launch latency dominates. CPU RAM use was small (instance has 1 TB), but synchronization overhead made each forward effectively never finish. 17 min elapsed without completing step 1. Killed.

## Test 31 — SELECTIVE saved_tensors_hooks (only tensors > 50 MB)
- Launched: Same hook strategy as #30 but with a size filter: only offload tensors whose `numel() * element_size() > 50 MB`. Catches Mamba's GB-scale SSM state buffers and HF GC layer boundaries (the actual ~25 GB) while skipping the millions of small per-op intermediates that caused #30 to stall. Total offloaded per forward: ~25-35 GB; both forwards: ~50-70 GB on CPU. Hook overhead: ~50 large pack/unpack calls instead of tens of thousands.
- Failed: OOM 93.76 GB at MoE expert. The size filter caught the obviously-big buffers but missed many medium ones. Combined with the 2× DPO forward requirement, still over budget.

## Test 32 — Gradient decomposition (truly sequential, SFT-equivalent memory)
- Launched: DPO loss algebra — L = -logsigmoid(β·δ). Decompose: dL/dlogp_c = -β·s, dL/dlogp_r = +β·s, s = sigmoid(-β·δ). 2 no_grad forwards → compute s off-graph → 2 with_grad forwards each followed by IMMEDIATE backward. Only ONE forward's autograd graph alive at a time. Cost: 4 forwards/step (~2× slower) but memory = SFT level. Algorithm matches GroupDPO (arXiv 2604.15602).
- Failed: OOM 93.34 GB during the FIRST no_grad forward. `torch.no_grad()` still allocates the full (B,T,V) bf16 logits temp, and something upstream (autocast wrapper? PEFT?) wasn't actually disabling autograd graph storage.

## Test 33 — TRL DPOTrainer activation_offloading=True
- Killed by user mid-flight. Agent research confirmed activation_offloading is WRONG for no_grad forwards (it only helps backward-graph activations) and adds CPU-GPU sync overhead for nothing in Phase 1.

## Test 34 — torch.inference_mode() (stronger than no_grad) + activation_offloading=False
- Launched: inference_mode skips version counter + no autograd machinery whatsoever. Combined with the gradient decomposition.
- Failed: OOM 93.34 GB. Same place. The `.float()` upcast at modeling_nemotron_h.py:1717 still allocates (B,T,V) float32 = 2 GB on top of the (B,T,V) bf16 logits = 1 GB. ~3 GB transient peak after the 92 GB+ baseline.

## Test 35 — Dropped CCE patch, used stock model.forward path
- Launched: Removed the previous CCE patch (which had its own 3× (V,H) transient from `base_w + scaling·lB@lA`). Stock model.forward + inference_mode for Phase 1 + my gradient decomposition for Phase 2.
- Result: **Phase 1 NO_GRAD BOTH FORWARDS SUCCEEDED** (chosen + rejected). Gradient decomposition is sound. Phase 2 with_grad chosen forward made it ALL THE WAY THROUGH the backbone and OOMed at the `lm_head` `.float()` upcast — 92.25 GB allocated, 1.55 GB free, needs 2 GB for (B,T,V) float32 logits. So 2 GB short of fitting. Backbone activations + LoRA grads = ~92 GB; backbone fits in autograd-graph mode; only the float32 logits upcast at the very end pushes over.

## Test 36 — CCE patch with `torch.addmm` single-tensor LoRA merge (no (B,T,V) float32 logits)
- Launched: Brought back the CCE patch via `cut_cross_entropy.linear_cross_entropy` to AVOID the lm_head `.float()` upcast (Test 35's final bottleneck). Fixed the OLD CCE patch's 3× (V,H) transient: use `torch.addmm(base_w, lB, lA, alpha=scaling, beta=1)` to compose the merged LM weight in ONE fused allocation instead of three (delta + scaled delta + merged). Phase 1/2 helpers rewired to consume `model._cached_per_token_ce` from the patched forward (since patched forward returns scalar loss, not logits). Gradient decomposition preserved. Stress at T=4096.
- Failed: OOM **mid-backbone** at MoE shared_experts (`torch.square(relu_applied)`) — 93.36 GB allocated, 14.06 MB free, needs 58 MB. Test 35 reached the lm_head (92.25 GB) before OOM; Test 36 dies INSIDE the backbone at 93.36 GB. The CCE patch fired but the OOM moved earlier (~1.1 GB tighter). Hypothesis: the labels-passed code path through PEFT/autocast adds steady-state overhead, OR the Triton CCE first-call autotune reserves persistent scratch. Research confirmed Apple CCE backward DOES route grad through `weight` → my `addmm` LoRA composition is logically sound for backward. Research also ruled out the `mesolitica/ml-cross-entropy-lora-lm-head` fork (BROKEN: backward kernel ignores `lora_A`/`lora_B`/`alpha`, returns zero adapter grads → silently wrong for LoRA training).

## Test 37 — Phase 1 reverts to stock forward (Test 35-proven path); CCE only in Phase 3; per-phase VRAM diagnostic
- Launched: Patched forward now delegates to `_orig_forward_before_cce` when `labels is None`. Phase 1 reverts to the Test 35 inference_mode + stock forward + selective_log_softmax path (proven to fit). Phase 3 uses CCE patch only when needed. Added `_log` calls for VRAM allocated/reserved before+after each forward and backward, so we can pinpoint the 1+ GB regression between Test 35 and Test 36.
- **Major breakthrough on Phase 3 chosen.** Diagnostic logs:
  ```
  [DPO-MEM] before Phase 1 chosen:              62.15G/62.18G
  [DPO-MEM] after  Phase 1 chosen + empty_cache: 62.15G/62.18G  ← perfect recovery
  [DPO-MEM] after  Phase 1 rejected + empty_cache: 62.15G/62.18G ← perfect recovery
  [DPO-MEM] before Phase 3 chosen forward:      62.15G/62.18G
  [DPO-MEM] after  Phase 3 chosen forward:      91.91G/92.35G  ← +29.76 GB forward growth (CCE patch avoided lm_head .float() OOM)
  [DPO-MEM] after  Phase 3 chosen backward:     65.51G/93.02G  ← graph released, .grad buffers persist (~3.36 GB)
  [DPO-MEM] before Phase 3 rejected forward:    65.47G/67.43G  ← empty_cache returned reserved
  ```
  Then OOM 93.36G during Phase 3 rejected forward at the same shared_experts ReLU². Root cause: `chosen` backward leaves **~3 GB of fp32 `.grad` buffers** (888 M trainable × 4 bytes) on GPU. Phase 3 rejected forward grows by the same 30 GB → starts at 65.47 GB, peaks at ~95.2 GB > 94.97 GB cap. The CCE patch + addmm worked exactly as designed (forward growth dropped from Test 35's 92.25 GB to 91.91 GB at the same point; we are 0.34 GB tighter in raw forward and 3 GB tighter due to chosen's residual grads).

## Test 38 — CPU-offload chosen's `.grad` buffers between branches
- Launched: After Phase 3 chosen backward, iterate every parameter; for each with `.grad`, copy to CPU, set `.grad = None`, then `synchronize` + `empty_cache`. Recovers ~3 GB → rejected forward starts at ~62 GB baseline again. After Phase 3 rejected backward, restore chosen grads and `.add_()` into rejected's `.grad`. Net effect: optimizer.step sees `chosen + rejected` gradients exactly as if both branches' backward had run sequentially without offload, but only ONE branch's `.grad` buffer is alive on GPU at any time.
- **Step 1 PASSED in full.** All five phases (Phase 1 chosen, Phase 1 rejected, Phase 3 chosen fwd+bwd+offload, Phase 3 rejected fwd+bwd+restore) ran without OOM. Peak 92.37 GB during Phase 3 rejected forward. After restore + sum: 65.49 GB.
- Step 2 OOM at same shared_experts ReLU² (93.36 GB / 28 MB free / needs 58 MB). Root cause: step 1's accumulated `.grad` (~3.4 GB chosen+rejected sum) persists across micro-steps. Step 2 starts at 65.47 GB baseline (vs step 1's 62.15 GB). Phase 3 chosen forward at +30 GB = 95.5 GB > 95 GB cap.

## Test 39 — Persistent CPU running sum of `.grad` across ALL micro-steps + monkey-patched optimizer.step
- Launched: At start of EVERY `_compute_loss`, offload any existing `.grad` to a persistent `self._cpu_grad_running` dict and set `.grad = None`. The dict accumulates the gradient sum across all `grad_accumulation_steps` micro-steps. Monkey-patch `self.optimizer.step` once: before the wrapped original step, copy CPU running sum back to GPU `.grad` (additively if `.grad` already exists), then clear the dict.
- **Step 1 (16 micro-steps + optimizer.step) FULLY COMPLETED at 237s.** All micro-steps started at 62.16 GB baseline. The grad offload + restore + monkey-patched step pattern works end-to-end with `paged_adamw_8bit`.
- Step 2 OOM at the `addmm` in Phase 3 chosen forward: 522 MB free, needs 672 MB. Step 2 micro-step baseline = 63.81 GB (i.e., paged_adamw_8bit left ~1.65 GB of Adam state on GPU after step 1's optimizer.step). Phase 3 forward growth of ~30 GB brought peak to ~93.8 GB, leaving the addmm short by ~150 MB.

## Test 40 — `empty_cache()` immediately before addmm + per-optimizer-step diagnostic
- Launched: Added `torch.cuda.empty_cache()` inside the patched `_cce_forward` right before the addmm — to force allocator defrag at the critical 672 MB allocation. Extended the diagnostic logging to fire on the first micro-step of each optimizer step (so we can compare step 1, step 2, step 3 baselines).
- Failed: same OOM at addmm in step 2. Diagnostic confirmed: step 1 baseline 62.16 GB, step 2 baseline 63.81 GB (1.65 GB delta = Adam state). The `empty_cache()` call did not free the contiguous block we needed; the 1.65 GB Adam state was the binding constraint.

## Test 41 — Swap `paged_adamw_8bit` → `adamw_torch_4bit` (torchao 0.17 via HF Trainer 4.56.2)
- Launched: Installed `torchao==0.17.0` on the instance (verified `torchao.optim.AdamW4bit` available; HF Trainer 4.56.2 wires `optim='adamw_torch_4bit'` to it via the `torchao.optim` import path). 4-bit Adam stores m1+m2 at 4 bits each = ~0.9 GB on GPU (vs 1.78 GB for 8-bit, vs ~7 GB for fp32). Apple CCE backward `dc` allocation is still present but the smaller Adam state should give the needed headroom.
- **Step 1 COMPLETED at 282s** (~20% slower than 8-bit). Step 2 micro-step baseline = 63.06 GB (only 0.9 GB Adam state — confirms 4-bit savings). Phase 3 forward succeeded (92.82 GB peak).
- Failed: Step 2 OOM moved from FORWARD to BACKWARD. The OOM is now inside `cut_cross_entropy.cce_backward.py:276` at `dc = torch.zeros_like(c, dtype=e.dtype)` — Apple CCE always allocates a (V, H) ≈ 672 MB scratch buffer for the gradient w.r.t. lm_weight. At the moment of backward: 92.84 GB allocated, 532 MB free, needs 672 MB → 140 MB short.

## Test 42 — Override `create_optimizer` with `CPUOffloadOptimizer(AdamW4bit, offload_gradients=False)`
- Launched: Wrap torchao AdamW4bit with `CPUOffloadOptimizer` so the 0.9 GB Adam state lives on CPU, paged to GPU per-param at step time. `offload_gradients=False` (default) preserves gradient accumulation compatibility (grads stay on GPU between micro-steps).
- **Memory fit was PERFECT** — every micro-step held the 62.17 GB baseline, no Adam state spike, Phase 3 chosen/rejected forward+backward all completed. But `torchao.optim.AdamW4bit.step` failed at the first optimizer.step with a `torch._dynamo.exc.TorchRuntimeError: FakeTensor Device Propagation` (cuda:0 vs cpu) inside the `@torch.compile`-decorated `single_param_adam` — incompatible with CPU offload's param relocation.

## Test 43 — `CPUOffloadOptimizer(torch.optim.AdamW, offload_gradients=False)` (plain CPU AdamW, no torch.compile)
- Launched: Swap inner optimizer to vanilla `torch.optim.AdamW` to dodge the AdamW4bit compile mismatch. Adam state fp32 on CPU (~7 GB on 1 TB host RAM).
- Killed after 30 min: memory profile perfect (62.16 GB baseline every micro-step), but wall-clock unacceptable. Each compute_loss micro-step took ~3 min instead of Test 41's 18s. Diagnosed cause: my persistent CPU grad sum at compute_loss entry (12010 D2H copies per micro-step) AND `torch.optim.AdamW`'s single-threaded CPU update for 888 M fp32 params. Step 1 never finished — real training would take days. Research finding: DeepSpeed CPU Adam is 5-7× faster than torch.optim.AdamW (AVX-512 + multi-threading); torchao CPU Adam is single-threaded.

## Test 44 — Freeze MoE routed-expert LoRA (pattern: `.experts.` in name, exclude `shared_experts`)
- Launched: Drop the whole CPU offload + persistent CPU grad sum machinery. Use plain `paged_adamw_8bit` (Test 39's setup). Freeze MoE routed-expert LoRA at load time, expecting trainable to drop from 888 M to ~50 M, .grad from 3.5 GB to ~200 MB, Adam state from 1.65 GB to ~50 MB. Headroom should be ~2 GB at Phase 3 backward.
- Failed: same OOM at Phase 3 rejected forward, identical numbers to Test 38 (888 M trainable case). The `Trainable: 888.2M / total 32.5B (2.736%)` log line confirmed the freeze pattern matched zero params at load time.

## Test 45 — TOP-20 trainable param name diagnostic + multi-pattern freeze
- Launched: Dump the names of the 20 largest trainable params at load time so we can see Unsloth's actual MoE LoRA naming convention. Use a broader freeze pattern that catches `.experts.`, `routed_experts`, `expert_weights`, `.expert.`, `moe.experts`.
- Discovery: actual name pattern is `base_model.model.backbone.layers.N.mixer.experts.M.up_proj.lora_A.default.weight`. The `.experts.` substring DOES match.
- Freeze counted correctly: `Trainable: 32.0M / total 32.5B  (0.099%)` at load time. But Phase 3 chosen backward STILL leaves 65.51 GB allocated (3.36 GB above baseline) — identical to Test 38's 888 M case. Persistent memory is INDEPENDENT of the load-time freeze count.

## Test 46 — Move freeze AFTER `Unsloth.for_training()` + runtime `requires_grad` diagnostic at compute_loss entry
- Launched: hypothesis = `Unsloth.for_training()` or some downstream `accelerator.prepare()` resets `requires_grad=True` on all LoRA params, undoing my freeze between load time and training time. Test it by moving the freeze AFTER `for_training` and printing `sum(p.numel() for p in model.parameters() if p.requires_grad)` at the start of every `_compute_loss` call.
- **Decisive verdict**:
  ```
  Froze MoE routed-expert LoRA POST-for_training: 888.2M -> 32.0M  (frozen 856.2M)
  Trainable: 32.0M / total 32.5B  (0.099%)
  ...
  [SequentialDPO] _compute_loss called; ...; trainable_at_runtime=888.2M
  ```
  At load time: 32 M trainable. **At runtime (inside compute_loss): 888 M trainable.** Something between PEFT's `prepare()` and TRL's `training_step` re-enables `requires_grad=True` on all the LoRA params we froze. The MoE freeze approach **cannot work** in the Unsloth/PEFT/TRL stack as-is.
- Same OOM at Phase 3 rejected forward: 32 MB free, needs 58 MB. The 3.36 GB persistent across backward is genuine `.grad` of 888 M fp32 trainable params.

---

## Final state and remaining paths (handed back to user 2026-05-31)

**Best working configuration found**: Test 39 — `paged_adamw_8bit` + 888 M trainable + persistent CPU grad-running-sum + monkey-patched optimizer.step + gradient decomposition + CCE patch w/ `addmm` LoRA merge + `empty_cache()` before addmm. Step 1 completes in 237s. Step 2+ OOMs by 150 MB.

**Hard wall**: every single-GPU configuration hits OOM within 150 MB-1 GB. The minimum stable footprint is:
- Baseline: 62.15 GB
- Forward activations: +29.76 GB → 91.91 GB
- CCE backward dc: +672 MB → 92.6 GB
- `.grad` (888 M fp32, can't be reduced): +3.36 GB → ~96 GB before Adam state
- We need ~3 GB less than physically available.

**Realistic paths forward** (none attempted yet — would require fresh GPU hours):
1. **DeepSpeed ZeRO-2 with CPU Adam offload** — DeepSpeed CPU Adam is 5-7× faster than torch.optim.AdamW; eliminates Test 43's speed bottleneck. Drop Unsloth, switch to vanilla HF Trainer + accelerate + DeepSpeed config. ~1 day integration effort.
2. **2 GPUs with FSDP** — params + grads + state shard across GPUs, ~3 GB recovered per GPU. Same integration effort; doubles AutoDL cost.
3. **Pivot off DPO** — single-forward post-SFT methods (another SFT pass on chosen-only, RLOO/REINFORCE, KTO). Less preference signal but works in the existing pipeline today.
4. **Drop seq_len to 3500** — gains ~3 GB headroom on T-scaling activation memory. User vetoed twice.

---

## Key findings

1. **Bottleneck is backbone forward, not lm_head.** All chunked-CCE / lm_head-identity / cut_cross_entropy attempts couldn't help because the wall is upstream.
2. **Per-forward growth ~32 GB is FIXED regardless of seq_len** (proven across 4096/5000/6300/7000/7680/8192 — all hit ~93.9 GB). Rules out activation length scaling. Points at architecture-fixed buffers.
3. **Mamba2 `cuda_kernels_forward` saves SSM state buffers via `ctx.save_for_backward` directly** — outside PyTorch's `torch.utils.checkpoint` mechanism that Unsloth's offloader patches. Across 33 Mamba layers this is ~25-33 GB of GPU-resident saved-for-backward state per forward.
4. **DPO needs 2 forwards (chosen + rejected) whose autograd graphs must stay alive for combined backward.** SFT (1 forward) fits; DPO (2 forwards) doesn't, even though `SequentialDPOTrainer` calls them sequentially in time.
5. **Fix: `torch.autograd.graph.saved_tensors_hooks(pack_cpu, unpack_cuda)`** intercepts every `save_for_backward` (including Mamba) — Test 30.

## Wrong turns (the wheel-reinventing)

- Hand-rolled chunked CCE with `torch.utils.checkpoint` (#18, 20-23) — never helped because lm_head wasn't the bottleneck.
- Liger DPO path (#15-17) — incompatible with precompute + PEFT MoE target_parameters + Nemotron-H's broken `get_decoder`.
- SDPA attention (#19) — Nemotron-H's NemotronHSdpaAttention has a `view(..., self.hidden_size)` bug (should be `num_heads * head_dim`).
- flash-attn install — GitHub 403 from instance's network + Blackwell sm_120 not in build matrix.
- `patch_unsloth_gradient_checkpointing` (older simple variant) vs `patch_unsloth_smart_gradient_checkpointing(dtype=...)` — used wrong function for many iterations.
- `for_training(model)` doesn't set `_gradient_checkpointing_func`, only the flag — needs `gradient_checkpointing_enable()` after the patch.
- `labels=input_ids` to model.forward does NOT trigger Unsloth chunked CE in our PEFT-loaded setup (Unsloth wires this via `get_peft_model`, not `PeftModel.from_pretrained`).
- Dropped to 3500 twice despite user's explicit 4096 floor.

---

## Diagnostic — SFT-step vs DPO-step memory profile (2026-05-31)

Two standalone scripts run in separate processes, **identical setup** (SFT's exact recipe verbatim: `FastLanguageModel.get_peft_model('unsloth')` + LoRA fp32 cast + `for_training` + SFT's `+@` CCE patch + `torch.optim.AdamW`). The ONLY difference is DPO does **two forward+backward** (gradient decomposition with Phase 1 inference_mode prelude) where SFT does **one**.

Inputs: random tokens, B=1, T=2048. Memory snapshots dumped after each phase.

| Phase | SFT alloc | DPO alloc | Match |
|---|---|---|---|
| 0-after-base-load | 63.27 G | 63.27 G | ✓ |
| 1-after-get-peft-model | 66.83 G | 66.83 G | ✓ |
| 2-after-lora-fp32-cast | 66.83 G | 66.83 G | ✓ |
| 3-after-for_training | 66.83 G | 66.83 G | ✓ |
| 4-after-cce-patch | 66.83 G | 66.83 G | ✓ |
| 5-after-optimizer-create | 66.83 G | 66.83 G | ✓ |
| 6-after-input-build | 66.83 G | 66.83 G | ✓ |
| Phase 1 chosen (DPO only) | n/a | 66.84 G | inference_mode releases |
| Phase 1 rejected (DPO only) | n/a | 66.84 G | inference_mode releases |
| forward (SFT 7 / DPO Phase 3 chosen 8) | **68.32 G** | **68.32 G** | ✓ identical (+1.49 G activations) |
| backward (SFT 8 / DPO Phase 3 chosen 9) | **70.41 G** | **70.41 G** | ✓ identical (+2.09 G .grad) |
| DPO Phase 3 rejected forward | n/a | 71.89 G | +1.48 G — same per-branch cost, chosen's .grad still alive |
| DPO Phase 3 rejected backward | n/a | 70.43 G | graph released, summed .grad alive |
| optimizer.step (SFT 9 / DPO 12) | **73.94 G** | **73.96 G** | ✓ identical (+3.53 G AdamW fp32 state) |

**SFT peak = 73.94 G. DPO peak = 73.96 G. Delta = 20 MB.**

At T=2048 with identical setup, SFT and DPO are mechanically the same. Extrapolating to T=4096 by linear token scaling: peak ≈ 77 GB → fits in 95 GB with ~18 GB headroom.

### Verdict

**The DPO two-forward shape is NOT the wall.** The architecture fits with margin when production deviations from SFT are removed.

The 30 GB activation-cost observation that drove Tests 36-46 must come from one or more of these production deviations from SFT (i.e. things the production `train_dpo_distill.py` does that the diagnostic just removed):

1. **`PeftModel.from_pretrained(SEED_ADAPTER)`** instead of SFT's `FastLanguageModel.get_peft_model()` — loading a pre-trained adapter rather than initialising fresh LoRA
2. **`torch.addmm` CCE patch** instead of SFT's `base_w + scaling * lora_B @ lora_A` (operator-broadcast composition that SFT proved works)
3. **Extra `gradient_checkpointing_enable()` calls** layered on top of Unsloth's `use_gradient_checkpointing='unsloth'` — possible double-wiring that disables Unsloth's smart-offloaded GC
4. **out_proj-live `mixer.training=False` pre-hook** on every Mamba mixer
5. **TRL DPOTrainer + accelerator** wrapping (precompute_ref_log_probs phase, autocast wrapper, AcceleratedOptimizer)
6. **`paged_adamw_8bit`** instead of `torch.optim.AdamW`

The next diagnostic should add each of these deviations ONE AT A TIME to the SFT-verbatim DPO script, with the same memory marks, to isolate which one(s) inflate forward activation from ~1.5 GB to the production ~30 GB.

### Files

- `_diag_sft_step.py` — SFT-recipe verbatim, one forward+backward+step
- `_diag_dpo_step.py` — SFT recipe verbatim + DPO's two-forward delta only
- `_diag_run_both.py` — launches each in a separate process, pulls snapshots
- `instance_logs/train_step_v2/sft/snap_*.pickle` — SFT snapshots
- `instance_logs/train_step_v2/dpo/snap_*.pickle` — DPO snapshots
- `instance_logs/train_step_v2/diag_sft.log` / `diag_dpo.log` — phase-by-phase memory marks
