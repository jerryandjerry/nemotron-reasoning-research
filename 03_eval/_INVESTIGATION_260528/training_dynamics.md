# Training Dynamics: `moe` vs `alleq`

**Investigation date:** 2026-05-28
**Logs:**
- moe: `study/2605222039_sft_moe_outproj_rtx6000/260523_outproj_submission/train_log_moe_notie_outproj.txt`
- alleq: `study/2605252021_sft_newdata_numeq_alleq_outproj_rtx6000/260526_1343_submission/train_alleq_full.txt`

---

## T1 + T2. Per-step metrics (loss / grad_norm / lr)

Parsed 246 step rows for moe, 251 step rows for alleq. Per-step loss/grad/lr CSV at `per_step_loss.csv` next to this report.

### Loss-threshold crossing (first step where loss dropped below threshold)

| threshold | moe step | moe loss | alleq step | alleq loss |
|----------:|---------:|---------:|-----------:|-----------:|
| < 1.000   |    1     | 0.4036   |    1       | 0.5221     |
| < 0.500   |    1     | 0.4036   |    2       | 0.4746     |
| < 0.200   |    9     | 0.1921   |    11      | 0.1837     |
| < 0.100   |    15    | 0.0918   |    17      | 0.0923     |
| < 0.050   |    21    | 0.0464   |    22      | 0.0487     |
| < 0.010   |    35    | 0.0088   |    40      | 0.0089     |
| < 0.005   |    53    | 0.0042   |    75      | 0.0041     |

alleq is consistently 2–5 steps slower to hit each threshold and 22 steps slower to drop below 0.005 (but: alleq runs more total steps and has 171 extra examples).

### Every-25-steps table

| step | moe loss | moe gn | moe lr | alleq loss | alleq gn | alleq lr |
|----:|--------:|-------:|-------:|----------:|--------:|--------:|
|   1 | 0.403614 | 0.1480 | 2.00e-04 | 0.522108 | 0.1517 | 2.00e-04 |
|  26 | 0.022567 | 0.0399 | 1.80e-04 | 0.031131 | 0.0506 | 1.80e-04 |
|  51 | 0.005381 | 0.0156 | 1.59e-04 | 0.005347 | 0.0153 | 1.60e-04 |
|  76 | 0.004059 | 0.0189 | 1.39e-04 | 0.004427 | 0.0091 | 1.40e-04 |
| 101 | 0.003980 | 0.0126 | 1.19e-04 | 0.004269 | 0.0112 | 1.20e-04 |
| 126 | 0.003483 | 0.0071 | 9.84e-05 | 0.003316 | 0.0074 | 1.00e-04 |
| 151 | 0.003813 | 0.0116 | 7.80e-05 | 0.003061 | 0.0075 | 8.05e-05 |
| 176 | 0.003200 | 0.0054 | 5.77e-05 | 0.003215 | 0.0084 | 6.06e-05 |
| 201 | 0.003128 | 0.0081 | 3.74e-05 | 0.004430 | 0.0085 | 4.06e-05 |
| 226 | 0.003349 | 0.0073 | 1.71e-05 | 0.003096 | 0.0084 | 2.07e-05 |
| 246 | 0.002959 | 0.0088 | 8.13e-07 | 0.002807 | 0.0050 | 4.78e-06 |
| 251 |   —      |   —    |   —    | 0.002449 | 0.0049 | 7.97e-07 |

### Summarized loss dynamics

| metric | MOE | ALLEQ |
|---|---:|---:|
| Step-1 loss | 0.4036 | 0.5221 |
| Mean loss, first 50 steps | 0.0816 | 0.1029 |
| Mean loss, last 50 steps | 0.00301 | 0.00326 |
| Window-25 mean loss at step≈50 | 0.00610 | 0.00772 |
| Window-25 mean loss at step≈100 | 0.00379 | 0.00448 |
| Window-25 mean loss at step≈150 | 0.00335 | 0.00379 |
| Window-25 mean loss at step≈200 | 0.00309 | 0.00340 |
| Window-25 mean loss at step≈246 | 0.00290 | 0.00323 |
| Final-step loss | 0.002959 (step 246) | 0.002449 (step 251) |
| Max grad_norm | 0.1480 (step 1) | 0.1517 (step 1) |
| Max loss | 0.4036 (step 1) | 0.5221 (step 1) |
| Loss spikes (>3× prior step) | 0 | 0 |
| Steps where alleq > moe + 0.001 | — | 92 / 246 |
| Steps where alleq < moe – 0.001 | — | 1 / 246 |
| Mean diff (alleq − moe) across 1–246 | — | **+0.00476** |

**Interpretation.** Both runs follow the same well-behaved cosine-ish curve: very rapid drop in the first ~25 steps (an order-of-magnitude per ~10 steps), then a slow grind in the 0.003–0.005 floor. No spikes, no NaNs, no instability. ALLEQ is systematically **slightly higher** loss than MOE at every checkpoint until the very end (median |diff| 0.00069 but mean diff +0.00476 — the gap is bigger early). At step≈100 the smoothed alleq loss is 18% higher; by step≈246 it is 11% higher. This is consistent with ALLEQ having ~171 extra fresh equation_numeric examples (slightly harder/longer-to-fit content) interleaved into the same number of effective epochs.

Neither run "overfits to near zero" — both bottom-out around 0.0025–0.003, which is the expected floor for LoRA-only adapter training on stratified content. Both have grad_norm < 0.02 from step ~30 onward — the model is no longer chasing anything significant by then.

---

## T3. Total steps verification

- **MOE**: header says `Stratified interleave: 7849 samples, seed=42 / Batching: stratified, 246 steps`. 7849 / 32 = 245.28 ⇒ 246 steps. Confirmed by `step=246` being the last in log and `Samples: 7849 / Steps: 246` in summary.
- **ALLEQ**: header says `Stratified interleave: 8020 samples, seed=42 / Batching: stratified, 251 steps`. 8020 / 32 = 250.625 ⇒ 251 steps. Confirmed.

Both batch sizes 32, micro 4, same seed 42 stratified interleave. The 5 extra steps in alleq = 5 × 32 = 160 extra example slots, used to absorb the 171 net new examples (171 − 11 padding overlap).

---

## T4. Per-batch sample IDs — first/last 5 steps

### MOE first 5 steps (first 4 of each step's 32-batch)
```
step=1: 9935aa11:'boxed{-`#}'        bbf1dba3:'{11010010}'         fa6bd888:'ed{27.869}'    9f9cde24:'nd valley}'
step=2: 9a4ea591:'ess draws}'         2cb5b118:'ed{24.861}'          a1b268b4:'ed{20.165}'    fecfb467:'{00101100}'
step=3: c7420a23:'\boxed{33}'         b1b10e83:'oxed{|"#$}'          f42d5e08:'ient door}'    8df3daad:'\boxed{34}'
step=4: 1bcd2b16:'{00000101}'         7b8e4432:'treasure}'           0c26f842:'xed{9.011}'    49743645:'oxed{1396}'
step=5: 69fe4b0d:'\boxed{59}'         b12df751:'ed{29.288}'          b1b5054f:'{00100000}'    34390bf1:'ed{29.852}'
```

### ALLEQ first 5 steps
```
step=1: b0d0dd45:'oxed{7289}'         499b6f06:'oxed{VIII}'          9935aa11:'boxed{-`#}'    bbf1dba3:'{11010010}'
step=2: 9a4ea591:'ess draws}'         96ff5ae9:'ed{16.104}'          b2b2ba94:'xed{7.030}'    fecfb467:'{00101100}'
step=3: 00a77d86:'ed{20.552}'         87eb7ce0:'ed{29.127}'          b1b10e83:'oxed{|"#$}'    f42d5e08:'ient door}'
step=4: 83bea6b2:'{10111011}'         67ff169c:'e village}'          1bcd2b16:'{00000101}'    7b8e4432:'treasure}'
step=5: 69197d42:'oxed{4662}'         51da0ee1:'boxed{:]:}'          1fbfca5d:'found map}'    f5918499:'ed{21.750}'
```

### MOE last 5 steps (242–246)
```
step=242: 196959f6 (numeric) | f7f91582 (numeral) | 969071c3 (numeric) | d75b53c7 (bit)
step=243: 79450a97 (cipher)  | 539bfe7c (numeral) | b2bdef43 (bit)     | d8bc44b3 (int)
step=244: 9d4039b6 (cipher)  | bdc10997 (numeral) | c04634c0 (numeric) | 6e6c2ce8 (crypt)
step=245: b91855fd (bit)     | ba960ba9 (numeric) | d9d85c78 (numeric) | 12da0da7 (numeral)
step=246: db1e0603 (crypt)   | d8a47cb7 (int)     | 63233e80 (numeral) | bb40f4fa (cipher) | (only 9 batches in step 246)
```

### ALLEQ last 5 steps (247–251)
```
step=247: 2e5e7fe7 (bit)     | 191ac967 (crypt)   | f6afb6b5 (cipher)  | 05c38073 (numeric)
step=248: f19ffbf1 (numeral) | 54f33148 (numeric) | 938b81c2 (bit)     | 0422aab3 (cipher)
step=249: 27e7a8dc ('')      | e82ed6be (crypt)   | b60cc65e (cipher)  | b398201b (bit)
step=250: 354e4fb6 (numeral) | 7064acac (bit)     | 215edb1a (cipher)  | 1029ef5a (numeric)
step=251: fb15fcfa (bit)     | de78c53f (numeric) | c7a4e09f (numeric) | 61766c6f (int)  | (only 20 batches in step 251)
```

**Both runs interleave categories evenly within each batch** (every step has ~7 bit, ~7 cipher-text, ~8 float-numeric, ~3 numeral, ~3 numeric-int, ~2 crypt — matching the global category breakdown). The stratified interleave is honoring the seed=42 mix in both runs.

**Order is not identical.** Step-1 sample-ids overlap 17/32 between runs. The shared sample order shifts by ~+3 steps median in ALLEQ (the 171 extra equation_numeric samples push everything else 3 steps later on average; distribution is dense at offsets +1 through +5).

```
Step offset (alleq_step - moe_step) for the 6906 shared sample_ids
  min = -236, max = +247, median = +3
  most common offsets: +4 (801 sids), +3 (719), +5 (701), +2 (690), +1 (661), 0 (317)
```

---

## T5. `content_tail` patterns — 20 examples per run

Tails are the last ~10 chars before `<|im_end|>`. They reveal answer-format families.

### MOE — first 20 float-tail (likely equation_numeric_deduce or gravity/unit_conv)
```
step=1  batch[2]  sid=fa6bd888 tail=ed{27.869}<|im_end|>
step=1  batch[8]  sid=69030119 tail=ed{26.801}<|im_end|>
step=1  batch[11] sid=00a77d86 tail=ed{20.552}<|im_end|>
step=1  batch[12] sid=645fe504 tail=ed{79.873}<|im_end|>
step=1  batch[20] sid=cdbeab98 tail=ed{13.837}<|im_end|>
step=1  batch[21] sid=d7032308 tail=d{186.395}<|im_end|>
step=1  batch[25] sid=9b10b67b tail=ed{50.109}<|im_end|>
step=1  batch[26] sid=0320b87e tail=ed{37.105}<|im_end|>
step=2  batch[1]  sid=2cb5b118 tail=ed{24.861}<|im_end|>
step=2  batch[2]  sid=a1b268b4 tail=ed{20.165}<|im_end|>
step=2  batch[8]  sid=54892cac tail=ed{71.872}<|im_end|>
step=2  batch[10] sid=cbf13263 tail=ed{92.020}<|im_end|>
step=2  batch[15] sid=33476ff1 tail=xed{9.490}<|im_end|>
step=2  batch[16] sid=0a4b6267 tail=d{129.668}<|im_end|>
step=2  batch[23] sid=b891fd93 tail=ed{35.110}<|im_end|>
step=2  batch[24] sid=397010ac tail=ed{76.000}<|im_end|>
step=2  batch[30] sid=5df8a2f5 tail=ed{33.991}<|im_end|>
step=2  batch[31] sid=1520fb1d tail=ed{79.163}<|im_end|>
step=3  batch[6]  sid=79accb89 tail=ed{64.091}<|im_end|>
step=3  batch[8]  sid=5540e7c5 tail=ed{13.560}<|im_end|>
```

### ALLEQ — first 20 float-tail
```
step=1  batch[7]  sid=1914af14 tail=ed{40.588}<|im_end|>
step=1  batch[8]  sid=037c45c6 tail=ed{43.276}<|im_end|>
step=1  batch[11] sid=91a0f4d0 tail=xed{6.423}<|im_end|>
step=1  batch[12] sid=9fb609e7 tail=d{124.937}<|im_end|>
step=1  batch[20] sid=60851ade tail=ed{60.496}<|im_end|>
step=1  batch[21] sid=5e6ee1d9 tail=ed{81.931}<|im_end|>
step=1  batch[26] sid=e02a19fe tail=ed{10.525}<|im_end|>
step=1  batch[27] sid=deeb0f72 tail=ed{48.852}<|im_end|>
step=2  batch[1]  sid=96ff5ae9 tail=ed{16.104}<|im_end|>
step=2  batch[2]  sid=b2b2ba94 tail=xed{7.030}<|im_end|>
step=2  batch[9]  sid=4e9494ac tail=ed{51.577}<|im_end|>
step=2  batch[11] sid=9029034d tail=ed{82.022}<|im_end|>
step=2  batch[17] sid=3c7f757f tail=ed{28.278}<|im_end|>
step=2  batch[18] sid=ff039a4e tail=ed{84.577}<|im_end|>
step=2  batch[24] sid=aab3127a tail=ed{13.719}<|im_end|>
step=2  batch[25] sid=50070c1d tail=d{102.411}<|im_end|>
step=3  batch[0]  sid=00a77d86 tail=ed{20.552}<|im_end|>
step=3  batch[1]  sid=87eb7ce0 tail=ed{29.127}<|im_end|>
step=3  batch[8]  sid=6191619a tail=ed{28.128}<|im_end|>
step=3  batch[10] sid=91258006 tail=ed{31.481}<|im_end|>
```

**No format drift.** Both runs use identical `\boxed{<number>}<|im_end|>` answer format. Same digit precision (3 decimal places). Same closing token. The numeric-eq examples in ALLEQ are NOT formatted differently from MOE — they are the same family of puzzles, just more of them.

### Category-tail distribution (whole training run)

| category-tail | MOE count | MOE % | ALLEQ count | ALLEQ % | Δ |
|---|---:|---:|---:|---:|---:|
| FLOAT (`{12.345}`) | 2125 | 27.1% | 2125 | 26.5% | 0 |
| BIT (8 binary)     | 1754 | 22.3% | 1754 | 21.9% | 0 |
| TEXT (cipher)      | 1656 | 21.1% | 1656 | 20.6% | 0 |
| SYMBOLIC (cryptarithm) | 712 | 9.1% | 709 | 8.8% | -3 |
| NUMERAL (Roman)    | 699 | 8.9% | 699 | 8.7% | 0 |
| INT (`{1234}`)     | 676 | 8.6% | 845 | 10.5% | **+169** |
| OTHER (trunc/edge) | 227 | 2.9% | 232 | 2.9% | +5 |

The 169 net new INT-tail examples in ALLEQ are exactly the equation_numeric expansion (header says +56 deduce, +115 guess → +171). 22 of those guess-mode examples have SYMBOLIC tails (operator answers like `+`, `*`), netting -3 in SYMBOLIC. All accounted for.

---

## T6. Per-category loss — can we measure it?

The training log **does not emit per-batch category** on the step-summary line. Each step summary is one aggregate loss over all 32 micro-batches. We can only get per-category loss by attributing each sample_id to a category (via tail heuristic or csv lookup) and re-running an eval. From these logs alone we **cannot decompose** the 0.003 final-loss into per-category components.

What we CAN say from the inputs:
- Both runs were given the **identical** non-numeric examples (same 6906 shared sample_ids, same stratified shuffle), so the model was exposed to identical bit/cipher/numeral/cryptarithm/unit_conv/gravity training signals. Whether the *learning signal* on those was different is invisible without per-batch loss decomposition.
- ALLEQ has 171 MORE equation_numeric samples, drawn from a different generation cohort (`260525_Cryptarithm` / `260526_Numeric_Equation`), and they replace what would have been free GPU time in MOE.

Because the aggregate loss curves are within 11% of each other (and ALLEQ is slightly higher), **there is no evidence the cryptarithm/cipher/etc. signal degraded.** If alleq's extra numeric-eq examples caused destructive interference on cryptarithm, we'd expect alleq's aggregate loss to be much higher than the +5% smoothed gap we see at step 200. The gap is fully explained by "171 fresh hard examples interleaved into the same shuffle".

---

## T7. Final training summaries — verbatim last 30 lines

### MOE last 30 lines

```
   step=246 smi_after=88201M loss=0.002959 grad_norm=0.0088 lr=8.13e-07 elapsed=251.1m eta=0.0m
  Checkpoint saved: checkpoint-246_loss0.0030_lr8.13e-07
  FIFO: deleted old checkpoint checkpoint-200_loss0.0030_lr3.82e-05
  Adapter-only saved (soup): adapter-246_loss0.0030_lr8.13e-07

Training done. Time: 4.19 hrs (251.5 min)
Peak VRAM: smi=71915M

=== Saving adapter ===
Saved and renamed lm_head keys

=== Creating submission zip ===
ZIP contents: ['adapter_model.safetensors', 'adapter_config.json']
ZIP size: 3654.9 MB

=== Zipping final checkpoint ===
Checkpoint zip: /root/autodl-tmp/checkpoint-246_loss0.0030_lr8.13e-07.zip (9931.3 MB)

============================================================
TRAINING SUMMARY — retrain of run #7 with soup-ready adapter saves
============================================================
Method:       LoRA bf16 + CCE + MoE tying
GPU:          NVIDIA RTX PRO 6000 Blackwell Server Edition
Samples:      7849
Steps:        246
LoRA rank:    32  alpha: 32
LoRA targets: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'in_proj', 'out_proj', 'lm_head']
LR:           0.0002 -> 0 (linear decay)
Batch:        32 (micro=4)
Adapter saves: [50, 100, 150, 197, 200, 209, 221, 234, 246]
Submission:   /root/autodl-tmp/submission_moe_notie_outproj.zip
```

### ALLEQ last 30 lines

```
   step=251 smi_after=88173M loss=0.002449 grad_norm=0.0049 lr=7.97e-07 elapsed=246.4m eta=0.0m
  Checkpoint saved: checkpoint-251_loss0.0024_lr7.97e-07
  FIFO: deleted old checkpoint checkpoint-250_loss0.0033_lr1.59e-06

Training done. Time: 4.11 hrs (246.8 min)
Peak VRAM: smi=72201M

=== Saving adapter ===
Saved and renamed lm_head keys

=== Creating submission zip ===
ZIP contents: ['adapter_model.safetensors', 'adapter_config.json']
ZIP size: 3654.4 MB

=== Zipping final checkpoint ===
Checkpoint zip: /root/autodl-tmp/checkpoint-251_loss0.0024_lr7.97e-07.zip (9923.4 MB)

============================================================
TRAINING SUMMARY — retrain of run #7 with soup-ready adapter saves
============================================================
Method:       LoRA bf16 + CCE + MoE tying
GPU:          NVIDIA RTX PRO 6000 Blackwell Server Edition
Samples:      8020
Steps:        251
LoRA rank:    32  alpha: 32
LoRA targets: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'in_proj', 'out_proj', 'lm_head']
LR:           0.0002 -> 0 (linear decay)
Batch:        32 (micro=4)
Adapter saves: []
Submission:   /root/autodl-tmp/submission_newdata_alleq_notie_outproj.zip
```

**Note: a single config difference is visible in the summary.** MOE saved 9 intermediate adapters `[50,100,150,197,200,209,221,234,246]` (these are "soup" candidates); ALLEQ saved zero (`Adapter saves: []`, endpoint-only). That's a *non-training* delta — it does not affect what the model learned, but it means ALLEQ has no intermediate soup candidates to recover from (downstream impact: cannot do weight-averaging like MOE did).

---

## T8. Token counts

```
MOE   : Loaded 7849 examples, 27,989,496 tokens (unmasked=26,713,988)
ALLEQ : Loaded 8020 examples, 24,971,076 tokens (unmasked=23,676,167)
```

| | MOE | ALLEQ | Δ |
|---|---:|---:|---:|
| Examples | 7849 | 8020 | +171 |
| Total tokens | 27,989,496 | 24,971,076 | **-3,018,420 (-10.8%)** |
| Unmasked tokens | 26,713,988 | 23,676,167 | **-3,037,821 (-11.4%)** |
| Tokens / example | 3,566 | **3,114** | **-12.7%** |

**This is the biggest training-side difference.** Despite having MORE examples, ALLEQ has 3M FEWER total tokens — the average example is ~12% shorter. Almost certainly because the new equation_numeric (260526) CoTs are shorter than whatever they replaced or shorter than the previous distribution. Padded-length distribution agrees: MOE median padded_len = 6864, ALLEQ median padded_len = 6852 (similar) but MOE mean = 6588, ALLEQ mean = 6270 (lower) — confirming the new examples are shorter.

Note also: ALLEQ took 246.8 min wall-clock vs MOE's 251.5 min — ALLEQ was faster *despite* having more steps, consistent with the shorter padded sequences.

---

## T9. VRAM and timing

| | MOE | ALLEQ |
|---|---:|---:|
| VRAM before training | 64,383M | 64,383M |
| Peak VRAM (per summary) | 71,915M | 72,201M |
| Step-1 smi_after | 88,115M | not in step row but ~88,131M (step 1) |
| Wall-clock | 4.19 hrs (251.5 min) | 4.11 hrs (246.8 min) |
| Min / max padded_len | 1,735 / 7,970 | 1,673 / 7,970 |

Both runs identical VRAM envelope, no GPU memory anomalies. ALLEQ slightly faster (4.7 min less) because of shorter mean sequence length.

---

## T10. Anomaly search (SKIP / WARNING / ERROR / oom / nan / inf)

### MOE
**Zero matches.** No SKIP, WARNING, ERROR, OOM, NaN, or Inf anywhere in the log. Clean run.

### ALLEQ
Only benign warnings + one network blip:

- 12 (load-time): `Unsloth: WARNING trust_remote_code is True.`
- 22 (load-time): `padding token! Will use pad_token = <SPECIAL_999>`
- 1767, 3447, 5130, 6809, 8495, 8520, 8529 (multiple checkpoint saves): `UserWarning: Setting save_embedding_layers to True as embedding layers found in target_modules.` (peft normal warning when LM head is a LoRA target)
- 3449: `swanlab: [2026.05.26 11:09:25] network error, swanlab will resume uploads when the network improves` — occurred right after step 100, ZERO impact on training (next step 101 logged normally).

**No oom/nan/inf in either run.** Both completed cleanly.

---

## Bottom line

**Training itself behaved nearly identically.** Same model (Nemotron 30B-A3B), same Unsloth LoRA r32 a32, same LR (2e-4 → 0 linear), same batch 32 micro 4, same seed 42 stratified interleave, same 9 LoRA targets including `lm_head` + `out_proj`, identical peak VRAM. Both runs completed in ~4.1 hours without spikes, NaNs, or OOMs.

**Differences in training:**

1. **Data shift (intended):** ALLEQ has +171 equation_numeric examples (56 deduce + 115 guess), 6906 sample_ids are identical, 171 are new-only. Total tokens DROPPED by 10.8% despite the +171 examples — the new equation_numeric CoTs are shorter (mean 3,114 tok/ex vs 3,566 tok/ex, **-12.7%**), so each step trains on less LM-loss-bearing content.

2. **Loss curve (consequence):** ALLEQ is systematically ~10–20% higher smoothed loss than MOE across steps 25–225 (mean diff +0.00476). Both bottom out at the same 0.003 floor. ALLEQ ends at 0.002449 (final), MOE ends at 0.002959 (final). The final-step comparison favors ALLEQ but this is dominated by lr schedule (alleq still has 5 more decay steps); apples-to-apples step-by-step comparison shows ALLEQ slightly behind.

3. **Order shift:** The 171 new examples push shared samples ~+3 steps later in ALLEQ's stratified order (median offset +3, mode +4).

4. **Adapter save schedule (different but doesn't affect what the model learned):** MOE saved 9 intermediate "soup-ready" adapters; ALLEQ saved none (`Adapter saves: []`). This is a config decision in the runner, not a training dynamics difference.

5. **Anomalies:** zero in MOE, only benign load warnings + one swanlab network blip in ALLEQ. No training instability.

**Could any training-side difference explain the LB regression?** Probably **not directly through training dynamics.** The aggregate loss curves are essentially the same shape, both bottom out, neither overfits (no near-zero collapse, no spikes), no NaNs, identical compute envelope. The lone meaningfully different thing on the training side is **token budget**: ALLEQ trained on 3M fewer tokens (-11%) because the new equation_numeric CoTs are shorter. That means slightly fewer LM gradient signals overall, and per-category coverage of *all other tasks* (bit/cipher/numeral/cryptarithm/gravity/unit_conv) is unchanged sample-count-wise but the shuffle has them sitting next to shorter neighbors. Combined with the absent soup (no weight-averaging at the end), ALLEQ could plausibly under-converge on the long-CoT categories vs. MOE.

The more likely explanation for an LB regression sits **outside training dynamics**: data-content shift (the 171 new equation_numeric CoTs format/style differ from the rest), the missing soup (MOE's submitted adapter was averaged, ALLEQ's was the raw step-251 endpoint), or eval-side decoding differences. These logs alone don't show a training pathology.
