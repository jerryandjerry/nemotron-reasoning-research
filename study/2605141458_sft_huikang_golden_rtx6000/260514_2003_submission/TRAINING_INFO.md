# Training Info — Huikang 0408 + lkevincc Golden Cryptarithm

> Trained: 2026-05-14 (Chicago) / finished 2026-05-15 09:03 China time
> Instance: AutoDL RTX PRO 6000 Blackwell (95 GB), port 21578
> SwanLab: `260514_1532_sft_huikang_golden_r32_a32_lr2e4_seq8192_bs32_rtx6000`

## Goal

Same proven 0.84 training setup, but swap the 65 weakly-oversampled cryptarithm
samples for 735 real golden cryptarithm solutions from lkevincc's solver
(handles arithmetic / little-endian / mixed-concat that huikang's solver couldn't).

## Data

`260514_huikang_golden.csv` — 6,906 unique rows → **7,849 expanded** via `oversampling` column.

| Source | Rows | Note |
|---|---|---|
| `pretok_0408_decoded` | 6,171 | exact 0.84 winning-run samples |
| `lkevincc_golden` | 735 | golden cryptarithm solutions (arithmetic, little_endian, mixed_concat, etc.) |

Category breakdown (expanded, 15 categories):
bit_manipulation 1754, cipher 1656, unit_conversion 1070, gravity 1055,
numeral 730, equation_numeric_deduce 658, arithmetic 315, little_endian 276,
equation_numeric_guess 126, mixed_concat 108, cryptarithm_deduce 54,
mixed_concat_little_endian 26, cryptarithm_guess 11, query_unseen_concat 5,
pure_concat 5.

vs the 0.84 run: cryptarithm oversampling dropped (627→54 deduce, 154→11 guess),
735 golden cryptarithm samples added. Net 7,849 vs 7,830.

## Config (identical to 0.84 run except data + ordering)

| Parameter | Value |
|---|---|
| Model | Nemotron-3-Nano-30B-A3B bf16 (Unsloth) |
| Method | LoRA bf16 + Cut Cross-Entropy + MoE weight tying |
| LoRA | r=32, alpha=32, dropout=0 |
| Targets | q/k/v/o_proj, up/down_proj, in/out_proj, lm_head (9 + manual lm_head) |
| Optimizer | AdamW, betas=(0.9, 0.95), eps=1e-8, wd=0 |
| LR | 2e-4 → 0 linear decay |
| Batch | 32 (micro=4, ga=8) |
| Max seq len | 8192 |
| MoE tying | True (5888 params) |
| Ordering | **Stratified** — interleave categories by rank/category_size so each batch is proportional; no adjacent duplicates |
| Steps | 246 |

Tokenization matches huikang's corpus.py exactly: prompt via apply_chat_template
(enable_thinking=True), completion via raw Tokenizer from tokenizer.json,
boxed regex with re.escape, response-only masking.

## Results

- **Train time:** 250.5 min (4.17 hrs), ~1.02 min/step
- **Final loss:** 0.0074 (step 246)
- **Loss curve:** 0.43 → 0.06 by step 10 → ~0.007–0.011 plateau
- **Peak VRAM:** 88.8 GB / 95 GB
- **Trainable params:** 888,154,112 / 32,466,091,456

## Outputs

- `submission.zip` (1.37 GB) — adapter_config.json + adapter_model.safetensors, CRC verified
- `checkpoint-246_loss0.0074_lr8.13e-07.zip` (2.86 GB) — full checkpoint for resume

Adapter post-processing applied: base_model_name_or_path →
`metric/nemotron-3-nano-30b-a3b-bf16`, inference_mode=True, lm_head keys renamed
to `base_model.model.backbone.lm_head.`

## Score

Pending Kaggle submission.
