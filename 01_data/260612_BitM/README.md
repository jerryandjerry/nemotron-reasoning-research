# 260612_BitM — bit_manipulation harvest of the 248 golden-uncovered puzzles

> Built 2026-06-12 (Chicago). Status: **generated, pending user approval of the new Maj/Ch wording** before any training CSV is built.

## What this is
The 248 puzzles are the bit_manipulation rows of the full competition data that have NO golden CoT
(full 1,602 minus the 1,354 in `260514_huikang_golden_stripped.csv`) — exactly the population the
provider's solver failed on (its own traces are GT-true on only 11/248). Source of prompts/golds:
`01_data/260514_huikang_update/huikang_unused.csv` (cross-checked against `01_data/archive/train.csv`).

## Folder structure (same convention as cryptarithm/numeric-eq)
- `bitm_<id>/question.txt` — THE ONLY INPUT the solver reads
- `bitm_<id>/answer.txt` — gold; used ONLY by the post-hoc GT filter, never by the solver
- `bitm_<id>/track/tree_cot.txt` — the generated CoT (only for expressible hypotheses)
- `_harvest_summary.csv` — per-puzzle status (GT / WRONG / UNEXPRESSIBLE) + token length

## The solver (`_gen_bitm.py`, entry `gen_cot(folder)`)
Blind, deterministic, prior-ordered **joint word-level hypothesis search** (the prior was mined from
the 1,354 winning train assignments — no test golds involved):
`out = u(x) | OP(u1,u2) | Maj(u1,u2,u3) | Ch(sel,a,b)`, u ∈ {rotl/rotr k, shl/shr k fill-0/1, NOT-ed} (72 forms),
OP ∈ {AND, OR, XOR}. First hypothesis reproducing ALL examples wins; it is then decomposed into
per-position vocabulary rules (truth-table canonical) and emitted as the native per-bit template.
3-term hypotheses (`x OP (y OP z)`) decompose outside the per-bit vocabulary → **not emitted** (UNEXPRESSIBLE).

The TRACE is the provider's template: Output blocks → bit columns (+bitsum hash, `a` = constant column)
→ 9 family sections (candidates with `match p`, `Matching output with X`, Left/Right shift chains with
`x` terminators) → Selecting (Lefts/Rights/longest/winner/Best/Truncated/Tentative/Preferred/Matching
rescan/Perfect match/Matched) → **Maj/Ch fallback** (only when no family matches a position) → Selected
→ Applying → `\boxed{}`. The boxed answer is COMPUTED by applying the Selected rules to the query —
never injected. Differences from the provider's emission, by design:
- this engine never emits `default 1` (the guess) — unresolved positions go through the Maj/Ch fallback or the row is not emitted. NOTE: `default 1` DOES exist in 66 of the 1,354 train CoTs (102 positions); if those stay in training alongside the new rows, the corpus teaches two different continuations of the same `Matched … none` cue (guess vs escalate) — resolution pending user decision (engine blind-solves 29/66 of them GT-true with honest Maj/Ch traces, 0 wrong);
- the Matching-rescan line shows, per family, the candidate the procedure adopts (provider sometimes
  printed `absent` contradicting its own tables);
- chain-winner ties break toward the hypothesis-consistent chain (deterministic prior; provider used family order).

## Harvest result (2026-06-12)
| | count |
|---|---|
| traces written (expressible) | 142 |
| **GT-true (keepable)** | **127** |
| GT-false (hypothesis fit examples, missed gold — excluded) | 15 |
| unexpressible (3-term / no hypothesis) | 106 |
| line-level soundness (every Selected rule fits all examples; boxed = rules applied) | **142/142** |
| GT-true traces using the Maj/Ch fallback | 75 |

Token length (prompt+cot), GT-true: min 5,748 / p50 6,778 / p90 7,490 / max 8,052; 10 rows ≥ 7,680.

## ⚠️ NEW ELEMENTS in the emitted traces (v2, native-only — needs approval)
All prose, the `lock` lines (numeric-eq vocabulary), and the Ch derivation lines were REMOVED.
The only non-native elements remaining:
1. A conditional `Maj` family section — emitted ONLY when `Matched` leaves `none` positions —
   in the exact native grammar: `jkl <column> <hash>[ match p…]` lines + `Matching output with Maj`
   block, followed by a second `Matched` that resolves (the native template already repeats
   near-identical blocks, e.g. Tentative/Preferred/Matched/Selected).
2. The rule labels `Maj{jkl}` in Matched/Selected/Applying (`1 Maj037 = Maj(1,0,1) = 1`,
   same line shape as `AND(x,y) = b`). Unavoidable: expressing Maj at all requires a name.
Rows needing Ch are HELD (status HELD_CH) until the user picks a Ch wording; no Ch lines exist.

## Harvest result (v2, 2026-06-12)
| | count |
|---|---|
| GT-true pure-native traces | **85** (33 use the Maj section, 52 need no fallback) |
| HELD_CH (need Ch; awaiting wording decision) | 42 |
| GT-false (filtered out) | 15 |
| unexpressible (3-term / Maj-Ch-with-NOT / no-fit) | 106 |
| line-level soundness of written traces | 100/100 |

Token length (prompt+cot), GT-true: p50 6,821 / p90 7,825 / max 8,109; 9 ≥ 7,680.


## ✅ ORIGINAL SOLVER FOUND AND PORTED (2026-06-12, later)
The provider's actual generator lives at `02_train/260512_huikang_085/repo/reasoners/bit_manipulation.py`
(blind — never reads `problem.answer`). Ported to `_gen_bitm_orig.py` (+ `_store_types.py`) with 13
convention edits to match the training revision (1-based Output/Input numbering, no header/legend lines,
`Matching output with <X>`, single `?key`, bare unary labels in the rescan, single Preferred-side block,
no `y` fail-marker, no `truncated` suffix, ascending mirror-pair order in match lists, 3-line ending).
**Byte-validation: 1,346/1,354 training CoTs reproduced EXACTLY (99.4%).** The 8 divergent rows are
precisely the 8 known-defective traces (Selected contradicts examples — old code state); the ported
version produces internally-consistent GT-TRUE traces for 7 of those 8 (ef2fe526 wrong).
NEXT: graft the word-prior selection + conditional Maj section into `_gen_bitm_orig.py` and re-emit the
248 harvest on the original machinery; retire the reverse-engineered emitter (`_gen_bitm.py`) for the
9-family portion.

## Harvest result (v3 = ported ORIGINAL machinery + blind steering, 2026-06-12 latest)
Engine: `_gen_bitm_v3.py` (= `_gen_bitm_orig.py` + steering hooks active ONLY when a word-prior is
passed; unsteered FULL regression: 1,346/1,354 byte-identical, unchanged). Steering ladder on the 248:
unsteered 11 -> tie-breaks 51 -> pref-over-length 62 -> placement gates 67 -> resolution steering **73 GT**.
| | count |
|---|---|
| GT-true traces (original grammar throughout) | **73** |
| HELD_CH (awaiting Ch wording) | 41 |
| WRONG (filtered) | 28 |
| unexpressible (3-term / Maj-Ch-with-NOT / no-fit) | 106 |
| soundness | 101/101 |
GT lengths: p50 7,014 / p90 8,116 / max 8,698 — 16 over 7,680, and the max exceeds the 8,192 TRAIN cap.
Reference: v2 emitter reached 85 GT (12-row gap = resolution-stage hooks not yet ported; closable).


## ✅ FINAL REGENERATION (2026-06-12, user-approved option 4 + 3 notations)
THE solver = `_gen_bitm_final.py` (one deterministic function question.txt → CoT): ported huikang
machinery + blind word-prior + tier ladder (families → Maj → Ch → NOT-wrap → 3-term → default 1
only after all scans absent). All 1,602 folders regenerated (`bitm_<id>/track/tree_cot.txt`).

| population | honest GT | default-assisted GT | wrong (0×) | byte-identical to stored |
|---|---|---|---|---|
| 1,354 train | 1,332 | 8 | 14 | 644 |
| 248 unused | 164 | 3 | 81 | — |
| **total** | **1,496** | **11** | 95 | |

**Trainable candidates: 1,507 / 1,602.** Soundness 1,602/1,602 (every Selected rule fits all
examples; boxed = rules applied). Foreign-wording scan: ZERO hits (no crypt/numeric-eq markers, no prose).
⚠️ Token lengths (GT rows): p50 6,978 / p90 7,621 / max 9,349 — 141 ≥ 7,680 and **75 ≥ 8,192 (train
hard cap — these cannot train as-is; needs a decision: drop, or revisit caps)**.
Per-row detail: `_regen_summary.csv`. Approved notation examples: `bitm_000b53cf` (Ch),
`bitm_796c8b63` (NOT-wrap), `bitm_0245b9bb` (3-term) — each also has `track/tree_cot_PROPOSED.txt`.

## Scripts
- `_setup_folders.py` — builds the 248 folders from huikang_unused.csv
- `_gen_bitm_final.py` — THE solver (gen_cot(folder)); `_gen_bitm_v3.py` = engine; `_gen_bitm.py` = blind prior
- `_regen_all.py` — regenerates all 1,602 + full verification → `_regen_summary.csv`
- `_gen_bitm.py` — the solver/emitter (question.txt → CoT string)
- `_harvest.py` — runs all folders, GT-checks, writes `_harvest_summary.csv`, prints token stats
