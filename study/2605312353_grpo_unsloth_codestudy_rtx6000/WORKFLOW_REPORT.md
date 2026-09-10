# Per-Step Distillation Reward Design — Deep-Read Workflow Report

**Date:** 2026-06-09 (CDT)
**Trigger:** User request to design a per-step + final-answer reward that distills the python-solver CoT structure, replacing the weak 4-reward stack in `train_grpo_v1.py`.
**Method:** 5-phase Workflow (7 subagents, 49 tool calls, 390K subagent tokens, 12.6 min runtime).

## Files read line-by-line

1. `study/2606090112_grpo/_gen_crypt.py` — cryptarithm solver
2. `study/2606090112_grpo/_gen_eq.py` — numeric-equation solver
3. `01_data/260525_Cryptarithm/260607_TrainData/260607_Cryptarithm_gtTrue.csv` — SFT training data (has `solver_cot` column)
4. `03_eval/2606071911_cryptarithm_gtTrue_260606swap_on_260607_cryptarithm_gtTrue/cryptarithm_gtTrue_260606swap_on_260607_with_probs.csv` — SFT model's actual generations

## Phase 1+2 findings — what the solver emits vs what the model produces

### The gold CoT has a DETERMINISTIC closed-set structure

Cryptarithm gold (every CoT, 100% coverage in the training CSV):

```
[opener]  "I need to infer the transformation rule from the examples."
[opener]  "Both the operands and operators are unknown..."
[BPE/symbol block]  "Split the BPE merged tokens..." OR "First, I should read each equation..."
[denotation]  N lines: "EX{i}  {lhs} = {rhs}  becomes  {spaced_lhs} = {spaced_rhs}"
[denotation]  "QUERY {q}  becomes  {spaced_q}"
[functions]   "Now denote them as unknown functions:" + per-op "{sym} -> {letter}"
[operands]    "The other symbols are operands; assign a letter to each in the order of appearance:"
[letter form] "I will convert all the equations to letter form:"
[priors]      "Prior knowledge for this kind of question:" + EXACTLY 6 numbered items
[tree open]   "Now start to traverse the solution tree, reading order = [rightward, leftward]:"
[per reading §n]
  "§{n} reading {one of: rightward | leftward fully... | leftward on digit only... | ...}:"
  "§{n}.1 writing equations:"
  "§{n}.2 quick check to see whether any operator is concatenation:"
  "§{n}.3 prune solution trees by the rhs digit-count before entering:"
  "§{n}.4 narrow the operand domains before branching:"
  (optional) "§{n}.5 search the remaining unknowns by branching..."
  EXACTLY ONE "Conclusion of step §{n}: {brace}, X of Y resolved."
[summary]     EXACTLY ONE "Summary: ... Confirm reading order = ..., {assignments}"
[query block] "Now solve the QUERY {letter}({a},{b}), applying {letter}(a,b) = {variant}"
              "{letter}({a},{b}) = {expr} = {val}"
              "map the digits back to symbols, {dmap}, so {transformed} -> {mid}"
              (optional) "mapping the subtraction sign... so {mid} -> {ans}"
[terminator]  "I will now return the answer in \\boxed{}, the answer is"
[terminator]  "\\boxed{<ans>}"
```

Closed enumerations (verbatim from solver source):
- `~tag` ∈ `{~add, ~sub, ~mul, ~concat}`
- variant ∈ `{a×b, a+b, |a-b|, b-a, a∥b, b∥a}` plus ±1/±2 noisy variants of `a×b` and `a+b`
- direction strings ∈ closed 4-element set
- 6 numbered prior rules (closed text)
- `_finalize` enforces: `in [` not `∈`, `×` not `·`, lowercase `lhs`/`rhs`, no `unknown` in `\boxed{}`

### The SFT model has memorized the SCAFFOLD near-perfectly

**Structural fidelity is ~100% even on wrong-answer rows.** Headers, priors, section markers, vocabulary all reproduced verbatim.

What the model gets wrong (the 1063 / 7077 = 15% failure rows):

| Failure mode | Frequency | Diagnosis |
|---|---|---|
| **C2: `unknown` in Summary brace** | 71% of crypt failures | Model writes `Conclusion of step §1: {f=unknown, g=a×b, h=unknown}, 1 of 3 resolved`, ships `unknown` to Summary, then... |
| **D2: `\boxed{D C E}` raw letters** | Same rows as C2 | Skips `map the digits back to symbols` block, boxes raw letters from `unknown` operators |
| **Truncation at 7680 tokens** | 507/7077 = 7.2% | No `</think>`, no final `\boxed{}` |
| **Duplicate `\boxed{}` line** | most non-truncated rows | Model emits boxed BOTH before and after `</think>` — gold emits ONCE |
| **`∈` instead of `in [`** | sporadic | `_finalize` invariant slip |

The model has NO `<think>` tags as gold content — those come from chat template, not the solver. Per user: drop any reward that requires `<think>`/`<reasoning>`/`<answer>` tags.

## Phase 4 design — 7 reward functions

| # | Function | Role | Range | Targets |
|---|---|---|---|---|
| 1 | `reward_final_answer` | **PRIMARY outcome** (Kaggle metric verbatim) | [−3.4, +3.7] | D1 wrong answer, D2 raw-letter box, D4 `unknown` in box, D5 duplicate-box |
| 2 | `reward_conclusion_semantic` | **PRIMARY process** | [−2.0, +2.0] | **C2 the 71% failure mode**: `unknown` in Summary brace |
| 3 | `reward_solve_query_remap` | **PRIMARY process** | [−1.3, +1.0] | Missing `map the digits back to symbols` (D2 source) |
| 4 | `reward_skeleton_alignment` | SCAFFOLD (closed-set headers) | [−1.6, +1.7] | Format collapse under exploration |
| 5 | `reward_closed_vocab_discipline` | SCAFFOLD (`_finalize` invariants) | [−1.6, +1.2] | `∈`, `·`, `LHS`/`RHS`, wrong `~tag` |
| 6 | `reward_termination_quality` | GUARDRAIL | [−1.5, +0.8] | Truncation + duplicate-box |
| 7 | `reward_solver_alignment` | RESERVED (off) | [+0.0, +0.5] | Diagnostic SequenceMatcher bonus |

### Why "primary outcome ~6× scaffolding"

`reward_final_answer` D1 alone: +2.5 / −1.5 = 4.0-point swing per trajectory. Scaffolding rewards (skeleton + vocab) are distributed across many sub-checks — typical per-trajectory advantage on those is far smaller after group normalization. **A trajectory that breaks gold structure but boxes the correct answer outscores a structurally perfect but wrong-answer trajectory.** This is the design invariant preventing collapse to "mimic scaffolding and ignore the puzzle."

### Anti-gaming defenses (each formally checked in the workflow)

| Game | Defense |
|---|---|
| Emit empty `\boxed{ }` to dodge D2 letter check | D1 still mismatches gold (−1.5), D4 fires (−1.0), D5 still requires exactly one box → net −2.7 |
| Spam many `\boxed{...}` hoping one matches | D5 penalty; D1 takes only LAST box, so spam doesn't help |
| Lie in Conclusion brace: `{f=a×b, g=a×b, h=a×b}, 3 of 3 resolved` | C1+C2 pass +1.3 but locked variants don't match EX checks → D1 fails −1.5 → net negative vs +5.5 for honest solve |
| Emit all headers with no content | Skeleton (+1.7) but C-block fails (−1.3) and D1 fails (−1.5) → net −1.1 vs +5.5 honest |
| Vacuous remap "`map the digits back to symbols, {}, so X -> X`" | Q3 passes (+0.2) but D1 still adjudicates real box content → wrong-box still −1.5 |

### Dataset columns needed (already in 260607 CSV per workflow inspection)

| Column | Source | Used by |
|---|---|---|
| `id` | CSV | logging |
| `category` | CSV (filter to crypt + eq_numeric only) | reward routing |
| `prompt` | CSV (re-rendered with chat template + suffix) | model input |
| `answer` (=`gold_answer`) | CSV | D1 |
| `gold_cot` | CSV `solver_cot` column | C-block reference, optional alignment bonus |
| `oversampling` | CSV | per-row weight on final reward |
| `ex_count` | parsed from prompt | skeleton S7 (`EX{i}` ≤ `ex_count`) |
| `symbol_pool` | parsed from prompt | D2 (cryptarithm raw-letter check) |

## What changed in `train_grpo_v1.py`

1. **Removed** the inline 4-reward stack (lines 277–385 old). Replaced with `from grpo_rewards import REWARD_FUNCS, extract_final_answer, verify`.
2. **Added** `KEEP_CATEGORIES = {cryptarithm_deduce, cryptarithm_guess, equation_numeric_deduce, equation_numeric_guess}` filter — rows in other categories are dropped before tokenization.
3. **Changed** `DATA_CSV` default to `/root/autodl-tmp/data/260607_Cryptarithm_gtTrue.csv` (has `solver_cot` column).
4. **Added** dataset prep helpers `_parse_ex_count` and `_parse_symbol_pool` that build the columns the rewards need from each prompt.
5. **Added** `gold_cot`, `gold_answer`, `oversampling`, `ex_count`, `symbol_pool` to each row TRL passes through as `**kwargs`.
6. **Added** `import re` at top (was previously inside the removed inline reward block).

## Smoke test (run by the workflow)

| Completion | Total score |
|---|---|
| Near-gold synthetic completion | **+9.9** |
| Garbage | **−3.2** |
| Gap | **13 points** |

`reward_final_answer` alone gives a 4.8-point gap (+3.5 vs −1.3), confirming the "primary dominates structure" invariant.

## What's left before launch

| | What | Who |
|---|---|---|
| ✅ | Training script updated to use new rewards + filtered dataset | done |
| ⏸ | User reviews `train_grpo_v1.py` + `grpo_rewards.py` | **user** |
| ⏸ | Power on AutoDL instance, share new SSH port | **user** |
| ⏸ | Confirm SEED_ADAPTER path on instance points to SFT 0.86 adapter | user or me |
| ⏸ | Upload data CSV (already exists locally) + script + rewards module via SFTP | me |
| ⏸ | Run `test_imports.py` to verify TRL 0.22.2 has all the GRPO knobs we set | me |
| ⏸ | Run stress test (`STRESS_TEST=1`) to verify VRAM fits with KL ref model | me |
| ⏸ | User approves final config per routine §2 | user |
| ⏸ | Launch real training, ~3 hr ETA, 5-min polls | me |

## Files in this study folder

```
2605312353_grpo_unsloth_codestudy_rtx6000/
├── REPORT.md                  ← code-study + adaptation plan (earlier)
├── WORKFLOW_REPORT.md         ← this file
├── train_grpo_v1.py           ← GRPO training script (updated to use new rewards)
├── code/                      ← 6 downloaded Unsloth GRPO notebooks
└── docs/                      ← 5 downloaded Unsloth RL docs

../../02_train/
└── grpo_rewards.py            ← the 7-reward module (smoke-tested)
```
