# Numeric Equation Puzzle Data — Build & Reproduction Guide

> **Audience:** an agent with zero prior context who must understand this folder and reproduce the work
> exactly. This file is the **index**: it explains the problem, the pipeline, the review/feedback loop,
> and the hard rules — and points to the canonical files for the details (do not duplicate their content
> here; open them).
>
> **Tackling a *different* Nemotron category from scratch (e.g. `gravity`)?** This folder is one worked
> example. The general, category-agnostic method is
> [`../HOW_TO_CRACK_A_NEMOTRON_CATEGORY.md`](../HOW_TO_CRACK_A_NEMOTRON_CATEGORY.md) — start there; it points
> back here as its reference implementation.

---

## 0. TL;DR (one screen)

- **Task:** for each `equation_numeric_*/question.txt` puzzle, produce an honest, deterministic,
  forward-only Chain-of-Thought (CoT) that deduces the answer, and write it to that puzzle's
  `track/tree_cot.txt`.
- **One generator produces all of it:** [`_gen_eq.py`](_gen_eq.py). Run it and it (re)writes all 732
  puzzle folders and prints the score.
  ```bash
  cd 01_data/260523_Numeric_Equation && python3 _gen_eq.py
  ```
- **Current state:** 618/732 on the reference metric; **732/732 = 10/10 on the multi-agent honesty
  audit** (a clean sweep). The score is a *sanity signal only* — never the optimization target.
- **The CoT spec** is [`cot_template.md`](cot_template.md). **The review rubric + the running defect log
  (#1–#10)** is [`agent_review_instruction.md`](agent_review_instruction.md). **The hard-won judgment —
  dead ends, leakage traps, tokenizer landmines, design rationale, philosophy — is
  [`LESSONS.md`](LESSONS.md).** Read all three before changing anything; `LESSONS.md` is the one that saves
  you from re-walking the wrong paths.

---

## 1. The problem

`equation_numeric` is a Nemotron puzzle category. Each `question.txt` gives a few example equations and
one query, e.g. ([equation_numeric_4c19c1cb](equation_numeric_4c19c1cb/question.txt)):

```
In Alice's Wonderland, a secret set of transformation rules is applied to equations. Below are a few examples:
76?45 = 8163
72?62 = 207
91&14 = 9114
Now, determine the result for: 35?09
```

The operands are **literal numbers** (no symbol→digit cipher). What must be deduced is, for each operator
symbol, **which operation it means** and **which direction the digits are read**. It is the cryptarithm
operator-locking sub-problem with the operands already settled. Concretely the solver must decide, per
puzzle:

1. **Reading order** — rightward (left-to-right) or leftward (units-digit-first / digits reversed).
2. **Each operator's family** — one of four *noisy* families, or "exotic" if none fit:
   `~mul`, `~add`, `~sub`, `~concat` (each "noisy" = exact value, or off by ±1/±2; `~sub` also allows the
   negated / operand-reversed forms; `~concat` is forward `ab` or reverse `ba`).

The full method, wording, and structure are specified in [`cot_template.md`](cot_template.md).

---

## 2. Directory map (the index)

**Whole setup at a glance.** Everything under `260523_Numeric_Equation/` is produced/maintained here; the
**input data** and the **metric spec** live *outside* this folder:

```
2026_Nemotron/                              ← repo root (workspace root)
├── ANSWER_EXTRACTION.md                    ← official-metric spec      (link: ../../ANSWER_EXTRACTION.md)
└── 01_data/
    ├── <two source CSVs>                    ← INPUT data; exact paths = the DATA / USED_CSV / UNUSED_CSV
    │                                          constants near the top of _gen_eq.py (read them there)
    └── 260523_Numeric_Equation/            ← THIS folder
        ├── README.md                       ← this guide — start here
        ├── LESSONS.md                      ← hard-won judgment: dead ends, leakage traps, rationale
        ├── cot_template.md                 ← the CoT spec (structure, wording, sign/char legend)
        ├── agent_review_instruction.md     ← review rubric + defect log #1–#10
        ├── operator_dict.md                ← per-symbol frequency prior
        ├── _gen_eq.py                      ← THE generator (reads the input CSVs, writes everything below)
        ├── _audit/                         ← disposable scratch for audit shard files
        └── equation_numeric_<id>/  × 732   ← one folder per puzzle:
            ├── question.txt                ←   the prompt (INPUT)
            ├── answer.txt                  ←   stored reference answer (sanity metric only, §5)
            ├── label.txt                   ←   our problem-type label (from classify())
            ├── cot_<reference>.txt         ←   third-party reference CoT (comparison only; not our pipeline)
            └── track/tree_cot.txt          ←   OUR generated CoT — THE DELIVERABLE
```

| Path | What it is |
|---|---|
| [`README.md`](README.md) | This guide — the index / entry point. |
| (two source CSVs under `01_data/`) | **The INPUT data.** Exact paths = `DATA` / `USED_CSV` / `UNUSED_CSV` in [`_gen_eq.py`](_gen_eq.py); the generator reads these to build every puzzle. They are not in this folder. |
| [`_gen_eq.py`](_gen_eq.py) | **The generator.** Reads the source data, solves every puzzle, writes all folders + the CoTs, prints the score. The single source of truth for the method — the CoTs are 100% generated, never hand-edited. |
| [`cot_template.md`](cot_template.md) | **The CoT spec.** Exact structure, wording, the `§`-tree, the sign/character legend, and the "CoT is a SYSTEM LOG" governing principle. Change the generator and this file together; they must always agree. |
| [`agent_review_instruction.md`](agent_review_instruction.md) | **The review rubric + defect log.** The prompt given to audit agents, plus the canonical list of every defect ever found (#1–#10) with its root cause and fix. **This is the changelog — read it to learn what has already gone wrong and must not recur.** |
| [`LESSONS.md`](LESSONS.md) | **The hard-won judgment.** Dead ends already tried and rejected (with *why*), the exact leakage patterns that kept getting caught, the tokenizer sign-collision landmines, the design rationale, the philosophy, and practical gotchas. The single most valuable file for not repeating mistakes. |
| [`operator_dict.md`](operator_dict.md) | Per-symbol frequency prior (`+`→~add, `-`→~sub, `*`→~mul are strong; arbitrary symbols ~uniform). Used by the generator's `get_prior()` to order the candidate families. |
| [`../../ANSWER_EXTRACTION.md`](../../ANSWER_EXTRACTION.md) | How the official Kaggle metric extracts and verifies the answer (last `\boxed{}` via `rfind` + 3-branch verify). The generator's `extract_final_answer()` / `verify()` mirror it exactly. |
| `_audit/` | Scratch space for audit chunk files (the sharded ID lists from a review run). Disposable. |
| `equation_numeric_<id>/` | One folder per puzzle (732 total). Contents below. |

**Per-puzzle folder** (e.g. [`equation_numeric_00d8b3db/`](equation_numeric_00d8b3db/)):

| File | Role |
|---|---|
| `question.txt` | The puzzle prompt (input). |
| `answer.txt` | The stored reference answer (used only for the GT-match sanity metric — **not** ground truth, see §5). |
| `label.txt` | Our own problem-type label (see §5), derived by `classify()` for our understanding. |
| `track/tree_cot.txt` | **Our generated CoT — the deliverable.** |
| `cot_*.txt` | A third-party reference CoT, kept for comparison only; **not** part of our pipeline and never used to justify anything. |

---

## 3. The method (high level — spec is in `cot_template.md`)

The CoT is a **literal system log** of a constraint-propagation search:

1. **Setup (unnumbered prose):** assign letters to symbols, convert equations to letter form, state the 6
   prior-knowledge items (families/variants, distinctness, leading-0 rule, sign-in-RHS rule, reading-order
   rule, per-operator frequency prior), and the sign-confirmation line.
2. **`§`-tree = the two readings.** `§1` = rightward (always tried first); `§2` = leftward (opened only if
   `§1` leaves an operator unresolved or is ruled illegal). Each `§N` has four sub-steps:
   - `§N.1` write equations in this reading's frame, then the **gut-check** (two passes: tail-sign, then
     leading-0 over the scan list `[all examples, then the QUERY operands]`).
   - `§N.2` concat check.
   - `§N.3` digit-count prune (pick each operator's family to try first).
   - `§N.4` enter the tree and search (greedy lock-and-verify, with backtracking).
3. **Summary** picks the winning reading (more operators resolved; `§1` on a tie), then the query is solved.
4. **Exotic / unseen fallbacks:** an operator no family fits in either reading → hardcoded
   `max(a,b) mod min(a,b)` guess; an unseen query operator → binary arithmetic/concat guess.

The exact line templates, the sign/character legend (every special char pinned to a codepoint + token id),
and the four governing system-log rules are in [`cot_template.md`](cot_template.md). Do not paraphrase its
wording — "same condition → 100% identical wording" is a hard rule (§8).

---

## 4. The generator (`_gen_eq.py`)

- **Inputs:** a source-solver CSV and an "unused" CSV. The exact paths are the `DATA` / `USED_CSV` /
  `UNUSED_CSV` constants near the top of [`_gen_eq.py`](_gen_eq.py) — read them there rather than
  hard-coding a path; `OUT_DIR` is the script's own directory.
- **Run:** `python3 _gen_eq.py` from this folder. It writes every `equation_numeric_<id>/` folder
  (question/answer/label/reference-CoT) and `track/tree_cot.txt`, then prints the GT-match score and the
  per-label breakdown to stdout.
- **Key functions (line numbers drift — grep the names):**
  | Function | Role |
  |---|---|
  | `gen_cot(d)` | Builds one puzzle's full CoT (the setup prose + the `§`-tree via `attempt`). |
  | `attempt(m, n)` | Narrates one reading `m` as `§n`: writes equations, runs the §N.1 gut-check, the §N.2 concat check, §N.3 prune, §N.4 search. Returns the resolved assignment or `None` (illegal/unresolved). |
  | `lock_concat(op)` | The **one shared** "lock an operator against the concat family" routine: decide direction from the first example whose operands differ, defer equal-operand examples, confirm on the rest. Used by both the §N.1 gut-check and §N.2. |
  | `consistent_ops`, `fam_variant_order`, `first_variant_match` | The arithmetic variant ordering + greedy lock walk (`~sub` order `\|a-b\| > a-b > -\|a-b\| > b-a`, then ±1/±2). |
  | `emit_answer` | Computes the query answer and the sign re-attach (raw signed value on the `answer =` line; sign re-attached only on the following line). |
  | `classify(d)` | Derives our `label.txt` problem-type (our taxonomy, not a model input). |
  | `extract_final_answer`, `verify` | Mirror the official metric (see `../../ANSWER_EXTRACTION.md`). |
  | `main()` | Iterates all puzzles, writes folders, prints the score. |

The CoTs are **only** produced by this generator — there is no manual editing of `tree_cot.txt`. To change
a CoT you change the generator, regenerate, and re-verify (§7).

---

## 5. Scoring / the metric

- The generator prints **`OUR GT-match: N/732`** — how many of our boxed answers equal `answer.txt` under
  the official extraction+verify logic ([`../../ANSWER_EXTRACTION.md`](../../ANSWER_EXTRACTION.md)).
- **This is a sanity signal ONLY.** `answer.txt` is *not* ground truth (it is not official; no one knows if
  it is correct). The GT-match must **never** drive or justify a logic choice, and a logic decision must
  never be called "correct" because it matches it. The only real correctness criterion is **honest +
  forward-only + deterministic + template-faithful** (verified by the audit, §6).
- `label.txt` categories (our own taxonomy, for understanding the landscape, not fed to any model):
  `arithmetic_right_to_left`, `arithmetic_left_to_right`, `unseen_operator`, `concatenation`,
  `ambiguous_arithmetic`, `exotic_operation`, `no_rule_fits`, `unseen_leading_zero`. Current per-label
  rates are in the generator's stdout.

---

## 6. The review & audit process

Correctness is established by a **multi-agent, line-by-line honesty audit**, not by the score.

1. **Shard the 732 puzzle IDs into 10 lists:**
   ```bash
   ls -d equation_numeric_*/ | sed 's#equation_numeric_##; s#/##' | sort > /tmp/all_ids.txt
   awk 'NR==FNR{total++; next} {idx=int((FNR-1)*10/total); print > sprintf("/tmp/shard_%02d.txt", idx)}' \
       /tmp/all_ids.txt /tmp/all_ids.txt
   ```
2. **Launch 10 fresh, no-context reviewer agents in parallel** (one per shard, run concurrently). Each
   agent is told to: read [`agent_review_instruction.md`](agent_review_instruction.md) and
   [`cot_template.md`](cot_template.md), then for every id in its shard read `question.txt` +
   `track/tree_cot.txt` and **recompute every line by hand** — never trust the CoT. They score each puzzle
   0–10 and report a table + a defect list. (See the exact reviewer prompt body in
   `agent_review_instruction.md`; the per-agent prompt also calls out the most-recent changes for extra
   scrutiny.)
3. **What reviewers verify** (the rubric, in full, is `agent_review_instruction.md`): forward-only (no line
   uses a later step's result); arithmetic (every lock, every "far from"/"not", concat fwd/rev, digit
   reversal, boxed answer); template/structure fidelity; and each specific defect #1–#10 not recurring.
4. **A puzzle is "clean" at 10/10.** The goal is 732/732. The audit is what cleared the set; the score did
   not.

> Reviewers must be **fresh / no shared context** — the point is an independent recomputation that catches
> any dishonesty or leakage the generator's author would be blind to.

---

## 7. The feedback & improvement loop

This is exactly how every fix in [`agent_review_instruction.md`](agent_review_instruction.md) (#1–#10) was
made. Follow it precisely.

```
        ┌─────────────────────────────────────────────────────────────┐
        │ 1. AUDIT (§6) surfaces a defect, OR the human flags a puzzle  │
        └─────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ 2. Send the human a CLICKABLE puzzle link + the exact lines,  │
        │    explain the root cause, and WAIT for the human's fix       │
        │    direction. NEVER assume or implement a fix unilaterally.   │
        └─────────────────────────────────────────────────────────────┘
                                   │  (human directs the fix)
                                   ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ 3. Implement in _gen_eq.py ONLY (no hand-editing CoTs).       │
        │    Keep cot_template.md in sync with the new wording/logic.   │
        └─────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ 4. Regenerate (python3 _gen_eq.py) and VERIFY:                │
        │    a. score printed (expect it to hold unless logic changed)  │
        │    b. diff boxed answers vs the pre-fix baseline (see below)  │
        │    c. enumerate which puzzles' narration changed = the scope  │
        │    d. hand-read every changed puzzle                          │
        └─────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────────────┐
        │ 5. Record the fix in agent_review_instruction.md (#N) and     │
        │    re-run the full 10-shard audit (§6) to re-clear all 732.   │
        └─────────────────────────────────────────────────────────────┘
```

**Verification techniques used at step 4 (reuse them):**

- **Prove a narration change moved zero answers** — capture boxed answers before and after, diff:
  ```bash
  for d in equation_numeric_*/; do id=${d%/}; id=${id#equation_numeric_};
    ans=$(grep -o '\\boxed{[^}]*}' "$d/track/tree_cot.txt" | tail -1); echo "$id $ans"; done | sort > /tmp/after.txt
  diff /tmp/before.txt /tmp/after.txt    # expect empty for a narration-only change
  ```
- **Reconstruct the pre-fix baseline definitively** — copy `_gen_eq.py` to `/tmp`, revert just the changed
  block in the copy, run it (its `OUT_DIR` is `/tmp`, so it does **not** touch the real folders), and diff
  the two outputs. This is how "0 of 732 answers changed" claims were proven.
- **Enumerate the scope of a narration change** — `grep -rl "<the new line's text>" equation_numeric_*/track/tree_cot.txt`
  tells you exactly which puzzles the change touched; hand-verify each.

---

## 8. Hard rules (non-negotiable — these caused most of the rework)

1. **The CoT is a SYSTEM LOG, not a polished explanation.** Do a step → print what it did → print the
   resulting running state; repeat. No refinement, no cover, no fake prints. (Top section of
   [`cot_template.md`](cot_template.md).)
2. **Forward-only.** A line may use *only* values derived by lines above it. Never pre-compute a later
   step's result into the current line (this rule killed defects #5 and #8).
3. **Honesty over score; stay true to the logic even if points are lost.** The GT-match is a reference
   signal, never the optimization target, never a justification (§5).
4. **Deterministic.** Every step is explainable and repeatable; the same logic applies to every puzzle. Any
   random choice scores 0. Guessing is allowed only when nothing is deducible, and the guess itself must be
   deterministic (e.g. the exotic `max-mod-min` and the unseen-operator binary rule).
5. **Same condition → 100% identical wording.** The generator's fixed line-templates *are* the vocabulary;
   `cot_template.md` documents it; no paraphrase. (Reviewers flag any drift.)
6. **One shared routine per concept.** "Lock an operator against a family" is a single function used
   everywhere (`lock_concat` for concat, the variant walk for arithmetic) — no special-case shortcuts
   (defects #5, #10).
7. **Human-directed fixes.** When a defect is found, surface it with a clickable puzzle link and **wait**
   for the human's fix direction; do not implement on your own (§7 step 2).
8. **Only ever edit `_gen_eq.py`** to change CoTs; never hand-edit a `tree_cot.txt`. Keep
   `cot_template.md` synced. Do not modify user-managed config/env files.

---

## 9. Current state (last validated)

- **618/732** reference GT-match.
- **732/732 = 10/10** on the full 10-shard honesty audit — clean sweep, zero defects.
- Defects #1–#10 in [`agent_review_instruction.md`](agent_review_instruction.md) are all `[fixed]`.
- Generator, `cot_template.md`, and the defect log are in sync.

---

## 10. Reproduce from scratch

1. Read [`cot_template.md`](cot_template.md) (the spec) and [`agent_review_instruction.md`](agent_review_instruction.md)
   (the rubric + what has already gone wrong).
2. Ensure the source CSVs referenced by `_gen_eq.py` (the `USED_CSV` / `UNUSED_CSV` constants) exist at
   their paths.
3. `cd 01_data/260523_Numeric_Equation && python3 _gen_eq.py` → regenerates all 732 folders + CoTs and
   prints the score. Confirm `OUR GT-match: 618/732`.
4. Run the full audit (§6). Confirm 732/732 = 10/10.
5. To improve: follow the feedback loop (§7) — never deviate from the hard rules (§8). Log every fix in
   `agent_review_instruction.md` and re-audit.
