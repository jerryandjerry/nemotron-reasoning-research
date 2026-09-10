# Cryptarithm training inspection — why our data didn't help (2026-05-25)

## Result
Our cryptarithm CoTs did **not** improve the model and slightly hurt it.
Per-category val (cryptarithm only, official metric):

| model | cryptarithm correct | LB |
|---|---|---|
| moe (baseline, no extra crypt) | 1/68 | 0.86 |
| newdata | 0/68 | 0.85 |
| goldonly | 0/68 | 0.84 |
| gt2x | 0/68 | 0.85 |

(equation_numeric stayed ~0.92; bit_manipulation dipped 0.77→0.74. Note: 68/71 val cryptarithm
puzzles are in our training set — not held out — so this is essentially *train-set* performance.)

## Two walls (both fatal, both empirical)
1. **Non-termination / looping.** Generation budget = 7,680 tokens. On cryptarithm the model's median
   generation **is** 7,680 and **75% truncate mid-search with no `\boxed{}`**. It happens even when
   our training CoT for the same puzzle is short (e.g. id 90feb0c5: our CoT 1,160 tok → model 7,680,
   truncated). The model learned the DFS *format* but not its *control*: it repeats a narrowing line
   verbatim until the ceiling. 780/800 of our CoTs branch (median 23 `Assume`, 16 backtracks) — the
   structure that triggers the loop.
2. **Symbol-parsing hallucination.** Of the ~25% that finish, **88% build a wrong symbol→letter
   legend** (drop a real symbol, invent an absent one, merge `>'` into one operator) and mis-transcribe
   the equations — then run a flawless DFS on a corrupted puzzle → wrong. 0/17 finished outputs
   reproduced our answer; 10 were puzzles our solver solves correctly that the model corrupted.

(A third item I earlier called "answer mangling" is real but **immaterial**: the eval CSV used a stale
first-`}` extractor; it differs on 3 rows and flips 0 wrong→correct vs the live metric.)

## Literature (confirms all of the above)
- Backtracking/reflection in CoT data is a documented **loop trigger** in reasoning models.
- Explicit backtracking-search CoT often **underperforms a direct method** on arithmetic puzzles (Kempner).
- LLMs **collapse on combinatorial/search reasoning** past a shallow depth (PuzzleBench, seqBench, SATBench).
- **PAL / Program-of-Thoughts** (offload the search to executed code) is the strongest documented fix.

## Fix hierarchy
1. **Program-of-Thoughts (offload to code) — strongest, but RULED OUT here.** The eval harness submits a
   LoRA adapter and runs a *fixed* vLLM `generate → extract_final_answer(\boxed) → verify` pipeline; the
   model's **text** is scored and nothing executes its code. So PoT is not usable (see "How to confirm").
2. **No-code path: drop the search log, go direct/linear.** Short (~≤1.5k tok), no backtracking display
   (only the forced solution path), plus loop-mitigated decoding. Terminates → beats the looping 0%, but
   the parsing wall caps the ceiling low.
3. **Exclude cryptarithm.** It's ~7% of the mix and currently net-negative. Given the depth-collapse +
   no-code constraints, highest expected value unless we commit to RL (GRPO) on direct-solution traces.

## How to confirm code execution is/ isn't allowed (answers "#1")
The eval notebook (`eval_kaggle_nemotron.ipynb`) and the competition Evaluation page (params "confirmed
by Kaggle staff") show the submission is a **LoRA adapter**, run through a fixed harness:
`LLM.generate(...) → extract_final_answer(text) → verify`. There is **no post-generation hook**, so the
model's output must already contain the answer in `\boxed{}` — its generated code would never run.
**Conclusion: no code execution → PoT out → pursue #2 or #3.**
