You are reviewing a Chain-of-Thought (CoT) written by a student for a puzzle, judging its correctness and logic.

You are given two things: the puzzle (`question.txt`) and the student's CoT. That is all. There is no answer key — no one knows the ground-truth answer. Judge the CoT purely on whether its reasoning makes sense.

## What makes a good CoT
- **Deterministic.** Every step follows by a rule, not a guess. Guessing is allowed only when nothing can be deduced, and even then the guess must follow a stated, repeatable rule (if it guesses A here, the same logic must force A in every comparable situation). **Any arbitrary or random choice scores 0.**
- **Forward-only.** Every line is justified by the puzzle and the lines above it. No line may use a fact that a later step discovers, or work backward from the answer.
- **Self-contained.** The reasoning reaches its answer from the puzzle alone. A step that only makes sense if you already knew the answer is not valid reasoning.

## Score
Read the CoT as a skeptic. For each step ask: does this follow from what came before, or is it asserted/assumed/guessed? Does the chain actually arrive at the answer it boxes? Score 0–10 on how sound, deterministic, and self-justifying the reasoning is.

Return a table, each row: question id, score (0–10), and reasons.
