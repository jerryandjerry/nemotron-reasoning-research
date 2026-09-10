# Cryptarithm CoT Template (current — matches `_honest_solver.py`, 2026-05-23)

A faithful template of the CoT now written to `<puzzle>/track/tree_cot.txt`. Placeholders use
`[UPPER_CASE]`; conditional branches are `// IF ... // ELSE ... // END`. The setup (sections 1–7)
comes from `cot_generator.make_prefix`; the DFS log (8) from `_honest_solver`; the answer (9) from
`emit_answer` + `_gen_solutions`. See `tree_cot_review.md` for the rationale behind each choice.

```
// 1. opening
I need to infer the transformation rule from the examples.

// 2. legend: symbols -> letters (A, B, ...), then operators -> f, g, h
First, let me assign letters to each symbol:
  [SYM] -> A
  [SYM] -> B
  ...
Operators:
  [OP_SYM] -> f
  [OP_SYM] -> g   // omit if fewer operators
  [OP_SYM] -> h

// 3. equations in letter form (examples then query)
I will convert all the equations to letter form:
  EX1 「[RAW_LHS]」 = 「[RAW_RHS]」 becomes [LHS] f [RHS] = [OUT]
  ...
  QUERY 「[RAW_QUERY]」 becomes [LHS] g [RHS]

// 4. operator background (priors from operator_dict.md)
In Alice world, multiplication, addition, subtraction and concatenating are the most common operations, but the results are usually noisy (off by ±1 or ±2, negated, or with operands reversed). For the operators in this puzzle:
  // for + - * (strong indicators):
  [OP_SYM], denoted as f, is usually [noisy_add|noisy_subtraction|noisy_mul], but I have seen it represent [FAMILY_LIST] in other puzzles, ordered by frequency
  // for an arbitrary symbol seen before:
  I have seen operator g represent [FAMILY_LIST] in other puzzles, ordered by frequency
  // for an arbitrary symbol never seen:
  I have not seen operator g in other puzzles
Anyway, I should analyze the pattern of the examples to figure out what they are.

// 5. reading-order / mode (RHS-operator-sign check)
// IF an operator symbol appears in an RHS:
[OP] appears in EX[N] at [the leading position|the trailing position]. In Alice world, only a subtraction operator can produce itself as a sign prefix in the output. This confirms [OP] represents a subtraction operation. Denote it as ~sub.
  // leading -> standard; trailing -> little-endian (state which)
// ELSE:
No operator symbol appears in any RHS. Reading order defaults to standard (tens digit first). If the full DFS finds no solution under standard order, I will retry with little-endian order (units digit first).
// END

// 6. is the query operator deducible?
// IF query operator appears in an example:
Operator [OP] in QUERY appears in EX[N], so I should analyze the equations and find out what the operator means.
// ELSE (unseen): no line here; the guess happens in section 9.

// 7. concat check
Let me also check if operators are concat operation:
  f: concat_fwd=[LETTERS], concat_rev=[LETTERS]. Output=[LETTERS]. [Neither matches, so f is not concat. | FWD/REV matches -> f is concat_fwd/rev.]
  ...
// IF all operators are concat: skip 8; go to the concat answer in 9.
// ELSE:
All [N] operator(s) remaining are arithmetic. We need to keep solving.
// END

// 8. the search (verbatim DFS log)
Writing equations (mode=[standard|little-endian], left is tens digit, right is units digit):
  Ex1: f([POLY], [POLY]) = [RHS_POLY]
  ...
[A..] each represent a distinct digit 0-9, leading symbol is never 0.
Unknown: op f, op g. [The operators are also distinct.]   // distinctness clause if >=2 arith ops

// digit-count pruning
The solution tree is huge, let me see if I can further prune some branches.
  f: EX[N] has [d]-digit RHS ([what it rules out]); ...
    // one survivor: Only [FAMILY] survives the digit-count filter. Try f = [FAMILY]. Denote it as [~mul|~add|~sub].
    // several:      Surviving candidates: [FAMILY] (denote ~X), [FAMILY] (denote ~Y). ...
  ...
[No further pruning found. | ...]

Now search for a digit assignment.
Try operators f = [FAMILY], g = [FAMILY]:
  EX[N]: LHS [EXPR] ~[op] [EXPR] in [LO,HI], RHS [LETTERS] in [LO,HI] -> [VAR] in {..}, ...   // narrow from one eqn
  ...
  // when one example's operands AND RHS are fully known, LOCK eagerly:
  We now know all the operands in EX[N]: [a] ~[op] [b], RHS = [r].
    [r] = [a]×[b], so lock f = [VARIANT].          // forced; or "more than one meaning ... try each:" then recurse
  // after locking, that operator is used by its EXACT variant and intervals tighten
  Assume [VAR] = [VAL]:                              // branch on the least-unknown equation
    ... [LHS] can never equal RHS -- backtrack       // backtrack on contradiction
  Every equation checks: f = [VARIANT], g = [VARIANT]. Solution: A = .., B = .., ...

// 9. answer (ONE of the five paths)

// (a) concat, derived from examples:
Query operator g is concat_[fwd|rev] (from the examples); answer = [LETTERS] = [SYMBOLS]

// (b) arithmetic, derived (query operator seen in examples):
Query operator g = [VARIANT] (from the examples), so QUERY is [a]×[b] = [r]; answer = [LETTERS] = [SYMBOLS]

// (c) guess the operator (query op unseen, examples solvable) — distinctness then prior:
The examples are arithmetically solvable, so I'll guess h is an arithmetic operator too[; PRIOR].
// forced by distinctness (no prior): Since [V] and [V] have appeared in the puzzle, by the distinct operator rule h must be the remaining operation, [FAMILY].
// tie: Since [V] has appeared in the puzzle, by the distinct operator rule h is [FAMILY] or [FAMILY]; [PRIOR].
Guess h = [VARIANT], so QUERY is [EXPR] = [r]; answer = [LETTERS] = [SYMBOLS]

// (d) guess a missing symbol (answer needs a digit absent from the examples):
... so QUERY is [EXPR] = [r].
But digit [D] doesn't show up in the examples, so I have to guess what symbol it maps to.
I know the full set of operand symbols, ordered by ASCII:
  ! " # $ % & ' ( ) / : < > ? @ [ \ ] ^ ` { | }
Excluding every symbol that already appears in this puzzle (marked X), I still have:
  [ALPHABET WITH USED SYMBOLS REPLACED BY X]
I know the symbol-to-digit mapping is assigned uniformly at random, so none of the remaining candidates is more likely than another.
Since I have to choose, I will just take the first one that is left: [D] -> [SYM], answer = [LETTERS] = [SYMBOLS]

// (e) blind fallback (no arithmetic solution found):
The examples aren't arithmetically solvable; guess concat_fwd; answer = [LETTERS] = [SYMBOLS]

// suffix (always, two lines)
I will now return the answer in \boxed{}, the answer is
\boxed{[ANSWER_SYMBOLS]}
```
