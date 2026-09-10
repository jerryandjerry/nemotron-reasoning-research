# NumericEq solver — self-contained, gold-blind

Portable copy of the deterministic `equation_numeric` ("Alice's Wonderland") CoT solver. Everything it
needs is in this folder; no other repo paths are referenced. It reads a puzzle's `question.txt` and
returns a chain-of-thought ending in `\boxed{<answer>}`. It never reads the answer — it solves blind.

## Requirements
- Python 3.7+
- Standard library only (no `pip install`).

## Solve a puzzle
Put each puzzle in a folder `equation_numeric_<id>/` next to `_gen_eq.py` with a single `question.txt`,
then:
```bash
python3 _gen_eq.py            # solve every equation_numeric_*/ beside the script
python3 _gen_eq.py <id>       # solve just one
```
Or from Python:
```python
import _gen_eq as g
cot, ans = g.gen_cot(g.load_folder("path/to/dir_containing_question_txt"))   # returns the CoT string
```
`gen_cot` reads only `question.txt`, runs the forward-only operator + reading-order search, and returns
the CoT ending in `\boxed{<answer>}`.

## Files
| file | role |
|---|---|
| `_gen_eq.py`  | **solver entry `gen_cot(d)`** — operator/reading search + template emission |
| `reasoners/`  | candidate-operation functions it imports (`equation_numeric.py`, `store_types.py`) |
| `run_tests.py`| self-test harness (reproduction check) |
| `test_cases/` | one golden `question.txt` + `cot.txt` pair per question type |

## Verify it works on this machine
```bash
python3 run_tests.py
```
Runs the solver on each bundled `test_cases/<name>/question.txt` and compares its output byte-for-byte
to the golden `cot.txt` generated on the reference machine. Expected:
`8/8 cases reproduced the reference CoT byte-for-byte. OK`. Exit code is 0 iff all pass.

The 8 cases are one per question type:

| type | case |
|---|---|
| arithmetic_left_to_right  | arithmetic_left_to_right_1db0c6fe |
| arithmetic_right_to_left  | arithmetic_right_to_left_7ecdae14 |
| concatenation             | concatenation_7c72ad99 |
| ambiguous_arithmetic      | ambiguous_arithmetic_094bf548 |
| unseen_operator           | unseen_operator_2f7e0e78 |
| exotic_operation          | exotic_operation_1f0fbe5f |
| unseen_leading_zero       | unseen_leading_zero_3687bc41 |
| no_rule_fits              | no_rule_fits_58fed63a |

A FAIL means the environment differs from the reference (a different Python version, or an edited/
missing file). The solver is fully deterministic, so a matching environment reproduces every CoT exactly.
