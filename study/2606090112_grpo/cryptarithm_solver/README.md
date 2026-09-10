# Cryptarithm solver — self-contained

Portable copy of the deterministic cryptarithm CoT solver. Everything it needs is in this
folder; no other repo paths are referenced (the original repo-relative tokenizer path was
patched to load the co-located `tokenizer.json`).

## Requirements
- Python 3.9+
- `pip install tokenizers`   (the only third-party dependency; tested with 0.22.2)

## Solve a puzzle
```bash
python3 solve.py path/to/question.txt      # a single question file
python3 solve.py path/to/puzzle_dir        # a directory containing question.txt
echo "<puzzle text>" | python3 solve.py    # puzzle on stdin
```
Or from Python:
```python
import _gen_crypt as gc
cot = gc.gen_cot("path/to/dir_containing_question_txt")   # returns the CoT string
```
`gen_cot` reads only `question.txt`, runs the forward-only constraint-propagation DFS, and
returns the chain-of-thought ending in `\boxed{<answer>}`. It writes nothing into your
source puzzle dir (a temp copy is used).

## Files
| file | role |
|---|---|
| `_gen_crypt.py`   | **solver entry `gen_cot(name)`** — §-tree joint digit+operator DFS + template emission |
| `cot_generator.py`| `parse(question.txt)`, deterministic prefix steps, `apply_op`/`op_str`/`VARIANT_FN`/`CORES` |
| `operator_dict.md`| operator data table (read by `cot_generator`) |
| `tokenizer.json`  | BPE tokenizer, used for the 7680-token length check |
| `solve.py`        | CLI/stdin convenience wrapper |
| `_gen_solutions.py`, `_gen_crypt_aug.py`, `harvest.py` | **optional** puzzle-generation chain (generate-and-test new gold-bearing puzzles); only needed if you want to *create* puzzles, not solve them |

## Verify it works on this machine
```bash
python3 run_tests.py
```
Runs the solver on each bundled `test_cases/<name>/question.txt` and compares its output
byte-for-byte to the golden `cot.txt` that was generated on the reference machine. Expected:
`10/10 cases reproduced the reference CoT byte-for-byte. OK`. Exit code is 0 iff all pass.

The 10 cases span every subtype and a range of sizes (66–2987 lines, incl. two deep-search
cases):

| subtype | cases |
|---|---|
| arithmetic | arithmetic_00457d26, arithmetic_01ef1e3e |
| little_endian | little_endian_00c032a8, little_endian_012cab1f |
| pure_concat | pure_concat_0133bcec, pure_concat_02a04b59 |
| mixed_concat | mixed_concat_042f1e53 (deep), mixed_concat_09d5ee68 |
| mixed_concat_little_endian | mixed_concat_little_endian_0e2d6796 (deep) |
| query_unseen_concat | query_unseen_concat_07b440f0 |

A FAIL — most likely in the "Split the BPE merged tokens" section — means the environment
differs from the reference (usually a different `tokenizers` version, or an edited/missing file).
The solver is fully deterministic, so a matching environment reproduces every CoT exactly.

## Generate new puzzles (optional)
`harvest.py` builds gold-bearing puzzles, blind-solves them, and keeps GT-matching ones under
the 7680-token cap. Edit its `PLAN`/`CAP` and run `python3 harvest.py`; output goes to a CSV
in this folder.

## Note on the tokenizer path
`_gen_crypt.py` loads `tokenizer.json` from this folder first, falling back to the in-repo
location `../../02_train/260512_huikang_085/repo/tokenizer.json` if the local copy is absent —
so this folder works both standalone and in place.
