# bit_manipulation CoT template (spec for format conformance)

The CoT is the forward-only log of the solver. Sections in this exact order. No prose anywhere except
the three fixed sentences given below. Position 0 = leftmost bit. N = number of examples (7–10).

## 1. Header (verbatim)
`We need to deduce the transformation by matching the example outputs.` then a blank line.

## 2. Output transcription — one block per example, 1-based
`Output <k>: <8 bits>` then 8 lines `<pos> <bit>` then a blank line.

## 3. Output bit columns
`Output bit columns (with bitsum as hash)` then 8 lines `<pos> <column> <hash>`.
hash = count of '1's in the column as one char; `a` if the column is uniform (all-0 or all-1).

## 4. Input transcription — `Input <k>: <8 bits>` blocks, same shape as §2.

## 5. Nine family sections, in order: Identity, NOT, Constant, AND, OR, XOR, AND-NOT, OR-NOT, XOR-NOT
Each section: name line, candidate lines, blank, `Matching output with <Name>` + 8 lines
`<pos> <labels…|absent>`, blank, `Left` block, `Right` block.
- Identity/NOT candidates: `<j> <column> <hash>[ match <p…>]` for j = 0..7.
- Constant candidates: `0 …` and `1 …`.
- AND/OR/XOR and XOR-NOT candidates: paired labels `<ab> <ba> <column> <hash>[ match <p…>]`,
  grouped by circular distance 1..4 with a blank line between groups (distance-4 group has 4 lines).
- AND-NOT/OR-NOT candidates: ordered labels `<ab> <column> <hash>[ match <p…>]`, distance groups 1..7.
- `Left`/`Right` blocks: chain-walk lines (each a sequence of stepped labels, the first failing label
  suffixed `x`), or `none`; then `Best: <prefixed labels>: <len>` or `Best: none`.

## 6. Selecting block
`Selecting`, blank, `Lefts` + 9 lines `<Family> <best|none>`, blank, `Rights` + 9 lines, blank,
`Left longest: <n>`, `Right longest: <n>`, blank, `Left winner: <Family> yes/no ×9` (comma-separated),
`Right winner: …`, blank, `Best left: …`, `Best right: …`, blank, `Truncated left: …`,
`Truncated right: …`, blank, `Tentative from right` (positions 7→0, rule or `pending`), blank,
`Tentative` (0→7), blank, `Preferred from left` (0→7; unresolved as `?<key>`), blank, `Preferred`,
blank, `Matching` (resolved: `<pos> <rule>`; pending: `<pos> ?<key> - <Family verdict ×9>` where each
verdict is `<Family> absent` or the adopted candidate label), blank, `Perfect match` + 9 lines
`<Family> yes/no`, blank, `Matched` + 8 lines (`<pos> <rule>` or `<pos> none`), blank.

**Winner-dependent orientation (both are conformant):** the WINNING side leads. The layout above is the
Left-winner case. When the RIGHT chain wins, the `Right winner` / `Best right` / `Truncated right` lines precede
their Left counterparts, and the `Preferred from <side>` header is `Preferred from right` (positions 7→0),
mirroring `Tentative from right`. So `Preferred from left` (0→7) and `Preferred from right` (7→0), and the
Right-before-Left ordering of the winner/Best/Truncated trio, are BOTH spec-conformant — chosen by which chain won.

## 7. Scan tiers — ONLY when `Matched` contains `none` positions, in this order
- `Maj` section: hits-only candidate lines `<jkl> <column> <hash> match <p…>` (combination order),
  blank, `Matching output with Maj` + 8 lines, blank, then a `Matched` checkpoint (8 lines, still-unresolved as `none`), blank.
- If unresolved remain: `Ch` section, hits-only: `<sab> <column> <hash>[ match <p…>]`
  (s = selector, a = ones-branch, b = zeros-branch; scan order s,a,b ascending), then `Matching output with Ch`.
- If unresolved remain: `NOT-AND` / `NOT-OR` / `NOT-XOR` (hits-only, `<ab> …`, only non-empty sections)
  and `NOT-Maj` (`<jkl> …`).
- If unresolved remain: 3-term sections named `<OP1>(<OP2>)` (e.g. `XOR(AND)`), hits-only lines `<a>(<bc>) <column> <hash> match <p…>`.
- After the tiers: final `Matched` (8 lines; positions no scan resolved stay `none`), blank.

## 7b. Pattern consistency check — when the global rule justifies the per-bit picks
Appears after the final `Matched` (and any scan tiers) when the single global rule that reproduces every
example either (a) assigns a position a different (but also example-fitting) rule than the per-bit selection
chose — revise it, or (b) confirms an example-ambiguous Maj/Ch pick that already matches the rule. Structure:
- `Pattern consistency check`
- `The resolved positions follow a single rule: output bit i = <readable global rule>.`
- case (a) — `Positions <p…> already match this rule.` then one line per revised position: `Position <p>: <old> was selected, but <new> (column <col>) matches output column <p> and is what this rule gives at position <p>; revise to <new>.`
- case (b) — one line per position spelling out the rule's value there in its reduced form (a majority/choice collapses to a simpler op where a shifted operand falls off the end or two operands coincide): `Position <p>: <rule-value> (column <col>) matches output column <p> and is what this rule gives at position <p>.`
- blank line.
In both cases the shown `(column <col>)` equals output column `<p>` from §3 — self-supporting evidence the rule fits every example; it need not have been surfaced by the (pending-scoped) scan tiers above.
The `<new>` rule's fit is proven in-line by the shown `(column <col>)` equalling output column `<p>` from §3; it need not have been surfaced by the (pending-scoped) scan tiers above.

## 7c. No-single-rule fallback — when NO uniform rule reproduces every example
If the search finds no single global rule (every uniform formula fails some example), state the fallback
explicitly instead of going silently to `Selected`, as one line after the final `Matched` (preceded by a blank
line, then `Selected` follows directly):
- `No single rule reproduces all examples; each output bit is taken from its own column match.`
This is mutually exclusive with §7b (a Pattern consistency check only appears when a global rule *does* exist).

## 7d. Composite (3-step) rule statement — when the single rule isn't per-bit nameable
If the single global rule is a 3-step composition (a `T3`, e.g. `OP(u, OP(u,u))`) whose per-position labels
aren't all nameable, state the rule (don't go silent), as one line after the final `Matched` (preceded by a
blank line, then `Selected` follows directly) — same rule wording as §7b, but standalone (no `Pattern
consistency check` header, no per-position lines, because the bits aren't cleanly nameable):
- `The resolved positions follow a single rule: output bit i = <readable 3-step rule>.`
The per-bit realization is the `Selected` block (whose per-column labels reproduce every example; they need
not textually match the stated rule's decomposition). Mutually exclusive with §7b and §7c.

## 8. `Selected` — 8 lines `<pos> <rule>`; if a Pattern consistency check ran, the revised rules; unresolved positions appear as `default 1`.

## 9. Applying (verbatim formats)
`Applying to <query>`, `Input` + 8 lines `<pos> <bit>`, `Output` + 8 lines, one per position:
- `<p> I<j> = <v>` · `<p> NOT<j> = NOT(<x>) = <v>` · `<p> C<d> = <d>`
- `<p> AND<ab> = AND(<x>,<y>) = <v>` (OR/XOR same) · `<p> AND-NOT<ab> = AND(<x>,NOT(<y>)) = <v>` (OR-/XOR- same)
- `<p> Maj<jkl> = Maj(<x>,<y>,<z>) = <v>` · `<p> Ch<sab> = Ch(<s>,<x>,<y>) = <v>`
- `<p> NOT-AND<ab> = NOT(AND(<x>,<y>)) = <v>` (NOT-OR/NOT-XOR same) · `<p> NOT-Maj<jkl> = NOT(Maj(<x>,<y>,<z>)) = <v>`
- `<p> <O1><a>(<O2><bc>) = <O1>(<x>,<O2>(<y>,<z>)) = <v>` · `<p> default 1 = 1`

## 10. Ending (verbatim, 3 lines)
`I will now return the answer in \boxed{}` / `The answer in \boxed is` / `\boxed{<8 bits>}`

## Forbidden anywhere
Any prose beyond §1/§10; `lock`; `§`; `∥`; `reading order`; `exotic`; `State`; `Conclusion`; the `y`
fail marker (only `x`); any cryptarithm/numeric-equation wording.
