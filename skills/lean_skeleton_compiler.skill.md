---
name: lean-skeleton-compiler
description: Compile a bridge plan into a high-quality multi-sorry Lean skeleton for real subgoal extraction
---

# Lean Skeleton Compiler Skill

You are compiling a validated bridge plan into Lean 4 code.

Your goal is NOT to prove the theorem outright.
Your goal is to produce a Lean skeleton that:

- elaborates as far as possible,
- exposes multiple real local subgoals via `sorry`,
- and preserves the target theorem goal.

## Input

You will receive:

- **Target theorem statement**
- **Optional formal statement**
- **Graph context**
- **Bridge plan JSON**
- **Compiler requirements JSON**
- **Optional previous Lean/decomposition error feedback**

## Hard requirements

1. Start with `import Mathlib`.
2. Use a theorem name starting with `discovery_`.
3. Keep the same theorem goal; do NOT weaken or replace the theorem.
4. Use tactic-style Lean.
5. Create explicit local lemmas / `have` blocks for bridge propositions and chain steps.
6. Use `sorry` only where a real local subgoal should remain open.
7. Precede each local bridge proposition with a comment line:

   `-- BRIDGE-PROP: <proposition-id>`

8. Precede each local reasoning step with a comment line:

   `-- BRIDGE-STEP: <step-id>`

9. Every proposition with grade `B` or `D` in the bridge plan MUST appear in the skeleton.
10. Do not use `admit`.
11. Do not use one giant final `sorry` as a substitute for the bridge chain.
12. Emit multiple local `have` blocks so that Lean can expose multiple subgoals.
13. When possible, the first several bridge lemmas should be sibling local goals that do NOT depend on earlier `sorry`-backed lemmas.
14. Do NOT use `section`, `namespace`, `end`, or custom notation blocks; produce one plain Lean file with imports, optional local `def`/`let`, and a single main theorem.
15. If `target_proposition_id` is provided, you MUST include an explicit marker line

   `-- BRIDGE-PROP: <target_proposition_id>`

   for that proposition in the skeleton body, even if it is also the theorem target.

## Quality goals

- Prefer many small local `have` goals over one giant unfinished block.
- Introduce minimal definitions first so later `have` statements are well-typed.
- Avoid referencing names or APIs you are not reasonably confident exist.
- If some bridge proposition is easier to encode as a comment + `have`, do that.
- If the prompt gives a minimum placeholder count, meet or exceed it.
- Prefer this pattern:
  - first define local notation / helper defs
  - then emit several independent `have hX : ... := by sorry`
  - only after that start combining them
- Keep the file syntactically minimal: imports, optional local helper defs, one theorem.
- If `preferred_sibling_proposition_ids` is provided, use those propositions as the first sibling local goals.
- If `target_dependency_ids` has multiple entries, you MUST make the theorem goal an explicit `And`-chain / conjunction over those dependency propositions and begin the proof with repeated `constructor`, so Lean can expose one unresolved goal per sibling.
- If `target_proposition_id` is provided, make sure the target proposition is represented explicitly as a local bridge proposition block rather than only implicitly as the final theorem statement.

## Output format

You MUST return exactly one JSON object:

```json
{
  "premises": [
    {"id": "existing_node_id", "statement": "premise used"}
  ],
  "steps": [
    "theorem discovery_example ...",
    "import Mathlib\n\n-- BRIDGE-PROP: P1\n-- BRIDGE-STEP: S1\ntheorem discovery_example ... := by\n  ...\n  sorry\n"
  ],
  "conclusion": {
    "statement": "Lean skeleton for [statement]",
    "formal_statement": "theorem discovery_example ..."
  },
  "module": "lean",
  "domain": "geometry"
}
```
