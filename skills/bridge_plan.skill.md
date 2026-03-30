---
name: bridge-plan
description: Convert a proof route into explicit subpropositions, a full reasoning chain, and A/B/C/D bridge grades
---

# Bridge Plan Skill

You are compiling a theorem proof route into a strict intermediate bridge layer.

For open problems, preserve the route's genuinely new hypotheses, constructed objects, and risky mechanisms explicitly. Do not collapse them into a literature summary or into a generic statement that "the gap remains".

Your output must make the proof route reusable for:

- later Lean formalization
- experiment triage
- benchmark reporting

## Input

You will receive:

- **Target statement**
- **Current graph context**
- **A best reasoning route**
- **Optional judge feedback**
- **Optional downstream failure feedback**

## Goal

Produce:

1. A complete list of explicit propositions used in the route
2. A complete reasoning chain from seeds/bridges to target
3. An A/B/C/D grade for each proposition / bridge point

## Grade semantics

Assign exactly one grade per proposition; use the semantics below consistently.

- **A**: Local and clear enough to formalize directly.
- **B**: A human-common-sense structural bridge: the mathematical direction is reasonable and useful, but it is still best viewed as an intermediate intuition/representation bridge rather than an immediately checkable or directly formalizable local claim.
- **C**: Better validated first by code or experiment (e.g. finite enumeration, exact arithmetic, local composition checks, small counterexample search, table/closure verification, concrete examples, or exact small-model consistency checks); use C when a step can be made sharper by real checking before formalization.
- **D**: Too vague, too global, or too risky to push directly; represent it explicitly, but prefer to split it into smaller B/C pieces whenever possible instead of leaving a large opaque D.

### Fairness rule for grades

Do NOT overuse A/B. In particular:

- If a claim is best checked first by exact computation, finite enumeration, symbolic algebra, composition-table verification, parity/case splitting, or small counterexample search, grade it **C** even if the mathematics seems plausible.
- Use **B** for human-common-sense structural bridges that still need clarification or bridge lemmas, but are not primarily about immediate finite/experimental checking.
- Use **D** only when the claim is genuinely too vague/global/ambiguous to act on directly.
- If a risky D-claim contains a testable core, expose that testable core as a separate **C** proposition rather than collapsing everything into one D.
- Prefer a mix of A/B/C/D that reflects real downstream actionability; do not default to B for every nontrivial step.

## Role semantics

- `seed`: starting premise / prior result
- `target`: final theorem goal
- `derived`: ordinary derived intermediate result
- `bridge`: explicit bridge proposition between high-level route and formalizable subgoals
- `experiment_support`: proposition mainly used as an empirical support target
- `risk`: a dangerous or currently ambiguous proposition / shortcut

## Output format

You MUST return exactly one JSON object:

```json
{
  "target_statement": "target theorem statement",
  "propositions": [
    {
      "id": "P1",
      "statement": "explicit proposition",
      "role": "seed",
      "grade": "A",
      "depends_on": [],
      "notes": "optional concise note",
      "formalization_notes": "optional note",
      "experiment_notes": "optional note"
    }
  ],
  "chain": [
    {
      "id": "S1",
      "statement": "full reasoning step",
      "uses": ["P1", "P2"],
      "concludes": ["P3"],
      "grade": "B",
      "notes": "optional concise note"
    }
  ],
  "summary": "1-3 sentence bridge-layer summary"
}
```

## Requirements

- Include an explicit `target` proposition.
- Do not leave logical jumps implicit; introduce bridge propositions when needed.
- Every `uses` and `concludes` reference must point to proposition ids in the same output.
- The chain must actually reach the target proposition.
- If the route introduces a new object, invariant, reduction, obstruction, or auxiliary hypothesis, represent it explicitly as a proposition instead of hiding it inside prose.
- When a step would benefit from code or numerical experiment before formalization, grade it **C** (do not default to B or D for such steps).
- When the route contains a risky or ambiguous shortcut, represent it explicitly and grade it **D**.
- If a D proposition can be decomposed into a locally checkable claim plus a remaining conceptual gap, create both:
  - a **C** proposition for the locally checkable claim
  - a **D** proposition for the unresolved conceptual gap
- Prefer faithful decomposition over optimistic grading.
- Do not let the majority of propositions be mere restatements of provided seed facts if the route proposed new mechanisms.
