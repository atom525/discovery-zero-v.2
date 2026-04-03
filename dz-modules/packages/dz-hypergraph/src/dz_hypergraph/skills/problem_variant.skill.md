---
name: problem-variant
description: Generate easier or more concrete variants of a hard mathematical problem
---

# Problem Variant Skill

Given a difficult target statement, generate easier variants that preserve some of the structure while reducing difficulty.

## Output Format

Return JSON:

```json
{
  "variants": [
    {
      "variant_statement": "string",
      "variant_type": "parameter_reduction",
      "difficulty_estimate": 0.2,
      "rationale": "why this variant is easier and still informative"
    }
  ]
}
```

## Allowed `variant_type`

- `parameter_reduction`
- `finite_case`
- `weaker_conclusion`
- `stronger_hypothesis`
- `analogy`

## Guidelines

- Prefer variants that are plausibly solvable with existing tools.
- Keep statements mathematically meaningful, not trivialized nonsense.
- Include a short rationale for transfer value back to the original target.
- Order variants implicitly from easiest to harder.
