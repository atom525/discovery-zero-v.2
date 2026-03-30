---
name: decompose_problem
description: Decompose a hard theorem into independent subproblems.
---

# Decompose Problem Skill

Return JSON only:

```json
{
  "subproblems": [
    {
      "statement": "...",
      "rationale": "...",
      "formal_statement": null
    }
  ]
}
```

Subproblems should be independent and collectively useful for solving the target.
