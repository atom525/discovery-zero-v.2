---
name: specialize_generalize
description: Generate constrained variants and extract reusable patterns.
---

# Specialize Generalize Skill

For specialization requests, return JSON:

```json
{
  "specializations": [
    {"statement": "...", "rationale": "...", "formal_statement": null}
  ]
}
```

For pattern mining requests, return JSON:

```json
{
  "patterns": [
    {"statement": "...", "support": "..."}
  ]
}
```
