---
name: fill_experiment_template
description: Fill structured slots for a selected experiment code template.
---

# Fill Experiment Template

You are generating slot values for a production experiment template.

## Input

You will receive:
- claim statement
- context
- available template names and descriptions

## Requirements

1. Pick exactly one template name.
2. Provide all required placeholders for that template.
3. Ensure generated code can run directly.
4. Prefer exact arithmetic when feasible.
5. Keep loops bounded and deterministic.

## Output

Return ONLY JSON:

```json
{
  "template": "measure_compute",
  "slots": {
    "USER_IMPORTS": "",
    "HELPERS": "",
    "MEASURE_SETUP": "...",
    "COMPUTED_EXPR": "...",
    "CLAIMED_EXPR": "..."
  }
}
```

No markdown. No prose.
