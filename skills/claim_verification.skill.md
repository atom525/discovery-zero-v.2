---
name: claim_verification
description: Plan computational checks for quantitative intermediate claims.
---

# Claim Verification Skill

Given a claim, return JSON only:

```json
{
  "strategy": [
    {"approach": "exact|enumeration|sampling", "notes": "..."}
  ],
  "preferred_template": "measure_compute",
  "validation_targets": [
    {"computed": "...", "claimed": "..."}
  ]
}
```

Focus on executable verification strategies, not narrative discussion.
