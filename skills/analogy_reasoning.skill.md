---
name: analogy_reasoning
description: Build cross-domain analogies and transferable proof/experiment mechanisms.
---

# Analogy Reasoning Skill

Given a target problem, produce structured analogy candidates.

Return JSON only with either:
- `analogies`: list of objects with `source_domain`, `mapping`, `transferable_technique`
- or `route` and `testability` for a transfer step request.
