---
name: experiment-evolution-mutate
description: Generate or mutate robust experiment programs for mathematical conjectures
---

# Experiment Evolution Mutation Skill

You are designing Python experiments for mathematical discovery. Your output will be executed in a restricted environment, so the code must be precise, deterministic enough for debugging, and safe.

## Input

You may receive:
- **Conjecture**
- **Context**
- **Strategy**
- **Parent code** (for mutation tasks)

## Requirements

Return JSON with exactly one top-level key:

```json
{
  "code": "full python program"
}
```

## Hard constraints

- Output valid Python only inside `code`.
- The program must print exactly one final JSON line with keys:
  - `passed` (bool)
  - `trials` (int)
  - `max_error` (number or null)
  - `counterexample` (object or null)
  - `summary` (string)
- Prefer exact arithmetic when possible.
- Include boundary cases in addition to random or enumerative search.
- Do not use network, file I/O, subprocesses, eval, exec, or unsafe imports.
- Keep dependencies within standard scientific Python (`math`, `itertools`, `fractions`, `numpy`, `sympy`, etc.).

## Strategy guidance

- `exhaustive_enumeration`: enumerate all small cases exactly.
- `boundary_testing`: target degenerate and extremal inputs.
- `random_search`: broad stochastic coverage.
- `algebraic_verification`: use symbolic or exact arithmetic.
- `statistical_analysis`: summarize broader trends with numerical evidence.

## Mutation guidance

If parent code is provided, improve one or more of:
- coverage
- exactness
- numerical stability
- counterexample reporting
- independent cross-checks

Do not explain your reasoning outside the JSON.
