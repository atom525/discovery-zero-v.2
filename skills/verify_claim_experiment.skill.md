---
name: verify_claim_experiment
description: Generate Python code to directly verify one quantitative claim.
---

# Verify Claim Experiment

You are testing one specific mathematical claim computationally.

## Input

- Claim
- Optional context and parameter hints

## Rules

1. Produce executable Python only.
2. Use exact arithmetic (`fractions.Fraction` or `sympy`) whenever possible.
3. Compute the claimed quantity directly.
4. Print exactly one final JSON line containing:
   - `passed` (bool)
   - `trials` (int)
   - `max_error` (number|null)
   - `counterexample` (object|null)
   - `summary` (string)
   - `computed_value` (string|number|null)
   - `claimed_value` (string|number|null)

## Output

Return ONLY Python code. No markdown fences. No explanation.
