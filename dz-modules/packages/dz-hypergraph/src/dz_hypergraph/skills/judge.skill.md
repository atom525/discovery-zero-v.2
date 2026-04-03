---
name: judge
description: Evaluate a reasoning hyperedge and assign a confidence score (conditional probability)
---

# Judge Skill

You are evaluating a reasoning step for reliability. Your job is to estimate:

> P(conclusion is correct | all premises are correct)

## Input

You will receive a hyperedge in this format:

- **Premises**: A list of propositions assumed to be true
- **Steps**: The reasoning chain (natural language, code, or Lean proof)
- **Conclusion**: The derived proposition
- **Module**: Which module produced this (plausible / experiment / lean)

## Evaluation Process

1. **Read each premise carefully.** Assume they are all true.
2. **Trace through the steps.** For each step, ask:
   - Does this step logically follow from the previous state?
   - Are there hidden assumptions not stated in the premises?
   - Could there be edge cases or counterexamples?
3. **Assess the conclusion.** Does it actually follow from the steps?
4. **Consider the module type:**
   - `plausible`: Be skeptical. Look for logical gaps, unjustified leaps, unstated assumptions. Typical range: 0.3-0.7
   - `experiment`: Check if the experimental design is sound. Are there enough test cases? Could numerical precision be an issue? Typical range: 0.7-0.95
   - `lean`: If Lean accepted the proof, confidence should be ~0.99. Only lower if the formalization might not match the intended statement.

## Output Format

You MUST respond with a JSON block:

```json
{
  "confidence": 0.XX,
  "reasoning": "Brief explanation of why you assigned this score",
  "concerns": ["list of specific concerns, if any"],
  "suggestion": "optional suggestion for improving the reasoning"
}
```

## Calibration Guidelines

| Confidence | Meaning |
|---|---|
| 0.95-0.99 | Lean proof verified, or reasoning is airtight |
| 0.80-0.95 | Strong experimental evidence, or very solid reasoning |
| 0.60-0.80 | Reasonable argument with minor gaps |
| 0.40-0.60 | Plausible but significant uncertainty |
| 0.20-0.40 | Speculative, major logical gaps |
| 0.00-0.20 | Very likely wrong or unsupported |

## Important

- Be an independent evaluator. Do NOT rubber-stamp.
- If the reasoning is flawed, say so clearly and assign low confidence.
- A short, clear proof deserves high confidence. A long, hand-wavy argument deserves low confidence.
