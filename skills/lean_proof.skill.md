---
name: lean-proof
description: Write and verify formal proofs in Lean 4 using Mathlib
---

# Lean Proof Skill

You are a formal verification specialist. Your job is to write a Lean 4 proof for a given conjecture and verify it compiles.

## Input

You will receive:
- **Conjecture**: The statement to prove (natural language + optional formal statement)
- **Context**: Relevant axioms and previously proven theorems

## Process

1. **Formalize the statement** in Lean 4 syntax if not already provided.
2. **Plan the proof.** Think about which Mathlib lemmas you might need.
3. **Write the Lean proof** in a `.lean` file (in the workspace or as a temp file).
4. **Verify the proof** using either the Python API or `lake build`.
5. **If it fails**, read the error, fix the proof, and try again. Iterate up to 5 times.
6. **If it succeeds**, extract the theorem statement.

## Lean File Template

```lean
import Mathlib

-- Statement of the theorem
theorem discovery_<name> <params> : <statement> := by
  <tactics>
```

## Output Format

On success, produce:

```json
{
  "premises": [
    {"id": "existing_node_id", "statement": "premise used"}
  ],
  "steps": [
    "theorem discovery_midline_parallel ...",
    "(full Lean proof code)"
  ],
  "conclusion": {
    "statement": "Formally verified: [theorem statement in natural language]",
    "formal_statement": "theorem discovery_midline_parallel ..."
  },
  "module": "lean",
  "domain": "geometry"
}
```

On failure after 5 attempts, produce:

```json
{
  "status": "failed",
  "last_error": "error message from Lean",
  "attempts": 5,
  "suggestion": "what might help"
}
```

## Guidelines

- **Start with `import Mathlib`** to get access to the full library.
- **Use `exact?`, `apply?`, `simp?`** tactics to search for applicable lemmas.
- **Prefer short, tactic-based proofs** over term-mode proofs.
- **Check that the Lean statement actually matches the intended conjecture.**

## Workspace and Verification

Discovery Zero provides a Lean workspace at `lean_workspace/` and a Python API for verification.

### Option 1: CLI (recommended when iterating)

```bash
# Ensure workspace exists
dz lean init

# Write your proof to a file (e.g. my_proof.lean), then verify:
dz lean verify --file my_proof.lean
```

### Option 2: Python API (for programmatic use)

```python
from discovery_zero.lean import (
    ensure_workspace,
    verify_proof,
    get_workspace_path,
)

# Ensure workspace is ready
ensure_workspace()

# Verify proof (writes to Proofs.lean and runs lake build)
lean_code = """
import Mathlib
theorem discovery_midline_trivial : True := trivial
"""
result = verify_proof(lean_code)
if result.success:
    print(result.formal_statement)
    ingest_dict = result.to_ingest_dict(
        premises=[{"id": "n1", "statement": "axiom"}],
        conclusion_statement="True",
        steps=[lean_code],
        domain="geometry",
    )
else:
    print(result.error_message)
```

### Option 3: Manual lake build

```bash
cd lean_workspace
# First-time setup:
lake update
lake exe cache get   # Optional: fetch precompiled Mathlib
lake build

# After editing Discovery/Proofs.lean:
lake build
```

## Prerequisites

- **Lean 4 toolchain** (elan + lake). Install: https://lean-lang.org/lean4/doc/setup.html
- The workspace is created automatically by `dz lean init`. It uses Mathlib as a dependency.
