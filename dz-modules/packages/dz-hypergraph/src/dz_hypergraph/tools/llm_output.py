"""
Multi-layer LLM output reliability system for Discovery Zero.

Four complementary layers that address the fundamental problem of monolithic
complex JSON generation from LLMs:

  Layer 1 — Schema-Constrained Decoding
    Each skill has a precise JSON Schema.  When the API supports
    json_schema response_format, syntax is guaranteed at the protocol level.

  Layer 2 — Incremental Stepwise Generation (IncrementalSkillRunner)
    Complex outputs are broken into multiple short, focused LLM calls.
    Each call produces a simple JSON fragment.  Intermediate outputs are
    validated before the next step proceeds.

  Layer 3 — Verifier-in-the-Loop Self-Correction (SelfCorrectingRunner)
    If a skill output fails schema validation, the error message is fed back
    to the LLM and the call is retried (up to max_corrections times).

  Layer 4 — Tactic-by-Tactic Lean Interaction
    Lean proofs are generated one tactic at a time (see lean_tactic.py).

This architecture is inspired by:
  - Pantograph / LLMLean: tactic-by-tactic proof state interaction
  - Goedel-Prover-V2: verifier-guided self-correction
  - DeepSeek-Prover-V2: subgoal decomposition into atomic steps
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from dz_hypergraph.tools.llm import (
    LLMError,
    chat_completion,
    load_skill_prompt,
    run_skill,
    extract_text_content,
    extract_json_block,
    get_llm_config,
)
from dz_hypergraph.tools.llm_transport import LLMTransport
from dz_hypergraph.tools.llm_budget import TokenBudget

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Layer 1: JSON Schemas for each skill                                #
# ------------------------------------------------------------------ #

PLAUSIBLE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "premises": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "statement": {"type": "string"},
                },
                "required": ["statement"],
                "additionalProperties": False,
            },
        },
        "steps": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "conclusion": {
            "type": "object",
            "properties": {
                "statement": {"type": "string"},
                "formal_statement": {"type": "string"},
            },
            "required": ["statement"],
            "additionalProperties": False,
        },
        "module": {
            "type": "string",
            "enum": ["plausible", "experiment", "lean"],
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "domain": {"type": "string"},
    },
    "required": ["premises", "steps", "conclusion", "module"],
    "additionalProperties": False,
}

BRIDGE_PLAN_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "target_statement": {"type": "string"},
        "propositions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "statement": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": ["seed", "target", "derived", "bridge", "experiment_support", "risk"],
                    },
                    "grade": {"type": "string", "enum": ["A", "B", "C", "D"]},
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "notes": {"type": "string"},
                    "formalization_notes": {"type": "string"},
                    "experiment_notes": {"type": "string"},
                },
                "required": ["id", "statement", "role", "grade"],
                "additionalProperties": False,
            },
        },
        "chain": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "statement": {"type": "string"},
                    "uses": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "concludes": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "grade": {"type": "string", "enum": ["A", "B", "C", "D"]},
                    "notes": {"type": "string"},
                },
                "required": ["id", "statement", "uses", "concludes", "grade"],
                "additionalProperties": False,
            },
            "minItems": 1,
        },
        "summary": {"type": "string"},
    },
    "required": ["target_statement", "propositions", "chain"],
    "additionalProperties": False,
}

EXPERIMENT_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "code": {"type": "string"},
        "expected_output": {"type": "string"},
        "module": {"type": "string", "enum": ["experiment"]},
        "domain": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["hypothesis", "code"],
    "additionalProperties": False,
}

JUDGE_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reasoning": {"type": "string"},
        "concerns": {
            "type": "array",
            "items": {"type": "string"},
        },
        "suggestion": {"type": "string"},
        "verdict": {"type": "string", "enum": ["accept", "replan", "reject"]},
    },
    "required": ["confidence", "reasoning"],
    "additionalProperties": False,
}

LEAN_SKELETON_OUTPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step_type": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["content"],
                "additionalProperties": False,
            },
        },
        "module": {"type": "string", "enum": ["lean"]},
        "lean_code": {"type": "string"},
    },
    "required": ["steps"],
    "additionalProperties": False,
}

DECOMPOSITION_PLAN_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "subgoals": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "statement": {"type": "string"},
                    "formal_statement": {"type": "string"},
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["trivial", "easy", "medium", "hard", "open"],
                    },
                },
                "required": ["id", "statement"],
                "additionalProperties": False,
            },
        },
        "strategy": {"type": "string"},
    },
    "required": ["subgoals"],
    "additionalProperties": False,
}

SKILL_SCHEMAS: Dict[str, Dict[str, Any]] = {
    "plausible.skill.md": PLAUSIBLE_OUTPUT_SCHEMA,
    "bridge_plan.skill.md": BRIDGE_PLAN_OUTPUT_SCHEMA,
    "experiment.skill.md": EXPERIMENT_OUTPUT_SCHEMA,
    "judge.skill.md": JUDGE_OUTPUT_SCHEMA,
    "lean_skeleton_compiler.skill.md": LEAN_SKELETON_OUTPUT_SCHEMA,
    "decompose.skill.md": DECOMPOSITION_PLAN_SCHEMA,
}


def get_skill_schema(skill_filename: str) -> Optional[Dict[str, Any]]:
    """Return the JSON Schema for a skill if one is registered."""
    return SKILL_SCHEMAS.get(skill_filename)


# ------------------------------------------------------------------ #
# Validation helpers                                                   #
# ------------------------------------------------------------------ #

@dataclass
class ValidationResult:
    valid: bool
    error_message: str = ""
    field_path: str = ""


def validate_against_schema(
    data: Any,
    schema: Dict[str, Any],
    *,
    skill_filename: str = "",
) -> ValidationResult:
    """
    Validate parsed JSON against a schema.

    Uses jsonschema if available; falls back to a minimal structural check.
    """
    try:
        import jsonschema
        try:
            jsonschema.validate(data, schema)
            return ValidationResult(valid=True)
        except jsonschema.ValidationError as exc:
            return ValidationResult(
                valid=False,
                error_message=str(exc.message),
                field_path=".".join(str(p) for p in exc.path),
            )
    except ImportError:
        pass

    # Minimal fallback: check required fields at top level
    if not isinstance(data, dict):
        return ValidationResult(valid=False, error_message="Output is not a JSON object", field_path="")
    required = schema.get("required", [])
    for key in required:
        if key not in data:
            return ValidationResult(
                valid=False,
                error_message=f"Missing required field: '{key}'",
                field_path=key,
            )
    return ValidationResult(valid=True)


# ------------------------------------------------------------------ #
# Layer 3: Verifier-in-the-Loop Self-Correction                       #
# ------------------------------------------------------------------ #

class SelfCorrectingRunner:
    """
    Generate → Validate → If invalid, feed error back → Regenerate.

    Inspired by Goedel-Prover-V2's verifier-guided self-correction.
    For JSON outputs: validation errors are the correction signal.
    For Lean: compiler errors are the correction signal (see lean_tactic.py).
    """

    def __init__(
        self,
        max_corrections: int = 3,
        transport: Optional[LLMTransport] = None,
        budget: Optional[TokenBudget] = None,
    ) -> None:
        self.max_corrections = max_corrections
        self.transport = transport
        self.budget = budget

    def run_with_correction(
        self,
        skill_filename: str,
        task_input: str,
        validator: Callable[[Any], ValidationResult],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        timeout: int = 300,
        node_id: str = "",
    ) -> Tuple[str, Any]:
        """
        Run a skill with automatic self-correction.

        Returns (raw_text, parsed_json) of the first validated output.
        Raises LLMError if all attempts (1 + max_corrections) fail.
        """
        prompt = task_input
        schema = get_skill_schema(skill_filename)

        for attempt in range(self.max_corrections + 1):
            try:
                result = run_skill(
                    skill_filename,
                    prompt,
                    model=model,
                    temperature=temperature,
                    timeout=timeout,
                    schema=schema,
                    transport=self.transport,
                    budget=self.budget,
                    node_id=node_id,
                )
                assert isinstance(result, tuple)
                raw, parsed = result
            except LLMError as exc:
                if attempt >= self.max_corrections:
                    raise
                parsed = None
                raw = ""
                error_msg = str(exc)
                field_path = ""
            else:
                if parsed is None:
                    if attempt >= self.max_corrections:
                        raise LLMError(f"Skill {skill_filename} returned None JSON after {attempt+1} attempts")
                    error_msg = "Output was None or could not be parsed as JSON"
                    field_path = ""
                else:
                    vr = validator(parsed)
                    if vr.valid:
                        return raw, parsed
                    error_msg = vr.error_message
                    field_path = vr.field_path

            if attempt < self.max_corrections:
                correction_context = (
                    f"\n\nYour previous output had a validation error:\n"
                    f"```\n{error_msg}\n```\n"
                )
                if field_path:
                    correction_context += f"Problematic field path: `{field_path}`\n"
                correction_context += "Fix ONLY this issue and regenerate the complete JSON."
                prompt = task_input + correction_context
                logger.info(
                    "Self-correction attempt %d/%d for %s: %s",
                    attempt + 1,
                    self.max_corrections,
                    skill_filename,
                    error_msg[:120],
                )

        raise LLMError(
            f"Skill {skill_filename} failed validation after {self.max_corrections + 1} attempts. "
            f"Last error: {error_msg}"
        )


# ------------------------------------------------------------------ #
# Layer 2: Incremental Stepwise Generation                            #
# ------------------------------------------------------------------ #

class _StepCall:
    """Helper for a single incremental LLM call producing a simple JSON."""

    def __init__(
        self,
        system_prompt: str,
        transport: Optional[LLMTransport],
        budget: Optional[TokenBudget],
    ) -> None:
        self._system = system_prompt
        self._transport = transport
        self._budget = budget

    def call(
        self,
        user_prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        timeout: int = 120,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Any:
        response = chat_completion(
            messages=[
                {"role": "system", "content": self._system},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=temperature,
            timeout=timeout,
            response_format=response_format or {"type": "json_object"},
            n=1,
            transport=self._transport,
            budget=self._budget,
        )
        assert isinstance(response, dict)
        raw = extract_text_content(response)
        return extract_json_block(raw)


class IncrementalSkillRunner:
    """
    Break a complex skill into multiple short, focused LLM calls.

    Instead of asking the LLM to produce a complete nested JSON in one shot
    (which causes truncation and parse failures on hard problems), we make
    multiple small calls that each produce a simple 1–2 level JSON.

    This mirrors the approach of Pantograph (tactic-by-tactic) but applied
    to planning and bridge generation, not just Lean proofs.
    """

    def __init__(
        self,
        transport: Optional[LLMTransport] = None,
        budget: Optional[TokenBudget] = None,
    ) -> None:
        self.transport = transport
        self.budget = budget

    # ---------------------------------------------------------- #
    # Bridge Plan — 3-step incremental generation                #
    # ---------------------------------------------------------- #

    def run_bridge_plan_incremental(
        self,
        target_statement: str,
        graph_context: str,
        *,
        model: Optional[str] = None,
        timeout: int = 120,
    ) -> Dict[str, Any]:
        """
        Generate a bridge plan in three focused steps:

        Step 1 → List proposition statements (simple string array)
        Step 2 → Assign grade + role + dependencies per proposition
        Step 3 → Write the reasoning chain connecting them
        """
        base_ctx = (
            f"Target statement: {target_statement}\n\n"
            f"Graph context:\n{graph_context}\n\n"
        )

        helper = _StepCall(
            system_prompt=(
                "You are a mathematical bridge planner. "
                "Answer with ONLY the JSON structure requested. "
                "Be concise and precise."
            ),
            transport=self.transport,
            budget=self.budget,
        )

        # Step 1: proposition statements
        step1_prompt = (
            base_ctx
            + "Step 1: List the 3–7 key intermediate propositions needed to bridge from "
            "the given knowledge to the target. Output a JSON object with key "
            '"propositions" containing an array of objects with "id" (p1, p2, ...) '
            "and \"statement\" (string). Do NOT include grades or dependencies yet."
        )
        props_data = helper.call(step1_prompt, model=model, timeout=timeout)
        raw_props: List[Dict[str, Any]] = props_data.get("propositions", [])
        if not raw_props or not isinstance(raw_props, list):
            raise LLMError("Bridge plan step 1: propositions list is empty or invalid")

        # Step 2: grades, roles, dependencies
        props_summary = json.dumps(
            [{"id": p.get("id", f"p{i}"), "statement": p.get("statement", "")}
             for i, p in enumerate(raw_props)],
            ensure_ascii=False,
        )
        step2_prompt = (
            base_ctx
            + f"Propositions identified:\n{props_summary}\n\n"
            "Step 2: For each proposition, assign:\n"
            "  - grade: A (formal/algebraic), B (analytical), C (computational/empirical), D (conjectural)\n"
            "  - role: one of 'seed', 'target', 'derived', 'bridge', 'experiment_support', 'risk'\n"
            "  - depends_on: array of proposition IDs this depends on (empty = depends only on seeds)\n"
            "  - notes / formalization_notes / experiment_notes (optional short strings)\n"
            'Output JSON: {"graded_propositions": [{"id":..., "statement":..., "grade":..., "role":..., "depends_on":[...], "notes":..., "formalization_notes":..., "experiment_notes":...}, ...]}'
        )
        graded_data = helper.call(step2_prompt, model=model, timeout=timeout)
        graded_props: List[Dict[str, Any]] = graded_data.get("graded_propositions", [])

        # Merge grades into props if graded list is shorter or ordering differs
        grade_map: Dict[str, Dict[str, Any]] = {
            p.get("id", ""): p for p in graded_props if p.get("id")
        }
        merged_props: List[Dict[str, Any]] = []
        for raw_p in raw_props:
            pid = raw_p.get("id", "")
            if pid in grade_map:
                merged_props.append({**raw_p, **grade_map[pid]})
            else:
                merged_props.append({**raw_p, "grade": "D", "role": "bridge", "depends_on": []})

        # Step 3: reasoning chain
        merged_summary = json.dumps(
            [{"id": p.get("id"), "statement": p.get("statement"), "grade": p.get("grade")}
             for p in merged_props],
            ensure_ascii=False,
        )
        step3_prompt = (
            base_ctx
            + f"Final propositions:\n{merged_summary}\n\n"
            "Step 3: Write a structured reasoning chain connecting these propositions "
            "to the target statement. "
            "Output JSON with:\n"
            '  - "chain": array of objects {id, statement, uses, concludes, grade}\n'
            '  - "summary": short 1-3 sentence bridge summary\n'
            '  - "target_statement": exact target statement\n'
        )
        chain_data = helper.call(step3_prompt, model=model, timeout=timeout)
        chain: List[Dict[str, Any]] = chain_data.get("chain", [])
        summary: str = chain_data.get("summary", "")

        return {
            "target_statement": chain_data.get("target_statement", target_statement),
            "propositions": merged_props,
            "chain": chain,
            "summary": summary,
        }

    # ---------------------------------------------------------- #
    # Plausible — 3-step incremental generation                  #
    # ---------------------------------------------------------- #

    def run_plausible_incremental(
        self,
        target_statement: str,
        graph_context: str,
        domain: str = "",
        *,
        model: Optional[str] = None,
        timeout: int = 120,
    ) -> Dict[str, Any]:
        """
        Generate a plausible reasoning output in three focused steps:

        Step 1 → Generate reasoning steps (list of strings)
        Step 2 → Identify premises used (statement references)
        Step 3 → State the conclusion and confidence
        """
        base_ctx = (
            f"Target statement: {target_statement}\n"
            f"Domain: {domain or 'mathematics'}\n\n"
            f"Graph context:\n{graph_context}\n\n"
        )

        helper = _StepCall(
            system_prompt=(
                "You are a mathematical reasoning assistant. "
                "Answer with ONLY the JSON structure requested."
            ),
            transport=self.transport,
            budget=self.budget,
        )

        # Step 1: reasoning steps
        step1_prompt = (
            base_ctx
            + "Step 1: Provide 3–8 reasoning steps that make the target statement plausible. "
            "Each step is a single, concrete mathematical sentence. "
            'Output JSON: {"steps": ["step1...", "step2...", ...]}'
        )
        steps_data = helper.call(step1_prompt, model=model, timeout=timeout)
        steps: List[str] = steps_data.get("steps", [])
        if not steps:
            raise LLMError("Plausible step 1: reasoning steps list is empty")

        # Step 2: premises
        steps_str = "\n".join(f"- {s}" for s in steps)
        step2_prompt = (
            base_ctx
            + f"Reasoning steps:\n{steps_str}\n\n"
            "Step 2: Identify which known facts/premises from the graph context "
            "are used in the reasoning. "
            'Output JSON: {"premises": [{"statement": "...", "id": "optional-node-id"}, ...]}'
        )
        prems_data = helper.call(step2_prompt, model=model, timeout=timeout)
        premises: List[Dict[str, Any]] = prems_data.get("premises", [])

        # Step 3: conclusion + confidence + module
        step3_prompt = (
            base_ctx
            + f"Reasoning steps:\n{steps_str}\n\n"
            "Step 3: State the conclusion and assess confidence. "
            "Also choose the best next module: 'plausible' (needs more reasoning), "
            "'experiment' (computational check helpful), or 'lean' (ready for formal proof). "
            "Output JSON: {\"conclusion\": {\"statement\": \"...\"}, "
            "\"confidence\": 0.0-1.0, \"module\": \"plausible|experiment|lean\"}"
        )
        concl_data = helper.call(step3_prompt, model=model, timeout=timeout)

        return {
            "premises": premises,
            "steps": steps,
            "conclusion": concl_data.get(
                "conclusion",
                {"statement": target_statement}
            ),
            "confidence": float(concl_data.get("confidence", 0.5)),
            "module": concl_data.get("module", "plausible"),
            "domain": domain,
        }

    # ---------------------------------------------------------- #
    # Generic helper                                             #
    # ---------------------------------------------------------- #

    def run_incremental(
        self,
        steps: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        timeout: int = 120,
    ) -> Dict[str, Any]:
        """
        Run an arbitrary multi-step generation defined as a list of step dicts.

        Each step dict has:
          - "prompt": str (user-side prompt)
          - "system": str (system context, optional, defaults to generic)
          - "key": str (key to extract from response and pass forward)
          - "accumulate": bool (if True, merge into running context)

        Returns a dict merging all extracted values.
        """
        context: Dict[str, Any] = {}
        for i, step in enumerate(steps):
            system = step.get("system", "You are a helpful assistant. Answer in JSON only.")
            prompt = step["prompt"]

            # Substitute {key} references with values from context
            for k, v in context.items():
                if isinstance(v, (str, int, float)):
                    prompt = prompt.replace("{" + k + "}", str(v))
                elif isinstance(v, (list, dict)):
                    prompt = prompt.replace("{" + k + "}", json.dumps(v, ensure_ascii=False))

            helper = _StepCall(system_prompt=system, transport=self.transport, budget=self.budget)
            result = helper.call(prompt, model=model, temperature=temperature, timeout=timeout)

            key = step.get("key", f"step_{i}")
            if isinstance(result, dict):
                if step.get("accumulate", True):
                    context.update(result)
                else:
                    context[key] = result
            else:
                context[key] = result

        return context
