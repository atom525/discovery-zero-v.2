"""Structured bridge-layer data models for Discovery Zero."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

BridgeGrade = Literal["A", "B", "C", "D"]
BridgeRole = Literal[
    "seed",
    "target",
    "derived",
    "bridge",
    "experiment_support",
    "risk",
]


class BridgeValidationError(RuntimeError):
    """Raised when a bridge plan is malformed or inconsistent."""


class BridgeProposition(BaseModel):
    """One explicit proposition in the bridge layer."""

    id: str
    statement: str
    role: BridgeRole
    grade: BridgeGrade
    depends_on: list[str] = Field(default_factory=list)
    notes: Optional[str] = None
    formalization_notes: Optional[str] = None
    experiment_notes: Optional[str] = None

    @field_validator("id", "statement")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Bridge proposition fields must be non-empty.")
        return cleaned

    @field_validator("depends_on")
    @classmethod
    def dedupe_dependencies(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in value:
            cleaned = item.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            ordered.append(cleaned)
        return ordered


class BridgeReasoningStep(BaseModel):
    """One explicit step in the reasoning chain."""

    id: str
    statement: str
    uses: list[str]
    concludes: list[str]
    grade: BridgeGrade
    notes: Optional[str] = None

    @field_validator("id", "statement")
    @classmethod
    def non_empty_step_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Bridge reasoning step fields must be non-empty.")
        return cleaned

    @field_validator("uses", "concludes")
    @classmethod
    def normalize_reference_list(cls, value: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in value:
            cleaned = item.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            ordered.append(cleaned)
        return ordered

    @model_validator(mode="after")
    def at_least_one_side_present(self) -> "BridgeReasoningStep":
        if not self.uses and not self.concludes:
            raise ValueError(
                "Bridge reasoning step must reference at least one used or concluded proposition."
            )
        overlap = sorted(set(self.uses) & set(self.concludes))
        if overlap:
            raise ValueError(
                "Bridge reasoning step cannot reference the same proposition in both "
                f"`uses` and `concludes`: {overlap}"
            )
        return self


class BridgePlan(BaseModel):
    """Validated bridge-layer plan for one theorem target."""

    target_statement: str
    propositions: list[BridgeProposition]
    chain: list[BridgeReasoningStep]
    summary: Optional[str] = None

    @field_validator("target_statement")
    @classmethod
    def non_empty_target(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("target_statement must be non-empty.")
        return cleaned

    @model_validator(mode="after")
    def validate_consistency(self) -> "BridgePlan":
        if not self.propositions:
            raise ValueError("Bridge plan must contain at least one proposition.")
        if not self.chain:
            raise ValueError("Bridge plan must contain at least one reasoning step.")

        proposition_ids = [item.id for item in self.propositions]
        if len(proposition_ids) != len(set(proposition_ids)):
            raise ValueError("Bridge proposition ids must be unique.")
        step_ids = [item.id for item in self.chain]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Bridge reasoning step ids must be unique.")

        known = set(proposition_ids)
        targets = [
            item for item in self.propositions
            if item.role == "target" or item.statement == self.target_statement
        ]
        if not targets:
            raise ValueError("Bridge plan must contain an explicit target proposition.")
        if len(targets) > 1:
            unique_target_ids = {item.id for item in targets}
            if len(unique_target_ids) > 1:
                raise ValueError("Bridge plan must not contain multiple target propositions.")

        for proposition in self.propositions:
            if proposition.id in proposition.depends_on:
                raise ValueError(
                    f"Bridge proposition '{proposition.id}' cannot depend on itself."
                )
            missing = [item for item in proposition.depends_on if item not in known]
            if missing:
                raise ValueError(
                    f"Bridge proposition '{proposition.id}' depends on unknown ids: {missing}"
                )

        concluded_ids: set[str] = set()
        for step in self.chain:
            unknown_uses = [item for item in step.uses if item not in known]
            if unknown_uses:
                raise ValueError(
                    f"Bridge step '{step.id}' uses unknown proposition ids: {unknown_uses}"
                )
            unknown_concludes = [item for item in step.concludes if item not in known]
            if unknown_concludes:
                raise ValueError(
                    f"Bridge step '{step.id}' concludes unknown proposition ids: {unknown_concludes}"
                )
            concluded_ids.update(step.concludes)

        target_ids = {item.id for item in targets}
        if not (target_ids & concluded_ids):
            raise ValueError("Bridge reasoning chain never concludes the target proposition.")

        return self

    def metrics(self) -> dict[str, int]:
        """Small bridge-layer summary for benchmarking or logs."""
        grade_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        for item in self.propositions:
            grade_counts[item.grade] += 1
        return {
            "num_propositions": len(self.propositions),
            "num_chain_steps": len(self.chain),
            "grade_a_count": grade_counts["A"],
            "grade_b_count": grade_counts["B"],
            "grade_c_count": grade_counts["C"],
            "grade_d_count": grade_counts["D"],
        }


def _dedup_chain_overlap(payload: dict) -> dict:
    """Remove from `uses` any prop ids that also appear in `concludes`."""
    chain = payload.get("chain")
    if not isinstance(chain, list):
        return payload
    cleaned: list[dict] = []
    for step in chain:
        if not isinstance(step, dict):
            cleaned.append(step)
            continue
        uses = step.get("uses", [])
        concludes = step.get("concludes", [])
        if isinstance(uses, list) and isinstance(concludes, list):
            concludes_set = {str(c).strip() for c in concludes}
            uses = [u for u in uses if str(u).strip() not in concludes_set]
            step = {**step, "uses": uses}
        cleaned.append(step)
    return {**payload, "chain": cleaned}


def validate_bridge_plan_payload(payload: dict) -> BridgePlan:
    """Strictly parse and validate a bridge plan payload."""
    try:
        plan = BridgePlan.model_validate(_dedup_chain_overlap(payload))
        if not any(item.role == "target" for item in plan.propositions):
            non_seed_indexes = [
                idx for idx, item in enumerate(plan.propositions) if item.role != "seed"
            ]
            if non_seed_indexes:
                last_idx = non_seed_indexes[-1]
                plan.propositions[last_idx].role = "target"
                plan = BridgePlan.model_validate(plan.model_dump())
        return plan
    except Exception as exc:
        raise BridgeValidationError(str(exc)) from exc
