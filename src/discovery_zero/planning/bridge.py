"""
Structured bridge-layer objects for Discovery Zero.

This module turns a natural-language proof route into a validated intermediate
representation: explicit subpropositions, a full reasoning chain, and A/B/C/D
formalization grades. It is intentionally strict: invalid or dangling bridge
plans must fail validation rather than being silently accepted.
"""

from __future__ import annotations

from collections import deque
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from discovery_zero.graph.models import Module


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
    """Remove from `uses` any prop ids that also appear in `concludes`.

    LLMs occasionally place the same proposition id on both sides of a
    reasoning step.  The Pydantic validator rejects such steps, causing
    silent bridge-plan save failures.  This pre-pass fixes the data so
    validation can succeed.
    """
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
    except Exception as exc:  # pragma: no cover - pydantic already structures details
        raise BridgeValidationError(str(exc)) from exc


def select_bridge_focus_proposition(plan: BridgePlan) -> BridgeProposition:
    """
    Select a bridge proposition to target for local decomposition.

    Priority:
    1. `bridge` role with grade D
    2. `bridge` role with grade B
    3. any grade D proposition
    4. target proposition
    """
    propositions = plan.propositions
    prop_map = {item.id: item for item in propositions}

    def non_seed_dep_count(item: BridgeProposition) -> int:
        return sum(
            1
            for dep in item.depends_on
            if dep in prop_map and prop_map[dep].role != "seed"
        )

    for predicate in (
        lambda p: p.role == "bridge" and p.grade == "D",
        lambda p: p.role == "bridge" and p.grade == "B",
        lambda p: p.grade == "D",
        lambda p: p.role == "target",
    ):
        matches = [item for item in propositions if predicate(item)]
        if matches:
            matches.sort(
                key=lambda item: (
                    non_seed_dep_count(item),
                    len(item.depends_on),
                    item.id,
                ),
                reverse=True,
            )
            return matches[0]
    raise BridgeValidationError("Bridge plan does not contain any selectable focus proposition.")


def derive_subplan(plan: BridgePlan, focus_proposition_id: str) -> BridgePlan:
    """Project a bridge plan onto the ancestor cone of one focus proposition."""
    prop_map = {item.id: item for item in plan.propositions}
    if focus_proposition_id not in prop_map:
        raise BridgeValidationError(
            f"Focus proposition '{focus_proposition_id}' is not present in bridge plan."
        )

    required: set[str] = set()
    queue = deque([focus_proposition_id])
    while queue:
        current = queue.popleft()
        if current in required:
            continue
        required.add(current)
        for parent in prop_map[current].depends_on:
            queue.append(parent)

    new_props: list[BridgeProposition] = []
    for item in plan.propositions:
        if item.id not in required:
            continue
        role = "target" if item.id == focus_proposition_id else item.role
        new_props.append(
            BridgeProposition(
                id=item.id,
                statement=item.statement,
                role=role,
                grade=item.grade,
                depends_on=[dep for dep in item.depends_on if dep in required],
                notes=item.notes,
                formalization_notes=item.formalization_notes,
                experiment_notes=item.experiment_notes,
            )
        )

    new_steps: list[BridgeReasoningStep] = []
    for step in plan.chain:
        uses = [item for item in step.uses if item in required]
        concludes = [item for item in step.concludes if item in required]
        if not uses and not concludes:
            continue
        new_steps.append(
            BridgeReasoningStep(
                id=step.id,
                statement=step.statement,
                uses=uses,
                concludes=concludes,
                grade=step.grade,
                notes=step.notes,
            )
        )

    return BridgePlan(
        target_statement=prop_map[focus_proposition_id].statement,
        propositions=new_props,
        chain=new_steps,
        summary=plan.summary,
    )


# Bridge node prior by proposition grade.
# Grade reflects expected verification difficulty / initial confidence in the
# proposition itself (node-level prior), not logical implication strength.
BRIDGE_GRADE_PRIOR: dict[str, float] = {
    "A": 0.92,
    "B": 0.70,
    "C": 0.50,
    "D": 0.30,
}

# Bridge dependency edges encode "if premises then conclusion" relations.
# Keep a strong, uniform implication confidence so BP can propagate along the
# planned chain without being bottlenecked by proposition grade labels.
BRIDGE_EDGE_CONFIDENCE: float = 0.85


def materialize_bridge_nodes(
    graph,
    plan: BridgePlan,
    *,
    default_domain: str | None = None,
    target_node_id: str | None = None,
) -> dict[str, str]:
    """Materialize bridge propositions as graph nodes and create dependency edges.

    Pass 1 creates or matches graph nodes for every proposition.  The proposition
    with ``role="target"`` is mapped to ``target_node_id`` (when supplied) so
    that the MCTS-tracked goal and the bridge target share the same node — this
    is essential for BP to propagate belief from verified bridge steps all the
    way to the real MCTS target.

    Pass 2 translates each ``depends_on`` relationship into a hyperedge, making
    the bridge plan's dependency structure visible to belief propagation.

    Args:
        graph: The HyperGraph to materialise into.
        plan: The validated BridgePlan.
        default_domain: Domain label for newly created nodes.
        target_node_id: The MCTS target node ID.  If provided, the bridge
            proposition whose role is "target" will be mapped to this node
            rather than creating a duplicate node with a different statement.

    Returns:
        Mapping from bridge proposition ID (e.g. "P3") to graph node ID.
    """
    def _grade_prior(grade: BridgeGrade) -> float:
        return BRIDGE_GRADE_PRIOR.get(grade, 0.5)

    mapping: dict[str, str] = {}
    risk_prop_ids = {
        item.id for item in plan.propositions if item.role == "risk"
    }

    # ------------------------------------------------------------------
    # Pass 1: create / match graph nodes
    # ------------------------------------------------------------------
    for item in plan.propositions:
        # TARGET proposition: always use the real MCTS target node when given.
        if item.role == "target" and target_node_id and target_node_id in graph.nodes:
            mapping[item.id] = target_node_id
            continue
        # All other propositions: try exact/canonical text match first.
        matches = graph.find_node_ids_by_statement(item.statement)
        if matches:
            matched_id = matches[0]
            mapping[item.id] = matched_id
            matched_node = graph.nodes[matched_id]
            if item.role == "risk" and not matched_node.is_locked():
                matched_node.provenance = "bridge_risk"
            elif (
                item.role != "seed"
                and item.role != "target"
                and not matched_node.is_locked()
                and matched_node.provenance is None
            ):
                matched_node.provenance = "bridge"
            if item.role == "seed":
                matched_node.state = "proven"
                matched_node.prior = 1.0
                matched_node.belief = 1.0
            elif not matched_node.is_locked():
                prior = _grade_prior(item.grade)
                matched_node.prior = max(float(matched_node.prior), prior)
                matched_node.belief = max(float(matched_node.belief), prior)
            continue
        prior = 1.0 if item.role == "seed" else _grade_prior(item.grade)
        node = graph.add_node(
            statement=item.statement,
            belief=prior,
            prior=prior,
            domain=default_domain,
            state="proven" if item.role == "seed" else "unverified",
            provenance=(
                "bridge_risk"
                if item.role == "risk"
                else ("bridge" if item.role not in ("seed", "target") else None)
            ),
        )
        mapping[item.id] = node.id

    # ------------------------------------------------------------------
    # Pass 2: create dependency hyperedges from depends_on relationships.
    # Seed propositions have no dependencies to create.
    # We skip creation when an identical edge already exists (idempotent).
    # ------------------------------------------------------------------
    for item in plan.propositions:
        if item.role == "seed" or not item.depends_on:
            continue
        conclusion_id = mapping.get(item.id)
        if conclusion_id is None:
            continue
        mapped_pairs = [(dep, mapping[dep]) for dep in item.depends_on if dep in mapping]
        premise_ids = [pid for _, pid in mapped_pairs]
        # Risk propositions are diagnostic blockers. They must not become
        # positive premises for non-risk conclusions.
        if item.role != "risk" and risk_prop_ids:
            filtered_premises = [
                pid for dep, pid in mapped_pairs if dep not in risk_prop_ids
            ]
            # Safety fallback: keep all premises if filtering would isolate
            # the conclusion completely.
            if filtered_premises:
                premise_ids = filtered_premises
        if not premise_ids:
            continue
        # Idempotency: skip if an edge with the same premise set and
        # conclusion already exists (e.g. from a previous iteration).
        premise_set = set(premise_ids)
        already_exists = any(
            set(e.premise_ids) == premise_set and e.conclusion_id == conclusion_id
            for e in graph.edges.values()
        )
        if already_exists:
            continue
        graph.add_hyperedge(
            premise_ids=premise_ids,
            conclusion_id=conclusion_id,
            module=Module.PLAUSIBLE,
            steps=[f"Bridge dependency: {', '.join(item.depends_on)} → {item.id}"],
            confidence=BRIDGE_EDGE_CONFIDENCE,
        )

    return mapping


def preferred_sibling_proposition_ids(plan: BridgePlan, focus_proposition_id: str) -> list[str]:
    """
    Return immediate non-seed dependencies to prefer as sibling local goals.

    These are the best candidates for parallel local `have` goals in a Lean
    skeleton because they feed directly into the focus proposition.
    """
    prop_map = {item.id: item for item in plan.propositions}
    if focus_proposition_id not in prop_map:
        raise BridgeValidationError(
            f"Focus proposition '{focus_proposition_id}' is not present in bridge plan."
        )
    focus = prop_map[focus_proposition_id]
    siblings = [
        dep for dep in focus.depends_on
        if dep in prop_map and prop_map[dep].role != "seed"
    ]
    return siblings


def derive_sibling_goal_subplan(plan: BridgePlan, focus_proposition_id: str) -> tuple[BridgePlan, str]:
    """
    Build a synthetic subplan whose target is to establish multiple sibling bridge propositions.

    This is used when a focus proposition has several direct non-seed dependencies;
    the synthetic target is a conjunction-style package over those siblings so Lean
    can expose them as independent local goals.
    """
    prop_map = {item.id: item for item in plan.propositions}
    if focus_proposition_id not in prop_map:
        raise BridgeValidationError(
            f"Focus proposition '{focus_proposition_id}' is not present in bridge plan."
        )
    sibling_ids = preferred_sibling_proposition_ids(plan, focus_proposition_id)
    if len(sibling_ids) < 2:
        raise BridgeValidationError(
            f"Focus proposition '{focus_proposition_id}' does not have enough sibling dependencies."
        )

    required: set[str] = set()
    queue = deque(sibling_ids)
    while queue:
        current = queue.popleft()
        if current in required:
            continue
        required.add(current)
        for parent in prop_map[current].depends_on:
            queue.append(parent)

    new_props: list[BridgeProposition] = []
    for item in plan.propositions:
        if item.id not in required:
            continue
        new_props.append(
            BridgeProposition(
                id=item.id,
                statement=item.statement,
                role=item.role,
                grade=item.grade,
                depends_on=[dep for dep in item.depends_on if dep in required],
                notes=item.notes,
                formalization_notes=item.formalization_notes,
                experiment_notes=item.experiment_notes,
            )
        )

    synth_id = f"{focus_proposition_id}__siblings"
    synth_statement = (
        "Simultaneously establish the following sibling bridge propositions needed for "
        f"[{prop_map[focus_proposition_id].statement}]: "
        + "; ".join(prop_map[item].statement for item in sibling_ids)
    )
    new_props.append(
        BridgeProposition(
            id=synth_id,
            statement=synth_statement,
            role="target",
            grade="B",
            depends_on=sibling_ids,
            notes="Synthetic conjunction-style decomposition target.",
            formalization_notes=(
                "Prefer a theorem goal that is an explicit conjunction/product of the sibling propositions "
                "so Lean exposes them as independent local goals."
            ),
        )
    )

    new_steps: list[BridgeReasoningStep] = []
    for step in plan.chain:
        uses = [item for item in step.uses if item in required]
        concludes = [item for item in step.concludes if item in required]
        if not uses and not concludes:
            continue
        new_steps.append(
            BridgeReasoningStep(
                id=step.id,
                statement=step.statement,
                uses=uses,
                concludes=concludes,
                grade=step.grade,
                notes=step.notes,
            )
        )

    new_steps.append(
        BridgeReasoningStep(
            id=f"{focus_proposition_id}__siblings_final",
            statement="Package the sibling bridge propositions into one explicit conjunction-style target.",
            uses=sibling_ids,
            concludes=[synth_id],
            grade="B",
            notes="Synthetic final packaging step for multi-goal decomposition.",
        )
    )

    return (
        BridgePlan(
            target_statement=synth_statement,
            propositions=new_props,
            chain=new_steps,
            summary=plan.summary,
        ),
        synth_id,
    )
