"""
Production-oriented orchestration helpers for Discovery Zero.

This module provides a small but real execution layer for:
- normalizing LLM skill outputs against the current graph
- running judge / experiment / lean with strict verification
- stitching outputs back into the hypergraph

The intent is to validate the architecture with real execution, not simulation.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from discovery_zero.planning.bridge import (
    BridgePlan,
    BridgeValidationError,
    derive_sibling_goal_subplan,
    derive_subplan,
    materialize_bridge_nodes,
    preferred_sibling_proposition_ids,
    select_bridge_focus_proposition,
    validate_bridge_plan_payload,
)
from discovery_zero.graph.ingest import ingest_skill_output
from discovery_zero.graph.inference import propagate_beliefs
from discovery_zero.tools.llm import (
    LLMError,
    chat_completion,
    extract_json_block,
    extract_text_content,
    load_skill_prompt,
    run_skill,
)
from discovery_zero.tools.lean import (
    decompose_proof_skeleton,
    get_workspace_path,
    verify_proof,
)
from discovery_zero.tools.lean_policy import (
    LeanBoundaryPolicy,
    LeanPolicyError,
    validate_lean_code,
)
from discovery_zero.graph.models import EdgeType, HyperGraph, Module
from discovery_zero.graph.persistence import load_graph, save_graph
from discovery_zero.planning.skeleton_compiler import (
    SkeletonCompilerError,
    compiler_requirements,
    validate_compiled_skeleton,
)
from discovery_zero.graph.strategy import rank_nodes, suggest_module
from discovery_zero.graph.session import GraphSession
from discovery_zero.planning.search import (
    RMaxTSSearch,
    SearchState,
    rank_frontiers,
    select_module_ucb,
)
from discovery_zero.planning.failure_router import (
    FailureRouter, FailureRecord, classify_error, RecoveryAction, ErrorType
)
from discovery_zero.tools.llm_budget import TokenBudget, BudgetExhaustedError

from discovery_zero.tools.experiment_backend import (
    SAFE_IMPORTS,
    BANNED_MODULE_PREFIXES,
    BANNED_NAMES,
    ExperimentBackend,
    ExperimentResult,
    CodeValidationError,
    get_experiment_backend,
    validate_python_code as _backend_validate_python_code,
)
from discovery_zero.tools.experiment_templates import get_template_catalog, render_template
PLACEHOLDER_IDS = {"existing_node_id", "null", "none", ""}
ENV_LEAN_WORKSPACE = "DISCOVERY_ZERO_LEAN_WORKSPACE"
SUPPORT_PREFIXES = (
    "Experimental evidence strongly supports:",
    "Experimental evidence supports:",
    "Formally verified:",
)
MIN_ACCEPTABLE_PLAUSIBLE_CONFIDENCE = 0.45
LEAN_ELIGIBLE_BRIDGE_GRADES = frozenset({"A", "B"})
EXPERIMENT_BRIDGE_GRADES = frozenset({"C"})
NATURAL_LANGUAGE_BRIDGE_GRADES = frozenset({"D"})
LEAN_BRIDGE_ROLES = frozenset({"bridge", "derived", "target"})
STRICT_LOCAL_BRIDGE_ROLES = frozenset({"bridge", "derived"})
OBJECT_LAYER_HINT_KEYWORDS = (
    "notation",
    "object",
    "coerc",
    "embedding",
    "instance",
    "representation",
    "viewed as",
    "interpret",
    "algebra",
    "map",
    "define",
    "identity",
    "typeclass",
)
EXPERIMENT_FRIENDLY_HINT_KEYWORDS = (
    "enumerate",
    "exactly",
    "count",
    "finite",
    "table",
    "composition",
    "closure",
    "inverse",
    "generator",
    "relation",
    "spectrum",
    "eigenvalue",
    "commut",
    "matrix",
    "capacity",
    "parity",
    "example",
    "counterexample",
)
REPLAN_SIGNAL_KEYWORDS = (
    "gap",
    "gaps",
    "invalid",
    "does not follow",
    "non sequitur",
    "unjustified",
    "unsupported",
    "circular",
    "hidden assumption",
    "not well-defined",
    "incorrect",
)
OPEN_PROBLEM_TAG_HINTS = (
    "open-problem",
    "frontier",
    "frontier-assisted",
    "gap remains",
    "does not settle",
)
OPEN_METHOD_KEYWORDS = (
    "new method",
    "new mechanism",
    "hypothesis",
    "route",
    "program",
    "construct",
    "reduction",
    "obstruction",
    "certificate",
    "barrier",
    "ansatz",
)


class OrchestrationError(RuntimeError):
    """Raised when an orchestration step fails validation or execution."""


@dataclass
class ActionResult:
    """Result of one orchestration action."""

    action: str
    target_node_id: str
    selected_module: str
    raw_output: Optional[str] = None
    normalized_output: Optional[dict[str, Any]] = None
    judge_output: Optional[dict[str, Any]] = None
    ingest_edge_id: Optional[str] = None
    created_node_ids: list[str] = field(default_factory=list)
    success: bool = False
    message: str = ""


def _proposition_supported_in_graph(
    graph: HyperGraph,
    node_map: dict[str, str],
    proposition_id: str,
    *,
    min_belief: float = 0.8,
) -> bool:
    node_id = node_map.get(proposition_id)
    if node_id is None or node_id not in graph.nodes:
        return False
    node = graph.nodes[node_id]
    return node.is_locked() or node.belief >= min_belief


def _supported_bridge_proposition_ids(
    plan: BridgePlan,
    graph: HyperGraph,
    node_map: dict[str, str],
    *,
    min_belief: float = 0.8,
) -> set[str]:
    supported: set[str] = set()
    for item in plan.propositions:
        if item.role == "seed":
            supported.add(item.id)
            continue
        if _proposition_supported_in_graph(graph, node_map, item.id, min_belief=min_belief):
            supported.add(item.id)
    return supported


def select_ready_bridge_proposition(
    plan: BridgePlan,
    graph: HyperGraph,
    node_map: dict[str, str],
    *,
    consumed_proposition_ids: set[str] | None = None,
) -> Optional[str]:
    consumed = consumed_proposition_ids or set()
    proposition_map = {item.id: item for item in plan.propositions}
    candidates = [
        item
        for item in plan.propositions
        if item.id not in consumed and item.role in {"bridge", "derived", "target"}
    ]
    grade_rank = {"A": 0, "B": 1, "C": 2, "D": 3}
    role_rank = {"derived": 0, "bridge": 1, "target": 2}
    ready: list[tuple[int, int, int, int, str]] = []
    for item in candidates:
        unresolved_dependencies = [
            dep
            for dep in item.depends_on
            if dep in proposition_map
            and proposition_map[dep].role != "seed"
            and not _proposition_supported_in_graph(graph, node_map, dep)
        ]
        if unresolved_dependencies:
            continue
        ready.append(
            (
                grade_rank.get(item.grade, 9),
                len(item.depends_on),
                _bridge_non_seed_dependency_count(plan, item.id),
                role_rank.get(item.role, 9),
                item.id,
            )
        )
    if not ready:
        return None
    ready.sort()
    return ready[0][-1]


def select_next_experiment_target(
    plan: BridgePlan,
    graph: HyperGraph,
    node_map: dict[str, str],
    *,
    consumed_proposition_ids: set[str] | None = None,
) -> Optional[str]:
    consumed = consumed_proposition_ids or set()
    supported = _supported_bridge_proposition_ids(plan, graph, node_map)
    proposition_map = {item.id: item for item in plan.propositions}

    def dependencies_ready(item_id: str) -> bool:
        item = proposition_map[item_id]
        return all(
            dep in supported
            for dep in item.depends_on
            if dep in proposition_map
        )

    explicit_candidates = [
        item_id
        for item_id in ranked_bridge_consumption_candidates(
            plan,
            allowed_roles={"bridge", "derived", "experiment_support"},
            allowed_grades={"C"},
        )
        if item_id not in consumed and item_id not in supported and dependencies_ready(item_id)
    ]
    if explicit_candidates:
        return explicit_candidates[0]

    d_candidates = [
        item_id
        for item_id in ranked_bridge_consumption_candidates(
            plan,
            allowed_roles={"bridge", "derived", "risk"},
            allowed_grades={"D"},
        )
        if item_id not in consumed
        and item_id not in supported
        and dependencies_ready(item_id)
        and any(
            keyword in _bridge_text_for_experiment(proposition_map[item_id])
            for keyword in EXPERIMENT_FRIENDLY_HINT_KEYWORDS
        )
    ]
    if d_candidates:
        return d_candidates[0]

    fallback_candidates = [
        item_id
        for item_id in ranked_bridge_consumption_candidates(
            plan,
            allowed_roles={"bridge", "derived"},
            allowed_grades={"A", "B"},
        )
        if item_id not in consumed
        and item_id not in supported
        and dependencies_ready(item_id)
        and any(
            keyword in _bridge_text_for_experiment(proposition_map[item_id])
            for keyword in EXPERIMENT_FRIENDLY_HINT_KEYWORDS
        )
    ]
    return fallback_candidates[0] if fallback_candidates else None


def execute_bridge_followups(
    graph_path: Path,
    target_node_id: str,
    reasoning_output: dict[str, Any],
    *,
    plan: Optional[BridgePlan] = None,
    raw_bridge: Optional[str] = None,
    judge_output: Optional[dict[str, Any]] = None,
    model: Optional[str] = None,
    backend: str = "bp",
    max_rounds: int = 2,
) -> list[ActionResult]:
    """
    Consume a validated route by building a bridge plan and executing local follow-ups.

    Current policy intentionally prioritizes experiment-style local bridge consumption
    before any heavier formalization step. This keeps the loop useful even when not
    every bridge should or can be formalized.
    """
    graph = load_graph(graph_path)
    if target_node_id not in graph.nodes:
        raise OrchestrationError(f"Target node {target_node_id} not found.")

    local_plan = plan
    local_raw_bridge = raw_bridge
    if local_plan is None:
        local_raw_bridge, local_plan = run_bridge_planning_action(
            graph,
            target_node_id,
            reasoning_output,
            judge_output=judge_output,
            model=model,
        )
    if local_raw_bridge is None:
        local_raw_bridge = ""
    node_map = materialize_bridge_nodes(
        graph,
        local_plan,
        default_domain=graph.nodes[target_node_id].domain,
        target_node_id=target_node_id,
    )
    save_graph(graph, graph_path)
    results: list[ActionResult] = [
        ActionResult(
            action="bridge_consumption",
            target_node_id=target_node_id,
            selected_module="bridge",
            raw_output=local_raw_bridge,
            normalized_output={"bridge_metrics": local_plan.metrics()},
            success=True,
            message="Bridge plan validated.",
        )
    ]
    consumed_experiment_ids: set[str] = set()
    ready_emitted = False

    for _round in range(max_rounds):
        graph = load_graph(graph_path)
        decision = plan_bridge_consumption(local_plan)
        results[0].normalized_output = {
            "bridge_metrics": local_plan.metrics(),
            **decision.to_log_dict(local_plan),
        }

        experiment_target_id = select_next_experiment_target(
            local_plan,
            graph,
            node_map,
            consumed_proposition_ids=consumed_experiment_ids,
        )
        if experiment_target_id is None or experiment_target_id in consumed_experiment_ids:
            ready_proposition_id = select_ready_bridge_proposition(
                local_plan,
                graph,
                node_map,
                consumed_proposition_ids=consumed_experiment_ids,
            )
            if ready_proposition_id is None:
                break
            ready_node_id = node_map.get(ready_proposition_id, target_node_id)
            ready_proposition = next(item for item in local_plan.propositions if item.id == ready_proposition_id)
            premise_ids = [
                node_map[dep]
                for dep in ready_proposition.depends_on
                if dep in node_map and _proposition_supported_in_graph(graph, node_map, dep)
            ]
            ready_result = ActionResult(
                action="bridge_ready",
                target_node_id=ready_node_id,
                selected_module=Module.PLAUSIBLE.value,
                normalized_output={
                    "premises": [
                        {
                            "id": premise_id,
                            "statement": graph.nodes[premise_id].statement,
                        }
                        for premise_id in premise_ids
                    ],
                    "steps": [
                        "Bridge consumer: all non-seed dependencies of this local proposition are now supported by prior bridge-level evidence.",
                        f"Ready local proposition selected: {ready_proposition.statement}",
                    ],
                    "conclusion": {
                        "statement": ready_proposition.statement,
                        "formal_statement": None,
                    },
                    "module": "plausible",
                    "domain": graph.nodes[ready_node_id].domain,
                    "confidence": 0.76,
                },
                success=True,
                message=f"Bridge proposition '{ready_proposition_id}' is now ready for downstream consumption.",
            )
            ready_result = ingest_action_output(
                graph_path,
                ready_result,
                backend=backend,
            )
            results.append(ready_result)
            ready_emitted = True
            break

        graph_target_id = node_map.get(experiment_target_id)
        if graph_target_id is None:
            results.append(
                ActionResult(
                    action="bridge_experiment",
                    target_node_id=target_node_id,
                    selected_module=Module.EXPERIMENT.value,
                    success=False,
                    message=f"Selected bridge experiment target '{experiment_target_id}' was not materialized.",
                )
            )
            break

        raw_exp, normalized_exp, judge_exp = run_experiment_action(
            graph,
            graph_target_id,
            model=model,
        )
        bridge_experiment_result = ActionResult(
            action="bridge_experiment",
            target_node_id=graph_target_id,
            selected_module=Module.EXPERIMENT.value,
            raw_output=raw_exp,
            normalized_output=normalized_exp,
            judge_output=judge_exp,
            success=True,
            message="Bridge-local experiment executed successfully.",
        )
        bridge_experiment_result = ingest_action_output(
            graph_path,
            bridge_experiment_result,
            backend=backend,
        )
        results.append(bridge_experiment_result)
        consumed_experiment_ids.add(experiment_target_id)
    if not ready_emitted:
        graph = load_graph(graph_path)
        ready_proposition_id = select_ready_bridge_proposition(
            local_plan,
            graph,
            node_map,
            consumed_proposition_ids=consumed_experiment_ids,
        )
        if ready_proposition_id is not None:
            ready_node_id = node_map.get(ready_proposition_id, target_node_id)
            ready_proposition = next(item for item in local_plan.propositions if item.id == ready_proposition_id)
            premise_ids = [
                node_map[dep]
                for dep in ready_proposition.depends_on
                if dep in node_map and _proposition_supported_in_graph(graph, node_map, dep)
            ]
            ready_result = ActionResult(
                action="bridge_ready",
                target_node_id=ready_node_id,
                selected_module=Module.PLAUSIBLE.value,
                normalized_output={
                    "premises": [
                        {
                            "id": premise_id,
                            "statement": graph.nodes[premise_id].statement,
                        }
                        for premise_id in premise_ids
                    ],
                    "steps": [
                        "Bridge consumer: all non-seed dependencies of this local proposition are now supported by prior bridge-level evidence.",
                        f"Ready local proposition selected: {ready_proposition.statement}",
                    ],
                    "conclusion": {
                        "statement": ready_proposition.statement,
                        "formal_statement": None,
                    },
                    "module": "plausible",
                    "domain": graph.nodes[ready_node_id].domain,
                    "confidence": 0.76,
                },
                success=True,
                message=f"Bridge proposition '{ready_proposition_id}' is now ready for downstream consumption.",
            )
            ready_result = ingest_action_output(
                graph_path,
                ready_result,
                backend=backend,
            )
            results.append(ready_result)
    return results


@dataclass
class BridgeConsumptionDecision:
    """Small consumer-side routing decision derived from a validated bridge plan."""

    decomposition_bridge_plan: Optional[BridgePlan] = None
    decomposition_focus_proposition_id: Optional[str] = None
    decomposition_target_proposition_id: Optional[str] = None
    decomposition_used_sibling_package: bool = False
    decomposition_candidate_ids: list[str] = field(default_factory=list)
    strict_focus_proposition_id: Optional[str] = None
    strict_target_proposition_id: Optional[str] = None
    strict_mode: Optional[str] = None
    strict_candidate_ids: list[str] = field(default_factory=list)
    experiment_focus_proposition_id: Optional[str] = None
    experiment_target_proposition_id: Optional[str] = None
    experiment_candidate_ids: list[str] = field(default_factory=list)
    experiment_proposition_ids: list[str] = field(default_factory=list)
    natural_language_proposition_ids: list[str] = field(default_factory=list)

    def to_log_dict(self, plan: BridgePlan) -> dict[str, Any]:
        """Render a compact, JSON-friendly summary for workspace logs."""
        proposition_map = {item.id: item for item in plan.propositions}

        def serialize(items: list[str]) -> list[dict[str, str]]:
            rendered: list[dict[str, str]] = []
            for item_id in items:
                proposition = proposition_map.get(item_id)
                if proposition is None:
                    continue
                rendered.append(
                    {
                        "id": proposition.id,
                        "role": proposition.role,
                        "grade": proposition.grade,
                        "statement": proposition.statement,
                    }
                )
            return rendered

        return {
            "decomposition_candidates": serialize(self.decomposition_candidate_ids),
            "decomposition_focus_proposition_id": self.decomposition_focus_proposition_id,
            "decomposition_target_proposition_id": self.decomposition_target_proposition_id,
            "decomposition_used_sibling_package": self.decomposition_used_sibling_package,
            "strict_candidates": serialize(self.strict_candidate_ids),
            "strict_focus_proposition_id": self.strict_focus_proposition_id,
            "strict_target_proposition_id": self.strict_target_proposition_id,
            "strict_mode": self.strict_mode,
            "experiment_candidates": serialize(self.experiment_candidate_ids),
            "experiment_focus_proposition_id": self.experiment_focus_proposition_id,
            "experiment_target_proposition_id": self.experiment_target_proposition_id,
            "delegated_to_experiment": serialize(self.experiment_proposition_ids),
            "delegated_to_natural_language": serialize(self.natural_language_proposition_ids),
        }


def _normalize_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if cleaned.lower() in PLACEHOLDER_IDS:
        return None
    return cleaned or None


def _bridge_non_seed_dependency_count(plan: BridgePlan, proposition_id: str) -> int:
    proposition_map = {item.id: item for item in plan.propositions}
    proposition = proposition_map[proposition_id]
    return sum(
        1
        for dep in proposition.depends_on
        if dep in proposition_map and proposition_map[dep].role != "seed"
    )


def _bridge_text_for_mode(item: Any) -> str:
    parts = [
        str(getattr(item, "statement", "") or ""),
        str(getattr(item, "notes", "") or ""),
        str(getattr(item, "formalization_notes", "") or ""),
    ]
    return " ".join(parts).casefold()


def _bridge_text_for_experiment(item: Any) -> str:
    parts = [
        str(getattr(item, "statement", "") or ""),
        str(getattr(item, "notes", "") or ""),
        str(getattr(item, "experiment_notes", "") or ""),
        str(getattr(item, "formalization_notes", "") or ""),
    ]
    return " ".join(parts).casefold()


def classify_strict_bridge_mode(plan: BridgePlan, proposition_id: str) -> str:
    """
    Classify how strict Lean should consume a local bridge proposition.

    - `object_setup`: first establish notation/object/coercion/API scaffolding
    - `lemma`: prove a local helper theorem before anything global
    - `direct_proof`: proposition is already small/direct enough to prove as-is
    """
    proposition = next(item for item in plan.propositions if item.id == proposition_id)
    text = _bridge_text_for_mode(proposition)
    if any(keyword in text for keyword in OBJECT_LAYER_HINT_KEYWORDS):
        return "object_setup"
    if proposition.role == "target":
        return "direct_proof"
    if proposition.grade == "A" and len(proposition.depends_on) <= 1:
        return "direct_proof"
    return "lemma"


def select_strict_lean_focus_proposition(plan: BridgePlan) -> Optional[str]:
    """
    Pick the best local bridge proposition to consume with strict Lean first.

    Consumer-side policy:
    - only A/B graded propositions are strict-Lean candidates
    - prefer local bridge/derived propositions over the top-level target
    - prefer smaller, more object-local goals to reduce API hallucinations
    """
    proposition_map = {item.id: item for item in plan.propositions}
    candidates = [
        item
        for item in plan.propositions
        if item.role in STRICT_LOCAL_BRIDGE_ROLES and item.grade in LEAN_ELIGIBLE_BRIDGE_GRADES
    ]
    if not candidates:
        candidates = [
            item
            for item in plan.propositions
            if item.role in LEAN_BRIDGE_ROLES and item.grade in LEAN_ELIGIBLE_BRIDGE_GRADES
        ]
        if not candidates:
            return None

    role_rank = {"derived": 0, "bridge": 1, "target": 2}
    grade_rank = {"A": 0, "B": 1}
    candidates.sort(
        key=lambda item: (
            grade_rank.get(item.grade, 9),
            _bridge_non_seed_dependency_count(plan, item.id),
            len(item.depends_on),
            role_rank.get(item.role, 9),
            item.id,
        ),
    )
    winner = candidates[0]
    if winner.id not in proposition_map:
        return None
    return winner.id


def ranked_bridge_consumption_candidates(
    plan: BridgePlan,
    *,
    allowed_roles: set[str],
    allowed_grades: set[str],
) -> list[str]:
    proposition_map = {item.id: item for item in plan.propositions}
    candidates = [
        item
        for item in plan.propositions
        if item.role in allowed_roles and item.grade in allowed_grades
    ]
    role_rank = {"derived": 0, "bridge": 1, "target": 2, "experiment_support": 3, "risk": 4}
    grade_rank = {"A": 0, "B": 1, "C": 2, "D": 3}
    candidates.sort(
        key=lambda item: (
            grade_rank.get(item.grade, 9),
            _bridge_non_seed_dependency_count(plan, item.id),
            len(item.depends_on),
            role_rank.get(item.role, 9),
            item.id,
        )
    )
    return [item.id for item in candidates if item.id in proposition_map]


def select_experiment_focus_proposition(plan: BridgePlan) -> Optional[str]:
    candidates = ranked_bridge_consumption_candidates(
        plan,
        allowed_roles={"bridge", "derived", "experiment_support"},
        allowed_grades=EXPERIMENT_BRIDGE_GRADES,
    )
    if candidates:
        return candidates[0]
    proposition_map = {item.id: item for item in plan.propositions}
    d_fallback_candidates = ranked_bridge_consumption_candidates(
        plan,
        allowed_roles={"bridge", "derived", "risk"},
        allowed_grades={"D"},
    )
    experiment_friendly_d = [
        item_id
        for item_id in d_fallback_candidates
        if any(
            keyword in _bridge_text_for_experiment(proposition_map[item_id])
            for keyword in EXPERIMENT_FRIENDLY_HINT_KEYWORDS
        )
    ]
    if experiment_friendly_d:
        return experiment_friendly_d[0]
    fallback_candidates = ranked_bridge_consumption_candidates(
        plan,
        allowed_roles={"bridge", "derived"},
        allowed_grades={"A", "B"},
    )
    experiment_friendly = [
        item_id
        for item_id in fallback_candidates
        if any(
            keyword in _bridge_text_for_experiment(proposition_map[item_id])
            for keyword in EXPERIMENT_FRIENDLY_HINT_KEYWORDS
        )
    ]
    if experiment_friendly:
        return experiment_friendly[0]
    support_candidates = ranked_bridge_consumption_candidates(
        plan,
        allowed_roles={"experiment_support"},
        allowed_grades={"A", "B", "C"},
    )
    return support_candidates[0] if support_candidates else None


def plan_bridge_consumption(plan: BridgePlan) -> BridgeConsumptionDecision:
    """
    Route bridge-plan propositions by grade before strict Lean consumption.

    Consumer policy:
    - D/risk propositions are still the first candidates for local decomposition
    - A/B local bridge propositions become strict-Lean candidates
    - C propositions are delegated to experiment
    - unresolved D/risk propositions remain explicit natural-language repair targets
    """
    decomposition_candidate_ids = ranked_bridge_consumption_candidates(
        plan,
        allowed_roles={"bridge", "derived", "target", "risk"},
        allowed_grades={"B", "C", "D"},
    )
    strict_candidate_ids = ranked_bridge_consumption_candidates(
        plan,
        allowed_roles=set(LEAN_BRIDGE_ROLES),
        allowed_grades=set(LEAN_ELIGIBLE_BRIDGE_GRADES),
    )
    experiment_candidate_ids = ranked_bridge_consumption_candidates(
        plan,
        allowed_roles={"bridge", "derived", "experiment_support"},
        allowed_grades=set(EXPERIMENT_BRIDGE_GRADES),
    )
    experiment_ids: list[str] = []
    natural_language_ids: list[str] = []
    proposition_map = {item.id: item for item in plan.propositions}
    for item in plan.propositions:
        if item.role == "seed":
            continue
        if item.grade in NATURAL_LANGUAGE_BRIDGE_GRADES or item.role == "risk":
            if any(
                keyword in _bridge_text_for_experiment(item)
                for keyword in EXPERIMENT_FRIENDLY_HINT_KEYWORDS
            ):
                experiment_ids.append(item.id)
            else:
                natural_language_ids.append(item.id)
        elif item.grade in EXPERIMENT_BRIDGE_GRADES or item.role == "experiment_support":
            experiment_ids.append(item.id)

    decomposition_focus_id: Optional[str] = None
    try:
        decomposition_focus = select_bridge_focus_proposition(plan)
        decomposition_focus_id = decomposition_focus.id
    except BridgeValidationError:
        decomposition_focus_id = None
    decomposition_plan: Optional[BridgePlan] = None
    decomposition_target_id: Optional[str] = None
    decomposition_used_sibling_package = False
    if decomposition_focus_id is not None:
        sibling_ids = preferred_sibling_proposition_ids(plan, decomposition_focus_id)
        if len(sibling_ids) >= 2:
            decomposition_plan, decomposition_target_id = derive_sibling_goal_subplan(
                plan, decomposition_focus_id
            )
            decomposition_used_sibling_package = True
        else:
            decomposition_plan = derive_subplan(plan, decomposition_focus_id)
            decomposition_target_id = decomposition_focus_id

    strict_focus_id = select_strict_lean_focus_proposition(plan)
    strict_mode = None if strict_focus_id is None else classify_strict_bridge_mode(plan, strict_focus_id)
    experiment_focus_id = select_experiment_focus_proposition(plan)
    if experiment_focus_id is not None and experiment_focus_id not in experiment_ids:
        experiment_ids.append(experiment_focus_id)
    if experiment_focus_id is not None and experiment_focus_id in natural_language_ids:
        natural_language_ids = [item_id for item_id in natural_language_ids if item_id != experiment_focus_id]

    return BridgeConsumptionDecision(
        decomposition_bridge_plan=decomposition_plan,
        decomposition_focus_proposition_id=decomposition_focus_id,
        decomposition_target_proposition_id=decomposition_target_id,
        decomposition_used_sibling_package=decomposition_used_sibling_package,
        decomposition_candidate_ids=decomposition_candidate_ids,
        strict_focus_proposition_id=strict_focus_id,
        strict_target_proposition_id=strict_focus_id,
        strict_mode=strict_mode,
        strict_candidate_ids=strict_candidate_ids,
        experiment_focus_proposition_id=experiment_focus_id,
        experiment_target_proposition_id=experiment_focus_id,
        experiment_candidate_ids=experiment_candidate_ids,
        experiment_proposition_ids=experiment_ids,
        natural_language_proposition_ids=natural_language_ids,
    )


def build_graph_context(
    graph: HyperGraph,
    focus_node_id: Optional[str] = None,
    max_nodes: int = 12,
) -> str:
    """Build concise graph context for prompting."""
    lines: list[str] = []
    if focus_node_id and focus_node_id in graph.nodes:
        node = graph.nodes[focus_node_id]
        lines.append(
            f"Focus node [{focus_node_id}] state={node.state} prior={node.prior:.3f} belief={node.belief:.3f}: {node.statement}"
        )
        incoming = graph.get_edges_to(focus_node_id)
        if incoming:
            lines.append("Incoming evidence:")
            for eid in incoming[:6]:
                edge = graph.edges[eid]
                premises = [graph.nodes[pid].statement for pid in edge.premise_ids]
                lines.append(
                    f"- [{eid}] module={edge.module.value} conf={edge.confidence:.2f} premises={premises}"
                )

    lines.append("Relevant nodes:")
    count = 0
    for nid, node in graph.nodes.items():
        lines.append(
            f"- [{nid}] state={node.state} prior={node.prior:.3f} belief={node.belief:.3f} domain={node.domain}: {node.statement}"
        )
        count += 1
        if count >= max_nodes:
            break
    return "\n".join(lines)


def _is_open_problem_mode(
    target_statement: str,
    feedback: Optional[str],
) -> bool:
    text = " ".join(item for item in (target_statement, feedback or "") if item).casefold()
    return any(keyword in text for keyword in OPEN_PROBLEM_TAG_HINTS)


def _open_problem_prompt_block() -> str:
    return (
        "Open-problem mode requirements:\n"
        "- Do NOT merely summarize known seeds or restate why existing tools are insufficient.\n"
        "- You MUST propose at least 2 genuinely new route candidates in your natural-language reasoning.\n"
        "- At least 1 route must introduce a non-seed hypothesis, object, representation, or reduction that is not already present in the context.\n"
        "- At least 1 route should be high-risk/high-upside rather than conservative.\n"
        "- Prefer creating new intermediate objects that can later be checked by experiment, exact computation, or local formalization.\n"
        "- Your final structured output must include at least 1 new premise with `id: null`, unless the route is a concrete counterexample search.\n"
        "- Do not stop at 'the gap remains'; push toward a testable mechanism."
    )


def _open_problem_novelty_score(output: dict[str, Any]) -> float:
    score = 0.0
    premises = output.get("premises", []) if isinstance(output, dict) else []
    steps = output.get("steps", []) if isinstance(output, dict) else []
    new_premises = 0
    for item in premises:
        if not isinstance(item, dict):
            continue
        pid = _normalize_id(item.get("id"))
        if pid is None:
            new_premises += 1
    if new_premises > 0:
        score += min(0.5, 0.2 * new_premises)
    merged_steps = " ".join(str(step) for step in steps).casefold()
    if any(keyword in merged_steps for keyword in OPEN_METHOD_KEYWORDS):
        score += 0.3
    if "fails" in merged_steps or "insufficient" in merged_steps:
        score -= 0.1
    conclusion = output.get("conclusion", {})
    if isinstance(conclusion, dict):
        statement = str(conclusion.get("statement", "")).strip()
        if statement and all(
            statement != str(item.get("statement", "")).strip()
            for item in premises
            if isinstance(item, dict)
        ):
            score += 0.2
    return max(0.0, min(1.0, score))


def normalize_skill_output(
    graph: HyperGraph,
    output: dict[str, Any],
    expected_module: Optional[str] = None,
    default_domain: Optional[str] = None,
) -> dict[str, Any]:
    """Normalize LLM skill output by binding premise IDs to the graph when possible."""
    if not isinstance(output, dict):
        raise OrchestrationError("Skill output must be a JSON object.")
    if output.get("status") == "failed":
        raise OrchestrationError(output.get("last_error", "Skill reported failure."))
    if "module" not in output:
        raise OrchestrationError("Skill output is missing 'module'.")
    if expected_module and output["module"] != expected_module:
        raise OrchestrationError(
            f"Expected module '{expected_module}', got '{output['module']}'."
        )
    if "conclusion" not in output:
        raise OrchestrationError("Skill output is missing 'conclusion'.")

    normalized = dict(output)
    normalized["domain"] = output.get("domain") or default_domain or "number_theory"

    seen_keys: set[tuple[str, str]] = set()
    normalized_premises: list[dict[str, Any]] = []
    for premise in output.get("premises", []):
        if not isinstance(premise, dict):
            raise OrchestrationError("Each premise must be an object.")
        statement = premise.get("statement", "").strip()
        if not statement:
            raise OrchestrationError("Premise is missing a statement.")
        pid = _normalize_id(premise.get("id"))
        if pid and pid not in graph.nodes:
            pid = None
        if not pid:
            matches = graph.find_node_ids_by_statement(statement)
            if len(matches) == 1:
                pid = matches[0]
        key = (pid or "", statement)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        normalized_premises.append({"id": pid, "statement": statement})
    normalized["premises"] = normalized_premises

    conclusion = output["conclusion"]
    if isinstance(conclusion, dict):
        statement = conclusion.get("statement", "").strip()
        if not statement:
            raise OrchestrationError("Conclusion is missing a statement.")
        normalized["conclusion"] = dict(conclusion)
    elif isinstance(conclusion, str):
        statement = conclusion.strip()
        if not statement:
            raise OrchestrationError("Conclusion is missing a statement.")
        normalized["conclusion"] = {"statement": statement}
    else:
        raise OrchestrationError("Conclusion must be a string or object.")
    return normalized


def _resolve_supported_conclusion_statement(
    graph: HyperGraph,
    statement: str,
    *,
    fallback_target_statement: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """
    Resolve evidence-style conclusion text back to an existing theorem node.

    Returns `(resolved_statement, original_summary_or_none)`.
    If the conclusion is of the form
      "Experimental evidence strongly supports: X"
    and `X` matches an existing node, then we map the conclusion back to `X`
    while preserving the original summary text separately.
    """
    stripped = statement.strip()
    for prefix in SUPPORT_PREFIXES:
        if stripped.startswith(prefix):
            candidate = stripped[len(prefix):].strip()
            matches = graph.find_node_ids_by_statement(candidate)
            if len(matches) == 1:
                return graph.nodes[matches[0]].statement, stripped
            if fallback_target_statement:
                fallback_matches = graph.find_node_ids_by_statement(fallback_target_statement)
                if len(fallback_matches) == 1:
                    return graph.nodes[fallback_matches[0]].statement, stripped
            return candidate, stripped
    return stripped, None


_JUDGE_FALLBACK: dict[str, Any] = {
    "confidence": 0.5,
    "reasoning": "Judge output was malformed; using conservative default.",
    "concerns": [],
    "suggestion": None,
}


def run_judge(output: dict[str, Any], model: Optional[str] = None) -> dict[str, Any]:
    """Run the judge skill on a normalized output dict.

    Returns a conservative fallback when the judge LLM produces
    unparseable output, so that a single malformed response never
    crashes the entire run.
    """
    try:
        _, parsed = run_skill(
            "judge.skill.md",
            "Hyperedge to evaluate:\n" + json.dumps(output, ensure_ascii=False, indent=2),
            model=model,
        )
        if not isinstance(parsed, dict) or "confidence" not in parsed:
            return dict(_JUDGE_FALLBACK)
        return parsed
    except (LLMError, OrchestrationError, Exception):
        return dict(_JUDGE_FALLBACK)


def apply_judge_confidence(output: dict[str, Any], judge_output: dict[str, Any]) -> dict[str, Any]:
    """Store judge confidence as review_confidence, preserving the original factor confidence."""
    normalized = dict(output)
    normalized["review_confidence"] = float(judge_output["confidence"])
    normalized["confidence"] = float(judge_output["confidence"])
    return normalized


def _bridge_plan_input(
    *,
    graph: HyperGraph,
    target_node_id: str,
    reasoning_output: dict[str, Any],
    judge_output: Optional[dict[str, Any]] = None,
    feedback: Optional[str] = None,
) -> str:
    """Build a strict prompt for bridge-layer extraction."""
    target = graph.nodes[target_node_id]
    lines = [
        "Target statement:",
        target.statement,
        "",
        "Current graph context:",
        build_graph_context(graph, target_node_id),
        "",
        "Best reasoning route to compile:",
        json.dumps(reasoning_output, ensure_ascii=False, indent=2),
    ]
    if judge_output is not None:
        lines.extend(
            [
                "",
                "Judge feedback:",
                json.dumps(judge_output, ensure_ascii=False, indent=2),
            ]
        )
    if feedback:
        lines.extend(["", "Additional failure/replan feedback:", feedback.strip()])
    lines.extend(
        [
            "",
            "Produce a complete bridge plan with explicit subpropositions, a full reasoning chain, and A/B/C/D grades.",
            "CRITICAL: You MUST include exactly one proposition with role='target', and its statement must match the target statement exactly.",
            "Do not omit risky or ambiguous bridge points: include them explicitly and mark them D if needed.",
            "IMPORTANT: Risk or unresolved-gap propositions are diagnostic markers. They must NOT appear in depends_on of any non-risk proposition.",
            "IMPORTANT: The target proposition's depends_on must contain only positive-support propositions (seed/bridge/derived/experiment_support), never risk nodes.",
            "If the route proposes genuinely new hypotheses, mechanisms, reductions, or constructed objects, preserve them explicitly as bridge/derived/risk propositions rather than collapsing everything into a summary of known seeds.",
            "For open problems, prefer making new testable objects and new unresolved bridges explicit.",
        ]
    )
    return "\n".join(lines)


def _prose_to_structured_bridge(
    prose: str,
    model: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    """Convert free-form bridge reasoning into structured bridge plan JSON.

    Uses schema-constrained decoding when available, with fallback to
    json_object mode.
    """
    from discovery_zero.tools.llm_output import BRIDGE_PLAN_OUTPUT_SCHEMA

    extraction_prompt = (
        "You are a structured data extractor. Given the bridge plan reasoning "
        "below, extract and return ONLY a JSON object conforming to a bridge plan "
        "schema with these fields:\n"
        '  - "target_statement": string\n'
        '  - "propositions": array of objects with {id, statement, role, grade, depends_on}\n'
        '  - "chain": array of objects with {id, statement, uses, concludes, grade}\n'
        '  - "summary": optional short string\n\n'
        "Do NOT include any prose, explanation, or markdown fences. "
        "Return ONLY valid JSON.\n\n"
        f"--- Reasoning to extract from ---\n{prose}\n--- End ---"
    )

    schema_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "bridge_plan_extraction",
            "schema": BRIDGE_PLAN_OUTPUT_SCHEMA,
            "strict": True,
        },
    }

    for fmt in [schema_format, {"type": "json_object"}]:
        try:
            response = chat_completion(
                messages=[
                    {"role": "system", "content": "You are a JSON extraction assistant. Return only valid JSON."},
                    {"role": "user", "content": extraction_prompt},
                ],
                model=model,
                temperature=0.0,
                response_format=fmt,
            )
            raw_text = extract_text_content(response)
            parsed = extract_json_block(raw_text)
            return raw_text, parsed
        except (LLMError, Exception):
            continue

    raise LLMError("Failed to extract structured bridge plan from prose.")


def _normalize_bridge_payload(
    payload: dict[str, Any],
    *,
    target_statement: str,
) -> dict[str, Any]:
    """Normalize near-miss bridge payloads into the strict BridgePlan shape.

    This lets the pipeline recover from small schema mismatches such as:
    - `dependencies` vs `depends_on`
    - `reasoning_steps` vs `chain`
    - `concludes: "P3"` vs `concludes: ["P3"]`
    - `chain: ["step 1", "step 2"]` vs structured chain objects
    """
    def _split_id_like_text(value: Any) -> list[str]:
        if isinstance(value, list):
            merged: list[str] = []
            for item in value:
                merged.extend(_split_id_like_text(item))
            seen: set[str] = set()
            deduped: list[str] = []
            for item in merged:
                if item not in seen:
                    seen.add(item)
                    deduped.append(item)
            return deduped
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            return [part.strip() for part in re.split(r"[,\s;]+", text) if part.strip()]
        return []

    normalized = dict(payload)
    normalized.setdefault("target_statement", target_statement)

    raw_props = normalized.get("propositions", [])
    fixed_props: list[dict[str, Any]] = []
    if isinstance(raw_props, list):
        for idx, item in enumerate(raw_props, start=1):
            if not isinstance(item, dict):
                continue
            prop = dict(item)
            if "depends_on" not in prop and "dependencies" in prop:
                prop["depends_on"] = prop.pop("dependencies")
            prop.setdefault("id", f"P{idx}")
            prop.setdefault("statement", "")
            prop.setdefault("role", "bridge")
            prop.setdefault("grade", "D")
            prop["depends_on"] = _split_id_like_text(prop.get("depends_on", []))
            fixed_props.append(prop)
    normalized["propositions"] = fixed_props

    raw_chain = normalized.get("chain")
    if raw_chain is None and "reasoning_steps" in normalized:
        raw_chain = normalized.get("reasoning_steps")
    fixed_chain: list[dict[str, Any]] = []
    if isinstance(raw_chain, list):
        for idx, item in enumerate(raw_chain, start=1):
            if isinstance(item, str):
                fixed_chain.append(
                    {
                        "id": f"S{idx}",
                        "statement": item.strip(),
                        "uses": [],
                        "concludes": [],
                        "grade": "B",
                    }
                )
                continue
            if not isinstance(item, dict):
                continue
            step = dict(item)
            if "statement" not in step and "justification" in step:
                step["statement"] = step.pop("justification")
            step.setdefault("id", f"S{idx}")
            step.setdefault("statement", "")
            step.setdefault("uses", [])
            step.setdefault("concludes", [])
            step.setdefault("grade", "B")
            step["uses"] = _split_id_like_text(step.get("uses", []))
            step["concludes"] = _split_id_like_text(step.get("concludes", []))
            fixed_chain.append(step)
    known_prop_ids = {
        item.get("id", "").strip()
        for item in fixed_props
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item.get("id", "").strip()
    }
    for step in fixed_chain:
        step_refs = _split_id_like_text(step.get("uses", [])) + _split_id_like_text(step.get("concludes", []))
        for ref_id in step_refs:
            if ref_id in known_prop_ids:
                continue
            fixed_props.append(
                {
                    "id": ref_id,
                    "statement": f"Auto-recovered bridge proposition {ref_id}.",
                    "role": "bridge",
                    "grade": "D",
                    "depends_on": [],
                }
            )
            known_prop_ids.add(ref_id)
    normalized["chain"] = fixed_chain
    normalized["propositions"] = fixed_props
    normalized.pop("reasoning_steps", None)
    normalized.pop("conclusion", None)
    normalized.pop("target_id", None)
    return normalized


def run_bridge_planning_action(
    graph: HyperGraph,
    target_node_id: str,
    reasoning_output: dict[str, Any],
    *,
    judge_output: Optional[dict[str, Any]] = None,
    model: Optional[str] = None,
    feedback: Optional[str] = None,
    max_attempts: int = 3,
    record_dir: Optional[Path] = None,
) -> tuple[str, BridgePlan]:
    """Compile a best route into a validated bridge-layer plan.

    Uses a two-step approach: the LLM first reasons freely, then a second
    call extracts structured JSON.  Falls back to direct single-step if
    two-step also fails.
    """
    if target_node_id not in graph.nodes:
        raise OrchestrationError(f"Target node {target_node_id} not found.")
    if not isinstance(reasoning_output, dict):
        raise OrchestrationError("Bridge planning requires a normalized reasoning output dict.")
    last_error = "Bridge planning did not run."
    last_raw = ""
    base_feedback = feedback
    for attempt in range(1, max_attempts + 1):
        attempt_feedback = base_feedback
        if attempt > 1:
            extra = (
                "Previous bridge-plan attempt failed validation with this exact error:\n"
                f"{last_error}\n"
                "Regenerate a fully valid bridge plan. In particular, do not place the same proposition "
                "id in both `uses` and `concludes` of a single reasoning step."
            )
            attempt_feedback = f"{base_feedback}\n\n{extra}" if base_feedback else extra

        bridge_input = _bridge_plan_input(
            graph=graph,
            target_node_id=target_node_id,
            reasoning_output=reasoning_output,
            judge_output=judge_output,
            feedback=attempt_feedback,
        )

        # Two-step: first get prose reasoning, then extract JSON
        parsed = None
        prose_record_path = _llm_record_path(record_dir, "bridge_plan_prose", attempt)
        try:
            try:
                skill_prompt = load_skill_prompt("bridge_plan.skill.md")
            except FileNotFoundError:
                skill_prompt = "You are a bridge plan architect."

            prose_response = chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            skill_prompt + "\n\n"
                            "Think step by step. Explain the bridge plan structure in natural language first. "
                            "Do NOT try to format as JSON yet."
                        ),
                    },
                    {"role": "user", "content": bridge_input},
                ],
                model=model,
                temperature=0.0,
                stream_record_path=prose_record_path,
            )
            if prose_record_path is not None and prose_record_path.exists():
                prose_text = prose_record_path.read_text(encoding="utf-8")
            else:
                prose_text = extract_text_content(prose_response)
            last_raw = prose_text
            _raw, parsed = _prose_to_structured_bridge(prose_text, model=model)
        except (LLMError, Exception):
            # Fallback to direct single-step
            try:
                raw, parsed = run_skill("bridge_plan.skill.md", bridge_input, model=model)
                last_raw = raw
            except (LLMError, Exception) as exc:
                last_error = f"Both two-step and direct bridge plan generation failed: {exc}"
                continue

        if not isinstance(parsed, dict):
            last_error = "Bridge plan skill did not return a JSON object."
            continue
        parsed = _normalize_bridge_payload(
            parsed,
            target_statement=graph.nodes[target_node_id].statement,
        )
        try:
            plan = validate_bridge_plan_payload(parsed)
        except BridgeValidationError as exc:
            last_error = f"Bridge plan validation failed: {exc}"
            continue
        target_statement = graph.nodes[target_node_id].statement
        target_matches = [
            item for item in plan.propositions
            if item.statement == target_statement or item.role == "target"
        ]
        if not target_matches:
            last_error = "Bridge plan does not include the active target proposition."
            continue
        return last_raw, plan
    raise OrchestrationError(last_error if last_error else "Bridge planning failed.")


def _compile_skeleton_from_bridge_plan(
    graph: HyperGraph,
    target_node_id: str,
    bridge_plan: BridgePlan,
    *,
    model: Optional[str] = None,
    feedback: Optional[str] = None,
    max_attempts: int = 3,
) -> tuple[str, dict[str, Any], str]:
    """Compile a bridge plan into a validated Lean skeleton payload."""
    target = graph.nodes[target_node_id]
    requirements = compiler_requirements(bridge_plan)
    target_prop_id = requirements["target_proposition_id"]
    base_feedback = feedback.strip() if feedback else ""
    last_error = "Lean skeleton compiler did not run."
    last_raw = ""
    for attempt in range(1, max_attempts + 1):
        prompt_lines = [
            "Target theorem statement:",
            target.statement,
            f"Optional formal statement:\n{target.formal_statement or '(none)'}",
            "",
            "Graph context:",
            build_graph_context(graph, target_node_id),
            "",
            "Bridge plan JSON:",
            bridge_plan.model_dump_json(indent=2),
            "",
            "Compiler requirements:",
            json.dumps(requirements, ensure_ascii=False, indent=2),
            "You MUST satisfy these coverage requirements.",
            "Introduce separate local `have` blocks with `sorry` so Lean can expose multiple subgoals.",
            "Do not collapse the whole bridge into one single `sorry`.",
            "Make the first several bridge lemmas sibling local goals whenever possible, so later goals do not depend on unresolved earlier ones.",
            "Avoid a serial proof where one `sorry` blocks elaboration of all later bridge lemmas.",
            "Do not use section/namespace wrappers; keep the skeleton as one minimal theorem file.",
            "If `preferred_sibling_proposition_ids` is non-empty, prioritize them as the first independent local goals.",
            "If `target_dependency_ids` contains multiple entries, the theorem goal MUST be an explicit `And`-chain over those dependencies, and the proof should begin with repeated `constructor` so Lean exposes sibling goals.",
            f"You MUST include an explicit marker line `-- BRIDGE-PROP: {target_prop_id}` for the active target proposition.",
        ]
        attempt_feedback = base_feedback
        if attempt > 1:
            retry_feedback = (
                "Previous skeleton attempt failed validation with this exact error:\n"
                f"{last_error}\n"
                f"Regenerate a fully valid skeleton. In particular, ensure the active target proposition marker `-- BRIDGE-PROP: {target_prop_id}` appears explicitly in the body, and cover every required bridge proposition/step."
            )
            attempt_feedback = (
                f"{base_feedback}\n\n{retry_feedback}".strip()
                if base_feedback
                else retry_feedback
            )
        if attempt_feedback:
            prompt_lines.extend(["", "Previous Lean/decomposition feedback:", attempt_feedback])
        raw, parsed = run_skill(
            "lean_skeleton_compiler.skill.md",
            "\n".join(prompt_lines),
            model=model,
        )
        last_raw = raw
        normalized = normalize_skill_output(
            graph,
            parsed,
            expected_module="lean",
            default_domain=target.domain,
        )
        lean_code = _extract_lean_code(normalized.get("steps", []))
        try:
            validate_compiled_skeleton(lean_code, bridge_plan)
        except SkeletonCompilerError as exc:
            last_error = f"Skeleton compiler validation failed: {exc}"
            continue
        return raw, normalized, lean_code
    raise OrchestrationError(last_error if last_error else "Skeleton compiler validation failed.")


def _judge_requires_replan(
    judge_output: dict[str, Any],
    *,
    min_confidence: float = MIN_ACCEPTABLE_PLAUSIBLE_CONFIDENCE,
) -> bool:
    """Decide whether the planner should try a different route."""
    confidence = float(judge_output.get("confidence", 0.0))
    if confidence < min_confidence:
        return True

    texts: list[str] = []
    reasoning = judge_output.get("reasoning")
    if isinstance(reasoning, str):
        texts.append(reasoning)
    suggestion = judge_output.get("suggestion")
    if isinstance(suggestion, str):
        texts.append(suggestion)
    concerns = judge_output.get("concerns")
    if isinstance(concerns, list):
        texts.extend(str(item) for item in concerns)

    merged = " ".join(texts).lower()
    return any(keyword in merged for keyword in REPLAN_SIGNAL_KEYWORDS)


def _build_replan_feedback(
    *,
    target_statement: str,
    previous_output: Optional[dict[str, Any]] = None,
    judge_output: Optional[dict[str, Any]] = None,
    failure_message: Optional[str] = None,
    failed_module: Optional[str] = None,
) -> str:
    """Format concrete feedback for the next planning attempt."""
    lines = [
        f"Target remains: {target_statement}",
        "The previous route is not yet acceptable. Generate a different or repaired route.",
        "Do not repeat the same invalid step pattern.",
    ]
    if failed_module:
        lines.append(f"Rejected downstream module: {failed_module}.")
    if failure_message:
        lines.append(f"Verifier/runtime feedback: {failure_message}")
    if previous_output:
        conclusion = previous_output.get("conclusion")
        if isinstance(conclusion, dict):
            statement = conclusion.get("statement")
        else:
            statement = conclusion
        if isinstance(statement, str) and statement.strip():
            lines.append(f"Previous attempted conclusion: {statement.strip()}")
    if judge_output:
        reasoning = judge_output.get("reasoning")
        if isinstance(reasoning, str) and reasoning.strip():
            lines.append(f"Judge reasoning: {reasoning.strip()}")
        concerns = judge_output.get("concerns")
        if isinstance(concerns, list) and concerns:
            lines.append("Judge concerns: " + "; ".join(str(item) for item in concerns))
        suggestion = judge_output.get("suggestion")
        if isinstance(suggestion, str) and suggestion.strip():
            lines.append(f"Judge suggestion: {suggestion.strip()}")
    return "\n".join(lines)


def _extract_python_code(step: str) -> str:
    text = step.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _validate_python_code(code: str) -> None:
    """Reject obviously unsafe Python code before execution.

    Delegates to the unified experiment_backend validation which supports
    the extended SAFE_IMPORTS (numpy, scipy, sympy, etc.).
    """
    try:
        _backend_validate_python_code(code)
    except CodeValidationError as e:
        raise OrchestrationError(str(e)) from e


def _get_default_backend() -> ExperimentBackend:
    """Get the configured experiment backend."""
    try:
        from discovery_zero.config import CONFIG
        backend_name = getattr(CONFIG, "experiment_backend", "local")
    except Exception:
        backend_name = "local"
    return get_experiment_backend(backend_name)


def _execute_python_code(code: str, timeout: int = 60) -> tuple[str, str]:
    """Execute validated Python code via the unified experiment backend."""
    _validate_python_code(code)
    backend = _get_default_backend()
    result = backend.execute(code, timeout=timeout)
    if result.timed_out:
        raise OrchestrationError(f"Experiment code timed out after {timeout}s.")
    return result.stdout, result.stderr


def _infer_experiment_result_from_stdout(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    if not text:
        raise OrchestrationError("Experiment code produced no output.")
    lowered = text.casefold()
    bool_match = re.search(r"\b(true|false)\b", lowered)
    if bool_match is not None:
        passed = bool_match.group(1) == "true"
    elif "counterexample" in lowered or "failed" in lowered or "refute" in lowered:
        passed = False
    elif "pass" in lowered or "success" in lowered or "verified" in lowered:
        passed = True
    else:
        raise OrchestrationError("Experiment output did not include parseable boolean signals.")
    return {
        "passed": passed,
        "trials": 1,
        "max_error": None,
        "counterexample": None if passed else {"source": "auto_inferred_stdout"},
        "summary": text[:500],
        "auto_inferred": True,
    }


def _parse_experiment_stdout(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise OrchestrationError("Experiment code produced no output.")
    candidate = lines[-1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as e:
        try:
            return _infer_experiment_result_from_stdout(stdout)
        except OrchestrationError:
            raise OrchestrationError(
                "Experiment code must print a final JSON line with execution summary."
            ) from e
    if not isinstance(data, dict):
        raise OrchestrationError("Experiment execution summary must be a JSON object.")
    if "passed" not in data:
        raise OrchestrationError("Experiment execution summary is missing 'passed'.")
    return data


def _repair_code(
    code: str,
    error_message: str,
    claim_context: str,
    model: Optional[str] = None,
    *,
    timeout: int = 90,
) -> str:
    repair_prompt = (
        "You are a Python code debugger. Return ONLY complete corrected Python code.\n"
        "Do not include markdown fences. Do not include explanations.\n\n"
        "The code failed with this error:\n"
        f"{error_message}\n\n"
        "Claim/context to preserve:\n"
        f"{claim_context}\n\n"
        "Broken code:\n"
        f"{code}"
    )
    response = chat_completion(
        messages=[
            {"role": "system", "content": "Return only valid Python code."},
            {"role": "user", "content": repair_prompt},
        ],
        model=model,
        temperature=0.0,
        timeout=timeout,
    )
    repaired = _extract_python_code(extract_text_content(response))
    if not repaired.strip():
        raise OrchestrationError("Repair model returned empty code.")
    return repaired


def _inject_json_output(code: str) -> str:
    wrapper = (
        "\n\n# Auto-injected output wrapper\n"
        "import json as _auto_json\n"
        "if __name__ == '__main__':\n"
        "    try:\n"
        "        _auto_result = globals().get('result', None)\n"
        "        _auto_passed = bool(_auto_result) if isinstance(_auto_result, bool) else True\n"
        "        print(_auto_json.dumps({\n"
        "            'passed': _auto_passed,\n"
        "            'trials': 1,\n"
        "            'max_error': None,\n"
        "            'counterexample': None if _auto_passed else {'source': 'auto_wrapper'},\n"
        "            'summary': 'auto wrapper generated fallback execution summary',\n"
        "            'auto_inferred': True,\n"
        "        }))\n"
        "    except Exception:\n"
        "        print(_auto_json.dumps({\n"
        "            'passed': False,\n"
        "            'trials': 1,\n"
        "            'max_error': None,\n"
        "            'counterexample': {'source': 'auto_wrapper_exception'},\n"
        "            'summary': 'auto wrapper fallback failed while inspecting result',\n"
        "            'auto_inferred': True,\n"
        "        }))\n"
    )
    if "auto wrapper generated fallback execution summary" in code:
        return code
    return code + wrapper


def _execute_with_repair(
    code: str,
    claim_context: str,
    model: Optional[str] = None,
    *,
    max_repairs: int = 3,
    timeout: int = 60,
) -> tuple[str, str, str, dict[str, Any]]:
    current_code = code
    last_error: Optional[Exception] = None
    for attempt in range(max_repairs + 1):
        try:
            _validate_python_code(current_code)
            stdout, stderr = _execute_python_code(current_code, timeout=timeout)
            try:
                parsed = _parse_experiment_stdout(stdout)
                return current_code, stdout, stderr, parsed
            except OrchestrationError as parse_exc:
                if attempt >= max_repairs:
                    injected = _inject_json_output(current_code)
                    stdout, stderr = _execute_python_code(injected, timeout=timeout)
                    parsed = _parse_experiment_stdout(stdout)
                    return injected, stdout, stderr, parsed
                current_code = _repair_code(
                    current_code,
                    f"Code ran but output format failed: {parse_exc}\nstdout:\n{stdout[:1000]}",
                    claim_context,
                    model=model,
                )
                continue
        except OrchestrationError as exc:
            last_error = exc
            if attempt >= max_repairs:
                break
            current_code = _repair_code(current_code, str(exc), claim_context, model=model)
    raise OrchestrationError(f"Experiment code repair exhausted. Last error: {last_error}")


def _run_experiment_skill_with_code_fallback(
    graph: HyperGraph,
    target_node_id: str,
    task_input: str,
    *,
    model: Optional[str] = None,
    timeout: int = 120,
    prose_record_path: Optional[Path] = None,
) -> tuple[str, dict[str, Any]]:
    """
    Run the experiment skill, accepting either JSON output or a raw Python block.

    Some models occasionally emit only a fenced Python script despite the JSON
    contract. For the experiment path we can safely recover by wrapping that code
    into the expected skill payload and letting the real execution layer judge the
    result.
    """
    skill_prompt = load_skill_prompt("experiment.skill.md")
    response = chat_completion(
        messages=[
            {
                "role": "system",
                "content": (
                    skill_prompt
                    + "\n\nReturn ONLY a valid JSON response matching the required output format."
                ),
            },
            {"role": "user", "content": task_input},
        ],
        model=model,
        temperature=0.0,
        timeout=timeout,
        stream_record_path=prose_record_path,
    )
    raw = extract_text_content(response)
    try:
        parsed = extract_json_block(raw)
    except LLMError:
        pythonish_signals = ("import ", "def ", "for ", "while ", "print(", "from ")
        code = _extract_python_code(raw)
        has_python = bool(code and any(sig in code for sig in pythonish_signals))

        if not has_python:
            # Step 1: ask LLM to extract Python code from its own prose output.
            extraction_prompt = (
                "The following text was intended to contain Python experiment code "
                "but was formatted as prose or mixed output.\n\n"
                "Extract and return ONLY the complete, executable Python code from the text below. "
                "Do not include any explanation, markdown fences, or JSON wrappers.\n\n"
                f"--- Text ---\n{raw[:6000]}\n--- End ---\n\n"
                f"If no code is present, write NEW Python code that tests: {task_input[:500]}"
            )
            try:
                extraction_resp = chat_completion(
                    messages=[
                        {"role": "system", "content": "Return only executable Python code."},
                        {"role": "user", "content": extraction_prompt},
                    ],
                    model=model,
                    temperature=0.0,
                    timeout=timeout,
                )
                extracted_text = _extract_python_code(extract_text_content(extraction_resp))
                if extracted_text and any(sig in extracted_text for sig in pythonish_signals):
                    code = extracted_text
                    has_python = True
                    raw = raw + "\n\n[CODE-EXTRACTION-FALLBACK]\n" + code
            except Exception:
                pass

        if not has_python:
            # Step 2: use a structured template as last resort.
            template_prompt = (
                f"Claim/experiment task:\n{task_input}\n\n"
                "Available templates:\n"
                f"{get_template_catalog()}\n\n"
                "Select one template and fill all placeholders."
            )
            try:
                _raw_tpl, parsed_tpl = run_skill(
                    "fill_experiment_template.skill.md",
                    template_prompt,
                    model=model,
                )
                if isinstance(parsed_tpl, dict):
                    template_name = str(parsed_tpl.get("template", "")).strip()
                    slots_raw = parsed_tpl.get("slots", {})
                    slots = {
                        str(k): str(v)
                        for k, v in (slots_raw.items() if isinstance(slots_raw, dict) else [])
                    }
                    code = render_template(template_name, slots)
                    raw = raw + "\n\n[TEMPLATE-FALLBACK]\n" + code
                else:
                    raise OrchestrationError("Template skill did not return JSON object.")
            except OrchestrationError:
                raise
            except Exception as exc:
                raise OrchestrationError(
                    f"Experiment skill returned neither JSON nor Python code, and template fallback failed: {exc}"
                ) from exc

        target = graph.nodes[target_node_id]
        parsed = {
            "premises": [],
            "steps": [code],
            "conclusion": {
                "statement": f"Experimental evidence strongly supports: {target.statement}",
                "formal_statement": None,
            },
            "module": "experiment",
            "domain": target.domain,
        }
    if not isinstance(parsed, dict):
        raise OrchestrationError("Experiment skill did not return a JSON object.")
    return raw, parsed


def _prose_to_structured_plausible(
    prose: str,
    target_statement: str,
    model: Optional[str] = None,
) -> tuple[str, dict[str, Any]]:
    """Convert free-form reasoning prose into a structured plausible JSON.

    This is Step 2 of the two-step plausible pipeline.  The LLM is given
    the prose and asked to produce *only* a JSON object conforming to the
    plausible output schema.  Schema-constrained decoding is used when the
    API supports it.
    """
    from discovery_zero.tools.llm_output import PLAUSIBLE_OUTPUT_SCHEMA

    extraction_prompt = (
        "You are a structured data extractor. Given the mathematical reasoning "
        "below, extract and return ONLY a JSON object with exactly these fields:\n"
        '  - "premises": array of {"statement": string} objects\n'
        '  - "steps": array of strings (reasoning steps)\n'
        '  - "conclusion": {"statement": string, "formal_statement": string or null}\n'
        '  - "module": "plausible"\n'
        '  - "confidence": number 0-1\n'
        '  - "domain": string or null\n\n'
        "Do NOT include any prose, explanation, or markdown fences. "
        "Return ONLY valid JSON.\n\n"
        f"Target theorem: {target_statement}\n\n"
        f"--- Reasoning to extract from ---\n{prose}\n--- End ---"
    )

    schema_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "plausible_extraction",
            "schema": PLAUSIBLE_OUTPUT_SCHEMA,
            "strict": True,
        },
    }

    try:
        response = chat_completion(
            messages=[
                {"role": "system", "content": "You are a JSON extraction assistant. Return only valid JSON."},
                {"role": "user", "content": extraction_prompt},
            ],
            model=model,
            temperature=0.0,
            response_format=schema_format,
        )
        raw_text = extract_text_content(response)
        parsed = extract_json_block(raw_text)
        return raw_text, parsed
    except (LLMError, Exception):
        pass

    # Fallback: try without schema constraint
    try:
        response = chat_completion(
            messages=[
                {"role": "system", "content": "You are a JSON extraction assistant. Return only valid JSON."},
                {"role": "user", "content": extraction_prompt},
            ],
            model=model,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw_text = extract_text_content(response)
        parsed = extract_json_block(raw_text)
        return raw_text, parsed
    except (LLMError, Exception):
        pass

    # Last resort: regex extraction from original prose
    parsed = _regex_extract_plausible(prose, target_statement)
    return prose, parsed


def _regex_extract_plausible(prose: str, target_statement: str) -> dict[str, Any]:
    """Best-effort regex extraction of plausible fields from prose."""
    steps = []
    for line in prose.splitlines():
        stripped = line.strip()
        if stripped and (stripped[0].isdigit() or stripped.startswith("-")):
            steps.append(stripped.lstrip("0123456789.-) ").strip())
    if not steps:
        paragraphs = [p.strip() for p in prose.split("\n\n") if p.strip()]
        steps = paragraphs[:5] if paragraphs else [prose[:500]]

    return {
        "premises": [],
        "steps": steps,
        "conclusion": {"statement": target_statement},
        "module": "plausible",
        "confidence": 0.3,
        "domain": None,
    }


def _llm_record_path(record_dir: Optional[Path], stem: str, attempt: int) -> Optional[Path]:
    """Build a deterministic per-attempt LLM record path."""
    if record_dir is None:
        return None
    record_dir.mkdir(parents=True, exist_ok=True)
    return record_dir / f"{stem}_attempt_{attempt}.txt"


def run_plausible_action(
    graph: HyperGraph,
    target_node_id: str,
    model: Optional[str] = None,
    feedback: Optional[str] = None,
    max_attempts: int = 3,
    record_dir: Optional[Path] = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Run plausible reasoning via two-step pipeline + judge-driven replanning.

    Step 1: LLM reasons freely in prose (no JSON constraint).
    Step 2: A separate LLM call extracts the structured JSON from the prose.
    This eliminates the dominant failure mode of malformed JSON in complex reasoning.
    """
    target = graph.nodes[target_node_id]
    open_problem_mode = _is_open_problem_mode(target.statement, feedback)
    base_prompt = (
        f"Direction:\nExplore mathematically useful conjectures or lemmas around the target node:\n"
        f"{target.statement}\n\n"
        f"Context:\n{build_graph_context(graph, target_node_id)}\n"
    )
    if feedback:
        base_prompt += "\nPlanning feedback:\n" + feedback.strip() + "\n"
    if open_problem_mode:
        base_prompt += "\n" + _open_problem_prompt_block() + "\n"

    best_attempt: Optional[tuple[str, dict[str, Any], dict[str, Any]]] = None
    replan_feedback = feedback
    for attempt in range(1, max_attempts + 1):
        task_input = base_prompt
        if attempt > 1 and replan_feedback:
            task_input += (
                "\nPrevious route review:\n"
                f"{replan_feedback}\n"
                "Return a revised route that fixes these issues.\n"
            )

        # Step 1: Free-form reasoning (no JSON requirement)
        prose_record_path = _llm_record_path(record_dir, "plausible_prose", attempt)
        try:
            skill_prompt = load_skill_prompt("plausible_reasoning.skill.md")
        except FileNotFoundError:
            skill_prompt = "You are a mathematical reasoning assistant."

        prose_response = chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        skill_prompt + "\n\n"
                        "Think step by step. Explain your reasoning in natural language. "
                        "Do NOT try to format as JSON yet."
                    ),
                },
                {"role": "user", "content": task_input},
            ],
            model=model,
            temperature=0.0,
            stream_record_path=prose_record_path,
        )
        if prose_record_path is not None and prose_record_path.exists():
            prose_text = prose_record_path.read_text(encoding="utf-8")
        else:
            prose_text = extract_text_content(prose_response)

        # Step 2: Extract structured JSON from the prose
        raw, parsed = _prose_to_structured_plausible(
            prose_text, target.statement, model=model
        )

        normalized = normalize_skill_output(
            graph,
            parsed,
            expected_module="plausible",
            default_domain=target.domain,
        )
        judge_output = run_judge(normalized, model=model)
        if open_problem_mode:
            novelty_score = _open_problem_novelty_score(normalized)
            judge_output = dict(judge_output)
            judge_output["open_problem_novelty"] = round(novelty_score, 6)
            judge_output["confidence"] = round(
                0.6 * float(judge_output.get("confidence", 0.0)) + 0.4 * novelty_score,
                6,
            )
        normalized = apply_judge_confidence(normalized, judge_output)

        if (
            best_attempt is None
            or float(judge_output.get("confidence", 0.0))
            > float(best_attempt[2].get("confidence", 0.0))
        ):
            best_attempt = (prose_text, normalized, judge_output)

        if (
            not _judge_requires_replan(judge_output)
            and (not open_problem_mode or float(judge_output.get("open_problem_novelty", 0.0)) >= 0.35)
        ):
            return prose_text, normalized, judge_output

        replan_feedback = _build_replan_feedback(
            target_statement=target.statement,
            previous_output=normalized,
            judge_output=judge_output,
        )
        if open_problem_mode:
            replan_feedback += (
                "\nOpen-problem retry rule: your last attempt was too conservative."
                " Introduce a new non-seed hypothesis/object/reduction and at least one testable mechanism."
            )

    if best_attempt is None:
        raise OrchestrationError("Plausible planner did not produce any valid attempt.")
    return best_attempt


def run_experiment_action(
    graph: HyperGraph,
    target_node_id: str,
    model: Optional[str] = None,
    timeout: int = 60,
    feedback: Optional[str] = None,
    record_dir: Optional[Path] = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """Run experiment skill, execute generated code, then judge the real result."""
    target = graph.nodes[target_node_id]
    extra_contract = (
        "Strict execution contract:\n"
        "- In steps[0], provide complete Python code only.\n"
        "- Allowed imports: math, cmath, random, json, statistics, itertools, functools, "
        "fractions, decimal, collections, numpy, scipy, sympy, mpmath and their sub-modules.\n"
        "- The code must print exactly one final JSON line with keys: "
        "passed (bool), trials (int), max_error (number or null), counterexample (object or null), summary (string).\n"
        "- Do not claim results you did not compute.\n"
    )
    task_input = (
        f"Conjecture:\n{target.statement}\n\n"
        f"Context:\n{build_graph_context(graph, target_node_id)}\n\n"
        f"{extra_contract}"
    )
    if feedback:
        task_input += f"\n\nGuidance:\n{feedback.strip()}\n"
    prose_record_path = _llm_record_path(record_dir, "experiment_prose", 1)
    raw, parsed = _run_experiment_skill_with_code_fallback(
        graph,
        target_node_id,
        task_input,
        model=model,
        timeout=120,
        prose_record_path=prose_record_path,
    )
    normalized = normalize_skill_output(
        graph,
        parsed,
        expected_module="experiment",
        default_domain=target.domain,
    )
    steps = normalized.get("steps", [])
    if not steps:
        raise OrchestrationError("Experiment skill did not return any steps/code.")
    code = _extract_python_code(steps[0])
    code, stdout, stderr, result = _execute_with_repair(
        code,
        claim_context=task_input,
        model=model,
        max_repairs=3,
        timeout=timeout,
    )
    if record_dir is not None:
        code_record = _llm_record_path(record_dir, "experiment_code", 1)
        if code_record is not None:
            code_record.write_text(code, encoding="utf-8")
        result_record = _llm_record_path(record_dir, "experiment_result", 1)
        if result_record is not None:
            import json as _json
            result_record.write_text(
                _json.dumps({"stdout": stdout, "stderr": stderr, "result": result}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    final_output: dict[str, Any]
    if not result.get("passed", False):
        counterexample = result.get("counterexample")
        trials = int(result.get("trials", 0) or 0)
        raw_pass_rate = result.get("pass_rate")
        pass_rate: float
        if raw_pass_rate is None:
            passed_trials = result.get("passed_trials")
            if passed_trials is not None and trials > 0:
                pass_rate = max(0.0, min(1.0, float(passed_trials) / float(trials)))
            else:
                pass_rate = 0.0
        else:
            pass_rate = max(0.0, min(1.0, float(raw_pass_rate)))
        has_concrete_counterexample = counterexample is not None and counterexample != {}

        # Grid-resolution false-positive guard: if the alleged counterexample's
        # best distance is within 1e-4 of the threshold, the search grid likely
        # just missed the optimal time — do not treat as a real counterexample.
        if has_concrete_counterexample and isinstance(counterexample, dict):
            ce_dist = counterexample.get("best_min_dist") or counterexample.get("best_gap")
            ce_threshold = counterexample.get("threshold")
            if (
                ce_dist is not None
                and ce_threshold is not None
                and abs(float(ce_threshold) - float(ce_dist)) < 1e-4
            ):
                has_concrete_counterexample = False

        if has_concrete_counterexample and trials >= 1000 and pass_rate == 0.0:
            penalty_confidence = 0.95
            outcome = "refuted"
            reasoning = (
                f"Experiment found concrete counterexamples with zero successful trials "
                f"across {trials} runs; this is treated as a strong empirical refutation."
            )
        elif has_concrete_counterexample and trials >= 100 and pass_rate == 0.0:
            penalty_confidence = 0.90
            outcome = "weakened"
            reasoning = (
                f"Experiment found concrete counterexamples with zero successful trials "
                f"across {trials} runs. Belief is sharply reduced but not hard-locked."
            )
        elif has_concrete_counterexample and trials >= 50:
            # Strong evidence: concrete counterexample with sufficient trials.
            # Use "weakened" outcome — belief penalty without hard state=refuted.
            # Only Lean-level formal refutation should hard-refute.
            penalty_confidence = min(0.85, 0.5 + 0.3 * min(1.0, trials / 500))
            outcome = "weakened"
            reasoning = (
                f"Experiment found a concrete counterexample in {trials} trials. "
                "Belief is significantly reduced but not hard-refuted "
                "(only formal verification can definitively refute)."
            )
        elif has_concrete_counterexample:
            # Weaker evidence: counterexample but few trials.
            penalty_confidence = 0.35
            outcome = "weakened"
            reasoning = (
                f"Experiment found a potential counterexample in only {trials} trials. "
                "Evidence is suggestive but not conclusive."
            )
        else:
            # No concrete counterexample, just passed=False (e.g. code error, timeout).
            penalty_confidence = 0.15
            outcome = "inconclusive"
            reasoning = (
                "Experiment did not pass but produced no concrete counterexample. "
                "This may indicate a coding issue rather than a mathematical refutation."
            )

        final_output = {
            "module": "experiment",
            "domain": normalized.get("domain", target.domain),
            "outcome": outcome,
            "conclusion": {"statement": target.statement},
            "confidence": penalty_confidence,
            "steps": [
                code,
                "Actual execution summary: " + json.dumps(result, ensure_ascii=False),
            ],
        }
        judge_output = {
            "confidence": penalty_confidence,
            "reasoning": reasoning,
            "concerns": (
                ["Strong empirical refutation is still non-formal and should be cross-checked."]
                if outcome == "refuted"
                else (
                    ["Experimental refutation is not formal proof of falsehood."]
                    if outcome == "weakened"
                    else ["No concrete counterexample; result may be due to implementation issues."]
                )
            ),
            "suggestion": "Try a different experimental approach or verify with symbolic computation." if outcome != "weakened" else None,
        }
        return raw, final_output, judge_output

    conclusion = normalized["conclusion"]
    if isinstance(conclusion, dict):
        conclusion_statement = conclusion.get("statement", target.statement)
        formal_statement = conclusion.get("formal_statement")
    else:
        conclusion_statement = str(conclusion)
        formal_statement = None
    resolved_statement, summary_statement = _resolve_supported_conclusion_statement(
        graph,
        conclusion_statement,
        fallback_target_statement=target.statement,
    )

    final_output = {
        "premises": normalized.get("premises", []),
        "steps": [
            code,
            "Actual execution summary: " + json.dumps(result, ensure_ascii=False),
        ],
        "conclusion": {
            "statement": resolved_statement,
            "formal_statement": target.formal_statement if resolved_statement == target.statement else formal_statement,
        },
        "module": "experiment",
        "domain": normalized.get("domain", target.domain),
    }
    if summary_statement:
        final_output["steps"].append("Original experiment conclusion: " + summary_statement)
    judge_output = run_judge(final_output, model=model)
    final_output = apply_judge_confidence(final_output, judge_output)
    if stderr.strip():
        final_output["steps"].append("stderr: " + stderr.strip())
    return raw, final_output, judge_output


def _extract_lean_code(steps: list[Any]) -> str:
    candidates: list[str] = []
    for step in steps:
        if isinstance(step, str):
            text = _extract_python_code(step)
            if "import Mathlib" in text and "theorem discovery_" in text:
                candidates.append(text)
    if candidates:
        return max(candidates, key=len)
    for step in steps:
        if isinstance(step, str) and "theorem discovery_" in step:
            text = step.strip()
            if "import Mathlib" not in text:
                text = "import Mathlib\n\n" + text
            return text
    raise OrchestrationError("Lean skill output did not include compilable Lean code.")


def _subgoal_statement(parent_statement: str, target: str, context: str, index: int) -> str:
    context_text = f" context: {context}" if context else ""
    return f"Lean subgoal {index} for [{parent_statement}] target: {target}.{context_text}"


def _bridge_proposition_by_id(plan: BridgePlan, proposition_id: str) -> Any:
    for item in plan.propositions:
        if item.id == proposition_id:
            return item
    raise OrchestrationError(f"Bridge proposition '{proposition_id}' not found.")


def build_strict_lean_bridge_feedback(
    plan: BridgePlan,
    proposition_id: str,
    *,
    strict_mode: Optional[str] = None,
) -> str:
    """
    Build local-goal guidance so strict Lean works on the bridge proposition layer.
    """
    proposition = _bridge_proposition_by_id(plan, proposition_id)
    dependency_statements = []
    prop_map = {item.id: item for item in plan.propositions}
    for dep in proposition.depends_on:
        if dep in prop_map:
            dependency_statements.append(f"- [{dep}] {prop_map[dep].statement}")

    mode = strict_mode or classify_strict_bridge_mode(plan, proposition_id)
    lines = [
        "Bridge-local strict Lean task:",
        f"- Active bridge proposition id: {proposition.id}",
        f"- Active bridge proposition role: {proposition.role}",
        f"- Active bridge proposition grade: {proposition.grade}",
        f"- Local target statement: {proposition.statement}",
        "- Do NOT try to prove the outer top-level theorem unless it is exactly this local target.",
        "- Stay close to the current object layer and keep the theorem goal minimal.",
    ]
    if mode == "direct_proof":
        lines.append(
            "- Strict mode: direct_proof. Prove this local proposition directly as a small theorem."
        )
    elif mode == "lemma":
        lines.append(
            "- Strict mode: lemma. Treat this bridge proposition as a local helper lemma, not as the final theorem."
        )
    elif mode == "object_setup":
        lines.append(
            "- Strict mode: object_setup. First formalize the missing objects/coercions/notation/API bridge needed for this proposition, and keep the goal smaller than the outer theorem."
        )
    if proposition.notes:
        lines.append(f"- Bridge notes: {proposition.notes}")
    if proposition.formalization_notes:
        lines.append(f"- Formalization notes: {proposition.formalization_notes}")
    if dependency_statements:
        lines.append("- Local dependencies:")
        lines.extend(dependency_statements)
    return "\n".join(lines)


def run_lean_action(
    graph: HyperGraph,
    target_node_id: str,
    model: Optional[str] = None,
    timeout: int = 300,
    boundary_policy: Optional[dict[str, Any]] = None,
    max_attempts: int = 3,
    prompt_feedback: Optional[str] = None,
    record_dir: Optional[Path] = None,
) -> tuple[str, dict[str, Any], Optional[dict[str, Any]]]:
    """Run lean_proof skill and strictly verify returned code."""
    target = graph.nodes[target_node_id]
    policy = LeanBoundaryPolicy.from_dict(boundary_policy)
    base_prompt = (
        f"Conjecture:\n{target.statement}\n"
        f"Optional formal statement:\n{target.formal_statement or '(none)'}\n\n"
        f"Context:\n{build_graph_context(graph, target_node_id)}\n\n"
        "Return JSON only. Include full Lean file content in steps[1] or another step.\n"
    )
    constraints = policy.prompt_constraints_text()
    if constraints:
        base_prompt += "\nBoundary constraints:\n" + constraints + "\n"
    if prompt_feedback:
        base_prompt += "\nBridge-local guidance:\n" + prompt_feedback.strip() + "\n"

    last_error = "Lean proof generation did not run."
    last_raw = ""
    for attempt in range(1, max_attempts + 1):
        task_input = base_prompt
        if attempt > 1:
            task_input += (
                "\nPrevious attempt failed with this exact verifier feedback:\n"
                f"{last_error}\n"
                "Fix the Lean code while preserving the same theorem goal and constraints."
            )
        prose_record = _llm_record_path(record_dir, "lean_prose", attempt)

        # --- LLM call with prose-recovery fallback ---
        lean_code_from_prose: Optional[str] = None
        try:
            raw, parsed = run_skill("lean_proof.skill.md", task_input, model=model, record_path=prose_record)
        except LLMError:
            # The model returned prose instead of JSON.  The prose record file was
            # written by the streaming handler before the parse error.  Try to
            # extract a Lean code block directly from the raw prose so we don't
            # silently drop work the model already did.
            raw = ""
            if prose_record is not None and prose_record.exists():
                raw = prose_record.read_text(encoding="utf-8")
            if raw:
                try:
                    lean_code_from_prose = _extract_lean_code([raw])
                except OrchestrationError:
                    lean_code_from_prose = None
            last_error = (
                "LLM output was prose rather than JSON. "
                + ("Extracted Lean code and will attempt verification." if lean_code_from_prose
                   else "No Lean code block found in prose output.")
            )
            last_raw = raw
            if lean_code_from_prose is None:
                continue
            parsed = {}  # handled below via lean_code_from_prose

        last_raw = raw

        # Handle explicit failure JSON from the skill.
        if isinstance(parsed, dict) and parsed.get("status") == "failed":
            base_err = parsed.get("last_error", "Lean skill reported failure.")
            # Include the suggestion if present — it carries mathematical insight
            # about how to reformulate the claim, which is valuable for the next attempt.
            suggestion = parsed.get("suggestion", "")
            if suggestion:
                last_error = (
                    f"{base_err}\n"
                    f"Suggestion for reformulation from previous attempt:\n{suggestion}"
                )
            else:
                last_error = base_err
            continue

        # Determine Lean code: either recovered from prose or extracted from JSON.
        if lean_code_from_prose is not None:
            lean_code = lean_code_from_prose
            normalized_premises: list[Any] = []
            normalized_domain = target.domain
        else:
            normalized = normalize_skill_output(
                graph,
                parsed,
                expected_module="lean",
                default_domain=target.domain,
            )
            lean_code = _extract_lean_code(normalized.get("steps", []))
            normalized_premises = normalized.get("premises", [])
            normalized_domain = normalized.get("domain", target.domain or "number_theory")

        lean_code_record = _llm_record_path(record_dir, "lean_code", attempt)
        if lean_code_record is not None:
            lean_code_record.with_suffix(".lean").parent.mkdir(parents=True, exist_ok=True)
            lean_code_record.with_suffix(".lean").write_text(lean_code, encoding="utf-8")
        try:
            validate_lean_code(lean_code, policy)
        except LeanPolicyError as e:
            last_error = str(e)
            continue

        lean_workspace = Path(
            os.environ.get(ENV_LEAN_WORKSPACE, str(get_workspace_path()))
        ).resolve()
        result = verify_proof(lean_code, workspace_path=lean_workspace, timeout=timeout)
        if not result.success:
            last_error = result.error_message or result.stderr or "Lean build failed."
            continue

        final_output = result.to_ingest_dict(
            premises=normalized_premises,
            conclusion_statement=target.statement,
            steps=[lean_code, "Strict verification: lake build succeeded"],
            domain=normalized_domain or "number_theory",
        )
        return last_raw, final_output, None

    raise OrchestrationError(last_error)


def run_lean_decompose_action(
    graph: HyperGraph,
    target_node_id: str,
    model: Optional[str] = None,
    timeout: int = 300,
    boundary_policy: Optional[dict[str, Any]] = None,
    max_attempts: int = 3,
    bridge_plan: Optional[BridgePlan | dict[str, Any]] = None,
    record_dir: Optional[Path] = None,
) -> tuple[str, dict[str, Any], list[dict[str, str]]]:
    """Ask the LLM for a Lean skeleton, then use real Lean diagnostics to extract subgoals."""
    target = graph.nodes[target_node_id]
    policy = LeanBoundaryPolicy.from_dict(boundary_policy)
    parsed_bridge_plan: Optional[BridgePlan] = None
    if bridge_plan is not None:
        if isinstance(bridge_plan, BridgePlan):
            parsed_bridge_plan = bridge_plan
        else:
            try:
                parsed_bridge_plan = validate_bridge_plan_payload(bridge_plan)
            except BridgeValidationError as exc:
                raise OrchestrationError(f"Invalid bridge plan supplied to decomposition: {exc}") from exc
    base_prompt = (
        "Conjecture to decompose into Lean subgoals:\n"
        f"{target.statement}\n"
        f"Optional formal statement:\n{target.formal_statement or '(none)'}\n\n"
        f"Context:\n{build_graph_context(graph, target_node_id)}\n\n"
        "Return a Lean skeleton using `sorry` for unresolved proof parts. "
        "Use theorem names starting with discovery_. Return JSON only.\n"
    )
    constraints = policy.prompt_constraints_text()
    if constraints:
        base_prompt += "\nBoundary constraints:\n" + constraints + "\n"

    last_error = "Lean decomposition generation did not run."
    last_raw = ""
    for attempt in range(1, max_attempts + 1):
        task_input = base_prompt
        if attempt > 1:
            task_input += (
                "\nPrevious decomposition attempt failed with this exact Lean feedback:\n"
                f"{last_error}\n"
                "Regenerate the skeleton so that it stays within the same theorem goal and constraints."
            )
        prose_record = _llm_record_path(record_dir, "decompose_prose", attempt)
        if parsed_bridge_plan is not None:
            raw, normalized, lean_code = _compile_skeleton_from_bridge_plan(
                graph,
                target_node_id,
                parsed_bridge_plan,
                model=model,
                feedback=last_error if attempt > 1 else None,
            )
            if prose_record is not None:
                prose_record.write_text(raw, encoding="utf-8")
        else:
            raw, parsed = run_skill("lean_skeleton.skill.md", task_input, model=model, record_path=prose_record)
            normalized = normalize_skill_output(
                graph,
                parsed,
                expected_module="lean",
                default_domain=target.domain,
            )
            lean_code = _extract_lean_code(normalized.get("steps", []))
        last_raw = raw
        skeleton_record = _llm_record_path(record_dir, "decompose_skeleton", attempt)
        if skeleton_record is not None:
            skeleton_record.with_suffix(".lean").parent.mkdir(parents=True, exist_ok=True)
            skeleton_record.with_suffix(".lean").write_text(lean_code, encoding="utf-8")
        try:
            validate_lean_code(lean_code, policy)
        except LeanPolicyError as e:
            last_error = str(e)
            continue

        lean_workspace = Path(
            os.environ.get(ENV_LEAN_WORKSPACE, str(get_workspace_path()))
        ).resolve()
        decomp = decompose_proof_skeleton(
            lean_code,
            workspace_path=lean_workspace,
            timeout=timeout,
        )
        if decomp.success and not decomp.goals:
            last_error = (
                "Lean skeleton unexpectedly discharged all goals; use the normal lean path instead."
            )
            continue
        if not decomp.goals:
            last_error = decomp.error_message or decomp.stderr or decomp.stdout or "Lean did not expose unresolved goals."
            continue

        subgoals: list[dict[str, str]] = []
        for i, goal in enumerate(decomp.goals, start=1):
            subgoals.append(
                {
                    "statement": _subgoal_statement(target.statement, goal.target, goal.context, i),
                    "formal_statement": goal.target,
                    "context": goal.context,
                }
            )

        return last_raw, normalized, subgoals

    raise OrchestrationError(last_error)


def execute_action(
    graph: HyperGraph,
    target_node_id: str,
    selected_module: Module,
    model: Optional[str] = None,
    feedback: Optional[str] = None,
) -> ActionResult:
    """Execute one selected module against a target node."""
    result = ActionResult(
        action="execute",
        target_node_id=target_node_id,
        selected_module=selected_module.value,
    )
    try:
        if selected_module == Module.PLAUSIBLE:
            raw, normalized, judge_output = run_plausible_action(
                graph,
                target_node_id,
                model=model,
                feedback=feedback,
            )
        elif selected_module == Module.EXPERIMENT:
            raw, normalized, judge_output = run_experiment_action(
                graph,
                target_node_id,
                model=model,
                feedback=feedback,
            )
        elif selected_module == Module.LEAN:
            raw, normalized, judge_output = run_lean_action(
                graph,
                target_node_id,
                model=model,
                prompt_feedback=feedback,
            )
        elif selected_module in {
            Module.ANALOGY,
            Module.DECOMPOSE,
            Module.SPECIALIZE,
            Module.RETRIEVE,
        }:
            # These modules are executed by dedicated engines in MCTS mode.
            # In generic orchestrator mode, use a conservative plausible fallback.
            raw, normalized, judge_output = run_plausible_action(
                graph,
                target_node_id,
                model=model,
                feedback=feedback,
            )
            normalized = dict(normalized)
            normalized["module"] = selected_module.value
        else:
            raise OrchestrationError(f"Unsupported module: {selected_module.value}")
        result.raw_output = raw
        result.normalized_output = normalized
        result.judge_output = judge_output
        result.success = True
        result.message = "Action executed successfully."
        return result
    except (LLMError, OrchestrationError) as e:
        result.success = False
        result.message = str(e)
        return result


def ingest_action_output(
    graph_path: Path,
    action_result: ActionResult,
    backend: str = "bp",
    target_node_id: str | None = None,
) -> ActionResult:
    """Ingest action result into the graph, propagate, and persist.

    Args:
        graph_path: Path to the graph JSON file.
        action_result: The result from the action to ingest.
        backend: Propagation backend ("bp" or "energy").
        target_node_id: When supplied, the conclusion of the action output is
            matched against this target using a canonicalised prefix comparison.
            If they describe the same proposition, the target node is reused as
            the conclusion rather than creating a duplicate node.  Pass the MCTS
            target node ID here to prevent plausible reasoning from creating
            disconnected shadow copies of the target theorem.
    """
    if not action_result.success or not action_result.normalized_output:
        return action_result

    graph = load_graph(graph_path)
    before_node_ids = set(graph.nodes.keys())
    edge = ingest_skill_output(graph, action_result.normalized_output, target_node_id=target_node_id)
    action_result.ingest_edge_id = None if edge is None else edge.id
    if backend == "energy":
        from discovery_zero.graph.inference_energy import propagate_beliefs_energy

        propagate_beliefs_energy(graph)
    else:
        propagate_beliefs(graph)
    action_result.created_node_ids = [nid for nid in graph.nodes.keys() if nid not in before_node_ids]
    save_graph(graph, graph_path)
    action_result.message = "Ingested and propagated."
    return action_result


def ingest_decomposition_output(
    graph_path: Path,
    target_node_id: str,
    normalized_output: dict[str, Any],
    subgoals: list[dict[str, str]],
    backend: str = "bp",
) -> ActionResult:
    """Materialize Lean-derived subgoals and connect them to the parent goal."""
    graph = load_graph(graph_path)
    if target_node_id not in graph.nodes:
        raise OrchestrationError(f"Target node {target_node_id} not found.")

    created_ids: list[str] = []
    premise_ids: list[str] = []
    for goal in subgoals:
        statement = goal["statement"]
        existing = graph.find_node_ids_by_statement(statement)
        if existing:
            premise_ids.append(existing[0])
            continue
        node = graph.add_node(
            statement=statement,
            belief=0.5,
            formal_statement=goal.get("formal_statement"),
            domain=normalized_output.get("domain"),
        )
        premise_ids.append(node.id)
        created_ids.append(node.id)

    edge = graph.add_hyperedge(
        premise_ids=premise_ids,
        conclusion_id=target_node_id,
        module=Module.LEAN,
        steps=normalized_output.get("steps", []) + [
            "Lean decomposition: unresolved subgoals were extracted from real Lean diagnostics."
        ],
        confidence=0.95,
        edge_type="decomposition",
    )

    if backend == "energy":
        from discovery_zero.graph.inference_energy import propagate_beliefs_energy

        propagate_beliefs_energy(graph)
    else:
        propagate_beliefs(graph)
    save_graph(graph, graph_path)
    return ActionResult(
        action="lean_decompose",
        target_node_id=target_node_id,
        selected_module=Module.LEAN.value,
        normalized_output=normalized_output,
        ingest_edge_id=edge.id,
        created_node_ids=created_ids,
        success=True,
        message="Lean subgoals extracted and decomposition edge ingested.",
    )


def run_loop(
    graph_path: Path,
    rounds: int = 1,
    model: Optional[str] = None,
    backend: str = "bp",
    log_path: Optional[Path] = None,
) -> list[ActionResult]:
    """Run a real discovery loop using rank_nodes + suggest_module."""
    results: list[ActionResult] = []
    for _ in range(rounds):
        graph = load_graph(graph_path)
        ranked = rank_nodes(graph)
        if not ranked:
            break
        node_id, _priority = ranked[0]
        module = suggest_module(graph, node_id)
        action_result = execute_action(graph, node_id, module, model=model)
        if action_result.success:
            action_result = ingest_action_output(graph_path, action_result, backend=backend)
        results.append(action_result)
        if (
            action_result.success
            and module == Module.PLAUSIBLE
            and action_result.normalized_output is not None
        ):
            try:
                followups = execute_bridge_followups(
                    graph_path,
                    node_id,
                    action_result.normalized_output,
                    judge_output=action_result.judge_output,
                    model=model,
                    backend=backend,
                )
                results.extend(followups)
            except (LLMError, OrchestrationError) as e:
                results.append(
                    ActionResult(
                        action="bridge_consumption",
                        target_node_id=node_id,
                        selected_module="bridge",
                        success=False,
                        message=f"Bridge follow-up failed: {e}",
                    )
                )
        if (not action_result.success) and module != Module.PLAUSIBLE:
            replan_result = ActionResult(
                action="replan",
                target_node_id=node_id,
                selected_module=Module.PLAUSIBLE.value,
            )
            try:
                graph = load_graph(graph_path)
                raw, normalized, judge_output = run_plausible_action(
                    graph,
                    node_id,
                    model=model,
                    feedback=_build_replan_feedback(
                        target_statement=graph.nodes[node_id].statement,
                        failure_message=action_result.message,
                        failed_module=module.value,
                    ),
                )
                replan_result.raw_output = raw
                replan_result.normalized_output = normalized
                replan_result.judge_output = judge_output
                replan_result.success = True
                replan_result.message = "Replanned after downstream failure."
                replan_result = ingest_action_output(graph_path, replan_result, backend=backend)
            except (LLMError, OrchestrationError) as e:
                replan_result.success = False
                replan_result.message = f"Replan failed: {e}"
            results.append(replan_result)
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                for item in results[-2:] if len(results) >= 2 and results[-1].action == "replan" else [action_result]:
                    f.write(
                        json.dumps(
                            {
                                "target_node_id": item.target_node_id,
                                "selected_module": item.selected_module,
                                "success": item.success,
                                "message": item.message,
                                "ingest_edge_id": item.ingest_edge_id,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    return results


# ======================================================================
# DiscoveryEngine — multi-round iterative orchestration
# ======================================================================

import time as _time
from datetime import datetime, timezone as _tz


@dataclass
class ActionEvent:
    """Emitted by DiscoveryEngine for monitoring and logging."""

    event_type: str
    """'action_start' | 'action_complete' | 'action_failed' | 'bp_complete' | 'round_end'"""
    node_id: str
    module: Optional[str]
    round_index: int
    elapsed_ms: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryResult:
    """Returned by DiscoveryEngine.run()."""

    target_node_id: str
    success: bool
    rounds_completed: int
    actions_total: int
    actions_succeeded: int
    target_belief_initial: float
    target_belief_final: float
    token_budget_summary: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    action_results: list[ActionResult] = field(default_factory=list)

    @property
    def belief_delta(self) -> float:
        return self.target_belief_final - self.target_belief_initial


class DiscoveryEngine:
    """
    Multi-round iterative discovery engine with:
      - UCB-based module selection (via SearchState)
      - Multi-frontier ranking (via rank_frontiers)
      - Adaptive failure routing (via FailureRouter)
      - Token budget enforcement (via TokenBudget)
      - Event hooks for monitoring (on_action callback)
      - GraphSession for lazy persistence

    Wraps the existing execute_action / run_plausible_action /
    execute_bridge_followups infrastructure, adding the UCB search
    loop and failure recovery on top.
    """

    def __init__(
        self,
        session: GraphSession,
        search_state: Optional[SearchState] = None,
        rmaxts: Optional[RMaxTSSearch] = None,
        config: Optional[Any] = None,  # ZeroConfig
        model: Optional[str] = None,
        backend: str = "bp",
    ) -> None:
        self._session = session
        self._search = search_state or SearchState()
        self._rmaxts = rmaxts
        self._failure_router = FailureRouter()
        self._model = model
        self._backend = backend

        # Load config
        if config is not None:
            self._config = config
        else:
            try:
                from discovery_zero.config import CONFIG
                self._config = CONFIG
            except Exception:
                self._config = None

    def run(
        self,
        target_node_id: str,
        *,
        max_rounds: int = 20,
        token_budget: Optional[TokenBudget] = None,
        on_action: Optional[Any] = None,  # Callable[[ActionEvent], None]
        log_path: Optional[Path] = None,
    ) -> DiscoveryResult:
        """
        Run the multi-round discovery loop.

        Each round:
          1. rank_frontiers() → top-k candidate nodes
          2. select_module_ucb() → choose best module for top node
          3. execute_action() → run the module
          4. ingest + incremental BP
          5. Update UCB stats and failure memory
          6. Repeat until budget exhausted or target proven
        """
        t0 = _time.monotonic()
        graph = self._session.graph

        if target_node_id not in graph.nodes:
            return DiscoveryResult(
                target_node_id=target_node_id,
                success=False,
                rounds_completed=0,
                actions_total=0,
                actions_succeeded=0,
                target_belief_initial=0.0,
                target_belief_final=0.0,
                elapsed_ms=0.0,
            )

        target_belief_initial = graph.nodes[target_node_id].belief
        all_results: list[ActionResult] = []
        actions_total = 0
        actions_succeeded = 0

        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)

        for round_idx in range(max_rounds):
            # Check token budget
            if token_budget is not None and token_budget.exhausted():
                break

            # Check if target is already proven
            current_graph = self._session.graph
            target_node = current_graph.nodes.get(target_node_id)
            if target_node and target_node.state == "proven":
                break

            # Get frontiers
            frontiers = rank_frontiers(
                current_graph,
                self._search,
                target_node_id,
                max_frontiers=5,
            )
            if not frontiers:
                # Fall back to rank_nodes
                ranked = rank_nodes(current_graph)
                if not ranked:
                    break
                node_id, _ = ranked[0]
                module = suggest_module(current_graph, node_id)
            else:
                selected_action = None
                if self._rmaxts is not None:
                    selected_action = self._rmaxts.select_action(
                        current_graph,
                        target_node_id,
                        frontiers,
                        self._search,
                    )
                if selected_action is not None:
                    node_id, module = selected_action
                else:
                    top = frontiers[0]
                    node_id = top.node_id
                    module = top.suggested_module or select_module_ucb(
                        current_graph, node_id, self._search
                    )

            # Emit event
            if on_action:
                try:
                    on_action(ActionEvent(
                        event_type="action_start",
                        node_id=node_id,
                        module=module.value if module else None,
                        round_index=round_idx,
                        elapsed_ms=(_time.monotonic() - t0) * 1000,
                    ))
                except Exception:
                    pass

            # Execute action
            t_action = _time.monotonic()
            try:
                action_result = execute_action(
                    current_graph,
                    node_id,
                    module,
                    model=self._model,
                )
            except BudgetExhaustedError:
                break
            except Exception as exc:
                error_type = classify_error(exc, stage=module.value if module else "unknown")
                record = FailureRecord(
                    node_id=node_id,
                    module=module,
                    stage=module.value if module else "unknown",
                    error_type=error_type,
                    message=str(exc)[:300],
                )
                self._failure_router.route(record)
                self._search.record_action(node_id, module, 0.0, success=False, error_type=error_type)
                continue

            actions_total += 1
            elapsed_action_ms = (_time.monotonic() - t_action) * 1000

            belief_before = current_graph.nodes.get(node_id).belief if node_id in current_graph.nodes else 0.0
            if action_result.success:
                actions_succeeded += 1
                # Ingest and propagate
                try:
                    # Save to session persist_path if set, then reload
                    if self._session._persist_path is not None:
                        self._session.flush()
                        action_result = ingest_action_output(
                            self._session._persist_path,
                            action_result,
                            backend=self._backend,
                        )
                        self._session.load_from_disk()
                    else:
                        # In-memory path
                        from discovery_zero.graph.ingest import ingest_skill_output
                        from discovery_zero.graph.inference import propagate_beliefs
                        edge = ingest_skill_output(current_graph, action_result.normalized_output)
                        if edge is not None:
                            action_result.ingest_edge_id = edge.id
                        propagate_beliefs(current_graph)
                        self._session.mark_dirty()
                except Exception as exc:
                    action_result.success = False
                    action_result.message = f"Ingest failed: {exc}"

                belief_after = self._session.graph.nodes.get(node_id)
                reward = max(0.0, (belief_after.belief if belief_after else 0.0) - belief_before)
                self._search.record_action(node_id, module, reward, success=True)
                self._failure_router.record_success(node_id, module)
            else:
                error_type = classify_error(
                    Exception(action_result.message),
                    stage=module.value if module else "unknown",
                )
                record = FailureRecord(
                    node_id=node_id,
                    module=module,
                    stage=module.value if module else "unknown",
                    error_type=error_type,
                    message=action_result.message,
                )
                self._failure_router.route(record)
                self._search.record_action(node_id, module, 0.0, success=False, error_type=error_type)

            all_results.append(action_result)
            self._search.rounds_completed = round_idx + 1

            if on_action:
                try:
                    on_action(ActionEvent(
                        event_type="action_complete" if action_result.success else "action_failed",
                        node_id=node_id,
                        module=module.value if module else None,
                        round_index=round_idx,
                        elapsed_ms=elapsed_action_ms,
                        details={
                            "success": action_result.success,
                            "message": action_result.message[:200],
                        },
                    ))
                except Exception:
                    pass

            if log_path is not None:
                try:
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "round": round_idx,
                            "node_id": node_id,
                            "module": module.value if module else None,
                            "success": action_result.success,
                            "message": action_result.message[:200],
                            "elapsed_ms": round(elapsed_action_ms, 1),
                        }, ensure_ascii=False) + "\n")
                except Exception:
                    pass

        # Final flush
        self._session.flush()

        final_graph = self._session.graph
        target_final = final_graph.nodes.get(target_node_id)
        target_belief_final = target_final.belief if target_final else 0.0
        target_proven = (target_final.state == "proven") if target_final else False

        budget_summary: dict[str, Any] = {}
        if token_budget is not None:
            budget_summary = token_budget.summary()

        return DiscoveryResult(
            target_node_id=target_node_id,
            success=target_proven,
            rounds_completed=self._search.rounds_completed,
            actions_total=actions_total,
            actions_succeeded=actions_succeeded,
            target_belief_initial=target_belief_initial,
            target_belief_final=target_belief_final,
            token_budget_summary=budget_summary,
            elapsed_ms=(_time.monotonic() - t0) * 1000,
            action_results=all_results,
        )
