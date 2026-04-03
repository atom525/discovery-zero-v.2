"""
Production-grade benchmark harness for Discovery Zero.

This module runs real benchmark suites against existing theorem workspaces,
stores isolated artifacts per run, and aggregates only factual metrics derived
from those artifacts. It does not simulate results or inject default passes.
"""

from __future__ import annotations

import json
import math
import os
import signal
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable, Optional

from dz_engine.bridge import BridgePlan, materialize_bridge_nodes, validate_bridge_plan_payload
from dz_hypergraph.inference import propagate_beliefs
from dz_hypergraph.inference_energy import propagate_beliefs_energy
from dz_hypergraph.models import HyperGraph, Module
from dz_engine.orchestrator import (
    ActionResult,
    _build_replan_feedback,
    _proposition_supported_in_graph,
    build_strict_lean_bridge_feedback,
    ingest_action_output,
    ingest_decomposition_output,
    plan_bridge_consumption,
    run_bridge_planning_action,
    run_experiment_action,
    run_lean_action,
    run_lean_decompose_action,
    run_plausible_action,
    select_ready_bridge_proposition,
)
from dz_hypergraph.persistence import load_graph, save_graph
from dz_hypergraph.config import CONFIG
from dz_verify.continuation_verifier import ContinuationConfig, ContinuationVerifier
from dz_engine.curiosity import CuriosityConfig, CuriosityDrivenExplorer, NoveltyTracker
from dz_engine.experiment_evolution import EvolutionConfig, ExperimentEvolver
from dz_engine.expert_iteration import ExperienceBuffer, ExpertIterationLoop
from dz_engine.analogy import AnalogyEngine
from dz_verify.claim_verifier import ClaimVerifier
from dz_engine.decompose import DecomposeEngine
from dz_engine.knowledge_retrieval import KnowledgeRetriever
from dz_verify.claim_pipeline import ClaimPipeline, ClaimPipelineConfig
from dz_verify.lean_feedback import LeanFeedbackParser, StructuralClaimRouter
from dz_engine.mcts_engine import MCTSConfig, MCTSDiscoveryEngine
from dz_engine.problem_variants import ProblemVariantGenerator
from dz_engine.specialize import SpecializeEngine
from dz_hypergraph.inference import SignalAccumulator
from dz_engine.value_net import ProcessAdvantageVerifier
from dz_hypergraph.tools.external_prm import ExternalPRM, ExternalPRMConfig
from dz_hypergraph.tools.gaia_client import build_gaia_client
from dz_hypergraph.tools.retrieval import HypergraphRetrievalIndex, RetrievalConfig
from dz_hypergraph.tools.lean import get_workspace_path, prepare_benchmark_lean_sandbox


DEFAULT_EVALUATION_ROOT = Path(__file__).resolve().parent.parent.parent / "evaluation"
DEFAULT_TIMEOUTS = {
    "experiment": 120,
    "decompose": 180,
    "lean": 300,
}
DEFAULT_LEAN_POLICY = {
    "mode": "selective",
    "enable_decomposition": False,
    "enable_strict_lean": True,
    "min_path_confidence": 0.85,
    "max_grade_d_ratio": 0.15,
    "allowed_strict_modes": ["direct_proof", "lemma"],
}
LOCALIZATION_NEGATIVE_HINTS = (
    "timed out",
    "timeout",
    "no output",
    "produced no output",
    "failed to reach llm endpoint",
    "invalid json",
)
LOCALIZATION_POSITIVE_HINTS = (
    "expected token",
    "function expected at",
    "unknown constant",
    "type mismatch",
    "ambiguous",
    "subgoal",
    "goal",
    "counterexample",
    "final json line",
    "bridge-prop",
    "placeholder",
    "sorry",
    "admit",
    "line ",
    "column ",
)
EXPERIMENT_EXACT_HINTS = (
    "fractions",
    "fraction(",
    "decimal",
    "exact arithmetic",
    "rational",
)
EXPERIMENT_CROSSCHECK_HINTS = (
    "crosscheck",
    "cross-check",
    "independent",
    "second method",
    "two methods",
    "compare",
    "faddeev",
    "interpolation",
)


class BenchmarkError(RuntimeError):
    """Raised when benchmark configuration or execution is invalid."""


@dataclass(slots=True)
class BenchmarkCaseConfig:
    case_id: str
    display_name: str
    source_proof_config: Path
    benchmark_scope: str
    repeats: Optional[int] = None
    tags: list[str] = field(default_factory=list)
    allow_refutation: bool = False
    timeouts: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_TIMEOUTS))
    model: Optional[str] = None
    lean_policy: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_LEAN_POLICY))
    planning_constraints: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BenchmarkSuiteConfig:
    suite_id: str
    display_name: str
    description: str
    case_files: list[Path]
    repeats: int = 5
    run_mode: str = "serial"


@dataclass(slots=True)
class SuiteRunResult:
    suite_run_dir: Path
    suite_summary_path: Path
    suite_scorecard_path: Path


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp_slug() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(base: Path, raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def load_case_config(path: Path) -> BenchmarkCaseConfig:
    raw = _read_json(path)
    timeouts = dict(DEFAULT_TIMEOUTS)
    timeouts.update(raw.get("timeouts", {}))
    lean_policy = dict(DEFAULT_LEAN_POLICY)
    lean_policy.update(raw.get("lean_policy", {}))
    case = BenchmarkCaseConfig(
        case_id=raw["case_id"],
        display_name=raw["display_name"],
        source_proof_config=_resolve_path(path.parent, raw["source_proof_config"]),
        benchmark_scope=raw["benchmark_scope"],
        repeats=raw.get("repeats"),
        tags=list(raw.get("tags", [])),
        allow_refutation=bool(raw.get("allow_refutation", False)),
        timeouts={
            "experiment": int(timeouts["experiment"]),
            "decompose": int(timeouts["decompose"]),
            "lean": int(timeouts["lean"]),
        },
        model=raw.get("model"),
        lean_policy={
            "mode": str(lean_policy["mode"]),
            "enable_decomposition": bool(lean_policy["enable_decomposition"]),
            "enable_strict_lean": bool(lean_policy["enable_strict_lean"]),
            "min_path_confidence": float(lean_policy["min_path_confidence"]),
            "max_grade_d_ratio": float(lean_policy["max_grade_d_ratio"]),
            "allowed_strict_modes": list(lean_policy["allowed_strict_modes"]),
        },
        planning_constraints=list(raw.get("planning_constraints", [])),
    )
    if not case.source_proof_config.exists():
        raise BenchmarkError(f"Case source config not found: {case.source_proof_config}")
    return case


def load_suite_config(path: Path) -> BenchmarkSuiteConfig:
    raw = _read_json(path)
    case_files = [_resolve_path(path.parent, item) for item in raw["cases"]]
    suite = BenchmarkSuiteConfig(
        suite_id=raw["suite_id"],
        display_name=raw["display_name"],
        description=raw["description"],
        case_files=case_files,
        repeats=int(raw.get("repeats", 5)),
        run_mode=raw.get("run_mode", "serial"),
    )
    if suite.run_mode != "serial":
        raise BenchmarkError("Only serial benchmark execution is supported.")
    for case_file in suite.case_files:
        if not case_file.exists():
            raise BenchmarkError(f"Suite case file not found: {case_file}")
    return suite


def _unique_directory(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = 1
    while True:
        candidate = path.with_name(f"{path.name}-{suffix:02d}")
        if not candidate.exists():
            return candidate
        suffix += 1


def _copy_resolved_config(source_config: dict[str, Any], run_dir: Path, case: BenchmarkCaseConfig) -> dict[str, Any]:
    resolved = json.loads(json.dumps(source_config))
    resolved["workspace"] = str(run_dir)
    if case.model:
        resolved["model"] = case.model
    resolved["experiment_timeout"] = case.timeouts["experiment"]
    resolved["decompose_timeout"] = case.timeouts["decompose"]
    resolved["lean_timeout"] = case.timeouts["lean"]
    resolved["benchmark_scope"] = case.benchmark_scope
    resolved["planning_constraints"] = list(case.planning_constraints)
    return resolved


def _prepare_benchmark_lean_workspace(resolved_config: dict[str, Any], run_dir: Path) -> tuple[Path, bool]:
    """
    Resolve the Lean workspace for this benchmark run.

    If ``proof_config`` sets ``lean_workspace``, that path is used as-is (shared;
    caller may backup/restore ``Proofs.lean``). If it is null, creates
    ``run_dir / "lean_workspace"`` as an isolated sandbox sharing ``.lake/packages``
    with the project template from ``get_workspace_path()`` (ignores
    ``DISCOVERY_ZERO_LEAN_WORKSPACE`` so parallel runs do not nest sandboxes).

    Returns:
        (workspace_path, should_backup_proofs)
    """
    explicit = resolved_config.get("lean_workspace")
    if explicit:
        return Path(explicit).resolve(), True
    sandbox = (run_dir / "lean_workspace").resolve()
    prepare_benchmark_lean_sandbox(get_workspace_path(), sandbox)
    resolved_config["lean_workspace"] = str(sandbox)
    return sandbox, False


def _snapshot(graph: HyperGraph, step: str) -> dict[str, Any]:
    return {
        "step": step,
        "nodes": {
            nid: {
                "prior": round(node.prior, 6),
                "belief": round(node.belief, 6),
                "state": node.state,
                "statement": node.statement,
            }
            for nid, node in graph.nodes.items()
        },
        "edges": {
            eid: {
                "module": edge.module.value,
                "edge_type": edge.edge_type,
                "confidence": edge.confidence,
                "conclusion_id": edge.conclusion_id,
                "premise_ids": edge.premise_ids,
            }
            for eid, edge in graph.edges.items()
        },
    }


def _detect_placeholder_guard(raw_output: Optional[str], success: bool) -> int:
    text = (raw_output or "").casefold()
    if "sorry" in text or "admit" in text:
        return 0 if success else 1
    return 1


def _extract_actual_execution_summary(step_values: Iterable[Any]) -> dict[str, Any] | None:
    for value in step_values:
        if not isinstance(value, str):
            continue
        prefix = "Actual execution summary: "
        if value.startswith(prefix):
            payload = value[len(prefix):].strip()
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                return None
            if isinstance(parsed, dict):
                return parsed
    return None


def _contains_any(text: str, hints: Iterable[str]) -> bool:
    folded = text.casefold()
    return any(item in folded for item in hints)


def _classify_failure_localization(message: Optional[str]) -> int:
    if not message:
        return 0
    folded = message.casefold()
    if any(item in folded for item in LOCALIZATION_NEGATIVE_HINTS):
        return 0
    if any(item in folded for item in LOCALIZATION_POSITIVE_HINTS):
        return 1
    return 1 if len(folded) >= 24 else 0


def _safe_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _safe_ratio(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _grade_counts_from_plan(plan: BridgePlan | None) -> tuple[int, int, int, int, int]:
    if plan is None:
        return 0, 0, 0, 0, 0
    a = sum(1 for item in plan.propositions if item.grade == "A")
    b = sum(1 for item in plan.propositions if item.grade == "B")
    c = sum(1 for item in plan.propositions if item.grade == "C")
    d = sum(1 for item in plan.propositions if item.grade == "D")
    return a, b, c, d, len(plan.propositions)


def _resolve_best_path_candidate(steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the plausible/replan step with highest judge confidence."""
    initial_plausible = next((item for item in steps if item.get("phase") == "plausible"), None)
    replan_steps = [item for item in steps if str(item.get("phase", "")).startswith("plausible_replan")]
    candidates = ([initial_plausible] if initial_plausible else []) + replan_steps
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda s: float((s.get("judge") or {}).get("confidence", 0.0)),
    )


def should_attempt_lean(
    case: BenchmarkCaseConfig,
    *,
    best_path_confidence: float,
    grade_d_ratio: float,
    strict_mode: str | None,
    has_decomposition_plan: bool,
    has_strict_target: bool,
) -> dict[str, Any]:
    policy = case.lean_policy
    mode = str(policy.get("mode", "selective"))
    if mode == "always":
        return {
            "attempt_decomposition": bool(policy.get("enable_decomposition", True)) and has_decomposition_plan,
            "attempt_strict_lean": bool(policy.get("enable_strict_lean", True)) and has_strict_target,
            "decomposition_reason": "Lean policy set to always.",
            "strict_lean_reason": "Lean policy set to always.",
        }

    min_path_confidence = float(policy.get("min_path_confidence", DEFAULT_LEAN_POLICY["min_path_confidence"]))
    max_grade_d_ratio = float(policy.get("max_grade_d_ratio", DEFAULT_LEAN_POLICY["max_grade_d_ratio"]))
    allowed_strict_modes = set(policy.get("allowed_strict_modes", DEFAULT_LEAN_POLICY["allowed_strict_modes"]))
    enable_decomposition = bool(policy.get("enable_decomposition", False))
    enable_strict_lean = bool(policy.get("enable_strict_lean", True))

    decomposition_reasons: list[str] = []
    strict_reasons: list[str] = []
    attempt_decomposition = enable_decomposition and has_decomposition_plan
    attempt_strict = enable_strict_lean and has_strict_target

    if best_path_confidence < min_path_confidence:
        attempt_decomposition = False
        attempt_strict = False
        reason = (
            f"Best path confidence {best_path_confidence:.3f} is below selective Lean threshold "
            f"{min_path_confidence:.3f}."
        )
        decomposition_reasons.append(reason)
        strict_reasons.append(reason)
    if grade_d_ratio > max_grade_d_ratio:
        attempt_decomposition = False
        attempt_strict = False
        reason = (
            f"Bridge D-grade ratio {grade_d_ratio:.3f} exceeds selective Lean threshold "
            f"{max_grade_d_ratio:.3f}."
        )
        decomposition_reasons.append(reason)
        strict_reasons.append(reason)
    if not has_decomposition_plan:
        attempt_decomposition = False
        decomposition_reasons.append("No decomposition bridge plan was selected.")
    if not enable_decomposition:
        attempt_decomposition = False
        decomposition_reasons.append(
            "Lean subgoal decomposition is disabled (lean_policy.enable_decomposition is false)."
        )
    if not has_strict_target:
        attempt_strict = False
        strict_reasons.append("No strict Lean target proposition was selected.")
    if not enable_strict_lean:
        attempt_strict = False
        strict_reasons.append("Strict Lean is disabled by case policy.")
    if strict_mode is None:
        attempt_strict = False
        strict_reasons.append("Strict Lean mode could not be determined.")
    elif strict_mode not in allowed_strict_modes:
        attempt_strict = False
        strict_reasons.append(
            f"Strict Lean mode `{strict_mode}` is outside the selective allowlist {sorted(allowed_strict_modes)}."
        )

    if attempt_decomposition and not decomposition_reasons:
        decomposition_reasons.append("Selective Lean gate passed for decomposition.")
    if attempt_strict and not strict_reasons:
        strict_reasons.append("Selective Lean gate passed for strict Lean.")
    return {
        "attempt_decomposition": attempt_decomposition,
        "attempt_strict_lean": attempt_strict,
        "decomposition_reason": " ".join(decomposition_reasons),
        "strict_lean_reason": " ".join(strict_reasons),
    }


def _compute_indices(metrics: dict[str, Any]) -> dict[str, float]:
    pqi = (
        35 * float(metrics["best_path_judge_confidence"])
        + 25 * float(metrics["grade_ab_ratio"])
        + 15 * min(float(metrics["best_path_step_count"]), 8.0) / 8.0
        + 15 * min(float(metrics["bridge_step_count"]), 4.0) / 4.0
        + 10 * (1.0 - float(metrics["grade_d_ratio"]))
    )
    fri = (
        30 * float(metrics["grade_a_ratio"])
        + 20 * float(metrics["grade_ab_ratio"])
        + 15 * float(metrics["lean_decompose_success"])
        + 10 * min(float(metrics["lean_subgoal_count"]), 4.0) / 4.0
        + 15 * float(metrics["first_failure_is_localized"])
        + 10 * float(metrics["placeholder_rejected_correctly"])
    )
    esi = (
        25 * float(metrics["experiment_ran"])
        + 20 * float(metrics["experiment_exact"])
        + 20 * min(float(metrics["experiment_trials"]), 500.0) / 500.0
        + 20 * (1.0 - float(metrics["experiment_found_counterexample"]))
        + 15 * float(metrics["experiment_independent_crosscheck"])
    )
    pqi *= 100.0 / 100.0
    fri *= 100.0 / 100.0
    esi *= 100.0 / 100.0
    odb = 0.4 * pqi + 0.35 * fri + 0.25 * esi

    # --- Frontier exploration indices (open/unsolved problems) ---
    # DBI: Discovery Breadth Index — knowledge gain beyond the seed graph
    dbi = (
        40 * min(float(metrics.get("new_nodes_created", 0)) / 12.0, 1.0)
        + 30 * float(metrics.get("max_new_node_belief", 0.0))
        + 30 * float(metrics["grade_ab_ratio"])
    )
    # ORI: Obstacle Recognition Index — did Zero locate where the problem is hard?
    ori = (
        50 * float(metrics["first_failure_is_localized"])
        + 30 * min(float(metrics["replan_triggered"]), 1.0)
        + 20 * float(metrics["bridge_plan_valid"])
    )
    # NVI: Novelty Index — did Zero propose approaches beyond the seeds?
    nvi = (
        40 * float(metrics["best_path_judge_confidence"])
        + 35 * (1.0 if float(metrics.get("replan_improved_confidence", 0)) > 0 else 0.0)
        + 25 * min(float(metrics["bridge_step_count"]) / 5.0, 1.0)
    )
    # OFC: Open Frontier Composite — primary score for open research problems
    ofc = 0.35 * dbi + 0.35 * ori + 0.30 * nvi

    return {
        "PQI": round(pqi, 4),
        "FRI": round(fri, 4),
        "ESI": round(esi, 4),
        "ODB": round(odb, 4),
        "DBI": round(dbi, 4),
        "ORI": round(ori, 4),
        "NVI": round(nvi, 4),
        "OFC": round(ofc, 4),
    }


def _step_reason(steps: list[dict[str, Any]], phase: str) -> str | None:
    step = next((item for item in steps if item.get("phase") == phase), None)
    if step is None:
        return None
    reason = step.get("reason")
    if isinstance(reason, str) and reason.strip():
        return reason.strip()
    return None


def _unique_nonempty_texts(values: Iterable[str | None]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value is None:
            continue
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _build_case_planning_feedback(case: BenchmarkCaseConfig) -> str | None:
    constraints = _unique_nonempty_texts(case.planning_constraints)
    open_problem_mode = any(
        tag in {"open-problem", "frontier", "frontier-assisted"}
        for tag in case.tags
    )
    if not constraints and not open_problem_mode:
        return None
    lines = [
        "Benchmark knowledge-boundary requirements:",
        "- Plan the route from the provided seed facts and target only.",
        "- Do not treat the target theorem itself, equivalent named theorems, or later textbook results as directly available facts.",
        "- Do not inject post-hoc closure or theorem shortcuts; derive intermediate lemmas explicitly if needed.",
    ]
    if open_problem_mode:
        lines.extend(
            [
                "- This is an open-problem benchmark: do not stop at seed summarization or literature-style gap listing.",
                "- Propose new intermediate objects, conjectures, reductions, or mechanisms that are not already explicit in the seeds.",
                "- At least one proposed route should be high-risk/high-upside and experimentally or structurally testable.",
                "- Prefer routes that create new local bridge targets over routes that only restate why the theorem is hard.",
            ]
        )
    lines.extend(f"- {item}" for item in constraints)
    return "\n".join(lines)


def _classify_current_bottleneck(
    *,
    first_failure_stage: str,
    experiment_found_counterexample: int,
    lean_decompose_skipped_by_policy: int,
    strict_lean_skipped_by_policy: int,
    steps: list[dict[str, Any]],
) -> tuple[str, str | None]:
    if first_failure_stage != "none":
        return first_failure_stage, None
    if experiment_found_counterexample:
        return "counterexample_found", "Experiment produced a concrete counterexample or failed execution summary."

    reasons = []
    decompose_reason = _step_reason(steps, "decomposition")
    strict_reason = _step_reason(steps, "strict_lean")
    reasons.extend(_unique_nonempty_texts([decompose_reason, strict_reason]))
    merged_reason = " ".join(reasons).strip() or None

    if lean_decompose_skipped_by_policy or strict_lean_skipped_by_policy:
        folded = (merged_reason or "").casefold()
        if "no validated bridge plan available" in folded:
            return "bridge_plan_unavailable", merged_reason
        if "no bridge consumption decision available" in folded:
            return "bridge_consumption_unavailable", merged_reason
        if "no strict lean target proposition was selected" in folded:
            return "no_strict_target_selected", merged_reason
        if "below selective lean threshold" in folded:
            return "lean_deferred_low_confidence", merged_reason
        if "d-grade ratio" in folded:
            return "lean_deferred_high_bridge_risk", merged_reason
        if "outside the selective allowlist" in folded:
            return "lean_deferred_mode_mismatch", merged_reason
        if "disabled by case policy" in folded or "prefers experiment/replan" in folded:
            return "lean_deferred_by_policy", merged_reason
        return "lean_deferred_by_policy", merged_reason

    return "none", None


def _classify_benchmark_outcome(
    *,
    final_target_state: str,
    allow_refutation: bool,
    experiment_found_counterexample: int,
    progress_without_closure: bool,
    bridge_plan_valid: int,
    bridge_consumption_ready: int,
) -> str:
    if final_target_state == "proven":
        return "formal_proof_success"
    if allow_refutation and final_target_state == "refuted":
        return "theorem_correction_success"
    if experiment_found_counterexample:
        return "counterexample_warning"
    if bridge_consumption_ready:
        return "bridge_consumption_ready"
    if bridge_plan_valid:
        return "bridge_plan_valid_only"
    if progress_without_closure:
        return "partial_progress"
    return "stalled"


def summarize_run(
    *,
    case: BenchmarkCaseConfig,
    run_dir: Path,
    graph_path: Path,
    log_path: Path,
    bridge_path: Path,
    repeat_index: int,
) -> dict[str, Any]:
    log = _read_json(log_path)
    graph = load_graph(graph_path)
    bridge_plan: BridgePlan | None = None
    if bridge_path.exists():
        try:
            bridge_plan = validate_bridge_plan_payload(_read_json(bridge_path))
        except Exception:
            bridge_plan = None

    steps = list(log.get("steps", []))
    node_ids = dict(log.get("node_ids", {}))
    theorem_id = node_ids.get("theorem")
    if theorem_id is None:
        try:
            resolved_config = _read_json(run_dir / "resolved_proof_config.json")
            target_key = str((resolved_config.get("target") or {}).get("key") or "").strip()
            if target_key:
                theorem_id = node_ids.get(target_key)
        except Exception:
            theorem_id = None
    theorem_node = graph.nodes.get(theorem_id) if theorem_id else None
    bridge_plan_valid = 1 if bridge_plan is not None else 0

    initial_plausible = next((item for item in steps if item.get("phase") == "plausible"), None)
    replan_steps = [item for item in steps if str(item.get("phase", "")).startswith("plausible_replan")]
    best_path_candidate = initial_plausible
    for item in replan_steps:
        if best_path_candidate is None:
            best_path_candidate = item
            continue
        current_conf = float(best_path_candidate.get("judge", {}).get("confidence", 0.0))
        item_conf = float(item.get("judge", {}).get("confidence", 0.0))
        if item_conf >= current_conf:
            best_path_candidate = item

    best_path_judge_confidence = float(
        (best_path_candidate or {}).get("judge", {}).get("confidence", 0.0)
    )
    best_path_step_count = len(
        ((best_path_candidate or {}).get("normalized") or {}).get("steps", [])
    )

    grade_a_count, grade_b_count, grade_c_count, grade_d_count, total_graded_steps = _grade_counts_from_plan(bridge_plan)
    grade_a_ratio = _safe_ratio(grade_a_count, total_graded_steps)
    grade_ab_ratio = _safe_ratio(grade_a_count + grade_b_count, total_graded_steps)
    grade_d_ratio = _safe_ratio(grade_d_count, total_graded_steps)
    bridge_step_count = 0 if bridge_plan is None else sum(
        1 for item in bridge_plan.propositions if item.role == "bridge"
    )

    experiment_step = next((item for item in steps if item.get("phase") == "experiment"), None)
    experiment_steps = ((experiment_step or {}).get("normalized") or {}).get("steps", [])
    execution_summary = _extract_actual_execution_summary(experiment_steps)
    experiment_ran = 1 if experiment_step is not None else 0
    experiment_trials = 0
    experiment_found_counterexample = 0
    if execution_summary:
        experiment_trials = int(execution_summary.get("trials") or 0)
        # Only count as "counterexample" if:
        # 1. The experiment found a concrete counterexample object, AND
        # 2. The refuted node is the MAIN TARGET (not an intermediate proposition).
        # Intermediate proposition failures are expected exploration behavior,
        # not evidence against the main conjecture.
        experiment_outcome = (experiment_step or {}).get("normalized", {}).get("outcome", "")
        experiment_target_node = (experiment_step or {}).get("target_node_id", "")
        if execution_summary.get("counterexample") and experiment_outcome == "refuted":
            # Check whether the refuted statement matches the main target.
            # If it doesn't, this is an intermediate proposition failure.
            target_statement = graph.nodes[theorem_id].statement if theorem_id in graph.nodes else ""
            refuted_statement = str(
                ((experiment_step or {}).get("normalized", {}).get("conclusion", {}) or {}).get("statement", "")
            )
            if refuted_statement == target_statement:
                experiment_found_counterexample = 1
            # else: intermediate failure — not a counterexample to the main target
        elif experiment_outcome in ("weakened", "inconclusive"):
            # Soft refutation or inconclusive: never counts as "counterexample found"
            experiment_found_counterexample = 0
    experiment_text = " ".join(str(item) for item in experiment_steps)
    experiment_exact = 1 if _contains_any(experiment_text, EXPERIMENT_EXACT_HINTS) else 0
    experiment_independent_crosscheck = 1 if _contains_any(experiment_text, EXPERIMENT_CROSSCHECK_HINTS) else 0

    bridge_consumption_step = next((item for item in steps if item.get("phase") == "bridge_consumption"), None)
    bridge_ready_step = next((item for item in steps if item.get("phase") == "bridge_ready"), None)
    bridge_consumption_ready = 1 if bridge_ready_step is not None and not bridge_ready_step.get("error") else 0
    bridge_experiment_step = next((item for item in steps if item.get("phase") == "bridge_experiment"), None)
    bridge_experiment_attempted = 1 if bridge_experiment_step is not None else 0
    bridge_experiment_success = 1 if bridge_experiment_step is not None and not bridge_experiment_step.get("error") else 0

    decompose_step = next((item for item in steps if item.get("phase") == "decomposition"), None)
    lean_decompose_attempted = 1 if decompose_step is not None and not decompose_step.get("skipped", False) else 0
    lean_subgoal_count = len((decompose_step or {}).get("subgoals", []))
    lean_decompose_success = 1 if lean_subgoal_count > 0 and not (decompose_step or {}).get("error") else 0
    lean_decompose_skipped_by_policy = 1 if (decompose_step or {}).get("skipped_by_policy") else 0

    strict_lean_step = next((item for item in steps if item.get("phase") == "strict_lean"), None)
    strict_lean_attempted = 1 if strict_lean_step is not None and not strict_lean_step.get("skipped", False) else 0
    strict_lean_success = 1 if (strict_lean_step or {}).get("success") else 0
    strict_lean_skipped_by_policy = 1 if ((strict_lean_step or {}).get("skipped_by_policy")) else 0
    placeholder_rejected_correctly = _detect_placeholder_guard(
        (strict_lean_step or {}).get("raw"),
        bool((strict_lean_step or {}).get("success")),
    ) if strict_lean_attempted else 0

    first_failure_stage = "none"
    first_failure_message = None
    for phase_name, alias in (
        ("plausible", "plausible"),
        ("experiment", "experiment"),
        ("decomposition", "lean_decompose"),
        ("strict_lean", "strict_lean"),
    ):
        phase_step = next((item for item in steps if item.get("phase") == phase_name), None)
        if phase_step is None:
            continue
        if phase_step.get("skipped", False):
            continue
        phase_error = phase_step.get("error")
        if phase_error:
            first_failure_stage = alias
            first_failure_message = str(phase_error)
            break
        if phase_name == "strict_lean" and phase_step.get("success") is False:
            first_failure_stage = alias
            first_failure_message = str(phase_step.get("error") or "strict Lean failed")
            break

    first_failure_is_localized = _classify_failure_localization(first_failure_message)
    replan_triggered = 1 if replan_steps else 0

    # --- Discovery metrics: nodes created beyond the initial seed graph ---
    seed_snapshot = next((s for s in log.get("snapshots", []) if s["step"] == "seed"), None)
    seed_node_ids: set[str] = set((seed_snapshot or {}).get("nodes", {}).keys())
    new_node_beliefs = [
        node.belief for nid, node in graph.nodes.items() if nid not in seed_node_ids
    ]
    new_nodes_created = len(new_node_beliefs)
    max_new_node_belief = round(max(new_node_beliefs), 6) if new_node_beliefs else 0.0
    high_confidence_new_nodes = sum(1 for b in new_node_beliefs if b >= 0.6)
    replan_improved_confidence = 0
    if initial_plausible and replan_steps:
        initial_conf = float(initial_plausible.get("judge", {}).get("confidence", 0.0))
        best_replan_conf = max(float(item.get("judge", {}).get("confidence", 0.0)) for item in replan_steps)
        if best_replan_conf > initial_conf + 1e-9:
            replan_improved_confidence = 1
        elif best_replan_conf + 1e-9 < initial_conf:
            replan_improved_confidence = -1

    theorem_state = theorem_node.state if theorem_node else "unverified"
    theorem_belief = theorem_node.belief if theorem_node else 0.0
    success = bool(strict_lean_success or theorem_state == "proven" or (case.allow_refutation and theorem_state == "refuted"))
    current_bottleneck, current_bottleneck_detail = _classify_current_bottleneck(
        first_failure_stage=first_failure_stage,
        experiment_found_counterexample=experiment_found_counterexample,
        lean_decompose_skipped_by_policy=lean_decompose_skipped_by_policy,
        strict_lean_skipped_by_policy=strict_lean_skipped_by_policy,
        steps=steps,
    )
    progress_without_closure = bool(
        not success
        and (
            (best_path_judge_confidence >= 0.6 and grade_ab_ratio >= 0.6)
            or (experiment_ran and not experiment_found_counterexample)
            or bridge_experiment_success
            or replan_triggered
            or lean_decompose_success
            or strict_lean_attempted
            or first_failure_is_localized
        )
    )
    benchmark_outcome = _classify_benchmark_outcome(
        final_target_state=theorem_state,
        allow_refutation=case.allow_refutation,
        experiment_found_counterexample=experiment_found_counterexample,
        progress_without_closure=progress_without_closure,
        bridge_plan_valid=bridge_plan_valid,
        bridge_consumption_ready=bridge_consumption_ready,
    )
    claims_extracted = 0
    claims_verified = 0
    claims_refuted = 0
    lean_gaps_identified = 0
    lean_gaps_resolved = 0
    verification_belief_progress = 0.0
    for step in steps:
        if not isinstance(step, dict):
            continue
        claims_extracted += int(step.get("claims_extracted", 0) or 0)
        claims_verified += int(step.get("claims_verified", 0) or 0)
        claims_refuted += int(step.get("claims_refuted", 0) or 0)
        lean_gaps_identified += int(step.get("lean_gaps_identified", 0) or 0)
        lean_gaps_resolved += int(step.get("lean_gaps_resolved", 0) or 0)
        if "belief_delta" in step:
            try:
                verification_belief_progress += float(step.get("belief_delta", 0.0) or 0.0)
            except (TypeError, ValueError):
                pass

    metrics = {
        "path_count": 1 + len(replan_steps),
        "new_nodes_created": new_nodes_created,
        "max_new_node_belief": max_new_node_belief,
        "high_confidence_new_nodes": high_confidence_new_nodes,
        "bridge_plan_valid": bridge_plan_valid,
        "bridge_consumption_ready": bridge_consumption_ready,
        "bridge_experiment_attempted": bridge_experiment_attempted,
        "bridge_experiment_success": bridge_experiment_success,
        "best_path_judge_confidence": round(best_path_judge_confidence, 6),
        "best_path_step_count": best_path_step_count,
        "bridge_step_count": bridge_step_count,
        "grade_a_count": grade_a_count,
        "grade_b_count": grade_b_count,
        "grade_c_count": grade_c_count,
        "grade_d_count": grade_d_count,
        "grade_a_ratio": round(grade_a_ratio, 6),
        "grade_ab_ratio": round(grade_ab_ratio, 6),
        "grade_d_ratio": round(grade_d_ratio, 6),
        "experiment_ran": experiment_ran,
        "experiment_exact": experiment_exact,
        "experiment_trials": experiment_trials,
        "experiment_found_counterexample": experiment_found_counterexample,
        "experiment_independent_crosscheck": experiment_independent_crosscheck,
        "lean_decompose_attempted": lean_decompose_attempted,
        "lean_decompose_success": lean_decompose_success,
        "lean_subgoal_count": lean_subgoal_count,
        "lean_decompose_skipped_by_policy": lean_decompose_skipped_by_policy,
        "strict_lean_attempted": strict_lean_attempted,
        "strict_lean_success": strict_lean_success,
        "strict_lean_skipped_by_policy": strict_lean_skipped_by_policy,
        "placeholder_rejected_correctly": placeholder_rejected_correctly,
        "first_failure_stage": first_failure_stage,
        "first_failure_is_localized": first_failure_is_localized,
        "replan_triggered": replan_triggered,
        "replan_improved_confidence": replan_improved_confidence,
        "claims_extracted_per_iteration": round(claims_extracted / max(1, len(steps)), 6),
        "claims_verified_per_iteration": round(claims_verified / max(1, len(steps)), 6),
        "claims_refuted_total": claims_refuted,
        "lean_gaps_identified": lean_gaps_identified,
        "lean_gaps_resolved": lean_gaps_resolved,
        "verification_driven_belief_progress": round(verification_belief_progress, 6),
    }
    metrics.update(_compute_indices(metrics))

    return {
        "case_id": case.case_id,
        "display_name": case.display_name,
        "benchmark_scope": case.benchmark_scope,
        "repeat_index": repeat_index,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "graph_path": str(graph_path),
        "bridge_plan_path": str(bridge_path) if bridge_path.exists() else None,
        "final_target_state": theorem_state,
        "final_target_belief": round(float(theorem_belief), 6),
        "success": success,
        "progress_without_closure": progress_without_closure,
        "strict_lean_success": bool(strict_lean_success),
        "strict_lean_skipped_by_policy": bool(strict_lean_skipped_by_policy),
        "lean_decompose_skipped_by_policy": bool(lean_decompose_skipped_by_policy),
        "first_failure_stage": first_failure_stage,
        "first_failure_message": first_failure_message,
        "benchmark_outcome": benchmark_outcome,
        "current_bottleneck": current_bottleneck,
        "current_bottleneck_detail": current_bottleneck_detail,
        "metrics": metrics,
    }


def _generate_partial_summary(
    *,
    case: BenchmarkCaseConfig,
    run_dir: Path,
    graph_path: Path,
    log_path: Path,
    bridge_path: Path,
    repeat_index: int,
    reason: str,
    error: Exception | None = None,
) -> dict[str, Any]:
    """Generate a best-effort summary even when the run crashes or is terminated."""
    try:
        summary = summarize_run(
            case=case,
            run_dir=run_dir,
            graph_path=graph_path,
            log_path=log_path,
            bridge_path=bridge_path,
            repeat_index=repeat_index,
        )
    except Exception as exc:
        graph = HyperGraph()
        if graph_path.exists():
            try:
                graph = load_graph(graph_path)
            except Exception:
                graph = HyperGraph()
        log_data: dict[str, Any] = {}
        if log_path.exists():
            try:
                log_data = _read_json(log_path)
            except Exception:
                log_data = {}
        theorem_id = str((log_data.get("node_ids") or {}).get("theorem") or "")
        theorem = graph.nodes.get(theorem_id) if theorem_id else None
        summary = {
            "case_id": case.case_id,
            "display_name": case.display_name,
            "benchmark_scope": case.benchmark_scope,
            "repeat_index": repeat_index,
            "run_dir": str(run_dir),
            "log_path": str(log_path),
            "graph_path": str(graph_path),
            "bridge_plan_path": str(bridge_path) if bridge_path.exists() else None,
            "final_target_state": theorem.state if theorem else "unverified",
            "final_target_belief": round(float(theorem.belief), 6) if theorem else 0.0,
            "success": False,
            "progress_without_closure": False,
            "strict_lean_success": False,
            "strict_lean_skipped_by_policy": False,
            "lean_decompose_skipped_by_policy": False,
            "first_failure_stage": "engine_crash",
            "first_failure_message": str(exc),
            "benchmark_outcome": "engine_crash",
            "current_bottleneck": "engine_crash",
            "current_bottleneck_detail": f"Failed to summarize run: {exc}",
            "metrics": {},
        }

    if reason != "completed":
        summary["success"] = False
        summary["benchmark_outcome"] = reason
    summary["partial_summary_reason"] = reason
    if error is not None:
        summary["partial_summary_error"] = str(error)
    return summary


def _write_scorecard_markdown(summary: dict[str, Any], path: Path) -> None:
    blocker = summary["current_bottleneck"]
    blocker_detail = summary.get("current_bottleneck_detail")
    outcome_map = {
        "formal_proof_success": "证明闭环成功",
        "theorem_correction_success": "命题纠错成功",
        "counterexample_warning": "发现可疑反例信号",
        "bridge_consumption_ready": "桥接已可继续消费",
        "bridge_plan_valid_only": "已生成有效桥接路径",
        "partial_progress": "有真实进展但未闭环",
        "stalled": "进展有限",
    }
    outcome_label = outcome_map.get(summary.get("benchmark_outcome"), "未闭环")
    lines = [
        f"# {summary['display_name']} 评测摘要",
        "",
        "## 运行概览",
        "",
        f"- `case_id`: `{summary['case_id']}`",
        f"- `repeat_index`: `{summary['repeat_index']}`",
        f"- `结果类型`: `{outcome_label}`",
        f"- `最终目标状态`: `{summary['final_target_state']}`",
        f"- `最终目标 belief`: `{summary['final_target_belief']}`",
        f"- `当前主瓶颈`: `{blocker}`",
        "",
        "## 精简指标",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
    ]
    metrics = summary["metrics"]
    for key in (
        "bridge_plan_valid",
        "bridge_consumption_ready",
        "best_path_judge_confidence",
        "path_count",
        "new_nodes_created",
        "max_new_node_belief",
        "high_confidence_new_nodes",
        "grade_a_count",
        "grade_b_count",
        "grade_c_count",
        "grade_d_count",
        "experiment_found_counterexample",
        "experiment_trials",
        "PQI",
        "FRI",
        "ESI",
        "ODB",
        "DBI",
        "ORI",
        "NVI",
        "OFC",
    ):
        lines.append(f"| `{key}` | `{metrics[key]}` |")
    if blocker_detail:
        lines.extend(
            [
                "",
                "## 瓶颈说明",
                "",
                blocker_detail,
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_case_once_mcts(
    case: BenchmarkCaseConfig,
    *,
    run_dir: Path,
    repeat_index: int,
    suite_id: str,
    experience_buffer: Optional[ExperienceBuffer] = None,
) -> dict[str, Any]:
    source_config = _read_json(case.source_proof_config)
    resolved_config = _copy_resolved_config(source_config, run_dir, case)
    run_dir.mkdir(parents=True, exist_ok=False)

    resolved_config_path = run_dir / "resolved_proof_config.json"
    graph_path = run_dir / "graph.json"
    log_path = run_dir / "exploration_log.json"
    bridge_path = run_dir / "bridge_plan.json"
    summary_path = run_dir / "summary.json"
    scorecard_path = run_dir / "PATH_BENCHMARK_SCORECARD_ZH.md"
    llm_record_dir = run_dir / "llm_records"

    lean_workspace_path, lean_backup_proofs = _prepare_benchmark_lean_workspace(resolved_config, run_dir)
    os.environ["DISCOVERY_ZERO_LEAN_WORKSPACE"] = str(lean_workspace_path)

    model = resolved_config.get("model") or os.environ.get("DISCOVERY_ZERO_LLM_MODEL") or CONFIG.llm_model
    resolved_config["model"] = model
    _save_json(resolved_config_path, resolved_config)
    boundary_policy = resolved_config.get("lean_boundary_policy")

    proofs_path: Path | None = None
    backup: str | None = None
    if lean_backup_proofs:
        proofs_path = lean_workspace_path / "Discovery" / "Discovery" / "Proofs.lean"
        if proofs_path.exists():
            backup = proofs_path.read_text(encoding="utf-8")

    log: dict[str, Any] = {
        "suite_id": suite_id,
        "case_id": case.case_id,
        "display_name": case.display_name,
        "repeat_index": repeat_index,
        "config_path": str(resolved_config_path),
        "workspace": str(run_dir),
        "metadata": {
            "benchmark_scope": case.benchmark_scope,
            "model": model,
            "tags": case.tags,
            "lean_policy": case.lean_policy,
            "timeouts": {
                "experiment_seconds": case.timeouts["experiment"],
                "lean_decompose_seconds": case.timeouts["decompose"],
                "lean_verify_seconds": case.timeouts["lean"],
            },
            "started_at": _utc_now().isoformat(),
            "llm_record_dir": str(llm_record_dir),
            "engine": "mcts",
            "lean_workspace": str(lean_workspace_path),
        },
        "steps": [],
        "snapshots": [],
    }

    planning_feedback = _build_case_planning_feedback(case)
    theorem_id: str | None = None
    run_reason = "completed"
    run_error: Exception | None = None
    mcts_result: Any | None = None

    try:
        graph = HyperGraph()
        node_ids: dict[str, str] = {}
        for seed in resolved_config.get("seed_nodes", []):
            node = graph.add_node(
                statement=seed["statement"],
                belief=seed.get("belief", 0.0),
                formal_statement=seed.get("formal_statement"),
                domain=seed.get("domain"),
                state=seed.get("state", "unverified"),
            )
            node_ids[seed["key"]] = node.id
        target = resolved_config["target"]
        theorem = graph.add_node(
            statement=target["statement"],
            formal_statement=target.get("formal_statement"),
            belief=target.get("belief", 0.5),
            domain=target.get("domain"),
            state=target.get("state", "unverified"),
        )
        theorem_id = theorem.id
        node_ids[target["key"]] = theorem.id
        node_ids["theorem"] = theorem.id
        save_graph(graph, graph_path)
        log["node_ids"] = node_ids
        log["snapshots"].append(_snapshot(graph, "seed"))
        _save_json(log_path, log)

        continuation_verifier = ContinuationVerifier(
            ContinuationConfig(
                num_continuations=CONFIG.continuation_num_samples,
                consistency_threshold=CONFIG.continuation_consistency_threshold,
            )
        ) if CONFIG.enable_continuation_verification else None
        prm_model = CONFIG.external_prm_model or model
        external_prm = ExternalPRM(
            ExternalPRMConfig(
                provider=CONFIG.external_prm_provider,
                api_base=CONFIG.external_prm_api_base or CONFIG.llm_api_base,
                api_key=CONFIG.external_prm_api_key or CONFIG.llm_api_key,
                model=prm_model,
            ),
            fallback_verifier=continuation_verifier,
        ) if prm_model else None

        # Build ProcessAdvantageVerifier only when explicitly enabled.
        pav = None
        if CONFIG.pav_enabled:
            _pav_model_path = None
            if CONFIG.pav_model_path:
                from pathlib import Path as _Path
                _candidate = _Path(CONFIG.pav_model_path)
                if _candidate.exists():
                    _pav_model_path = _candidate
                else:
                    import logging as _logging
                    _logging.getLogger(__name__).warning(
                        "pav_model_path %s does not exist; PAV starts untrained.", _candidate
                    )
            pav = ProcessAdvantageVerifier(
                model_path=_pav_model_path,
                external_prm=external_prm,
                blend_ratio=1.0 if external_prm is not None else 0.0,
            )
        novelty_tracker = NoveltyTracker()
        curiosity = CuriosityDrivenExplorer(
            novelty_tracker=novelty_tracker,
            pav=pav,
            config=CuriosityConfig(),
        )
        # Build retrieval index. When gaia_api_base is configured, also construct
        # a GaiaClient for cross-run global graph search (used inside retrieval).
        _gaia_client = build_gaia_client(CONFIG.gaia_api_base or None)
        retrieval_index = HypergraphRetrievalIndex(
            config=RetrievalConfig(
                max_results=CONFIG.retrieval_max_results,
                min_similarity=CONFIG.retrieval_min_similarity,
                graph_proximity_weight=CONFIG.retrieval_graph_proximity_weight,
                embedding_api_base=CONFIG.embedding_api_base,
                use_gaia_storage=CONFIG.retrieval_use_gaia_storage,
                gaia_vector_top_k=CONFIG.retrieval_gaia_vector_top_k,
            ),
            gaia_client=_gaia_client,
        ) if CONFIG.enable_retrieval else None
        experiment_evolver = ExperimentEvolver(
            EvolutionConfig(backend=CONFIG.experiment_backend)
        ) if CONFIG.enable_evolutionary_experiments else None
        claim_verifier = ClaimVerifier(
            backend_name=CONFIG.experiment_backend,
            verification_model=CONFIG.claim_verification_model or None,
            max_claims_per_call=CONFIG.claim_verification_max_claims,
            lean_verify_timeout=case.timeouts["lean"],
        ) if CONFIG.enable_claim_verifier else None
        analogy_engine = AnalogyEngine() if CONFIG.enable_analogy else None
        specialize_engine = SpecializeEngine() if CONFIG.enable_specialize else None
        decompose_engine = DecomposeEngine() if CONFIG.enable_decompose else None
        knowledge_retriever = KnowledgeRetriever() if CONFIG.enable_knowledge_retrieval else None

        # Verification-driven pipeline components.
        claim_pipeline = ClaimPipeline(
            ClaimPipelineConfig(max_claims_per_memo=CONFIG.max_claims_per_memo)
        ) if CONFIG.verification_loop_enabled else None
        lean_feedback_parser = LeanFeedbackParser() if CONFIG.lean_feedback_enabled else None
        structural_claim_router = StructuralClaimRouter(
            max_decompose_depth=CONFIG.max_decompose_depth,
            structural_complexity_threshold=CONFIG.structural_complexity_threshold,
            decompose_engine=decompose_engine,
            decompose_timeout=case.timeouts["decompose"],
            verify_timeout=case.timeouts["lean"],
            workspace_path=lean_workspace_path,
        ) if CONFIG.lean_feedback_enabled else None
        signal_accumulator = SignalAccumulator(
            threshold=CONFIG.bp_propagation_threshold
        ) if CONFIG.verification_loop_enabled else None

        engine = MCTSDiscoveryEngine(
            graph_path=graph_path,
            target_node_id=theorem.id,
            config=MCTSConfig(
                max_iterations=CONFIG.mcts_max_iterations,
                max_time_seconds=CONFIG.mcts_max_time_seconds,
                post_action_budget_seconds=max(
                    CONFIG.mcts_post_action_budget_seconds,
                    float(case.timeouts["lean"]),
                ),
                c_puct=CONFIG.mcts_c_puct,
                num_simulations_per_expand=CONFIG.mcts_num_simulations_per_expand,
                enable_evolutionary_experiments=CONFIG.enable_evolutionary_experiments,
                enable_continuation_verification=CONFIG.enable_continuation_verification,
                enable_retrieval=CONFIG.enable_retrieval,
                enable_problem_variants=CONFIG.enable_problem_variants,
                specialization_threshold=CONFIG.mcts_specialization_threshold,
                progressive_widening_base=CONFIG.mcts_progressive_widening_base,
                replan_on_stuck=CONFIG.mcts_replan_on_stuck,
            ),
            pav=pav,
            novelty_tracker=novelty_tracker,
            curiosity_explorer=curiosity,
            continuation_verifier=continuation_verifier,
            experiment_evolver=experiment_evolver,
            retrieval_index=retrieval_index,
            experience_buffer=experience_buffer,
            problem_variant_generator=ProblemVariantGenerator() if CONFIG.enable_problem_variants else None,
            claim_verifier=claim_verifier,
            analogy_engine=analogy_engine,
            specialize_engine=specialize_engine,
            decompose_engine=decompose_engine,
            knowledge_retriever=knowledge_retriever,
            claim_pipeline=claim_pipeline,
            lean_feedback_parser=lean_feedback_parser,
            structural_claim_router=structural_claim_router,
            signal_accumulator=signal_accumulator,
            lean_policy=case.lean_policy,
            model=model,
            backend="bp",
            llm_record_dir=llm_record_dir,
            bridge_path=bridge_path,
            lean_timeout=case.timeouts["lean"],
        )
        # Incremental log flush: called at the end of every MCTS iteration.
        # Keeps exploration_log.json up-to-date so partial progress is visible
        # in real time and the data is not lost if the run is interrupted.
        _flushed_steps: int = 0

        def _flush_iteration(iteration: int, result: "MCTSDiscoveryResult") -> None:
            nonlocal _flushed_steps
            new_steps = result.steps[_flushed_steps:]
            if new_steps:
                log["steps"].extend(new_steps)
                _flushed_steps = len(result.steps)
            # Snapshot belief state of key nodes after this iteration.
            try:
                _g = load_graph(graph_path)
                log["snapshots"].append(
                    {
                        **_snapshot(_g, f"iteration_{iteration}"),
                        "iteration": iteration,
                        "target_belief": round(
                            float(_g.nodes[theorem_id].belief)
                            if theorem_id in _g.nodes else 0.0, 6
                        ),
                    }
                )
            except Exception:
                pass
            log["metadata"]["last_iteration"] = iteration
            log["metadata"]["last_flush_at"] = _utc_now().isoformat()
            _save_json(log_path, log)

        class _RunTerminated(RuntimeError):
            pass

        sigterm_received = {"flag": False}
        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
        previous_sighup_handler = signal.getsignal(signal.SIGHUP) if hasattr(signal, "SIGHUP") else None

        def _handle_sigterm(_signum: int, _frame: Any) -> None:
            sigterm_received["flag"] = True
            raise _RunTerminated(f"Received signal {_signum} during MCTS run")

        signal.signal(signal.SIGTERM, _handle_sigterm)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, _handle_sigterm)
        try:
            try:
                mcts_result = engine.run(
                    planning_feedback=planning_feedback or "",
                    boundary_policy=boundary_policy,
                    on_iteration_complete=_flush_iteration,
                )
            except _RunTerminated as exc:
                run_reason = "terminated"
                run_error = exc
                log["steps"].append(
                    {
                        "phase": "engine_termination",
                        "error": str(exc),
                    }
                )
                log["metadata"]["terminated_at"] = _utc_now().isoformat()
                _save_json(log_path, log)
            except Exception as exc:
                run_reason = "engine_crash"
                run_error = exc
                log["steps"].append(
                    {
                        "phase": "engine_crash",
                        "error": str(exc),
                    }
                )
                log["metadata"]["crashed_at"] = _utc_now().isoformat()
                _save_json(log_path, log)
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
            if hasattr(signal, "SIGHUP") and previous_sighup_handler is not None:
                signal.signal(signal.SIGHUP, previous_sighup_handler)

        # Flush any remaining steps not captured by the last callback.
        if mcts_result is not None:
            remaining_steps = mcts_result.steps[_flushed_steps:]
            if remaining_steps:
                log["steps"].extend(remaining_steps)
        graph = load_graph(graph_path)
        log["snapshots"].append(_snapshot(graph, "after_mcts"))
        log["metadata"]["finished_at"] = _utc_now().isoformat()
        _save_json(log_path, log)
    except Exception as exc:
        run_reason = "engine_crash"
        run_error = exc
        log["steps"].append(
            {
                "phase": "run_case_mcts_error",
                "error": str(exc),
            }
        )
        log["metadata"]["crashed_at"] = _utc_now().isoformat()
        _save_json(log_path, log)
    finally:
        try:
            if proofs_path is not None and backup is not None:
                proofs_path.write_text(backup, encoding="utf-8")
        except Exception as restore_exc:
            logger.warning("Failed to restore proofs file: %s", restore_exc)
        try:
            summary = _generate_partial_summary(
                case=case,
                run_dir=run_dir,
                graph_path=graph_path,
                log_path=log_path,
                bridge_path=bridge_path,
                repeat_index=repeat_index,
                reason=run_reason,
                error=run_error,
            )
            _save_json(summary_path, summary)
            _write_scorecard_markdown(summary, scorecard_path)
        except Exception as summary_exc:
            logger.error("Failed to generate summary: %s", summary_exc, exc_info=True)
            summary = {"error": str(summary_exc), "reason": run_reason}
            _save_json(summary_path, summary)

    return summary


def run_case_once(
    case: BenchmarkCaseConfig,
    *,
    run_dir: Path,
    repeat_index: int,
    suite_id: str,
    experience_buffer: Optional[ExperienceBuffer] = None,
) -> dict[str, Any]:
    if CONFIG.enable_mcts:
        return _run_case_once_mcts(
            case,
            run_dir=run_dir,
            repeat_index=repeat_index,
            suite_id=suite_id,
            experience_buffer=experience_buffer,
        )
    source_config = _read_json(case.source_proof_config)
    resolved_config = _copy_resolved_config(source_config, run_dir, case)
    run_dir.mkdir(parents=True, exist_ok=False)

    resolved_config_path = run_dir / "resolved_proof_config.json"
    graph_path = run_dir / "graph.json"
    log_path = run_dir / "exploration_log.json"
    bridge_path = run_dir / "bridge_plan.json"
    summary_path = run_dir / "summary.json"
    scorecard_path = run_dir / "PATH_BENCHMARK_SCORECARD_ZH.md"
    llm_record_dir = run_dir / "llm_records"

    lean_workspace_path, lean_backup_proofs = _prepare_benchmark_lean_workspace(resolved_config, run_dir)
    os.environ["DISCOVERY_ZERO_LEAN_WORKSPACE"] = str(lean_workspace_path)

    model = resolved_config.get("model") or os.environ.get("DISCOVERY_ZERO_LLM_MODEL") or CONFIG.llm_model
    resolved_config["model"] = model
    _save_json(resolved_config_path, resolved_config)
    boundary_policy = resolved_config.get("lean_boundary_policy")

    proofs_path: Path | None = None
    backup: str | None = None
    if lean_backup_proofs:
        proofs_path = lean_workspace_path / "Discovery" / "Discovery" / "Proofs.lean"
        if proofs_path.exists():
            backup = proofs_path.read_text(encoding="utf-8")

    log: dict[str, Any] = {
        "suite_id": suite_id,
        "case_id": case.case_id,
        "display_name": case.display_name,
        "repeat_index": repeat_index,
        "config_path": str(resolved_config_path),
        "workspace": str(run_dir),
        "metadata": {
            "benchmark_scope": case.benchmark_scope,
            "model": model,
            "tags": case.tags,
            "lean_policy": case.lean_policy,
            "timeouts": {
                "experiment_seconds": case.timeouts["experiment"],
                "lean_decompose_seconds": case.timeouts["decompose"],
                "lean_verify_seconds": case.timeouts["lean"],
            },
            "started_at": _utc_now().isoformat(),
            "llm_record_dir": str(llm_record_dir),
            "lean_workspace": str(lean_workspace_path),
        },
        "steps": [],
        "snapshots": [],
    }

    best_bridge_confidence = float("-inf")
    best_bridge_plan: BridgePlan | None = None
    best_bridge_node_map: dict[str, str] = {}
    theorem_id: str | None = None
    target_statement = resolved_config["target"]["statement"]
    plausible_output: dict[str, Any] | None = None
    plausible_judge: dict[str, Any] | None = None
    planning_feedback = _build_case_planning_feedback(case)

    def flush_log() -> None:
        _save_json(log_path, log)

    def resolve_target_id(graph: HyperGraph) -> str:
        if theorem_id and theorem_id in graph.nodes:
            return theorem_id
        matches = graph.find_node_ids_by_statement(target_statement)
        if len(matches) == 1:
            return matches[0]
        raise BenchmarkError("Could not resolve target theorem node id.")

    def record_error(phase: str, exc: Exception, **extra: Any) -> None:
        log["steps"].append({"phase": phase, "error": str(exc), **extra})
        flush_log()

    def save_bridge_plan(
        graph: HyperGraph,
        reasoning_output: dict[str, Any],
        judge_output: dict[str, Any] | None,
        phase: str,
        feedback: str | None = None,
    ) -> None:
        nonlocal best_bridge_confidence, best_bridge_plan, best_bridge_node_map
        raw_bridge, plan = run_bridge_planning_action(
            graph,
            resolve_target_id(graph),
            reasoning_output,
            judge_output=judge_output,
            model=model,
            feedback=feedback,
            record_dir=llm_record_dir,
        )
        log["steps"].append(
            {
                "phase": f"{phase}_bridge_plan",
                "raw": raw_bridge,
                "bridge_metrics": plan.metrics(),
            }
        )
        confidence = float((judge_output or {}).get("confidence", 0.0))
        if confidence >= best_bridge_confidence:
            best_bridge_node_map = materialize_bridge_nodes(
                graph,
                plan,
                default_domain=graph.nodes[resolve_target_id(graph)].domain,
            )
            save_graph(graph, graph_path)
            bridge_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
            best_bridge_confidence = confidence
            best_bridge_plan = plan
        flush_log()

    try:
        graph = HyperGraph()
        node_ids: dict[str, str] = {}
        for seed in resolved_config.get("seed_nodes", []):
            node = graph.add_node(
                statement=seed["statement"],
                belief=seed.get("belief", 0.0),
                formal_statement=seed.get("formal_statement"),
                domain=seed.get("domain"),
                state=seed.get("state", "unverified"),
            )
            node_ids[seed["key"]] = node.id
        target = resolved_config["target"]
        theorem = graph.add_node(
            statement=target["statement"],
            formal_statement=target.get("formal_statement"),
            belief=target.get("belief", 0.5),
            domain=target.get("domain"),
            state=target.get("state", "unverified"),
        )
        theorem_id = theorem.id
        node_ids[target["key"]] = theorem.id
        node_ids["theorem"] = theorem.id
        save_graph(graph, graph_path)
        log["node_ids"] = node_ids
        log["snapshots"].append(_snapshot(graph, "seed"))
        flush_log()

        try:
            raw_p, out_p, judge_p = run_plausible_action(
                graph,
                theorem.id,
                model=model,
                feedback=planning_feedback,
                max_attempts=4,
                record_dir=llm_record_dir,
            )
            plausible_output = out_p
            plausible_judge = judge_p
            for seed_key in resolved_config.get("extra_plausible_premises", []):
                out_p["premises"].append(
                    {
                        "id": node_ids[seed_key],
                        "statement": graph.nodes[node_ids[seed_key]].statement,
                    }
                )
            edge_p = ingest_action_output(
                graph_path,
                ActionResult(
                    action="plausible",
                    target_node_id=theorem.id,
                    selected_module=Module.PLAUSIBLE.value,
                    raw_output=raw_p,
                    normalized_output=out_p,
                    judge_output=judge_p,
                    success=True,
                    message="plausible planning complete",
                ),
                backend="bp",
            )
            graph = load_graph(graph_path)
            log["steps"].append(
                {
                    "phase": "plausible",
                    "raw": raw_p,
                    "normalized": out_p,
                    "judge": judge_p,
                    "edge_id": edge_p.ingest_edge_id,
                    "message": edge_p.message,
                }
            )
            try:
                save_bridge_plan(graph, out_p, judge_p, "plausible")
            except Exception as exc:
                record_error("plausible_bridge_plan", exc)
            log["snapshots"].append(_snapshot(graph, "after_plausible"))
            flush_log()
        except Exception as exc:
            record_error("plausible", exc)
            graph = load_graph(graph_path)

        if plausible_output is not None:
            try:
                raw_e, out_e, judge_e = run_experiment_action(
                    graph,
                    theorem.id,
                    model=model,
                    timeout=case.timeouts["experiment"],
                )
                res_e = ingest_action_output(
                    graph_path,
                    ActionResult(
                        action="experiment",
                        target_node_id=theorem.id,
                        selected_module=Module.EXPERIMENT.value,
                        raw_output=raw_e,
                        normalized_output=out_e,
                        judge_output=judge_e,
                        success=True,
                        message="experiment complete",
                    ),
                    backend="bp",
                )
                graph = load_graph(graph_path)
                log["steps"].append(
                    {
                        "phase": "experiment",
                        "raw": raw_e,
                        "normalized": out_e,
                        "judge": judge_e,
                        "edge_id": res_e.ingest_edge_id,
                        "message": res_e.message,
                    }
                )
                log["snapshots"].append(_snapshot(graph, "after_experiment"))
                flush_log()
            except Exception as exc:
                record_error("experiment", exc)
                graph = load_graph(graph_path)

        bridge_consumption = None
        lean_gate = {
            "attempt_decomposition": False,
            "attempt_strict_lean": False,
            "decomposition_reason": "No bridge consumption decision available.",
            "strict_lean_reason": "No bridge consumption decision available.",
        }
        if best_bridge_plan is not None:
            try:
                bridge_consumption = plan_bridge_consumption(best_bridge_plan)
                grade_a_count, grade_b_count, grade_c_count, grade_d_count, total_graded_steps = _grade_counts_from_plan(best_bridge_plan)
                grade_d_ratio = _safe_ratio(grade_d_count, total_graded_steps)
                best_path_confidence = float((_resolve_best_path_candidate(log["steps"]) or {}).get("judge", {}).get("confidence", 0.0))
                lean_gate = should_attempt_lean(
                    case,
                    best_path_confidence=best_path_confidence,
                    grade_d_ratio=grade_d_ratio,
                    strict_mode=bridge_consumption.strict_mode,
                    has_decomposition_plan=bridge_consumption.decomposition_bridge_plan is not None,
                    has_strict_target=bridge_consumption.strict_focus_proposition_id is not None,
                )
                log["steps"].append(
                    {
                        "phase": "bridge_consumption",
                        **bridge_consumption.to_log_dict(best_bridge_plan),
                        "lean_gate": lean_gate,
                    }
                )
                flush_log()
            except Exception as exc:
                record_error("bridge_consumption", exc)

        if plausible_output is not None and bridge_consumption is not None and bridge_consumption.experiment_target_proposition_id is not None:
            try:
                graph = load_graph(graph_path)
                experiment_target_id = best_bridge_node_map.get(
                    bridge_consumption.experiment_target_proposition_id,
                    theorem.id,
                )
                raw_be, out_be, judge_be = run_experiment_action(
                    graph,
                    experiment_target_id,
                    model=model,
                    timeout=case.timeouts["experiment"],
                )
                res_be = ingest_action_output(
                    graph_path,
                    ActionResult(
                        action="bridge_experiment",
                        target_node_id=experiment_target_id,
                        selected_module=Module.EXPERIMENT.value,
                        raw_output=raw_be,
                        normalized_output=out_be,
                        judge_output=judge_be,
                        success=True,
                        message="bridge experiment complete",
                    ),
                    backend="bp",
                )
                graph = load_graph(graph_path)
                log["steps"].append(
                    {
                        "phase": "bridge_experiment",
                        "raw": raw_be,
                        "normalized": out_be,
                        "judge": judge_be,
                        "focus_bridge_proposition_id": bridge_consumption.experiment_focus_proposition_id,
                        "focus_graph_node_id": experiment_target_id,
                        "edge_id": res_be.ingest_edge_id,
                        "message": res_be.message,
                    }
                )
                log["snapshots"].append(_snapshot(graph, "after_bridge_experiment"))
                flush_log()

                if best_bridge_plan is not None:
                    ready_prop_id = select_ready_bridge_proposition(
                        best_bridge_plan,
                        graph,
                        best_bridge_node_map,
                        consumed_proposition_ids={bridge_consumption.experiment_target_proposition_id},
                    )
                    if ready_prop_id is not None:
                        ready_graph_id = best_bridge_node_map.get(ready_prop_id, theorem.id)
                        ready_prop = next(item for item in best_bridge_plan.propositions if item.id == ready_prop_id)
                        premise_ids = [
                            best_bridge_node_map[dep]
                            for dep in ready_prop.depends_on
                            if dep in best_bridge_node_map and _proposition_supported_in_graph(graph, best_bridge_node_map, dep)
                        ]
                        ready_result = ingest_action_output(
                            graph_path,
                            ActionResult(
                                action="bridge_ready",
                                target_node_id=ready_graph_id,
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
                                        f"Ready local proposition selected: {ready_prop.statement}",
                                    ],
                                    "conclusion": {
                                        "statement": ready_prop.statement,
                                        "formal_statement": None,
                                    },
                                    "module": "plausible",
                                    "domain": graph.nodes[ready_graph_id].domain,
                                    "confidence": 0.76,
                                },
                                success=True,
                                message="bridge ready",
                            ),
                            backend="bp",
                        )
                        graph = load_graph(graph_path)
                        log["steps"].append(
                            {
                                "phase": "bridge_ready",
                                "ready_bridge_proposition_id": ready_prop_id,
                                "ready_graph_node_id": ready_graph_id,
                                "edge_id": ready_result.ingest_edge_id,
                                "message": ready_result.message,
                            }
                        )
                        log["snapshots"].append(_snapshot(graph, "after_bridge_ready"))
                        flush_log()
            except Exception as exc:
                record_error(
                    "bridge_experiment",
                    exc,
                    focus_bridge_proposition_id=bridge_consumption.experiment_focus_proposition_id,
                )

        if plausible_output is not None:
            try:
                if best_bridge_plan is None or bridge_consumption is None:
                    log["steps"].append(
                        {
                            "phase": "decomposition",
                            "skipped": True,
                            "reason": "No validated bridge plan available for Lean decomposition.",
                            "subgoals": [],
                        }
                    )
                    flush_log()
                elif bridge_consumption.decomposition_bridge_plan is None:
                    log["steps"].append(
                        {
                            "phase": "decomposition",
                            "skipped": True,
                            "reason": "No bridge proposition selected for further Lean decomposition.",
                            "subgoals": [],
                        }
                    )
                    flush_log()
                elif not lean_gate["attempt_decomposition"]:
                    log["steps"].append(
                        {
                            "phase": "decomposition",
                            "skipped": True,
                            "skipped_by_policy": True,
                            "reason": lean_gate["decomposition_reason"],
                            "subgoals": [],
                        }
                    )
                    flush_log()
                else:
                    graph = load_graph(graph_path)
                    extra_map = materialize_bridge_nodes(
                        graph,
                        bridge_consumption.decomposition_bridge_plan,
                        default_domain=graph.nodes[theorem.id].domain,
                    )
                    best_bridge_node_map.update(extra_map)
                    save_graph(graph, graph_path)
                    graph = load_graph(graph_path)
                    decompose_target_id = best_bridge_node_map.get(
                        bridge_consumption.decomposition_target_proposition_id or "",
                        theorem.id,
                    )
                    raw_d, norm_d, subgoals = run_lean_decompose_action(
                        graph,
                        decompose_target_id,
                        model=model,
                        timeout=case.timeouts["decompose"],
                        boundary_policy=boundary_policy,
                        max_attempts=3,
                        bridge_plan=bridge_consumption.decomposition_bridge_plan,
                        record_dir=llm_record_dir,
                    )
                    res_d = ingest_decomposition_output(
                        graph_path,
                        decompose_target_id,
                        norm_d,
                        subgoals,
                        backend="bp",
                    )
                    graph = load_graph(graph_path)
                    log["steps"].append(
                        {
                            "phase": "decomposition",
                            "raw": raw_d,
                            "normalized": norm_d,
                            "subgoals": subgoals,
                            "focus_bridge_proposition_id": bridge_consumption.decomposition_focus_proposition_id,
                            "focus_graph_node_id": decompose_target_id,
                            "edge_id": res_d.ingest_edge_id,
                            "created_node_ids": res_d.created_node_ids,
                        }
                    )
                    log["snapshots"].append(_snapshot(graph, "after_decomposition"))
                    flush_log()
            except Exception as exc:
                record_error("decomposition", exc, subgoals=[])
                if plausible_output is not None and plausible_judge is not None:
                    try:
                        graph = load_graph(graph_path)
                        raw_r1, out_r1, judge_r1 = run_plausible_action(
                            graph,
                            theorem.id,
                            model=model,
                                feedback="\n\n".join(
                                    item
                                    for item in (
                                        planning_feedback,
                                        _build_replan_feedback(
                                            target_statement=graph.nodes[theorem.id].statement,
                                            previous_output=plausible_output,
                                            judge_output=plausible_judge,
                                            failure_message=str(exc),
                                            failed_module="lean_decompose",
                                        ),
                                    )
                                    if item
                                ),
                            max_attempts=3,
                        )
                        res_r1 = ingest_action_output(
                            graph_path,
                            ActionResult(
                                action="plausible_replan_after_decomposition",
                                target_node_id=theorem.id,
                                selected_module=Module.PLAUSIBLE.value,
                                raw_output=raw_r1,
                                normalized_output=out_r1,
                                judge_output=judge_r1,
                                success=True,
                                message="replanned after decomposition failure",
                            ),
                            backend="bp",
                        )
                        graph = load_graph(graph_path)
                        save_bridge_plan(
                            graph,
                            out_r1,
                            judge_r1,
                            "plausible_replan_after_decomposition",
                            feedback=str(exc),
                        )
                        log["steps"].append(
                            {
                                "phase": "plausible_replan_after_decomposition",
                                "raw": raw_r1,
                                "normalized": out_r1,
                                "judge": judge_r1,
                                "edge_id": res_r1.ingest_edge_id,
                                "message": res_r1.message,
                            }
                        )
                        log["snapshots"].append(_snapshot(graph, "after_replan_decomposition_failure"))
                        flush_log()
                    except Exception as replan_exc:
                        record_error("plausible_replan_after_decomposition", replan_exc)

        if plausible_output is not None:
            strict_target_id = theorem.id
            strict_focus_prop_id = None
            strict_mode = None
            strict_prompt_feedback = None
            if best_bridge_plan is not None:
                try:
                    latest_bridge_consumption = plan_bridge_consumption(best_bridge_plan)
                    strict_focus_prop_id = latest_bridge_consumption.strict_focus_proposition_id
                    strict_mode = latest_bridge_consumption.strict_mode
                    if strict_focus_prop_id is not None:
                        strict_target_id = best_bridge_node_map.get(
                            latest_bridge_consumption.strict_target_proposition_id or "",
                            theorem.id,
                        )
                        strict_prompt_feedback = build_strict_lean_bridge_feedback(
                            best_bridge_plan,
                            strict_focus_prop_id,
                            strict_mode=strict_mode,
                        )
                except Exception as exc:
                    record_error("strict_lean_selection", exc)
            if not lean_gate["attempt_strict_lean"]:
                log["steps"].append(
                    {
                        "phase": "strict_lean",
                        "skipped": True,
                        "skipped_by_policy": True,
                        "reason": lean_gate["strict_lean_reason"],
                        "success": False,
                        "strict_focus_bridge_proposition_id": strict_focus_prop_id,
                        "strict_mode": strict_mode,
                        "strict_graph_node_id": strict_target_id,
                    }
                )
                flush_log()
            else:
                try:
                    graph = load_graph(graph_path)
                    raw_l, out_l, _ = run_lean_action(
                        graph,
                        strict_target_id,
                        model=model,
                        timeout=case.timeouts["lean"],
                        boundary_policy=boundary_policy,
                        max_attempts=3,
                        prompt_feedback=strict_prompt_feedback,
                    )
                    res_l = ingest_action_output(
                        graph_path,
                        ActionResult(
                            action="lean",
                            target_node_id=strict_target_id,
                            selected_module=Module.LEAN.value,
                            raw_output=raw_l,
                            normalized_output=out_l,
                            success=True,
                            message="lean complete",
                        ),
                        backend="bp",
                    )
                    graph = load_graph(graph_path)
                    log["steps"].append(
                        {
                            "phase": "strict_lean",
                            "raw": raw_l,
                            "normalized": out_l,
                            "success": True,
                            "error": None,
                            "strict_focus_bridge_proposition_id": strict_focus_prop_id,
                            "strict_mode": strict_mode,
                            "strict_graph_node_id": strict_target_id,
                            "edge_id": res_l.ingest_edge_id,
                            "message": res_l.message,
                        }
                    )
                    log["snapshots"].append(_snapshot(graph, "after_strict_lean"))
                    flush_log()
                except Exception as exc:
                    record_error(
                        "strict_lean",
                        exc,
                        raw=None,
                        success=False,
                        strict_focus_bridge_proposition_id=strict_focus_prop_id,
                        strict_mode=strict_mode,
                        strict_graph_node_id=strict_target_id,
                    )
                    if plausible_output is not None:
                        try:
                            graph = load_graph(graph_path)
                            raw_r2, out_r2, judge_r2 = run_plausible_action(
                                graph,
                                theorem.id,
                                model=model,
                                feedback="\n\n".join(
                                    item
                                    for item in (
                                        planning_feedback,
                                        _build_replan_feedback(
                                            target_statement=graph.nodes[theorem.id].statement,
                                            failure_message=str(exc),
                                            failed_module="lean",
                                        ),
                                    )
                                    if item
                                ),
                                max_attempts=3,
                            )
                            res_r2 = ingest_action_output(
                                graph_path,
                                ActionResult(
                                    action="plausible_replan_after_lean",
                                    target_node_id=theorem.id,
                                    selected_module=Module.PLAUSIBLE.value,
                                    raw_output=raw_r2,
                                    normalized_output=out_r2,
                                    judge_output=judge_r2,
                                    success=True,
                                    message="replanned after lean failure",
                                ),
                                backend="bp",
                            )
                            graph = load_graph(graph_path)
                            save_bridge_plan(
                                graph,
                                out_r2,
                                judge_r2,
                                "plausible_replan_after_lean",
                                feedback=str(exc),
                            )
                            log["steps"].append(
                                {
                                    "phase": "plausible_replan_after_lean",
                                    "raw": raw_r2,
                                    "normalized": out_r2,
                                    "judge": judge_r2,
                                    "edge_id": res_r2.ingest_edge_id,
                                    "message": res_r2.message,
                                }
                            )
                            log["snapshots"].append(_snapshot(graph, "after_replan_lean_failure"))
                            flush_log()
                        except Exception as replan_exc:
                            record_error("plausible_replan_after_lean", replan_exc)

        graph = load_graph(graph_path)
        if CONFIG.bp_backend == "energy":
            propagate_beliefs_energy(graph)
            save_graph(graph, graph_path)
            log["snapshots"].append(_snapshot(graph, "after_energy"))
        log["metadata"]["finished_at"] = _utc_now().isoformat()
        flush_log()
    finally:
        if proofs_path is not None and backup is not None:
            proofs_path.write_text(backup, encoding="utf-8")

    summary = summarize_run(
        case=case,
        run_dir=run_dir,
        graph_path=graph_path,
        log_path=log_path,
        bridge_path=bridge_path,
        repeat_index=repeat_index,
    )
    _save_json(summary_path, summary)
    _write_scorecard_markdown(summary, scorecard_path)
    return summary


def _aggregate_metric_values(values: list[float]) -> dict[str, float]:
    return {
        "mean": round(_safe_mean(values), 6),
        "stdev": round(_safe_std(values), 6),
        "min": round(min(values), 6) if values else 0.0,
        "max": round(max(values), 6) if values else 0.0,
    }


def aggregate_case_runs(case: BenchmarkCaseConfig, run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    metric_keys = [
        "bridge_plan_valid",
        "bridge_consumption_ready",
        "bridge_experiment_attempted",
        "bridge_experiment_success",
        "best_path_judge_confidence",
        "grade_a_count",
        "grade_b_count",
        "grade_c_count",
        "grade_d_count",
        "experiment_trials",
        "strict_lean_attempted",
        "lean_decompose_success",
        "lean_decompose_attempted",
        "lean_subgoal_count",
        "strict_lean_success",
        "new_nodes_created",
        "max_new_node_belief",
        "high_confidence_new_nodes",
        "PQI",
        "FRI",
        "ESI",
        "ODB",
        "DBI",
        "ORI",
        "NVI",
        "OFC",
    ]
    metrics_aggregate: dict[str, Any] = {}
    for key in metric_keys:
        values = [float(item["metrics"].get(key, 0)) for item in run_summaries]
        metrics_aggregate[key] = _aggregate_metric_values(values)

    failure_counts = Counter(item["first_failure_stage"] for item in run_summaries)
    blocker_counts = Counter(item["current_bottleneck"] for item in run_summaries)
    outcome_counts = Counter(item["benchmark_outcome"] for item in run_summaries)
    run_count = len(run_summaries)
    success_count = sum(1 for item in run_summaries if item["success"])
    strict_success_count = sum(1 for item in run_summaries if item["strict_lean_success"])
    progress_count = sum(1 for item in run_summaries if item["progress_without_closure"])
    decompose_success_count = sum(int(item["metrics"]["lean_decompose_success"]) for item in run_summaries)
    decompose_attempt_count = sum(int(item["metrics"]["lean_decompose_attempted"]) for item in run_summaries)
    experiment_ran_count = sum(int(item["metrics"]["experiment_ran"]) for item in run_summaries)
    localized_count = sum(int(item["metrics"]["first_failure_is_localized"]) for item in run_summaries)
    strict_attempt_count = sum(int(item["metrics"]["strict_lean_attempted"]) for item in run_summaries)
    replan_count = sum(int(item["metrics"]["replan_triggered"]) for item in run_summaries)
    decompose_policy_skip_count = sum(int(item["metrics"]["lean_decompose_skipped_by_policy"]) for item in run_summaries)
    strict_policy_skip_count = sum(int(item["metrics"]["strict_lean_skipped_by_policy"]) for item in run_summaries)
    counterexample_count = sum(int(item["metrics"]["experiment_found_counterexample"]) for item in run_summaries)
    high_pqi_count = sum(1 for item in run_summaries if float(item["metrics"]["PQI"]) >= 80.0)
    formalization_ready_count = sum(1 for item in run_summaries if float(item["metrics"]["FRI"]) >= 50.0)
    strong_experiment_support_count = sum(1 for item in run_summaries if float(item["metrics"]["ESI"]) >= 70.0)
    mean_final_target_belief = _safe_mean([float(item["final_target_belief"]) for item in run_summaries])
    bridge_plan_valid_count = sum(int(item["metrics"]["bridge_plan_valid"]) for item in run_summaries)
    bridge_consumption_ready_count = sum(int(item["metrics"]["bridge_consumption_ready"]) for item in run_summaries)
    bridge_experiment_attempt_count = sum(int(item["metrics"]["bridge_experiment_attempted"]) for item in run_summaries)
    bridge_experiment_success_count = sum(int(item["metrics"]["bridge_experiment_success"]) for item in run_summaries)
    theorem_correction_success_count = sum(1 for item in run_summaries if item["benchmark_outcome"] == "theorem_correction_success")
    counterexample_warning_count = sum(1 for item in run_summaries if item["benchmark_outcome"] == "counterexample_warning")
    dominant_outcome = max(outcome_counts.items(), key=lambda item: item[1])[0] if outcome_counts else "stalled"

    return {
        "case_id": case.case_id,
        "display_name": case.display_name,
        "benchmark_scope": case.benchmark_scope,
        "run_count": run_count,
        "dominant_outcome": dominant_outcome,
        "success_rate": round(_safe_ratio(success_count, run_count), 6),
        "strict_lean_success_rate": round(_safe_ratio(strict_success_count, run_count), 6),
        "progress_without_closure_rate": round(_safe_ratio(progress_count, run_count), 6),
        "counterexample_rate": round(_safe_ratio(counterexample_count, run_count), 6),
        "bridge_plan_valid_rate": round(_safe_ratio(bridge_plan_valid_count, run_count), 6),
        "bridge_consumption_ready_rate": round(_safe_ratio(bridge_consumption_ready_count, run_count), 6),
        "bridge_experiment_attempt_rate": round(_safe_ratio(bridge_experiment_attempt_count, run_count), 6),
        "bridge_experiment_success_rate": round(_safe_ratio(bridge_experiment_success_count, run_count), 6),
        "theorem_correction_success_rate": round(_safe_ratio(theorem_correction_success_count, run_count), 6),
        "counterexample_warning_rate": round(_safe_ratio(counterexample_warning_count, run_count), 6),
        "high_pqi_rate": round(_safe_ratio(high_pqi_count, run_count), 6),
        "formalization_ready_rate": round(_safe_ratio(formalization_ready_count, run_count), 6),
        "strong_experiment_support_rate": round(_safe_ratio(strong_experiment_support_count, run_count), 6),
        "mean_final_target_belief": round(mean_final_target_belief, 6),
        "strict_lean_attempt_rate": round(_safe_ratio(strict_attempt_count, run_count), 6),
        "lean_decompose_attempt_rate": round(_safe_ratio(decompose_attempt_count, run_count), 6),
        "lean_decompose_success_rate": round(_safe_ratio(decompose_success_count, run_count), 6),
        "experiment_ran_rate": round(_safe_ratio(experiment_ran_count, run_count), 6),
        "first_failure_is_localized_rate": round(_safe_ratio(localized_count, run_count), 6),
        "replan_triggered_rate": round(_safe_ratio(replan_count, run_count), 6),
        "lean_decompose_skipped_by_policy_rate": round(_safe_ratio(decompose_policy_skip_count, run_count), 6),
        "strict_lean_skipped_by_policy_rate": round(_safe_ratio(strict_policy_skip_count, run_count), 6),
        "first_failure_stage_counts": dict(failure_counts),
        "current_bottleneck_counts": dict(blocker_counts),
        "benchmark_outcome_counts": dict(outcome_counts),
        "metrics": metrics_aggregate,
        "runs": run_summaries,
    }


def aggregate_suite_results(
    *,
    suite: BenchmarkSuiteConfig,
    cases: list[BenchmarkCaseConfig],
    case_results: list[dict[str, Any]],
    suite_run_dir: Path,
    actual_repeats: int,
) -> dict[str, Any]:
    valid_results = [item for item in case_results if "run_count" in item]
    total_runs = sum(item["run_count"] for item in valid_results)
    success_runs = sum(round(item["success_rate"] * item["run_count"]) for item in valid_results)
    strict_runs = sum(round(item["strict_lean_success_rate"] * item["run_count"]) for item in valid_results)
    localized_runs = sum(round(item["first_failure_is_localized_rate"] * item["run_count"]) for item in valid_results)
    progress_runs = sum(round(item["progress_without_closure_rate"] * item["run_count"]) for item in valid_results)
    strict_attempt_runs = sum(round(item["strict_lean_attempt_rate"] * item["run_count"]) for item in valid_results)
    decompose_attempt_runs = sum(round(item["lean_decompose_attempt_rate"] * item["run_count"]) for item in valid_results)
    replan_runs = sum(round(item["replan_triggered_rate"] * item["run_count"]) for item in valid_results)
    counterexample_runs = sum(round(item["counterexample_rate"] * item["run_count"]) for item in valid_results)
    bridge_plan_valid_runs = sum(round(item["bridge_plan_valid_rate"] * item["run_count"]) for item in valid_results)
    bridge_consumption_ready_runs = sum(round(item["bridge_consumption_ready_rate"] * item["run_count"]) for item in valid_results)
    bridge_experiment_attempt_runs = sum(round(item["bridge_experiment_attempt_rate"] * item["run_count"]) for item in valid_results)
    bridge_experiment_success_runs = sum(round(item["bridge_experiment_success_rate"] * item["run_count"]) for item in valid_results)
    theorem_correction_success_runs = sum(round(item["theorem_correction_success_rate"] * item["run_count"]) for item in valid_results)
    counterexample_warning_runs = sum(round(item["counterexample_warning_rate"] * item["run_count"]) for item in valid_results)
    high_pqi_runs = sum(round(item["high_pqi_rate"] * item["run_count"]) for item in valid_results)
    formalization_ready_runs = sum(round(item["formalization_ready_rate"] * item["run_count"]) for item in valid_results)
    strong_experiment_support_runs = sum(round(item["strong_experiment_support_rate"] * item["run_count"]) for item in valid_results)
    mean_final_target_belief = _safe_mean(
        [
            float(run["final_target_belief"])
            for item in valid_results
            for run in item.get("runs", [])
        ]
    )

    failure_counts = Counter()
    blocker_counts = Counter()
    outcome_counts = Counter()
    for item in valid_results:
        failure_counts.update(item.get("first_failure_stage_counts", {}))
        blocker_counts.update(item.get("current_bottleneck_counts", {}))
        outcome_counts.update(item.get("benchmark_outcome_counts", {}))

    overall_metric_vectors: dict[str, list[float]] = {}
    for metric_key in ("PQI", "FRI", "ESI", "ODB", "DBI", "ORI", "NVI", "OFC"):
        overall_metric_vectors[metric_key] = []
        for item in valid_results:
            overall_metric_vectors[metric_key].extend(
                float(run["metrics"].get(metric_key, 0)) for run in item.get("runs", [])
            )

    return {
        "suite_id": suite.suite_id,
        "display_name": suite.display_name,
        "description": suite.description,
        "suite_run_dir": str(suite_run_dir),
        "repeats": actual_repeats,
        "cases": case_results,
        "overall": {
            "case_count": len(cases),
            "total_runs": total_runs,
            "success_rate": round(_safe_ratio(success_runs, total_runs), 6),
            "strict_lean_success_rate": round(_safe_ratio(strict_runs, total_runs), 6),
            "progress_without_closure_rate": round(_safe_ratio(progress_runs, total_runs), 6),
            "counterexample_rate": round(_safe_ratio(counterexample_runs, total_runs), 6),
            "bridge_plan_valid_rate": round(_safe_ratio(bridge_plan_valid_runs, total_runs), 6),
            "bridge_consumption_ready_rate": round(_safe_ratio(bridge_consumption_ready_runs, total_runs), 6),
            "bridge_experiment_attempt_rate": round(_safe_ratio(bridge_experiment_attempt_runs, total_runs), 6),
            "bridge_experiment_success_rate": round(_safe_ratio(bridge_experiment_success_runs, total_runs), 6),
            "theorem_correction_success_rate": round(_safe_ratio(theorem_correction_success_runs, total_runs), 6),
            "counterexample_warning_rate": round(_safe_ratio(counterexample_warning_runs, total_runs), 6),
            "high_pqi_rate": round(_safe_ratio(high_pqi_runs, total_runs), 6),
            "formalization_ready_rate": round(_safe_ratio(formalization_ready_runs, total_runs), 6),
            "strong_experiment_support_rate": round(_safe_ratio(strong_experiment_support_runs, total_runs), 6),
            "mean_final_target_belief": round(mean_final_target_belief, 6),
            "strict_lean_attempt_rate": round(_safe_ratio(strict_attempt_runs, total_runs), 6),
            "lean_decompose_attempt_rate": round(_safe_ratio(decompose_attempt_runs, total_runs), 6),
            "first_failure_is_localized_rate": round(_safe_ratio(localized_runs, total_runs), 6),
            "replan_triggered_rate": round(_safe_ratio(replan_runs, total_runs), 6),
            "first_failure_stage_counts": dict(failure_counts),
            "current_bottleneck_counts": dict(blocker_counts),
            "benchmark_outcome_counts": dict(outcome_counts),
            "metrics": {
                key: _aggregate_metric_values(values)
                for key, values in overall_metric_vectors.items()
            },
        },
    }


def write_suite_scorecard(summary: dict[str, Any], path: Path) -> None:
    blocker_counts = summary["overall"].get("current_bottleneck_counts", {})
    top_blocker = max(blocker_counts.items(), key=lambda item: item[1])[0] if blocker_counts else "none"
    outcome_counts = summary["overall"].get("benchmark_outcome_counts", {})
    top_outcome = max(outcome_counts.items(), key=lambda item: item[1])[0] if outcome_counts else "stalled"
    outcome_labels = {
        "formal_proof_success": "证明闭环成功",
        "theorem_correction_success": "命题纠错成功",
        "counterexample_warning": "可疑反例信号",
        "bridge_consumption_ready": "桥接已可继续消费",
        "bridge_plan_valid_only": "已生成有效桥接路径",
        "partial_progress": "局部进展",
        "stalled": "进展有限",
    }
    lines = [
        f"# {summary['display_name']}",
        "",
        "## 总览",
        "",
        f"- `suite_id`: `{summary['suite_id']}`",
        f"- `total_runs`: `{summary['overall']['total_runs']}`",
        f"- `progress_without_closure_rate`: `{summary['overall']['progress_without_closure_rate']}`",
        f"- `counterexample_rate`: `{summary['overall']['counterexample_rate']}`",
        f"- `theorem_correction_success_rate`: `{summary['overall']['theorem_correction_success_rate']}`",
        f"- `counterexample_warning_rate`: `{summary['overall']['counterexample_warning_rate']}`",
        f"- `bridge_plan_valid_rate`: `{summary['overall']['bridge_plan_valid_rate']}`",
        f"- `bridge_consumption_ready_rate`: `{summary['overall']['bridge_consumption_ready_rate']}`",
        f"- `bridge_experiment_success_rate`: `{summary['overall']['bridge_experiment_success_rate']}`",
        f"- `mean_final_target_belief`: `{summary['overall']['mean_final_target_belief']}`",
        f"- `DBI_mean`: `{summary['overall']['metrics'].get('DBI', {}).get('mean', 0.0)}`",
        f"- `ORI_mean`: `{summary['overall']['metrics'].get('ORI', {}).get('mean', 0.0)}`",
        f"- `NVI_mean`: `{summary['overall']['metrics'].get('NVI', {}).get('mean', 0.0)}`",
        f"- `OFC_mean`: `{summary['overall']['metrics'].get('OFC', {}).get('mean', 0.0)}`",
        f"- `repeats`: `{summary['repeats']}`",
        "",
        "## 当前主瓶颈分布",
        "",
        f"- `top_blocker`: `{top_blocker}`",
        f"- `bottleneck_counts`: `{summary['overall'].get('current_bottleneck_counts', {})}`",
        f"- `top_outcome`: `{top_outcome}`",
        f"- `outcome_counts`: `{summary['overall'].get('benchmark_outcome_counts', {})}`",
        "",
        "## 分题摘要",
        "",
    ]
    header = [
        "题目",
        "结果类型",
        "进展率",
        "新节点均值",
        "桥接有效率",
        "消费就绪率",
        "平均Belief",
        "DBI",
        "ORI",
        "NVI",
        "OFC",
        "平均PQI",
        "主瓶颈",
    ]
    divider = ["---", "---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(divider) + " |")
    for case_summary in summary["cases"]:
        case_blocker_counts = case_summary["current_bottleneck_counts"]
        case_top_blocker = max(case_blocker_counts.items(), key=lambda item: item[1])[0] if case_blocker_counts else "none"
        outcome_label = outcome_labels.get(case_summary.get("dominant_outcome", "stalled"), case_summary.get("dominant_outcome", "stalled"))
        row = [
            case_summary["display_name"],
            outcome_label,
            f"`{case_summary['progress_without_closure_rate']}`",
            f"`{case_summary['metrics']['new_nodes_created']['mean']}`",
            f"`{case_summary['bridge_plan_valid_rate']}`",
            f"`{case_summary['bridge_consumption_ready_rate']}`",
            f"`{case_summary['mean_final_target_belief']}`",
            f"`{case_summary['metrics']['DBI']['mean']}`",
            f"`{case_summary['metrics']['ORI']['mean']}`",
            f"`{case_summary['metrics']['NVI']['mean']}`",
            f"`{case_summary['metrics']['OFC']['mean']}`",
            f"`{case_summary['metrics']['PQI']['mean']}`",
            f"`{case_top_blocker}`",
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## 迭代建议",
            "",
            f"- 当前最常见瓶颈是 `{top_blocker}`，下一轮应优先围绕这一层做修复或细化。",
            "- 当 `high_pqi_rate` 高、`bridge_plan_valid_rate` 高，但 `bridge_consumption_ready_rate` 低时，优先优化 bridge consumer，而不是加强 Lean。",
            "- 当 `bridge_consumption_ready_rate` 提升但 `bridge_experiment_success_rate` 仍低时，说明选桥能力在进步，但局部 bridge 的真实消费动作仍然不稳。",
            "- 当 `counterexample_rate` 提升时，需区分是 `theorem_correction_success_rate` 上升，还是 `counterexample_warning_rate` 上升。",
            "- 如果 `mean_final_target_belief` 持续高但仍无法进入 `bridge_consumption_ready`，说明路径到消费层的映射仍然偏弱。",
            "- **前沿问题专用**：`DBI低+ORI低` → 种子太少/太抽象，补充更多基础工具节点；`DBI高+ORI低` → 产生了新节点但找不到真正障碍，planning 层需改进；`ORI高+NVI低` → 知道难点但提不出新方案，加强实验/Lean 反馈循环；`NVI高+DBI低` → 新路径未物化为节点，bridge materialization 有问题。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_suite(
    suite_config_path: Path,
    *,
    repeats_override: Optional[int] = None,
    output_root: Optional[Path] = None,
    max_parallel: int = 1,
    resume: bool = False,
    resume_dir: Optional[Path] = None,
    experience_buffer: Optional[ExperienceBuffer] = None,
) -> SuiteRunResult:
    """
    Run a benchmark suite, optionally in parallel and with checkpoint/resume.

    Args:
        suite_config_path: Path to suite.json.
        repeats_override: Override the per-case repeat count.
        output_root: Override the output root directory.
        max_parallel: Number of cases to run in parallel (default 1 = sequential).
        resume: If True, skip cases that already have a completed summary.json.
        resume_dir: Path to a previous suite_run_dir to resume from.
                    Implies resume=True.
    """
    import concurrent.futures as _futures
    import signal as _signal
    import threading as _threading

    if resume_dir is not None:
        resume = True

    suite = load_suite_config(suite_config_path)
    cases = [load_case_config(path) for path in suite.case_files]
    evaluation_root = output_root or DEFAULT_EVALUATION_ROOT
    if resume_dir is not None:
        suite_run_dir = Path(resume_dir).resolve()
    else:
        suite_run_dir = _unique_directory(
            evaluation_root / "runs" / suite.suite_id / _timestamp_slug()
        )
    reports_dir = evaluation_root / "reports" / suite.suite_id / suite_run_dir.name
    suite_run_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # When running in MCTS mode, automatically collect experiences so they are
    # available for expert iteration and PAV training even without explicit --buffer flag.
    _auto_buffer_save = False
    if CONFIG.enable_mcts and experience_buffer is None:
        experience_buffer = ExperienceBuffer()
        _auto_buffer_save = True

    # Checkpoint/resume: load progress.json if it exists
    progress_path = suite_run_dir / "progress.json"
    completed_case_ids: set[str] = set()
    if resume and progress_path.exists():
        try:
            progress_data = json.loads(progress_path.read_text())
            completed_case_ids = set(progress_data.get("completed", []))
        except Exception:
            pass

    repeats = int(repeats_override or suite.repeats)
    _results_lock = _threading.Lock()
    case_results: list[dict[str, Any]] = []
    default_case_timeout_seconds = int(
        max(300.0, float(getattr(CONFIG, "mcts_max_time_seconds", 0.0) or 0.0) * 1.5)
    )

    def _mark_case_completed(case_id: str) -> None:
        with _results_lock:
            completed_case_ids.add(case_id)
            try:
                progress_path.write_text(
                    json.dumps({"completed": sorted(completed_case_ids)}, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass

    def _run_one_case(case: BenchmarkCaseConfig) -> dict[str, Any]:
        case_repeats = int(case.repeats or repeats)
        run_summaries: list[dict[str, Any]] = []
        case_dir = suite_run_dir / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        for repeat_index in range(1, case_repeats + 1):
            run_dir = case_dir / f"run_{repeat_index:02d}"
            # Check per-run checkpoint
            summary_checkpoint = run_dir / "summary.json"
            if resume and summary_checkpoint.exists():
                try:
                    run_summaries.append(json.loads(summary_checkpoint.read_text()))
                    continue
                except Exception:
                    pass

            run_summary = run_case_once(
                case,
                run_dir=run_dir,
                repeat_index=repeat_index,
                suite_id=suite.suite_id,
                experience_buffer=experience_buffer,
            )
            run_summaries.append(run_summary)

            # Save per-run checkpoint
            try:
                run_dir.mkdir(parents=True, exist_ok=True)
                summary_checkpoint.write_text(
                    json.dumps(run_summary, ensure_ascii=False, default=str, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass

        agg = aggregate_case_runs(case, run_summaries)

        # Update progress
        _mark_case_completed(case.case_id)

        return agg

    def _run_one_case_with_timeout(case: BenchmarkCaseConfig) -> dict[str, Any]:
        timeout_seconds = default_case_timeout_seconds
        if timeout_seconds <= 0:
            return _run_one_case(case)
        if _threading.current_thread() is not _threading.main_thread():
            # signal-based timeout only works on main thread; fall back to normal call.
            return _run_one_case(case)

        def _raise_timeout(_signum: int, _frame: Any) -> None:
            raise TimeoutError(
                f"Case '{case.case_id}' timed out after {timeout_seconds}s."
            )

        previous_handler = _signal.getsignal(_signal.SIGALRM)
        try:
            _signal.signal(_signal.SIGALRM, _raise_timeout)
            _signal.alarm(timeout_seconds)
            return _run_one_case(case)
        finally:
            _signal.alarm(0)
            _signal.signal(_signal.SIGALRM, previous_handler)

    def _flush_suite_report() -> tuple[Path, Path]:
        """Generate suite summary and scorecard from whatever results are available."""
        ss = aggregate_suite_results(
            suite=suite,
            cases=cases,
            case_results=case_results,
            suite_run_dir=suite_run_dir,
            actual_repeats=repeats,
        )
        sp = reports_dir / "suite_summary.json"
        sc = reports_dir / "suite_scorecard_zh.md"
        _save_json(sp, ss)
        write_suite_scorecard(ss, sc)
        return sp, sc

    suite_summary_path = reports_dir / "suite_summary.json"
    suite_scorecard_path = reports_dir / "suite_scorecard_zh.md"

    try:
        if max_parallel > 1:
            with _futures.ThreadPoolExecutor(max_workers=max_parallel) as pool:
                pending_cases = [c for c in cases if c.case_id not in completed_case_ids]
                future_to_case = {pool.submit(_run_one_case, c): c for c in pending_cases}
                for future in _futures.as_completed(future_to_case):
                    case = future_to_case[future]
                    try:
                        case_results.append(future.result(timeout=default_case_timeout_seconds))
                    except _futures.TimeoutError:
                        case_results.append({
                            "case_id": case.case_id,
                            "error": f"Case exceeded wall-clock timeout ({default_case_timeout_seconds}s)",
                            "status": "timeout",
                        })
                    except Exception as exc:
                        case_results.append({
                            "case_id": case.case_id,
                            "error": str(exc),
                            "status": "error",
                        })
        else:
            for case in cases:
                if resume and case.case_id in completed_case_ids:
                    case_dir = suite_run_dir / case.case_id
                    found_summaries: list[dict[str, Any]] = []
                    for run_dir in sorted(case_dir.glob("run_*")) if case_dir.exists() else []:
                        sp = run_dir / "summary.json"
                        if sp.exists():
                            try:
                                found_summaries.append(json.loads(sp.read_text()))
                            except Exception:
                                pass
                    if found_summaries:
                        case_results.append(aggregate_case_runs(case, found_summaries))
                        continue
                try:
                    case_results.append(_run_one_case_with_timeout(case))
                except TimeoutError as exc:
                    _mark_case_completed(case.case_id)
                    case_results.append(
                        {
                            "case_id": case.case_id,
                            "error": str(exc),
                            "status": "timeout",
                        }
                    )
                except Exception as exc:
                    _mark_case_completed(case.case_id)
                    case_results.append(
                        {
                            "case_id": case.case_id,
                            "error": str(exc),
                            "status": "error",
                        }
                    )
    finally:
        suite_summary_path, suite_scorecard_path = _flush_suite_report()

    # Persist collected experiences when auto-buffer was created.
    if _auto_buffer_save and experience_buffer is not None and len(experience_buffer) > 0:
        try:
            buffer_path = suite_run_dir / "experience_buffer.json"
            experience_buffer.save(buffer_path)
        except Exception:
            pass

    if (
        CONFIG.expert_iter_enabled
        and experience_buffer is not None
        and len(experience_buffer) >= 100
    ):
        loop = ExpertIterationLoop(
            experience_buffer=experience_buffer,
            checkpoint_dir=Path(CONFIG.expert_iter_checkpoint_dir),
        )
        try:
            loop.run_iteration()
        except Exception:
            pass
    return SuiteRunResult(
        suite_run_dir=suite_run_dir,
        suite_summary_path=suite_summary_path,
        suite_scorecard_path=suite_scorecard_path,
    )
