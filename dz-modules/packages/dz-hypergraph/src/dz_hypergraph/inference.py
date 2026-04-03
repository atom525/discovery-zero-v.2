"""Belief propagation on the reasoning hypergraph via Gaia's loopy BP engine.

Supports:
  - Full-graph BP (default)
  - Incremental BP: only re-propagate the subgraph affected by changed edges
  - Warmstart: initialise Gaia BP from the current beliefs rather than priors
  - InferenceConfig: typed configuration object replacing hard-coded defaults
  - Optional NeuralBPCorrector post-processing when neural_bp_enabled is set
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Set

from gaia.bp.engine import EngineConfig, InferenceEngine
from gaia.bp.factor_graph import CROMWELL_EPS

from dz_hypergraph.config import CONFIG
from dz_hypergraph import adapter as adapter_v1
from dz_hypergraph.adapter import run_gaia_bp, write_back_beliefs
from dz_hypergraph.adapter_v2 import adapt_zero_graph_v2
from dz_hypergraph.memo import VerificationResult
from dz_hypergraph.models import HyperGraph

logger = logging.getLogger(__name__)

# Module-level Neural BP corrector, loaded lazily based on CONFIG.
_neural_bp_corrector: Optional[Any] = None
_neural_bp_corrector_loaded: bool = False


def _get_neural_bp_corrector() -> Optional[Any]:
    """Return a NeuralBPCorrector instance if configured, else None.

    Loaded once per process; subsequent calls reuse the cached instance.
    Uses CONFIG.neural_bp_enabled and CONFIG.neural_bp_model_path.
    """
    global _neural_bp_corrector, _neural_bp_corrector_loaded
    if _neural_bp_corrector_loaded:
        return _neural_bp_corrector
    _neural_bp_corrector_loaded = True
    try:
        from dz_hypergraph.config import CONFIG
        if not getattr(CONFIG, "neural_bp_enabled", False):
            return None
        model_path_str = getattr(CONFIG, "neural_bp_model_path", "")
        if not model_path_str:
            return None
        from pathlib import Path
        from dz_hypergraph.neural_bp import NeuralBPCorrector
        model_path = Path(model_path_str)
        if not model_path.exists():
            logger.warning(
                "neural_bp_model_path %s does not exist; Neural BP disabled.", model_path
            )
            return None
        strength = float(getattr(CONFIG, "neural_bp_correction_strength", 0.3))
        corrector = NeuralBPCorrector(
            model_path=model_path,
            correction_strength=strength,
        )
        _neural_bp_corrector = corrector
        logger.info("NeuralBPCorrector loaded from %s (strength=%.2f).", model_path, strength)
        return corrector
    except Exception as exc:
        logger.warning("Failed to load NeuralBPCorrector: %s", exc)
        return None


REFUTATION_PENALTY_RATIO = 0.25


# ------------------------------------------------------------------ #
# Configuration                                                        #
# ------------------------------------------------------------------ #

@dataclass
class InferenceConfig:
    """Typed configuration for a BP run."""

    max_iterations: int = 50
    damping: float = 0.5
    tol: float = 1e-6
    refutation_penalty_ratio: float = REFUTATION_PENALTY_RATIO

    @classmethod
    def from_config(cls) -> "InferenceConfig":
        """Build from the global ZeroConfig singleton."""
        try:
            from dz_hypergraph.config import CONFIG
            return cls(
                max_iterations=CONFIG.bp_max_iterations,
                damping=CONFIG.bp_damping,
                tol=CONFIG.bp_tolerance,
                refutation_penalty_ratio=REFUTATION_PENALTY_RATIO,
            )
        except Exception:
            return cls()


_DEFAULT_INFERENCE_CONFIG = InferenceConfig()

# BP marginals are not Cromwell-clamped on output; values slightly past the
# theoretical (ε, 1−ε) band are common from loopy BP float error. Warn only
# when clearly invalid or far outside the band.
_CROMWELL_BP_DIAG_ATOL = 5e-4


def _bp_marginal_should_warn_outside_cromwell(belief: float) -> bool:
    if not (0.0 <= belief <= 1.0):
        return True
    lo = CROMWELL_EPS - _CROMWELL_BP_DIAG_ATOL
    hi = 1.0 - CROMWELL_EPS + _CROMWELL_BP_DIAG_ATOL
    return belief < lo or belief > hi


def run_inference_v2(
    graph: HyperGraph,
    *,
    warmstart: bool = False,
    config: Optional[InferenceConfig] = None,
) -> adapter_v1.InferenceResult:
    """Run inference_v2 and convert output to v1-compatible InferenceResult."""
    adapted = adapt_zero_graph_v2(graph, warmstart=warmstart)
    eff_config = config or _DEFAULT_INFERENCE_CONFIG
    engine = InferenceEngine(
        EngineConfig(
            bp_max_iter=eff_config.max_iterations,
            bp_damping=eff_config.damping,
            bp_threshold=eff_config.tol,
        )
    )
    v2_result = engine.run(adapted.factor_graph, method="auto")
    node_beliefs = {
        var_id: belief
        for var_id, belief in v2_result.beliefs.items()
        if var_id not in adapted.synthetic_var_ids
    }
    for var_id, belief in node_beliefs.items():
        if _bp_marginal_should_warn_outside_cromwell(belief):
            logger.warning(
                "BP marginal outside Cromwell band (check numerical stability): %s=%.8g",
                var_id,
                belief,
            )
    return adapter_v1.InferenceResult(
        node_beliefs=node_beliefs,
        converged=bool(v2_result.is_exact or v2_result.diagnostics.converged),
        iterations=int(v2_result.diagnostics.iterations_run),
        diagnostics=v2_result.diagnostics,
    )


@dataclass
class SignalAccumulator:
    """Accumulate deterministic verification signals for threshold-triggered BP."""

    threshold: int
    pending_signals: int = 0

    def add(self, count: int) -> bool:
        self.pending_signals += max(0, count)
        return self.pending_signals >= self.threshold

    def clear(self) -> None:
        self.pending_signals = 0


# ------------------------------------------------------------------ #
# Incremental subgraph selection                                       #
# ------------------------------------------------------------------ #

def _collect_affected_subgraph(
    graph: HyperGraph,
    changed_edge_ids: Set[str],
) -> tuple[Set[str], Set[str]]:
    """
    From a set of changed edge IDs, BFS downstream to collect all nodes and
    edges that could be affected by the change.

    Returns (affected_node_ids, affected_edge_ids).
    """
    affected_nodes: Set[str] = set()
    affected_edges: Set[str] = set(changed_edge_ids)

    # Seeds: conclusion nodes of changed edges
    frontier: list[str] = []
    for eid in changed_edge_ids:
        edge = graph.edges.get(eid)
        if edge is not None:
            frontier.append(edge.conclusion_id)

    # BFS: follow outgoing edges of affected nodes
    while frontier:
        nid = frontier.pop()
        if nid in affected_nodes:
            continue
        affected_nodes.add(nid)
        for eid in graph.get_edges_from(nid):
            if eid not in affected_edges:
                affected_edges.add(eid)
                edge = graph.edges.get(eid)
                if edge is not None and edge.conclusion_id not in affected_nodes:
                    frontier.append(edge.conclusion_id)

    return affected_nodes, affected_edges


# ------------------------------------------------------------------ #
# Main propagation entry-points                                        #
# ------------------------------------------------------------------ #

def propagate_beliefs(
    graph: HyperGraph,
    max_iterations: Optional[int] = None,
    damping: Optional[float] = None,
    tol: Optional[float] = None,
    *,
    changed_edge_ids: Optional[Set[str]] = None,
    warmstart: bool = False,
    config: Optional[InferenceConfig] = None,
) -> int:
    """
    Update beliefs using Gaia's loopy BP.

    Args:
        graph: The reasoning hypergraph to update in-place.
        max_iterations: Maximum BP iterations.
        damping: Message-passing damping factor in [0, 1].
        tol: Convergence tolerance.
        changed_edge_ids: If provided, only the subgraph reachable from
            these changed edges is re-run through BP (incremental mode).
            When None, the full graph is updated.
        warmstart: If True, use current beliefs (instead of priors) as BP
            starting point.  Useful for incremental re-computation when
            beliefs are already meaningful.  Default False so that the
            first full-graph run uses the original priors.
        config: Optional typed configuration override.

    Returns:
        Number of BP iterations performed.
    """
    # Config priority: explicit params > InferenceConfig > defaults
    eff_config = config or _DEFAULT_INFERENCE_CONFIG
    eff_max_iter = max_iterations if max_iterations is not None else eff_config.max_iterations
    eff_damping = damping if damping is not None else eff_config.damping
    eff_tol = tol if tol is not None else eff_config.tol

    backend = getattr(CONFIG, "bp_backend", "gaia")

    if changed_edge_ids is not None and len(changed_edge_ids) > 0 and backend != "gaia_v2":
        affected_nodes, affected_edges = _collect_affected_subgraph(graph, changed_edge_ids)
        if len(affected_nodes) == 0:
            return 0
        # Build a subgraph view for BP.
        # Approach: temporarily create a sub-HyperGraph with only affected nodes/edges,
        # run BP on it, write back, then apply penalties on full graph.
        sub_graph = _build_subgraph(graph, affected_nodes, affected_edges)
        result = run_gaia_bp(
            sub_graph,
            max_iterations=eff_max_iter,
            damping=eff_damping,
            tol=eff_tol,
            warmstart=warmstart,
        )
        write_back_beliefs(sub_graph, result)
        # Write beliefs back to the full graph
        for nid, node in sub_graph.nodes.items():
            if nid in graph.nodes and not graph.nodes[nid].is_locked():
                graph.nodes[nid].belief = node.belief
        return result.iterations

    # Full-graph BP
    if backend == "gaia_v2":
        try:
            result = run_inference_v2(
                graph,
                warmstart=warmstart,
                config=eff_config,
            )
        except Exception as exc:
            logger.warning("gaia_v2 inference failed (%s); fallback to gaia v1", exc)
            result = run_gaia_bp(
                graph,
                max_iterations=eff_max_iter,
                damping=eff_damping,
                tol=eff_tol,
                warmstart=warmstart,
            )
    else:
        result = run_gaia_bp(
            graph,
            max_iterations=eff_max_iter,
            damping=eff_damping,
            tol=eff_tol,
            warmstart=warmstart,
        )
    write_back_beliefs(graph, result)

    # Optional Neural BP correction: adjusts beliefs using a trained GNN corrector.
    # Only applied when CONFIG.neural_bp_enabled is True and a model path is set.
    corrector = _get_neural_bp_corrector()
    if corrector is not None:
        try:
            corrector.apply_to_graph(graph)
        except Exception as exc:
            logger.warning("NeuralBPCorrector.apply_to_graph failed: %s", exc)

    return result.iterations


def propagate_verification_signals(
    graph: HyperGraph,
    verification_results: list[VerificationResult | dict[str, Any]],
    *,
    threshold: int = 3,
    accumulator: Optional[SignalAccumulator] = None,
    force: bool = False,
    max_iterations: int = 50,
    damping: float = 0.5,
    tol: float = 1e-6,
) -> int:
    """Trigger BP for deterministic verification outcomes.

    IMPORTANT:
    - Prior/state updates happen in ingestion (`ingest_verified_claim`).
    - This function only accumulates deterministic signals and schedules BP.
    """
    deterministic_count = 0
    for item in verification_results:
        if isinstance(item, VerificationResult):
            verdict = item.verdict
            claim_id = item.claim_id
        else:
            verdict = str(item.get("verdict", "inconclusive")).strip().lower()
            claim_id = str(item.get("claim_id", "")).strip()
        if verdict not in {"verified", "refuted"}:
            continue
        target_node = None
        if claim_id and claim_id in graph.nodes:
            target_node = graph.nodes[claim_id]
        elif claim_id:
            matches = graph.find_node_ids_by_statement(claim_id)
            if matches:
                target_node = graph.nodes[matches[0]]
        if target_node is None:
            continue
        deterministic_count += 1

    effective_acc = accumulator or SignalAccumulator(threshold=max(1, threshold))
    should_run = effective_acc.add(deterministic_count)
    if force and effective_acc.pending_signals > 0:
        should_run = True
    if not should_run:
        return 0
    iterations = propagate_beliefs(
        graph,
        max_iterations=max_iterations,
        damping=damping,
        tol=tol,
        warmstart=False if getattr(CONFIG, "bp_backend", "gaia") == "gaia_v2" else True,
    )
    effective_acc.clear()
    return iterations


def _build_subgraph(
    graph: HyperGraph,
    affected_node_ids: Set[str],
    affected_edge_ids: Set[str],
) -> HyperGraph:
    """
    Build a sub-HyperGraph containing only the specified nodes and edges.

    Nodes that are premises/conclusions of included edges but not in
    affected_node_ids are included as "context" nodes (with their current
    belief as fixed priors).
    """
    from dz_hypergraph.models import HyperGraph as HG, Node, Hyperedge

    sub = HG()

    # Collect all referenced node IDs (including non-affected premises)
    all_nids: Set[str] = set(affected_node_ids)
    for eid in affected_edge_ids:
        edge = graph.edges.get(eid)
        if edge:
            all_nids.update(edge.premise_ids)
            all_nids.add(edge.conclusion_id)

    # Add nodes
    for nid in all_nids:
        if nid in graph.nodes:
            sub.nodes[nid] = graph.nodes[nid].model_copy()

    # Add edges
    for eid in affected_edge_ids:
        if eid in graph.edges:
            sub.edges[eid] = graph.edges[eid].model_copy()

    # Rebuild adjacency indices
    sub.model_post_init(None)
    return sub
