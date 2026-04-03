"""Hypergraph package for Discovery Zero modular stack."""

from dz_hypergraph.belief_gap import BeliefGapAnalyser
from dz_hypergraph.bridge_models import (
    BridgePlan,
    BridgeProposition,
    BridgeReasoningStep,
    BridgeValidationError,
    validate_bridge_plan_payload,
)
from dz_hypergraph.config import CONFIG, ZeroConfig
from dz_hypergraph.inference import SignalAccumulator, propagate_beliefs, propagate_verification_signals
from dz_hypergraph.models import (
    EdgeType,
    HyperGraph,
    Hyperedge,
    Module,
    Node,
    NodeState,
    canonicalize_statement_text,
)
from dz_hypergraph.persistence import load_graph, save_graph


def create_graph() -> HyperGraph:
    """Create an empty reasoning hypergraph."""
    return HyperGraph()


def analyze_belief_gaps(
    graph: HyperGraph,
    target_node_id: str,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Convenience wrapper around BeliefGapAnalyser."""
    return BeliefGapAnalyser().find_critical_gaps(
        graph=graph,
        target_node_id=target_node_id,
        top_k=top_k,
    )
