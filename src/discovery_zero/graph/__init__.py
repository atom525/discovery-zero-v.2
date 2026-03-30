"""Graph layer: data model, inference, persistence (depends on Gaia)."""

from discovery_zero.graph.models import (
    Node,
    Hyperedge,
    HyperGraph,
    Module,
    NodeState,
    EdgeType,
    canonicalize_statement_text,
)
from discovery_zero.graph.inference import (
    propagate_beliefs,
    apply_refutation_penalties,
    propagate_verification_signals,
    SignalAccumulator,
)
from discovery_zero.graph.inference_energy import propagate_beliefs_energy
from discovery_zero.graph.ingest import ingest_skill_output
from discovery_zero.graph.persistence import save_graph, load_graph
from discovery_zero.graph.strategy import rank_nodes, suggest_module
from discovery_zero.graph.adapter import (
    ZeroInferenceGraph,
    InferenceResult,
    adapt_zero_graph,
    run_gaia_bp,
    write_back_beliefs,
)
