"""Zero HyperGraph <-> Gaia FactorGraph adapter.

Converts Zero's HyperGraph into Gaia's FactorGraph for loopy belief propagation,
runs inference, and writes posterior beliefs back into the Zero graph.

Edge-type mapping uses Gaia's full semantic vocabulary:
  - formal (Lean-verified)  -> "deduction" at near-certainty
  - heuristic (plausible/experiment) -> "deduction" at stated confidence
  - contradiction pairs      -> "relation_contradiction" gated factor
  - equivalence annotations  -> "relation_equivalence" gated factor
  - decomposition            -> excluded from BP (structural only)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from libs.inference import FactorGraph, BeliefPropagation, InconsistentGraphError
from libs.inference.bp import BPDiagnostics
from libs.inference.factor_graph import CROMWELL_EPS

from dz_hypergraph.models import HyperGraph

logger = logging.getLogger(__name__)


@dataclass
class ZeroInferenceGraph:
    """Mapping between Zero node/edge IDs and Gaia variable/factor indices."""

    factor_graph: FactorGraph
    node_id_to_var_id: dict[str, int] = field(default_factory=dict)
    var_id_to_node_id: dict[int, str] = field(default_factory=dict)
    edge_id_to_factor_idx: dict[str, int] = field(default_factory=dict)
    excluded_edge_ids: set[str] = field(default_factory=set)
    node_priors: dict[str, float] = field(default_factory=dict)


@dataclass
class InferenceResult:
    """Result of a Gaia BP run mapped back to Zero node IDs."""

    node_beliefs: dict[str, float] = field(default_factory=dict)
    converged: bool = False
    iterations: int = 0
    diagnostics: BPDiagnostics | None = None


def _detect_contradictions(graph: HyperGraph) -> list[tuple[str, str, str]]:
    """Find pairs of edges that give contradictory conclusions for the same node.

    A contradiction is detected when one edge supports a node (heuristic/formal
    leading to conclusion X) and another edge's conclusion *refutes* X, or when
    two edges reach the same conclusion from incompatible premise sets where one
    premise set contains a refuted node.

    Returns list of (edge_id_a, edge_id_b, conclusion_node_id).
    """
    edges_by_conclusion: dict[str, list[str]] = defaultdict(list)
    for eid, edge in graph.edges.items():
        if edge.edge_type != "decomposition":
            edges_by_conclusion[edge.conclusion_id].append(eid)

    contradictions: list[tuple[str, str, str]] = []
    for cid, eids in edges_by_conclusion.items():
        if len(eids) < 2:
            continue
        for i, eid_a in enumerate(eids):
            edge_a = graph.edges[eid_a]
            for eid_b in eids[i + 1:]:
                edge_b = graph.edges[eid_b]
                a_has_refuted = any(
                    graph.nodes.get(p) and graph.nodes[p].state == "refuted"
                    for p in edge_a.premise_ids
                )
                b_has_refuted = any(
                    graph.nodes.get(p) and graph.nodes[p].state == "refuted"
                    for p in edge_b.premise_ids
                )
                if a_has_refuted != b_has_refuted:
                    contradictions.append((eid_a, eid_b, cid))
    return contradictions


def _detect_equivalences(graph: HyperGraph) -> list[tuple[str, str]]:
    """Find node pairs connected by symmetric edges (A->B and B->A exist).

    Self-loops (A->A) are excluded.
    Returns list of (node_id_a, node_id_b).
    """
    directed: set[tuple[str, str]] = set()
    for edge in graph.edges.values():
        if edge.edge_type == "decomposition":
            continue
        if len(edge.premise_ids) == 1 and edge.premise_ids[0] != edge.conclusion_id:
            directed.add((edge.premise_ids[0], edge.conclusion_id))

    equivalences: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()
    for a, b in directed:
        if a != b and (b, a) in directed:
            pair = frozenset((a, b))
            if pair not in seen:
                seen.add(pair)
                equivalences.append((a, b))
    return equivalences


def adapt_zero_graph(
    graph: HyperGraph,
    *,
    warmstart: bool = False,
) -> ZeroInferenceGraph:
    """Convert a Zero HyperGraph to a Gaia FactorGraph.

    - Each Zero node becomes a Gaia variable with its prior.
    - Proven nodes are clamped to 1 - CROMWELL_EPS.
    - Refuted nodes are clamped to CROMWELL_EPS.
    - Decomposition edges are excluded from BP.
    - Formal (Lean) edges -> "deduction" with high confidence.
    - Heuristic edges -> "deduction" at stated confidence.
    - Detected contradiction pairs -> "relation_contradiction" gated factor.
    - Detected equivalences -> "relation_equivalence" gated factor.
    - warmstart: use current belief instead of prior as the BP input.
    """
    fg = FactorGraph()
    zig = ZeroInferenceGraph(factor_graph=fg)

    for var_int_id, (nid, node) in enumerate(graph.nodes.items()):
        if warmstart and not node.is_locked():
            prior = max(CROMWELL_EPS, min(1.0 - CROMWELL_EPS, node.belief))
        else:
            prior = node.prior
        if node.state == "proven":
            prior = 1.0 - CROMWELL_EPS
        elif node.state == "refuted":
            prior = CROMWELL_EPS
        fg.add_variable(var_int_id, prior)
        zig.node_id_to_var_id[nid] = var_int_id
        zig.var_id_to_node_id[var_int_id] = nid
        zig.node_priors[nid] = prior

    factor_idx = 0
    for eid, edge in graph.edges.items():
        if edge.edge_type == "decomposition":
            zig.excluded_edge_ids.add(eid)
            continue

        premise_var_ids = [
            zig.node_id_to_var_id[pid]
            for pid in edge.premise_ids
            if pid in zig.node_id_to_var_id
        ]
        conclusion_var_id = zig.node_id_to_var_id.get(edge.conclusion_id)
        if conclusion_var_id is None:
            zig.excluded_edge_ids.add(eid)
            continue

        if edge.edge_type == "formal":
            prob = max(edge.confidence, 1.0 - CROMWELL_EPS)
        else:
            prob = edge.confidence

        fg.add_factor(
            edge_id=factor_idx,
            premises=premise_var_ids,
            conclusions=[conclusion_var_id],
            probability=prob,
            edge_type="deduction",
        )
        zig.edge_id_to_factor_idx[eid] = factor_idx
        factor_idx += 1

    # Contradiction factors: gated factors penalising jointly-true contradictory paths
    for eid_a, eid_b, cid in _detect_contradictions(graph):
        edge_a = graph.edges[eid_a]
        edge_b = graph.edges[eid_b]
        a_premises = [zig.node_id_to_var_id[p] for p in edge_a.premise_ids if p in zig.node_id_to_var_id]
        b_premises = [zig.node_id_to_var_id[p] for p in edge_b.premise_ids if p in zig.node_id_to_var_id]
        gate_var = zig.node_id_to_var_id.get(cid)
        if gate_var is None:
            continue
        all_premises = list(set(a_premises + b_premises))
        if not all_premises:
            continue
        fg.add_factor(
            edge_id=factor_idx,
            premises=all_premises,
            conclusions=[],
            probability=CROMWELL_EPS,
            edge_type="relation_contradiction",
            gate_var=gate_var,
        )
        zig.edge_id_to_factor_idx[f"_contra_{eid_a}_{eid_b}"] = factor_idx
        factor_idx += 1

    # Equivalence factors: symmetric belief coupling (requires exactly 2 premises)
    for nid_a, nid_b in _detect_equivalences(graph):
        var_a = zig.node_id_to_var_id.get(nid_a)
        var_b = zig.node_id_to_var_id.get(nid_b)
        if var_a is None or var_b is None:
            continue
        fg.add_factor(
            edge_id=factor_idx,
            premises=[var_a, var_b],
            conclusions=[],
            probability=1.0 - CROMWELL_EPS,
            edge_type="relation_equivalence",
        )
        zig.edge_id_to_factor_idx[f"_equiv_{nid_a}_{nid_b}"] = factor_idx
        factor_idx += 1

    return zig


def write_back_beliefs(graph: HyperGraph, result: InferenceResult) -> None:
    """Write Gaia BP posterior beliefs back into Zero's nodes.

    Locked nodes (proven/refuted) are skipped. Beliefs are clamped to
    [CROMWELL_EPS, 1-CROMWELL_EPS].
    """
    for nid, belief in result.node_beliefs.items():
        node = graph.nodes.get(nid)
        if node is None:
            continue
        if node.is_locked():
            continue
        clamped = max(CROMWELL_EPS, min(1.0 - CROMWELL_EPS, belief))
        node.belief = clamped


def run_gaia_bp(
    graph: HyperGraph,
    *,
    max_iterations: int = 50,
    damping: float = 0.5,
    tol: float = 1e-6,
    diagnostics: bool = True,
    warmstart: bool = False,
) -> InferenceResult:
    """Build a Gaia FactorGraph from a Zero HyperGraph and run loopy BP.

    Args:
        warmstart: If True, use current beliefs (not priors) as BP starting
            point.  Accelerates convergence on incremental updates.

    Returns an InferenceResult with beliefs mapped back to Zero node IDs.
    """
    zig = adapt_zero_graph(graph, warmstart=warmstart)

    if not zig.factor_graph.variables:
        return InferenceResult(node_beliefs={}, converged=True, iterations=0)

    bp = BeliefPropagation(
        damping=damping,
        max_iterations=max_iterations,
        convergence_threshold=tol,
    )

    try:
        if diagnostics:
            beliefs_by_var, diag = bp.run_with_diagnostics(zig.factor_graph)
        else:
            beliefs_by_var = bp.run(zig.factor_graph)
            diag = None
    except InconsistentGraphError as exc:
        logger.error(
            "InconsistentGraphError during BP on graph with %d nodes, %d edges: %s. "
            "Returning current beliefs unchanged. The graph likely contains "
            "contradictory hard constraints that prevent convergence.",
            len(graph.nodes),
            len(graph.edges),
            exc,
        )
        return InferenceResult(
            node_beliefs={nid: graph.nodes[nid].belief for nid in graph.nodes},
            converged=False,
            iterations=0,
            diagnostics=None,
        )

    node_beliefs: dict[str, float] = {}
    for var_id, belief in beliefs_by_var.items():
        nid = zig.var_id_to_node_id.get(var_id)
        if nid is not None:
            node_beliefs[nid] = belief

    converged = diag.converged if diag is not None else True
    iterations = diag.iterations_run if diag is not None else 0

    return InferenceResult(
        node_beliefs=node_beliefs,
        converged=converged,
        iterations=iterations,
        diagnostics=diag,
    )
