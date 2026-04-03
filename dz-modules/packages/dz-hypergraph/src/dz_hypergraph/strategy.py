"""Exploration strategy: value-uncertainty trade-off and module selection."""

from dz_hypergraph.models import HyperGraph, Module

BRIDGE_READY_MARKER = "Bridge consumer:"
BRIDGE_READY_BONUS = 0.75


def _node_value(graph: HyperGraph, node_id: str) -> float:
    incoming = len(graph.get_edges_to(node_id))
    outgoing = len(graph.get_edges_from(node_id))
    return 1.0 + 0.3 * incoming + 0.5 * outgoing


def _bridge_ready_bonus(graph: HyperGraph, node_id: str) -> float:
    bonus = 0.0
    for edge_id in graph.get_edges_to(node_id):
        edge = graph.edges[edge_id]
        if any(BRIDGE_READY_MARKER in step for step in edge.steps):
            bonus = max(bonus, BRIDGE_READY_BONUS * edge.confidence)
    return bonus


def rank_nodes(graph: HyperGraph) -> list[tuple[str, float]]:
    ranked = []
    for nid, node in graph.nodes.items():
        if getattr(node, "is_locked", lambda: False)():
            continue
        if node.belief == 1.0:
            continue
        value = _node_value(graph, nid)
        priority = value * (1.0 - node.belief) + _bridge_ready_bonus(graph, nid)
        ranked.append((nid, priority))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def suggest_module(graph: HyperGraph, node_id: str) -> Module:
    belief = graph.nodes[node_id].belief
    if belief < 0.4:
        return Module.PLAUSIBLE
    elif belief < 0.8:
        return Module.EXPERIMENT
    else:
        return Module.LEAN
