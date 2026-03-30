from __future__ import annotations

from libs.inference_v2.factor_graph import FactorType

from discovery_zero.graph.adapter_v2 import adapt_zero_graph_v2
from discovery_zero.graph.models import HyperGraph, Module


def test_adapter_v2_maps_heuristic_formal_and_skips_decomposition():
    graph = HyperGraph()
    a = graph.add_node("A", belief=0.8, prior=0.8)
    b = graph.add_node("B", belief=0.2, prior=0.2)
    c = graph.add_node("C", belief=0.2, prior=0.2)
    graph.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, ["h"], confidence=0.7, edge_type="heuristic")
    graph.add_hyperedge([b.id], c.id, Module.LEAN, ["f"], confidence=0.99, edge_type="formal")
    graph.add_hyperedge([a.id], c.id, Module.DECOMPOSE, ["d"], confidence=0.5, edge_type="decomposition")

    adapted = adapt_zero_graph_v2(graph)
    factor_types = [f.factor_type for f in adapted.factor_graph.factors]

    assert factor_types.count(FactorType.SOFT_IMPLICATION) == 2
    assert all(f.factor_type != FactorType.CONJUNCTION for f in adapted.factor_graph.factors)


def test_adapter_v2_creates_relation_vars_for_constraints():
    graph = HyperGraph()
    p_refuted = graph.add_node("Refuted premise", belief=0.0, prior=0.0, state="refuted")
    p_ok = graph.add_node("Healthy premise", belief=0.8, prior=0.8)
    c = graph.add_node("Conclusion", belief=0.4, prior=0.4)
    a = graph.add_node("A", belief=0.4, prior=0.4)
    b = graph.add_node("B", belief=0.4, prior=0.4)

    # Contradiction pattern: two edges to same conclusion, one side uses refuted premise.
    graph.add_hyperedge([p_refuted.id], c.id, Module.PLAUSIBLE, ["r"], confidence=0.6)
    graph.add_hyperedge([p_ok.id], c.id, Module.PLAUSIBLE, ["s"], confidence=0.6)
    # Equivalence pattern: A->B and B->A.
    graph.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, ["ab"], confidence=0.7)
    graph.add_hyperedge([b.id], a.id, Module.PLAUSIBLE, ["ba"], confidence=0.7)

    adapted = adapt_zero_graph_v2(graph)
    contra_factors = [f for f in adapted.factor_graph.factors if f.factor_type == FactorType.CONTRADICTION]
    equiv_factors = [f for f in adapted.factor_graph.factors if f.factor_type == FactorType.EQUIVALENCE]

    assert contra_factors
    assert equiv_factors
    assert adapted.synthetic_var_ids
    for factor in contra_factors + equiv_factors:
        assert factor.relation_var is not None
        assert factor.relation_var in adapted.factor_graph.variables
