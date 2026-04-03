from __future__ import annotations

import pytest

from gaia.bp.factor_graph import CROMWELL_EPS, FactorType

from dz_hypergraph.adapter_v2 import adapt_zero_graph_v2
from dz_hypergraph.models import HyperGraph, Module


def test_adapter_v2_maps_heuristic_formal_and_includes_decomposition():
    graph = HyperGraph()
    a = graph.add_node("A", belief=0.8, prior=0.8)
    b = graph.add_node("B", belief=0.2, prior=0.2)
    c = graph.add_node("C", belief=0.2, prior=0.2)
    graph.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, ["h"], confidence=0.7, edge_type="heuristic")
    graph.add_hyperedge([b.id], c.id, Module.LEAN, ["f"], confidence=0.99, edge_type="formal")
    graph.add_hyperedge([a.id], c.id, Module.DECOMPOSE, ["d"], confidence=0.5, edge_type="decomposition")

    adapted = adapt_zero_graph_v2(graph)
    factor_types = [f.factor_type for f in adapted.factor_graph.factors]

    assert factor_types.count(FactorType.SOFT_ENTAILMENT) == 1
    assert factor_types.count(FactorType.IMPLICATION) == 2
    assert all(f.factor_type != FactorType.CONJUNCTION for f in adapted.factor_graph.factors)


def test_decomposition_uses_deterministic_conjunction_then_implication():
    graph = HyperGraph()
    a = graph.add_node("A", belief=0.05, prior=0.05)
    b = graph.add_node("B", belief=0.05, prior=0.05)
    c = graph.add_node("C", belief=0.5, prior=0.5)
    graph.add_hyperedge(
        [a.id, b.id],
        c.id,
        Module.DECOMPOSE,
        ["d"],
        confidence=0.72,
        edge_type="decomposition",
    )
    adapted = adapt_zero_graph_v2(graph)
    se_factors = [f for f in adapted.factor_graph.factors if f.factor_type == FactorType.SOFT_ENTAILMENT]
    imp_factors = [f for f in adapted.factor_graph.factors if f.factor_type == FactorType.IMPLICATION]
    conj_factors = [f for f in adapted.factor_graph.factors if f.factor_type == FactorType.CONJUNCTION]
    assert not se_factors
    assert len(imp_factors) == 1
    assert len(conj_factors) == 1


def test_adapter_v2_creates_helper_claims_for_constraints():
    graph = HyperGraph()
    p_refuted = graph.add_node("Refuted premise", belief=0.0, prior=0.0, state="refuted")
    p_ok = graph.add_node("Healthy premise", belief=0.8, prior=0.8)
    c = graph.add_node("Conclusion", belief=0.4, prior=0.4)
    a = graph.add_node("A", belief=0.4, prior=0.4)
    b = graph.add_node("B", belief=0.4, prior=0.4)

    graph.add_hyperedge([p_refuted.id], c.id, Module.PLAUSIBLE, ["r"], confidence=0.6)
    graph.add_hyperedge([p_ok.id], c.id, Module.PLAUSIBLE, ["s"], confidence=0.6)
    graph.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, ["ab"], confidence=0.7)
    graph.add_hyperedge([b.id], a.id, Module.PLAUSIBLE, ["ba"], confidence=0.7)

    adapted = adapt_zero_graph_v2(graph)
    contra_factors = [f for f in adapted.factor_graph.factors if f.factor_type == FactorType.CONTRADICTION]
    equiv_factors = [f for f in adapted.factor_graph.factors if f.factor_type == FactorType.EQUIVALENCE]

    assert not contra_factors
    assert equiv_factors
    assert adapted.synthetic_var_ids
    for factor in contra_factors + equiv_factors:
        assert factor.conclusion is not None
        assert factor.conclusion in adapted.factor_graph.variables
        assert len(factor.variables) == 2


def test_equivalence_relation_variable_is_near_true_prior():
    graph = HyperGraph()
    a = graph.add_node("A", belief=0.4, prior=0.4)
    b = graph.add_node("B", belief=0.4, prior=0.4)
    graph.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, ["ab"], confidence=0.7)
    graph.add_hyperedge([b.id], a.id, Module.PLAUSIBLE, ["ba"], confidence=0.7)
    adapted = adapt_zero_graph_v2(graph)
    rel_vars = [
        vid for vid in adapted.synthetic_var_ids
        if vid.startswith("equiv_rel_")
    ]
    assert rel_vars
    for vid in rel_vars:
        assert adapted.factor_graph.variables[vid] == pytest.approx(1.0 - CROMWELL_EPS, abs=1e-9)


def test_experiment_edges_use_noisy_and_p2():
    graph = HyperGraph()
    a = graph.add_node("A", belief=0.7, prior=0.7)
    b = graph.add_node("B", belief=0.4, prior=0.4)
    graph.add_hyperedge([a.id], b.id, Module.EXPERIMENT, ["exp"], confidence=0.8, edge_type="heuristic")
    adapted = adapt_zero_graph_v2(graph)
    se = [f for f in adapted.factor_graph.factors if f.factor_type == FactorType.SOFT_ENTAILMENT]
    assert len(se) == 1
    assert se[0].p2 == pytest.approx(1.0 - CROMWELL_EPS, abs=1e-9)


def test_plausible_same_conclusion_edges_are_deduplicated():
    graph = HyperGraph()
    a = graph.add_node("A", belief=0.8, prior=0.8)
    b = graph.add_node("B", belief=0.75, prior=0.75)
    c = graph.add_node("C", belief=0.5, prior=0.5)
    graph.add_hyperedge([a.id], c.id, Module.PLAUSIBLE, ["a->c"], confidence=0.9, edge_type="heuristic")
    graph.add_hyperedge([b.id], c.id, Module.PLAUSIBLE, ["b->c"], confidence=0.7, edge_type="heuristic")
    adapted = adapt_zero_graph_v2(graph)
    se = [f for f in adapted.factor_graph.factors if f.factor_type == FactorType.SOFT_ENTAILMENT]
    assert len(se) == 1
