"""Tests for the Zero <-> Gaia adapter layer."""

import pytest

from dz_hypergraph.models import HyperGraph, Module
from dz_hypergraph.adapter import (
    adapt_zero_graph,
    run_gaia_bp,
    write_back_beliefs,
    InferenceResult,
    ZeroInferenceGraph,
)
from libs.inference.factor_graph import CROMWELL_EPS


class TestAdaptZeroGraph:
    def test_empty_graph(self):
        g = HyperGraph()
        zig = adapt_zero_graph(g)
        assert len(zig.factor_graph.variables) == 0
        assert len(zig.factor_graph.factors) == 0

    def test_single_node(self):
        g = HyperGraph()
        g.add_node("A is true", belief=0.6, prior=0.6)
        zig = adapt_zero_graph(g)
        assert len(zig.factor_graph.variables) == 1
        assert len(zig.factor_graph.factors) == 0

    def test_linear_chain(self):
        g = HyperGraph()
        a = g.add_node("A", belief=0.9, prior=0.9)
        b = g.add_node("B", belief=0.5, prior=0.5)
        c = g.add_node("C", belief=0.5, prior=0.5)
        g.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, ["step1"], confidence=0.8)
        g.add_hyperedge([b.id], c.id, Module.PLAUSIBLE, ["step2"], confidence=0.7)
        zig = adapt_zero_graph(g)
        assert len(zig.factor_graph.variables) == 3
        assert len(zig.factor_graph.factors) == 2

    def test_diamond_dag(self):
        g = HyperGraph()
        a = g.add_node("A", belief=0.9, prior=0.9)
        b = g.add_node("B", belief=0.5, prior=0.5)
        c = g.add_node("C", belief=0.5, prior=0.5)
        d = g.add_node("D", belief=0.5, prior=0.5)
        g.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, ["s1"], confidence=0.8)
        g.add_hyperedge([a.id], c.id, Module.PLAUSIBLE, ["s2"], confidence=0.7)
        g.add_hyperedge([b.id, c.id], d.id, Module.PLAUSIBLE, ["s3"], confidence=0.6)
        zig = adapt_zero_graph(g)
        assert len(zig.factor_graph.variables) == 4
        assert len(zig.factor_graph.factors) == 3

    def test_decomposition_edges_excluded(self):
        g = HyperGraph()
        a = g.add_node("A", belief=0.5, prior=0.5)
        b = g.add_node("B", belief=0.5, prior=0.5)
        g.add_hyperedge([a.id], b.id, Module.LEAN, ["decomp"], confidence=0.95, edge_type="decomposition")
        zig = adapt_zero_graph(g)
        assert len(zig.factor_graph.factors) == 0
        assert len(zig.excluded_edge_ids) == 1

    def test_proven_node_clamped(self):
        g = HyperGraph()
        a = g.add_node("A", belief=1.0, prior=1.0, state="proven")
        zig = adapt_zero_graph(g)
        var_id = zig.node_id_to_var_id[a.id]
        prior_value = zig.factor_graph.variables[var_id]
        assert prior_value == pytest.approx(1.0 - CROMWELL_EPS, abs=1e-6)

    def test_refuted_node_clamped(self):
        g = HyperGraph()
        a = g.add_node("A", belief=0.0, prior=0.0, state="refuted")
        zig = adapt_zero_graph(g)
        var_id = zig.node_id_to_var_id[a.id]
        prior_value = zig.factor_graph.variables[var_id]
        assert prior_value == pytest.approx(CROMWELL_EPS, abs=1e-6)

    def test_mixed_edge_types(self):
        g = HyperGraph()
        a = g.add_node("A", belief=0.9, prior=0.9, state="proven")
        b = g.add_node("B", belief=0.5, prior=0.5)
        c = g.add_node("C", belief=0.5, prior=0.5)
        g.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, ["heuristic"], confidence=0.7)
        g.add_hyperedge([a.id], c.id, Module.LEAN, ["formal"], confidence=0.99)
        zig = adapt_zero_graph(g)
        assert len(zig.factor_graph.factors) == 2

    def test_id_mappings_bijective(self):
        g = HyperGraph()
        for i in range(5):
            g.add_node(f"Node {i}", belief=0.5, prior=0.5)
        zig = adapt_zero_graph(g)
        assert len(zig.node_id_to_var_id) == 5
        assert len(zig.var_id_to_node_id) == 5
        for nid, var_id in zig.node_id_to_var_id.items():
            assert zig.var_id_to_node_id[var_id] == nid


class TestWriteBackBeliefs:
    def test_unlocked_nodes_updated(self):
        g = HyperGraph()
        a = g.add_node("A", belief=0.5, prior=0.5)
        result = InferenceResult(node_beliefs={a.id: 0.8})
        write_back_beliefs(g, result)
        assert g.nodes[a.id].belief == pytest.approx(0.8)

    def test_locked_nodes_not_updated(self):
        g = HyperGraph()
        a = g.add_node("A", belief=1.0, prior=1.0, state="proven")
        result = InferenceResult(node_beliefs={a.id: 0.5})
        write_back_beliefs(g, result)
        assert g.nodes[a.id].belief == 1.0

    def test_beliefs_clamped(self):
        from gaia.bp.factor_graph import CROMWELL_EPS
        g = HyperGraph()
        a = g.add_node("A", belief=0.5, prior=0.5)
        result = InferenceResult(node_beliefs={a.id: 1.5})
        write_back_beliefs(g, result)
        assert g.nodes[a.id].belief == pytest.approx(1.0 - CROMWELL_EPS, abs=1e-9)


class TestRunGaiaBp:
    def test_simple_chain_propagation(self):
        g = HyperGraph()
        a = g.add_node("Axiom", belief=0.95, prior=0.95)
        b = g.add_node("Conjecture", belief=0.5, prior=0.5)
        g.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, ["reasoning"], confidence=0.8)
        result = run_gaia_bp(g)
        assert result.node_beliefs[b.id] > 0.5

    def test_multiple_support_edges(self):
        g = HyperGraph()
        a = g.add_node("A", belief=0.8, prior=0.8)
        b = g.add_node("B", belief=0.7, prior=0.7)
        c = g.add_node("C", belief=0.3, prior=0.3)
        g.add_hyperedge([a.id], c.id, Module.PLAUSIBLE, ["s1"], confidence=0.6)
        g.add_hyperedge([b.id], c.id, Module.EXPERIMENT, ["s2"], confidence=0.7)
        result = run_gaia_bp(g)
        assert result.node_beliefs[c.id] > 0.3

    def test_empty_graph_returns_empty(self):
        g = HyperGraph()
        result = run_gaia_bp(g)
        assert result.node_beliefs == {}
        assert result.converged is True
