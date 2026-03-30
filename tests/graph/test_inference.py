"""Tests for Gaia BP-backed belief propagation and refutation penalties."""

import pytest
from discovery_zero.config import CONFIG
from discovery_zero.graph.models import HyperGraph, Module
from discovery_zero.graph.inference import (
    apply_refutation_penalties,
    propagate_beliefs,
    run_inference_v2,
)


class TestGaiaBPPropagation:
    def test_axiom_stays_at_1(self):
        g = HyperGraph()
        n = g.add_node("axiom", belief=1.0, prior=1.0)
        propagate_beliefs(g)
        assert g.nodes[n.id].belief == pytest.approx(1.0, abs=0.01)

    def test_single_edge_propagation(self):
        g = HyperGraph()
        n1 = g.add_node("axiom", belief=0.95, prior=0.95)
        n2 = g.add_node("conclusion", belief=0.5, prior=0.5)
        g.add_hyperedge([n1.id], n2.id, Module.PLAUSIBLE, [], 0.7)
        propagate_beliefs(g)
        assert g.nodes[n2.id].belief > 0.5

    def test_multi_premise_edge(self):
        g = HyperGraph()
        n1 = g.add_node("axiom 1", belief=0.95, prior=0.95)
        n2 = g.add_node("axiom 2", belief=0.8, prior=0.8)
        n3 = g.add_node("conclusion", belief=0.1, prior=0.1)
        g.add_hyperedge([n1.id, n2.id], n3.id, Module.PLAUSIBLE, [], 0.6)
        propagate_beliefs(g)
        assert g.nodes[n3.id].belief > 0.1

    def test_multi_path_enhancement(self):
        g = HyperGraph()
        n1 = g.add_node("axiom", belief=0.95, prior=0.95)
        n2 = g.add_node("conclusion", belief=0.5, prior=0.5)
        g.add_hyperedge([n1.id], n2.id, Module.PLAUSIBLE, [], 0.5)
        g.add_hyperedge([n1.id], n2.id, Module.EXPERIMENT, [], 0.9)
        propagate_beliefs(g)
        assert g.nodes[n2.id].belief > 0.7

    def test_lean_proof_collapses_belief(self):
        g = HyperGraph()
        n1 = g.add_node("axiom", belief=0.95, prior=0.95)
        n2 = g.add_node("conjecture", belief=0.3, prior=0.3)
        g.add_hyperedge([n1.id], n2.id, Module.LEAN, [], 0.99)
        propagate_beliefs(g)
        assert g.nodes[n2.id].belief > 0.85

    def test_chain_propagation(self):
        g = HyperGraph()
        a = g.add_node("axiom", belief=0.95, prior=0.95)
        b = g.add_node("lemma", belief=0.5, prior=0.5)
        c = g.add_node("theorem", belief=0.5, prior=0.5)
        g.add_hyperedge([a.id], b.id, Module.LEAN, [], 0.99)
        g.add_hyperedge([b.id], c.id, Module.PLAUSIBLE, [], 0.6)
        propagate_beliefs(g)
        assert g.nodes[b.id].belief > 0.8
        assert g.nodes[c.id].belief > 0.3

    def test_no_edges_belief_unchanged(self):
        g = HyperGraph()
        n = g.add_node("isolated conjecture", belief=0.3, prior=0.3)
        propagate_beliefs(g)
        assert g.nodes[n.id].belief == pytest.approx(0.3, abs=0.01)

    def test_proven_node_stays_locked(self):
        g = HyperGraph()
        a = g.add_node("axiom", belief=1.0, prior=1.0, state="proven")
        b = g.add_node("conclusion", belief=0.5, prior=0.5)
        g.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, [], 0.3)
        propagate_beliefs(g)
        assert g.nodes[a.id].belief == 1.0

    def test_refuted_node_stays_locked(self):
        g = HyperGraph()
        a = g.add_node("false claim", belief=0.0, prior=0.0, state="refuted")
        propagate_beliefs(g)
        assert g.nodes[a.id].belief == 0.0

    def test_returns_iteration_count(self):
        g = HyperGraph()
        a = g.add_node("axiom", belief=0.95, prior=0.95)
        b = g.add_node("conclusion", belief=0.1, prior=0.1)
        g.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, [], 0.7)
        iterations = propagate_beliefs(g)
        assert isinstance(iterations, int)
        assert iterations >= 0


class TestApplyRefutationPenalties:
    def test_refutation_penalizes_premises(self):
        g = HyperGraph()
        a = g.add_node("premise", belief=0.8, prior=0.8)
        b = g.add_node("refuted conclusion", belief=0.0, prior=0.0, state="refuted")
        g.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, ["step"], confidence=0.6)
        changed = apply_refutation_penalties(g)
        assert changed > 0
        assert g.nodes[a.id].belief < 0.8

    def test_formal_edges_not_penalized(self):
        g = HyperGraph()
        a = g.add_node("premise", belief=0.8, prior=0.8)
        b = g.add_node("refuted conclusion", belief=0.0, prior=0.0, state="refuted")
        g.add_hyperedge([a.id], b.id, Module.LEAN, ["formal proof"], confidence=0.99)
        changed = apply_refutation_penalties(g)
        assert changed == 0
        assert g.nodes[a.id].belief == 0.8

    def test_locked_premises_not_penalized(self):
        g = HyperGraph()
        a = g.add_node("proven premise", belief=1.0, prior=1.0, state="proven")
        b = g.add_node("refuted conclusion", belief=0.0, prior=0.0, state="refuted")
        g.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, ["step"], confidence=0.6)
        changed = apply_refutation_penalties(g)
        assert changed == 0
        assert g.nodes[a.id].belief == 1.0

    def test_no_refuted_nodes_no_change(self):
        g = HyperGraph()
        a = g.add_node("premise", belief=0.8, prior=0.8)
        b = g.add_node("conclusion", belief=0.5, prior=0.5)
        g.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, ["step"], confidence=0.6)
        changed = apply_refutation_penalties(g)
        assert changed == 0


def test_run_inference_v2_filters_synthetic_relation_vars():
    g = HyperGraph()
    a = g.add_node("A", belief=0.4, prior=0.4)
    b = g.add_node("B", belief=0.4, prior=0.4)
    c = g.add_node("C", belief=0.4, prior=0.4)
    g.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, ["ab"], confidence=0.7)
    g.add_hyperedge([b.id], a.id, Module.PLAUSIBLE, ["ba"], confidence=0.7)
    g.add_hyperedge([a.id], c.id, Module.PLAUSIBLE, ["ac"], confidence=0.7)
    g.add_hyperedge([b.id], c.id, Module.PLAUSIBLE, ["bc"], confidence=0.7)
    result = run_inference_v2(g)
    assert result.node_beliefs
    assert set(result.node_beliefs.keys()).issubset(set(g.nodes.keys()))
    assert all(not key.startswith("equiv_") for key in result.node_beliefs)
    assert all(not key.startswith("contra_") for key in result.node_beliefs)


def test_propagate_beliefs_with_gaia_v2_backend(monkeypatch):
    monkeypatch.setattr(CONFIG, "bp_backend", "gaia_v2")
    g = HyperGraph()
    a = g.add_node("Axiom", belief=0.9, prior=0.9)
    b = g.add_node("Claim", belief=0.2, prior=0.2)
    g.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, ["support"], confidence=0.8)
    iterations = propagate_beliefs(g)
    assert isinstance(iterations, int)
    assert g.nodes[b.id].belief > 0.2
