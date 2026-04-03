"""Tests for Gaia BP-backed belief propagation and refutation penalties."""

import pytest
from dz_hypergraph.config import CONFIG
from dz_hypergraph.adapter import InferenceResult
from dz_hypergraph.models import HyperGraph, Module
from dz_hypergraph.inference import (
    _bp_marginal_should_warn_outside_cromwell,
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
        """Multi-premise with strong premises, strong confidence, neutral prior.

        With SOFT_ENTAILMENT ↝(0.85, 0.5) and CONJUNCTION mediator:
          When mediator is true (premises jointly hold): p₁=0.85 for conclusion.
          When mediator is false: p₂=0.5 (MaxEnt, no information).
        For strong premises (0.95, 0.8), mediator ≈ 0.76.

        With neutral prior (0.5), the positive evidence should raise belief.
        """
        g = HyperGraph()
        n1 = g.add_node("axiom 1", belief=0.95, prior=0.95)
        n2 = g.add_node("axiom 2", belief=0.8, prior=0.8)
        n3 = g.add_node("conclusion", belief=0.5, prior=0.5)
        g.add_hyperedge([n1.id, n2.id], n3.id, Module.PLAUSIBLE, [], 0.85)
        propagate_beliefs(g)
        # Strong evidence (p=0.85) with strong premises raises belief above neutral prior
        assert g.nodes[n3.id].belief > 0.5

    def test_multi_premise_edge_low_prior(self):
        """Multi-premise with strong premises but skeptical prior (0.1).

        With SOFT_ENTAILMENT ↝(0.6, 0.5): moderate evidence combined with
        low prior. The posterior is a Bayesian update — it may increase or
        stay near the prior depending on the evidence strength.
        """
        g = HyperGraph()
        n1 = g.add_node("axiom 1", belief=0.95, prior=0.95)
        n2 = g.add_node("axiom 2", belief=0.8, prior=0.8)
        n3 = g.add_node("conclusion", belief=0.1, prior=0.1)
        g.add_hyperedge([n1.id, n2.id], n3.id, Module.PLAUSIBLE, [], 0.6)
        propagate_beliefs(g)
        # Posterior is near the Bayes-optimal value, belief changed from prior
        assert g.nodes[n3.id].belief != 0.1  # belief actually updates
        assert 0.0 < g.nodes[n3.id].belief < 0.5  # moderate evidence, low prior

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


class TestCromwellDiagnosticTolerance:
    def test_no_warn_for_marginal_slightly_above_cromwell_ceiling(self):
        # Strict band upper is 0.999; BP often returns ~0.9993 — not a real violation.
        assert not _bp_marginal_should_warn_outside_cromwell(0.9993)

    def test_warn_for_exact_one(self):
        assert _bp_marginal_should_warn_outside_cromwell(1.0)

    def test_warn_for_exact_zero(self):
        assert _bp_marginal_should_warn_outside_cromwell(0.0)

    def test_no_warn_interior(self):
        assert not _bp_marginal_should_warn_outside_cromwell(0.5)

    def test_warn_clearly_above_soft_band(self):
        assert _bp_marginal_should_warn_outside_cromwell(0.9999)


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


def test_incremental_bp_writes_back_result_node_beliefs(monkeypatch):
    monkeypatch.setattr(CONFIG, "bp_backend", "gaia")
    g = HyperGraph()
    a = g.add_node("A", belief=0.9, prior=0.9)
    b = g.add_node("B", belief=0.2, prior=0.2)
    edge = g.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, ["support"], confidence=0.7)

    def _fake_run_gaia_bp(*args, **kwargs):
        return InferenceResult(
            node_beliefs={a.id: 0.9, b.id: 0.88},
            converged=True,
            iterations=3,
            diagnostics=None,
        )

    monkeypatch.setattr("dz_hypergraph.inference.run_gaia_bp", _fake_run_gaia_bp)
    iters = propagate_beliefs(g, changed_edge_ids={edge.id}, warmstart=True)
    assert iters == 3
    assert g.nodes[b.id].belief == pytest.approx(0.88, abs=1e-6)


def test_gaia_v2_path_passes_warmstart_to_run_inference_v2(monkeypatch):
    monkeypatch.setattr(CONFIG, "bp_backend", "gaia_v2")
    g = HyperGraph()
    a = g.add_node("A", belief=0.9, prior=0.9)
    b = g.add_node("B", belief=0.2, prior=0.2)
    g.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, ["support"], confidence=0.8)
    seen: dict[str, bool] = {}

    def _fake_run_inference_v2(graph, *, warmstart=False, config=None):
        seen["warmstart"] = warmstart
        return InferenceResult(
            node_beliefs={a.id: 0.9, b.id: 0.6},
            converged=True,
            iterations=2,
            diagnostics=None,
        )

    monkeypatch.setattr("dz_hypergraph.inference.run_inference_v2", _fake_run_inference_v2)
    propagate_beliefs(g, warmstart=True)
    assert seen["warmstart"] is True
