"""Tests for energy minimization (dual-state RENEW)."""

import pytest
from discovery_zero.graph.models import HyperGraph, Module
from discovery_zero.graph.inference import propagate_beliefs
from discovery_zero.graph.inference_energy import (
    EnergyConfig,
    propagate_beliefs_energy,
    _single_edge_energy,
    _global_energy,
)


class TestEnergyPropagation:
    def test_proven_node_unchanged(self):
        g = HyperGraph()
        a = g.add_node("axiom", belief=1.0, state="proven")
        b = g.add_node("conj", belief=0.5)
        g.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, ["step"], 0.8)
        it = propagate_beliefs_energy(g)
        assert g.nodes[a.id].belief == 1.0
        assert g.nodes[a.id].state == "proven"
        assert it >= 1

    def test_refuted_node_unchanged(self):
        g = HyperGraph()
        a = g.add_node("refuted", belief=0.0, state="refuted")
        it = propagate_beliefs_energy(g)
        assert g.nodes[a.id].belief == 0.0
        assert it == 0

    def test_single_unverified_converges(self):
        g = HyperGraph()
        a = g.add_node("premise", belief=1.0, state="proven")
        b = g.add_node("conclusion", belief=0.0)
        g.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, [], 0.9)
        it = propagate_beliefs_energy(g, EnergyConfig(step_size=0.2, max_iterations=100))
        assert it >= 1
        assert g.nodes[b.id].belief > 0.5

    def test_formal_edge_dominates(self):
        g = HyperGraph()
        a = g.add_node("ax", belief=1.0, state="proven")
        b = g.add_node("conj", belief=0.3)
        g.add_hyperedge([a.id], b.id, Module.LEAN, ["proof"], 0.99)
        it = propagate_beliefs_energy(g, EnergyConfig(step_size=0.1, max_iterations=150))
        assert g.nodes[b.id].belief > 0.9


class TestEdgeEnergy:
    def test_edge_energy_zero_when_satisfied(self):
        g = HyperGraph()
        a = g.add_node("p", belief=1.0)
        b = g.add_node("c", belief=1.0)
        e = g.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, [], 0.5)
        x = {a.id: 1.0, b.id: 1.0}
        E = _single_edge_energy(g, e, x)
        assert E == 0.0

    def test_edge_energy_positive_when_violated(self):
        g = HyperGraph()
        a = g.add_node("p", belief=1.0)
        b = g.add_node("c", belief=0.0)
        e = g.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, [], 0.5)
        x = {a.id: 1.0, b.id: 0.0}
        E = _single_edge_energy(g, e, x)
        assert E > 0
