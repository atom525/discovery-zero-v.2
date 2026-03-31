"""Tests for the Gaia theory-aligned BP adapter (adapter_v2).

Per Gaia theory (03-propositional-operators.md §4, 06-factor-graphs.md §3.7):
  - ALL reasoning edges → SOFT_IMPLICATION ↝(p₁, p₂)
  - p₂ = 0.5 (MaxEnt default): premise false → uniform (no info)
  - Formal edges: p₁ ≈ 1 (strict implication)
  - Heuristic edges: p₁ = confidence
  - Multi-premise: CONJUNCTION + SOFT_IMPLICATION
  - Decomposition: skipped (structural only)
"""
from __future__ import annotations

import pytest
from gaia_bp.factor_graph import CROMWELL_EPS, FactorType

from discovery_zero.graph.adapter_v2 import MAXENT_P2, adapt_zero_graph_v2
from discovery_zero.graph.models import HyperGraph, Module


def test_adapter_v2_all_reasoning_edges_use_soft_implication():
    """All reasoning edges (heuristic + formal) → SOFT_IMPLICATION ↝(p₁, 0.5).

    Per 03-propositional-operators.md §4: ↝ is the ONLY parameterized factor.
    Per 06-factor-graphs.md §3.7: p₂ = 0.5 (MaxEnt default).
    Decomposition edges are structural and excluded from BP.
    """
    graph = HyperGraph()
    a = graph.add_node("A", belief=0.8, prior=0.8)
    b = graph.add_node("B", belief=0.2, prior=0.2)
    c = graph.add_node("C", belief=0.2, prior=0.2)
    graph.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, ["h"], confidence=0.7, edge_type="heuristic")
    graph.add_hyperedge([b.id], c.id, Module.LEAN, ["f"], confidence=0.99, edge_type="formal")
    graph.add_hyperedge([a.id], c.id, Module.DECOMPOSE, ["d"], confidence=0.5, edge_type="decomposition")

    adapted = adapt_zero_graph_v2(graph)
    factor_types = [f.factor_type for f in adapted.factor_graph.factors]

    # All reasoning edges → SOFT_IMPLICATION
    assert FactorType.SOFT_IMPLICATION in factor_types
    assert factor_types.count(FactorType.SOFT_IMPLICATION) == 2
    # Decomposition edges are skipped
    assert FactorType.CONJUNCTION not in factor_types
    # INDUCTION and ENTAILMENT are NOT used (per Gaia theory, only ↝)
    assert FactorType.INDUCTION not in factor_types
    assert FactorType.ENTAILMENT not in factor_types
    assert len(adapted.factor_graph.factors) == 2

    # All factors have p₂ = 0.5 (MaxEnt)
    for f in adapted.factor_graph.factors:
        assert f.p2 == pytest.approx(MAXENT_P2)


def test_adapter_v2_formal_edge_p1_near_one():
    """Formal (Lean) edges: strict implication → ↝(1, 0.5).

    Per 03-propositional-operators.md §4.7: strict implication is embedded
    as ↝(1, 0.5), Cromwell-clamped to ↝(1-eps, 0.5).
    """
    graph = HyperGraph()
    a = graph.add_node("premise", belief=0.9, prior=0.9)
    b = graph.add_node("conclusion", belief=0.3, prior=0.3)
    graph.add_hyperedge([a.id], b.id, Module.LEAN, ["proof"], confidence=0.99, edge_type="formal")

    adapted = adapt_zero_graph_v2(graph)
    factor = adapted.factor_graph.factors[0]

    assert factor.factor_type == FactorType.SOFT_IMPLICATION
    assert factor.p >= 1.0 - CROMWELL_EPS  # p₁ ≈ 1
    assert factor.p2 == pytest.approx(MAXENT_P2)  # p₂ = 0.5


def test_adapter_v2_multi_premise_conjunction_plus_soft_implication():
    """Multi-premise → CONJUNCTION mediator + SOFT_IMPLICATION ↝(p₁, 0.5).

    Per theory §5: ∧ (deterministic) separates how premises combine from
    how they support the conclusion.
    """
    graph = HyperGraph()
    a = graph.add_node("A", belief=0.8, prior=0.8)
    b = graph.add_node("B", belief=0.8, prior=0.8)
    c = graph.add_node("C", belief=0.2, prior=0.2)
    graph.add_hyperedge([a.id, b.id], c.id, Module.EXPERIMENT, ["s"], confidence=0.85)

    adapted = adapt_zero_graph_v2(graph)
    factor_types = [f.factor_type for f in adapted.factor_graph.factors]

    assert factor_types.count(FactorType.CONJUNCTION) == 1
    assert factor_types.count(FactorType.SOFT_IMPLICATION) == 1
    assert len(adapted.synthetic_var_ids) == 1  # mediator variable

    si = [f for f in adapted.factor_graph.factors if f.factor_type == FactorType.SOFT_IMPLICATION][0]
    assert si.p == pytest.approx(0.85)
    assert si.p2 == pytest.approx(MAXENT_P2)


def test_adapter_v2_p1_p2_constraint_clamps_low_confidence():
    """p₁ + p₂ > 1 constraint: confidence ≤ 0.5 is clamped to 0.5 + eps.

    This prevents negative-relevance factors from entering BP while
    satisfying the Gaia theory's positive support constraint.
    """
    graph = HyperGraph()
    a = graph.add_node("A", belief=0.8, prior=0.8)
    b = graph.add_node("B", belief=0.2, prior=0.2)
    c = graph.add_node("C", belief=0.2, prior=0.2)
    graph.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, ["s"], confidence=0.5)
    graph.add_hyperedge([a.id], c.id, Module.RETRIEVE, ["s"], confidence=0.4)

    adapted = adapt_zero_graph_v2(graph)

    for f in adapted.factor_graph.factors:
        # All factors must satisfy p₁ + p₂ > 1
        assert f.p + f.p2 > 1.0
        # p₂ is always 0.5 (MaxEnt)
        assert f.p2 == pytest.approx(MAXENT_P2)
        # p₁ is at least 0.5 + CROMWELL_EPS
        assert f.p >= MAXENT_P2 + CROMWELL_EPS - 1e-9


def test_adapter_v2_creates_relation_vars_for_constraints():
    """CONTRADICTION and EQUIVALENCE are deterministic constraint factors."""
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

    assert contra_factors
    assert equiv_factors
    assert adapted.synthetic_var_ids
    for factor in contra_factors + equiv_factors:
        assert factor.relation_var is not None
        assert factor.relation_var in adapted.factor_graph.variables


def test_adapter_v2_no_cold_start_collapse():
    """SOFT_IMPLICATION ↝(p₁, 0.5) must NOT cause cold-start belief collapse.

    Per 07-belief-propagation.md §3: when p₂=0.5 and premise is false,
    the factor sends a uniform message (no information). Unverified
    intermediate nodes (low belief) should NOT suppress downstream.

    This is the critical property that INDUCTION (leak=eps, p₂≈1) violated:
    a chain seed(0.15) → intermediate(0.15) → target(0.1) would collapse
    target to ~0.009. With ↝(p₁, 0.5), target stays near its prior.
    """
    from gaia_bp.engine import EngineConfig, InferenceEngine

    g = HyperGraph()
    seed = g.add_node("seed", belief=0.15, prior=0.15)
    intermediate = g.add_node("intermediate", belief=0.15, prior=0.15)
    target = g.add_node("target", belief=0.1, prior=0.1)
    g.add_hyperedge([seed.id], intermediate.id, Module.PLAUSIBLE, ["s"], confidence=0.6)
    g.add_hyperedge([intermediate.id], target.id, Module.PLAUSIBLE, ["s"], confidence=0.6)

    adapted = adapt_zero_graph_v2(g)
    engine = InferenceEngine(EngineConfig(bp_max_iter=50, bp_damping=0.5))
    result = engine.run(adapted.factor_graph, method="auto")

    target_belief = result.beliefs[target.id]

    # Target must NOT collapse below 0.05 (cold-start collapse threshold)
    assert target_belief > 0.05, (
        f"Cold-start collapse: target belief = {target_belief:.4f}. "
        "SOFT_IMPLICATION with p2=0.5 should prevent this."
    )


def test_adapter_v2_strong_evidence_raises_belief():
    """With strong premise and high confidence, ↝ raises conclusion belief.

    Validates that the adapter produces factors that actually propagate
    positive evidence correctly (not just avoiding cold-start collapse).
    """
    from gaia_bp.engine import EngineConfig, InferenceEngine

    g = HyperGraph()
    axiom = g.add_node("axiom", belief=0.95, prior=0.95)
    conclusion = g.add_node("conclusion", belief=0.3, prior=0.3)
    g.add_hyperedge([axiom.id], conclusion.id, Module.PLAUSIBLE, ["r"], confidence=0.8)

    adapted = adapt_zero_graph_v2(g)
    engine = InferenceEngine(EngineConfig(bp_max_iter=50, bp_damping=0.5))
    result = engine.run(adapted.factor_graph, method="auto")

    # Strong evidence should raise belief above prior
    assert result.beliefs[conclusion.id] > 0.3
