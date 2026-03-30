"""Tests for the bridge plan connectivity fixes.

Covers:
  - materialize_bridge_nodes creates dependency hyperedges
  - bridge TARGET proposition maps to the MCTS target node
  - BRIDGE_GRADE_PRIOR / BRIDGE_EDGE_CONFIDENCE are exported
  - ingest_verified_claim target_node_id bypass
  - ingest_verified_claim parent_edge_id creates edge for new nodes
  - ClaimPipeline.extract_claims bridges bridge_proposition_id via bridge_plan
"""
from __future__ import annotations

import pytest

from discovery_zero.graph.ingest import ingest_verified_claim
from discovery_zero.graph.models import HyperGraph, Module
from discovery_zero.planning.bridge import (
    BRIDGE_EDGE_CONFIDENCE,
    BRIDGE_GRADE_PRIOR,
    BridgePlan,
    BridgeProposition,
    BridgeReasoningStep,
    materialize_bridge_nodes,
    validate_bridge_plan_payload,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_plan(target_statement: str) -> BridgePlan:
    """Two-proposition plan: P1 (seed A) → P2 (derived C) → TARGET (target B)."""
    return BridgePlan(
        target_statement=target_statement,
        propositions=[
            BridgeProposition(id="P1", statement="Seed fact", role="seed", grade="A"),
            BridgeProposition(id="P2", statement="Intermediate lemma", role="derived", grade="C",
                              depends_on=["P1"]),
            BridgeProposition(id="TARGET", statement=target_statement, role="target", grade="B",
                              depends_on=["P2"]),
        ],
        chain=[
            BridgeReasoningStep(id="S1", statement="Derive P2", uses=["P1"], concludes=["P2"], grade="C"),
            BridgeReasoningStep(id="S2", statement="Conclude target", uses=["P2"], concludes=["TARGET"], grade="B"),
        ],
        summary="Simple test plan.",
    )


# ---------------------------------------------------------------------------
# Test bridge prior/confidence constants
# ---------------------------------------------------------------------------

def test_bridge_grade_prior_keys():
    assert set(BRIDGE_GRADE_PRIOR.keys()) == {"A", "B", "C", "D"}
    assert BRIDGE_GRADE_PRIOR["A"] > BRIDGE_GRADE_PRIOR["B"]
    assert BRIDGE_GRADE_PRIOR["B"] > BRIDGE_GRADE_PRIOR["C"]
    assert BRIDGE_GRADE_PRIOR["C"] > BRIDGE_GRADE_PRIOR["D"]
    assert BRIDGE_EDGE_CONFIDENCE == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Test materialize_bridge_nodes: creates nodes AND edges
# ---------------------------------------------------------------------------

def test_materialize_creates_dependency_edges():
    graph = HyperGraph()
    target = graph.add_node("The main theorem", belief=0.1, prior=0.1)
    plan = _make_simple_plan("The main theorem")
    node_map = materialize_bridge_nodes(graph, plan, target_node_id=target.id)

    # All propositions must be in the mapping.
    assert set(node_map.keys()) == {"P1", "P2", "TARGET"}

    # TARGET must map to the real MCTS target, not a duplicate.
    assert node_map["TARGET"] == target.id

    # Two dependency edges must have been created (P1→P2 and P2→TARGET).
    edges_created = list(graph.edges.values())
    # P1 is seed, so no edge is created for P1 itself.
    # Edge for P2: premises=[P1_node], conclusion=P2_node
    # Edge for TARGET: premises=[P2_node], conclusion=target.id
    assert len(edges_created) == 2

    conclusions = {e.conclusion_id for e in edges_created}
    assert node_map["P2"] in conclusions
    assert target.id in conclusions
    assert all(e.confidence == pytest.approx(BRIDGE_EDGE_CONFIDENCE) for e in edges_created)

    # Node priors/beliefs follow proposition grade.
    p2 = graph.nodes[node_map["P2"]]
    assert p2.prior == pytest.approx(BRIDGE_GRADE_PRIOR["C"])
    assert p2.belief == pytest.approx(BRIDGE_GRADE_PRIOR["C"])


def test_materialize_target_maps_to_existing_mcts_node():
    """TARGET proposition is mapped to the MCTS target even when statement differs."""
    graph = HyperGraph()
    # MCTS target uses natural-language statement.
    mcts_target = graph.add_node(
        "LRC holds for n=11 (natural language version)", belief=0.1
    )
    # Bridge plan TARGET uses a formalized statement that would not text-match.
    plan = BridgePlan(
        target_statement="LRC holds for n=11 (natural language version)",
        propositions=[
            BridgeProposition(id="P1", statement="Bad set measure", role="seed", grade="A"),
            BridgeProposition(
                id="TARGET",
                statement="∀ v₁ ... v₁₀ : ℕ, ∃ t, ...",  # different text
                role="target",
                grade="B",
                depends_on=["P1"],
            ),
        ],
        chain=[
            BridgeReasoningStep(id="S1", statement="step", uses=["P1"], concludes=["TARGET"], grade="B"),
        ],
        summary="Formalized target test.",
    )
    node_map = materialize_bridge_nodes(graph, plan, target_node_id=mcts_target.id)
    # TARGET must be remapped to the MCTS node, not a new duplicate.
    assert node_map["TARGET"] == mcts_target.id
    # No new node with the formal statement should exist.
    formal_matches = graph.find_node_ids_by_statement("∀ v₁ ... v₁₀ : ℕ, ∃ t, ...")
    assert not formal_matches


def test_materialize_idempotent_no_duplicate_edges():
    """Calling materialize_bridge_nodes twice must not create duplicate edges."""
    graph = HyperGraph()
    target = graph.add_node("Target", belief=0.1)
    plan = _make_simple_plan("Target")
    materialize_bridge_nodes(graph, plan, target_node_id=target.id)
    edge_count_after_first = len(graph.edges)
    materialize_bridge_nodes(graph, plan, target_node_id=target.id)
    assert len(graph.edges) == edge_count_after_first


def test_materialize_without_target_node_id_still_creates_edges():
    """When target_node_id is not given, edges are still created (TARGET gets new node)."""
    graph = HyperGraph()
    plan = _make_simple_plan("Some target statement")
    node_map = materialize_bridge_nodes(graph, plan)
    # Edges for P1→P2 and P2→TARGET must still exist.
    assert len(graph.edges) == 2


def test_materialize_filters_risk_from_non_risk_conclusions():
    graph = HyperGraph()
    target = graph.add_node("Main target", belief=0.2, prior=0.2)
    plan = BridgePlan(
        target_statement="Main target",
        propositions=[
            BridgeProposition(id="S", statement="Seed", role="seed", grade="A"),
            BridgeProposition(id="R", statement="Unresolved gap", role="risk", grade="D", depends_on=["S"]),
            BridgeProposition(id="B", statement="Bridge lemma", role="bridge", grade="B", depends_on=["S", "R"]),
            BridgeProposition(id="T", statement="Main target", role="target", grade="B", depends_on=["B", "R"]),
        ],
        chain=[
            BridgeReasoningStep(id="S1", statement="Derive bridge", uses=["S"], concludes=["B"], grade="B"),
            BridgeReasoningStep(id="S2", statement="Conclude target", uses=["B"], concludes=["T"], grade="B"),
        ],
        summary="risk filtering test",
    )
    node_map = materialize_bridge_nodes(graph, plan, target_node_id=target.id)

    risk_node_id = node_map["R"]
    bridge_node_id = node_map["B"]
    bridge_edges = [e for e in graph.edges.values() if e.conclusion_id == bridge_node_id]
    assert len(bridge_edges) == 1
    assert risk_node_id not in bridge_edges[0].premise_ids

    target_edges = [e for e in graph.edges.values() if e.conclusion_id == target.id]
    assert len(target_edges) == 1
    assert risk_node_id not in target_edges[0].premise_ids
    assert graph.nodes[risk_node_id].provenance == "bridge_risk"


def test_validate_bridge_plan_autofixes_missing_target_role():
    payload = {
        "target_statement": "Main target",
        "propositions": [
            {"id": "P1", "statement": "Seed", "role": "seed", "grade": "A", "depends_on": []},
            {"id": "P2", "statement": "Main target", "role": "bridge", "grade": "B", "depends_on": ["P1"]},
        ],
        "chain": [
            {"id": "S1", "statement": "Derive bridge", "uses": ["P1"], "concludes": ["P2"], "grade": "B"},
        ],
    }
    plan = validate_bridge_plan_payload(payload)
    target_props = [p for p in plan.propositions if p.role == "target"]
    assert len(target_props) == 1
    assert target_props[0].id == "P2"


# ---------------------------------------------------------------------------
# Test ingest_verified_claim: target_node_id bypass
# ---------------------------------------------------------------------------

def test_ingest_verified_claim_target_node_id_skips_text_search():
    """When target_node_id is provided, the function updates that node directly."""
    graph = HyperGraph()
    existing = graph.add_node("For pairwise coprime speeds, overlap is (2/11)^2", belief=0.5)
    # Claim text deliberately does not match the existing statement.
    different_text = "Pairwise coprime overlap equals 4/121"
    returned_id = ingest_verified_claim(
        graph,
        claim_text=different_text,
        verification_source="experiment",
        verdict="verified",
        target_node_id=existing.id,
    )
    # Must have updated the existing node, not created a new one.
    assert returned_id == existing.id
    assert graph.nodes[existing.id].belief >= 0.85
    # No new node should have been created.
    assert len(graph.nodes) == 1


# ---------------------------------------------------------------------------
# Test ingest_verified_claim: parent_edge_id creates edge for new nodes
# ---------------------------------------------------------------------------

def test_ingest_verified_claim_parent_edge_id_creates_edge():
    """New claim nodes created without a text match should be linked via parent_edge_id."""
    graph = HyperGraph()
    premise = graph.add_node("Known fact", belief=1.0, state="proven")
    conclusion = graph.add_node("Target theorem", belief=0.2)
    # Create a parent edge whose conclusion is the target.
    parent_edge = graph.add_hyperedge(
        premise_ids=[premise.id],
        conclusion_id=conclusion.id,
        module=Module.PLAUSIBLE,
        steps=["plausible reasoning"],
        confidence=0.7,
    )

    returned_id = ingest_verified_claim(
        graph,
        claim_text="Entirely new claim not in graph",
        verification_source="experiment",
        verdict="verified",
        parent_edge_id=parent_edge.id,
    )

    # A new node was created.
    assert returned_id not in {premise.id, conclusion.id}
    assert returned_id in graph.nodes

    # An edge connecting the new node → conclusion must exist.
    new_node_edges = [
        e for e in graph.edges.values()
        if returned_id in e.premise_ids and e.conclusion_id == conclusion.id
    ]
    assert len(new_node_edges) == 1


def test_ingest_verified_claim_existing_node_no_extra_edge():
    """When an existing node is matched, no spurious edge should be created."""
    graph = HyperGraph()
    existing = graph.add_node("Known claim", belief=0.3)
    premise = graph.add_node("P", belief=1.0, state="proven")
    conclusion = graph.add_node("C", belief=0.2)
    parent_edge = graph.add_hyperedge(
        premise_ids=[premise.id],
        conclusion_id=conclusion.id,
        module=Module.PLAUSIBLE,
        steps=["step"],
        confidence=0.6,
    )
    initial_edge_count = len(graph.edges)

    returned_id = ingest_verified_claim(
        graph,
        claim_text="Known claim",  # matches existing node
        verification_source="experiment",
        verdict="verified",
        parent_edge_id=parent_edge.id,
    )

    assert returned_id == existing.id
    # No new edge should have been created.
    assert len(graph.edges) == initial_edge_count


# ---------------------------------------------------------------------------
# Test ClaimPipeline: bridge_plan injects valid bridge_proposition_id
# ---------------------------------------------------------------------------

def test_extract_claims_accepts_valid_bridge_proposition_id(monkeypatch):
    """bridge_proposition_id from LLM output is accepted if it maps to a known proposition."""
    from discovery_zero.planning.claim_pipeline import ClaimPipeline
    from discovery_zero.graph.memo import ClaimType

    pipeline = ClaimPipeline()
    plan = _make_simple_plan("The main theorem")

    def fake_run_skill(*args, **kwargs):
        return "{}", {
            "claims": [
                {
                    "claim_text": "Intermediate lemma is true",
                    "claim_type": "structural",
                    "confidence": 0.7,
                    "evidence": "derived",
                    "bridge_proposition_id": "P2",  # valid proposition ID
                },
                {
                    "claim_text": "Some heuristic claim",
                    "claim_type": "heuristic",
                    "confidence": 0.4,
                    "evidence": "",
                    "bridge_proposition_id": "INVALID_ID",  # invalid → should be None
                },
            ]
        }

    monkeypatch.setattr("discovery_zero.planning.claim_pipeline.run_skill", fake_run_skill)
    claims = pipeline.extract_claims(
        prose="Some reasoning prose",
        context="",
        source_memo_id="memo_test",
        model=None,
        bridge_plan=plan,
    )
    assert len(claims) == 2
    # First claim maps to valid bridge proposition P2.
    assert claims[0].bridge_proposition_id == "P2"
    # Second claim has invalid bridge_proposition_id, should be None.
    assert claims[1].bridge_proposition_id is None


def test_extract_claims_without_bridge_plan_no_bridge_id(monkeypatch):
    """Without a bridge plan, bridge_proposition_id must always be None."""
    from discovery_zero.planning.claim_pipeline import ClaimPipeline

    pipeline = ClaimPipeline()

    def fake_run_skill(*args, **kwargs):
        return "{}", {
            "claims": [
                {
                    "claim_text": "Claim without bridge",
                    "claim_type": "heuristic",
                    "confidence": 0.5,
                    "evidence": "",
                    "bridge_proposition_id": "P2",  # even if LLM output it
                }
            ]
        }

    monkeypatch.setattr("discovery_zero.planning.claim_pipeline.run_skill", fake_run_skill)
    claims = pipeline.extract_claims(
        prose="prose",
        context="",
        source_memo_id="memo_test",
        model=None,
        bridge_plan=None,  # no bridge plan → no valid IDs
    )
    assert len(claims) == 1
    # Without a valid set of proposition IDs, bridge_proposition_id is None.
    assert claims[0].bridge_proposition_id is None
