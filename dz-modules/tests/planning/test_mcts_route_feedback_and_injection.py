from __future__ import annotations

from types import SimpleNamespace

from dz_hypergraph.memo import Claim, ClaimType, VerificationStatus
from dz_hypergraph.models import HyperGraph, Module
from dz_hypergraph.persistence import save_graph
from dz_engine.bridge import BridgePlan, BridgeProposition, BridgeReasoningStep
from dz_engine.mcts_engine import MCTSConfig, MCTSDiscoveryEngine
from dz_engine.orchestrator import ActionResult


def test_select_action_breaks_experiment_loop(tmp_graph_dir, monkeypatch):
    graph_path = tmp_graph_dir / "graph.json"
    graph = HyperGraph()
    target = graph.add_node("Target theorem", belief=0.2, prior=0.2)
    premise = graph.add_node("Some premise", belief=0.5, prior=0.5)
    graph.add_hyperedge(
        premise_ids=[premise.id],
        conclusion_id=target.id,
        module=Module.PLAUSIBLE,
        steps=["candidate route"],
        confidence=0.6,
    )
    save_graph(graph, graph_path)

    engine = MCTSDiscoveryEngine(
        graph_path=graph_path,
        target_node_id=target.id,
        config=MCTSConfig(max_iterations=1, enable_retrieval=False),
    )
    # Simulate 3 consecutive EXPERIMENT selections on the same node.
    for _ in range(3):
        engine.search_state.record_selection(premise.id, Module.EXPERIMENT)

    monkeypatch.setattr(
        "dz_engine.mcts_engine.select_module_ucb",
        lambda _g, _nid, _state: Module.EXPERIMENT,
    )

    node_id, module, _path = engine._select_action(graph)
    # After 3 consecutive experiments, the loop-breaker should force PLAUSIBLE.
    assert module == Module.PLAUSIBLE


def test_run_verification_pipeline_skips_parent_edge_when_parent_concludes_target(
    tmp_graph_dir,
    monkeypatch,
):
    graph_path = tmp_graph_dir / "graph.json"
    graph = HyperGraph()
    target = graph.add_node("Target theorem", belief=0.1, prior=0.1)
    premise = graph.add_node("Known premise", state="proven", belief=1.0, prior=1.0)
    parent_edge = graph.add_hyperedge(
        premise_ids=[premise.id],
        conclusion_id=target.id,
        module=Module.PLAUSIBLE,
        steps=["plausible proof route"],
        confidence=0.7,
    )
    save_graph(graph, graph_path)

    class FakePipeline:
        def extract_claims(self, **_kwargs):
            return [
                Claim(
                    claim_text="Quantitative check for overlap bound",
                    claim_type=ClaimType.QUANTITATIVE,
                    verification_status=VerificationStatus.PENDING,
                    source_memo_id="memo_test",
                    confidence=0.7,
                )
            ]

        def prioritize_claims(self, *, claims, **_kwargs):
            return claims

    class FakeClaimVerifier:
        def verify_claims(self, **_kwargs):
            return [SimpleNamespace(verdict="verified", evidence="ok", code="")]

        def update_graph_beliefs(self, **_kwargs):
            return None

    engine = MCTSDiscoveryEngine(
        graph_path=graph_path,
        target_node_id=target.id,
        config=MCTSConfig(max_iterations=1, enable_retrieval=False),
        claim_pipeline=FakePipeline(),
        claim_verifier=FakeClaimVerifier(),
    )

    captured_parent_edge_ids: list[str | None] = []

    def fake_ingest_verified_claim(_graph, **kwargs):
        captured_parent_edge_ids.append(kwargs.get("parent_edge_id"))
        return target.id

    monkeypatch.setattr(
        "dz_engine.mcts_engine.ingest_verified_claim",
        fake_ingest_verified_claim,
    )

    _steps, _lean_feedback, _verification_bonus = engine._run_verification_pipeline(
        action_result=ActionResult(
            action="plausible",
            target_node_id=target.id,
            selected_module=Module.PLAUSIBLE.value,
            raw_output="prose",
            normalized_output={"steps": ["..."]},
            ingest_edge_id=parent_edge.id,
            success=True,
            message="ok",
        ),
        combined_feedback="",
        bridge_plan=None,
        bridge_node_map=None,
    )

    assert captured_parent_edge_ids
    assert captured_parent_edge_ids[0] is None


def test_run_enriches_rolling_feedback_with_strategy_concerns_and_refutations(
    tmp_graph_dir,
    monkeypatch,
):
    graph_path = tmp_graph_dir / "graph.json"
    graph = HyperGraph()
    target = graph.add_node("Target theorem", belief=0.2, prior=0.2)
    helper = graph.add_node("Helper premise", belief=0.6, prior=0.6)
    graph.add_hyperedge(
        premise_ids=[helper.id],
        conclusion_id=target.id,
        module=Module.PLAUSIBLE,
        steps=["existing route"],
        confidence=0.5,
    )
    save_graph(graph, graph_path)

    engine = MCTSDiscoveryEngine(
        graph_path=graph_path,
        target_node_id=target.id,
        config=MCTSConfig(max_iterations=2, enable_retrieval=False, enable_problem_variants=False),
    )

    feedback_inputs: list[str] = []

    def fake_select_action(_graph):
        return target.id, Module.PLAUSIBLE, []

    def fake_execute_selected_action(*, graph, node_id, module, boundary_policy, feedback):
        feedback_inputs.append(feedback)
        save_graph(graph, graph_path)
        return ActionResult(
            action="plausible",
            target_node_id=node_id,
            selected_module=module.value,
            raw_output="plausible output",
            normalized_output={"steps": ["candidate route"]},
            judge_output={"concerns": ["All key lemmas are still unproven."]},
            success=True,
            message="ok",
        )

    def fake_handle_plausible_followups(*, result, **_kwargs):
        result.best_bridge_plan = BridgePlan(
            target_statement="Target theorem",
            propositions=[
                BridgeProposition(id="P1", statement="Seed", role="seed", grade="A"),
                BridgeProposition(id="TARGET", statement="Target theorem", role="target", grade="B", depends_on=["P1"]),
            ],
            chain=[
                BridgeReasoningStep(
                    id="S1",
                    statement="Conclude target from seed",
                    uses=["P1"],
                    concludes=["TARGET"],
                    grade="B",
                )
            ],
            summary="Third-order Bonferroni with overlap correction.",
        )

    def fake_run_verification_pipeline(**_kwargs):
        return (
            [
                {
                    "phase": "claim_verification",
                    "summary": "CRT independence claim => refuted; backup bound => verified",
                    "lean_gaps_identified": 1,
                }
            ],
            "Excess overlap lemma fails when k >= 6.",
            0.0,
        )

    monkeypatch.setattr(engine, "_select_action", fake_select_action)
    monkeypatch.setattr(engine, "_execute_selected_action", fake_execute_selected_action)
    monkeypatch.setattr(engine, "_handle_plausible_followups", fake_handle_plausible_followups)
    monkeypatch.setattr(engine, "_run_verification_pipeline", fake_run_verification_pipeline)
    monkeypatch.setattr("dz_engine.mcts_engine.ingest_action_output", lambda *_args, **_kwargs: _args[1])

    engine.run(planning_feedback="")

    assert len(feedback_inputs) >= 2
    second_feedback = feedback_inputs[1]
    assert "[PROOF STRATEGY]" in second_feedback
    assert "[JUDGE CONCERN]" in second_feedback
    assert "[CLAIM REFUTED]" in second_feedback
    assert "[LEAN DISCOVERY]" in second_feedback
