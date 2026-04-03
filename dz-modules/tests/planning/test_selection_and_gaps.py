from __future__ import annotations

from pathlib import Path

from dz_hypergraph.config import CONFIG
from dz_hypergraph.models import HyperGraph, Module
from dz_hypergraph.persistence import save_graph
from dz_engine.discovery_engine import BeliefGapAnalyser
from dz_engine.mcts_engine import MCTSConfig, MCTSDiscoveryEngine
from dz_engine.orchestrator import ActionResult
from dz_engine.search import SearchState, _infer_claim_type as search_infer_claim_type


def test_claim_type_structural_priority_matches_mcts():
    statement = "Lonely Runner Conjecture for n=11"
    assert search_infer_claim_type(statement) == "structural"
    assert MCTSDiscoveryEngine._infer_claim_type(statement) == "structural"


def test_find_critical_gaps_prefers_ready_nodes_and_decay():
    graph = HyperGraph()
    target = graph.add_node("Target", belief=0.2, prior=0.2)
    ready = graph.add_node("Ready lemma", belief=0.2, prior=0.2)
    cold = graph.add_node("Cold lemma", belief=0.2, prior=0.2)
    proved = graph.add_node("Proved premise", belief=1.0, prior=1.0, state="proven")
    weak = graph.add_node("Weak premise", belief=0.1, prior=0.1)

    graph.add_hyperedge([ready.id], target.id, Module.PLAUSIBLE, ["r->t"], confidence=0.8)
    graph.add_hyperedge([cold.id], target.id, Module.PLAUSIBLE, ["c->t"], confidence=0.8)
    graph.add_hyperedge([proved.id], ready.id, Module.PLAUSIBLE, ["p->r"], confidence=0.8)
    graph.add_hyperedge([weak.id], cold.id, Module.PLAUSIBLE, ["w->c"], confidence=0.8)

    analyser = BeliefGapAnalyser()
    ranked = analyser.find_critical_gaps(graph, target.id, top_k=5)
    ids = [nid for nid, _ in ranked]
    assert ready.id in ids and cold.id in ids
    assert ids.index(ready.id) < ids.index(cold.id)

    state = SearchState()
    state.visit_counts[ready.id] = 15
    ranked_with_decay = analyser.find_critical_gaps(graph, target.id, top_k=5, search_state=state)
    ids_decay = [nid for nid, _ in ranked_with_decay]
    assert cold.id in ids_decay
    assert ids_decay.index(cold.id) <= ids_decay.index(ready.id)


def test_select_action_uses_stall_escalation(tmp_path, monkeypatch):
    graph_path = Path(tmp_path) / "graph.json"
    graph = HyperGraph()
    target = graph.add_node("Target theorem", belief=0.2, prior=0.2)
    helper = graph.add_node("Helper", belief=0.4, prior=0.4)
    graph.add_hyperedge([helper.id], target.id, Module.PLAUSIBLE, ["route"], confidence=0.7)
    save_graph(graph, graph_path)

    engine = MCTSDiscoveryEngine(
        graph_path=graph_path,
        target_node_id=target.id,
        config=MCTSConfig(max_iterations=1, enable_retrieval=False),
    )
    engine._belief_stall_count = 4
    engine._plausible_stall_cycles = 3
    monkeypatch.setattr(engine, "_default_select", lambda _g: (helper.id, Module.EXPERIMENT, []))
    monkeypatch.setattr(engine, "_count_live_plausible_routes_to_target", lambda _g: 0)

    node_id, module, _path = engine._select_action(graph)
    assert node_id == target.id
    assert module == Module.DECOMPOSE


def test_select_action_suppresses_globally_failing_module(tmp_path, monkeypatch):
    graph_path = Path(tmp_path) / "graph.json"
    graph = HyperGraph()
    target = graph.add_node("Target theorem", belief=0.2, prior=0.2)
    helper = graph.add_node("Helper", belief=0.4, prior=0.4)
    graph.add_hyperedge([helper.id], target.id, Module.PLAUSIBLE, ["route"], confidence=0.7)
    save_graph(graph, graph_path)

    engine = MCTSDiscoveryEngine(
        graph_path=graph_path,
        target_node_id=target.id,
        config=MCTSConfig(max_iterations=1, enable_retrieval=False),
    )
    engine._recent_module_history = [(Module.LEAN, False)] * 8
    monkeypatch.setattr(engine, "_default_select", lambda _g: (helper.id, Module.LEAN, []))

    node_id, module, _path = engine._select_action(graph)
    assert node_id == target.id
    assert module == Module.PLAUSIBLE


def test_stall_counter_not_reset_by_tiny_belief_delta(tmp_path, monkeypatch):
    graph_path = Path(tmp_path) / "graph.json"
    graph = HyperGraph()
    target = graph.add_node("Target theorem", belief=0.2, prior=0.2)
    helper = graph.add_node("Helper", belief=0.4, prior=0.4)
    graph.add_hyperedge([helper.id], target.id, Module.PLAUSIBLE, ["route"], confidence=0.7)
    save_graph(graph, graph_path)

    engine = MCTSDiscoveryEngine(
        graph_path=graph_path,
        target_node_id=target.id,
        config=MCTSConfig(max_iterations=1, enable_retrieval=False),
    )
    engine._belief_stall_count = 2
    engine._plausible_stall_cycles = 3

    monkeypatch.setattr(engine, "_select_action", lambda _g: (helper.id, Module.EXPERIMENT, []))

    def _fake_execute_selected_action(*, graph, node_id, module, boundary_policy, feedback):
        graph.nodes[target.id].belief = float(graph.nodes[target.id].belief) + 0.001
        save_graph(graph, graph_path)
        return ActionResult(
            action="experiment",
            target_node_id=node_id,
            selected_module=module.value,
            raw_output="",
            normalized_output=None,
            judge_output=None,
            success=True,
            message="ok",
        )

    monkeypatch.setattr(engine, "_execute_selected_action", _fake_execute_selected_action)
    engine.run()
    assert engine._belief_stall_count == 2
    assert engine._plausible_stall_cycles == 3


def test_signal_accumulator_uses_config_threshold(tmp_path, monkeypatch):
    graph_path = Path(tmp_path) / "graph.json"
    graph = HyperGraph()
    target = graph.add_node("Target theorem", belief=0.2, prior=0.2)
    save_graph(graph, graph_path)
    monkeypatch.setattr(CONFIG, "bp_propagation_threshold", 1)
    engine = MCTSDiscoveryEngine(
        graph_path=graph_path,
        target_node_id=target.id,
        config=MCTSConfig(max_iterations=1, enable_retrieval=False),
        signal_accumulator=None,
    )
    assert engine.signal_accumulator.threshold == 1
