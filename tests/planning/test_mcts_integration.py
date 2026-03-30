from discovery_zero.graph.models import HyperGraph, Module
from discovery_zero.graph.persistence import save_graph
from discovery_zero.planning.mcts_engine import MCTSConfig, MCTSDiscoveryEngine
from discovery_zero.planning.orchestrator import ActionResult


def test_mcts_engine_collects_experience_and_updates_belief(tmp_graph_dir, monkeypatch):
    graph_path = tmp_graph_dir / "graph.json"
    graph = HyperGraph()
    target = graph.add_node("Target theorem", belief=0.3, prior=0.3)
    save_graph(graph, graph_path)

    engine = MCTSDiscoveryEngine(
        graph_path=graph_path,
        target_node_id=target.id,
        config=MCTSConfig(max_iterations=1, enable_retrieval=False, enable_problem_variants=False),
    )

    def fake_select_action(graph):
        return target.id, Module.PLAUSIBLE, []

    def fake_execute_selected_action(*, graph, node_id, module, boundary_policy, feedback):
        updated = HyperGraph.model_validate_json(graph.model_dump_json())
        updated.nodes[target.id].belief = 0.8
        save_graph(updated, graph_path)
        return ActionResult(
            action="plausible",
            target_node_id=node_id,
            selected_module=module.value,
            normalized_output=None,
            success=True,
            message="ok",
        )

    monkeypatch.setattr(engine, "_select_action", fake_select_action)
    monkeypatch.setattr(engine, "_execute_selected_action", fake_execute_selected_action)
    monkeypatch.setattr(engine, "_handle_plausible_followups", lambda **kwargs: None)

    result = engine.run(planning_feedback="")

    assert result.iterations_completed == 1
    assert result.target_belief_final == 0.8
    assert len(result.experiences) == 1
    assert result.experiences[0].belief_delta > 0
