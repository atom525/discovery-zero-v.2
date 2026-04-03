from dz_hypergraph.models import HyperGraph, Module
from dz_hypergraph.persistence import save_graph
from dz_engine.mcts_engine import ActionResult, MCTSConfig, MCTSDiscoveryEngine


def test_compute_verification_reward_prefers_explicit_verified(tmp_graph_dir):
    graph_path = tmp_graph_dir / "graph.json"
    graph = HyperGraph()
    target = graph.add_node("Target theorem", belief=0.2, prior=0.2)
    save_graph(graph, graph_path)
    engine = MCTSDiscoveryEngine(
        graph_path=graph_path,
        target_node_id=target.id,
        config=MCTSConfig(max_iterations=1, enable_retrieval=False),
    )
    action = ActionResult(
        action="experiment",
        target_node_id=target.id,
        selected_module=Module.EXPERIMENT.value,
        normalized_output={"outcome": "verified"},
        judge_output={"confidence": 0.7},
        success=True,
    )
    reward = engine._compute_verification_reward(
        action_result=action,
        target_belief_before=0.2,
        target_belief_after=0.25,
    )
    # Verified outcome adds 0.4 (from the outcome bonus).
    # The experiment module bonus (0.05 or 0.25) is now conditional on whether
    # the target node is on the bridge reasoning chain.  In this test the graph
    # has no edges, so the target is not on any chain → module bonus = 0.05.
    # Total: 0.4 (verified) + 0.05 (experiment, non-bridge) = 0.45, plus a
    # small belief-delta contribution.  The reward should be clearly positive.
    assert reward >= 0.4


def test_select_action_uses_claim_type_routing(tmp_graph_dir):
    graph_path = tmp_graph_dir / "graph.json"
    graph = HyperGraph()
    target = graph.add_node("if A then B", belief=0.2, prior=0.2)
    candidate = graph.add_node("n >= 2 implies n^2 - n is even", belief=0.1, prior=0.1)
    graph.add_hyperedge([candidate.id], target.id, Module.PLAUSIBLE, ["step"], 0.6)
    save_graph(graph, graph_path)
    engine = MCTSDiscoveryEngine(
        graph_path=graph_path,
        target_node_id=target.id,
        config=MCTSConfig(max_iterations=1, enable_retrieval=False),
    )
    node_id, module, _path = engine._select_action(graph)
    assert node_id in {candidate.id, target.id}
    assert module in {Module.EXPERIMENT, Module.LEAN, Module.PLAUSIBLE}

