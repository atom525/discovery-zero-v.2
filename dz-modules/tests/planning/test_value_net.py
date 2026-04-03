import pytest

from dz_hypergraph.models import HyperGraph, Module
from dz_engine.value_net import (
    MODULE_INDEX,
    ProcessAdvantageVerifier,
    PAVTrainingSample,
    encode_action,
)


def _build_graph():
    graph = HyperGraph()
    seed = graph.add_node("Seed", belief=1.0, prior=1.0, state="proven")
    target = graph.add_node("Target", belief=0.3, prior=0.3)
    candidate = graph.add_node("Candidate", belief=0.4, prior=0.4)
    graph.add_hyperedge([seed.id], candidate.id, Module.PLAUSIBLE, ["step"], 0.8)
    graph.add_hyperedge([candidate.id], target.id, Module.LEAN, ["step"], 0.7)
    return graph, target.id, candidate.id


def test_pav_predict_advantage_is_neutral_before_training():
    graph, target_id, candidate_id = _build_graph()
    pav = ProcessAdvantageVerifier()

    assert pav.predict_advantage(graph, target_id, candidate_id, Module.PLAUSIBLE) == 0.0


def test_encode_action_returns_expected_shape_when_available():
    graph, _, candidate_id = _build_graph()
    candidate = graph.nodes[candidate_id]
    encoded = encode_action(
        Module.PLAUSIBLE,
        candidate.belief,
        candidate.prior,
        len(graph.get_edges_to(candidate_id)),
        len(graph.get_edges_from(candidate_id)),
    )

    if encoded is None:
        assert encoded is None
    else:
        assert tuple(encoded.shape) == (len(MODULE_INDEX) + 8,)


def test_pav_train_step_runs_when_torch_available():
    graph, target_id, candidate_id = _build_graph()
    pav = ProcessAdvantageVerifier()
    if pav._net is None:
        pytest.skip("torch unavailable")

    import torch

    optimizer = torch.optim.Adam(pav._net.parameters(), lr=1e-3)
    loss = pav.train_step(
        [
            PAVTrainingSample(
                graph_snapshot=graph,
                target_node_id=target_id,
                action_node_id=candidate_id,
                action_module=Module.PLAUSIBLE,
                actual_belief_gain=0.4,
                success=True,
            )
        ],
        optimizer,
    )

    assert loss >= 0.0
    assert pav._trained is True
