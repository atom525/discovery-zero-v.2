import json

import pytest

from discovery_zero.graph.models import HyperGraph, Module
from discovery_zero.graph.neural_bp import (
    BPTrainingCollector,
    BPTrainingSample,
    NeuralBPCorrector,
    encode_graph_for_nn,
)
from discovery_zero.graph.persistence import save_graph


def _build_graph():
    graph = HyperGraph()
    seed = graph.add_node("Seed fact", belief=1.0, prior=1.0, state="proven")
    target = graph.add_node("Target fact", belief=0.3, prior=0.3)
    graph.add_hyperedge([seed.id], target.id, Module.PLAUSIBLE, ["step"], 0.8)
    return graph, target.id


def test_encode_graph_for_nn_exposes_shapes_or_fallback():
    graph, _ = _build_graph()
    encoding = encode_graph_for_nn(graph)

    if encoding.node_features is None:
        assert encoding.factor_features is None
    else:
        assert tuple(encoding.node_features.shape) == (2, 8)
        assert tuple(encoding.factor_features.shape) == (1, 6)


def test_neural_bp_inactive_returns_standard_beliefs():
    graph, target_id = _build_graph()
    corrector = NeuralBPCorrector(correction_strength=0.0)
    standard = {nid: node.belief for nid, node in graph.nodes.items()}

    assert corrector.correct_beliefs(graph, standard) == standard


def test_neural_bp_train_step_runs_when_torch_available():
    graph, target_id = _build_graph()
    corrector = NeuralBPCorrector(correction_strength=0.2)
    if corrector._net is None:
        pytest.skip("torch unavailable")

    import torch

    optimizer = torch.optim.Adam(corrector._net.parameters(), lr=1e-3)
    loss = corrector.train_step(
        [
            BPTrainingSample(
                graph=graph,
                standard_beliefs={nid: node.belief for nid, node in graph.nodes.items()},
                true_beliefs={
                    target_id: 0.8,
                    next(iter(nid for nid in graph.nodes if nid != target_id)): 1.0,
                },
                run_id="run-1",
                node_id=target_id,
            )
        ],
        optimizer,
    )

    assert loss >= 0.0


def test_bp_training_collector_reads_snapshot_runs(monkeypatch, tmp_path):
    graph, target_id = _build_graph()
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    save_graph(graph, run_dir / "graph.json")
    (run_dir / "exploration_log.json").write_text(
        json.dumps(
            {
                "target_node_id": target_id,
                "graph_snapshot": graph.model_dump_json(),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "discovery_zero.graph.adapter.run_gaia_bp",
        lambda graph: {nid: node.belief for nid, node in graph.nodes.items()},
    )
    monkeypatch.setattr(
        "discovery_zero.graph.adapter.write_back_beliefs",
        lambda graph, beliefs: None,
    )

    samples = BPTrainingCollector().collect_from_run_dirs([run_dir])

    assert len(samples) == 1
    assert samples[0].node_id == target_id
