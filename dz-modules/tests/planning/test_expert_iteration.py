from dz_hypergraph.models import Module
from dz_engine.expert_iteration import (
    ExperienceBuffer,
    ExperienceRecord,
    ExpertIterationLoop,
    build_dpo_pairs,
)
from dz_engine.search import IntrinsicReward


def _make_record(
    *,
    snapshot: str,
    module: str = Module.PLAUSIBLE.value,
    belief_delta: float = 0.0,
    novelty: float = 0.0,
    success: bool = True,
):
    return ExperienceRecord(
        graph_snapshot_json=snapshot,
        target_node_id="target-1",
        action_node_id="node-1",
        action_module=module,
        intrinsic_reward=IntrinsicReward(
            belief_gain=belief_delta,
            graph_novelty=novelty,
            strategy_surprise=0.0,
        ),
        belief_delta=belief_delta,
        success=success,
        bridge_plan_valid=success,
        next_graph_snapshot_json=snapshot + "-next",
        run_id="run-1",
    )


def test_experience_buffer_save_and_load_roundtrip(tmp_path):
    buffer = ExperienceBuffer()
    record = _make_record(snapshot='{"nodes": {"n1": 1}}', belief_delta=0.4, novelty=0.2)
    buffer.add(record)

    path = tmp_path / "experience_buffer.json"
    buffer.save(path)

    loaded = ExperienceBuffer()
    loaded.load(path)
    restored = loaded.sample_batch(1)[0]

    assert len(loaded) == 1
    assert restored.graph_snapshot_json == record.graph_snapshot_json
    assert restored.next_graph_snapshot_json == record.next_graph_snapshot_json
    assert restored.intrinsic_reward.graph_novelty == record.intrinsic_reward.graph_novelty


def test_build_dpo_pairs_groups_by_target_and_state():
    buffer = ExperienceBuffer()
    buffer.add(_make_record(snapshot="same-state", belief_delta=0.6, novelty=0.3))
    buffer.add(_make_record(snapshot="same-state", belief_delta=0.1, novelty=0.0))
    buffer.add(_make_record(snapshot="other-state", belief_delta=0.9, novelty=0.4))
    buffer.add(_make_record(snapshot="other-state", belief_delta=0.85, novelty=0.4))

    pairs = build_dpo_pairs(buffer, min_reward_gap=0.15)

    assert len(pairs) == 1
    assert pairs[0].preferred.total_reward > pairs[0].rejected.total_reward


def test_expert_iteration_loop_collects_and_checkpoints(tmp_path):
    buffer = ExperienceBuffer()

    def suite_runner(*, experience_buffer):
        experience_buffer.add(_make_record(snapshot="s1", belief_delta=0.2))
        experience_buffer.add(_make_record(snapshot="s2", belief_delta=0.3))

    loop = ExpertIterationLoop(
        experience_buffer=buffer,
        suite_runner=suite_runner,
        checkpoint_dir=tmp_path,
        min_buffer_size_for_training=999,
    )
    result = loop.run_iteration()

    assert result.iteration == 1
    assert result.experiences_collected == 2
    assert (tmp_path / "iter_0001" / "experience_buffer.json").exists()
