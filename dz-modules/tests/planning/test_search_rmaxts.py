import pytest

from dz_hypergraph.models import HyperGraph, Module
from dz_engine.search import (
    FrontierNode,
    IntrinsicReward,
    RMaxTSConfig,
    RMaxTSSearch,
    SearchState,
)


class DummyPAV:
    def predict_advantage(self, graph, target_node_id, candidate_node_id, candidate_module):
        scores = {
            ("strong", Module.LEAN): 0.9,
            ("weak", Module.PLAUSIBLE): 0.2,
        }
        return scores.get((candidate_node_id, candidate_module), 0.0)


class DummyCuriosity:
    def exploration_bonus(self, graph, target_node_id, candidate):
        return candidate.diversity_bonus + candidate.bridge_bonus


def test_rmaxts_select_action_prefers_best_scored_candidate():
    graph = HyperGraph()
    target = graph.add_node("Target theorem", belief=0.4)
    strong = graph.add_node("Strong frontier", belief=0.8)
    weak = graph.add_node("Weak frontier", belief=0.3)

    state = SearchState()
    state.record_action(strong.id, Module.LEAN, 0.6, success=True)
    state.record_action(weak.id, Module.PLAUSIBLE, 0.1, success=True)

    candidates = [
        FrontierNode(
            node_id=strong.id,
            belief=strong.belief,
            priority=0.8,
            bridge_bonus=0.2,
            diversity_bonus=0.1,
            suggested_module=Module.LEAN,
        ),
        FrontierNode(
            node_id=weak.id,
            belief=weak.belief,
            priority=0.7,
            bridge_bonus=0.0,
            diversity_bonus=0.0,
            suggested_module=Module.PLAUSIBLE,
        ),
    ]

    selector = RMaxTSSearch(
        pav=DummyPAV(),
        curiosity=DummyCuriosity(),
        config=RMaxTSConfig(c_puct=1.2, c_intrinsic=0.5),
    )
    node_id, module = selector.select_action(graph, target.id, candidates, state)

    assert node_id == strong.id
    assert module == Module.LEAN


def test_intrinsic_reward_uses_weighted_components():
    selector = RMaxTSSearch(config=RMaxTSConfig(
        belief_weight=0.5,
        novelty_weight=0.3,
        surprise_weight=0.2,
    ))

    reward = selector.compute_intrinsic_reward(
        belief_before=0.2,
        belief_after=0.7,
        novelty=0.4,
        surprise=0.5,
    )

    assert isinstance(reward, IntrinsicReward)
    assert reward.belief_gain == pytest.approx(0.5)
    assert reward.total == pytest.approx(0.5 * 0.5 + 0.3 * 0.4 + 0.2 * 0.5)
