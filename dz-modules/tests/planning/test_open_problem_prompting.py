from dz_hypergraph.models import HyperGraph
from dz_engine.orchestrator import (
    _is_open_problem_mode,
    _open_problem_novelty_score,
)


def test_open_problem_mode_detects_frontier_feedback():
    assert _is_open_problem_mode(
        "A frontier open problem",
        "This is an open-problem benchmark with frontier-assisted seeds.",
    ) is True


def test_open_problem_novelty_score_rewards_new_premises_and_methods():
    score = _open_problem_novelty_score(
        {
            "premises": [{"id": None, "statement": "Introduce a new obstruction invariant"}],
            "steps": ["New method: construct an obstruction certificate and reduce to a finite search."],
            "conclusion": {"statement": "A new reduction route for the target"},
        }
    )
    assert score >= 0.5
