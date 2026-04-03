from dz_hypergraph.models import HyperGraph
from dz_hypergraph.session import GraphSession
from dz_engine.bridge import validate_bridge_plan_payload
from dz_engine.bridge_executor import SpeculativeDecomposer


class DummyPAV:
    def predict_advantage(self, graph, target_node_id, candidate_node_id, candidate_module):
        node = graph.nodes[candidate_node_id]
        if "lemma" in node.statement.lower():
            return 0.6
        return 0.1


def _make_plan(target_statement: str, bridge_grade: str):
    return validate_bridge_plan_payload(
        {
            "target_statement": target_statement,
            "propositions": [
                {
                    "id": "P1",
                    "statement": "Seed fact",
                    "role": "seed",
                    "grade": "A",
                    "depends_on": [],
                },
                {
                    "id": "P2",
                    "statement": f"{bridge_grade} bridge lemma",
                    "role": "bridge",
                    "grade": bridge_grade,
                    "depends_on": ["P1"],
                },
                {
                    "id": "P3",
                    "statement": target_statement,
                    "role": "target",
                    "grade": "B",
                    "depends_on": ["P2"],
                },
            ],
            "chain": [
                {
                    "id": "S1",
                    "statement": "Build the bridge lemma.",
                    "uses": ["P1"],
                    "concludes": ["P2"],
                    "grade": bridge_grade,
                },
                {
                    "id": "S2",
                    "statement": "Conclude the target.",
                    "uses": ["P2"],
                    "concludes": ["P3"],
                    "grade": "B",
                },
            ],
        }
    )


def test_speculative_decomposer_scores_candidates_and_rolls_back(monkeypatch):
    graph = HyperGraph()
    graph.add_node("Seed fact", belief=1.0, state="proven")
    target = graph.add_node("Open target", belief=0.2, prior=0.2)
    session = GraphSession(graph)

    plans_by_temperature = {
        0.3: _make_plan("Open target", "D"),
        0.5: _make_plan("Open target", "B"),
        0.7: _make_plan("Open target", "B"),
    }

    def generate_plan_fn(**kwargs):
        return plans_by_temperature[kwargs["temperature"]]

    def fake_propagate_beliefs(graph, *args, **kwargs):
        target_node = graph.nodes[target.id]
        lean_like_edges = sum(
            1
            for edge in graph.edges.values()
            if edge.edge_type == "decomposition" and edge.confidence >= 0.8
        )
        target_node.belief = min(1.0, 0.2 + 0.25 * lean_like_edges)
        return 1

    monkeypatch.setattr(
        "dz_hypergraph.inference.propagate_beliefs",
        fake_propagate_beliefs,
    )

    decomposer = SpeculativeDecomposer(
        session,
        target.id,
        generate_plan_fn=generate_plan_fn,
        pav=DummyPAV(),
        num_candidates=3,
        temperatures=[0.3, 0.5, 0.7],
    )

    original_node_count = len(session.graph.nodes)
    original_edge_count = len(session.graph.edges)
    candidates = decomposer.generate_candidates()
    scored = decomposer.score_candidates(candidates)

    assert len(candidates) == 2
    assert scored[0].candidate.plan.propositions[1].grade == "B"
    assert scored[0].belief_gain > scored[-1].belief_gain
    assert len(session.graph.nodes) == original_node_count
    assert len(session.graph.edges) == original_edge_count


def test_speculative_decomposer_generate_and_score_returns_best(monkeypatch):
    graph = HyperGraph()
    graph.add_node("Seed fact", belief=1.0, state="proven")
    target = graph.add_node("Main target", belief=0.3)
    session = GraphSession(graph)

    def generate_plan_fn(**kwargs):
        if kwargs["temperature"] < 0.4:
            return _make_plan("Main target", "D")
        return _make_plan("Main target", "A")

    monkeypatch.setattr(
        "dz_hypergraph.inference.propagate_beliefs",
        lambda graph, *args, **kwargs: graph.nodes[target.id].__setattr__("belief", 0.85),
    )

    best = SpeculativeDecomposer(
        session,
        target.id,
        generate_plan_fn=generate_plan_fn,
        num_candidates=2,
        temperatures=[0.3, 0.5],
    ).generate_and_score()

    assert best is not None
    assert best.candidate.plan.target_statement == "Main target"
    assert best.total_score > 0
