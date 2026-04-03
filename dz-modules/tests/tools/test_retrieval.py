from dz_hypergraph.models import HyperGraph, Module
from dz_hypergraph.tools.retrieval import HypergraphRetrievalIndex, RetrievalConfig


class FakeEmbeddingModel:
    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "runner" in lowered else 0.0,
                    1.0 if "ergodic" in lowered else 0.0,
                    1.0 if "density" in lowered else 0.0,
                ]
            )
        return vectors


def test_retrieval_prefers_semantically_matching_nodes():
    graph = HyperGraph()
    seed = graph.add_node("A density lemma", belief=1.0, prior=1.0, state="proven")
    runner = graph.add_node("A lonely runner spacing lemma", belief=0.92, prior=0.92, state="proven")
    target = graph.add_node("Lonely runner conjecture", belief=0.4, prior=0.4)
    graph.add_hyperedge([runner.id], target.id, Module.PLAUSIBLE, ["step"], 0.8)
    graph.add_hyperedge([seed.id], runner.id, Module.PLAUSIBLE, ["step"], 0.8)

    index = HypergraphRetrievalIndex(
        config=RetrievalConfig(max_results=2, min_similarity=0.0),
        embedding_model=FakeEmbeddingModel(),
    )
    assert index.build_from_graph(graph) == 2

    results = index.retrieve(
        "runner spacing",
        graph=graph,
        target_node_id=target.id,
    )

    assert results
    assert results[0].node_id == runner.id
    assert "runner" in results[0].statement.lower()


def test_format_retrieval_context_includes_scores():
    graph = HyperGraph()
    node = graph.add_node("Useful lemma", belief=0.95, prior=0.95, state="proven")
    index = HypergraphRetrievalIndex(
        config=RetrievalConfig(max_results=1, min_similarity=0.0),
        embedding_model=FakeEmbeddingModel(),
    )
    index.build_from_graph(graph)
    results = index.retrieve("lemma", graph=graph, target_node_id=node.id)
    text = index.format_retrieval_context(results, graph)
    assert "Possible relevant established results" in text
    assert node.id in text
