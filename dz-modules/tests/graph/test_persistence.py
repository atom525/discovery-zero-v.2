from pathlib import Path

from dz_hypergraph.models import HyperGraph, Module
from dz_hypergraph.persistence import save_graph, load_graph


class TestPersistence:
    def test_save_and_load_empty(self, tmp_graph_dir):
        path = tmp_graph_dir / "graph.json"
        g = HyperGraph()
        save_graph(g, path)
        loaded = load_graph(path)
        assert loaded.summary()["num_nodes"] == 0

    def test_save_and_load_with_data(self, tmp_graph_dir):
        path = tmp_graph_dir / "graph.json"
        g = HyperGraph()
        n1 = g.add_node("axiom A", belief=1.0)
        n2 = g.add_node("conjecture B")
        g.add_hyperedge([n1.id], n2.id, Module.PLAUSIBLE, ["reasoning"], 0.6)
        save_graph(g, path)
        loaded = load_graph(path)
        assert loaded.summary()["num_nodes"] == 2
        assert loaded.summary()["num_edges"] == 1
        assert loaded.nodes[n1.id].statement == "axiom A"
        assert loaded.nodes[n1.id].belief == 1.0

    def test_load_nonexistent_returns_empty(self, tmp_graph_dir):
        path = tmp_graph_dir / "nonexistent.json"
        g = load_graph(path)
        assert g.summary()["num_nodes"] == 0
