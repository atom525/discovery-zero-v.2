import pytest
from dz_hypergraph.models import Node, Hyperedge, HyperGraph, Module


class TestNode:
    def test_create_node(self):
        node = Node(statement="For all triangles, the angle sum is 180 degrees")
        assert node.statement == "For all triangles, the angle sum is 180 degrees"
        assert node.belief == 0.5
        assert node.id is not None

    def test_create_axiom_node(self):
        node = Node(statement="Two points determine a line", belief=1.0)
        assert node.belief == 1.0

    def test_node_ids_are_unique(self):
        n1 = Node(statement="A")
        n2 = Node(statement="B")
        assert n1.id != n2.id

    def test_node_backward_compat_without_new_metadata_fields(self):
        node = Node.model_validate({"statement": "legacy", "belief": 0.2})
        assert node.verification_source is None
        assert node.memo_ref is None


class TestHyperedge:
    def test_create_hyperedge(self):
        edge = Hyperedge(
            premise_ids=["n1", "n2"],
            conclusion_id="n3",
            module=Module.PLAUSIBLE,
            steps=["step 1", "step 2"],
            confidence=0.6,
        )
        assert edge.module == Module.PLAUSIBLE
        assert edge.confidence == 0.6
        assert len(edge.premise_ids) == 2

    def test_confidence_clamped(self):
        edge = Hyperedge(
            premise_ids=["n1"],
            conclusion_id="n2",
            module=Module.EXPERIMENT,
            steps=[],
            confidence=1.5,
        )
        assert edge.confidence == 1.0

    def test_hyperedge_claim_refs_default(self):
        edge = Hyperedge(
            premise_ids=["n1"],
            conclusion_id="n2",
            module=Module.EXPERIMENT,
            steps=[],
        )
        assert edge.claim_refs == []


class TestHyperGraph:
    def test_add_node(self):
        g = HyperGraph()
        node = g.add_node("Two points determine a line", belief=1.0)
        assert node.id in g.nodes
        assert g.nodes[node.id].statement == "Two points determine a line"

    def test_add_hyperedge(self):
        g = HyperGraph()
        n1 = g.add_node("premise 1", belief=1.0)
        n2 = g.add_node("premise 2", belief=1.0)
        n3 = g.add_node("conclusion")
        edge = g.add_hyperedge(
            premise_ids=[n1.id, n2.id],
            conclusion_id=n3.id,
            module=Module.PLAUSIBLE,
            steps=["reasoning step"],
            confidence=0.5,
        )
        assert edge.id in g.edges
        assert edge.id in g.get_edges_to(n3.id)

    def test_add_hyperedge_invalid_node(self):
        g = HyperGraph()
        with pytest.raises(ValueError, match="not found"):
            g.add_hyperedge(
                premise_ids=["nonexistent"],
                conclusion_id="also_nonexistent",
                module=Module.PLAUSIBLE,
                steps=[],
                confidence=0.5,
            )

    def test_get_edges_to(self):
        g = HyperGraph()
        n1 = g.add_node("A", belief=1.0)
        n2 = g.add_node("B")
        e1 = g.add_hyperedge([n1.id], n2.id, Module.PLAUSIBLE, ["step"], 0.5)
        e2 = g.add_hyperedge([n1.id], n2.id, Module.EXPERIMENT, ["code"], 0.9)
        edges_to_b = g.get_edges_to(n2.id)
        assert len(edges_to_b) == 2

    def test_get_edges_from(self):
        g = HyperGraph()
        n1 = g.add_node("A", belief=1.0)
        n2 = g.add_node("B")
        n3 = g.add_node("C")
        g.add_hyperedge([n1.id], n2.id, Module.PLAUSIBLE, [], 0.5)
        g.add_hyperedge([n1.id], n3.id, Module.PLAUSIBLE, [], 0.5)
        edges_from_a = g.get_edges_from(n1.id)
        assert len(edges_from_a) == 2

    def test_add_hyperedge_filters_self_loop(self):
        g = HyperGraph()
        n1 = g.add_node("A", belief=1.0)
        n2 = g.add_node("B", belief=0.5)
        edge = g.add_hyperedge(
            premise_ids=[n1.id, n2.id],
            conclusion_id=n2.id,
            module=Module.EXPERIMENT,
            steps=["self-referential step"],
            confidence=0.8,
        )
        assert n2.id not in edge.premise_ids
        assert edge.premise_ids == [n1.id]

    def test_add_hyperedge_self_loop_only_premise(self):
        g = HyperGraph()
        n1 = g.add_node("A", belief=1.0)
        edge = g.add_hyperedge(
            premise_ids=[n1.id],
            conclusion_id=n1.id,
            module=Module.EXPERIMENT,
            steps=["pure self-loop"],
            confidence=0.8,
        )
        assert edge.premise_ids == []

    def test_summary(self):
        g = HyperGraph()
        g.add_node("A", belief=1.0)
        g.add_node("B")
        summary = g.summary()
        assert summary["num_nodes"] == 2
        assert summary["num_edges"] == 0
