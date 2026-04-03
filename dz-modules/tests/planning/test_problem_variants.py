from dz_hypergraph.models import HyperGraph
from dz_engine.problem_variants import ProblemVariant, ProblemVariantGenerator


def test_materialize_variants_creates_or_reuses_nodes():
    graph = HyperGraph()
    target = graph.add_node("Original conjecture", belief=0.4, prior=0.4)
    generator = ProblemVariantGenerator()
    variants = [
        ProblemVariant(
            original_node_id=target.id,
            variant_statement="Finite case of the original conjecture",
            variant_type="finite_case",
            difficulty_estimate=0.2,
        )
    ]

    created = generator.materialize_variants(graph, variants, domain="number_theory")

    assert len(created) == 1
    created_node = graph.nodes[created[0]]
    assert created_node.statement == "Finite case of the original conjecture"
    assert created_node.provenance == "variant:finite_case"
