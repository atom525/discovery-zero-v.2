from gaia.bp.factor_graph import CROMWELL_EPS

from dz_hypergraph.config import CONFIG
from dz_hypergraph.ingest import DEFAULT_CONFIDENCE, ingest_verified_claim
from dz_hypergraph.models import HyperGraph, Module


def test_default_confidence_is_config_driven():
    assert DEFAULT_CONFIDENCE[Module.PLAUSIBLE] == CONFIG.default_confidence_plausible
    assert DEFAULT_CONFIDENCE[Module.EXPERIMENT] == CONFIG.default_confidence_experiment
    assert DEFAULT_CONFIDENCE[Module.LEAN] == CONFIG.default_confidence_lean


def test_non_formal_refutation_uses_cromwell_floor():
    graph = HyperGraph()
    node = graph.add_node("claim", belief=0.7, prior=0.7)
    node_id = ingest_verified_claim(
        graph,
        claim_text="claim",
        verification_source="experiment",
        verdict="refuted",
    )
    assert node_id == node.id
    assert graph.nodes[node_id].prior >= CROMWELL_EPS
    assert graph.nodes[node_id].prior < 0.7
