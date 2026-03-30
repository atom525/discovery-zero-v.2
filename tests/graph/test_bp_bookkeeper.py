from discovery_zero.graph.ingest import ingest_verified_claim
from discovery_zero.graph.inference import SignalAccumulator, propagate_verification_signals
from discovery_zero.graph.models import HyperGraph


def test_ingest_verified_claim_sets_expected_priors():
    graph = HyperGraph()
    node_experiment = ingest_verified_claim(
        graph,
        claim_text="experiment claim",
        verification_source="experiment",
        verdict="verified",
    )
    node_lean = ingest_verified_claim(
        graph,
        claim_text="lean claim",
        verification_source="lean",
        verdict="verified",
    )
    node_refuted = ingest_verified_claim(
        graph,
        claim_text="bad claim",
        verification_source="experiment",
        verdict="refuted",
    )
    assert graph.nodes[node_experiment].belief >= 0.85
    assert graph.nodes[node_lean].state == "proven"
    assert graph.nodes[node_refuted].state == "refuted"


def test_propagate_verification_signals_threshold_trigger():
    graph = HyperGraph()
    node = graph.add_node("target", belief=0.2, prior=0.2)
    accumulator = SignalAccumulator(threshold=2)
    # First verified signal should not trigger propagation yet.
    iters1 = propagate_verification_signals(
        graph,
        [{"claim_id": node.id, "verdict": "verified", "backend": "judge"}],
        accumulator=accumulator,
        threshold=2,
        force=False,
    )
    assert iters1 == 0
    # Second deterministic signal triggers propagation.
    iters2 = propagate_verification_signals(
        graph,
        [{"claim_id": node.id, "verdict": "verified", "backend": "judge"}],
        accumulator=accumulator,
        threshold=2,
        force=False,
    )
    assert isinstance(iters2, int)
    assert graph.nodes[node.id].belief >= 0.6

