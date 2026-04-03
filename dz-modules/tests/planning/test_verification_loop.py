from dz_hypergraph.memo import Claim, ClaimType, ResearchMemo, VerificationResult
from dz_hypergraph.models import HyperGraph
from dz_verify.verification_loop import VerificationLoop, VerificationLoopConfig


def test_verification_loop_ingests_and_propagates(monkeypatch):
    graph = HyperGraph()
    target = graph.add_node("Target theorem", belief=0.2, prior=0.2)
    loop = VerificationLoop(config=VerificationLoopConfig(max_iterations=1, bp_propagation_threshold=1))

    monkeypatch.setattr(
        loop,
        "_generate_reasoning",
        lambda **kwargs: ResearchMemo(
            raw_prose="Claim: target theorem holds for all tested cases.",
            source_node_id=target.id,
            iteration=1,
        ),
    )
    monkeypatch.setattr(
        loop.claim_pipeline,
        "extract_claims",
        lambda **kwargs: [
            Claim(
                claim_text="Target theorem",
                claim_type=ClaimType.HEURISTIC,
                source_memo_id="memo_1",
                confidence=0.7,
            )
        ],
    )
    monkeypatch.setattr(
        loop,
        "_verify_claims_parallel",
        lambda claims, **kwargs: [
            VerificationResult(
                claim_id=claims[0].id,
                verdict="verified",
                evidence_text="validated",
                confidence_delta=0.3,
                backend="judge",
            )
        ],
    )

    result = loop.run(graph=graph, target_node_id=target.id)
    assert result.iterations_completed == 1
    assert graph.nodes[target.id].belief >= 0.6
    assert result.traces[0].verified == 1

