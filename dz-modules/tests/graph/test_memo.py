from dz_hypergraph.memo import (
    Claim,
    ClaimType,
    ResearchMemo,
    VerificationResult,
    VerificationStatus,
)


def test_claim_roundtrip_serialization():
    claim = Claim(
        claim_text="For all n >= 2, n^2 - n is even.",
        claim_type=ClaimType.STRUCTURAL,
        source_memo_id="memo_123",
        confidence=0.75,
    )
    payload = claim.model_dump_json()
    loaded = Claim.model_validate_json(payload)
    assert loaded.claim_text == claim.claim_text
    assert loaded.claim_type == ClaimType.STRUCTURAL
    assert loaded.verification_status == VerificationStatus.PENDING


def test_research_memo_claims_roundtrip():
    memo = ResearchMemo(
        raw_prose="If A then B. Empirically B holds for sampled inputs.",
        reasoning_structure=["A -> B", "sample check"],
        claims=[
            Claim(
                claim_text="A implies B",
                claim_type=ClaimType.STRUCTURAL,
                source_memo_id="memo_src",
            ),
            Claim(
                claim_text="B holds on 1000 samples",
                claim_type=ClaimType.QUANTITATIVE,
                source_memo_id="memo_src",
            ),
        ],
    )
    loaded = ResearchMemo.model_validate_json(memo.model_dump_json())
    assert len(loaded.claims) == 2
    assert loaded.claims[1].claim_type == ClaimType.QUANTITATIVE


def test_verification_result_delta_clamped():
    result = VerificationResult(
        claim_id="claim_x",
        verdict="verified",
        confidence_delta=2.0,
    )
    assert result.confidence_delta == 1.0

