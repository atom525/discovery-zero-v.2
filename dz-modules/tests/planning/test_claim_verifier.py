from dz_verify.claim_verifier import ClaimVerifier, VerifiableClaim


def test_extract_claims_includes_structural_and_heuristic():
    verifier = ClaimVerifier()
    claims = verifier.extract_claims(
        {
            "premises": [{"id": None, "statement": "if A then B"}],
            "conclusion": {"statement": "This mechanism should generalize by analogy."},
            "steps": ["n >= 2 implies n^2 - n is even"],
        }
    )
    claim_types = {c.claim_type for c in claims}
    assert "structural" in claim_types
    assert "heuristic" in claim_types
    assert "quantitative" in claim_types


def test_verify_heuristic_claim(monkeypatch):
    verifier = ClaimVerifier()
    monkeypatch.setattr(
        "dz_verify.claim_verifier.chat_completion",
        lambda **kwargs: {"choices": [{"message": {"content": '{"verdict":"verified","evidence":"coherent"}'}}]},
    )
    results = verifier.verify_claims(
        claims=[VerifiableClaim(claim_text="this seems plausible", source_prop_id=None, quantitative=False, claim_type="heuristic")],
        context="",
        model=None,
    )
    assert len(results) == 1
    assert results[0].verdict == "verified"

