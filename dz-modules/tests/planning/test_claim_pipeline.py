import os

import pytest

from dz_hypergraph.memo import Claim, ClaimType
from dz_hypergraph.models import HyperGraph
from dz_verify.claim_pipeline import ClaimPipeline


def test_classify_claims_groups_by_type():
    pipeline = ClaimPipeline()
    claims = [
        Claim(claim_text="x^2 >= 0", claim_type=ClaimType.QUANTITATIVE, source_memo_id="m1"),
        Claim(claim_text="if A then B", claim_type=ClaimType.STRUCTURAL, source_memo_id="m1"),
        Claim(claim_text="this appears plausible", claim_type=ClaimType.HEURISTIC, source_memo_id="m1"),
    ]
    grouped = pipeline.classify_claims(claims)
    assert len(grouped[ClaimType.QUANTITATIVE]) == 1
    assert len(grouped[ClaimType.STRUCTURAL]) == 1
    assert len(grouped[ClaimType.HEURISTIC]) == 1


def test_prioritize_claims_prefers_novel_structural():
    g = HyperGraph()
    existing = g.add_node("known fact", belief=0.6, prior=0.6)
    pipeline = ClaimPipeline()
    claims = [
        Claim(claim_text="known fact", claim_type=ClaimType.HEURISTIC, source_memo_id="m2", confidence=0.7),
        Claim(claim_text="if X then Y", claim_type=ClaimType.STRUCTURAL, source_memo_id="m2", confidence=0.4),
    ]
    ranked = pipeline.prioritize_claims(claims=claims, graph=g, target_node_id=existing.id)
    assert ranked[0].claim_text == "if X then Y"


def test_extract_claims_uses_skill_output(monkeypatch):
    pipeline = ClaimPipeline()

    def fake_run_skill(*args, **kwargs):
        return "{}", {
            "claims": [
                {
                    "claim_text": "for all n>=1, n+n=2n",
                    "claim_type": "structural",
                    "confidence": 0.8,
                    "evidence": "algebraic identity",
                }
            ]
        }

    monkeypatch.setattr("dz_verify.claim_pipeline.run_skill", fake_run_skill)
    claims = pipeline.extract_claims(
        prose="for all n>=1, n+n=2n",
        context="",
        source_memo_id="memo_unit",
        model=None,
    )
    assert len(claims) == 1
    assert claims[0].claim_type == ClaimType.STRUCTURAL
    assert claims[0].confidence == pytest.approx(0.8)


@pytest.mark.llm
def test_extract_claims_real_llm_integration():
    if not os.environ.get("LITELLM_PROXY_API_BASE") or not os.environ.get("LITELLM_PROXY_API_KEY"):
        pytest.skip("LLM endpoint not configured")
    pipeline = ClaimPipeline()
    claims = pipeline.extract_claims(
        prose=(
            "For all n >= 2, n^2 - n is even. "
            "If n is prime then n has no nontrivial divisors."
        ),
        context="number theory basics",
        source_memo_id="memo_live",
        model=None,
    )
    assert claims
    assert any(c.claim_type in (ClaimType.QUANTITATIVE, ClaimType.STRUCTURAL) for c in claims)

