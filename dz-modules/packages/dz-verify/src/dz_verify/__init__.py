"""Verification layer facade for Discovery Zero."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dz_hypergraph.ingest import ingest_verified_claim
from dz_hypergraph.memo import Claim
from dz_hypergraph.models import HyperGraph
from dz_verify.claim_pipeline import ClaimPipeline
from dz_verify.claim_verifier import ClaimVerificationResult, ClaimVerifier, VerifiableClaim
from dz_verify.verification_loop import VerificationLoop


@dataclass
class VerificationSummary:
    claims: list[Claim]
    results: list[ClaimVerificationResult]


def extract_claims(
    *,
    prose: str,
    context: str,
    source_memo_id: str,
    model: Optional[str] = None,
    record_dir: Optional[Path] = None,
) -> list[Claim]:
    """Extract claims from reasoning prose."""
    pipeline = ClaimPipeline()
    return pipeline.extract_claims(
        prose=prose,
        context=context,
        source_memo_id=source_memo_id,
        model=model,
        record_dir=record_dir,
    )


def verify_claims(
    *,
    prose: str,
    context: str,
    graph: HyperGraph,
    source_memo_id: str,
    model: Optional[str] = None,
    record_dir: Optional[Path] = None,
    claim_verifier: Optional[ClaimVerifier] = None,
) -> VerificationSummary:
    """Run extraction + verification and ingest results back into the graph."""
    pipeline = ClaimPipeline()
    verifier = claim_verifier or ClaimVerifier()
    claims = pipeline.extract_claims(
        prose=prose,
        context=context,
        source_memo_id=source_memo_id,
        model=model,
        record_dir=record_dir,
    )
    verifiable = [
        VerifiableClaim(
            claim_text=item.claim_text,
            source_prop_id=item.node_id,
            quantitative=item.claim_type.value == "quantitative",
            claim_type=item.claim_type.value,
        )
        for item in claims
    ]
    results = verifier.verify_claims(verifiable, context=context, model=model, record_dir=record_dir)
    for result in results:
        ingest_verified_claim(
            graph,
            claim_text=result.claim.claim_text,
            verification_source="experiment"
            if result.claim.claim_type == "quantitative"
            else "llm_judge",
            verdict=result.verdict,
        )
    return VerificationSummary(claims=claims, results=results)


def run_verification_loop(*args, **kwargs):
    """Compatibility facade for existing verification loop call sites."""
    loop = VerificationLoop(*args, **kwargs)
    return loop.run()
