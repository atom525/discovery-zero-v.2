"""Verification-driven discovery loop."""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass, field
from typing import Optional

from dz_hypergraph.ingest import ingest_verified_claim
from dz_hypergraph.inference import SignalAccumulator, propagate_verification_signals
from dz_hypergraph.memo import Claim, ResearchMemo, VerificationResult
from dz_hypergraph.models import HyperGraph
from dz_hypergraph.config import CONFIG
from dz_verify.claim_pipeline import ClaimPipeline
from dz_verify.claim_verifier import ClaimVerifier, VerifiableClaim
from dz_verify.lean_feedback import LeanFeedbackParser, StructuralClaimRouter
from dz_hypergraph.tools.llm import chat_completion, extract_text_content

logger = logging.getLogger(__name__)


@dataclass
class VerificationLoopConfig:
    max_iterations: int = 8
    verification_parallel_workers: int = 3
    max_claims_per_memo: int = 10
    bp_propagation_threshold: int = 3
    max_decompose_depth: int = 4


@dataclass
class VerificationIterationTrace:
    iteration: int
    extracted_claims: int = 0
    verified: int = 0
    refuted: int = 0
    inconclusive: int = 0
    bp_iterations: int = 0


@dataclass
class VerificationLoopResult:
    success: bool = False
    iterations_completed: int = 0
    traces: list[VerificationIterationTrace] = field(default_factory=list)
    latest_feedback: str = ""


class VerificationLoop:
    def __init__(
        self,
        *,
        claim_pipeline: Optional[ClaimPipeline] = None,
        claim_verifier: Optional[ClaimVerifier] = None,
        lean_feedback_parser: Optional[LeanFeedbackParser] = None,
        structural_router: Optional[StructuralClaimRouter] = None,
        config: Optional[VerificationLoopConfig] = None,
        model: Optional[str] = None,
    ) -> None:
        self.config = config or VerificationLoopConfig()
        self.model = model
        self.claim_pipeline = claim_pipeline or ClaimPipeline()
        self.claim_verifier = claim_verifier or ClaimVerifier(
            max_claims_per_call=self.config.max_claims_per_memo,
            lean_verify_timeout=CONFIG.lean_timeout,
        )
        self.lean_feedback_parser = lean_feedback_parser or LeanFeedbackParser()
        self.structural_router = structural_router or StructuralClaimRouter(max_decompose_depth=self.config.max_decompose_depth)
        self._signal_accumulator = SignalAccumulator(threshold=max(1, self.config.bp_propagation_threshold))

    def run(
        self,
        *,
        graph: HyperGraph,
        target_node_id: str,
        max_iterations: Optional[int] = None,
        initial_feedback: str = "",
    ) -> VerificationLoopResult:
        if target_node_id not in graph.nodes:
            return VerificationLoopResult(success=False, iterations_completed=0, latest_feedback="target node not found")

        feedback = initial_feedback
        result = VerificationLoopResult(success=False)
        rounds = max_iterations if max_iterations is not None else self.config.max_iterations

        for iteration in range(1, rounds + 1):
            target = graph.nodes.get(target_node_id)
            if target is None:
                break
            if target.state in {"proven", "refuted"}:
                result.success = target.state == "proven"
                result.iterations_completed = iteration - 1
                break

            memo = self._generate_reasoning(graph=graph, target_node_id=target_node_id, iteration=iteration, feedback=feedback)
            claims = self.claim_pipeline.extract_claims(
                prose=memo.raw_prose,
                context=feedback,
                source_memo_id=memo.id,
                model=self.model,
            )
            claims = self.claim_pipeline.prioritize_claims(claims=claims, graph=graph, target_node_id=target_node_id)
            if claims:
                memo.claims = claims[: self.config.max_claims_per_memo]
            verification_results = self._verify_claims_parallel(memo.claims, feedback=feedback)
            ingested_results = self._ingest_verified_claims(graph, memo, verification_results)
            bp_iters = propagate_verification_signals(
                graph,
                ingested_results,
                threshold=self.config.bp_propagation_threshold,
                accumulator=self._signal_accumulator,
                force=True,
            )
            trace = VerificationIterationTrace(
                iteration=iteration,
                extracted_claims=len(memo.claims),
                verified=sum(1 for item in ingested_results if item.verdict == "verified"),
                refuted=sum(1 for item in ingested_results if item.verdict == "refuted"),
                inconclusive=sum(1 for item in ingested_results if item.verdict == "inconclusive"),
                bp_iterations=bp_iters,
            )
            result.traces.append(trace)
            result.iterations_completed = iteration
            feedback = self._build_feedback(ingested_results)
            result.latest_feedback = feedback
            target = graph.nodes.get(target_node_id)
            if target and target.state in {"proven", "refuted"}:
                result.success = target.state == "proven"
                break

        return result

    def _generate_reasoning(
        self,
        *,
        graph: HyperGraph,
        target_node_id: str,
        iteration: int,
        feedback: str,
    ) -> ResearchMemo:
        target = graph.nodes[target_node_id]
        prompt = (
            f"Iteration: {iteration}\n"
            f"Target claim:\n{target.statement}\n\n"
            "Write a concise research memo containing concrete, verifiable claims "
            "that can move the target toward proof or refutation.\n\n"
            f"Feedback from previous verification rounds:\n{feedback or '(none)'}\n"
        )
        response = chat_completion(
            messages=[
                {"role": "system", "content": "You are a mathematical discovery assistant. Produce focused reasoning prose."},
                {"role": "user", "content": prompt},
            ],
            model=self.model,
            temperature=0.2,
        )
        prose = extract_text_content(response).strip()
        return ResearchMemo(
            raw_prose=prose or target.statement,
            source_node_id=target_node_id,
            iteration=iteration,
        )

    def _verify_claims_parallel(self, claims: list[Claim], *, feedback: str) -> list[VerificationResult]:
        if not claims:
            return []

        def _verify_single(claim: Claim) -> VerificationResult:
            verifiable = VerifiableClaim(
                claim_text=claim.claim_text,
                source_prop_id=claim.node_id,
                quantitative=(claim.claim_type.value == "quantitative"),
                claim_type=claim.claim_type.value,
            )
            # Structural claims are routed through router policy before claim verifier.
            if claim.claim_type.value == "structural":
                route = self.structural_router.route_structural_claim(claim, depth=claim.depth)
                if route.mode == "decompose":
                    subclaims = self.structural_router.decompose_to_subclaims(claim, source_memo_id=claim.source_memo_id)
                    if subclaims:
                        return VerificationResult(
                            claim_id=claim.id,
                            verdict="inconclusive",
                            evidence_text=f"decomposed into {len(subclaims)} structural subclaims",
                            confidence_delta=0.0,
                            backend="lean_decompose",
                            raw_result={"subclaims": [item.claim_text for item in subclaims]},
                        )
            verified = self.claim_verifier.verify_claims(
                claims=[verifiable],
                context=feedback,
                model=self.model,
            )
            if not verified:
                return VerificationResult(
                    claim_id=claim.id,
                    verdict="inconclusive",
                    evidence_text="verification backend returned no result",
                    backend="none",
                )
            item = verified[0]
            backend = "experiment"
            if verifiable.claim_type == "structural":
                backend = "lean"
            elif verifiable.claim_type == "heuristic":
                backend = "judge"
            delta = 0.4 if item.verdict == "verified" else (-0.5 if item.verdict == "refuted" else 0.0)
            return VerificationResult(
                claim_id=claim.id,
                verdict=item.verdict,
                evidence_text=item.evidence,
                confidence_delta=delta,
                code=item.code,
                lean_error=str(item.raw_result.get("error_message", "")) if isinstance(item.raw_result, dict) else "",
                backend=backend,
                raw_result=item.raw_result,
            )

        workers = max(1, self.config.verification_parallel_workers)
        per_claim_timeout = float(CONFIG.lean_timeout) + 60.0
        out: list[VerificationResult] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_verify_single, claim) for claim in claims]
            for future in concurrent.futures.as_completed(futures, timeout=per_claim_timeout * len(claims) + 120):
                try:
                    out.append(future.result(timeout=per_claim_timeout))
                except Exception as exc:
                    logger.warning("Verification future timed out or failed: %s", exc)
        # Preserve deterministic order by claim_id appearance order
        order = {claim.id: idx for idx, claim in enumerate(claims)}
        out.sort(key=lambda item: order.get(item.claim_id, 10_000))
        return out

    def _ingest_verified_claims(
        self,
        graph: HyperGraph,
        memo: ResearchMemo,
        verification_results: list[VerificationResult],
    ) -> list[VerificationResult]:
        by_id = {claim.id: claim for claim in memo.claims}
        ingested: list[VerificationResult] = []
        for vr in verification_results:
            claim = by_id.get(vr.claim_id)
            if claim is None:
                continue
            node_id = ingest_verified_claim(
                graph,
                claim_text=claim.claim_text,
                verification_source=vr.backend or "judge",
                verdict=vr.verdict,
                domain=None,
                source_memo_id=claim.source_memo_id,
                claim_id=claim.id,
            )
            claim.node_id = node_id
            # Use node ID for downstream BP mapping.
            ingested.append(
                VerificationResult(
                    claim_id=node_id,
                    verdict=vr.verdict,
                    evidence_text=vr.evidence_text,
                    confidence_delta=vr.confidence_delta,
                    code=vr.code,
                    lean_error=vr.lean_error,
                    backend=vr.backend,
                    raw_result=vr.raw_result,
                )
            )
        return ingested

    def _build_feedback(self, verification_results: list[VerificationResult]) -> str:
        lines: list[str] = []
        for item in verification_results:
            prefix = item.verdict.upper()
            line = f"- [{prefix}] claim={item.claim_id} backend={item.backend}: {item.evidence_text}"
            lines.append(line[:600])
            if item.backend.startswith("lean") and item.lean_error:
                gap = self.lean_feedback_parser.parse_lean_error(item.lean_error)
                lines.append(f"  Lean gap: {gap.gap_type} -> {gap.message[:200]}")
        return "\n".join(lines[:30])

