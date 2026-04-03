"""Claim extraction, classification, and prioritization pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from dz_hypergraph.memo import Claim, ClaimType, VerificationStatus
from dz_hypergraph.models import HyperGraph
from dz_hypergraph.belief_gap import BeliefGapAnalyser
from dz_hypergraph.tools.llm import run_skill

if TYPE_CHECKING:
    from dz_hypergraph.bridge_models import BridgePlan


@dataclass
class ClaimPipelineConfig:
    max_claims_per_memo: int = 10


class ClaimPipeline:
    """Production claim pipeline backed by real LLM extraction."""

    def __init__(self, config: Optional[ClaimPipelineConfig] = None) -> None:
        self.config = config or ClaimPipelineConfig()
        self._gap_analyser = BeliefGapAnalyser()

    def extract_claims(
        self,
        *,
        prose: str,
        context: str,
        source_memo_id: str,
        model: Optional[str] = None,
        record_dir: Optional[Path] = None,
        bridge_plan: Optional["BridgePlan"] = None,
    ) -> list[Claim]:
        """Extract claims from free-form reasoning prose via LLM skill.

        When ``bridge_plan`` is provided, the list of non-seed bridge
        propositions is injected into the prompt.  The LLM is asked to also
        output a ``bridge_proposition_id`` for each claim, indicating which
        bridge proposition the claim corresponds to (or null if none match).
        This enables verification results to be written back to the correct
        graph node via ``bridge_node_map`` without any text matching.
        """
        task_input = (
            "Extract verifiable claims from the reasoning memo.\n\n"
            f"Memo:\n{prose}\n\n"
            f"Context:\n{context}\n\n"
            "Return JSON: {\"claims\": [{\"claim_text\": str, \"claim_type\": \"quantitative|structural|heuristic\","
            " \"confidence\": number, \"evidence\": str, \"bridge_proposition_id\": str|null}]}"
        )

        # Inject bridge plan propositions as context so the LLM can map
        # each extracted claim to a specific bridge proposition ID.
        if bridge_plan is not None:
            non_seed_props = [
                p for p in bridge_plan.propositions if p.role != "seed"
            ]
            if non_seed_props:
                prop_lines = "\n".join(
                    f"  {p.id} [{p.grade}]: {p.statement}"
                    for p in non_seed_props
                )
                task_input += (
                    "\n\nBridge plan propositions (non-seed). "
                    "For each extracted claim, set 'bridge_proposition_id' to the ID of the "
                    "most relevant proposition below, or null if none matches closely:\n"
                    f"{prop_lines}\n"
                )

        schema = {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim_text": {"type": "string"},
                            "claim_type": {"type": "string", "enum": ["quantitative", "structural", "heuristic"]},
                            "confidence": {"type": "number"},
                            "evidence": {"type": "string"},
                            "bridge_proposition_id": {"type": ["string", "null"]},
                        },
                        "required": ["claim_text", "claim_type"],
                        "additionalProperties": True,
                    },
                }
            },
            "required": ["claims"],
            "additionalProperties": True,
        }
        record_path: Optional[Path] = None
        if record_dir is not None:
            record_dir.mkdir(parents=True, exist_ok=True)
            record_path = record_dir / "claim_extraction_attempt_1.txt"
        _, parsed = run_skill(
            "claim_extraction.skill.md",
            task_input,
            model=model,
            schema=schema,
            record_path=record_path,
        )
        claim_items = parsed.get("claims", []) if isinstance(parsed, dict) else []

        # Build a set of valid bridge proposition IDs for validation.
        valid_bridge_ids: set[str] = set()
        if bridge_plan is not None:
            valid_bridge_ids = {p.id for p in bridge_plan.propositions if p.role != "seed"}

        claims: list[Claim] = []
        for item in claim_items:
            if not isinstance(item, dict):
                continue
            text = str(item.get("claim_text", "")).strip()
            if not text:
                continue
            raw_type = str(item.get("claim_type", "")).strip().lower()
            if raw_type not in {"quantitative", "structural", "heuristic"}:
                raw_type = self._infer_claim_type(text)
            # Only accept bridge_proposition_id if it maps to a known proposition.
            raw_bridge_id = item.get("bridge_proposition_id")
            bridge_proposition_id: Optional[str] = None
            if raw_bridge_id and str(raw_bridge_id).strip() in valid_bridge_ids:
                bridge_proposition_id = str(raw_bridge_id).strip()
            claims.append(
                Claim(
                    claim_text=text,
                    claim_type=ClaimType(raw_type),
                    verification_status=VerificationStatus.PENDING,
                    source_memo_id=source_memo_id,
                    evidence=str(item.get("evidence", "")).strip(),
                    confidence=float(item.get("confidence", 0.0) or 0.0),
                    bridge_proposition_id=bridge_proposition_id,
                )
            )
        if not claims:
            claims = self._fallback_extract_claims(prose=prose, source_memo_id=source_memo_id)
        return claims[: self.config.max_claims_per_memo]

    def classify_claims(self, claims: list[Claim]) -> dict[ClaimType, list[Claim]]:
        grouped: dict[ClaimType, list[Claim]] = {
            ClaimType.QUANTITATIVE: [],
            ClaimType.STRUCTURAL: [],
            ClaimType.HEURISTIC: [],
        }
        for claim in claims:
            grouped[claim.claim_type].append(claim)
        return grouped

    def prioritize_claims(
        self,
        *,
        claims: list[Claim],
        graph: HyperGraph,
        target_node_id: Optional[str] = None,
    ) -> list[Claim]:
        """Prioritize by novelty, critical-path relevance, and confidence."""
        critical_nodes: set[str] = set()
        if target_node_id and target_node_id in graph.nodes:
            critical_nodes = {node_id for node_id, _ in self._gap_analyser.find_critical_gaps(graph, target_node_id, top_k=12)}

        def _score(claim: Claim) -> float:
            score = max(0.0, min(1.0, float(claim.confidence)))
            existing = graph.find_node_ids_by_statement(claim.claim_text)
            if not existing:
                score += 0.5
            else:
                score -= 0.15
            if claim.claim_type == ClaimType.STRUCTURAL:
                score += 0.2
            if claim.claim_type == ClaimType.QUANTITATIVE:
                score += 0.1
            if any(node_id in critical_nodes for node_id in existing):
                score += 0.35
            return score

        return sorted(claims, key=_score, reverse=True)

    def _fallback_extract_claims(self, *, prose: str, source_memo_id: str) -> list[Claim]:
        claims: list[Claim] = []
        seen: set[str] = set()
        for raw_line in prose.splitlines():
            line = raw_line.strip(" -\t")
            if not line or len(line) < 8:
                continue
            if line in seen:
                continue
            seen.add(line)
            claim_type = self._infer_claim_type(line)
            claims.append(
                Claim(
                    claim_text=line,
                    claim_type=ClaimType(claim_type),
                    verification_status=VerificationStatus.PENDING,
                    source_memo_id=source_memo_id,
                    confidence=0.2,
                )
            )
        return claims

    @staticmethod
    def _infer_claim_type(text: str) -> str:
        if re.search(r"\d|=|<|>|≤|≥|mu\(|μ\(", text):
            return "quantitative"
        lowered = text.casefold()
        if any(
            keyword in lowered
            for keyword in ("if ", " then ", "implies", "for all", "forall", "exists", "lemma", "theorem", "subgoal")
        ):
            return "structural"
        return "heuristic"

