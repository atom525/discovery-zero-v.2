"""Research memo and claim models for verification-driven discovery."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ClaimType(str, Enum):
    QUANTITATIVE = "quantitative"
    STRUCTURAL = "structural"
    HEURISTIC = "heuristic"


class VerificationStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    REFUTED = "refuted"
    INCONCLUSIVE = "inconclusive"


class Claim(BaseModel):
    id: str = Field(default_factory=lambda: f"claim_{uuid.uuid4().hex[:12]}")
    claim_text: str
    claim_type: ClaimType
    verification_status: VerificationStatus = VerificationStatus.PENDING
    source_memo_id: str
    evidence: str = ""
    node_id: Optional[str] = None
    confidence: float = 0.0
    depth: int = 0
    # Set by ClaimPipeline when a bridge plan is provided as context.
    # When non-null, identifies the bridge proposition ID (e.g. "P4") that this
    # claim corresponds to, allowing verification results to be written back to
    # the correct graph node via bridge_node_map without any text matching.
    bridge_proposition_id: Optional[str] = None

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, value))


class VerificationResult(BaseModel):
    claim_id: str
    verdict: Literal["verified", "refuted", "inconclusive"]
    evidence_text: str = ""
    confidence_delta: float = 0.0
    code: str = ""
    lean_error: str = ""
    backend: str = ""
    raw_result: dict[str, Any] = Field(default_factory=dict)

    @field_validator("confidence_delta")
    @classmethod
    def _clamp_delta(cls, value: float) -> float:
        return max(-1.0, min(1.0, value))


class ResearchMemo(BaseModel):
    id: str = Field(default_factory=lambda: f"memo_{uuid.uuid4().hex[:12]}")
    raw_prose: str
    claims: list[Claim] = Field(default_factory=list)
    reasoning_structure: list[str] = Field(default_factory=list)
    source_node_id: Optional[str] = None
    iteration: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

