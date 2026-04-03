"""
Continuation-sampling verification for bridge / plausible reasoning.

The verifier samples multiple independent continuations for a reasoning step,
clusters them coarsely by lexical overlap, and uses the dominant cluster mass as
a cheap consistency proxy. This is designed as a practical fallback when no
specialized PRM is available.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from gaia.bp.factor_graph import CROMWELL_EPS

from dz_hypergraph.bridge_models import BridgePlan
from dz_hypergraph.tools.llm import chat_completion, extract_text_content


_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass
class ContinuationConfig:
    num_continuations: int = 4
    consistency_threshold: float = 0.6
    temperature: float = 0.6
    max_tokens: int = 256
    enable_code_check: bool = True


@dataclass
class StepVerificationResult:
    step_id: str
    statement: str
    continuations: list[str] = field(default_factory=list)
    cluster_sizes: list[int] = field(default_factory=list)
    consistency_score: float = 0.0
    code_check_passed: bool = False
    accepted: bool = False


class ContinuationVerifier:
    def __init__(self, config: Optional[ContinuationConfig] = None) -> None:
        self.config = config or ContinuationConfig()

    def verify_bridge_plan(
        self,
        *,
        target_statement: str,
        bridge_plan: BridgePlan,
        model: Optional[str] = None,
        retrieval_context: str = "",
    ) -> list[StepVerificationResult]:
        results: list[StepVerificationResult] = []
        for step in bridge_plan.chain:
            results.append(
                self.verify_step(
                    target_statement=target_statement,
                    step_id=step.id,
                    step_statement=step.statement,
                    supporting_statements=[
                        prop.statement
                        for prop in bridge_plan.propositions
                        if prop.id in step.uses
                    ],
                    model=model,
                    retrieval_context=retrieval_context,
                )
            )
        return results

    def verify_step(
        self,
        *,
        target_statement: str,
        step_id: str,
        step_statement: str,
        supporting_statements: list[str],
        model: Optional[str] = None,
        retrieval_context: str = "",
    ) -> StepVerificationResult:
        prompt_lines = [
            "Assess the plausibility of the following mathematical reasoning step.",
            f"Target: {target_statement}",
            f"Step: {step_statement}",
        ]
        if supporting_statements:
            prompt_lines.append("Supporting facts:")
            prompt_lines.extend(f"- {item}" for item in supporting_statements[:8])
        if retrieval_context:
            prompt_lines.extend(["", "Retrieved context:", retrieval_context])
        prompt_lines.extend(
            [
                "",
                "Write a short independent continuation that either supports or questions the step.",
                "Be concrete and mathematical.",
            ]
        )
        try:
            responses = chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a mathematical verifier. Respond with concise analysis only.",
                    },
                    {"role": "user", "content": "\n".join(prompt_lines)},
                ],
                model=model,
                n=self.config.num_continuations,
                temperature=self.config.temperature,
                stream=False,
            )
        except Exception:
            responses = []

        if isinstance(responses, dict):
            raw_continuations = [extract_text_content(responses)]
        else:
            raw_continuations = [extract_text_content(item) for item in responses]
        continuations = [text.strip() for text in raw_continuations if text and text.strip()]
        cluster_sizes = self._cluster_sizes(continuations)
        dominant = max(cluster_sizes) if cluster_sizes else 0
        consistency = dominant / max(len(continuations), 1)
        code_ok = self._code_check(step_statement) if self.config.enable_code_check else False
        accepted = consistency >= self.config.consistency_threshold or code_ok
        return StepVerificationResult(
            step_id=step_id,
            statement=step_statement,
            continuations=continuations,
            cluster_sizes=cluster_sizes,
            consistency_score=round(consistency, 6),
            code_check_passed=code_ok,
            accepted=accepted,
        )

    def calibrated_belief(self, *, prior: float, consistency: float) -> float:
        prior = max(CROMWELL_EPS, min(1.0 - CROMWELL_EPS, float(prior)))
        consistency = max(0.0, min(1.0, float(consistency)))
        logit = math.log(prior / (1.0 - prior))
        adjusted = logit + 2.0 * (consistency - 0.5)
        return 1.0 / (1.0 + math.exp(-adjusted))

    def _cluster_sizes(self, texts: list[str]) -> list[int]:
        clusters: list[set[str]] = []
        sizes: list[int] = []
        for text in texts:
            tokens = set(_TOKEN_RE.findall(text.lower()))
            if not tokens:
                continue
            assigned = False
            for idx, cluster in enumerate(clusters):
                overlap = len(tokens & cluster) / max(len(tokens | cluster), 1)
                if overlap >= 0.35:
                    clusters[idx] = cluster | tokens
                    sizes[idx] += 1
                    assigned = True
                    break
            if not assigned:
                clusters.append(tokens)
                sizes.append(1)
        sizes.sort(reverse=True)
        return sizes

    def _code_check(self, step_statement: str) -> bool:
        fenced = re.findall(r"```(?:python)?\n(.*?)```", step_statement, re.DOTALL)
        snippets = fenced or ([step_statement] if "def " in step_statement or "=" in step_statement else [])
        for snippet in snippets:
            try:
                ast.parse(snippet)
                return True
            except SyntaxError:
                continue
        return False
