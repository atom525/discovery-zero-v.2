"""
External process-reward bridge for cold-start value estimation.

The interface is intentionally PAV-compatible: when a self-trained value model
is unavailable or still weak, callers can ask this module for a coarse estimate
of whether a proposed reasoning step or action looks promising.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from dz_hypergraph.models import HyperGraph, Module
from dz_hypergraph.tools.llm import chat_completion, extract_text_content


@dataclass
class ExternalPRMConfig:
    provider: str = "chat_completion"
    api_base: str = ""
    api_key: str = ""
    model: str = ""
    timeout_seconds: float = 30.0
    fallback_score: float = 0.0


class ExternalPRM:
    def __init__(
        self,
        config: ExternalPRMConfig,
        *,
        fallback_verifier: Any = None,
    ) -> None:
        self.config = config
        self._fallback_verifier = fallback_verifier

    @property
    def enabled(self) -> bool:
        return bool(self.config.model or self.config.api_base or self.config.provider == "chat_completion")

    def score_step(
        self,
        *,
        problem: str,
        step: str,
        context: str = "",
    ) -> float:
        if not self.enabled:
            return self._fallback(problem=problem, step=step, context=context)
        prompt = (
            "You are a process reward model for mathematical reasoning.\n"
            "Score the candidate step on a continuous scale from 0.0 to 1.0.\n"
            "Return only the numeric score.\n\n"
            f"Problem:\n{problem}\n\n"
            f"Context:\n{context or '(none)'}\n\n"
            f"Candidate step:\n{step}\n"
        )
        if self.config.provider == "chat_completion":
            try:
                response = chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": "Return only a single numeric score between 0.0 and 1.0.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    model=self.config.model or None,
                    api_base=self.config.api_base or None,
                    api_key=self.config.api_key or None,
                    temperature=0.0,
                    timeout=int(self.config.timeout_seconds),
                    stream=False,
                )
            except Exception:
                return self._fallback(problem=problem, step=step, context=context)
            return self._extract_score({"choices": [{"text": extract_text_content(response)}]})
        body = {
            "model": self.config.model,
            "prompt": prompt,
            "max_tokens": 8,
            "temperature": 0.0,
            "logprobs": 5,
        }
        try:
            payload = self._post_json("/completions", body)
        except Exception:
            return self._fallback(problem=problem, step=step, context=context)
        return self._extract_score(payload)

    def score_solution(
        self,
        *,
        problem: str,
        steps: list[str],
        context: str = "",
    ) -> float:
        if not steps:
            return self.config.fallback_score
        scores = [
            self.score_step(problem=problem, step=step, context=context)
            for step in steps
            if step.strip()
        ]
        if not scores:
            return self.config.fallback_score
        return sum(scores) / len(scores)

    def estimate_value(
        self,
        graph: HyperGraph,
        target_node_id: str,
        candidate_node_id: str,
        candidate_module: Module,
    ) -> float:
        target = graph.nodes.get(target_node_id)
        candidate = graph.nodes.get(candidate_node_id)
        if target is None or candidate is None:
            return self.config.fallback_score
        context_lines = [
            f"Target belief: {target.belief:.3f}",
            f"Candidate belief: {candidate.belief:.3f}",
            f"Candidate module: {candidate_module.value}",
        ]
        if graph.get_edges_to(candidate_node_id):
            context_lines.append("Candidate already has supporting incoming edges.")
        prompt_step = (
            f"Explore node '{candidate.statement}' with module '{candidate_module.value}' "
            f"to improve confidence in target '{target.statement}'."
        )
        score = self.score_step(
            problem=target.statement,
            step=prompt_step,
            context="\n".join(context_lines),
        )
        return max(-1.0, min(1.0, 2.0 * score - 1.0))

    def _fallback(self, *, problem: str, step: str, context: str) -> float:
        verifier = self._fallback_verifier
        if verifier is not None and hasattr(verifier, "calibrated_belief"):
            try:
                return float(verifier.calibrated_belief(prior=0.5, consistency=0.0))
            except Exception:
                pass
        return self.config.fallback_score

    def _post_json(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        api_base = self.config.api_base.rstrip("/")
        if api_base.endswith("/v1"):
            url = api_base + endpoint
        else:
            url = api_base + "/v1" + endpoint
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **(
                    {"Authorization": f"Bearer {self.config.api_key}"}
                    if self.config.api_key
                    else {}
                ),
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _extract_score(self, payload: dict[str, Any]) -> float:
        choices = payload.get("choices", [])
        if not choices:
            return self.config.fallback_score
        choice = choices[0] or {}

        logprobs = choice.get("logprobs") or {}
        top_logprobs = logprobs.get("top_logprobs") or []
        if top_logprobs and isinstance(top_logprobs, list):
            first = top_logprobs[0]
            if isinstance(first, dict):
                numeric_candidates = []
                for token, logp in first.items():
                    try:
                        value = float(token.strip())
                    except Exception:
                        continue
                    numeric_candidates.append((value, float(logp)))
                if numeric_candidates:
                    best_value = max(numeric_candidates, key=lambda item: item[1])[0]
                    return max(0.0, min(1.0, best_value))

        text = str(choice.get("text", "")).strip()
        match = re.search(r"([01](?:\.\d+)?)", text)
        if match:
            return max(0.0, min(1.0, float(match.group(1))))
        return self.config.fallback_score
