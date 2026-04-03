"""Constraint specialization and pattern transfer engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from dz_hypergraph.models import HyperGraph
from dz_engine.orchestrator import (
    OrchestrationError,
    apply_judge_confidence,
    normalize_skill_output,
    run_judge,
)
from dz_hypergraph.tools.llm import run_skill


@dataclass
class Specialization:
    statement: str
    rationale: str


@dataclass
class Pattern:
    statement: str
    support: str


class SpecializeEngine:
    def generate_specializations(self, problem: str, model: Optional[str]) -> list[Specialization]:
        raw, parsed = run_skill(
            "specialize_generalize.skill.md",
            (
                f"Problem:\n{problem}\n\n"
                "Generate 3-5 constrained specializations that are easier to test computationally."
            ),
            model=model,
        )
        items = parsed.get("specializations", []) if isinstance(parsed, dict) else []
        results: list[Specialization] = []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    results.append(
                        Specialization(
                            statement=str(item.get("statement", "")),
                            rationale=str(item.get("rationale", "")),
                        )
                    )
        if not results:
            results.append(Specialization(statement=problem, rationale="fallback specialization"))
        return results

    def mine_patterns(self, solved: list[Specialization], model: Optional[str]) -> list[Pattern]:
        if not solved:
            return []
        payload = "\n".join(f"- {item.statement} ({item.rationale})" for item in solved)
        raw, parsed = run_skill(
            "specialize_generalize.skill.md",
            "From these solved specializations, extract reusable patterns:\n" + payload,
            model=model,
        )
        patterns_raw = parsed.get("patterns", []) if isinstance(parsed, dict) else []
        patterns: list[Pattern] = []
        if isinstance(patterns_raw, list):
            for item in patterns_raw:
                if isinstance(item, dict):
                    patterns.append(
                        Pattern(
                            statement=str(item.get("statement", "")),
                            support=str(item.get("support", "")),
                        )
                    )
        if not patterns:
            patterns.append(Pattern(statement=raw[:300], support="fallback"))
        return patterns

    def run(
        self,
        *,
        graph: HyperGraph,
        node_id: str,
        model: Optional[str],
        feedback: str,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        if node_id not in graph.nodes:
            raise OrchestrationError(f"Target node {node_id} not found for specialization.")
        node = graph.nodes[node_id]
        specs = self.generate_specializations(node.statement, model)
        patterns = self.mine_patterns(specs[:3], model)
        # Specializations and patterns become premise nodes for MCTS to later verify
        # or attack individually. Each gets belief=0.5 on ingest.
        premises: list[dict[str, object]] = []
        for spec in specs:
            if spec.statement.strip():
                premises.append({"id": None, "statement": spec.statement})
        for pat in patterns:
            if pat.statement.strip():
                premises.append({"id": None, "statement": pat.statement})
        payload = {
            "module": "specialize",
            "domain": node.domain,
            "premises": premises,
            "steps": [
                "Generated constrained specializations:",
                *[f"{idx+1}. {spec.statement} | rationale: {spec.rationale}" for idx, spec in enumerate(specs)],
                "Transferred patterns:",
                *[f"- {pat.statement} | support: {pat.support}" for pat in patterns],
                f"Feedback context: {feedback.strip()}",
            ],
            "conclusion": {
                "statement": node.statement,
                "formal_statement": node.formal_statement,
            },
        }
        normalized = normalize_skill_output(graph, payload, expected_module="specialize", default_domain=node.domain)
        judge = run_judge(normalized, model=model)
        normalized = apply_judge_confidence(normalized, judge)
        return "\n".join(payload["steps"]), normalized, judge
