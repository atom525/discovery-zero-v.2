"""Analogy-driven reasoning module."""

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
class Analogy:
    source_domain: str
    mapping: str
    transferable_technique: str


@dataclass
class TransferResult:
    route: str
    testability: str


class AnalogyEngine:
    def find_analogies(self, problem: str, graph: HyperGraph, model: Optional[str]) -> list[Analogy]:
        context = "\n".join(
            f"- {node.statement}"
            for _, node in list(graph.nodes.items())[:20]
        )
        prompt = (
            f"Target problem:\n{problem}\n\n"
            f"Current graph context:\n{context}\n\n"
            "Propose three cross-domain analogies and include domain, mapping, and transferable technique."
        )
        raw, parsed = run_skill("analogy_reasoning.skill.md", prompt, model=model)
        if not isinstance(parsed, dict):
            raise OrchestrationError("Analogy skill did not return JSON object.")
        items = parsed.get("analogies", [])
        analogies: list[Analogy] = []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                analogies.append(
                    Analogy(
                        source_domain=str(item.get("source_domain", "unknown")),
                        mapping=str(item.get("mapping", "")),
                        transferable_technique=str(item.get("transferable_technique", "")),
                    )
                )
        if not analogies:
            analogies.append(
                Analogy(
                    source_domain="fallback",
                    mapping="No structured analogies provided; use conservative transfer.",
                    transferable_technique=raw[:400],
                )
            )
        return analogies

    def transfer_technique(self, analogy: Analogy, target: str, model: Optional[str]) -> TransferResult:
        prompt = (
            f"Target:\n{target}\n\n"
            f"Analogy source domain: {analogy.source_domain}\n"
            f"Mapping: {analogy.mapping}\n"
            f"Transferable technique: {analogy.transferable_technique}\n\n"
            "Produce a concrete transferable route and explain how to test it."
        )
        raw, parsed = run_skill("analogy_reasoning.skill.md", prompt, model=model)
        if isinstance(parsed, dict):
            return TransferResult(
                route=str(parsed.get("route", raw[:300])),
                testability=str(parsed.get("testability", "requires further evaluation")),
            )
        return TransferResult(route=raw[:300], testability="requires further evaluation")

    def run(
        self,
        *,
        graph: HyperGraph,
        node_id: str,
        model: Optional[str],
        feedback: str,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        if node_id not in graph.nodes:
            raise OrchestrationError(f"Target node {node_id} not found for analogy engine.")
        node = graph.nodes[node_id]
        analogies = self.find_analogies(node.statement, graph, model)
        transfers = [self.transfer_technique(a, node.statement, model) for a in analogies]
        # Transferable techniques become premise nodes so MCTS can attack them later.
        premises: list[dict[str, object]] = []
        for analogy, transfer in zip(analogies, transfers):
            technique = analogy.transferable_technique.strip()
            if technique:
                premises.append({
                    "id": None,
                    "statement": f"[Analogy from {analogy.source_domain}] {technique}",
                })
            route = transfer.route.strip()
            if route:
                premises.append({
                    "id": None,
                    "statement": f"[Transfer route] {route}",
                })
        # Deduplicate by statement text
        seen_stmts: set[str] = set()
        deduped_premises: list[dict[str, object]] = []
        for p in premises:
            stmt = str(p["statement"])
            if stmt not in seen_stmts:
                seen_stmts.add(stmt)
                deduped_premises.append(p)
        transfer0 = transfers[0]
        payload = {
            "module": "analogy",
            "domain": node.domain,
            "premises": deduped_premises,
            "steps": [
                f"Analogy source domain: {analogies[0].source_domain}",
                f"Mapping: {analogies[0].mapping}",
                f"Transferable mechanism: {analogies[0].transferable_technique}",
                f"Route proposal: {transfer0.route}",
                f"Testability: {transfer0.testability}",
            ],
            "conclusion": {"statement": node.statement, "formal_statement": node.formal_statement},
        }
        normalized = normalize_skill_output(graph, payload, expected_module="analogy", default_domain=node.domain)
        judge = run_judge(normalized, model=model)
        normalized = apply_judge_confidence(normalized, judge)
        raw = "\n".join(str(s) for s in payload["steps"]) + "\n\nFeedback:\n" + feedback.strip()
        return raw, normalized, judge
