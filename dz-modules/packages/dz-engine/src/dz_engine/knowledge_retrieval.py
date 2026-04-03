"""LLM-assisted retrieval of node-relevant known facts."""

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
class KnowledgeFact:
    statement: str
    relation: str
    source: str


class KnowledgeRetriever:
    def retrieve_for_node(
        self,
        *,
        node_statement: str,
        graph: HyperGraph,
        model: Optional[str],
    ) -> list[KnowledgeFact]:
        local_context = "\n".join(f"- {node.statement}" for _, node in list(graph.nodes.items())[:15])
        raw, parsed = run_skill(
            "knowledge_retrieval.skill.md",
            (
                f"Target statement:\n{node_statement}\n\n"
                f"Known local graph statements:\n{local_context}\n\n"
                "Retrieve 5-10 relevant known facts/theorems with relation type."
            ),
            model=model,
        )
        facts_raw = parsed.get("facts", []) if isinstance(parsed, dict) else []
        facts: list[KnowledgeFact] = []
        if isinstance(facts_raw, list):
            for item in facts_raw:
                if isinstance(item, dict):
                    facts.append(
                        KnowledgeFact(
                            statement=str(item.get("statement", "")),
                            relation=str(item.get("relation", "related")),
                            source=str(item.get("source", "llm_retrieval")),
                        )
                    )
        if not facts:
            facts.append(KnowledgeFact(statement=raw[:300], relation="related", source="llm_retrieval"))
        return facts

    def run(
        self,
        *,
        graph: HyperGraph,
        node_id: str,
        model: Optional[str],
        feedback: str,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        if node_id not in graph.nodes:
            raise OrchestrationError(f"Target node {node_id} not found for retrieval.")
        node = graph.nodes[node_id]
        facts = self.retrieve_for_node(node_statement=node.statement, graph=graph, model=model)
        premises = [{"id": None, "statement": fact.statement} for fact in facts]
        payload = {
            "module": "retrieve",
            "domain": node.domain,
            "premises": premises,
            "steps": [
                "Retrieved known relevant facts for current node.",
                *[f"- ({fact.relation}) {fact.statement} [{fact.source}]" for fact in facts],
                f"Feedback context: {feedback.strip()}",
            ],
            "conclusion": {"statement": node.statement, "formal_statement": node.formal_statement},
        }
        normalized = normalize_skill_output(graph, payload, expected_module="retrieve", default_domain=node.domain)
        judge = run_judge(normalized, model=model)
        normalized = apply_judge_confidence(normalized, judge)
        return "\n".join(payload["steps"]), normalized, judge
