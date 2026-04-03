"""
Generate easier or more concrete variants when the main search stalls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from dz_hypergraph.models import HyperGraph
from dz_hypergraph.tools.llm import run_skill


@dataclass
class ProblemVariant:
    original_node_id: str
    variant_statement: str
    variant_type: str
    difficulty_estimate: float
    rationale: str = ""


class ProblemVariantGenerator:
    def generate_variants(
        self,
        graph: HyperGraph,
        target_node_id: str,
        *,
        model: Optional[str] = None,
        max_variants: int = 3,
    ) -> list[ProblemVariant]:
        target = graph.nodes.get(target_node_id)
        if target is None:
            return []
        task_input = "\n".join(
            [
                "Generate easier specialized variants of the target problem.",
                "Return JSON with key `variants`, a list of objects with keys:",
                "`variant_statement`, `variant_type`, `difficulty_estimate`, `rationale`.",
                "Allowed variant_type values: parameter_reduction, finite_case, weaker_conclusion, stronger_hypothesis, analogy.",
                f"Target statement: {target.statement}",
            ]
        )
        try:
            _raw, parsed = run_skill("problem_variant.skill.md", task_input, model=model)
        except Exception:
            return []
        items = parsed.get("variants", []) if isinstance(parsed, dict) else []
        variants: list[ProblemVariant] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            statement = str(item.get("variant_statement", "")).strip()
            if not statement:
                continue
            variants.append(
                ProblemVariant(
                    original_node_id=target_node_id,
                    variant_statement=statement,
                    variant_type=str(item.get("variant_type", "finite_case")),
                    difficulty_estimate=float(item.get("difficulty_estimate", 0.5)),
                    rationale=str(item.get("rationale", "")),
                )
            )
        variants.sort(key=lambda item: item.difficulty_estimate)
        return variants[:max_variants]

    def materialize_variants(
        self,
        graph: HyperGraph,
        variants: list[ProblemVariant],
        *,
        domain: Optional[str] = None,
    ) -> list[str]:
        created_ids: list[str] = []
        for item in variants:
            existing = graph.find_node_ids_by_statement(item.variant_statement)
            if existing:
                created_ids.append(existing[0])
                continue
            node = graph.add_node(
                statement=item.variant_statement,
                belief=0.35,
                prior=0.35,
                domain=domain,
                provenance=f"variant:{item.variant_type}",
            )
            created_ids.append(node.id)
        return created_ids

    def transfer_insight(
        self,
        *,
        original_statement: str,
        solved_variant_statement: str,
        solved_variant_summary: str,
        model: Optional[str] = None,
    ) -> str:
        task_input = "\n".join(
            [
                "Generalize insight from a solved easier variant back to the original problem.",
                f"Original problem: {original_statement}",
                f"Solved variant: {solved_variant_statement}",
                f"Variant solution summary: {solved_variant_summary}",
                "Return concise natural-language transfer guidance.",
            ]
        )
        try:
            raw, _parsed = run_skill("transfer_insight.skill.md", task_input, model=model)
        except Exception:
            return ""
        return raw.strip()
