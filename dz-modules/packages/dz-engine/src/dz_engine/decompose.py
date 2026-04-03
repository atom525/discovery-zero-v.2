"""Problem decomposition engine with informal + Lean-formal modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from dz_hypergraph.models import HyperGraph
from dz_engine.orchestrator import (
    OrchestrationError,
    apply_judge_confidence,
    normalize_skill_output,
    run_experiment_action,
    run_judge,
    run_lean_action,
    run_lean_decompose_action,
)
from dz_hypergraph.tools.llm import run_skill


@dataclass
class SubProblem:
    statement: str
    rationale: str
    formal_statement: Optional[str] = None


@dataclass
class FormalSubGoal:
    statement: str
    formal_statement: str
    context: str


class DecomposeEngine:
    def decompose_informal(
        self,
        *,
        problem: str,
        graph: HyperGraph,
        model: Optional[str],
    ) -> list[SubProblem]:
        raw, parsed = run_skill(
            "decompose_problem.skill.md",
            (
                f"Problem:\n{problem}\n\n"
                "Produce a decomposition into independent subproblems with rationale."
            ),
            model=model,
        )
        items = parsed.get("subproblems", []) if isinstance(parsed, dict) else []
        out: list[SubProblem] = []
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    out.append(
                        SubProblem(
                            statement=str(item.get("statement", "")),
                            rationale=str(item.get("rationale", "")),
                            formal_statement=item.get("formal_statement"),
                        )
                    )
        if not out:
            out.append(SubProblem(statement=problem, rationale=raw[:240]))
        return out

    def decompose_formal(
        self,
        *,
        graph: HyperGraph,
        node_id: str,
        model: Optional[str],
    ) -> list[FormalSubGoal]:
        raw, normalized, subgoals = run_lean_decompose_action(
            graph,
            node_id,
            model=model,
            timeout=180,
        )
        formal_subgoals: list[FormalSubGoal] = []
        for item in subgoals:
            formal_statement = str(item.get("formal_statement", "")).strip()
            if not formal_statement:
                continue
            formal_subgoals.append(
                FormalSubGoal(
                    statement=str(item.get("statement", "")),
                    formal_statement=formal_statement,
                    context=str(item.get("context", "")),
                )
            )
        return formal_subgoals

    def attack_subproblem(
        self,
        *,
        graph: HyperGraph,
        node_id: str,
        model: Optional[str],
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        raw_exp, normalized_exp, judge_exp = run_experiment_action(graph, node_id, model=model, timeout=45)
        confidence = float(judge_exp.get("confidence", 0.0))
        if confidence >= 0.85:
            return raw_exp, normalized_exp, judge_exp
        raw_lean, normalized_lean, judge_lean = run_lean_action(graph, node_id, model=model, timeout=120)
        if float(judge_lean.get("confidence", 0.0)) >= confidence:
            return raw_lean, normalized_lean, judge_lean
        return raw_exp, normalized_exp, judge_exp

    def run(
        self,
        *,
        graph: HyperGraph,
        node_id: str,
        model: Optional[str],
        feedback: str,
        try_formal: bool = True,
    ) -> tuple[str, dict[str, Any], dict[str, Any]]:
        if node_id not in graph.nodes:
            raise OrchestrationError(f"Target node {node_id} not found for decomposition.")
        node = graph.nodes[node_id]
        informal = self.decompose_informal(problem=node.statement, graph=graph, model=model)
        formal: list[FormalSubGoal] = []
        if try_formal:
            try:
                formal = self.decompose_formal(graph=graph, node_id=node_id, model=model)
            except Exception:
                formal = []
        # Subproblems go into premises so ingest_skill_output creates graph nodes
        # (belief=0.5 each), enabling MCTS to select and attack them in future iterations.
        premises: list[dict[str, object]] = [
            {"id": None, "statement": sp.statement}
            for sp in informal
            if sp.statement.strip()
        ]
        if formal:
            for sg in formal:
                if sg.statement.strip():
                    premises.append({"id": None, "statement": sg.statement})
        steps = [
            "Generated informal decomposition subproblems.",
            *[f"- {sp.statement} | rationale: {sp.rationale}" for sp in informal],
        ]
        if formal:
            steps.append("Validated formal Lean decomposition subgoals.")
            steps.extend(f"- {sg.statement} | formal: {sg.formal_statement}" for sg in formal)
        steps.append(f"Feedback context: {feedback.strip()}")
        payload = {
            "module": "decompose",
            "domain": node.domain,
            "premises": premises,
            "steps": steps,
            "conclusion": {"statement": node.statement, "formal_statement": node.formal_statement},
        }
        normalized = normalize_skill_output(graph, payload, expected_module="decompose", default_domain=node.domain)
        judge = run_judge(normalized, model=model)
        normalized = apply_judge_confidence(normalized, judge)
        return "\n".join(steps), normalized, judge
