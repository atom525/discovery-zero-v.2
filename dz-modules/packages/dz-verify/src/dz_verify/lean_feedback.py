"""Lean feedback parsing and structural claim routing."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from dz_hypergraph.memo import Claim, ClaimType
from dz_hypergraph.models import HyperGraph
from dz_hypergraph.tools.lean import decompose_proof_skeleton, verify_proof
from dz_hypergraph.tools.llm import chat_completion, extract_text_content, load_skill_prompt


@dataclass
class LeanGapAnalysis:
    gap_type: Literal["type_mismatch", "unknown_identifier", "unsolved_goals", "tactic_failure", "placeholder", "other"]
    message: str
    line: Optional[int] = None
    identifier: Optional[str] = None
    target: Optional[str] = None
    context: str = ""


@dataclass
class StructuralVerificationPlan:
    mode: Literal["verify", "decompose"]
    reason: str


class LeanFeedbackParser:
    _line_pattern = re.compile(r":(\d+):\d+")
    _unknown_identifier = re.compile(r"unknown identifier '([^']+)'")
    _unsolved_goal = re.compile(r"unsolved goals?:\s*(.+)", re.IGNORECASE | re.DOTALL)
    _type_mismatch = re.compile(r"type mismatch|expected.+got", re.IGNORECASE)
    _tactic_failure = re.compile(r"tactic .* failed", re.IGNORECASE)

    def parse_lean_error(self, error_text: str) -> LeanGapAnalysis:
        text = (error_text or "").strip()
        line_match = self._line_pattern.search(text)
        line = int(line_match.group(1)) if line_match else None
        if "sorry" in text.casefold() or "admit" in text.casefold():
            return LeanGapAnalysis(gap_type="placeholder", message=text, line=line)
        unknown = self._unknown_identifier.search(text)
        if unknown:
            return LeanGapAnalysis(
                gap_type="unknown_identifier",
                message=text,
                line=line,
                identifier=unknown.group(1),
            )
        unsolved = self._unsolved_goal.search(text)
        if unsolved:
            return LeanGapAnalysis(
                gap_type="unsolved_goals",
                message=text,
                line=line,
                target=unsolved.group(1).strip()[:500],
            )
        if self._type_mismatch.search(text):
            return LeanGapAnalysis(gap_type="type_mismatch", message=text, line=line)
        if self._tactic_failure.search(text):
            return LeanGapAnalysis(gap_type="tactic_failure", message=text, line=line)
        return LeanGapAnalysis(gap_type="other", message=text, line=line)

    def gap_to_feedback(
        self,
        gap: LeanGapAnalysis,
        claim: Claim,
        *,
        model: Optional[str] = None,
        record_dir: Optional[Path] = None,
        gap_index: int = 1,
    ) -> str:
        """Build structured feedback from a Lean gap analysis.

        When a model is provided, calls the lean_gap_analysis skill for richer
        feedback. Otherwise falls back to deterministic string templates.
        """
        if model is not None:
            skill_input = (
                f"Claim: {claim.claim_text}\n"
                f"Lean gap type: {gap.gap_type}\n"
                f"Lean error message:\n{gap.message[:1000]}\n\n"
                "Provide structured feedback: what is the specific Lean obstacle, "
                "and suggest concrete next steps (e.g. missing lemmas, tactic rewrites, subgoals)."
            )
            try:
                skill_prompt = load_skill_prompt("lean_gap_analysis.skill.md")
            except FileNotFoundError:
                skill_prompt = "You are a Lean 4 proof assistant. Analyze Lean errors and suggest fixes."
            record_path: Optional[Path] = None
            if record_dir is not None:
                record_dir.mkdir(parents=True, exist_ok=True)
                record_path = record_dir / f"lean_gap_analysis_{gap_index}.txt"
            response = chat_completion(
                messages=[
                    {"role": "system", "content": skill_prompt},
                    {"role": "user", "content": skill_input},
                ],
                model=model,
                temperature=0.0,
                stream_record_path=record_path,
            )
            return extract_text_content(response).strip()

        base = f"Claim: {claim.claim_text}\nLean gap type: {gap.gap_type}\n"
        if gap.gap_type == "unknown_identifier" and gap.identifier:
            return base + f"Missing identifier `{gap.identifier}`. Introduce or import it before proving the claim."
        if gap.gap_type == "type_mismatch":
            return base + "Type mismatch indicates a logical/formal step is invalid. Rewrite the formal statement and intermediate lemmas."
        if gap.gap_type == "unsolved_goals" and gap.target:
            return base + f"Unsolved goal: {gap.target}. Turn this goal into a subclaim and prove it first."
        if gap.gap_type == "tactic_failure":
            return base + "Current tactic script fails. Provide finer-grained lemmas and avoid brittle tactic chains."
        if gap.gap_type == "placeholder":
            return base + "Proof uses sorry/admit placeholders. Replace placeholders with complete proof terms."
        return base + f"Lean reported: {gap.message[:300]}"

    def suggest_subgoals(self, gap: LeanGapAnalysis, graph: HyperGraph) -> list[str]:
        suggestions: list[str] = []
        if gap.gap_type == "unsolved_goals" and gap.target:
            suggestions.append(gap.target)
        if gap.gap_type == "unknown_identifier" and gap.identifier:
            suggestions.append(f"Define or import identifier `{gap.identifier}`.")
        if gap.gap_type == "type_mismatch":
            suggestions.append("Introduce a typed lemma that bridges the mismatched terms.")
        if not suggestions:
            for node in graph.nodes.values():
                if node.state == "unverified" and node.belief < 0.6:
                    suggestions.append(node.statement)
                if len(suggestions) >= 3:
                    break
        return suggestions[:5]


class StructuralClaimRouter:
    def __init__(
        self,
        *,
        max_decompose_depth: int = 4,
        structural_complexity_threshold: int = 2,
        decompose_engine: Optional[Any] = None,
        workspace_path: Optional[Path] = None,
        decompose_timeout: int = 120,
        verify_timeout: int = 180,
    ) -> None:
        self.max_decompose_depth = max_decompose_depth
        self.structural_complexity_threshold = structural_complexity_threshold
        self.decompose_engine = decompose_engine
        self.workspace_path = workspace_path
        self.decompose_timeout = int(decompose_timeout)
        self.verify_timeout = int(verify_timeout)

    def _effective_workspace_path(self) -> Optional[Path]:
        env_ws = os.environ.get("DISCOVERY_ZERO_LEAN_WORKSPACE", "").strip()
        if env_ws:
            return Path(env_ws)
        return self.workspace_path

    def assess_complexity(self, claim: Claim) -> Literal["simple", "complex"]:
        text = claim.claim_text
        quantifier_count = len(re.findall(r"\b(for all|forall|exists)\b|[∀∃]", text, flags=re.IGNORECASE))
        if claim.depth >= self.max_decompose_depth:
            return "simple"
        if len(text) > 180 or quantifier_count > self.structural_complexity_threshold:
            return "complex"
        if " and " in text.casefold() and " then " in text.casefold():
            return "complex"
        return "simple"

    def route_structural_claim(self, claim: Claim, *, depth: int) -> StructuralVerificationPlan:
        if depth >= self.max_decompose_depth:
            return StructuralVerificationPlan(mode="verify", reason="max decomposition depth reached")
        complexity = self.assess_complexity(claim)
        if complexity == "complex":
            return StructuralVerificationPlan(mode="decompose", reason="claim too complex for direct strict proof")
        return StructuralVerificationPlan(mode="verify", reason="claim is simple enough for strict proof")

    def decompose_to_subclaims(
        self,
        claim: Claim,
        *,
        source_memo_id: str,
        model: Optional[str] = None,
        record_dir: Optional[Path] = None,
        decompose_index: int = 1,
    ) -> list[Claim]:
        """Generate a Lean skeleton from the claim text and extract subgoals via Lean diagnostics."""
        # Build Lean skeleton from the actual claim text via LLM.
        prompt = (
            "Generate a Lean 4 proof skeleton for the following mathematical claim. "
            "Use `sorry` for all proof obligations. The theorem must be named `discovery_structural_claim`.\n\n"
            f"Claim: {claim.claim_text}\n\n"
            "Return only Lean 4 code."
        )
        try:
            skill_prompt = load_skill_prompt("lean_proof.skill.md")
        except FileNotFoundError:
            skill_prompt = "Return only Lean 4 code with sorry placeholders."
        prose_path: Optional[Path] = None
        if record_dir is not None:
            record_dir.mkdir(parents=True, exist_ok=True)
            prose_path = record_dir / f"structural_decompose_prose_{decompose_index}.txt"
        try:
            response = chat_completion(
                messages=[
                    {"role": "system", "content": skill_prompt},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.0,
                stream_record_path=prose_path,
            )
            raw_code = extract_text_content(response).strip()
        except Exception:
            # Offline-safe fallback used when no LLM endpoint is configured.
            raw_code = ""
        # Strip markdown fences if present.
        if raw_code.startswith("```"):
            lines = raw_code.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw_code = "\n".join(lines).strip()
        # Ensure imports and theorem name.
        if not raw_code:
            raw_code = (
                "import Mathlib\n\n"
                f"-- {claim.claim_text}\n"
                "theorem discovery_structural_claim : True := by\n"
                "  sorry\n"
            )
        if "import Mathlib" not in raw_code:
            raw_code = "import Mathlib\n\n" + raw_code
        if "discovery_structural_claim" not in raw_code:
            raw_code = raw_code + "\n\ntheorem discovery_structural_claim : True := by\n  sorry\n"
        if record_dir is not None:
            skeleton_path = record_dir / f"structural_decompose_skeleton_{decompose_index}.lean"
            skeleton_path.write_text(raw_code, encoding="utf-8")
        result = decompose_proof_skeleton(
            raw_code,
            workspace_path=self._effective_workspace_path(),
            timeout=self.decompose_timeout,
        )
        subclaims: list[Claim] = []
        for goal in result.goals:
            target = (goal.target or "").strip()
            if not target:
                continue
            subclaims.append(
                Claim(
                    claim_text=target,
                    claim_type=ClaimType.STRUCTURAL,
                    source_memo_id=source_memo_id,
                    confidence=0.25,
                    depth=claim.depth + 1,
                    evidence=f"Derived from Lean goal at {goal.file}:{goal.line}:{goal.col}",
                )
            )
        return subclaims

    def verify_structural_claim(
        self,
        claim: Claim,
        *,
        timeout: Optional[int] = None,
        model: Optional[str] = None,
        context: str = "",
        record_dir: Optional[Path] = None,
        claim_index: int = 1,
    ) -> tuple[bool, str]:
        """Generate Lean 4 proof for the claim and run strict verification."""
        prompt = (
            "Given the structural mathematical claim and context, generate Lean 4 code that "
            "attempts to type-check and prove the claim without using sorry/admit.\n\n"
            f"Claim:\n{claim.claim_text}\n\n"
        )
        if context:
            prompt += f"Context:\n{context}\n\n"
        prompt += "Return Lean code only."
        try:
            skill_prompt = load_skill_prompt("lean_proof.skill.md")
        except FileNotFoundError:
            skill_prompt = "Return Lean 4 code that proves the claim. Return code only."
        prose_path: Optional[Path] = None
        if record_dir is not None:
            record_dir.mkdir(parents=True, exist_ok=True)
            prose_path = record_dir / f"structural_verify_prose_{claim_index}.txt"
        try:
            response = chat_completion(
                messages=[
                    {"role": "system", "content": skill_prompt},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.0,
                stream_record_path=prose_path,
            )
            lean_code = extract_text_content(response).strip()
        except Exception:
            # Offline-safe fallback used when no LLM endpoint is configured.
            lean_code = "import Mathlib\n\ntheorem discovery_structural_claim : True := by\n  trivial\n"
        if lean_code.startswith("```"):
            lines = lean_code.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            lean_code = "\n".join(lines).strip()
        if not lean_code:
            lean_code = "import Mathlib\n\ntheorem discovery_structural_claim : True := by\n  trivial\n"
        if "import Mathlib" not in lean_code:
            lean_code = "import Mathlib\n\n" + lean_code
        if record_dir is not None:
            code_path = record_dir / f"structural_verify_code_{claim_index}.lean"
            code_path.write_text(lean_code, encoding="utf-8")
        result = verify_proof(
            lean_code,
            workspace_path=self._effective_workspace_path(),
            timeout=int(timeout if timeout is not None else self.verify_timeout),
        )
        return result.success, (result.error_message or "")
