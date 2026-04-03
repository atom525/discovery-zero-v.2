"""
Universal Lean skeleton compiler validation.

This module validates whether an LLM-generated Lean skeleton meaningfully covers
the bridge plan: enough placeholder goals, explicit bridge proposition markers,
and explicit chain-step markers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from dz_engine.bridge import BridgePlan, preferred_sibling_proposition_ids

BRIDGE_PROP_MARKER_RE = re.compile(r"^\s*--\s*BRIDGE-PROP:\s*([A-Za-z0-9_-]+)\s*$", re.MULTILINE)
BRIDGE_STEP_MARKER_RE = re.compile(r"^\s*--\s*BRIDGE-STEP:\s*([A-Za-z0-9_-]+)\s*$", re.MULTILINE)
PLACEHOLDER_RE = re.compile(r"\b(sorry)\b")


class SkeletonCompilerError(RuntimeError):
    """Raised when a compiled Lean skeleton does not satisfy bridge requirements."""


@dataclass
class SkeletonCoverageReport:
    placeholder_count: int
    proposition_markers: set[str]
    step_markers: set[str]
    required_proposition_ids: list[str]
    required_step_ids: list[str]


def extract_skeleton_coverage(lean_code: str, bridge_plan: BridgePlan) -> SkeletonCoverageReport:
    """Extract bridge markers and placeholder count from Lean skeleton code."""
    proposition_markers = set(BRIDGE_PROP_MARKER_RE.findall(lean_code))
    step_markers = set(BRIDGE_STEP_MARKER_RE.findall(lean_code))
    placeholder_count = len(PLACEHOLDER_RE.findall(lean_code))
    required_proposition_ids = [
        item.id
        for item in bridge_plan.propositions
        if item.role in ("bridge", "target")
    ]
    required_step_ids = [item.id for item in bridge_plan.chain]
    return SkeletonCoverageReport(
        placeholder_count=placeholder_count,
        proposition_markers=proposition_markers,
        step_markers=step_markers,
        required_proposition_ids=required_proposition_ids,
        required_step_ids=required_step_ids,
    )


def _minimum_placeholder_count(bridge_plan: BridgePlan) -> int:
    chain_len = len(bridge_plan.chain)
    if chain_len >= 6:
        return 2
    return 1


def compiler_requirements(bridge_plan: BridgePlan) -> dict[str, object]:
    """Return explicit coverage requirements to feed back into the compiler prompt."""
    required_prop_ids = [
        item.id
        for item in bridge_plan.propositions
        if item.role in ("bridge", "target")
    ]
    required_step_ids = [item.id for item in bridge_plan.chain]
    target_prop = next(
        item for item in bridge_plan.propositions
        if item.role == "target" or item.statement == bridge_plan.target_statement
    )
    focus_target_id = next(
        item.id for item in bridge_plan.propositions
        if item.role == "target" or item.statement == bridge_plan.target_statement
    )
    return {
        "required_proposition_ids": required_prop_ids,
        "required_step_ids": required_step_ids,
        "target_proposition_id": focus_target_id,
        "target_statement": target_prop.statement,
        "minimum_placeholder_count": _minimum_placeholder_count(bridge_plan),
        "preferred_sibling_proposition_ids": preferred_sibling_proposition_ids(
            bridge_plan, focus_target_id
        ),
        "target_dependency_ids": target_prop.depends_on,
    }


def validate_compiled_skeleton(lean_code: str, bridge_plan: BridgePlan) -> SkeletonCoverageReport:
    """Strictly validate that a compiled skeleton covers the bridge plan."""
    if "import Mathlib" not in lean_code:
        raise SkeletonCompilerError("Compiled skeleton is missing `import Mathlib`.")
    if "theorem discovery_" not in lean_code:
        raise SkeletonCompilerError("Compiled skeleton is missing a discovery_ theorem declaration.")

    report = extract_skeleton_coverage(lean_code, bridge_plan)
    missing_props = [
        item for item in report.required_proposition_ids
        if item not in report.proposition_markers
    ]
    target_ids = [item.id for item in bridge_plan.propositions if item.role == "target"]
    theorem_mentions_target = "theorem discovery_" in lean_code
    if any(item not in report.proposition_markers for item in target_ids) and not theorem_mentions_target:
        raise SkeletonCompilerError(
            "Compiled skeleton is missing the target proposition marker."
        )
    if report.required_proposition_ids:
        covered = len(report.required_proposition_ids) - len(missing_props)
        coverage_ratio = covered / len(report.required_proposition_ids)
        if coverage_ratio < 0.6:
            raise SkeletonCompilerError(
                "Compiled skeleton covers too few bridge propositions. "
                f"Coverage={coverage_ratio:.2f}; missing: {', '.join(missing_props)}"
            )

    min_step_markers = min(len(report.required_step_ids), max(1, len(report.required_step_ids) // 2))
    if len(report.step_markers) < min_step_markers:
        raise SkeletonCompilerError(
            "Compiled skeleton does not cover enough bridge steps. "
            f"Expected at least {min_step_markers}, found {len(report.step_markers)}."
        )

    min_placeholders = _minimum_placeholder_count(bridge_plan)
    if report.placeholder_count < min_placeholders:
        raise SkeletonCompilerError(
            "Compiled skeleton exposes too few placeholder goals. "
            f"Expected at least {min_placeholders}, found {report.placeholder_count}."
        )

    return report
