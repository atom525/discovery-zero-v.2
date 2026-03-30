"""Factor graph representation for BP v2 — strictly follows theory docs.

Theory reference: docs/foundations/theory/belief-propagation.md
                  docs/foundations/theory/reasoning-hypergraph.md

Design decisions mandated by theory:
- Binary variables only: each Claim is x ∈ {0, 1}
- Cromwell's rule: all priors and probabilities clamped to (eps, 1-eps)
- Seven operator types:
  ENTAILMENT, INDUCTION, ABDUCTION, CONTRADICTION, EQUIVALENCE,
  CONJUNCTION, SOFT_IMPLICATION
- String variable IDs throughout (no int-mapping layer)
- For CONTRADICTION and EQUIVALENCE: the relation variable is a full BP
  participant with its own prior, not a read-only gate. This implements the
  theory requirement (bp.md §2.5) that relation nodes receive bidirectional
  messages.
- For reasoning operators: ENTAILMENT uses silence (bp.md §2.6, C4 = "通常沉默"),
  INDUCTION/ABDUCTION use noisy-AND + leak (bp.md §2.1, C4 ✓). See potentials.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Sequence

logger = logging.getLogger(__name__)

# Cromwell's rule: never assign P=0 or P=1 to any empirical proposition.
# bp.md §4 enforces this at construction AND inside potentials (leak=eps).
CROMWELL_EPS: float = 1e-3


def _cromwell_clamp(value: float, label: str = "") -> float:
    """Clamp a probability to (eps, 1-eps) per Cromwell's rule.

    Logs a debug message if clamping occurs so the caller can detect
    when inputs violate the rule.
    """
    clamped = max(CROMWELL_EPS, min(1.0 - CROMWELL_EPS, value))
    if clamped != value and label:
        logger.debug("Cromwell clamp: %s %.6g -> %.6g", label, value, clamped)
    return clamped


class FactorType(Enum):
    """Operator types from reasoning-hypergraph.md §7.3.

    Seven types mapping to potential families (potentials.py):
      - ENTAILMENT: silence model (bp.md §2.6, C4 = "通常沉默")
        Premise false → factor silent (potential = 1.0).
      - INDUCTION / ABDUCTION: noisy-AND + leak (bp.md §2.1, C4 ✓)
        Premise false → conclusion suppressed (potential = eps).
      - CONTRADICTION / EQUIVALENCE: fixed-eps constraints (bp.md §2.5)
        Constraint strength fixed by Cromwell, not a free parameter.
      - CONJUNCTION: deterministic factor enforcing M = (A1 ∧ ... ∧ Ak)
      - SOFT_IMPLICATION: two-parameter binary factor (p1, p2)
    """

    ENTAILMENT = auto()  # Deterministic implication, p ≈ 1
    INDUCTION = auto()  # Pattern → general law, p < 1
    ABDUCTION = auto()  # Observation → causal hypothesis, p < 1
    CONTRADICTION = auto()  # Mutual exclusion constraint
    EQUIVALENCE = auto()  # True-value consistency constraint
    CONJUNCTION = auto()  # Deterministic conjunction M=(A1∧...∧Ak)
    SOFT_IMPLICATION = auto()  # Binary soft implication with (p1, p2)


@dataclass
class Factor:
    """A single factor node in the bipartite factor graph.

    For ENTAILMENT / INDUCTION / ABDUCTION:
      - premises: list of variable IDs that are joint necessary conditions
      - conclusions: list of variable IDs that are jointly supported
      - p: P(all conclusions=1 | all premises=1), the author-supplied strength
      - relation_var: None

    For CONTRADICTION:
      - premises: the constrained claim variable IDs
      - conclusions: []
      - p: unused (potential uses CROMWELL_EPS directly)
      - relation_var: ID of the relation variable (full BP participant).
        The relation variable is included in the factor's participant set;
        it is also a regular variable node in FactorGraph.variables.

    For EQUIVALENCE:
      - premises: exactly two claim variable IDs [A, B]
      - conclusions: []
      - p: unused (potential uses CROMWELL_EPS directly)
      - relation_var: ID of the relation variable (full BP participant).
    """

    factor_id: str
    factor_type: FactorType
    premises: list[str]
    conclusions: list[str]
    p: float  # p1 for SOFT_IMPLICATION; ignored by pure deterministic factors
    p2: float | None = None  # second parameter for SOFT_IMPLICATION
    relation_var: str | None = None  # For CONTRADICTION / EQUIVALENCE

    @property
    def all_vars(self) -> list[str]:
        """All variable IDs involved in this factor, including relation_var."""
        base = self.premises + self.conclusions
        if self.relation_var is not None:
            base = [self.relation_var] + base
        return base


class FactorGraph:
    """Bipartite factor graph for belief propagation.

    Implements the structure from reasoning-hypergraph.md §5:
      - Variable nodes: binary propositions with a prior belief π ∈ (eps, 1-eps)
      - Factor nodes: reasoning operators encoding the joint constraint

    All five operator types from §7.3 are supported. The graph never deletes
    variables or factors once added (immutability per reasoning-hypergraph.md §8).

    String IDs are used throughout (no int-to-str mapping needed).

    Usage example:
        fg = FactorGraph()
        fg.add_variable("A", prior=0.5)
        fg.add_variable("B", prior=0.5)
        fg.add_factor("f1", FactorType.ENTAILMENT, premises=["A"],
                       conclusions=["B"], p=0.999)
    """

    def __init__(self) -> None:
        # variable_id -> Cromwell-clamped prior π(x=1)
        self.variables: dict[str, float] = {}
        # ordered list of Factor objects
        self.factors: list[Factor] = []

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def add_variable(self, var_id: str, prior: float) -> None:
        """Register a variable node with its prior belief π(x=1).

        Cromwell's rule is enforced: prior is clamped to (eps, 1-eps).
        Adding the same variable twice updates its prior (last write wins).
        """
        clamped = _cromwell_clamp(prior, label=f"variable '{var_id}' prior")
        self.variables[var_id] = clamped

    def add_factor(
        self,
        factor_id: str,
        factor_type: FactorType,
        premises: Sequence[str],
        conclusions: Sequence[str],
        p: float,
        p2: float | None = None,
        relation_var: str | None = None,
    ) -> None:
        """Add a factor node to the graph.

        Parameters
        ----------
        factor_id:
            Unique string identifier for this factor.
        factor_type:
            One of the five FactorType values.
        premises:
            Variable IDs that are the joint antecedents.
        conclusions:
            Variable IDs that are jointly supported (empty for constraints).
        p:
            For reasoning operators: P(all conclusions=1 | all premises=1).
            Cromwell-clamped. Unused for CONTRADICTION/EQUIVALENCE (the
            potential uses CROMWELL_EPS directly, making the constraint
            strength a system constant, not a free parameter — per bp.md §2.5).
        relation_var:
            For CONTRADICTION/EQUIVALENCE: the relation variable ID. Must
            already be registered via add_variable(), or will raise KeyError
            when BP runs. The relation_var is a full participant receiving
            and sending messages bidirectionally.
        """
        clamped_p = _cromwell_clamp(p, label=f"factor '{factor_id}' p")
        clamped_p2 = (
            _cromwell_clamp(p2, label=f"factor '{factor_id}' p2") if p2 is not None else None
        )

        # Validate that constraint factors have the right structure
        if factor_type in (FactorType.CONTRADICTION, FactorType.EQUIVALENCE):
            if relation_var is None:
                raise ValueError(
                    f"Factor '{factor_id}' of type {factor_type.name} requires a relation_var."
                )
            if conclusions:
                raise ValueError(
                    f"Factor '{factor_id}' of type {factor_type.name} "
                    "must have empty conclusions list."
                )
            if factor_type == FactorType.EQUIVALENCE and len(premises) != 2:
                raise ValueError(
                    f"EQUIVALENCE factor '{factor_id}' requires exactly "
                    f"2 premises, got {len(premises)}."
                )
        elif factor_type == FactorType.CONJUNCTION:
            if len(premises) < 1:
                raise ValueError(
                    f"CONJUNCTION factor '{factor_id}' requires at least 1 premise."
                )
            if len(conclusions) != 1:
                raise ValueError(
                    f"CONJUNCTION factor '{factor_id}' requires exactly 1 conclusion, "
                    f"got {len(conclusions)}."
                )
            if relation_var is not None:
                raise ValueError(
                    f"CONJUNCTION factor '{factor_id}' must not set relation_var."
                )
        elif factor_type == FactorType.SOFT_IMPLICATION:
            if len(premises) != 1 or len(conclusions) != 1:
                raise ValueError(
                    f"SOFT_IMPLICATION factor '{factor_id}' requires exactly 1 premise "
                    "and 1 conclusion."
                )
            if clamped_p2 is None:
                raise ValueError(
                    f"SOFT_IMPLICATION factor '{factor_id}' requires p2."
                )
            if clamped_p + clamped_p2 <= 1.0:
                raise ValueError(
                    f"SOFT_IMPLICATION factor '{factor_id}' requires p1+p2>1, "
                    f"got p1={clamped_p:.6g}, p2={clamped_p2:.6g}."
                )
            if relation_var is not None:
                raise ValueError(
                    f"SOFT_IMPLICATION factor '{factor_id}' must not set relation_var."
                )

        factor = Factor(
            factor_id=factor_id,
            factor_type=factor_type,
            premises=list(premises),
            conclusions=list(conclusions),
            p=clamped_p,
            p2=clamped_p2,
            relation_var=relation_var,
        )
        self.factors.append(factor)

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------

    def get_var_to_factors(self) -> dict[str, list[int]]:
        """Build reverse index: variable_id -> list of factor indices.

        Every variable that appears in a factor's all_vars list is mapped
        to that factor's index. Used by the BP engine to implement the
        exclude-self rule efficiently.
        """
        index: dict[str, list[int]] = {vid: [] for vid in self.variables}
        for fi, factor in enumerate(self.factors):
            for vid in factor.all_vars:
                if vid in index:
                    index[vid].append(fi)
                else:
                    logger.warning(
                        "Factor '%s' references undeclared variable '%s'.",
                        factor.factor_id,
                        vid,
                    )
        return index

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty if graph is consistent).

        Checks:
        - All variables referenced by factors are registered
        - CONTRADICTION/EQUIVALENCE relation_var is registered
        - No factor has duplicate variable IDs in all_vars
        """
        errors: list[str] = []
        for fi, factor in enumerate(self.factors):
            seen: set[str] = set()
            for vid in factor.all_vars:
                if vid not in self.variables:
                    errors.append(
                        f"Factor[{fi}] '{factor.factor_id}': variable '{vid}' not registered."
                    )
                if vid in seen:
                    errors.append(
                        f"Factor[{fi}] '{factor.factor_id}': "
                        f"variable '{vid}' appears more than once in all_vars."
                    )
                seen.add(vid)
        return errors

    def summary(self) -> str:
        """Human-readable summary of the graph for debugging."""
        lines = [f"FactorGraph: {len(self.variables)} variables, {len(self.factors)} factors"]
        lines.append("Variables:")
        for vid, prior in sorted(self.variables.items()):
            lines.append(f"  {vid:30s}  prior={prior:.4f}")
        lines.append("Factors:")
        for factor in self.factors:
            rel = f"  relation={factor.relation_var}" if factor.relation_var else ""
            p2 = f"  p2={factor.p2:.4f}" if factor.p2 is not None else ""
            lines.append(
                f"  [{factor.factor_type.name:15s}] {factor.factor_id}"
                f"  premises={factor.premises}"
                f"  conclusions={factor.conclusions}"
                f"  p={factor.p:.4f}{p2}{rel}"
            )
        return "\n".join(lines)
