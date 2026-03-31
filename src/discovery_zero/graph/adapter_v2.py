"""Zero HyperGraph -> Gaia BP v2 FactorGraph adapter.

Converts Discovery-Zero's HyperGraph into Gaia's FactorGraph for belief
propagation, strictly following Gaia theory documents:

  - 03-propositional-operators.md §4:  Soft Implication ↝(p₁, p₂)
  - 06-factor-graphs.md §3.7:  ↝ is the ONLY parameterized factor
  - 07-belief-propagation.md §3:  when p₂=0.5, premise false → uniform (no info)

Factor mapping from HyperGraph edge semantics:

  Edge                          FactorType             Parameters
  ────────────────────────────  ─────────────────────  ───────────────────────
  formal (Lean-verified)        SOFT_IMPLICATION        p₁≈1, p₂=0.5
  heuristic (plausible/exp/...) SOFT_IMPLICATION        p₁=confidence, p₂=0.5
  decomposition                 (skipped)               structural only
  multi-premise                 CONJUNCTION + ↝         ∧ mediator + ↝(p₁,p₂)
  detected contradiction        CONTRADICTION           deterministic constraint
  detected equivalence          EQUIVALENCE             deterministic constraint

Key design decisions per Gaia theory:
  - ALL reasoning edges use SOFT_IMPLICATION ↝(p₁, p₂).
  - p₂ = 0.5 (MaxEnt default): premise false → factor sends uniform message
    (no information), equivalent to "not expressing an opinion when the
    antecedent is absent" (07-belief-propagation.md §3).
  - p₁ + p₂ > 1 constraint (positive support): edges with confidence ≤ 0.5
    are clamped to 0.5 + eps to satisfy the constraint minimally.
  - Formal (Lean) edges: strict implication → embedded as ↝(1, 0.5)
    (03-propositional-operators.md §4.7).
  - Multi-premise: ∧ (CONJUNCTION) + ↝ (SOFT_IMPLICATION), clean separation
    of "how premises combine" from "how they support conclusion" (theory §5).
  - Decomposition edges are structural (used by MCTS planner), not
    probabilistic. Excluded from BP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations

from gaia_bp.factor_graph import CROMWELL_EPS, FactorGraph, FactorType

from discovery_zero.graph.adapter import _detect_contradictions, _detect_equivalences
from discovery_zero.graph.models import HyperGraph

logger = logging.getLogger(__name__)

# MaxEnt default for p₂: when premise is false, factor provides no information.
# Per 07-belief-propagation.md §3: "当 p₂ = 0.5 时, ψ(0,0) = ψ(0,1) = 0.5,
# 即 A 为假时因子对 B 提供均匀（无信息）权重"
MAXENT_P2: float = 0.5


@dataclass
class ZeroInferenceGraphV2:
    factor_graph: FactorGraph
    synthetic_var_ids: set[str] = field(default_factory=set)


def _compute_p1(edge) -> float:
    """Compute p₁ for a ↝ factor from the edge's confidence and type.

    Per Gaia theory:
      - Formal edges (Lean-verified): p₁ ≈ 1 (strict implication embedded
        as ↝(1, 0.5), 03-propositional-operators.md §4.7)
      - Heuristic edges: p₁ = confidence (author-supplied strength)

    The p₁ + p₂ > 1 constraint (with p₂ = 0.5) requires p₁ > 0.5.
    Edges with confidence ≤ 0.5 are clamped to 0.5 + CROMWELL_EPS.
    """
    if edge.edge_type == "formal":
        # Strict implication: p₁ ≈ 1
        return max(float(edge.confidence), 1.0 - CROMWELL_EPS)

    p1 = float(edge.confidence)

    # Enforce p₁ + p₂ > 1 constraint: with p₂ = 0.5, need p₁ > 0.5
    min_p1 = MAXENT_P2 + CROMWELL_EPS  # 0.501
    if p1 <= MAXENT_P2:
        logger.debug(
            "Edge %s confidence %.4f <= 0.5; clamped to %.4f for p1+p2>1 constraint",
            edge.id, p1, min_p1,
        )
        p1 = min_p1

    return p1


def adapt_zero_graph_v2(
    graph: HyperGraph,
    *,
    warmstart: bool = False,
) -> ZeroInferenceGraphV2:
    """Convert a Zero HyperGraph into Gaia BP v2's FactorGraph.

    Strictly follows Gaia theory: all reasoning edges are mapped to
    SOFT_IMPLICATION ↝(p₁, p₂) with p₂ = 0.5 (MaxEnt default).

    Parameters
    ----------
    graph:
        The Zero HyperGraph to convert.
    warmstart:
        If True, use current beliefs (instead of priors) as BP starting point.
    """
    fg = FactorGraph()
    synthetic_var_ids: set[str] = set()

    # ── Register all nodes as BP variables ──
    for nid, node in graph.nodes.items():
        if warmstart and not node.is_locked():
            prior = max(CROMWELL_EPS, min(1.0 - CROMWELL_EPS, float(node.belief)))
        else:
            prior = float(node.prior)
        if node.state == "proven":
            prior = 1.0 - CROMWELL_EPS
        elif node.state == "refuted":
            prior = CROMWELL_EPS
        fg.add_variable(nid, prior)

    # ── Convert edges to factors ──
    for eid, edge in graph.edges.items():
        premises = [pid for pid in edge.premise_ids if pid in fg.variables]
        conclusion = edge.conclusion_id
        if conclusion not in fg.variables:
            continue

        # Decomposition edges are structural (subgoal breakdown), not
        # probabilistic evidence.  Including them would force
        # conclusion_belief ≈ ∏(subgoal_beliefs), crushing beliefs when
        # subgoals are unverified.  Skip from BP; decomposition structure
        # is used by the MCTS planner.
        if edge.edge_type == "decomposition":
            continue

        if not premises:
            continue

        # Compute p₁ from edge confidence/type
        p1 = _compute_p1(edge)

        # Single-premise edge → direct SOFT_IMPLICATION ↝(p₁, p₂)
        if len(premises) == 1:
            fg.add_factor(
                factor_id=eid,
                factor_type=FactorType.SOFT_IMPLICATION,
                premises=premises,
                conclusions=[conclusion],
                p=p1,
                p2=MAXENT_P2,
            )
            continue

        # Multi-premise edge → CONJUNCTION mediator + SOFT_IMPLICATION
        # Per theory §5: ∧ (deterministic) separates "how premises combine"
        # from "how they support conclusion"
        mediator = f"{eid}_M"
        if mediator not in fg.variables:
            fg.add_variable(mediator, 0.5)
        synthetic_var_ids.add(mediator)

        # CONJUNCTION: premises → mediator (deterministic AND)
        fg.add_factor(
            factor_id=f"{eid}_conj",
            factor_type=FactorType.CONJUNCTION,
            premises=premises,
            conclusions=[mediator],
            p=1.0 - CROMWELL_EPS,
        )

        # SOFT_IMPLICATION: mediator → conclusion ↝(p₁, p₂)
        fg.add_factor(
            factor_id=eid,
            factor_type=FactorType.SOFT_IMPLICATION,
            premises=[mediator],
            conclusions=[conclusion],
            p=p1,
            p2=MAXENT_P2,
        )

    # ── Contradiction factors ──
    for eid_a, eid_b, _cid in _detect_contradictions(graph):
        edge_a = graph.edges[eid_a]
        edge_b = graph.edges[eid_b]
        claim_vars = sorted({
            pid for pid in edge_a.premise_ids + edge_b.premise_ids if pid in fg.variables
        })
        if not claim_vars:
            continue
        for a, b in combinations(claim_vars, 2):
            relation_var = f"contra_rel_{eid_a}_{eid_b}_{a}_{b}"
            if relation_var not in fg.variables:
                fg.add_variable(relation_var, 0.5)
            synthetic_var_ids.add(relation_var)
            fg.add_factor(
                factor_id=f"contra_factor_{eid_a}_{eid_b}_{a}_{b}",
                factor_type=FactorType.CONTRADICTION,
                premises=[a, b],
                conclusions=[],
                p=1.0 - CROMWELL_EPS,
                relation_var=relation_var,
            )

    # ── Equivalence factors ──
    for nid_a, nid_b in _detect_equivalences(graph):
        if nid_a not in fg.variables or nid_b not in fg.variables:
            continue
        relation_var = f"equiv_rel_{nid_a}_{nid_b}"
        if relation_var not in fg.variables:
            fg.add_variable(relation_var, 0.5)
        synthetic_var_ids.add(relation_var)
        fg.add_factor(
            factor_id=f"equiv_factor_{nid_a}_{nid_b}",
            factor_type=FactorType.EQUIVALENCE,
            premises=[nid_a, nid_b],
            conclusions=[],
            p=1.0 - CROMWELL_EPS,
            relation_var=relation_var,
        )

    return ZeroInferenceGraphV2(factor_graph=fg, synthetic_var_ids=synthetic_var_ids)
