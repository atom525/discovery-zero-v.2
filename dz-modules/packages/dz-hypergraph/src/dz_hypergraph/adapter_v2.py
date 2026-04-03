"""Zero HyperGraph -> Gaia BP v2 FactorGraph adapter (Gaia IR-aligned API).

Maps HyperGraph edges to FactorType IMPLICATION, CONJUNCTION, SOFT_ENTAILMENT,
CONTRADICTION, and EQUIVALENCE per docs/foundations/theory/06-factor-graphs.md
and docs/foundations/bp/potentials.md.

SOFT_ENTAILMENT parameters are derived from edge metadata only (static); never
from current node beliefs (Gaia 06-factor-graphs §4.1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from gaia.bp.factor_graph import CROMWELL_EPS, FactorGraph, FactorType

from dz_hypergraph.adapter import _detect_contradictions, _detect_equivalences
from dz_hypergraph.models import HyperGraph, Hyperedge

logger = logging.getLogger(__name__)

# Gaia 06-factor-graphs §3.7: when M=0, ψ(0,0)=p2 and ψ(0,1)=1−p2; p2=0.5 gives
# uniform row — "premise absent provides no directional evidence" (07-bp §2.3 C4).
P2_MAXENT_NEUTRAL_SOFT_ENTAILMENT: float = 0.5
PLAUSIBLE_DEDUP_DECAY: float = 0.3


def _module_value(edge: Hyperedge) -> str:
    mod = edge.module
    return mod.value if hasattr(mod, "value") else str(mod)


def _edge_is_deterministic(edge: Hyperedge) -> bool:
    return edge.edge_type in {"formal", "decomposition"}


def _derive_p2_soft_entailment(edge: Hyperedge) -> float:
    mod = _module_value(edge)
    if mod == "experiment":
        return 1.0 - CROMWELL_EPS
    return P2_MAXENT_NEUTRAL_SOFT_ENTAILMENT


def _derive_p1_soft_entailment(edge: Hyperedge) -> float:
    """Author-supplied forward strength for SOFT_ENTAILMENT (static, not BP output).

    formal (Lean): near-certain conditional support (06-fg §3.3 strict → compiled).
    heuristic / decomposition: module-assigned edge.confidence.
    """
    if _edge_is_deterministic(edge):
        return 1.0 - CROMWELL_EPS
    return float(edge.confidence)


def _effective_p1_soft_entailment(edge: Hyperedge, p2: float, *, edge_id: str) -> float:
    """p1 for SOFT_ENTAILMENT: respect author value but enforce Gaia p1+p2>1 after Cromwell.

    Gaia 03-propositional-operators §4 / factor_graph SOFT_ENTAILMENT validation require
    strictly positive support in the isolated two-node channel. The minimal p1 compatible
    with fixed p2 and Cromwell clamp is derived from CROMWELL_EPS (not an arbitrary magic
    number).
    """
    raw = _derive_p1_soft_entailment(edge)
    p2c = max(CROMWELL_EPS, min(1.0 - CROMWELL_EPS, p2))
    # Smallest p1 such that worst-case clamp still yields p1c + p2c > 1.
    floor = 1.0 - p2c + CROMWELL_EPS
    if raw < floor:
        logger.warning(
            "SOFT_ENTAILMENT p1 clamped to floor for edge %s: raw=%.6f floor=%.6f (p2=%.6f)",
            edge_id,
            raw,
            floor,
            p2c,
        )
    if raw <= CROMWELL_EPS:
        logger.warning(
            "Edge %s has near-zero confidence for SOFT_ENTAILMENT (raw=%.6f); "
            "clamp may change semantics.",
            edge_id,
            raw,
        )
    return max(raw, floor)


def _effective_p1_from_raw(raw: float, p2: float, *, edge_id: str) -> float:
    p2c = max(CROMWELL_EPS, min(1.0 - CROMWELL_EPS, p2))
    floor = 1.0 - p2c + CROMWELL_EPS
    if raw < floor:
        logger.warning(
            "SOFT_ENTAILMENT p1 clamped to floor for edge %s: raw=%.6f floor=%.6f (p2=%.6f)",
            edge_id,
            raw,
            floor,
            p2c,
        )
    if raw <= CROMWELL_EPS:
        logger.warning(
            "Edge %s has near-zero confidence for SOFT_ENTAILMENT (raw=%.6f); "
            "clamp may change semantics.",
            edge_id,
            raw,
        )
    return max(raw, floor)


def _add_implication_or_conjunction_then_implication(
    fg: FactorGraph,
    *,
    edge_id: str,
    premises: list[str],
    conclusion: str,
    synthetic_var_ids: set[str],
) -> None:
    if len(premises) == 1:
        fg.add_factor(
            edge_id,
            FactorType.IMPLICATION,
            premises,
            conclusion,
        )
        return
    mediator = f"{edge_id}_M"
    if mediator not in fg.variables:
        fg.add_variable(mediator, 0.5)
    synthetic_var_ids.add(mediator)
    fg.add_factor(
        f"{edge_id}_conj",
        FactorType.CONJUNCTION,
        premises,
        mediator,
    )
    fg.add_factor(
        edge_id,
        FactorType.IMPLICATION,
        [mediator],
        conclusion,
    )


@dataclass
class ZeroInferenceGraphV2:
    factor_graph: FactorGraph
    synthetic_var_ids: set[str] = field(default_factory=set)


def adapt_zero_graph_v2(
    graph: HyperGraph,
    *,
    warmstart: bool = False,
) -> ZeroInferenceGraphV2:
    """Convert a Zero HyperGraph into Gaia BP v2 FactorGraph."""
    fg = FactorGraph()
    synthetic_var_ids: set[str] = set()

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

    plausible_groups: dict[tuple[str, str], list[tuple[str, Hyperedge]]] = {}
    grouped_plausible_eids: set[str] = set()
    for eid, edge in graph.edges.items():
        if (
            _module_value(edge) == "plausible"
            and edge.edge_type not in {"formal", "decomposition"}
            and edge.conclusion_id in fg.variables
            and any(pid in fg.variables for pid in edge.premise_ids)
        ):
            key = (edge.conclusion_id, _module_value(edge))
            plausible_groups.setdefault(key, []).append((eid, edge))

    for group in plausible_groups.values():
        if len(group) <= 1:
            continue
        for eid, _ in group:
            grouped_plausible_eids.add(eid)
        group.sort(key=lambda item: float(item[1].confidence), reverse=True)
        best_eid, best_edge = group[0]
        conclusion = best_edge.conclusion_id
        premise_union: list[str] = []
        seen: set[str] = set()
        for _eid, edge in group:
            for pid in edge.premise_ids:
                if pid in fg.variables and pid not in seen:
                    seen.add(pid)
                    premise_union.append(pid)
        if not premise_union:
            continue
        p2_group = _derive_p2_soft_entailment(best_edge)
        p1_best = float(best_edge.confidence)
        p1_effective_raw = 1.0 - (1.0 - p1_best) * (PLAUSIBLE_DEDUP_DECAY ** (len(group) - 1))
        p1_effective = _effective_p1_from_raw(p1_effective_raw, p2_group, edge_id=best_eid)
        dedup_eid = f"{best_eid}_dedup"
        if len(premise_union) == 1:
            try:
                fg.add_factor(
                    dedup_eid,
                    FactorType.SOFT_ENTAILMENT,
                    premise_union,
                    conclusion,
                    p1=p1_effective,
                    p2=p2_group,
                )
            except ValueError as exc:
                logger.warning("Skipping deduplicated SOFT_ENTAILMENT %s: %s", dedup_eid, exc)
            continue
        mediator = f"{dedup_eid}_M"
        if mediator not in fg.variables:
            fg.add_variable(mediator, 0.5)
        synthetic_var_ids.add(mediator)
        try:
            fg.add_factor(
                f"{dedup_eid}_conj",
                FactorType.CONJUNCTION,
                premise_union,
                mediator,
            )
            fg.add_factor(
                dedup_eid,
                FactorType.SOFT_ENTAILMENT,
                [mediator],
                conclusion,
                p1=p1_effective,
                p2=p2_group,
            )
        except ValueError as exc:
            logger.warning("Skipping deduplicated multi-premise factor %s: %s", dedup_eid, exc)

    for eid, edge in graph.edges.items():
        if eid in grouped_plausible_eids:
            continue
        premises = [pid for pid in edge.premise_ids if pid in fg.variables]
        conclusion = edge.conclusion_id
        if conclusion not in fg.variables:
            continue
        if not premises:
            continue

        if _edge_is_deterministic(edge):
            try:
                _add_implication_or_conjunction_then_implication(
                    fg,
                    edge_id=eid,
                    premises=premises,
                    conclusion=conclusion,
                    synthetic_var_ids=synthetic_var_ids,
                )
            except ValueError as exc:
                logger.warning("Skipping formal IMPLICATION factor %s: %s", eid, exc)
            continue

        p2 = _derive_p2_soft_entailment(edge)
        p1_se = _effective_p1_soft_entailment(edge, p2, edge_id=eid)

        if len(premises) == 1:
            try:
                fg.add_factor(
                    eid,
                    FactorType.SOFT_ENTAILMENT,
                    premises,
                    conclusion,
                    p1=p1_se,
                    p2=p2,
                )
            except ValueError as exc:
                logger.warning("Skipping SOFT_ENTAILMENT factor %s: %s", eid, exc)
            continue

        try:
            mediator = f"{eid}_M"
            if mediator not in fg.variables:
                fg.add_variable(mediator, 0.5)
            synthetic_var_ids.add(mediator)
            fg.add_factor(
                f"{eid}_conj",
                FactorType.CONJUNCTION,
                premises,
                mediator,
            )
            fg.add_factor(
                eid,
                FactorType.SOFT_ENTAILMENT,
                [mediator],
                conclusion,
                p1=p1_se,
                p2=p2,
            )
        except ValueError as exc:
            logger.warning("Skipping multi-premise factors for edge %s: %s", eid, exc)

    # CONTRADICTION factors are intentionally disabled.  _detect_contradictions
    # finds edge pairs sharing the same conclusion where one path has a refuted
    # premise — but BP already handles this via refuted-node prior=ε propagating
    # backward through SOFT_ENTAILMENT modus tollens.  Adding explicit
    # CONTRADICTION factors would double-count the evidence.  A future version
    # may introduce semantic contradiction detection (e.g. "A" vs "¬A" node
    # pairs), but the current heuristic is too coarse (see plan §1.5).
    _unused_contradictions = _detect_contradictions(graph)
    if _unused_contradictions:
        logger.debug(
            "Detected %d contradiction edge pairs (not converted to factors; "
            "handled by modus tollens via refuted priors).",
            len(_unused_contradictions),
        )

    for nid_a, nid_b in _detect_equivalences(graph):
        if nid_a not in fg.variables or nid_b not in fg.variables:
            continue
        relation_var = f"equiv_rel_{nid_a}_{nid_b}"
        if relation_var not in fg.variables:
            fg.add_variable(relation_var, 1.0 - CROMWELL_EPS)
        synthetic_var_ids.add(relation_var)
        try:
            fg.add_factor(
                f"equiv_factor_{nid_a}_{nid_b}",
                FactorType.EQUIVALENCE,
                [nid_a, nid_b],
                relation_var,
            )
        except ValueError as exc:
            logger.warning("Skipping EQUIVALENCE %s/%s: %s", nid_a, nid_b, exc)

    validation_errors = fg.validate()
    if validation_errors:
        raise RuntimeError(
            "FactorGraph validation failed after adapt_zero_graph_v2: "
            + "; ".join(validation_errors)
        )

    incident: set[str] = set()
    for fac in fg.factors:
        for vid in fac.all_vars:
            incident.add(vid)
    isolated = set(fg.variables.keys()) - incident
    if isolated:
        logger.info(
            "Variables with no incident factors (posterior equals prior under BP): %s",
            sorted(isolated),
        )

    return ZeroInferenceGraphV2(factor_graph=fg, synthetic_var_ids=synthetic_var_ids)
