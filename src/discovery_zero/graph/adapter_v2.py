"""Zero HyperGraph -> Gaia inference_v2 FactorGraph adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from libs.inference_v2.factor_graph import CROMWELL_EPS, FactorGraph, OperatorType

from discovery_zero.graph.adapter import _detect_contradictions, _detect_equivalences
from discovery_zero.graph.models import HyperGraph


@dataclass
class ZeroInferenceGraphV2:
    factor_graph: FactorGraph
    synthetic_var_ids: set[str] = field(default_factory=set)


def adapt_zero_graph_v2(
    graph: HyperGraph,
    *,
    warmstart: bool = False,
) -> ZeroInferenceGraphV2:
    """Convert a Zero HyperGraph into inference_v2's FactorGraph."""
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

    for eid, edge in graph.edges.items():
        if edge.edge_type == "decomposition":
            continue
        premises = [pid for pid in edge.premise_ids if pid in fg.variables]
        conclusion = edge.conclusion_id
        if conclusion not in fg.variables:
            continue
        if not premises:
            continue
        p1_raw = max(float(edge.confidence), 1.0 - CROMWELL_EPS) if edge.edge_type == "formal" else float(edge.confidence)
        # Neutral completion for absent premise branch.
        p2 = 0.5
        # inference_v2 requires p1 + p2 > 1; keep a tiny positive margin.
        p1 = max(p1_raw, 1.0 - p2 + 1e-3)
        if len(premises) == 1:
            fg.add_factor(
                factor_id=eid,
                factor_type=OperatorType.SOFT_IMPLICATION,
                premises=premises,
                conclusions=[conclusion],
                p=p1,
                p2=p2,
            )
            continue
        mediator = f"{eid}_M"
        if mediator not in fg.variables:
            fg.add_variable(mediator, 0.5)
        synthetic_var_ids.add(mediator)
        fg.add_factor(
            factor_id=f"{eid}_conj",
            factor_type=OperatorType.CONJUNCTION,
            premises=premises,
            conclusions=[mediator],
            p=1.0 - CROMWELL_EPS,
        )
        fg.add_factor(
            factor_id=eid,
            factor_type=OperatorType.SOFT_IMPLICATION,
            premises=[mediator],
            conclusions=[conclusion],
            p=p1,
            p2=p2,
        )

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
                factor_type=OperatorType.CONTRADICTION,
                premises=[a, b],
                conclusions=[],
                p=1.0 - CROMWELL_EPS,
                relation_var=relation_var,
            )

    for nid_a, nid_b in _detect_equivalences(graph):
        if nid_a not in fg.variables or nid_b not in fg.variables:
            continue
        relation_var = f"equiv_rel_{nid_a}_{nid_b}"
        if relation_var not in fg.variables:
            fg.add_variable(relation_var, 0.5)
        synthetic_var_ids.add(relation_var)
        fg.add_factor(
            factor_id=f"equiv_factor_{nid_a}_{nid_b}",
            factor_type=OperatorType.EQUIVALENCE,
            premises=[nid_a, nid_b],
            conclusions=[],
            p=1.0 - CROMWELL_EPS,
            relation_var=relation_var,
        )

    return ZeroInferenceGraphV2(factor_graph=fg, synthetic_var_ids=synthetic_var_ids)
