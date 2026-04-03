"""JSON persistence for the reasoning hypergraph, with Gaia Graph IR export.

Exports use real Gaia types (``LocalCanonicalGraph``, ``LocalParameterization``,
``BeliefSnapshot``) so that Zero's graphs are consumable by any Gaia pipeline
(e.g. ``run_local_bp.py``, storage ingest, global graph merge).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dz_hypergraph.models import HyperGraph

DEFAULT_GRAPH_PATH = Path("./discovery_zero_graph.json")

# Gaia Graph IR types
from libs.graph_ir.models import (
    FactorNode as GaiaFactorNode,
    FactorParams,
    LocalCanonicalGraph,
    LocalCanonicalNode,
    LocalParameterization,
    SourceRef,
)

# Prefer Gaia's canonical BeliefSnapshot model when available.
try:
    from libs.storage.models import BeliefSnapshot  # type: ignore
except Exception:
    # Fallback local schema-compatible model when storage deps are unavailable.
    from pydantic import BaseModel as _BM, Field as _F

    class BeliefSnapshot(_BM):
        knowledge_id: str
        version: int
        belief: float = _F(ge=0, le=1)
        bp_run_id: str
        computed_at: datetime


# ------------------------------------------------------------------ #
# Core save / load (Zero-native JSON)                                  #
# ------------------------------------------------------------------ #

def save_graph(graph: HyperGraph, path: Path = DEFAULT_GRAPH_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(graph.model_dump_json(indent=2))


def load_graph(path: Path = DEFAULT_GRAPH_PATH) -> HyperGraph:
    if not path.exists():
        return HyperGraph()
    return HyperGraph.model_validate_json(path.read_text())


# ------------------------------------------------------------------ #
# Gaia Graph IR export                                                 #
# ------------------------------------------------------------------ #

_ZERO_EDGE_TYPE_TO_FACTOR_TYPE = {
    "heuristic": "reasoning",
    "formal": "reasoning",
    "decomposition": "instantiation",
}


def export_as_gaia_local_graph(
    graph: HyperGraph,
    package_name: str = "discovery_zero",
    version: str = "0.1.0",
) -> tuple[LocalCanonicalGraph, LocalParameterization]:
    """Export the Zero graph as real Gaia ``LocalCanonicalGraph`` + ``LocalParameterization``.

    Returns a ``(local_graph, parameterization)`` pair that can be passed
    directly to ``adapt_local_graph_to_factor_graph`` or serialised via
    ``model_dump_json()``.
    """
    knowledge_nodes: list[LocalCanonicalNode] = []
    factor_nodes: list[GaiaFactorNode] = []
    node_priors: dict[str, float] = {}
    factor_parameters: dict[str, FactorParams] = {}

    for nid, node in graph.nodes.items():
        source_refs = []
        if node.provenance:
            source_refs.append(SourceRef(
                package=package_name,
                version=version,
                module=node.provenance,
                knowledge_name=nid,
            ))

        knowledge_nodes.append(LocalCanonicalNode(
            local_canonical_id=nid,
            package=package_name,
            knowledge_type="claim",
            representative_content=node.statement,
            source_refs=source_refs,
            metadata={
                "formal_statement": node.formal_statement,
                "state": node.state,
                "domain": node.domain,
            },
        ))
        node_priors[nid] = node.prior

    for eid, edge in graph.edges.items():
        factor_type = _ZERO_EDGE_TYPE_TO_FACTOR_TYPE.get(edge.edge_type, "reasoning")
        factor_nodes.append(GaiaFactorNode(
            factor_id=eid,
            type=factor_type,
            premises=list(edge.premise_ids),
            conclusion=edge.conclusion_id,
            metadata={
                "module": edge.module.value,
                "edge_type": edge.edge_type,
                "review_confidence": edge.review_confidence,
                "steps": edge.steps,
            },
        ))
        factor_parameters[eid] = FactorParams(
            conditional_probability=edge.confidence,
        )

    local_graph = LocalCanonicalGraph(
        package=package_name,
        version=version,
        knowledge_nodes=knowledge_nodes,
        factor_nodes=factor_nodes,
    )

    parameterization = LocalParameterization(
        graph_hash=local_graph.graph_hash(),
        node_priors=node_priors,
        factor_parameters=factor_parameters,
    )

    return local_graph, parameterization


def export_as_gaia_beliefs(
    graph: HyperGraph,
    bp_run_id: str = "latest",
) -> list[BeliefSnapshot]:
    """Export belief snapshots as real Gaia ``BeliefSnapshot`` instances.

    Each node yields one snapshot with ``version=1`` (Zero doesn't track
    knowledge versions).
    """
    now = datetime.now(timezone.utc)
    snapshots: list[BeliefSnapshot] = []
    for nid, node in graph.nodes.items():
        snapshots.append(BeliefSnapshot(
            knowledge_id=nid,
            version=1,
            belief=max(0.0, min(1.0, node.belief)),
            bp_run_id=bp_run_id,
            computed_at=now,
        ))
    return snapshots
