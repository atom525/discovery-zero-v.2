"""Core data model: Node, Hyperedge, HyperGraph.

Dual-state design (RENEW): nodes have discrete state (unverified/proven/refuted)
and continuous belief; edges are either heuristic (soft) or formal (hard).

Extended for Gaia integration:
- Node.prior: the BP input prior (distinct from posterior belief)
- Node.provenance: source annotation for traceability
- Hyperedge.review_confidence: judge-assigned score (separate from factor confidence)
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator


NodeState = Literal["unverified", "proven", "refuted"]
EdgeType = Literal["heuristic", "formal", "decomposition"]


def canonicalize_statement_text(statement: str) -> str:
    """
    Canonicalize a natural-language mathematical statement for matching.

    The goal is conservative deduplication:
    - normalize unicode width/compatibility
    - collapse repeated whitespace
    - ignore trailing sentence punctuation
    - compare case-insensitively
    """
    normalized = unicodedata.normalize("NFKC", statement).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.rstrip(" .。")
    return normalized.casefold()


class Module(str, Enum):
    PLAUSIBLE = "plausible"
    EXPERIMENT = "experiment"
    LEAN = "lean"
    ANALOGY = "analogy"
    DECOMPOSE = "decompose"
    SPECIALIZE = "specialize"
    RETRIEVE = "retrieve"


class OperatorType(str, Enum):
    """Deprecated: kept for backward compatibility with serialized graphs.

    Per Gaia theory (03-propositional-operators.md §4), heuristic edges compile
    to SOFT_ENTAILMENT ↝(p₁, p₂).  The adapter does not read this field; it
    maps edge_type + edge.confidence to Gaia factors (see adapter_v2).
    """

    ENTAILMENT = "entailment"
    INDUCTION = "induction"
    ABDUCTION = "abduction"


class Node(BaseModel):
    """A proposition node in the reasoning hypergraph.

    Dual-state: state is discrete (unverified/proven/refuted); belief is
    continuous in [0,1]. Proven => belief locked 1; Refuted => belief locked 0.

    prior: the BP input (what we believe before running inference).
    belief: the BP output (posterior after inference). For backward compat,
            old JSON without 'prior' will use the existing belief value.
    provenance: optional source tag (e.g. "plausible", "bridge", "axiom").
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    statement: str
    formal_statement: Optional[str] = None
    belief: float = 0.5
    prior: float = 0.5
    domain: Optional[str] = None
    provenance: Optional[str] = None
    verification_source: Optional[str] = None
    memo_ref: Optional[str] = None
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    state: NodeState = "unverified"

    @field_validator("belief")
    @classmethod
    def clamp_belief(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @field_validator("prior")
    @classmethod
    def clamp_prior(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @model_validator(mode="before")
    @classmethod
    def migrate_prior_from_belief(cls, data):
        """Backward compatibility: if prior is missing in serialized data, copy from belief."""
        if isinstance(data, dict) and "prior" not in data:
            data["prior"] = data.get("belief", 0.5)
        return data

    @model_validator(mode="after")
    def sync_state_belief(self) -> "Node":
        if self.state == "proven":
            if self.belief != 1.0:
                object.__setattr__(self, "belief", 1.0)
            if self.prior != 1.0:
                object.__setattr__(self, "prior", 1.0)
        if self.state == "refuted":
            if self.belief != 0.0:
                object.__setattr__(self, "belief", 0.0)
            if self.prior != 0.0:
                object.__setattr__(self, "prior", 0.0)
        return self

    def is_locked(self) -> bool:
        """True if belief must not be updated by propagation (proven or refuted)."""
        return self.state in ("proven", "refuted")


class Hyperedge(BaseModel):
    """A reasoning step: premises -> conclusion.

    Heuristic edges (plausible/experiment) carry confidence and act as soft
    constraints; formal edges (lean) are hard constraints (no probability).
    Decomposition edges are Lean-derived subgoal structures used to connect a
    parent goal to unresolved child goals.

    review_confidence: score from the judge skill, kept separate from the
        factor confidence used in BP.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    premise_ids: list[str]
    conclusion_id: str
    module: Module
    steps: list[str]
    confidence: float = 0.5
    review_confidence: Optional[float] = None
    claim_refs: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    edge_type: EdgeType = "heuristic"
    operator_type: Optional[OperatorType] = None

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class HyperGraph(BaseModel):
    """The reasoning hypergraph: nodes (propositions) + edges (reasoning steps).

    Adjacency indices (_edges_to, _edges_from) provide O(1) lookups.
    They are private (not serialized) and rebuilt on model_post_init.
    """

    nodes: dict[str, Node] = Field(default_factory=dict)
    edges: dict[str, Hyperedge] = Field(default_factory=dict)

    # Private adjacency indices — not serialised, rebuilt from edges on load
    _edges_to: dict[str, list[str]] = PrivateAttr(default_factory=dict)
    """conclusion_id → [edge_id, ...]"""

    _edges_from: dict[str, list[str]] = PrivateAttr(default_factory=dict)
    """premise_node_id → [edge_id, ...]"""

    def model_post_init(self, __context: object) -> None:
        """Rebuild adjacency indices from the loaded edges dict."""
        self._edges_to = {}
        self._edges_from = {}
        for eid, edge in self.edges.items():
            self._edges_to.setdefault(edge.conclusion_id, []).append(eid)
            for pid in edge.premise_ids:
                self._edges_from.setdefault(pid, []).append(eid)

    def add_node(
        self,
        statement: str,
        belief: float = 0.5,
        formal_statement: str | None = None,
        domain: str | None = None,
        state: NodeState | None = None,
        prior: float | None = None,
        provenance: str | None = None,
        verification_source: str | None = None,
        memo_ref: str | None = None,
    ) -> Node:
        if state is None:
            state = "unverified"
        resolved_prior = prior if prior is not None else belief
        node = Node(
            statement=statement,
            belief=belief,
            prior=resolved_prior,
            formal_statement=formal_statement,
            domain=domain,
            state=state,
            provenance=provenance,
            verification_source=verification_source,
            memo_ref=memo_ref,
        )
        self.nodes[node.id] = node
        return node

    def find_node_ids_by_statement(
        self,
        statement: str,
        *,
        exact_first: bool = True,
        canonicalize: bool = True,
    ) -> list[str]:
        """
        Find matching nodes by statement.

        This is intentionally non-destructive: it does not merge nodes, it only
        helps callers resolve references robustly across minor punctuation /
        whitespace variations.
        """
        if exact_first:
            exact = [nid for nid, n in self.nodes.items() if n.statement == statement]
            if exact or not canonicalize:
                return exact
        if not canonicalize:
            return []
        target = canonicalize_statement_text(statement)
        return [
            nid
            for nid, n in self.nodes.items()
            if canonicalize_statement_text(n.statement) == target
        ]

    def add_hyperedge(
        self,
        premise_ids: list[str],
        conclusion_id: str,
        module: Module,
        steps: list[str],
        confidence: float = 0.5,
        edge_type: EdgeType | None = None,
        operator_type: OperatorType | None = None,
    ) -> Hyperedge:
        premise_ids = [pid for pid in premise_ids if pid != conclusion_id]
        for nid in premise_ids + [conclusion_id]:
            if nid not in self.nodes:
                raise ValueError(f"Node '{nid}' not found in hypergraph")
        if edge_type is None:
            edge_type = "formal" if module == Module.LEAN else "heuristic"
        # operator_type is deprecated (kept for backward compat with serialized
        # graphs).  The adapter derives factor parameters from edge_type and
        # confidence per Gaia theory.  Do not auto-infer.
        edge = Hyperedge(
            premise_ids=premise_ids,
            conclusion_id=conclusion_id,
            module=module,
            steps=steps,
            confidence=confidence,
            edge_type=edge_type,
            operator_type=operator_type,
        )
        self.edges[edge.id] = edge
        # Maintain adjacency indices
        self._edges_to.setdefault(edge.conclusion_id, []).append(edge.id)
        for pid in premise_ids:
            self._edges_from.setdefault(pid, []).append(edge.id)
        return edge

    def remove_edge(self, edge_id: str) -> Optional["Hyperedge"]:
        """Remove a hyperedge and update adjacency indices."""
        edge = self.edges.pop(edge_id, None)
        if edge is None:
            return None
        # Update indices
        to_list = self._edges_to.get(edge.conclusion_id, [])
        if edge_id in to_list:
            to_list.remove(edge_id)
        for pid in edge.premise_ids:
            from_list = self._edges_from.get(pid, [])
            if edge_id in from_list:
                from_list.remove(edge_id)
        return edge

    def get_edges_to(self, node_id: str) -> list[str]:
        """O(1) lookup: edge IDs whose conclusion is node_id."""
        return list(self._edges_to.get(node_id, []))

    def get_edges_from(self, node_id: str) -> list[str]:
        """O(1) lookup: edge IDs that have node_id as a premise."""
        return list(self._edges_from.get(node_id, []))

    def summary(self) -> dict:
        return {
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "num_axioms": sum(
                1 for n in self.nodes.values()
                if n.state == "proven" and not self.get_edges_to(n.id)
            ),
            "num_proven": sum(1 for n in self.nodes.values() if n.state == "proven"),
            "num_refuted": sum(1 for n in self.nodes.values() if n.state == "refuted"),
            "num_unverified": sum(
                1 for n in self.nodes.values() if n.state == "unverified"
            ),
        }
