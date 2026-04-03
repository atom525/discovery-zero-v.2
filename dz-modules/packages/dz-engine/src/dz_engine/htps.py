"""HyperTree Proof Search (HTPS): selection, evaluation, backup.

Enhanced over the original with:
  - PAV-informed policy prior (replaces uniform 1/N)
  - Virtual loss for concurrent search safety
  - Progressive widening (expand new children only when visit count threshold met)
  - Integration with SearchState from planning/search.py
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dz_hypergraph.models import HyperGraph, Module
from dz_engine.search import (
    SearchState,
    ModuleStats,
    compute_policy_prior,
)


# ------------------------------------------------------------------ #
# HTPS State (extended)                                               #
# ------------------------------------------------------------------ #

@dataclass
class HTPSState:
    """
    Visit counts N(node, edge) and action values Q(node, edge).

    Extended to integrate with SearchState for shared statistics.
    """

    N: Dict[Tuple[str, str], int] = field(default_factory=dict)
    Q: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # Virtual loss: temporarily reduce Q values for concurrent workers
    virtual_loss: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def get_V(self, node_id: str, graph: HyperGraph) -> float:
        """Value of node: 1 if proven, 0 if refuted, else belief."""
        node = graph.nodes.get(node_id)
        if node is None:
            return 0.0
        if node.state == "proven":
            return 1.0
        if node.state == "refuted":
            return 0.0
        return node.belief

    def get_Q(self, node_id: str, edge_id: str) -> float:
        vl = self.virtual_loss.get((node_id, edge_id), 0.0)
        return self.Q.get((node_id, edge_id), 0.0) - vl

    def get_N(self, node_id: str, edge_id: str) -> int:
        return self.N.get((node_id, edge_id), 0)

    def add_virtual_loss(
        self, node_id: str, edge_id: str, delta: float = 1.0
    ) -> None:
        key = (node_id, edge_id)
        self.virtual_loss[key] = self.virtual_loss.get(key, 0.0) + delta

    def remove_virtual_loss(
        self, node_id: str, edge_id: str, delta: float = 1.0
    ) -> None:
        key = (node_id, edge_id)
        current = self.virtual_loss.get(key, 0.0)
        self.virtual_loss[key] = max(0.0, current - delta)

    def to_json_dict(self) -> dict:
        return {
            "N": {f"{nid}::{eid}": n for (nid, eid), n in self.N.items()},
            "Q": {f"{nid}::{eid}": q for (nid, eid), q in self.Q.items()},
        }

    @classmethod
    def from_json_dict(cls, data: dict) -> "HTPSState":
        state = cls()
        for key, value in data.get("N", {}).items():
            nid, eid = key.split("::", 1)
            state.N[(nid, eid)] = int(value)
        for key, value in data.get("Q", {}).items():
            nid, eid = key.split("::", 1)
            state.Q[(nid, eid)] = float(value)
        return state


def save_htps_state(state: HTPSState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_json_dict(), indent=2), encoding="utf-8")


def load_htps_state(path: Path) -> HTPSState:
    if not path.exists():
        return HTPSState()
    return HTPSState.from_json_dict(json.loads(path.read_text(encoding="utf-8")))


# ------------------------------------------------------------------ #
# Progressive widening                                                 #
# ------------------------------------------------------------------ #

def _should_widen(
    n_visits: int,
    n_children: int,
    *,
    pw_base: float = 1.5,
    pw_exponent: float = 0.5,
) -> bool:
    """
    Progressive widening: expand a new child only when:
      n_visits >= pw_base * n_children^(1/pw_exponent)

    Returns True if we should consider adding a new action.
    """
    if n_children == 0:
        return True
    threshold = pw_base * (n_children ** (1.0 / max(pw_exponent, 0.1)))
    return n_visits >= threshold


# ------------------------------------------------------------------ #
# Edge selection with informed prior                                   #
# ------------------------------------------------------------------ #

def _pick_edge_puct(
    graph: HyperGraph,
    state: HTPSState,
    node_id: str,
    edge_ids: List[str],
    c_puct: float = 1.4,
    search_state: Optional[SearchState] = None,
) -> str:
    """
    Select an edge by PUCT with an informed policy prior.

    score(e) = Q(n,e) + c_puct * P(n,e) * sqrt(Σ_e' N(n,e')) / (1 + N(n,e))

    P(n,e) is computed from compute_policy_prior() when search_state is
    available; otherwise falls back to uniform 1/N.
    """
    if not edge_ids:
        return ""

    total_n = sum(state.get_N(node_id, eid) for eid in edge_ids)
    sqrt_total = math.sqrt(max(1, total_n))

    # Compute policy priors
    if search_state is not None:
        priors = compute_policy_prior(graph, node_id, edge_ids, search_state)
    else:
        priors = {eid: 1.0 / len(edge_ids) for eid in edge_ids}

    best_e = edge_ids[0]
    best_score = -1e9

    # Progressive widening check
    n_node_visits = total_n
    n_active = sum(1 for eid in edge_ids if state.get_N(node_id, eid) > 0)

    for eid in edge_ids:
        n = state.get_N(node_id, eid)
        # Skip unvisited edges unless progressive widening allows
        if n == 0 and not _should_widen(n_node_visits, n_active):
            continue
        q = state.get_Q(node_id, eid)  # includes virtual loss adjustment
        p = priors.get(eid, 1.0 / len(edge_ids))
        puct = c_puct * p * sqrt_total / (1 + n)
        score = q + puct
        if score > best_score:
            best_score = score
            best_e = eid

    return best_e


def _min_premise_value(
    graph: HyperGraph, state: HTPSState, premise_ids: List[str]
) -> float:
    return min(state.get_V(pid, graph) for pid in premise_ids) if premise_ids else 1.0


# ------------------------------------------------------------------ #
# HTPS core algorithms                                                 #
# ------------------------------------------------------------------ #

def htps_select(
    graph: HyperGraph,
    state: HTPSState,
    root_id: str,
    max_depth: int = 20,
    c_puct: float = 1.4,
    search_state: Optional[SearchState] = None,
    *,
    apply_virtual_loss: bool = False,
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Select a path from root to a leaf.

    Returns (leaf_id, path) where path is [(node_id, edge_id), ...].
    Leaf: node with no incoming edges (axiom/source), proven/refuted, or max_depth.

    When apply_virtual_loss=True, temporarily decrements Q values along the
    selected path to discourage other concurrent workers from choosing the same
    path (virtual loss for parallel MCTS).
    """
    path: List[Tuple[str, str]] = []
    node_id = root_id

    for _ in range(max_depth):
        node = graph.nodes.get(node_id)
        if node is None or node.state in ("proven", "refuted"):
            break
        incoming = graph.get_edges_to(node_id)
        if not incoming:
            break
        edge_id = _pick_edge_puct(
            graph, state, node_id, incoming, c_puct, search_state
        )
        if not edge_id:
            break

        if apply_virtual_loss:
            state.add_virtual_loss(node_id, edge_id)

        path.append((node_id, edge_id))
        edge = graph.edges.get(edge_id)
        if edge is None or not edge.premise_ids:
            break
        candidate_premises = [
            pid
            for pid in edge.premise_ids
            if pid in graph.nodes and graph.nodes[pid].provenance != "bridge_risk"
        ]
        premise_pool = candidate_premises if candidate_premises else edge.premise_ids
        worst_premise = min(
            premise_pool,
            key=lambda pid: state.get_V(pid, graph),
        )
        node_id = worst_premise

    return (node_id, path)


def htps_backup(
    state: HTPSState,
    path: List[Tuple[str, str]],
    value: float,
    *,
    remove_virtual_loss: bool = False,
) -> None:
    """
    Backup value along path; update N and Q for each (node, edge).

    When remove_virtual_loss=True, restores virtual losses applied during
    selection (call this after the action has been executed).
    """
    for (node_id, edge_id) in path:
        key = (node_id, edge_id)
        n = state.N.get(key, 0) + 1
        q_old = state.Q.get(key, 0.0)
        state.N[key] = n
        state.Q[key] = q_old + (value - q_old) / n
        if remove_virtual_loss:
            state.remove_virtual_loss(node_id, edge_id)


def htps_step(
    graph: HyperGraph,
    state: HTPSState,
    root_id: str,
    max_depth: int = 20,
    c_puct: float = 1.4,
    search_state: Optional[SearchState] = None,
) -> Tuple[str, List[Tuple[str, str]], float]:
    """
    One HTPS step: select to leaf, evaluate leaf, backup.

    Returns (leaf_id, path, value).
    """
    leaf_id, path = htps_select(
        graph, state, root_id, max_depth, c_puct, search_state
    )
    value = state.get_V(leaf_id, graph)
    htps_backup(state, path, value)
    return (leaf_id, path, value)
