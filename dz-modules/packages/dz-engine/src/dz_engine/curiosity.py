"""
Curiosity-Driven Exploration for Discovery Zero.

Three complementary signals that measure how "novel" and "surprising" an
action is from the system's perspective.  These drive exploration of under-
investigated regions of the knowledge hypergraph even when immediate belief
gain is low.

  NoveltyTracker:
    Tracks what statement types, edge patterns, and bridge topologies
    the system has already seen.  Novelty = fraction of new elements
    introduced by an action.

  StrategySurprise:
    Measures the divergence between what Gaia BP *predicted* after an action
    and what actually happened.  High surprise → BP model inadequate → exactly
    the region Neural BP needs to learn.  Inspired by SuS (2026).

  CuriosityDrivenExplorer:
    Combines NoveltyTracker + StrategySurprise + PAV uncertainty into a
    unified exploration bonus added to the UCB score.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from dz_hypergraph.models import HyperGraph, Module
from dz_engine.search import FrontierNode


# ------------------------------------------------------------------ #
# Novelty Tracker                                                      #
# ------------------------------------------------------------------ #

class NoveltyTracker:
    """
    Track what the discovery system has previously encountered to
    compute a novelty reward for new actions.

    Three novelty dimensions:
      1. Statement novelty: is the proposition statement meaningfully new?
      2. Edge pattern novelty: is the (premise_types, conclusion_type, module) pattern new?
      3. Bridge topology novelty: is the DAG structure of a bridge plan new?
    """

    def __init__(self) -> None:
        self.seen_statement_hashes: Set[str] = set()
        self.seen_edge_patterns: Set[Tuple[str, ...]] = set()
        self.seen_bridge_topologies: Set[str] = set()
        self._history: List[Dict[str, Any]] = []

    def _statement_hash(self, statement: str) -> str:
        # Short hash of normalised statement
        normalised = " ".join(statement.lower().split())[:300]
        return hashlib.sha1(normalised.encode()).hexdigest()[:12]

    def _edge_pattern(
        self,
        premise_states: List[str],
        conclusion_state: str,
        module: Module,
    ) -> Tuple[str, ...]:
        return tuple(sorted(premise_states)) + (conclusion_state, module.value)

    def _bridge_topo_hash(self, propositions: List[Any]) -> str:
        """Hash the DAG structure of a bridge plan (ignoring statement content)."""
        topo = [(p.id, tuple(sorted(p.depends_on))) for p in propositions]
        return hashlib.sha1(json.dumps(topo, sort_keys=True).encode()).hexdigest()[:12]

    def observe_action(
        self,
        new_nodes: List[Any],       # list of Node objects added
        new_edges: List[Any],       # list of Hyperedge objects added
        graph: HyperGraph,
        bridge_propositions: Optional[List[Any]] = None,
    ) -> float:
        """
        Record what was introduced by an action and return novelty score in [0, 1].

        Novelty = weighted fraction of new elements:
          0.5 * statement_novelty + 0.3 * edge_pattern_novelty + 0.2 * bridge_topo_novelty
        """
        new_stmt_count = 0
        for node in new_nodes:
            h = self._statement_hash(node.statement)
            if h not in self.seen_statement_hashes:
                self.seen_statement_hashes.add(h)
                new_stmt_count += 1
        stmt_novelty = min(1.0, new_stmt_count / max(len(new_nodes), 1))

        new_pattern_count = 0
        for edge in new_edges:
            conclusion = graph.nodes.get(edge.conclusion_id)
            premises = [graph.nodes.get(p) for p in edge.premise_ids]
            if conclusion is None:
                continue
            pattern = self._edge_pattern(
                [p.state if p else "unknown" for p in premises],
                conclusion.state,
                edge.module,
            )
            if pattern not in self.seen_edge_patterns:
                self.seen_edge_patterns.add(pattern)
                new_pattern_count += 1
        edge_novelty = min(1.0, new_pattern_count / max(len(new_edges), 1))

        bridge_novelty = 0.0
        if bridge_propositions is not None and bridge_propositions:
            topo_hash = self._bridge_topo_hash(bridge_propositions)
            if topo_hash not in self.seen_bridge_topologies:
                self.seen_bridge_topologies.add(topo_hash)
                bridge_novelty = 1.0

        total = 0.5 * stmt_novelty + 0.3 * edge_novelty + 0.2 * bridge_novelty
        self._history.append({
            "stmt_novelty": round(stmt_novelty, 3),
            "edge_novelty": round(edge_novelty, 3),
            "bridge_novelty": round(bridge_novelty, 3),
            "total": round(total, 3),
        })
        return total

    def compute_novelty(self, action_result: Any, graph: HyperGraph) -> float:
        """
        Compute novelty from an ActionResult object.

        Extracts new nodes/edges from action_result.created_node_ids /
        action_result.ingest_edge_id and delegates to observe_action.
        """
        new_nodes = []
        for nid in getattr(action_result, "created_node_ids", []):
            node = graph.nodes.get(nid)
            if node:
                new_nodes.append(node)

        new_edges = []
        eid = getattr(action_result, "ingest_edge_id", None)
        if eid and eid in graph.edges:
            new_edges.append(graph.edges[eid])

        return self.observe_action(new_nodes, new_edges, graph)


# ------------------------------------------------------------------ #
# Strategy Surprise                                                    #
# ------------------------------------------------------------------ #

class StrategySurprise:
    """
    Measures the divergence between BP-predicted beliefs and actual beliefs
    after an action.

    High surprise = the action changed beliefs in ways that Gaia's standard
    BP did not predict.  This indicates a genuinely novel discovery direction
    that the current model cannot explain — exactly the region Neural BP
    correction should learn.

    Inspired by SuS (Strategy-aware Surprise, 2026) but operating at the
    hypergraph level rather than the tactic level.
    """

    def compute(
        self,
        beliefs_before: Dict[str, float],
        bp_predicted: Dict[str, float],
        beliefs_actual: Dict[str, float],
    ) -> float:
        """
        Compute mean squared divergence between predicted and actual beliefs,
        normalised to [0, 1].

        Only considers nodes that exist in all three dicts.
        """
        common = (
            set(beliefs_before.keys())
            & set(bp_predicted.keys())
            & set(beliefs_actual.keys())
        )
        if not common:
            return 0.0

        total = 0.0
        for nid in common:
            diff = beliefs_actual[nid] - bp_predicted[nid]
            total += diff * diff
        mse = total / len(common)
        # Normalise: max MSE for [0,1] beliefs is 1.0 (delta = ±1)
        return min(1.0, 4.0 * mse)  # scale so 0.5 MSE → 1.0

    def compute_kl_divergence(
        self,
        bp_predicted: Dict[str, float],
        beliefs_actual: Dict[str, float],
        eps: float = 1e-6,
    ) -> float:
        """
        KL divergence KL(actual || predicted) as an alternative surprise measure.

        Higher KL = more surprised.  Normalised to [0, 1] via tanh.
        """
        common = set(bp_predicted.keys()) & set(beliefs_actual.keys())
        if not common:
            return 0.0

        kl = 0.0
        for nid in common:
            p = max(eps, min(1 - eps, beliefs_actual[nid]))
            q = max(eps, min(1 - eps, bp_predicted[nid]))
            kl += p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

        kl /= len(common)
        # Normalise KL to [0, 1]
        return math.tanh(kl)


# ------------------------------------------------------------------ #
# Curiosity-Driven Explorer                                            #
# ------------------------------------------------------------------ #

@dataclass
class CuriosityConfig:
    novelty_weight: float = 0.4
    surprise_weight: float = 0.4
    pav_uncertainty_weight: float = 0.2
    # Decay factor for novelty bonus over time (exploration → exploitation)
    decay_per_round: float = 0.99


class CuriosityDrivenExplorer:
    """
    Combines NoveltyTracker + StrategySurprise + PAV uncertainty into a
    unified exploration bonus.

    The bonus is added on top of the UCB score in RMaxTSSearch to guide
    the agent toward genuinely novel and surprising regions of the hypergraph.

    Core insight: when PAV uncertainty is high AND novelty is high AND
    strategy surprise is high, this frontier deserves extra exploration even
    if the immediate belief gain signal is weak.
    """

    def __init__(
        self,
        novelty_tracker: Optional[NoveltyTracker] = None,
        surprise: Optional[StrategySurprise] = None,
        pav: Optional[Any] = None,  # ProcessAdvantageVerifier
        config: Optional[CuriosityConfig] = None,
    ) -> None:
        self.novelty_tracker = novelty_tracker or NoveltyTracker()
        self.surprise = surprise or StrategySurprise()
        self.pav = pav
        self.config = config or CuriosityConfig()
        self._round = 0

    def next_round(self) -> None:
        """Call at the start of each discovery round to apply decay."""
        self._round += 1

    def exploration_bonus(
        self,
        graph: HyperGraph,
        target_node_id: str,
        candidate: FrontierNode,
        *,
        beliefs_before: Optional[Dict[str, float]] = None,
        bp_predicted: Optional[Dict[str, float]] = None,
        beliefs_actual: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Compute the exploration bonus for a candidate frontier node.

        Args:
            graph: Current hypergraph.
            target_node_id: Target theorem node.
            candidate: FrontierNode being evaluated.
            beliefs_before/bp_predicted/beliefs_actual: Optional belief dicts
                for computing strategy surprise.

        Returns:
            Bonus in [0, 1] (to be added to UCB score).
        """
        decay = self.config.decay_per_round ** self._round

        # Novelty component: how many new node/edge types does this candidate introduce?
        # Use the candidate's bridge_bonus as a proxy for potential novelty.
        novelty = min(1.0, candidate.bridge_bonus + candidate.diversity_bonus)

        # Surprise component
        surprise = 0.0
        if bp_predicted is not None and beliefs_actual is not None:
            surprise = self.surprise.compute(
                beliefs_before or {},
                bp_predicted,
                beliefs_actual,
            )

        # PAV uncertainty: high variance across multiple PAV predictions → high uncertainty
        pav_uncertainty = 0.0
        if self.pav is not None:
            try:
                node = graph.nodes.get(candidate.node_id)
                if node is not None:
                    # Estimate uncertainty as |PAV(plausible) - PAV(lean)|
                    pav_p = self.pav.predict_advantage(
                        graph, target_node_id, candidate.node_id, Module.PLAUSIBLE
                    )
                    pav_l = self.pav.predict_advantage(
                        graph, target_node_id, candidate.node_id, Module.LEAN
                    )
                    pav_uncertainty = abs(pav_p - pav_l)
            except Exception:
                pass

        bonus = (
            self.config.novelty_weight * novelty
            + self.config.surprise_weight * surprise
            + self.config.pav_uncertainty_weight * pav_uncertainty
        )
        return min(1.0, bonus * decay)

    def should_explore(
        self,
        graph: HyperGraph,
        target_node_id: str,
        candidate: FrontierNode,
        ucb_score: float,
        threshold: float = 0.5,
    ) -> bool:
        """
        Return True if the exploration bonus justifies selecting this candidate
        even when its raw UCB score is below the threshold.
        """
        bonus = self.exploration_bonus(graph, target_node_id, candidate)
        return (ucb_score + bonus) >= threshold
