"""
Search Engine for Discovery Zero — UCB-driven multi-frontier exploration.

Replaces fixed-threshold module selection (belief < 0.4 → plausible, etc.)
with principled exploration-exploitation via UCB1-Tuned bandit statistics.

Key components:

  SearchState:
    Unified mutable search state shared across all rounds of a discovery run.
    Tracks visit counts, action values, per-(node, module) UCB statistics,
    failure memory, and the current exploration frontier.

  ModuleStats:
    Per-(node, module) UCB1-Tuned statistics for adaptive module selection.

  select_module_ucb():
    UCB1-Tuned bandit that selects the best next module for a given node,
    informed by the node's belief (Bayesian prior) and failure history.

  rank_frontiers():
    Multi-frontier ranking that returns the top-k promising nodes to explore,
    with diversity regularisation so we don't get stuck on one subgraph.

  FrontierNode:
    Priority-annotated node descriptor used in the exploration frontier.

The module also hosts RMaxTS integration stubs (see phase6-rmaxts for the full
implementation that overlays intrinsic rewards on top of this UCB baseline).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from dz_hypergraph.models import HyperGraph, Module
def _infer_claim_type(statement: str) -> str:
    lowered = statement.casefold()
    if any(token in lowered for token in ("conjecture", "theorem", "lemma", "holds for", "for all", "forall", "there exists", "exists", "if ", " then ", "implies")):
        return "structural"
    if any(ch.isdigit() for ch in statement) or any(op in statement for op in ("=", "<", ">", "≤", "≥")):
        return "quantitative"
    return "heuristic"




# ------------------------------------------------------------------ #
# Per-(node, module) UCB statistics                                    #
# ------------------------------------------------------------------ #

@dataclass
class ModuleStats:
    """UCB1-Tuned statistics for one (node_id, module) arm."""

    successes: int = 0
    attempts: int = 0
    total_reward: float = 0.0
    total_reward_sq: float = 0.0  # for variance estimate

    @property
    def mean_reward(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.total_reward / self.attempts

    @property
    def reward_variance(self) -> float:
        """Sample variance estimate: E[X²] - E[X]²."""
        if self.attempts < 2:
            return 0.25  # max variance for [0,1] reward
        mean_sq = self.total_reward_sq / self.attempts
        return max(0.0, mean_sq - self.mean_reward ** 2)

    def record(self, reward: float, *, success: bool = True) -> None:
        self.attempts += 1
        self.total_reward += reward
        self.total_reward_sq += reward * reward
        if success:
            self.successes += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "successes": self.successes,
            "attempts": self.attempts,
            "mean_reward": round(self.mean_reward, 4),
        }


# ------------------------------------------------------------------ #
# Frontier node descriptor                                             #
# ------------------------------------------------------------------ #

@dataclass
class FrontierNode:
    """A candidate node to explore, annotated with priority scores."""

    node_id: str
    belief: float
    priority: float  # higher = more urgent to explore
    value_uncertainty: float = 0.0
    diversity_bonus: float = 0.0
    bridge_bonus: float = 0.0
    suggested_module: Optional[Module] = None

    def __lt__(self, other: "FrontierNode") -> bool:
        return self.priority > other.priority  # max-heap semantics


# ------------------------------------------------------------------ #
# Failure memory                                                       #
# ------------------------------------------------------------------ #

@dataclass
class FailureMemory:
    """Tracks consecutive failures for a (node, module) arm."""

    consecutive: int = 0
    total: int = 0
    last_error_type: str = ""

    def record_failure(self, error_type: str = "") -> None:
        self.consecutive += 1
        self.total += 1
        self.last_error_type = error_type

    def record_success(self) -> None:
        self.consecutive = 0

    @property
    def penalty_factor(self) -> float:
        """Multiplicative penalty for UCB reward due to consecutive failures."""
        if self.consecutive == 0:
            return 1.0
        # Each consecutive failure halves the effective reward
        return max(0.05, 0.5 ** self.consecutive)


# ------------------------------------------------------------------ #
# Unified search state                                                 #
# ------------------------------------------------------------------ #

@dataclass
class SearchState:
    """
    Unified mutable search state for a full discovery run.

    Owned by the DiscoveryEngine; persisted between rounds so UCB
    statistics accumulate across the whole run.
    """

    # Per-node total visit count (across all modules)
    visit_counts: Dict[str, int] = field(default_factory=dict)

    # Per-node Q-value (expected belief gain toward target)
    action_values: Dict[str, float] = field(default_factory=dict)

    # Per-(node_id, module_name) UCB stats
    module_stats: Dict[Tuple[str, str], ModuleStats] = field(default_factory=dict)

    # Per-(node_id, module_name) failure memory
    failure_memory: Dict[Tuple[str, str], FailureMemory] = field(default_factory=dict)

    # Current top-k frontier (updated by rank_frontiers)
    frontier: List[FrontierNode] = field(default_factory=list)

    # Virtual loss table for parallel workers: (node_id, module_name) → count
    virtual_loss: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # Total rounds completed
    rounds_completed: int = 0

    # Last selected module per node (for consecutive-selection tracking)
    _last_selected: Dict[str, str] = field(default_factory=dict)
    _consecutive_same: Dict[str, int] = field(default_factory=dict)

    def record_selection(self, node_id: str, module: Module) -> None:
        """Track which module was just selected for a node (for diversity enforcement)."""
        key = node_id
        if self._last_selected.get(key) == module.value:
            self._consecutive_same[key] = self._consecutive_same.get(key, 0) + 1
        else:
            self._last_selected[key] = module.value
            self._consecutive_same[key] = 1

    def get_consecutive_same_count(self, node_id: str, module: Module) -> int:
        """How many times this module was consecutively selected for this node."""
        if self._last_selected.get(node_id) == module.value:
            return self._consecutive_same.get(node_id, 0)
        return 0

    def get_module_stats(self, node_id: str, module: Module) -> ModuleStats:
        key = (node_id, module.value)
        if key not in self.module_stats:
            self.module_stats[key] = ModuleStats()
        return self.module_stats[key]

    def get_failure_memory(self, node_id: str, module: Module) -> FailureMemory:
        key = (node_id, module.value)
        if key not in self.failure_memory:
            self.failure_memory[key] = FailureMemory()
        return self.failure_memory[key]

    def record_action(
        self,
        node_id: str,
        module: Module,
        reward: float,
        *,
        success: bool = True,
        error_type: str = "",
    ) -> None:
        """Record the outcome of an (node, module) action."""
        key = (node_id, module.value)
        self.visit_counts[node_id] = self.visit_counts.get(node_id, 0) + 1
        ms = self.get_module_stats(node_id, module)
        ms.record(reward, success=success)
        fm = self.get_failure_memory(node_id, module)
        if success:
            fm.record_success()
        else:
            fm.record_failure(error_type)
        # Update Q value
        n = self.visit_counts[node_id]
        q_old = self.action_values.get(node_id, 0.0)
        self.action_values[node_id] = q_old + (reward - q_old) / n

    def add_virtual_loss(self, node_id: str, module: Module, delta: float = 1.0) -> None:
        key = (node_id, module.value)
        self.virtual_loss[key] = self.virtual_loss.get(key, 0.0) + delta

    def remove_virtual_loss(self, node_id: str, module: Module, delta: float = 1.0) -> None:
        key = (node_id, module.value)
        current = self.virtual_loss.get(key, 0.0)
        self.virtual_loss[key] = max(0.0, current - delta)

    def get_virtual_loss(self, node_id: str, module: Module) -> float:
        return self.virtual_loss.get((node_id, module.value), 0.0)

    def total_visits(self) -> int:
        return sum(self.visit_counts.values())

    def summary(self) -> Dict[str, Any]:
        return {
            "total_visits": self.total_visits(),
            "rounds_completed": self.rounds_completed,
            "num_nodes_visited": len(self.visit_counts),
            "num_arms": len(self.module_stats),
        }


# ------------------------------------------------------------------ #
# UCB1-Tuned module selection                                          #
# ------------------------------------------------------------------ #

def select_module_ucb(
    graph: HyperGraph,
    node_id: str,
    state: SearchState,
    *,
    c_explore: float = 1.4,
    belief_prior: Optional[Dict[Module, float]] = None,
) -> Module:
    """
    Select the best module to apply to a node using UCB1-Tuned.

    UCB1-Tuned score for arm (node, module):
      score = Q(n,m) + c_explore * sqrt(ln(N_total) / N(n,m))
                     * min(1/4, V_hat + sqrt(2 * ln(N_total) / N(n,m)))
      minus virtual_loss
      times penalty_factor from failure memory

    When no statistics exist (cold start), uses belief-based priors:
      - plausible: highest prior when belief is low (< 0.5)
      - experiment: moderate prior when belief is medium (0.3–0.7)
      - lean: highest prior when belief is high (> 0.7)

    Args:
        graph: The current hypergraph.
        node_id: The node to select a module for.
        state: Mutable search state.
        c_explore: UCB exploration coefficient.
        belief_prior: Optional override for module priors.

    Returns:
        Selected Module enum value.
    """
    node = graph.nodes.get(node_id)
    if node is None:
        return Module.PLAUSIBLE

    belief = node.belief
    modules = list(Module)
    total_visits = max(1, state.total_visits())

    # Initialise priors based on belief if not provided
    if belief_prior is None:
        claim_type = _infer_claim_type(node.statement)
        if claim_type == "quantitative":
            belief_prior = {
                Module.PLAUSIBLE: 0.08,
                Module.EXPERIMENT: 0.48,
                Module.LEAN: 0.10,
                Module.ANALOGY: 0.08,
                Module.DECOMPOSE: 0.07,
                Module.SPECIALIZE: 0.07,
                Module.RETRIEVE: 0.12,
            }
        elif claim_type == "structural":
            belief_prior = {
                Module.PLAUSIBLE: 0.10,
                Module.EXPERIMENT: 0.12,
                Module.LEAN: 0.42,
                Module.ANALOGY: 0.10,
                Module.DECOMPOSE: 0.12,
                Module.SPECIALIZE: 0.08,
                Module.RETRIEVE: 0.06,
            }
        elif claim_type == "heuristic":
            belief_prior = {
                Module.PLAUSIBLE: 0.32,
                Module.EXPERIMENT: 0.16,
                Module.LEAN: 0.08,
                Module.ANALOGY: 0.16,
                Module.DECOMPOSE: 0.12,
                Module.SPECIALIZE: 0.10,
                Module.RETRIEVE: 0.06,
            }
        else:
            belief_prior = None
    if belief_prior is None:
        if belief < 0.35:
            # Low belief: open-problem territory. Boost DECOMPOSE and EXPERIMENT
            # so subgoal decomposition and experiments start early.
            belief_prior = {
                Module.PLAUSIBLE: 0.22,
                Module.EXPERIMENT: 0.20,
                Module.LEAN: 0.03,
                Module.ANALOGY: 0.18,
                Module.DECOMPOSE: 0.20,
                Module.SPECIALIZE: 0.12,
                Module.RETRIEVE: 0.05,
            }
        elif belief < 0.65:
            belief_prior = {
                Module.PLAUSIBLE: 0.18,
                Module.EXPERIMENT: 0.25,
                Module.LEAN: 0.12,
                Module.ANALOGY: 0.15,
                Module.DECOMPOSE: 0.12,
                Module.SPECIALIZE: 0.10,
                Module.RETRIEVE: 0.08,
            }
        else:
            belief_prior = {
                Module.PLAUSIBLE: 0.08,
                Module.EXPERIMENT: 0.20,
                Module.LEAN: 0.40,
                Module.ANALOGY: 0.10,
                Module.DECOMPOSE: 0.08,
                Module.SPECIALIZE: 0.07,
                Module.RETRIEVE: 0.07,
            }
    plausible_attempts = state.get_module_stats(node_id, Module.PLAUSIBLE).attempts
    experiment_attempts = state.get_module_stats(node_id, Module.EXPERIMENT).attempts
    specialize_attempts = state.get_module_stats(node_id, Module.SPECIALIZE).attempts
    # After at least one plausible or specialize attempt, boost experiment —
    # but only when belief is already reasonably high (>= 0.5), meaning a credible
    # proof route is established and targeted verification makes sense.
    # For frontier open problems with low belief, suppressing plausible exploration
    # prevents the system from discovering new proof routes.
    if (plausible_attempts >= 1 or specialize_attempts >= 1) and experiment_attempts == 0:
        if belief >= 0.5:
            belief_prior = dict(belief_prior)
            belief_prior[Module.PLAUSIBLE] = max(0.05, belief_prior.get(Module.PLAUSIBLE, 0.1) * 0.4)
            belief_prior[Module.EXPERIMENT] = max(0.40, belief_prior.get(Module.EXPERIMENT, 0.1) * 2.0)
        # else: low belief → let UCB explore naturally without suppressing plausible

    best_module = Module.PLAUSIBLE
    best_score = -1e9
    preferred_order = [
        Module.RETRIEVE,
        Module.ANALOGY,
        Module.SPECIALIZE,
        Module.PLAUSIBLE,
        Module.EXPERIMENT,
        Module.DECOMPOSE,
        Module.LEAN,
    ]
    preferred_module: Optional[Module] = None
    for candidate in preferred_order:
        if state.get_module_stats(node_id, candidate).attempts == 0:
            preferred_module = candidate
            break

    for mod in modules:
        ms = state.get_module_stats(node_id, mod)
        fm = state.get_failure_memory(node_id, mod)
        vl = state.get_virtual_loss(node_id, mod)
        prior = belief_prior.get(mod, 1.0 / len(modules))

        if ms.attempts == 0:
            # Cold-start: use prior directly + small bonus for unexplored
            score = prior + 0.1 * c_explore
        else:
            q = ms.mean_reward
            n = ms.attempts
            ln_n_total = math.log(total_visits)
            # UCB1-Tuned
            exploration_term = math.sqrt(ln_n_total / n)
            v_hat = ms.reward_variance
            ucb_tuned_factor = min(0.25, v_hat + math.sqrt(2 * ln_n_total / n))
            score = q + c_explore * exploration_term * ucb_tuned_factor
            # Blend with prior (lower weight as arms get explored)
            explore_blend = max(0.0, 1.0 - n / 20.0)
            score = (1.0 - explore_blend) * score + explore_blend * prior

        # Apply failure penalty and virtual loss
        score *= fm.penalty_factor
        score -= vl * 0.1
        if preferred_module is not None and mod == preferred_module:
            score += 0.15

        # Consecutive-selection decay: penalise modules that have been selected
        # for the same node multiple times in a row, forcing exploration diversity.
        consecutive = state.get_consecutive_same_count(node_id, mod)
        if consecutive >= 2:
            score *= max(0.1, 1.0 - 0.25 * (consecutive - 1))

        if score > best_score:
            best_score = score
            best_module = mod

    return best_module


# ------------------------------------------------------------------ #
# Multi-frontier ranking                                               #
# ------------------------------------------------------------------ #

def rank_frontiers(
    graph: HyperGraph,
    state: SearchState,
    target_node_id: str,
    *,
    max_frontiers: int = 5,
    diversity_weight: float = 0.3,
    min_belief_threshold: float = 0.0,
    exclude_node_ids: Optional[Set[str]] = None,
) -> List[FrontierNode]:
    """
    Rank the top-k most promising nodes to explore.

    Priority = value_uncertainty * (1 + bridge_bonus) * diversity_factor

    where:
      value_uncertainty = 1 - belief (unverified nodes) — higher is better
      bridge_bonus = fraction of incoming edges that are bridges to target
      diversity_factor = 1 + diversity_weight * avg_graph_distance_to_selected

    Returns an ordered list of FrontierNode (highest priority first).
    """
    if exclude_node_ids is None:
        exclude_node_ids = set()

    candidates: List[FrontierNode] = []

    for nid, node in graph.nodes.items():
        if node.state != "unverified":
            continue
        if node.provenance == "bridge_risk":
            continue
        if nid == target_node_id:
            continue
        if nid in exclude_node_ids:
            continue
        if node.belief < min_belief_threshold:
            continue

        # Value uncertainty: how much we can improve belief
        value_uncertainty = max(0.0, 1.0 - node.belief)

        # Bridge bonus: has this node been bridge-planned?
        incoming = graph.get_edges_to(nid)
        bridge_edges = [
            eid for eid in incoming
            if graph.edges[eid].edge_type in ("heuristic", "decomposition")
        ]
        bridge_bonus = 0.3 * min(1.0, len(bridge_edges) / 3.0)

        # Action value from UCB history
        action_val = state.action_values.get(nid, 0.0)

        # Base priority
        base_priority = (
            0.5 * value_uncertainty
            + 0.3 * (1.0 - node.belief)
            + 0.2 * action_val
        ) * (1.0 + bridge_bonus)

        candidates.append(FrontierNode(
            node_id=nid,
            belief=node.belief,
            priority=base_priority,
            value_uncertainty=value_uncertainty,
            bridge_bonus=bridge_bonus,
            suggested_module=select_module_ucb(graph, nid, state),
        ))

    if not candidates:
        return []

    # Sort by base priority
    candidates.sort(key=lambda f: f.priority, reverse=True)

    if max_frontiers >= len(candidates):
        return candidates

    # Diversity selection: greedy max-diversity among top candidates
    selected: List[FrontierNode] = [candidates[0]]
    remaining = candidates[1:]

    while len(selected) < max_frontiers and remaining:
        best_idx = 0
        best_combined = -1.0
        selected_ids = {f.node_id for f in selected}

        for i, cand in enumerate(remaining):
            # Diversity: average graph-neighbourhood distance to selected nodes
            diversity = _estimate_graph_diversity(graph, cand.node_id, selected_ids)
            diversity_factor = 1.0 + diversity_weight * diversity
            combined = cand.priority * diversity_factor
            if combined > best_combined:
                best_combined = combined
                best_idx = i

        chosen = remaining.pop(best_idx)
        chosen.diversity_bonus = best_combined - chosen.priority
        chosen.priority = best_combined
        selected.append(chosen)

    return selected


def _estimate_graph_diversity(
    graph: HyperGraph,
    node_id: str,
    selected_ids: Set[str],
    max_hops: int = 3,
) -> float:
    """
    Estimate graph-structural diversity of node_id relative to selected set.

    Returns a value in [0, 1]: 1.0 = node is structurally distant from all
    selected nodes; 0.0 = node shares immediate neighbourhood with all.

    Uses BFS up to max_hops to estimate shared-neighbourhood overlap.
    """
    if not selected_ids:
        return 1.0

    # BFS neighbourhood of node_id
    def bfs_neighbourhood(start: str) -> Set[str]:
        visited: Set[str] = {start}
        frontier = [start]
        for _ in range(max_hops):
            next_frontier = []
            for nid in frontier:
                for eid in graph.get_edges_to(nid) + graph.get_edges_from(nid):
                    edge = graph.edges.get(eid)
                    if edge is None:
                        continue
                    for connected in edge.premise_ids + [edge.conclusion_id]:
                        if connected not in visited:
                            visited.add(connected)
                            next_frontier.append(connected)
            frontier = next_frontier
            if not frontier:
                break
        return visited

    node_neighbourhood = bfs_neighbourhood(node_id)
    min_overlap = 1.0

    for sid in selected_ids:
        sel_neighbourhood = bfs_neighbourhood(sid)
        if not node_neighbourhood or not sel_neighbourhood:
            continue
        union = node_neighbourhood | sel_neighbourhood
        intersection = node_neighbourhood & sel_neighbourhood
        overlap = len(intersection) / len(union) if union else 0.0
        min_overlap = min(min_overlap, 1.0 - overlap)

    return min_overlap


# ------------------------------------------------------------------ #
# Enhanced HTPS integration helpers                                    #
# ------------------------------------------------------------------ #

def compute_policy_prior(
    graph: HyperGraph,
    node_id: str,
    edge_ids: List[str],
    state: SearchState,
) -> Dict[str, float]:
    """
    Compute an informed policy prior over edges for HTPS.

    Replaces the uniform 1/N prior with a distribution informed by:
    - Edge confidence (higher = higher prior)
    - Module statistics (higher success rate = higher prior)
    - Edge type (formal > decomposition > heuristic for proven nodes)

    Returns {edge_id: prior_probability} (sums to ~1.0).
    """
    if not edge_ids:
        return {}

    raw_scores: Dict[str, float] = {}
    for eid in edge_ids:
        edge = graph.edges.get(eid)
        if edge is None:
            raw_scores[eid] = 1.0
            continue

        # Base: edge confidence
        base = edge.confidence

        # Module stats bonus
        mod = edge.module
        ms = state.get_module_stats(node_id, mod)
        if ms.attempts > 0:
            base *= (0.5 + 0.5 * ms.mean_reward)

        # Edge type preference (when near proven)
        node = graph.nodes.get(node_id)
        if node and node.belief > 0.6:
            if edge.edge_type == "formal":
                base *= 2.0
            elif edge.edge_type == "decomposition":
                base *= 1.5

        # Failure penalty
        fm = state.get_failure_memory(node_id, mod)
        base *= fm.penalty_factor

        raw_scores[eid] = max(0.01, base)

    # Normalise to probability distribution
    total = sum(raw_scores.values())
    if total <= 0:
        return {eid: 1.0 / len(edge_ids) for eid in edge_ids}
    return {eid: s / total for eid, s in raw_scores.items()}


# ------------------------------------------------------------------ #
# RMaxTS — Intrinsic Reward MCTS                                       #
# ------------------------------------------------------------------ #

@dataclass
class IntrinsicReward:
    """
    Three-dimensional intrinsic reward for RMaxTS on the hypergraph.

    Adapted from DeepSeek-Prover-V1.5's RMaxTS (tactic-state novelty)
    to the discovery setting where rewards are continuous and multidimensional.
    """

    belief_gain: float = 0.0
    """Delta belief on target node after BP."""

    graph_novelty: float = 0.0
    """Fraction of new node/edge types not seen before."""

    strategy_surprise: float = 0.0
    """Divergence between BP-predicted and actual beliefs after action."""

    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)

    @property
    def total(self) -> float:
        return (
            self.weights[0] * self.belief_gain
            + self.weights[1] * self.graph_novelty
            + self.weights[2] * self.strategy_surprise
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "belief_gain": round(self.belief_gain, 4),
            "graph_novelty": round(self.graph_novelty, 4),
            "strategy_surprise": round(self.strategy_surprise, 4),
            "total": round(self.total, 4),
        }


@dataclass
class RMaxTSConfig:
    c_puct: float = 1.4
    c_intrinsic: float = 0.5
    belief_weight: float = 0.5
    novelty_weight: float = 0.3
    surprise_weight: float = 0.2


class RMaxTSSearch:
    """
    RMaxTS adapted for hypergraph discovery.

    Key differences from DeepSeek's original:
    - State = hypergraph snapshot (not tactic state)
    - Action = (node_id, module) pair (not single tactic)
    - Reward = IntrinsicReward (not binary proof success)
    - Value = PAV prediction (not rollout-to-end)

    PUCT selection score:
      score(n, m) = Q(n, m)
                 + c_puct * P(n, m) * sqrt(N_parent) / (1 + N(n, m))
                 + c_intrinsic * novelty_bonus(n, m)

    where:
      P(n, m) = PAV.predict_advantage() if PAV available,
                else belief_prior(n, m)  (rule-based fallback)
    """

    def __init__(
        self,
        pav: Optional[Any] = None,         # ProcessAdvantageVerifier
        novelty_tracker: Optional[Any] = None,  # NoveltyTracker
        curiosity: Optional[Any] = None,   # CuriosityDrivenExplorer
        config: Optional[RMaxTSConfig] = None,
    ) -> None:
        self._pav = pav
        self._novelty = novelty_tracker
        self._curiosity = curiosity
        self._config = config or RMaxTSConfig()

    def select_action(
        self,
        graph: HyperGraph,
        target_node_id: str,
        candidates: List[FrontierNode],
        state: SearchState,
    ) -> Optional[Tuple[str, Module]]:
        """
        PUCT selection with PAV as value function and intrinsic bonus.

        Returns (node_id, module) for the best action, or None if no candidates.
        """
        if not candidates:
            return None

        best_score = -1e9
        best_action: Optional[Tuple[str, Module]] = None
        total_n = max(1, state.total_visits())

        for candidate in candidates:
            nid = candidate.node_id
            module = candidate.suggested_module or select_module_ucb(graph, nid, state)

            # Q value
            ms = state.get_module_stats(nid, module)
            q = ms.mean_reward if ms.attempts > 0 else 0.0

            # Policy prior
            if self._pav is not None:
                p = (self._pav.predict_advantage(graph, target_node_id, nid, module) + 1) / 2
            else:
                # Rule-based fallback: belief-informed prior
                node = graph.nodes.get(nid)
                belief = node.belief if node else 0.5
                if module == Module.PLAUSIBLE:
                    p = max(0.1, 1.0 - belief)
                elif module == Module.EXPERIMENT:
                    p = max(0.1, abs(belief - 0.5) * 2)
                elif module == Module.LEAN:
                    p = max(0.1, belief)
                elif module == Module.RETRIEVE:
                    p = max(0.1, 1.0 - 0.8 * belief)
                elif module == Module.ANALOGY:
                    p = max(0.1, 1.0 - 0.5 * belief)
                elif module == Module.SPECIALIZE:
                    p = max(0.1, 1.0 - 0.4 * belief)
                elif module == Module.DECOMPOSE:
                    p = max(0.1, 0.2 + 0.6 * belief)
                else:
                    p = 0.1

            # PUCT exploration term
            n_arm = ms.attempts if ms.attempts > 0 else 0
            puct = self._config.c_puct * p * math.sqrt(total_n) / (1 + n_arm)

            # Intrinsic bonus
            intrinsic = 0.0
            if self._curiosity is not None:
                intrinsic = self._curiosity.exploration_bonus(graph, target_node_id, candidate)
            elif self._novelty is not None:
                # Proxy: candidate diversity_bonus as novelty estimate
                intrinsic = candidate.diversity_bonus

            score = q + puct + self._config.c_intrinsic * intrinsic

            # Virtual loss adjustment
            vl = state.get_virtual_loss(nid, module)
            score -= vl * 0.1

            if score > best_score:
                best_score = score
                best_action = (nid, module)

        return best_action

    def compute_intrinsic_reward(
        self,
        belief_before: float,
        belief_after: float,
        novelty: float,
        surprise: float,
    ) -> IntrinsicReward:
        """Construct a 3D intrinsic reward from component signals."""
        belief_gain = max(0.0, belief_after - belief_before)
        return IntrinsicReward(
            belief_gain=belief_gain,
            graph_novelty=novelty,
            strategy_surprise=surprise,
            weights=(
                self._config.belief_weight,
                self._config.novelty_weight,
                self._config.surprise_weight,
            ),
        )
