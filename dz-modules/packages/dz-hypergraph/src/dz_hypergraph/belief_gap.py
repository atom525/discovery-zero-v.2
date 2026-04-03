"""Belief gap analysis utilities extracted from discovery engine."""

from __future__ import annotations

from typing import Any, List, Set, Tuple

from dz_hypergraph.models import HyperGraph


class BeliefGapAnalyser:
    """
    Identify "critical lemma" positions: nodes where if belief were raised,
    the largest downstream belief gain would occur.

    This is a graph-structural analysis, not LLM-based.
    """

    def find_critical_gaps(
        self,
        graph: HyperGraph,
        target_node_id: str,
        top_k: int = 5,
        search_state: Any | None = None,
    ) -> List[Tuple[str, float]]:
        """
        For each unverified node reachable upstream from target,
        estimate the marginal belief gain on target if that node
        were proven (belief → 1.0).

        Returns [(node_id, estimated_target_belief_gain)] sorted descending.
        """
        target = graph.nodes.get(target_node_id)
        if target is None:
            return []

        upstream: Set[str] = set()
        frontier = [target_node_id]
        while frontier:
            nid = frontier.pop()
            for eid in graph.get_edges_to(nid):
                edge = graph.edges[eid]
                for pid in edge.premise_ids:
                    if pid not in upstream and pid in graph.nodes:
                        node = graph.nodes[pid]
                        if node.state == "unverified":
                            upstream.add(pid)
                            frontier.append(pid)

        gains: List[Tuple[str, float]] = []
        for nid in upstream:
            gain = self._estimate_marginal_gain(graph, nid, target_node_id)
            if gain <= 0.01:
                continue
            readiness = self._premise_readiness(graph, nid)
            visit_count = 0
            if search_state is not None and hasattr(search_state, "visit_counts"):
                visit_count = int(getattr(search_state, "visit_counts", {}).get(nid, 0))
            effective_gain = gain * (0.3 + 0.7 * readiness) / (1.0 + 0.3 * visit_count)
            if effective_gain > 0.0:
                gains.append((nid, effective_gain))

        gains.sort(key=lambda x: x[1], reverse=True)
        return gains[:top_k]

    def _estimate_marginal_gain(
        self, graph: HyperGraph, node_id: str, target_id: str
    ) -> float:
        """
        Cheap heuristic: count how many paths from node_id to target_id
        have node_id as the weakest link (min belief premise).
        """
        node = graph.nodes.get(node_id)
        if node is None:
            return 0.0
        current_belief = node.belief

        paths_found = 0
        bottleneck_count = 0
        visited: Set[str] = set()
        stack: List[Tuple[str, float]] = [(node_id, current_belief)]

        while stack and paths_found < 20:
            nid, min_belief_on_path = stack.pop()
            if nid == target_id:
                paths_found += 1
                if min_belief_on_path <= current_belief + 0.01:
                    bottleneck_count += 1
                continue
            if nid in visited:
                continue
            visited.add(nid)

            for eid in graph.get_edges_from(nid):
                edge = graph.edges[eid]
                cid = edge.conclusion_id
                conclusion = graph.nodes.get(cid)
                if conclusion is None:
                    continue
                new_min = min(min_belief_on_path, conclusion.belief)
                stack.append((cid, new_min))

        if paths_found == 0:
            return 0.0
        return (bottleneck_count / paths_found) * (1.0 - current_belief) * 0.5

    def _premise_readiness(self, graph: HyperGraph, node_id: str) -> float:
        """How ready this node is for verification based on premise coverage."""
        edges_to_node = graph.get_edges_to(node_id)
        if not edges_to_node:
            return 1.0
        best_readiness = 0.0
        for eid in edges_to_node:
            edge = graph.edges[eid]
            if not edge.premise_ids:
                best_readiness = max(best_readiness, 1.0)
                continue
            verified_count = sum(
                1
                for pid in edge.premise_ids
                if pid in graph.nodes and graph.nodes[pid].belief >= 0.7
            )
            readiness = verified_count / len(edge.premise_ids)
            best_readiness = max(best_readiness, readiness)
        return best_readiness
