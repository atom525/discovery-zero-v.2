"""
Theorem Discovery Engine — conjecture generation, mutation, and structural induction.

This module is deliberately decoupled from the existing proof-search pipeline.
It interfaces with the rest of Zero exclusively through HyperGraph and standard
ingest/propagation utilities.  It does NOT modify orchestrator, mcts_engine,
or benchmark code.

Three discovery modes:

  PatternInductor:
    Scans proven/high-belief subgraphs for recurring structural motifs
    (e.g. "all edges of form A→B with module=experiment also have a
    plausible edge A→B") and proposes generalisations as new conjectures.

  CounterexampleRefiner:
    When an experiment refutes a conjecture, analyses the counterexample
    structure and proposes weakened/strengthened variants that survive
    the counterexample.

  AnalogyConjecturer:
    Given two proven subgraphs with similar topology, proposes a conjecture
    that "completes the analogy" in a third, less-explored region.

All three modes produce ConjectureProposal objects that can be ingested
into the graph as new unverified nodes with appropriate provenance.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from dz_hypergraph.models import HyperGraph, Hyperedge, Module, Node
from dz_hypergraph.ingest import ingest_skill_output
from dz_hypergraph.inference import propagate_beliefs
from dz_hypergraph.tools.llm import (
    LLMError,
    chat_completion,
    extract_json_block,
    extract_text_content,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Data types                                                           #
# ------------------------------------------------------------------ #

@dataclass
class ConjectureProposal:
    """A proposed new conjecture discovered by the system."""

    statement: str
    rationale: str
    discovery_mode: str  # "pattern_induction" | "counterexample_refinement" | "analogy"
    confidence: float = 0.3
    parent_node_ids: List[str] = field(default_factory=list)
    evidence_summary: str = ""
    formal_statement: Optional[str] = None
    domain: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement": self.statement,
            "rationale": self.rationale,
            "discovery_mode": self.discovery_mode,
            "confidence": round(self.confidence, 4),
            "parent_node_ids": self.parent_node_ids,
            "evidence_summary": self.evidence_summary,
            "formal_statement": self.formal_statement,
            "domain": self.domain,
            "metadata": self.metadata,
        }


@dataclass
class StructuralMotif:
    """A recurring pattern in the hypergraph."""

    description: str
    edge_pattern: Tuple[str, ...]  # (premise_states..., conclusion_state, module)
    frequency: int
    example_edge_ids: List[str] = field(default_factory=list)
    avg_confidence: float = 0.0


@dataclass
class DiscoveryResult:
    """Result of one discovery cycle."""

    proposals: List[ConjectureProposal] = field(default_factory=list)
    motifs_found: int = 0
    refinements_attempted: int = 0
    analogies_found: int = 0
    nodes_created: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_proposals": len(self.proposals),
            "motifs_found": self.motifs_found,
            "refinements_attempted": self.refinements_attempted,
            "analogies_found": self.analogies_found,
            "nodes_created": self.nodes_created,
            "proposals": [p.to_dict() for p in self.proposals],
        }


# ------------------------------------------------------------------ #
# Pattern Inductor                                                     #
# ------------------------------------------------------------------ #

class PatternInductor:
    """
    Scan proven/high-belief subgraphs for recurring structural motifs
    and propose generalisations as new conjectures.

    Strategy:
      1. Extract all edge patterns: (sorted premise states, conclusion state, module)
      2. Count frequencies; patterns appearing ≥ min_frequency are motifs
      3. For each motif, find unverified nodes that *could* participate
         but currently lack an edge of that pattern → gap
      4. Ask LLM to generalise the motif into a conjecture filling the gap
    """

    def __init__(self, min_frequency: int = 2, min_belief: float = 0.6) -> None:
        self.min_frequency = min_frequency
        self.min_belief = min_belief

    def extract_motifs(self, graph: HyperGraph) -> List[StructuralMotif]:
        pattern_counter: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
        pattern_confidence: Dict[Tuple[str, ...], List[float]] = defaultdict(list)

        for eid, edge in graph.edges.items():
            conclusion = graph.nodes.get(edge.conclusion_id)
            if conclusion is None:
                continue
            premise_states = []
            for pid in edge.premise_ids:
                p = graph.nodes.get(pid)
                if p:
                    premise_states.append(p.state)
                else:
                    premise_states.append("unknown")
            pattern = tuple(sorted(premise_states)) + (conclusion.state, edge.module.value)
            pattern_counter[pattern].append(eid)
            pattern_confidence[pattern].append(edge.confidence)

        motifs: List[StructuralMotif] = []
        for pattern, edge_ids in pattern_counter.items():
            if len(edge_ids) < self.min_frequency:
                continue
            confs = pattern_confidence[pattern]
            motifs.append(StructuralMotif(
                description=f"Pattern: {' + '.join(pattern[:-2])} → {pattern[-2]} via {pattern[-1]}",
                edge_pattern=pattern,
                frequency=len(edge_ids),
                example_edge_ids=edge_ids[:5],
                avg_confidence=sum(confs) / len(confs) if confs else 0.0,
            ))
        motifs.sort(key=lambda m: m.frequency, reverse=True)
        return motifs

    def find_gaps(
        self, graph: HyperGraph, motif: StructuralMotif
    ) -> List[Tuple[List[str], str]]:
        """
        Find (premise_ids, conclusion_id) tuples where the motif pattern
        *could* apply but no such edge exists yet.
        """
        module_val = motif.edge_pattern[-1]
        conclusion_state = motif.edge_pattern[-2]
        premise_states = list(motif.edge_pattern[:-2])

        # Find candidate conclusion nodes
        candidates: List[str] = []
        for nid, node in graph.nodes.items():
            if node.state == "unverified" and node.belief >= 0.3:
                existing_modules = {
                    graph.edges[eid].module.value
                    for eid in graph.get_edges_to(nid)
                }
                if module_val not in existing_modules:
                    candidates.append(nid)

        gaps: List[Tuple[List[str], str]] = []
        for cid in candidates[:10]:  # cap to avoid explosion
            # Find premise nodes that match the pattern's premise states
            premise_candidates: List[str] = []
            for nid, node in graph.nodes.items():
                if nid == cid:
                    continue
                if node.belief >= self.min_belief:
                    premise_candidates.append(nid)
            if len(premise_candidates) >= len(premise_states):
                gaps.append((premise_candidates[:len(premise_states)], cid))
        return gaps[:5]

    def propose_from_motif(
        self,
        graph: HyperGraph,
        motif: StructuralMotif,
        model: Optional[str] = None,
    ) -> List[ConjectureProposal]:
        """Ask LLM to generalise a motif into a conjecture."""
        # Collect example edges for context
        examples = []
        for eid in motif.example_edge_ids[:3]:
            edge = graph.edges.get(eid)
            if edge is None:
                continue
            premises = [graph.nodes[p].statement for p in edge.premise_ids if p in graph.nodes]
            conclusion = graph.nodes.get(edge.conclusion_id)
            if conclusion:
                examples.append({
                    "premises": premises,
                    "conclusion": conclusion.statement,
                    "module": edge.module.value,
                    "confidence": edge.confidence,
                })

        if not examples:
            return []

        prompt = (
            f"The following pattern appears {motif.frequency} times in our reasoning graph:\n"
            f"Pattern: {motif.description}\n\n"
            f"Examples:\n{json.dumps(examples, indent=2, ensure_ascii=False)}\n\n"
            "Based on this recurring pattern, propose 1-3 NEW conjectures that:\n"
            "1. Generalise or extend the pattern to unexplored territory\n"
            "2. Are precise and falsifiable\n"
            "3. Are non-trivial (not direct restatements of the examples)\n\n"
            'Return JSON: {"conjectures": [{"statement": "...", "rationale": "...", "confidence": 0.3}]}'
        )
        try:
            response = chat_completion(
                messages=[
                    {"role": "system", "content": "You are a mathematical conjecture generator. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.7,
            )
            raw = extract_text_content(response)
            parsed = extract_json_block(raw)
        except (LLMError, Exception):
            return []

        proposals: List[ConjectureProposal] = []
        items = parsed.get("conjectures", []) if isinstance(parsed, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            stmt = str(item.get("statement", "")).strip()
            if not stmt:
                continue
            proposals.append(ConjectureProposal(
                statement=stmt,
                rationale=str(item.get("rationale", "pattern induction")),
                discovery_mode="pattern_induction",
                confidence=min(0.5, float(item.get("confidence", 0.3))),
                parent_node_ids=[
                    graph.edges[eid].conclusion_id
                    for eid in motif.example_edge_ids[:3]
                    if eid in graph.edges
                ],
                evidence_summary=f"Induced from motif with frequency={motif.frequency}, avg_conf={motif.avg_confidence:.2f}",
                metadata={"motif_description": motif.description, "motif_frequency": motif.frequency},
            ))
        return proposals


# ------------------------------------------------------------------ #
# Counterexample Refiner                                               #
# ------------------------------------------------------------------ #

class CounterexampleRefiner:
    """
    When an experiment refutes a conjecture, analyse the counterexample
    and propose modified conjectures that survive it.

    Strategy:
      1. Find refuted nodes that have incoming experiment edges
      2. Extract the counterexample/refutation evidence from edge steps
      3. Ask LLM to propose weakened (add hypothesis) or strengthened
         (weaken conclusion) variants
    """

    def find_refuted_with_evidence(
        self, graph: HyperGraph
    ) -> List[Tuple[str, str, List[str]]]:
        """Returns (node_id, statement, evidence_lines) for refuted nodes."""
        results: List[Tuple[str, str, List[str]]] = []
        for nid, node in graph.nodes.items():
            if node.state != "refuted":
                continue
            evidence: List[str] = []
            for eid in graph.get_edges_to(nid):
                edge = graph.edges[eid]
                if edge.module in (Module.EXPERIMENT, Module.PLAUSIBLE):
                    for step in edge.steps:
                        if any(kw in step.lower() for kw in
                               ("counterexample", "refut", "violat", "fail", "false")):
                            evidence.append(step)
            if evidence:
                results.append((nid, node.statement, evidence))
        return results

    def propose_refinements(
        self,
        graph: HyperGraph,
        node_id: str,
        statement: str,
        evidence: List[str],
        model: Optional[str] = None,
    ) -> List[ConjectureProposal]:
        evidence_text = "\n".join(f"- {e}" for e in evidence[:5])
        # Collect premises that led to this node (potential hypotheses to strengthen)
        premise_stmts: List[str] = []
        for eid in graph.get_edges_to(node_id):
            edge = graph.edges[eid]
            for pid in edge.premise_ids:
                p = graph.nodes.get(pid)
                if p:
                    premise_stmts.append(p.statement)

        prompt = (
            f"The following conjecture was REFUTED by experiment:\n"
            f"Conjecture: {statement}\n\n"
            f"Counterexample / refutation evidence:\n{evidence_text}\n\n"
        )
        if premise_stmts:
            prompt += f"Original premises:\n" + "\n".join(f"- {s}" for s in premise_stmts[:5]) + "\n\n"

        prompt += (
            "Propose 2-3 MODIFIED conjectures that:\n"
            "1. Survive the known counterexample (add a hypothesis, or weaken the conclusion)\n"
            "2. Are still non-trivial and interesting\n"
            "3. Each is clearly stated\n\n"
            "For each, explain which modification strategy you used (weaken_conclusion, "
            "add_hypothesis, restrict_domain, change_bound).\n\n"
            'Return JSON: {"refinements": [{"statement": "...", "strategy": "...", "rationale": "..."}]}'
        )
        try:
            response = chat_completion(
                messages=[
                    {"role": "system", "content": "You are a mathematical conjecture refiner. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.5,
            )
            raw = extract_text_content(response)
            parsed = extract_json_block(raw)
        except (LLMError, Exception):
            return []

        proposals: List[ConjectureProposal] = []
        items = parsed.get("refinements", []) if isinstance(parsed, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            stmt = str(item.get("statement", "")).strip()
            if not stmt:
                continue
            strategy = str(item.get("strategy", "unknown"))
            proposals.append(ConjectureProposal(
                statement=stmt,
                rationale=str(item.get("rationale", f"Refined via {strategy}")),
                discovery_mode="counterexample_refinement",
                confidence=0.35,
                parent_node_ids=[node_id],
                evidence_summary=f"Refined from refuted conjecture via {strategy}",
                metadata={"strategy": strategy, "original_node_id": node_id},
            ))
        return proposals


# ------------------------------------------------------------------ #
# Analogy Conjecturer                                                  #
# ------------------------------------------------------------------ #

class AnalogyConjecturer:
    """
    Given proven subgraphs with similar topology, propose a conjecture
    that "completes" an analogous but incomplete subgraph.

    Strategy:
      1. Find pairs of nodes with similar graph neighbourhoods (Jaccard on
         edge types / module types of their 2-hop subgraphs)
      2. If one is proven and the other unverified, the unverified node's
         missing edges suggest conjectures
      3. LLM proposes the conjecture by analogical transfer
    """

    def __init__(self, similarity_threshold: float = 0.3) -> None:
        self.similarity_threshold = similarity_threshold

    def _node_signature(self, graph: HyperGraph, nid: str) -> Set[str]:
        """Collect multiset of (edge_type, module) in 2-hop neighbourhood."""
        sig: Set[str] = set()
        visited: Set[str] = {nid}
        frontier = [nid]
        for _ in range(2):
            next_frontier: List[str] = []
            for n in frontier:
                for eid in graph.get_edges_to(n) + graph.get_edges_from(n):
                    edge = graph.edges.get(eid)
                    if edge is None:
                        continue
                    sig.add(f"{edge.edge_type}:{edge.module.value}")
                    for connected in edge.premise_ids + [edge.conclusion_id]:
                        if connected not in visited:
                            visited.add(connected)
                            next_frontier.append(connected)
            frontier = next_frontier
        return sig

    def find_analogous_pairs(
        self, graph: HyperGraph
    ) -> List[Tuple[str, str, float]]:
        """Find (proven_id, unverified_id, similarity) pairs."""
        proven_nodes = [
            nid for nid, n in graph.nodes.items()
            if n.state == "proven" or n.belief >= 0.9
        ]
        unverified_nodes = [
            nid for nid, n in graph.nodes.items()
            if n.state == "unverified" and n.belief < 0.7
        ]

        signatures: Dict[str, Set[str]] = {}
        for nid in proven_nodes + unverified_nodes:
            signatures[nid] = self._node_signature(graph, nid)

        pairs: List[Tuple[str, str, float]] = []
        for pid in proven_nodes:
            for uid in unverified_nodes:
                sig_p = signatures[pid]
                sig_u = signatures[uid]
                if not sig_p or not sig_u:
                    continue
                jaccard = len(sig_p & sig_u) / len(sig_p | sig_u)
                if jaccard >= self.similarity_threshold:
                    pairs.append((pid, uid, jaccard))

        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:10]

    def propose_from_analogy(
        self,
        graph: HyperGraph,
        proven_id: str,
        unverified_id: str,
        similarity: float,
        model: Optional[str] = None,
    ) -> List[ConjectureProposal]:
        proven = graph.nodes.get(proven_id)
        unverified = graph.nodes.get(unverified_id)
        if not proven or not unverified:
            return []

        # Collect edges of the proven node as structural template
        proven_edges = []
        for eid in graph.get_edges_to(proven_id):
            edge = graph.edges[eid]
            premises = [graph.nodes[p].statement for p in edge.premise_ids if p in graph.nodes]
            proven_edges.append({
                "premises": premises,
                "module": edge.module.value,
                "confidence": edge.confidence,
            })

        # Edges of unverified node (what we already have)
        unverified_edges = []
        for eid in graph.get_edges_to(unverified_id):
            edge = graph.edges[eid]
            premises = [graph.nodes[p].statement for p in edge.premise_ids if p in graph.nodes]
            unverified_edges.append({
                "premises": premises,
                "module": edge.module.value,
                "confidence": edge.confidence,
            })

        prompt = (
            f"Proven statement (high confidence):\n{proven.statement}\n"
            f"Supporting evidence: {json.dumps(proven_edges, ensure_ascii=False)}\n\n"
            f"Analogous unverified statement:\n{unverified.statement}\n"
            f"Current evidence: {json.dumps(unverified_edges, ensure_ascii=False)}\n\n"
            f"Structural similarity: {similarity:.2f}\n\n"
            "The proven statement has stronger support. By analogy, propose 1-2 conjectures "
            "that would help establish the unverified statement, inspired by the proven one's "
            "support structure. Focus on the GAP — what kind of evidence is the unverified "
            "statement missing that the proven one has?\n\n"
            'Return JSON: {"conjectures": [{"statement": "...", "rationale": "..."}]}'
        )
        try:
            response = chat_completion(
                messages=[
                    {"role": "system", "content": "You are a mathematical analogy expert. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                temperature=0.6,
            )
            raw = extract_text_content(response)
            parsed = extract_json_block(raw)
        except (LLMError, Exception):
            return []

        proposals: List[ConjectureProposal] = []
        items = parsed.get("conjectures", []) if isinstance(parsed, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            stmt = str(item.get("statement", "")).strip()
            if not stmt:
                continue
            proposals.append(ConjectureProposal(
                statement=stmt,
                rationale=str(item.get("rationale", "analogy-based discovery")),
                discovery_mode="analogy",
                confidence=0.3,
                parent_node_ids=[proven_id, unverified_id],
                evidence_summary=f"Analogical transfer from proven node (similarity={similarity:.2f})",
                domain=unverified.domain,
                metadata={"proven_id": proven_id, "unverified_id": unverified_id, "similarity": similarity},
            ))
        return proposals


# ------------------------------------------------------------------ #
# Belief Gap Analyser                                                  #
# ------------------------------------------------------------------ #

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

        # Collect upstream nodes via BFS on edges leading to target
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

        # For each upstream node, simulate setting it to proven and estimate gain
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

        # BFS forward: find paths from node_id to target_id
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


# ------------------------------------------------------------------ #
# Discovery Engine — unified orchestrator                              #
# ------------------------------------------------------------------ #

class DiscoveryEngine:
    """
    Main entry point for theorem discovery.

    Runs all three discovery modes and ingests proposals into the graph.
    Designed to be called periodically alongside the proof-search loop
    without modifying any existing code.

    Usage:
        engine = DiscoveryEngine(model="cds/Claude-4.6-opus")
        result = engine.discover(graph, target_node_id="abc123")
        # result.proposals contains new conjectures
        # result.nodes_created contains IDs of nodes added to graph
    """

    def __init__(
        self,
        model: Optional[str] = None,
        max_proposals_per_mode: int = 3,
        auto_ingest: bool = True,
        min_ingest_confidence: float = 0.2,
    ) -> None:
        self.model = model
        self.max_proposals = max_proposals_per_mode
        self.auto_ingest = auto_ingest
        self.min_ingest_confidence = min_ingest_confidence
        self.pattern_inductor = PatternInductor()
        self.counterexample_refiner = CounterexampleRefiner()
        self.analogy_conjecturer = AnalogyConjecturer()
        self.belief_gap_analyser = BeliefGapAnalyser()

    def discover(
        self,
        graph: HyperGraph,
        target_node_id: Optional[str] = None,
        *,
        enable_pattern: bool = True,
        enable_refine: bool = True,
        enable_analogy: bool = True,
    ) -> DiscoveryResult:
        """
        Run all enabled discovery modes and return proposals.

        If auto_ingest is True, proposals above min_ingest_confidence are
        added to the graph as new unverified nodes with plausible edges.
        """
        result = DiscoveryResult()
        all_proposals: List[ConjectureProposal] = []

        # 1. Pattern induction
        if enable_pattern:
            motifs = self.pattern_inductor.extract_motifs(graph)
            result.motifs_found = len(motifs)
            for motif in motifs[:3]:
                proposals = self.pattern_inductor.propose_from_motif(graph, motif, self.model)
                all_proposals.extend(proposals[:self.max_proposals])

        # 2. Counterexample refinement
        if enable_refine:
            refuted = self.counterexample_refiner.find_refuted_with_evidence(graph)
            result.refinements_attempted = len(refuted)
            for node_id, statement, evidence in refuted[:3]:
                proposals = self.counterexample_refiner.propose_refinements(
                    graph, node_id, statement, evidence, self.model
                )
                all_proposals.extend(proposals[:self.max_proposals])

        # 3. Analogy-based discovery
        if enable_analogy:
            pairs = self.analogy_conjecturer.find_analogous_pairs(graph)
            result.analogies_found = len(pairs)
            for proven_id, unverified_id, sim in pairs[:3]:
                proposals = self.analogy_conjecturer.propose_from_analogy(
                    graph, proven_id, unverified_id, sim, self.model
                )
                all_proposals.extend(proposals[:self.max_proposals])

        # Deduplicate by statement hash
        seen: Set[str] = set()
        unique: List[ConjectureProposal] = []
        for p in all_proposals:
            h = hashlib.sha1(p.statement.lower().encode()).hexdigest()[:12]
            if h not in seen:
                seen.add(h)
                unique.append(p)
        result.proposals = unique

        # Auto-ingest into graph
        if self.auto_ingest:
            result.nodes_created = self._ingest_proposals(graph, unique)

        # Critical gap analysis
        if target_node_id:
            gaps = self.belief_gap_analyser.find_critical_gaps(graph, target_node_id)
            if gaps:
                result.proposals.append(ConjectureProposal(
                    statement=f"Critical lemma candidates for {target_node_id}: "
                              + "; ".join(f"{graph.nodes[nid].statement[:60]}(gain={g:.2f})" for nid, g in gaps[:3]),
                    rationale="Belief gap analysis identified these as bottleneck nodes",
                    discovery_mode="belief_gap_analysis",
                    confidence=0.0,  # meta-proposal, not a conjecture
                    metadata={"gaps": [(nid, g) for nid, g in gaps]},
                ))

        return result

    def _ingest_proposals(
        self, graph: HyperGraph, proposals: List[ConjectureProposal]
    ) -> List[str]:
        """Add proposals to graph as new unverified nodes with provenance."""
        created: List[str] = []
        for p in proposals:
            if p.confidence < self.min_ingest_confidence:
                continue
            # Check if statement already exists
            existing = graph.find_node_ids_by_statement(p.statement)
            if existing:
                continue

            node = graph.add_node(
                statement=p.statement,
                belief=p.confidence,
                prior=p.confidence,
                domain=p.domain,
                provenance=f"discovery:{p.discovery_mode}",
                formal_statement=p.formal_statement,
            )
            created.append(node.id)

            # If there are parent nodes, create a plausible edge
            valid_parents = [pid for pid in p.parent_node_ids if pid in graph.nodes]
            if valid_parents:
                graph.add_hyperedge(
                    premise_ids=valid_parents,
                    conclusion_id=node.id,
                    module=Module.PLAUSIBLE,
                    steps=[
                        f"Discovery mode: {p.discovery_mode}",
                        f"Rationale: {p.rationale}",
                        f"Evidence: {p.evidence_summary}",
                    ],
                    confidence=p.confidence,
                    edge_type="heuristic",
                )

        if created:
            propagate_beliefs(graph)
        return created
