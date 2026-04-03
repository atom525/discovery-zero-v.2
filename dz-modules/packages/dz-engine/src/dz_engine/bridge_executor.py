"""
Bridge Executor — topological execution of bridge proposition chains.

Replaces the original `execute_bridge_followups` which ran a fixed 2-pass loop
with no topological awareness.  The BridgeExecutor:

  1. Topologically sorts bridge propositions by their dependency graph.
  2. Processes propositions from leaves (no dependencies) toward the target.
  3. Dispatches each proposition to the right module based on grade:
       A/B  → strict Lean verification (via TacticByTacticProver if available)
       C    → computational experiment
       D    → plausible reasoning (attempt to upgrade grade)
  4. After each proposition, runs incremental BP on the affected subgraph.
  5. Parallel-executes propositions in the same topological level.
  6. Retries and re-queues on failure using FailureRouter guidance.

The SpeculativeDecomposer (see Phase 6-M) extends this class to generate
and score multiple candidate decomposition trees before committing.
"""

from __future__ import annotations

import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from dz_hypergraph.models import HyperGraph, Module
from dz_hypergraph.session import GraphSession
from dz_engine.bridge import BridgePlan, BridgeProposition

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Result types                                                         #
# ------------------------------------------------------------------ #

@dataclass
class PropositionResult:
    """Outcome of executing one bridge proposition."""

    prop_id: str
    prop_statement: str
    grade: str
    module_used: Module
    success: bool
    belief_after: float = 0.0
    node_id: Optional[str] = None
    error_message: str = ""
    elapsed_ms: float = 0.0


@dataclass
class BridgeExecutionResult:
    """Aggregate result of executing a full bridge plan."""

    target_node_id: str
    total_propositions: int
    successful: int
    failed: int
    skipped: int
    target_belief_before: float
    target_belief_after: float
    proposition_results: List[PropositionResult] = field(default_factory=list)
    passes_completed: int = 0
    elapsed_ms: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_propositions == 0:
            return 0.0
        return self.successful / self.total_propositions

    @property
    def belief_delta(self) -> float:
        return self.target_belief_after - self.target_belief_before

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_node_id": self.target_node_id,
            "total": self.total_propositions,
            "successful": self.successful,
            "failed": self.failed,
            "skipped": self.skipped,
            "target_belief_before": round(self.target_belief_before, 4),
            "target_belief_after": round(self.target_belief_after, 4),
            "belief_delta": round(self.belief_delta, 4),
            "passes": self.passes_completed,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


# ------------------------------------------------------------------ #
# Topological sort                                                     #
# ------------------------------------------------------------------ #

def topological_sort_propositions(
    propositions: List[BridgeProposition],
) -> List[List[BridgeProposition]]:
    """
    Kahn's algorithm topological sort of bridge propositions.

    Returns a list of levels, where each level is a group of propositions
    that can be processed in parallel (no dependencies between them within
    the same level).

    Cycle detection: if a cycle is found, remaining propositions are
    appended at the end in their original order.
    """
    prop_map: Dict[str, BridgeProposition] = {p.id: p for p in propositions}
    dep_count: Dict[str, int] = {p.id: 0 for p in propositions}
    dependents: Dict[str, List[str]] = {p.id: [] for p in propositions}

    for p in propositions:
        for dep_id in p.depends_on:
            if dep_id in dep_count:
                dep_count[p.id] += 1
                dependents[dep_id].append(p.id)
            # Ignore dependencies on nodes outside the plan (seed refs)

    levels: List[List[BridgeProposition]] = []
    ready = [pid for pid, count in dep_count.items() if count == 0]
    processed: Set[str] = set()

    while ready:
        level_props = [prop_map[pid] for pid in ready if pid in prop_map]
        levels.append(level_props)
        processed.update(ready)

        next_ready: List[str] = []
        for pid in ready:
            for dep_pid in dependents.get(pid, []):
                dep_count[dep_pid] -= 1
                if dep_count[dep_pid] == 0:
                    next_ready.append(dep_pid)
        ready = next_ready

    # Handle any remaining (cycle members)
    remaining = [p for p in propositions if p.id not in processed]
    if remaining:
        logger.warning(
            "Bridge topology: %d propositions form a cycle — appending unsorted",
            len(remaining),
        )
        levels.append(remaining)

    return levels


# ------------------------------------------------------------------ #
# BridgeExecutor                                                       #
# ------------------------------------------------------------------ #

class BridgeExecutor:
    """
    Topologically-ordered execution engine for bridge plans.

    Accepts action callbacks for each module type so it can be integrated
    with the orchestrator without circular imports.

    Usage::

        executor = BridgeExecutor(
            session=session,
            plan=bridge_plan,
            target_node_id=target_id,
            run_plausible_fn=...,
            run_experiment_fn=...,
            run_lean_fn=...,
            run_incremental_bp_fn=...,
        )
        result = executor.execute(max_passes=5)
    """

    def __init__(
        self,
        session: GraphSession,
        plan: BridgePlan,
        target_node_id: str,
        *,
        # Callbacks injected by the orchestrator
        run_plausible_fn: Optional[Callable] = None,
        run_experiment_fn: Optional[Callable] = None,
        run_lean_fn: Optional[Callable] = None,
        run_incremental_bp_fn: Optional[Callable] = None,
        max_parallel: int = 3,
        lean_gate_min_belief: float = 0.6,
    ) -> None:
        self._session = session
        self._plan = plan
        self._target_node_id = target_node_id
        self._run_plausible = run_plausible_fn
        self._run_experiment = run_experiment_fn
        self._run_lean = run_lean_fn
        self._run_bp = run_incremental_bp_fn
        self._max_parallel = max_parallel
        self._lean_gate_min_belief = lean_gate_min_belief
        self._lock = threading.Lock()

    def execute(self, max_passes: int = 5) -> BridgeExecutionResult:
        """
        Execute the bridge plan with topological ordering and optional parallelism.

        Multi-pass: after each full topological sweep, re-checks which
        propositions still have unmet dependencies (due to runtime failures)
        and queues them for the next pass.
        """
        t0 = time.monotonic()
        graph = self._session.graph
        target_node = graph.nodes.get(self._target_node_id)
        target_belief_before = target_node.belief if target_node else 0.0

        propositions = list(self._plan.propositions)
        if not propositions:
            return BridgeExecutionResult(
                target_node_id=self._target_node_id,
                total_propositions=0,
                successful=0,
                failed=0,
                skipped=0,
                target_belief_before=target_belief_before,
                target_belief_after=target_belief_before,
            )

        # Topological levels
        levels = topological_sort_propositions(propositions)

        all_results: List[PropositionResult] = []
        successful = 0
        failed = 0
        skipped = 0
        passes_done = 0

        for pass_num in range(max_passes):
            passes_done = pass_num + 1
            any_progress = False

            for level in levels:
                if not level:
                    continue

                # Filter propositions that haven't been successfully processed
                pending = [
                    p for p in level
                    if not self._is_done(p.id, all_results)
                ]
                if not pending:
                    continue

                # Execute this level's propositions (optionally in parallel)
                if self._max_parallel > 1 and len(pending) > 1:
                    level_results = self._execute_parallel(pending)
                else:
                    level_results = [self._execute_one(p) for p in pending]

                for pr in level_results:
                    all_results.append(pr)
                    if pr.success:
                        successful += 1
                        any_progress = True
                    elif pr.error_message:
                        failed += 1
                    else:
                        skipped += 1

                # Incremental BP after this level
                if self._run_bp is not None:
                    try:
                        self._run_bp(self._session.graph)
                    except Exception as exc:
                        logger.warning("Incremental BP error: %s", exc)

            if not any_progress:
                logger.info(
                    "BridgeExecutor: no progress in pass %d — stopping", pass_num + 1
                )
                break

            # Check if target is proven
            target = self._session.graph.nodes.get(self._target_node_id)
            if target and target.state == "proven":
                break

        target_node_after = self._session.graph.nodes.get(self._target_node_id)
        target_belief_after = target_node_after.belief if target_node_after else 0.0

        return BridgeExecutionResult(
            target_node_id=self._target_node_id,
            total_propositions=len(propositions),
            successful=successful,
            failed=failed,
            skipped=skipped,
            target_belief_before=target_belief_before,
            target_belief_after=target_belief_after,
            proposition_results=all_results,
            passes_completed=passes_done,
            elapsed_ms=(time.monotonic() - t0) * 1000,
        )

    def _is_done(self, prop_id: str, results: List[PropositionResult]) -> bool:
        """Return True if a proposition has already been successfully processed."""
        return any(r.prop_id == prop_id and r.success for r in results)

    def _execute_one(self, prop: BridgeProposition) -> PropositionResult:
        """Execute a single bridge proposition."""
        t0 = time.monotonic()
        graph = self._session.graph
        grade = prop.grade

        try:
            if grade in ("A", "B"):
                result = self._dispatch_lean(prop, graph)
            elif grade == "C":
                result = self._dispatch_experiment(prop, graph)
            else:  # D
                result = self._dispatch_plausible(prop, graph)

            elapsed_ms = (time.monotonic() - t0) * 1000
            if result is not None:
                result.elapsed_ms = elapsed_ms
                return result
        except Exception as exc:
            logger.warning("Bridge prop %s execution error: %s", prop.id, exc)
            return PropositionResult(
                prop_id=prop.id,
                prop_statement=prop.statement[:200],
                grade=grade,
                module_used=Module.PLAUSIBLE,
                success=False,
                error_message=str(exc)[:200],
                elapsed_ms=(time.monotonic() - t0) * 1000,
            )

        # Fallback: unattempted
        return PropositionResult(
            prop_id=prop.id,
            prop_statement=prop.statement[:200],
            grade=grade,
            module_used=Module.PLAUSIBLE,
            success=False,
            error_message="No executor available for this proposition",
        )

    def _execute_parallel(
        self, propositions: List[BridgeProposition]
    ) -> List[PropositionResult]:
        """Execute a group of independent propositions in parallel."""
        results: List[PropositionResult] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self._max_parallel, len(propositions))
        ) as pool:
            futures = {pool.submit(self._execute_one, p): p for p in propositions}
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    prop = futures[future]
                    results.append(PropositionResult(
                        prop_id=prop.id,
                        prop_statement=prop.statement[:200],
                        grade=prop.grade,
                        module_used=Module.PLAUSIBLE,
                        success=False,
                        error_message=f"Thread error: {exc}",
                    ))
        return results

    def _dispatch_lean(
        self, prop: BridgeProposition, graph: HyperGraph
    ) -> PropositionResult:
        """Attempt Lean verification for an A/B grade proposition."""
        node = self._find_or_skip_node(prop, graph)
        if node is None:
            return PropositionResult(
                prop_id=prop.id,
                prop_statement=prop.statement[:200],
                grade=prop.grade,
                module_used=Module.LEAN,
                success=False,
                error_message="Proposition node not found in graph",
            )

        # Check lean gate
        if node.belief < self._lean_gate_min_belief:
            logger.debug("Lean gate: node %s belief %.3f below threshold — falling back to experiment", node.id, node.belief)
            return self._dispatch_experiment(prop, graph)

        if self._run_lean is not None:
            try:
                outcome = self._run_lean(node.id, prop)
                success = outcome.get("success", False)
                return PropositionResult(
                    prop_id=prop.id,
                    prop_statement=prop.statement[:200],
                    grade=prop.grade,
                    module_used=Module.LEAN,
                    success=success,
                    node_id=node.id,
                    belief_after=graph.nodes[node.id].belief if success else 0.0,
                    error_message=outcome.get("error", ""),
                )
            except Exception as exc:
                return PropositionResult(
                    prop_id=prop.id,
                    prop_statement=prop.statement[:200],
                    grade=prop.grade,
                    module_used=Module.LEAN,
                    success=False,
                    error_message=str(exc)[:200],
                )
        return PropositionResult(
            prop_id=prop.id, prop_statement=prop.statement[:200], grade=prop.grade,
            module_used=Module.LEAN, success=False, error_message="No Lean executor",
        )

    def _dispatch_experiment(
        self, prop: BridgeProposition, graph: HyperGraph
    ) -> PropositionResult:
        """Run a computational experiment for a C grade proposition."""
        node = self._find_or_skip_node(prop, graph)
        if node is None:
            return PropositionResult(
                prop_id=prop.id, prop_statement=prop.statement[:200], grade=prop.grade,
                module_used=Module.EXPERIMENT, success=False, error_message="Node not found",
            )
        if self._run_experiment is not None:
            try:
                outcome = self._run_experiment(node.id, prop)
                success = outcome.get("success", False)
                return PropositionResult(
                    prop_id=prop.id, prop_statement=prop.statement[:200], grade=prop.grade,
                    module_used=Module.EXPERIMENT, success=success, node_id=node.id,
                    belief_after=graph.nodes[node.id].belief if success else 0.0,
                    error_message=outcome.get("error", ""),
                )
            except Exception as exc:
                return PropositionResult(
                    prop_id=prop.id, prop_statement=prop.statement[:200], grade=prop.grade,
                    module_used=Module.EXPERIMENT, success=False, error_message=str(exc)[:200],
                )
        return PropositionResult(
            prop_id=prop.id, prop_statement=prop.statement[:200], grade=prop.grade,
            module_used=Module.EXPERIMENT, success=False, error_message="No experiment executor",
        )

    def _dispatch_plausible(
        self, prop: BridgeProposition, graph: HyperGraph
    ) -> PropositionResult:
        """Run plausible reasoning for a D grade proposition."""
        node = self._find_or_skip_node(prop, graph)
        if node is None:
            return PropositionResult(
                prop_id=prop.id, prop_statement=prop.statement[:200], grade=prop.grade,
                module_used=Module.PLAUSIBLE, success=False, error_message="Node not found",
            )
        if self._run_plausible is not None:
            try:
                outcome = self._run_plausible(node.id, prop)
                success = outcome.get("success", False)
                return PropositionResult(
                    prop_id=prop.id, prop_statement=prop.statement[:200], grade=prop.grade,
                    module_used=Module.PLAUSIBLE, success=success, node_id=node.id,
                    belief_after=graph.nodes[node.id].belief if success else 0.0,
                    error_message=outcome.get("error", ""),
                )
            except Exception as exc:
                return PropositionResult(
                    prop_id=prop.id, prop_statement=prop.statement[:200], grade=prop.grade,
                    module_used=Module.PLAUSIBLE, success=False, error_message=str(exc)[:200],
                )
        return PropositionResult(
            prop_id=prop.id, prop_statement=prop.statement[:200], grade=prop.grade,
            module_used=Module.PLAUSIBLE, success=False, error_message="No plausible executor",
        )

    def _find_or_skip_node(
        self, prop: BridgeProposition, graph: HyperGraph
    ) -> Optional[Any]:
        """
        Find the node corresponding to a bridge proposition statement.

        Returns the Node object if found, None if not found in graph.
        """
        # Try exact statement match
        matches = graph.find_node_ids_by_statement(prop.statement)
        if matches:
            return graph.nodes[matches[0]]
        # Try partial/canonical match
        canon_matches = graph.find_node_ids_by_statement(prop.statement, canonicalize=True)
        if canon_matches:
            return graph.nodes[canon_matches[0]]
        return None


# ------------------------------------------------------------------ #
# Speculative Decomposer                                               #
# ------------------------------------------------------------------ #

@dataclass
class DecompositionConstraint:
    """VERIFY-RL style structural validity constraints for a decomposition step."""

    complexity_decreasing: bool
    """Subproblem complexity < parent complexity (prevents trivial decompositions)."""

    solution_containment: bool
    """Conjunction of subproblem solutions implies the parent solution."""

    derivation_valid: bool
    """Decomposition step can be derived from formal rules (e.g. Lean can check it)."""

    @property
    def is_valid(self) -> bool:
        return self.complexity_decreasing and self.solution_containment


@dataclass
class DecompositionCandidate:
    """One candidate bridge plan for speculative decomposition."""

    plan: BridgePlan
    temperature_used: float
    candidate_index: int


@dataclass
class ScoredDecomposition:
    """A scored candidate decomposition."""

    candidate: DecompositionCandidate
    belief_gain: float
    grade_quality: float
    constraint: DecompositionConstraint
    pav_advantage: float = 0.0
    total_score: float = 0.0

    def compute_score(self) -> float:
        validity = 1.0 if self.constraint.is_valid else 0.3
        self.total_score = (
            0.4 * self.belief_gain
            + 0.3 * self.grade_quality
            + 0.2 * validity
            + 0.1 * self.pav_advantage
        )
        return self.total_score


class SpeculativeDecomposer:
    """
    Generate multiple candidate decomposition trees for an open problem,
    score them with Gaia BP, and select the most promising one.

    This is fundamentally different from Hilbert/DeepSeek-V2 which decompose
    KNOWN theorems.  We decompose UNKNOWN conjectures, where each candidate
    is a hypothesis about the proof strategy.

    Scoring:
      1. Materialise each candidate virtually (using GraphSession.checkpoint/rollback).
      2. Run Gaia BP on each virtual graph.
      3. Score = belief_gain * grade_quality * constraint_score * pav_advantage.
      4. Select the best; rollback all others.

    Integrates VERIFY-RL constraints for structural validity checking and
    uses the hypergraph itself as a KG (KG-Prover idea) for relevant context.
    """

    def __init__(
        self,
        session: "GraphSession",
        target_node_id: str,
        *,
        generate_plan_fn: Optional[Callable] = None,  # LLM bridge plan generator
        pav: Optional[Any] = None,                     # ProcessAdvantageVerifier
        verify_constraints_fn: Optional[Callable[..., DecompositionConstraint]] = None,
        embedding_model: Optional[Any] = None,
        num_candidates: int = 3,
        temperatures: Optional[List[float]] = None,
        model: Optional[str] = None,
    ) -> None:
        self._session = session
        self._target_node_id = target_node_id
        self._generate_plan = generate_plan_fn
        self._pav = pav
        self._verify_constraints = verify_constraints_fn
        self._embedding_model = embedding_model
        self._num_candidates = num_candidates
        self._temperatures = temperatures or [0.3, 0.5, 0.7][:num_candidates]
        self._model = model

    def generate_candidates(self) -> List[DecompositionCandidate]:
        """Generate distinct bridge-plan candidates with temperature diversity."""
        if self._generate_plan is None:
            return []

        graph = self._session.graph
        target_node = graph.nodes.get(self._target_node_id)
        if target_node is None:
            return []

        candidates: List[DecompositionCandidate] = []
        seen_payloads: Set[str] = set()
        retrieved_nodes = self.retrieve_relevant_subgraph(
            graph,
            target_node.statement,
        )

        for i, temp in enumerate(self._temperatures):
            try:
                plan = self._call_generate_plan(
                    graph=graph,
                    target_node_id=self._target_node_id,
                    target_statement=target_node.statement,
                    temperature=temp,
                    model=self._model,
                    retrieved_nodes=retrieved_nodes,
                )
                if plan is not None:
                    payload = plan.model_dump_json()
                    if payload in seen_payloads:
                        continue
                    seen_payloads.add(payload)
                    candidates.append(
                        DecompositionCandidate(
                            plan=plan,
                            temperature_used=temp,
                            candidate_index=i,
                        )
                    )
            except Exception:
                continue

        return candidates[: self._num_candidates]

    def score_candidates(
        self,
        candidates: List[DecompositionCandidate],
    ) -> List[ScoredDecomposition]:
        """Score candidates on virtual graph checkpoints and return best-first."""
        scored: List[ScoredDecomposition] = []
        for cand in candidates:
            scored_cand = self._score_candidate(cand)
            if scored_cand is not None:
                scored.append(scored_cand)
        scored.sort(key=lambda s: s.total_score, reverse=True)
        return scored

    def generate_and_score(self) -> Optional[ScoredDecomposition]:
        """
        Generate multiple candidate plans, score them virtually with Gaia BP,
        and return the best one (or None if generation fails entirely).
        """
        candidates = self.generate_candidates()
        if not candidates:
            return None

        scored = self.score_candidates(candidates)
        if not scored:
            return None

        return scored[0]

    def retrieve_relevant_subgraph(
        self,
        graph: HyperGraph,
        target_statement: str,
        *,
        top_k: int = 8,
    ) -> List[Any]:
        """
        Retrieve the most relevant existing nodes to condition decomposition.

        Uses Gaia's EmbeddingModel when available and falls back to a cheap
        lexical overlap scorer otherwise.
        """
        nodes = list(graph.nodes.values())
        if not nodes:
            return []

        if self._embedding_model is not None:
            try:
                target_embedding = self._embedding_model.embed([target_statement])[0]
                node_embeddings = self._embedding_model.embed([n.statement for n in nodes])
                scored = []
                for node, embedding in zip(nodes, node_embeddings):
                    dot = sum(a * b for a, b in zip(target_embedding, embedding))
                    norm_a = sum(a * a for a in target_embedding) ** 0.5
                    norm_b = sum(b * b for b in embedding) ** 0.5
                    sim = dot / max(norm_a * norm_b, 1e-8)
                    scored.append((sim, node))
                scored.sort(key=lambda item: item[0], reverse=True)
                return [node for _, node in scored[:top_k]]
            except Exception:
                pass

        target_terms = set(target_statement.lower().split())
        lexical = []
        for node in nodes:
            terms = set(node.statement.lower().split())
            overlap = len(target_terms & terms) / max(len(target_terms | terms), 1)
            lexical.append((overlap + 0.1 * node.belief, node))
        lexical.sort(key=lambda item: item[0], reverse=True)
        return [node for _, node in lexical[:top_k]]

    def _call_generate_plan(self, **kwargs: Any) -> Optional[BridgePlan]:
        """Call the injected generator with a few compatible signatures."""
        if self._generate_plan is None:
            return None
        plan = None
        try:
            plan = self._generate_plan(**kwargs)
        except TypeError:
            try:
                plan = self._generate_plan(
                    kwargs["graph"],
                    kwargs["target_node_id"],
                    temperature=kwargs.get("temperature"),
                    model=kwargs.get("model"),
                )
            except TypeError:
                plan = self._generate_plan(kwargs["graph"], kwargs["target_node_id"])
        if plan is None:
            return None
        if isinstance(plan, BridgePlan):
            return plan
        if isinstance(plan, dict):
            return BridgePlan.model_validate(plan)
        raise TypeError("generate_plan_fn must return BridgePlan or dict payload")

    def _score_candidate(
        self, cand: DecompositionCandidate
    ) -> Optional["ScoredDecomposition"]:
        """Score one candidate by materialising it in a virtual graph and running BP."""
        from dz_hypergraph.inference import propagate_beliefs
        from dz_engine.bridge import materialize_bridge_nodes

        graph = self._session.graph
        target_node = graph.nodes.get(self._target_node_id)
        if target_node is None:
            return None

        belief_before = target_node.belief

        # Checkpoint → materialise → run BP → score → rollback
        cid = self._session.checkpoint(f"spec_decomp_cand_{cand.candidate_index}")
        try:
            virtual_graph = self._session.graph
            try:
                node_map = materialize_bridge_nodes(
                    virtual_graph,
                    cand.plan,
                    default_domain=target_node.domain,
                )
                self._materialize_decomposition_edges(virtual_graph, cand.plan, node_map)
            except Exception:
                return None

            propagate_beliefs(virtual_graph)

            target_after = virtual_graph.nodes.get(self._target_node_id)
            belief_after = target_after.belief if target_after else belief_before
            belief_gain = max(0.0, belief_after - belief_before)

            grade_weights = {"A": 1.0, "B": 0.8, "C": 0.5, "D": 0.2}
            grade_quality = sum(
                grade_weights.get(p.grade, 0.2) for p in cand.plan.propositions
            ) / max(len(cand.plan.propositions), 1)

            constraint = self._check_constraints(cand.plan)

            pav_adv = 0.0
            if self._pav is not None:
                for prop in cand.plan.propositions[:3]:  # check first 3
                    matches = virtual_graph.find_node_ids_by_statement(prop.statement)
                    if matches:
                        pav_adv = max(pav_adv, self._pav.predict_advantage(
                            virtual_graph, self._target_node_id,
                            matches[0], Module.PLAUSIBLE
                        ))

            scored = ScoredDecomposition(
                candidate=cand,
                belief_gain=belief_gain,
                grade_quality=grade_quality,
                constraint=constraint,
                pav_advantage=max(0.0, pav_adv),
            )
            scored.compute_score()
            return scored

        finally:
            self._session.rollback(cid)

    def _materialize_decomposition_edges(
        self,
        graph: HyperGraph,
        plan: BridgePlan,
        node_map: Dict[str, str],
    ) -> None:
        """Add soft decomposition edges so BP can score the candidate plan."""
        existing = {
            (
                tuple(edge.premise_ids),
                edge.conclusion_id,
                edge.edge_type,
            )
            for edge in graph.edges.values()
        }
        grade_to_module = {
            "A": Module.LEAN,
            "B": Module.LEAN,
            "C": Module.EXPERIMENT,
            "D": Module.PLAUSIBLE,
        }
        grade_to_confidence = {
            "A": 0.95,
            "B": 0.8,
            "C": 0.65,
            "D": 0.55,
        }
        for prop in plan.propositions:
            if not prop.depends_on:
                continue
            premise_ids = [node_map[dep] for dep in prop.depends_on if dep in node_map]
            if not premise_ids or prop.id not in node_map:
                continue
            signature = (tuple(premise_ids), node_map[prop.id], "decomposition")
            if signature in existing:
                continue
            edge = graph.add_hyperedge(
                premise_ids=premise_ids,
                conclusion_id=node_map[prop.id],
                module=grade_to_module.get(prop.grade, Module.PLAUSIBLE),
                steps=[f"[speculative decomposition] {prop.statement}"],
                confidence=grade_to_confidence.get(prop.grade, 0.55),
                edge_type="decomposition",
            )
            existing.add((tuple(edge.premise_ids), edge.conclusion_id, edge.edge_type))

    def _check_constraints(self, plan: BridgePlan) -> DecompositionConstraint:
        """
        Lightweight VERIFY-RL style constraint checking without Lean.

        Full Lean validation would be: use TacticByTacticProver to check
        that the skeleton compiles with sorry.  For now we use heuristics.
        """
        if self._verify_constraints is not None:
            try:
                return self._verify_constraints(plan=plan, graph=self._session.graph)
            except Exception:
                pass

        props = list(plan.propositions)
        prop_map = {prop.id: prop for prop in props}
        target_props = [prop for prop in props if prop.role == "target"]
        target_dep_count = max((len(prop.depends_on) for prop in target_props), default=0)

        complexity_ok = any(
            prop.role != "seed"
            and len(prop.depends_on) < max(target_dep_count, 1)
            for prop in props
        )
        concluded = {
            conclusion
            for step in plan.chain
            for conclusion in step.concludes
        }
        target_ids = {prop.id for prop in target_props}
        solution_containment = bool(target_ids & concluded)
        derivation_valid = all(
            prop.role == "seed" or prop.id in concluded or prop.id in target_ids
            for prop in props
            if prop.id in prop_map
        )

        return DecompositionConstraint(
            complexity_decreasing=complexity_ok,
            solution_containment=solution_containment,
            derivation_valid=derivation_valid,
        )
