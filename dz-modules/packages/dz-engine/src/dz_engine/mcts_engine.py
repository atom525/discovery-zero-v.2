"""
MCTS-style iterative discovery engine built on top of Zero's existing
orchestration stack.

This engine does not replace the mature benchmark/orchestrator code paths.
Instead, it composes:
  - HTPS path selection for graph-aware leaf choice
  - SearchState / UCB / RMaxTS for action selection
  - optional retrieval, continuation verification, experiment evolution, and
    problem variants
  - ExperienceRecord collection for expert iteration
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from dz_engine.bridge import BridgePlan

from dz_hypergraph.inference import SignalAccumulator, propagate_beliefs, propagate_verification_signals
from dz_hypergraph.ingest import ingest_verified_claim
from dz_hypergraph.memo import ResearchMemo, VerificationResult
from dz_hypergraph.models import HyperGraph, Module
from dz_hypergraph.persistence import load_graph, save_graph
from dz_engine.bridge import BridgePlan, materialize_bridge_nodes
from dz_verify.claim_pipeline import ClaimPipeline
from dz_verify.claim_verifier import ClaimVerifier, VerifiableClaim
from dz_verify.continuation_verifier import ContinuationVerifier
from dz_engine.curiosity import CuriosityDrivenExplorer, NoveltyTracker
from dz_engine.decompose import DecomposeEngine
from dz_hypergraph.belief_gap import BeliefGapAnalyser
from dz_engine.experiment_evolution import ExperimentEvolver
from dz_engine.expert_iteration import ExperienceBuffer, ExperienceRecord
from dz_engine.htps import HTPSState, htps_backup, htps_select
from dz_engine.knowledge_retrieval import KnowledgeRetriever
from dz_verify.lean_feedback import LeanFeedbackParser, StructuralClaimRouter
from dz_engine.analogy import AnalogyEngine
from dz_engine.specialize import SpecializeEngine
from dz_engine.orchestrator import (
    ActionResult,
    execute_bridge_followups,
    ingest_action_output,
    run_bridge_planning_action,
    run_experiment_action,
    run_lean_action,
    run_plausible_action,
)
from dz_engine.problem_variants import ProblemVariantGenerator
from dz_engine.search import RMaxTSSearch, SearchState, rank_frontiers, select_module_ucb
from dz_hypergraph.tools.retrieval import HypergraphRetrievalIndex

logger = logging.getLogger(__name__)


@dataclass
class MCTSConfig:
    max_iterations: int = 30
    max_time_seconds: float = 14400.0
    post_action_budget_seconds: float = 300.0
    c_puct: float = 1.4
    num_simulations_per_expand: int = 3
    enable_evolutionary_experiments: bool = True
    enable_continuation_verification: bool = True
    enable_retrieval: bool = True
    enable_problem_variants: bool = True
    specialization_threshold: int = 3
    progressive_widening_base: float = 1.5
    replan_on_stuck: int = 2


@dataclass
class MCTSIterationTrace:
    iteration: int
    node_id: str
    module: str
    target_belief_before: float
    target_belief_after: float
    reward: float


@dataclass
class MCTSDiscoveryResult:
    success: bool
    iterations_completed: int
    target_belief_initial: float
    target_belief_final: float
    traces: list[MCTSIterationTrace] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    experiences: list[ExperienceRecord] = field(default_factory=list)
    best_bridge_plan: Optional[BridgePlan] = None
    best_bridge_confidence: float = float("-inf")
    best_bridge_node_map: dict[str, str] = field(default_factory=dict)
    elapsed_ms: float = 0.0


class MCTSDiscoveryEngine:
    def __init__(
        self,
        graph_path: Path,
        target_node_id: str,
        config: MCTSConfig,
        *,
        pav: Any = None,
        novelty_tracker: Optional[NoveltyTracker] = None,
        curiosity_explorer: Optional[CuriosityDrivenExplorer] = None,
        continuation_verifier: Optional[ContinuationVerifier] = None,
        experiment_evolver: Optional[ExperimentEvolver] = None,
        retrieval_index: Optional[HypergraphRetrievalIndex] = None,
        experience_buffer: Optional[ExperienceBuffer] = None,
        problem_variant_generator: Optional[ProblemVariantGenerator] = None,
        claim_verifier: Optional[ClaimVerifier] = None,
        analogy_engine: Optional[AnalogyEngine] = None,
        specialize_engine: Optional[SpecializeEngine] = None,
        decompose_engine: Optional[DecomposeEngine] = None,
        knowledge_retriever: Optional[KnowledgeRetriever] = None,
        claim_pipeline: Optional[ClaimPipeline] = None,
        lean_feedback_parser: Optional[LeanFeedbackParser] = None,
        structural_claim_router: Optional[StructuralClaimRouter] = None,
        signal_accumulator: Optional[SignalAccumulator] = None,
        lean_policy: Optional[dict[str, Any]] = None,
        model: Optional[str] = None,
        backend: str = "bp",
        llm_record_dir: Optional[Path] = None,
        bridge_path: Optional[Path] = None,
        lean_timeout: Optional[int] = None,
    ) -> None:
        self.graph_path = graph_path
        self.target_node_id = target_node_id
        self.config = config
        self.model = model
        self.backend = backend
        self.llm_record_dir = llm_record_dir
        self.bridge_path = bridge_path
        self.search_state = SearchState()
        self.htps_state = HTPSState()
        self.novelty_tracker = novelty_tracker or NoveltyTracker()
        self.curiosity = curiosity_explorer
        self.continuation_verifier = continuation_verifier
        self.experiment_evolver = experiment_evolver
        self.retrieval_index = retrieval_index
        self.experience_buffer = experience_buffer
        self.problem_variant_generator = problem_variant_generator
        self.claim_verifier = claim_verifier
        self.analogy_engine = analogy_engine
        self.specialize_engine = specialize_engine
        self.decompose_engine = decompose_engine
        self.knowledge_retriever = knowledge_retriever
        self.claim_pipeline = claim_pipeline
        self.lean_feedback_parser = lean_feedback_parser
        self.structural_claim_router = structural_claim_router
        from dz_hypergraph.config import CONFIG as _cfg
        self.signal_accumulator = signal_accumulator or SignalAccumulator(
            threshold=max(1, int(getattr(_cfg, "bp_propagation_threshold", 1)))
        )
        self.lean_policy: dict[str, Any] = lean_policy or {}
        self._lean_timeout = int(
            lean_timeout
            if lean_timeout is not None
            else getattr(_cfg, "lean_timeout", 300)
        )
        self._rmaxts = RMaxTSSearch(
            pav=pav,
            novelty_tracker=self.novelty_tracker,
            curiosity=self.curiosity,
        )
        self._gap_analyser = BeliefGapAnalyser()
        # Per-run counter for naming verification-related LLM records uniquely.
        self._verification_run_counter = 0
        # Global stall trackers used by unified action override logic.
        self._belief_stall_count = 0
        self._plausible_stall_cycles = 0
        # Track module outcomes across nodes to suppress globally failing modes.
        self._recent_module_history: list[tuple[Module, bool]] = []
        # Consecutive iterations where target node has no incoming support edges.
        self._target_isolation_streak: int = 0

    def run(
        self,
        *,
        planning_feedback: str = "",
        boundary_policy: Optional[dict[str, Any]] = None,
        on_iteration_complete: Optional[Callable[[int, "MCTSDiscoveryResult"], None]] = None,
    ) -> MCTSDiscoveryResult:
        """Run the MCTS discovery loop.

        Args:
            planning_feedback: Seed feedback injected into every LLM prompt.
            boundary_policy: Optional Lean boundary policy dict.
            on_iteration_complete: Optional callback invoked after each MCTS
                iteration completes (success or failure). Receives the iteration
                number (1-based) and the current result snapshot. Use this to
                flush incremental logs to disk.
        """
        t0 = time.monotonic()
        graph = load_graph(self.graph_path)
        if self.target_node_id not in graph.nodes:
            return MCTSDiscoveryResult(
                success=False,
                iterations_completed=0,
                target_belief_initial=0.0,
                target_belief_final=0.0,
            )

        target_belief_initial = float(graph.nodes[self.target_node_id].belief)
        result = MCTSDiscoveryResult(
            success=False,
            iterations_completed=0,
            target_belief_initial=target_belief_initial,
            target_belief_final=target_belief_initial,
        )
        stuck_rounds = 0
        plausible_attempts = 0
        experiment_attempts = 0
        lean_attempts = 0
        rolling_feedback = planning_feedback
        iter_budget_seconds = float(os.environ.get("DISCOVERY_ZERO_MCTS_ITER_BUDGET", "0").strip() or "0")
        if iter_budget_seconds <= 0:
            per_iter = self.config.max_time_seconds / max(self.config.max_iterations, 1)
            iter_budget_seconds = min(
                max(1800.0, per_iter * 2.0),
                self.config.max_time_seconds * 0.5,
            )
        post_action_budget_seconds = float(self.config.post_action_budget_seconds)
        checkpoint_dir = self.graph_path.parent / "action_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for iteration in range(1, self.config.max_iterations + 1):
            iter_start = time.monotonic()
            steps_before_iteration = len(result.steps)
            lean_feedback_iteration = ""
            if time.monotonic() - t0 > self.config.max_time_seconds:
                break
            graph = load_graph(self.graph_path)
            target_node = graph.nodes.get(self.target_node_id)
            if target_node is None:
                break
            if target_node.state in {"proven", "refuted"}:
                result.success = target_node.state == "proven"
                break
            target_incoming_edges = graph.get_edges_to(self.target_node_id)
            if target_incoming_edges:
                self._target_isolation_streak = 0
            else:
                self._target_isolation_streak += 1

            retrieval_context = ""
            if self.config.enable_retrieval and self.retrieval_index is not None:
                self.retrieval_index.build_from_graph(graph)

            selected_node_id, selected_module, selected_path = self._select_action(graph)
            if self._target_isolation_streak > 2:
                selected_node_id = self.target_node_id
                selected_module = Module.PLAUSIBLE
                selected_path = [self.target_node_id]
                result.steps.append(
                    {
                        "phase": "target_isolation_recovery",
                        "iteration": iteration,
                        "message": (
                            "Target node has remained isolated for more than two "
                            "iterations; forcing PLAUSIBLE action on root goal."
                        ),
                        "target_node_id": self.target_node_id,
                    }
                )
            if selected_node_id not in graph.nodes:
                result.steps.append(
                    {
                        "phase": "selection_mcts",
                        "error": f"Selected node '{selected_node_id}' not found in graph.",
                    }
                )
                result.iterations_completed = iteration
                stuck_rounds += 1
                continue

            selected_node = graph.nodes[selected_node_id]
            if self.config.enable_retrieval and self.retrieval_index is not None:
                retrieved = self.retrieval_index.retrieve(
                    f"{target_node.statement}\n{selected_node.statement}",
                    graph=graph,
                    target_node_id=self.target_node_id,
                    exclude_node_ids={selected_node_id},
                )
                retrieval_context = self.retrieval_index.format_retrieval_context(retrieved, graph)

            combined_feedback = "\n\n".join(
                part for part in (rolling_feedback, retrieval_context) if part
            )
            target_belief_before = float(target_node.belief)

            try:
                action_result = self._execute_selected_action(
                    graph=graph,
                    node_id=selected_node_id,
                    module=selected_module,
                    boundary_policy=boundary_policy,
                    feedback=combined_feedback,
                )
            except Exception as exc:
                logger.exception(
                    "Iteration %d action execution failed for node=%s module=%s",
                    iteration,
                    selected_node_id,
                    selected_module.value,
                )
                result.steps.append(
                    {
                        "phase": "iteration_error",
                        "iteration": iteration,
                        "node_id": selected_node_id,
                        "action_module": selected_module.value,
                        "error": str(exc),
                    }
                )
                result.iterations_completed = iteration
                stuck_rounds += 1
                try:
                    save_graph(graph, self.graph_path)
                except Exception as save_exc:
                    logger.warning("Failed to save graph after iteration error: %s", save_exc)
                if on_iteration_complete is not None:
                    on_iteration_complete(iteration, result)
                continue
            try:
                checkpoint_path = checkpoint_dir / f"iter_{iteration:04d}_action_result.json"
                checkpoint_payload = {
                    "iteration": iteration,
                    "node_id": selected_node_id,
                    "module": selected_module.value,
                    "created_at": time.time(),
                    "result": asdict(action_result),
                }
                checkpoint_path.write_text(
                    json.dumps(checkpoint_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.warning("Failed to checkpoint action result at iteration %d: %s", iteration, exc)

            elapsed_after_action = time.monotonic() - iter_start
            remaining_budget = iter_budget_seconds - elapsed_after_action
            skip_optional_followups = remaining_budget < post_action_budget_seconds
            if (
                selected_module == Module.PLAUSIBLE
                and result.best_bridge_plan is None
            ):
                skip_optional_followups = False
            if skip_optional_followups:
                result.steps.append(
                    {
                        "phase": "iteration_budget_warning",
                        "iteration": iteration,
                        "node_id": selected_node_id,
                        "action_module": selected_module.value,
                        "message": (
                            f"Remaining budget {max(0.0, remaining_budget):.1f}s < post-action "
                            f"critical budget {post_action_budget_seconds:.1f}s; skipping optional followups."
                        ),
                    }
                )
            if not action_result.success:
                self._record_recent_module_outcome(selected_module, False)
                self.search_state.record_action(
                    selected_node_id,
                    selected_module,
                    0.0,
                    success=False,
                    error_type=action_result.message[:80],
                )
                fail_step = {
                    "phase": f"{selected_module.value}_mcts",
                    "error": action_result.message,
                    "node_id": selected_node_id,
                    "iteration": iteration,
                    "belief_before": target_belief_before,
                    "belief_after": target_belief_before,  # unchanged on failure
                    "belief_delta": 0.0,
                    "node_beliefs": {
                        nid: round(float(nd.belief), 6)
                        for nid, nd in graph.nodes.items()
                    },
                }
                result.steps.append(fail_step)
                result.iterations_completed = iteration
                stuck_rounds += 1
                if on_iteration_complete is not None:
                    on_iteration_complete(iteration, result)
                continue

            verify_bonus: float = 0.0
            try:
                if action_result.normalized_output is not None:
                    # Only force the conclusion to map to the MCTS target node when
                    # the action was explicitly targeting that node.  Passing
                    # target_node_id unconditionally would hijack conclusions for
                    # actions on bridge propositions (whose conclusions should be
                    # their own nodes, not the overall goal).
                    action_targets_mcts_goal = (
                        getattr(action_result, "target_node_id", "") == self.target_node_id
                    )
                    action_result = ingest_action_output(
                        self.graph_path,
                        action_result,
                        backend=self.backend,
                        target_node_id=self.target_node_id if action_targets_mcts_goal else None,
                    )
                    # For PLAUSIBLE actions: first generate the bridge plan so that
                    # bridge nodes are materialised into the graph BEFORE claim
                    # verification runs.  This lets the claim pipeline map claims
                    # to bridge propositions by ID (via bridge_node_map), which
                    # avoids text-matching and ensures verification results flow
                    # to connected graph nodes.
                    if selected_module == Module.PLAUSIBLE and not skip_optional_followups:
                        plausible_attempts += 1
                        graph = load_graph(self.graph_path)
                        self._handle_plausible_followups(
                            graph=graph,
                            action_result=action_result,
                            feedback=combined_feedback,
                            result=result,
                        )
                        # Now run verification pipeline with bridge plan context.
                        self._verification_run_counter += 1
                        verification_steps, lean_feedback, verify_bonus = self._run_verification_pipeline(
                            action_result=action_result,
                            combined_feedback=combined_feedback,
                            bridge_plan=result.best_bridge_plan,
                            bridge_node_map=result.best_bridge_node_map,
                        )
                        result.steps.extend(verification_steps)
                        lean_feedback_iteration = lean_feedback
                        if lean_feedback:
                            combined_feedback = "\n\n".join(
                                part for part in (combined_feedback, lean_feedback) if part
                            )
                    elif selected_module == Module.PLAUSIBLE and skip_optional_followups:
                        result.steps.append(
                            {
                                "phase": "plausible_followups_skipped",
                                "iteration": iteration,
                                "node_id": selected_node_id,
                                "reason": "insufficient_followup_budget_after_action",
                            }
                        )
            except Exception as exc:
                logger.exception(
                    "Iteration %d post-action processing failed for node=%s module=%s",
                    iteration,
                    selected_node_id,
                    selected_module.value,
                )
                result.steps.append(
                    {
                        "phase": "iteration_error",
                        "iteration": iteration,
                        "node_id": selected_node_id,
                        "action_module": selected_module.value,
                        "error": str(exc),
                    }
                )
                result.iterations_completed = iteration
                stuck_rounds += 1
                try:
                    save_graph(load_graph(self.graph_path), self.graph_path)
                except Exception as save_exc:
                    logger.warning("Failed to persist graph after post-action error: %s", save_exc)
                if on_iteration_complete is not None:
                    on_iteration_complete(iteration, result)
                continue
            graph = load_graph(self.graph_path)
            updated_target = graph.nodes.get(self.target_node_id)
            target_belief_after = float(updated_target.belief) if updated_target is not None else target_belief_before
            exploration_reward = self._action_exploration_reward(action_result)
            if selected_module in (Module.EXPERIMENT, Module.LEAN):
                delta = max(0.0, target_belief_after - target_belief_before)
                dampener = max(0.1, min(1.0, delta * 10.0))
                exploration_reward *= dampener
            verification_reward = self._compute_verification_reward(
                action_result=action_result,
                target_belief_before=target_belief_before,
                target_belief_after=target_belief_after,
            )
            reward = verification_reward + exploration_reward + verify_bonus

            result.steps.append(
                {
                    "phase": self._phase_name(
                        selected_module,
                        plausible_attempts=plausible_attempts,
                        experiment_attempts=experiment_attempts,
                        lean_attempts=lean_attempts,
                    ),
                    "iteration": iteration,
                    "belief_before": round(target_belief_before, 6),
                    "belief_after": round(target_belief_after, 6),
                    "belief_delta": round(target_belief_after - target_belief_before, 6),
                    "reward": round(reward, 6),
                    "node_count": len(graph.nodes),
                    "raw": (action_result.raw_output or "")[:500],
                    "normalized": None,
                    "judge": action_result.judge_output,
                    "edge_id": action_result.ingest_edge_id,
                    "message": action_result.message,
                    "target_node_id": selected_node_id,
                    "action_module": selected_module.value,
                    "created_node_ids": getattr(action_result, "created_node_ids", []) or [],
                }
            )

            # Track module selection for diversity enforcement.
            self.search_state.record_selection(selected_node_id, selected_module)

            # Record action outcome so FailureMemory can penalise modules that
            # keep failing on the same node.  Without this call, penalty_factor
            # in select_module_ucb always stays at 1.0 regardless of consecutive
            # failures, making it impossible for UCB to suppress LEAN (or any
            # other repeatedly-failing module) automatically.
            error_hint = ""
            if not action_result.success:
                error_hint = str(action_result.message or "")[:80]
            self.search_state.record_action(
                selected_node_id,
                selected_module,
                reward,
                success=action_result.success,
                error_type=error_hint,
            )
            self._record_recent_module_outcome(selected_module, action_result.success)

            # PLAUSIBLE followups (bridge plan + verification) are handled
            # above inside the normalized_output block so bridge nodes are
            # available before claim verification runs.  We only track
            # attempt counts for non-PLAUSIBLE modules here.
            if selected_module == Module.EXPERIMENT:
                experiment_attempts += 1
            elif selected_module == Module.LEAN:
                lean_attempts += 1

            novelty = self.novelty_tracker.compute_novelty(action_result, graph)
            surprise = 0.0
            intrinsic = self._rmaxts.compute_intrinsic_reward(
                target_belief_before,
                target_belief_after,
                min(1.0, novelty + exploration_reward),
                surprise,
            )
            htps_backup(self.htps_state, selected_path, target_belief_after)

            experience = ExperienceRecord(
                graph_snapshot_json="",
                target_node_id=self.target_node_id,
                action_node_id=selected_node_id,
                action_module=selected_module.value,
                intrinsic_reward=intrinsic,
                belief_delta=target_belief_after - target_belief_before,
                success=True,
                bridge_plan_valid=result.best_bridge_plan is not None,
                next_graph_snapshot_json="",
                run_id=str(self.graph_path.parent),
            )
            result.experiences.append(experience)
            if self.experience_buffer is not None:
                self.experience_buffer.add(experience)

            result.traces.append(
                MCTSIterationTrace(
                    iteration=iteration,
                    node_id=selected_node_id,
                    module=selected_module.value,
                    target_belief_before=target_belief_before,
                    target_belief_after=target_belief_after,
                    reward=reward,
                )
            )
            result.iterations_completed = iteration
            result.target_belief_final = target_belief_after

            delta_belief = abs(target_belief_after - target_belief_before)
            if delta_belief < 1e-5:
                if selected_module in (Module.EXPERIMENT, Module.LEAN):
                    self._belief_stall_count += 1
            elif delta_belief > 0.01:
                self._belief_stall_count = 0
                self._plausible_stall_cycles = 0

            if reward <= 1e-9:
                stuck_rounds += 1
            else:
                stuck_rounds = 0

            if (
                self.config.enable_problem_variants
                and self.problem_variant_generator is not None
                and stuck_rounds >= self.config.specialization_threshold
            ):
                created_variant_ids = self._spawn_problem_variants(graph, selected_node_id, result)
                if created_variant_ids:
                    stuck_rounds = 0

            # Notify caller that this iteration is complete (enables incremental log flushing).
            if on_iteration_complete is not None:
                on_iteration_complete(iteration, result)

            if updated_target is not None and updated_target.state == "proven":
                result.success = True
                break

            verified_facts: list[str] = []
            seen_facts: set[str] = set()

            def _append_fact(tag: str, text: str, *, limit: int = 180) -> None:
                cleaned = " ".join(str(text or "").split())
                if not cleaned:
                    return
                fact = f"- [{tag}] {cleaned[:limit]}"
                if fact not in seen_facts:
                    seen_facts.add(fact)
                    verified_facts.append(fact)

            for _, node in graph.nodes.items():
                if node.state == "proven" or (node.belief >= 0.8 and node.provenance == "experiment"):
                    _append_fact("VERIFIED", f"{node.statement} (belief={node.belief:.3f})")
                elif node.state == "refuted":
                    _append_fact("REFUTED", node.statement)

            if result.best_bridge_plan is not None and result.best_bridge_plan.summary:
                _append_fact("PROOF STRATEGY", result.best_bridge_plan.summary, limit=220)

            recent_steps = result.steps[max(0, steps_before_iteration - 24):]
            for step in recent_steps:
                judge = step.get("judge") if isinstance(step, dict) else None
                if isinstance(judge, dict):
                    concerns = judge.get("concerns", [])
                    if isinstance(concerns, list):
                        for concern in concerns[:2]:
                            _append_fact("JUDGE CONCERN", str(concern), limit=160)

                if not isinstance(step, dict):
                    continue
                if step.get("phase") != "claim_verification":
                    continue

                summary = step.get("summary")
                if isinstance(summary, str):
                    for part in summary.split(";"):
                        if "refuted" in part.casefold():
                            _append_fact("CLAIM REFUTED", part.strip(), limit=170)

                if int(step.get("lean_gaps_identified", 0) or 0) > 0 and lean_feedback_iteration:
                    lean_lines = [ln.strip() for ln in lean_feedback_iteration.splitlines() if ln.strip()]
                    for line in lean_lines[:2]:
                        _append_fact("LEAN DISCOVERY", line, limit=170)
                if step.get("phase") == "bridge_plan_mcts" and step.get("error"):
                    _append_fact("BRIDGE PLAN FAILED", step["error"], limit=170)

            if verified_facts:
                rolling_feedback = (
                    "Verified results from previous iterations:\n"
                    + "\n".join(verified_facts[:28])
                )
                if planning_feedback:
                    rolling_feedback = rolling_feedback + "\n\n" + planning_feedback

        result.elapsed_ms = (time.monotonic() - t0) * 1000
        graph = load_graph(self.graph_path)
        if self.target_node_id in graph.nodes:
            result.target_belief_final = float(graph.nodes[self.target_node_id].belief)
            result.success = result.success or graph.nodes[self.target_node_id].state == "proven"
        return result

    def _action_exploration_reward(self, action_result: ActionResult) -> float:
        normalized = action_result.normalized_output or {}
        reward = 0.0
        created_nodes = len(getattr(action_result, "created_node_ids", []) or [])
        reward += min(0.25, 0.08 * created_nodes)
        premises = normalized.get("premises", []) if isinstance(normalized, dict) else []
        new_premises = 0
        for item in premises:
            if not isinstance(item, dict):
                continue
            pid = item.get("id")
            if pid in (None, "", "existing_node_id", "null", "none"):
                new_premises += 1
        reward += min(0.25, 0.1 * new_premises)
        steps_text = " ".join(str(step) for step in normalized.get("steps", []))
        lowered = steps_text.casefold()
        if any(
            keyword in lowered
            for keyword in (
                "new method",
                "new mechanism",
                "construct",
                "reduction",
                "obstruction",
                "certificate",
                "ansatz",
                "hypothesis",
            )
        ):
            reward += 0.2
        # Experiment gets no unconditional bonus here; its reward is determined
        # in _compute_verification_reward based on whether the target is on the
        # bridge reasoning chain.
        return min(0.6, reward)

    def _compute_verification_reward(
        self,
        *,
        action_result: ActionResult,
        target_belief_before: float,
        target_belief_after: float,
    ) -> float:
        """Reward grounded on explicit verification outcomes (not raw belief delta)."""
        normalized = action_result.normalized_output or {}
        judge = action_result.judge_output or {}
        reward = 0.0

        outcome = str(normalized.get("outcome", "")).strip().lower()
        if outcome == "verified":
            reward += 0.4
        elif outcome == "refuted":
            reward += 0.2
        elif outcome == "inconclusive":
            reward += 0.0

        if action_result.selected_module == Module.LEAN.value:
            if action_result.success:
                reward += 0.35
            else:
                reward += 0.05
        elif action_result.selected_module == Module.EXPERIMENT.value and action_result.success:
            # Only give the full experiment reward when the target node participates
            # in the bridge reasoning chain (has outgoing edges toward the target).
            # Generic numerical sampling on open problems should not receive the
            # same reward as targeted verification of a bridge proposition.
            target_nid = getattr(action_result, "target_node_id", "") or ""
            node_on_bridge = False
            if target_nid:
                try:
                    _g = load_graph(self.graph_path)
                    node_on_bridge = any(
                        target_nid in e.premise_ids for e in _g.edges.values()
                    )
                except Exception as exc:
                    logger.warning(
                        "Verification future failed in run %s: %s",
                        self._verification_run_counter,
                        exc,
                    )
            if node_on_bridge:
                reward += 0.25  # bridge proposition experiment: full reward
            else:
                reward += 0.05  # generic open-problem sampling: minimal reward
        elif action_result.selected_module == Module.PLAUSIBLE.value:
            reward += 0.1 * max(0.0, min(1.0, float(judge.get("confidence", 0.0))))

        # Small fallback signal for monotonic progress if explicit verdict is absent.
        if outcome not in {"verified", "refuted"}:
            reward += max(0.0, target_belief_after - target_belief_before) * 0.15

        return min(1.0, reward)

    def _select_action(self, graph: HyperGraph) -> tuple[str, Module, list[tuple[str, str]]]:
        # When there are no edges in the graph yet, no reasoning path has been
        # established.  Always start with plausible reasoning so that the LLM
        # can build a proof route before experiments or Lean are attempted.
        if not graph.edges:
            return self.target_node_id, Module.PLAUSIBLE, []

        target_node = graph.nodes.get(self.target_node_id)
        target_belief = float(target_node.belief) if target_node is not None else 0.0
        node_id_for_ucb, module, path = self._default_select(graph)
        if self._is_module_globally_failing(module):
            if module != Module.PLAUSIBLE:
                module = Module.PLAUSIBLE
                node_id_for_ucb = self.target_node_id
            else:
                module = Module.DECOMPOSE
                node_id_for_ucb = self.target_node_id
        consecutive = self.search_state.get_consecutive_same_count(
            node_id_for_ucb, module
        )

        # Unified override logic (loop-breaker + global stall detector).
        if module in (Module.EXPERIMENT, Module.LEAN):
            if consecutive >= 3 or self._belief_stall_count >= 4:
                live_routes = self._count_live_plausible_routes_to_target(graph)
                if target_belief < 0.5 and live_routes < 5:
                    if self._plausible_stall_cycles < 2:
                        module = Module.PLAUSIBLE
                    elif self._plausible_stall_cycles < 4:
                        module = Module.DECOMPOSE
                    else:
                        module = Module.ANALOGY
                    node_id_for_ucb = self.target_node_id
                    if self._belief_stall_count >= 4:
                        self._belief_stall_count = 0
                        self._plausible_stall_cycles += 1
        elif module == Module.PLAUSIBLE and consecutive >= 3:
            critical = self._gap_analyser.find_critical_gaps(
                graph,
                self.target_node_id,
                top_k=3,
                search_state=self.search_state,
            )
            for crit_id, _ in critical:
                if crit_id in graph.nodes and not graph.nodes[crit_id].is_locked():
                    crit_module = self._module_for_claim_type(
                        graph.nodes[crit_id].statement
                    )
                    return crit_id, crit_module, path
        return node_id_for_ucb, module, path

    def _record_recent_module_outcome(self, module: Module, success: bool) -> None:
        self._recent_module_history.append((module, bool(success)))
        if len(self._recent_module_history) > 10:
            self._recent_module_history.pop(0)

    def _is_module_globally_failing(self, module: Module) -> bool:
        if len(self._recent_module_history) < 8:
            return False
        recent_window = self._recent_module_history[-8:]
        module_attempts = sum(1 for m, _ in recent_window if m == module)
        if module_attempts < 5:
            return False
        module_failures = sum(1 for m, ok in recent_window if m == module and not ok)
        return module_failures >= 5

    def _default_select(
        self, graph: HyperGraph
    ) -> tuple[str, Module, list[tuple[str, str]]]:
        """Standard selection logic: critical-gaps → frontiers → UCB fallback."""
        # First try critical-gaps routing for verification-centric exploration.
        critical = self._gap_analyser.find_critical_gaps(
            graph,
            self.target_node_id,
            top_k=5,
            search_state=self.search_state,
        )
        for node_id, _gain in critical:
            if node_id not in graph.nodes:
                continue
            node = graph.nodes[node_id]
            if node.is_locked():
                continue
            inferred_type = self._infer_claim_type(node.statement)
            if inferred_type == "quantitative":
                return node_id, Module.EXPERIMENT, []
            if inferred_type == "structural":
                # Do NOT hard-route to LEAN here: structural claims often live on
                # frontier open problems where Lean always fails.  Let UCB decide
                # based on the actual reward/failure history for this node so that
                # FailureMemory can suppress LEAN after repeated failures and the
                # system can fall back to PLAUSIBLE for further exploration.
                module = select_module_ucb(graph, node_id, self.search_state)
                return node_id, module, []
            return node_id, Module.PLAUSIBLE, []

        leaf_id, path = htps_select(
            graph,
            self.htps_state,
            self.target_node_id,
            c_puct=self.config.c_puct,
            search_state=self.search_state,
        )
        frontiers = rank_frontiers(
            graph,
            self.search_state,
            self.target_node_id,
            max_frontiers=max(3, self.config.num_simulations_per_expand),
        )
        if frontiers:
            selected = self._rmaxts.select_action(
                graph,
                self.target_node_id,
                frontiers,
                self.search_state,
            )
            if selected is not None:
                return selected[0], selected[1], path
        if leaf_id in graph.nodes and not graph.nodes[leaf_id].is_locked():
            module = self._module_for_claim_type(graph.nodes[leaf_id].statement)
            return leaf_id, module, path
        target_module = self._module_for_claim_type(graph.nodes[self.target_node_id].statement)
        return self.target_node_id, target_module, path

    def _count_live_plausible_routes_to_target(self, graph: HyperGraph) -> int:
        live_routes = 0
        for edge in graph.edges.values():
            edge_module = edge.module.value if isinstance(edge.module, Module) else str(edge.module)
            if edge_module != Module.PLAUSIBLE.value:
                continue
            if edge.conclusion_id != self.target_node_id:
                continue
            if self._is_bridge_dependency_edge(edge):
                continue
            if any(
                premise_id in graph.nodes
                and graph.nodes[premise_id].state != "refuted"
                for premise_id in edge.premise_ids
            ):
                live_routes += 1
        return live_routes

    @staticmethod
    def _is_bridge_dependency_edge(edge: Any) -> bool:
        joined_steps = " ".join(str(s) for s in (edge.steps or []))
        return "bridge dependency" in joined_steps.casefold()

    @staticmethod
    def _infer_claim_type(statement: str) -> str:
        lowered = statement.casefold()
        # Structural keywords take priority: theorem/conjecture statements contain
        # digits and operators but are not quantitative experiments.
        if any(
            token in lowered
            for token in (
                "conjecture", "theorem", "lemma", "holds for", "for all", "forall",
                "there exists", "exists", "if ", " then ", "implies",
            )
        ):
            return "structural"
        # Only classify as quantitative when statement is purely computational
        # (digits/operators present and no structural connectives).
        if any(ch.isdigit() for ch in statement) or any(op in statement for op in ("=", "<", ">", "≤", "≥")):
            return "quantitative"
        return "heuristic"

    def _module_for_claim_type(self, statement: str) -> Module:
        claim_type = self._infer_claim_type(statement)
        if claim_type == "quantitative":
            return Module.EXPERIMENT
        if claim_type == "structural":
            return Module.LEAN
        return Module.PLAUSIBLE

    def _run_verification_pipeline(
        self,
        *,
        action_result: ActionResult,
        combined_feedback: str,
        bridge_plan: Optional["BridgePlan"] = None,
        bridge_node_map: Optional[dict[str, str]] = None,
    ) -> tuple[list[dict[str, Any]], str, float]:
        """Run the full verification pipeline after a PLAUSIBLE action.

        This method should be called AFTER _handle_plausible_followups so that
        bridge proposition nodes are already present in the graph.  The bridge
        plan and node map are passed in so that extracted claims can be matched
        to specific bridge propositions by ID — no text-matching required.

        Returns:
            A 3-tuple of:
            - list of step dicts to append to result.steps
            - lean_feedback_str for injection into the next iteration
            - verification_bonus to add to the MCTS reward signal
        """
        from dz_hypergraph.config import CONFIG

        steps: list[dict[str, Any]] = []
        lean_feedback_parts: list[str] = []
        run_idx = self._verification_run_counter

        # ---- 1. Extract claims ----
        prose = action_result.raw_output or ""
        context = "\n\n".join(
            part
            for part in (combined_feedback, str(action_result.normalized_output or ""))
            if part
        )
        memo_id = f"memo_mcts_{run_idx}"

        if self.claim_pipeline is not None:
            try:
                claims = self.claim_pipeline.extract_claims(
                    prose=prose,
                    context=context,
                    source_memo_id=memo_id,
                    model=self.model,
                    record_dir=self.llm_record_dir,
                    bridge_plan=bridge_plan,
                )
                claims = self.claim_pipeline.prioritize_claims(
                    claims=claims,
                    graph=load_graph(self.graph_path),
                    target_node_id=self.target_node_id,
                )
            except Exception as exc:
                steps.append({"phase": "claim_extraction_error", "error": str(exc)})
                return steps, "", 0.0
        elif self.claim_verifier is not None:
            # Fallback: use ClaimVerifier.extract_claims (old code path).
            # These claims cannot have bridge_proposition_id.
            raw_claims = self.claim_verifier.extract_claims(action_result.normalized_output or {})
            from dz_hypergraph.memo import Claim, ClaimType, VerificationStatus
            claims = [
                Claim(
                    claim_text=c.claim_text,
                    claim_type=ClaimType(c.claim_type),
                    verification_status=VerificationStatus.PENDING,
                    source_memo_id=memo_id,
                    confidence=0.5,
                )
                for c in raw_claims
            ]
        else:
            return steps, "", 0.0

        if not claims:
            steps.append({"phase": "claim_verification", "claims_extracted": 0, "message": "no claims extracted"})
            return steps, "", 0.0

        enable_decomposition = bool(self.lean_policy.get("enable_decomposition", False))
        # Only run real Lean on structural claims when explicitly enabled by lean_policy
        # AND the bridge confidence is high enough.  For frontier open problems where the
        # target is still unverified, avoid burning Lean time on intermediate claims.
        enable_lean_claim_verify = bool(self.lean_policy.get("enable_strict_lean", True))

        # ---- 2. Parallel verification ----
        verification_results: list[VerificationResult] = []
        old_style_results = []  # ClaimVerificationResult for legacy update_graph_beliefs
        decomposed_subclaims: list[tuple[str, Any]] = []

        max_workers = CONFIG.verification_parallel_workers

        def _verify_one(claim: Any) -> tuple[Any, Any]:
            """Returns (VerificationResult | None, ClaimVerificationResult | None)."""
            from dz_hypergraph.memo import VerificationResult as VR
            try:
                if claim.claim_type.value == "structural":
                    if (
                        self.structural_claim_router is not None
                        and enable_lean_claim_verify
                    ):
                        plan = self.structural_claim_router.route_structural_claim(
                            claim, depth=claim.depth
                        )
                        if plan.mode == "decompose":
                            subs = self.structural_claim_router.decompose_to_subclaims(
                                claim,
                                source_memo_id=memo_id,
                                model=self.model,
                                record_dir=self.llm_record_dir,
                                decompose_index=run_idx,
                            )
                            vr = VR(
                                claim_id=claim.id,
                                verdict="inconclusive",
                                evidence_text=f"Decomposed into {len(subs)} subclaims.",
                                backend="lean_decompose",
                            )
                            return vr, subs
                        else:
                            ok, error_msg = self.structural_claim_router.verify_structural_claim(
                                claim,
                                model=self.model,
                                context=context,
                                record_dir=self.llm_record_dir,
                                claim_index=run_idx,
                            )
                            verdict = "verified" if ok else "inconclusive"
                            vr = VR(
                                claim_id=claim.id,
                                verdict=verdict,
                                evidence_text=error_msg or ("Lean verification succeeded." if ok else "Lean verification failed."),
                                lean_error="" if ok else error_msg,
                                backend="lean",
                            )
                            return vr, None
                    elif self.claim_verifier is not None:
                        vc = VerifiableClaim(
                            claim_text=claim.claim_text,
                            source_prop_id=None,
                            quantitative=False,
                            claim_type="structural",
                        )
                        cvr_list = self.claim_verifier.verify_claims(
                            claims=[vc],
                            context=context,
                            model=self.model,
                            record_dir=self.llm_record_dir,
                        )
                        if cvr_list:
                            cvr = cvr_list[0]
                            vr = VR(
                                claim_id=claim.id,
                                verdict=cvr.verdict,
                                evidence_text=cvr.evidence,
                                lean_error=cvr.code if cvr.verdict != "verified" else "",
                                backend="lean",
                            )
                            return vr, cvr
                    return None, None

                elif claim.claim_type.value == "quantitative":
                    if self.claim_verifier is not None:
                        vc = VerifiableClaim(
                            claim_text=claim.claim_text,
                            source_prop_id=None,
                            quantitative=True,
                            claim_type="quantitative",
                        )
                        cvr_list = self.claim_verifier.verify_claims(
                            claims=[vc],
                            context=context,
                            model=self.model,
                            record_dir=self.llm_record_dir,
                        )
                        if cvr_list:
                            cvr = cvr_list[0]
                            vr = VR(
                                claim_id=claim.id,
                                verdict=cvr.verdict,
                                evidence_text=cvr.evidence,
                                backend="experiment",
                            )
                            return vr, cvr
                    return None, None

                else:  # heuristic
                    if self.claim_verifier is not None:
                        vc = VerifiableClaim(
                            claim_text=claim.claim_text,
                            source_prop_id=None,
                            quantitative=False,
                            claim_type="heuristic",
                        )
                        cvr_list = self.claim_verifier.verify_claims(
                            claims=[vc],
                            context=context,
                            model=self.model,
                            record_dir=self.llm_record_dir,
                        )
                        if cvr_list:
                            cvr = cvr_list[0]
                            vr = VR(
                                claim_id=claim.id,
                                verdict=cvr.verdict,
                                evidence_text=cvr.evidence,
                                backend="llm_judge",
                            )
                            return vr, cvr
                    return None, None
            except Exception as exc:
                from dz_hypergraph.memo import VerificationResult as VR2
                vr = VR2(
                    claim_id=claim.id,
                    verdict="inconclusive",
                    evidence_text=f"verification error: {exc}",
                    backend="error",
                )
                return vr, None

        # claims_by_id maps internal claim ID → Claim for lookups after parallel verify.
        claims_by_id: dict[str, Any] = {c.id: c for c in claims}

        per_claim_timeout = float(self._lean_timeout) + 60.0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_verify_one, claim): claim for claim in claims}
            for fut in concurrent.futures.as_completed(futures, timeout=per_claim_timeout * len(claims) + 120):
                try:
                    vr, extra = fut.result(timeout=per_claim_timeout)
                    if vr is not None:
                        verification_results.append(vr)
                        if isinstance(extra, list):
                            decomposed_subclaims.extend((vr.claim_id, sub) for sub in extra)
                        elif extra is not None:
                            # Fix 7: only collect ClaimVerificationResult into
                            # old_style_results when the claim has NO bridge mapping.
                            # If a bridge_proposition_id is present, the write-back
                            # below handles the node update precisely by ID.
                            orig_claim = claims_by_id.get(vr.claim_id)
                            if orig_claim is None or not orig_claim.bridge_proposition_id:
                                old_style_results.append(extra)
                except Exception:
                    pass

        # ---- 3. Ingest verified claims into graph ----
        # Build a mapping from VR object identity to the original Claim BEFORE
        # vr.claim_id is overwritten with the real graph node ID.  This mapping
        # is used later for Lean feedback collection and bonus computation.
        vr_to_original_claim: dict[int, Any] = {
            id(vr): claims_by_id[vr.claim_id]
            for vr in verification_results
            if vr.claim_id in claims_by_id
        }

        graph_updated = load_graph(self.graph_path)
        ingested = 0
        for vr in verification_results:
            if vr.verdict not in ("verified", "refuted"):
                continue
            claim = claims_by_id.get(vr.claim_id)
            if claim is None:
                continue
            try:
                # Fix 2 write-back: if the claim was mapped to a bridge proposition,
                # resolve the exact graph node ID via bridge_node_map.  This avoids
                # text-matching and writes directly to the connected graph node.
                exact_target: Optional[str] = None
                if (
                    claim.bridge_proposition_id
                    and bridge_node_map
                    and claim.bridge_proposition_id in bridge_node_map
                ):
                    exact_target = bridge_node_map[claim.bridge_proposition_id]
                parent_edge_id: Optional[str] = None
                if exact_target is None and action_result.ingest_edge_id:
                    parent_edge = graph_updated.edges.get(action_result.ingest_edge_id)
                    if parent_edge is not None and parent_edge.conclusion_id != self.target_node_id:
                        parent_edge_id = action_result.ingest_edge_id

                real_node_id = ingest_verified_claim(
                    graph_updated,
                    claim_text=claim.claim_text,
                    verification_source=vr.backend,
                    verdict=vr.verdict,
                    source_memo_id=memo_id,
                    claim_id=claim.id,
                    target_node_id=exact_target,
                    parent_edge_id=parent_edge_id,
                )
                # Fix 6: write the real graph node ID back into VR.claim_id so
                # that propagate_verification_signals can locate the node for
                # prior updates.
                vr.claim_id = real_node_id
                ingested += 1
            except Exception as exc:
                logger.warning(
                    "Failed to ingest verification result for claim '%s': %s",
                    claim.claim_text[:120],
                    exc,
                )

        decomposed_nodes_ingested = 0
        decomposed_edges_ingested = 0
        for parent_claim_id, subclaim in decomposed_subclaims:
            parent_claim = claims_by_id.get(parent_claim_id)
            if parent_claim is None:
                continue
            parent_node_id: Optional[str] = None
            if (
                parent_claim.bridge_proposition_id
                and bridge_node_map
                and parent_claim.bridge_proposition_id in bridge_node_map
            ):
                parent_node_id = bridge_node_map[parent_claim.bridge_proposition_id]
            elif self.target_node_id in graph_updated.nodes:
                parent_node_id = self.target_node_id
            if parent_node_id is None or parent_node_id not in graph_updated.nodes:
                continue
            statement = str(getattr(subclaim, "claim_text", "")).strip()
            if not statement:
                continue
            matches = graph_updated.find_node_ids_by_statement(statement)
            if matches:
                sub_node_id = matches[0]
            else:
                node = graph_updated.add_node(
                    statement=statement,
                    belief=0.5,
                    prior=0.5,
                    domain=graph_updated.nodes[parent_node_id].domain,
                    provenance="lean_decompose",
                )
                sub_node_id = node.id
                decomposed_nodes_ingested += 1
            already = any(
                e.edge_type == "decomposition"
                and e.conclusion_id == parent_node_id
                and set(e.premise_ids) == {sub_node_id}
                for e in graph_updated.edges.values()
            )
            if not already and sub_node_id != parent_node_id:
                graph_updated.add_hyperedge(
                    premise_ids=[sub_node_id],
                    conclusion_id=parent_node_id,
                    module=Module.LEAN,
                    steps=["Lean decomposition-derived subclaim"],
                    confidence=0.95,
                    edge_type="decomposition",
                )
                decomposed_edges_ingested += 1

        # Apply legacy belief updates from ClaimVerifier results for claims
        # that were NOT handled via bridge_node_map (unmapped claims only).
        if old_style_results and self.claim_verifier is not None:
            self.claim_verifier.update_graph_beliefs(
                graph=graph_updated,
                results=old_style_results,
            )

        if ingested > 0 or decomposed_nodes_ingested > 0 or decomposed_edges_ingested > 0:
            save_graph(graph_updated, self.graph_path)

        # ---- 4. BP signal accumulation ----
        deterministic_count = sum(
            1 for vr in verification_results if vr.verdict in ("verified", "refuted")
        )
        if deterministic_count > 0:
            try:
                threshold = max(1, int(getattr(CONFIG, "bp_propagation_threshold", 1)))
                force_now = any(vr.verdict == "refuted" for vr in verification_results)
                graph_bp = load_graph(self.graph_path)
                propagate_verification_signals(
                    graph_bp,
                    verification_results,
                    threshold=threshold,
                    accumulator=self.signal_accumulator,
                    force=force_now,
                )
                save_graph(graph_bp, self.graph_path)
            except Exception as exc:
                logger.warning("Failed to propagate verification signals: %s", exc)

        # ---- 5. Collect Lean feedback for next iteration ----
        # vr.claim_id has been overwritten with the real graph node ID in step 3.
        # Use vr_to_original_claim (built before the overwrite) to resolve the
        # original Claim object for each VerificationResult.
        for vr in verification_results:
            if vr.backend.startswith("lean") and vr.lean_error and self.lean_feedback_parser is not None:
                original_claim = vr_to_original_claim.get(id(vr))
                if original_claim is None:
                    continue
                gap = self.lean_feedback_parser.parse_lean_error(vr.lean_error)
                feedback_str = self.lean_feedback_parser.gap_to_feedback(
                    gap,
                    original_claim,
                    model=self.model,
                    record_dir=self.llm_record_dir,
                    gap_index=run_idx,
                )
                if feedback_str:
                    lean_feedback_parts.append(feedback_str)

        # ---- 6. Compute verification bonus for MCTS reward ----
        # Weight by whether the claim is mapped to the bridge reasoning chain:
        # - bridge-mapped verified: highest reward (directly advances proof strategy)
        # - bridge-mapped refuted: medium reward (eliminates a route, also informative)
        # - isolated/unmapped verified: minimal reward (not on proof chain)
        bridge_verified = sum(
            1 for vr in verification_results
            if vr.verdict == "verified"
            and vr_to_original_claim.get(id(vr)) is not None
            and vr_to_original_claim[id(vr)].bridge_proposition_id
        )
        bridge_refuted = sum(
            1 for vr in verification_results
            if vr.verdict == "refuted"
            and vr_to_original_claim.get(id(vr)) is not None
            and vr_to_original_claim[id(vr)].bridge_proposition_id
        )
        verified_count = sum(1 for vr in verification_results if vr.verdict == "verified")
        refuted_count = sum(1 for vr in verification_results if vr.verdict == "refuted")
        isolated_verified = verified_count - bridge_verified
        verification_bonus = min(
            0.4,
            0.10 * bridge_verified       # bridge-mapped verified: high reward
            + 0.06 * bridge_refuted      # bridge-mapped refuted: medium reward
            + 0.02 * isolated_verified,  # isolated/trivial verified: minimal
        )

        # ---- 7. Log step ----
        steps.append({
            "phase": "claim_verification",
            "claims_extracted": len(claims),
            "claims_verified": verified_count,
            "claims_refuted": refuted_count,
            "claims_inconclusive": len(verification_results) - verified_count - refuted_count,
            "claims_ingested": ingested,
            "lean_gaps_identified": len(lean_feedback_parts),
            "decomposed_subclaims": len(decomposed_subclaims),
            "decomposed_subclaims_nodes_ingested": decomposed_nodes_ingested,
            "decomposed_subclaims_edges_ingested": decomposed_edges_ingested,
            "bp_pending_signals": self.signal_accumulator.pending_signals,
            "verification_bonus": round(verification_bonus, 4),
            "summary": "; ".join(
                f"{claim.claim_text[:60]} => {vr.verdict}"
                for claim, vr in zip(claims, verification_results)
            )[:500],
        })

        lean_feedback = "\n\n".join(lean_feedback_parts)
        return steps, lean_feedback, verification_bonus

    def _execute_selected_action(
        self,
        *,
        graph: HyperGraph,
        node_id: str,
        module: Module,
        boundary_policy: Optional[dict[str, Any]],
        feedback: str,
    ) -> ActionResult:
        try:
            return self._execute_selected_action_impl(
                graph=graph,
                node_id=node_id,
                module=module,
                boundary_policy=boundary_policy,
                feedback=feedback,
            )
        except Exception as exc:
            return ActionResult(
                action=module.value,
                target_node_id=node_id,
                selected_module=module.value,
                success=False,
                message=f"{type(exc).__name__}: {exc}"[:200],
            )

    def _execute_selected_action_impl(
        self,
        *,
        graph: HyperGraph,
        node_id: str,
        module: Module,
        boundary_policy: Optional[dict[str, Any]],
        feedback: str,
    ) -> ActionResult:
        if module == Module.PLAUSIBLE:
            # Enrich the plausible prompt with cross-domain analogy suggestions.
            # Analogies are injected as thinking material — the LLM is free to
            # adopt, adapt, or ignore them.  This is more natural than a separate
            # MCTS analogy module because: (a) analogies inform reasoning rather
            # than being standalone actions, (b) their quality is reflected in
            # the plausible judge confidence (good analogies → better routes →
            # higher reward), (c) they add no extra MCTS arm to explore.
            enriched_feedback = feedback
            if self.analogy_engine is not None:
                try:
                    analogies = self.analogy_engine.find_analogies(
                        graph.nodes[node_id].statement, graph, self.model
                    )
                    if analogies:
                        hints = "\n".join(
                            f"- [{a.source_domain}] {a.mapping}: {a.transferable_technique}"
                            for a in analogies[:3]
                        )
                        analogy_block = (
                            "\n\nCross-domain analogies to consider (adopt, adapt, or ignore):\n"
                            + hints
                        )
                        enriched_feedback = feedback + analogy_block if feedback else analogy_block
                except Exception:
                    pass  # analogy failure must never block plausible reasoning

            raw, normalized, judge_output = run_plausible_action(
                graph,
                node_id,
                model=self.model,
                feedback=enriched_feedback,
                record_dir=self.llm_record_dir,
            )
            if (
                self.config.enable_continuation_verification
                and self.continuation_verifier is not None
                and normalized.get("steps")
            ):
                consistency_scores = []
                for idx, step in enumerate(normalized.get("steps", []), start=1):
                    verification = self.continuation_verifier.verify_step(
                        target_statement=graph.nodes[node_id].statement,
                        step_id=f"step_{idx}",
                        step_statement=str(step),
                        supporting_statements=[
                            str(item.get("statement", ""))
                            for item in normalized.get("premises", [])
                            if isinstance(item, dict)
                        ],
                        model=self.model,
                        retrieval_context=feedback,
                    )
                    consistency_scores.append(verification.consistency_score)
                if consistency_scores:
                    avg_consistency = sum(consistency_scores) / len(consistency_scores)
                    judge_output = dict(judge_output or {})
                    judge_output["continuation_consistency"] = round(avg_consistency, 6)
                    judge_output["confidence"] = round(
                        0.5 * float(judge_output.get("confidence", 0.0)) + 0.5 * avg_consistency,
                        6,
                    )
            return ActionResult(
                action="plausible",
                target_node_id=node_id,
                selected_module=module.value,
                raw_output=raw,
                normalized_output=normalized,
                judge_output=judge_output,
                success=True,
                message="plausible planning complete",
            )

        if module == Module.EXPERIMENT:
            if self.config.enable_evolutionary_experiments and self.experiment_evolver is not None:
                population = self.experiment_evolver.evolve(
                    conjecture=graph.nodes[node_id].statement,
                    context=feedback,
                    model=self.model,
                )
                if population:
                    feedback = "\n\n".join(
                        part
                        for part in (
                            feedback,
                            "Top evolved experiment strategies:\n"
                            + "\n".join(
                                f"- {item.strategy}: fitness={item.fitness:.3f}"
                                for item in population[:3]
                            ),
                        )
                        if part
                    )
            raw, normalized, judge_output = run_experiment_action(
                graph,
                node_id,
                model=self.model,
                feedback=feedback,
                record_dir=self.llm_record_dir,
            )
            return ActionResult(
                action="experiment",
                target_node_id=node_id,
                selected_module=module.value,
                raw_output=raw,
                normalized_output=normalized,
                judge_output=judge_output,
                success=True,
                message="experiment complete",
            )

        if module == Module.ANALOGY:
            if self.analogy_engine is None:
                return ActionResult(
                    action="analogy",
                    target_node_id=node_id,
                    selected_module=module.value,
                    success=False,
                    message="analogy engine is not configured",
                )
            raw, normalized, judge_output = self.analogy_engine.run(
                graph=graph,
                node_id=node_id,
                model=self.model,
                feedback=feedback,
            )
            return ActionResult(
                action="analogy",
                target_node_id=node_id,
                selected_module=module.value,
                raw_output=raw,
                normalized_output=normalized,
                judge_output=judge_output,
                success=True,
                message="analogy complete",
            )

        if module == Module.SPECIALIZE:
            if self.specialize_engine is None:
                return ActionResult(
                    action="specialize",
                    target_node_id=node_id,
                    selected_module=module.value,
                    success=False,
                    message="specialize engine is not configured",
                )
            raw, normalized, judge_output = self.specialize_engine.run(
                graph=graph,
                node_id=node_id,
                model=self.model,
                feedback=feedback,
            )
            return ActionResult(
                action="specialize",
                target_node_id=node_id,
                selected_module=module.value,
                raw_output=raw,
                normalized_output=normalized,
                judge_output=judge_output,
                success=True,
                message="specialize complete",
            )

        if module == Module.RETRIEVE:
            if self.knowledge_retriever is None:
                return ActionResult(
                    action="retrieve",
                    target_node_id=node_id,
                    selected_module=module.value,
                    success=False,
                    message="knowledge retriever is not configured",
                )
            raw, normalized, judge_output = self.knowledge_retriever.run(
                graph=graph,
                node_id=node_id,
                model=self.model,
                feedback=feedback,
            )
            return ActionResult(
                action="retrieve",
                target_node_id=node_id,
                selected_module=module.value,
                raw_output=raw,
                normalized_output=normalized,
                judge_output=judge_output,
                success=True,
                message="retrieve complete",
            )

        if module == Module.DECOMPOSE:
            if self.decompose_engine is None:
                return ActionResult(
                    action="decompose",
                    target_node_id=node_id,
                    selected_module=module.value,
                    success=False,
                    message="decompose engine is not configured",
                )
            raw, normalized, judge_output = self.decompose_engine.run(
                graph=graph,
                node_id=node_id,
                model=self.model,
                feedback=feedback,
                try_formal=True,
            )
            return ActionResult(
                action="decompose",
                target_node_id=node_id,
                selected_module=module.value,
                raw_output=raw,
                normalized_output=normalized,
                judge_output=judge_output,
                success=True,
                message="decompose complete",
            )

        # Module.LEAN always runs strict Lean proof (not decompose).
        # Module.DECOMPOSE (a separate UCB arm) handles subgoal decomposition.
        # Respect lean_policy: if enable_strict_lean is explicitly False, skip.
        if not self.lean_policy.get("enable_strict_lean", True):
            return ActionResult(
                action="lean",
                target_node_id=node_id,
                selected_module=module.value,
                success=False,
                message="strict Lean skipped by lean_policy (enable_strict_lean=false)",
            )
        raw, normalized, judge_output = run_lean_action(
            graph,
            node_id,
            model=self.model,
            timeout=self._lean_timeout,
            boundary_policy=boundary_policy,
            prompt_feedback=feedback,
            record_dir=self.llm_record_dir,
        )
        return ActionResult(
            action="lean",
            target_node_id=node_id,
            selected_module=module.value,
            raw_output=raw,
            normalized_output=normalized,
            judge_output=judge_output,
            success=True,
            message="lean complete",
        )

    def _handle_plausible_followups(
        self,
        *,
        graph: HyperGraph,
        action_result: ActionResult,
        feedback: str,
        result: MCTSDiscoveryResult,
    ) -> None:
        if action_result.normalized_output is None:
            return
        plan: Optional[BridgePlan] = None
        raw_bridge: str = ""
        from dz_hypergraph.config import CONFIG
        bridge_max_attempts = getattr(CONFIG, "engine_bridge_max_attempts", 3)
        last_bridge_error: str = ""
        for bridge_attempt in range(1, bridge_max_attempts + 1):
            try:
                attempt_feedback = feedback
                if bridge_attempt > 1 and last_bridge_error:
                    attempt_feedback = (
                        (feedback + "\n\n" if feedback else "")
                        + f"Previous bridge plan attempt {bridge_attempt - 1} failed: {last_bridge_error}\n"
                        "Make sure every proposition has a valid role (seed/bridge/derived/risk/target) "
                        "and that exactly one proposition has role='target'."
                    )
                if (
                    bridge_attempt == bridge_max_attempts
                    and "target proposition" in last_bridge_error.casefold()
                    and self.target_node_id in graph.nodes
                ):
                    target_statement = graph.nodes[self.target_node_id].statement
                    target_hint = (
                        '{"id":"T_target","statement":"'
                        + target_statement.replace('"', '\\"')
                        + '","role":"target","grade":"D","depends_on":["<all_non_seed_ids>"]}'
                    )
                    attempt_feedback = (
                        (attempt_feedback + "\n\n") if attempt_feedback else ""
                    ) + (
                        "CRITICAL: You MUST include exactly one proposition with role='target'. "
                        "Use this exact target statement and schema template:\n"
                        + target_hint
                    )
                raw_bridge, plan = run_bridge_planning_action(
                    graph,
                    self.target_node_id,
                    action_result.normalized_output,
                    judge_output=action_result.judge_output,
                    model=self.model,
                    feedback=attempt_feedback,
                    record_dir=self.llm_record_dir,
                )
                confidence = float((action_result.judge_output or {}).get("confidence", 0.0))
                if confidence >= result.best_bridge_confidence:
                    node_map = materialize_bridge_nodes(
                        graph,
                        plan,
                        default_domain=graph.nodes[self.target_node_id].domain,
                        target_node_id=self.target_node_id,
                    )
                    save_graph(graph, self.graph_path)
                    result.best_bridge_plan = plan
                    result.best_bridge_confidence = confidence
                    result.best_bridge_node_map = node_map
                    if self.bridge_path is not None:
                        self.bridge_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
                result.steps.append(
                    {
                        "phase": "bridge_plan_mcts",
                        "raw": raw_bridge,
                        "bridge_metrics": plan.metrics(),
                        "attempt": bridge_attempt,
                    }
                )
                break  # success
            except Exception as exc:
                last_bridge_error = str(exc)
                result.steps.append({
                    "phase": "bridge_plan_mcts",
                    "error": last_bridge_error,
                    "attempt": bridge_attempt,
                })
                plan = None

        followups: list[ActionResult] = []
        if plan is not None:
            try:
                from dz_hypergraph.config import CONFIG as _cfg

                if getattr(_cfg, "spec_decomp_enabled", False):
                    from dz_hypergraph.session import GraphSession
                    from dz_engine.bridge_executor import SpeculativeDecomposer

                    session = GraphSession(graph.model_copy(deep=True))

                    def _generate_spec_plan(**kwargs: Any) -> BridgePlan:
                        _raw_spec, spec_plan = run_bridge_planning_action(
                            kwargs["graph"],
                            kwargs["target_node_id"],
                            action_result.normalized_output or {},
                            judge_output=action_result.judge_output,
                            model=kwargs.get("model") or self.model,
                            feedback=feedback,
                            max_attempts=1,
                        )
                        return spec_plan

                    speculative = SpeculativeDecomposer(
                        session=session,
                        target_node_id=self.target_node_id,
                        generate_plan_fn=_generate_spec_plan,
                        pav=None,
                        num_candidates=max(1, int(getattr(_cfg, "spec_decomp_num_candidates", 3))),
                        model=self.model,
                    )
                    best_spec = speculative.generate_and_score()
                    if best_spec is not None:
                        result.steps.append(
                            {
                                "phase": "spec_decomposition",
                                "enabled": True,
                                "candidate_index": best_spec.candidate.candidate_index,
                                "total_score": round(float(best_spec.total_score), 6),
                                "belief_gain": round(float(best_spec.belief_gain), 6),
                                "grade_quality": round(float(best_spec.grade_quality), 6),
                                "constraint_valid": bool(best_spec.constraint.is_valid),
                            }
                        )
                    else:
                        result.steps.append(
                            {
                                "phase": "spec_decomposition",
                                "enabled": True,
                                "message": "No valid speculative decomposition candidate found.",
                            }
                        )
            except Exception as exc:
                result.steps.append(
                    {
                        "phase": "spec_decomposition",
                        "enabled": True,
                        "error": str(exc),
                    }
                )
            try:
                from dz_hypergraph.config import CONFIG as _cfg
                from dz_engine.orchestrator import plan_bridge_consumption as _pbc
                _decision = _pbc(plan)
                _delegated_count = len(_decision.experiment_proposition_ids)
                _max_rounds = max(1, min(_delegated_count, getattr(_cfg, "max_bridge_followup_rounds", 5)))
                followups = execute_bridge_followups(
                    self.graph_path,
                    self.target_node_id,
                    action_result.normalized_output,
                    plan=plan,
                    raw_bridge=raw_bridge,
                    judge_output=action_result.judge_output,
                    model=self.model,
                    backend=self.backend,
                    max_rounds=_max_rounds,
                    record_dir=self.llm_record_dir,
                )
            except Exception as exc:
                result.steps.append({"phase": "bridge_consumption", "error": str(exc)})

        for followup in followups:
            payload: dict[str, Any] = {
                "phase": followup.action,
                "message": followup.message,
                "edge_id": followup.ingest_edge_id,
            }
            if followup.normalized_output is not None:
                payload["normalized"] = followup.normalized_output
            result.steps.append(payload)

        from dz_hypergraph.config import CONFIG as _cfg
        graph_bp = load_graph(self.graph_path)
        propagate_beliefs(
            graph_bp,
            warmstart=(getattr(_cfg, "bp_backend", "gaia") != "gaia_v2"),
        )
        save_graph(graph_bp, self.graph_path)

    def _spawn_problem_variants(
        self,
        graph: HyperGraph,
        node_id: str,
        result: MCTSDiscoveryResult,
    ) -> list[str]:
        if self.problem_variant_generator is None:
            return []
        variants = self.problem_variant_generator.generate_variants(
            graph,
            node_id,
            model=self.model,
        )
        if not variants:
            return []
        created_ids = self.problem_variant_generator.materialize_variants(
            graph,
            variants,
            domain=graph.nodes[node_id].domain if node_id in graph.nodes else None,
        )
        save_graph(graph, self.graph_path)
        result.steps.append(
            {
                "phase": "problem_variants",
                "created_node_ids": created_ids,
                "variants": [
                    {
                        "statement": item.variant_statement,
                        "variant_type": item.variant_type,
                        "difficulty_estimate": item.difficulty_estimate,
                    }
                    for item in variants
                ],
            }
        )
        return created_ids

    def _phase_name(
        self,
        module: Module,
        *,
        plausible_attempts: int,
        experiment_attempts: int,
        lean_attempts: int,
    ) -> str:
        if module == Module.PLAUSIBLE:
            return "plausible" if plausible_attempts == 0 else f"plausible_replan_mcts_{plausible_attempts:02d}"
        if module == Module.EXPERIMENT:
            return "experiment" if experiment_attempts == 0 else f"experiment_mcts_{experiment_attempts:02d}"
        if module == Module.LEAN:
            return "strict_lean" if lean_attempts == 0 else f"strict_lean_mcts_{lean_attempts:02d}"
        return module.value
