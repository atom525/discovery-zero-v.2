"""
Expert Iteration Self-Improvement Loop for Discovery Zero.

Original innovation: applies expert iteration (ExIt) at the discovery-level
(not tactic-level).  Instead of learning to choose tactics for a known theorem,
we learn to:
  1. Select which frontier node to explore (PAV training)
  2. Choose which module to apply (policy DPO training)
  3. Correct BP messages (Neural BP training)

The training loop is:
  Collect → Train PAV + Policy + Neural BP → Deploy → Repeat

Training signals:
  - PAV: regression on (graph_state, action) → actual belief_gain
  - Policy (DPO): (preferred_action, rejected_action) pairs ranked by belief_gain
  - Neural BP: (standard_bp_beliefs, actual_beliefs) correction

This is inspired by:
  - BFS-Prover's DPO from compiler feedback
  - DeepSeek-Prover-V2's GRPO training
  - AlphaProof's RL on auto-formalized problems

But operates at the knowledge-discovery level, not the proof level.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from discovery_zero.graph.models import HyperGraph, Module
from discovery_zero.planning.search import IntrinsicReward

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Experience record                                                    #
# ------------------------------------------------------------------ #

@dataclass
class ExperienceRecord:
    """
    One (state, action, reward, next_state) transition in the discovery MDP.

    Collected from DiscoveryEngine.run() via ActionEvent hooks.
    """

    # State
    graph_snapshot_json: str
    """Serialised HyperGraph at decision time."""

    target_node_id: str
    action_node_id: str
    action_module: str  # Module.value

    # Outcome
    intrinsic_reward: IntrinsicReward = field(default_factory=IntrinsicReward)
    belief_delta: float = 0.0
    """Actual belief change on target after executing action + BP."""

    success: bool = False
    bridge_plan_valid: bool = False

    # Next state
    next_graph_snapshot_json: str = ""

    # Metadata
    run_id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_snapshot_json": self.graph_snapshot_json,
            "target_node_id": self.target_node_id,
            "action_node_id": self.action_node_id,
            "action_module": self.action_module,
            "intrinsic_reward": self.intrinsic_reward.to_dict(),
            "belief_delta": round(self.belief_delta, 4),
            "success": self.success,
            "bridge_plan_valid": self.bridge_plan_valid,
            "next_graph_snapshot_json": self.next_graph_snapshot_json,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperienceRecord":
        reward_payload = data.get("intrinsic_reward", {}) or {}
        intrinsic_reward = IntrinsicReward(
            belief_gain=float(reward_payload.get("belief_gain", 0.0)),
            graph_novelty=float(reward_payload.get("graph_novelty", 0.0)),
            strategy_surprise=float(reward_payload.get("strategy_surprise", 0.0)),
        )
        return cls(
            graph_snapshot_json=data.get("graph_snapshot_json", ""),
            target_node_id=data.get("target_node_id", ""),
            action_node_id=data.get("action_node_id", ""),
            action_module=data.get("action_module", Module.PLAUSIBLE.value),
            intrinsic_reward=intrinsic_reward,
            belief_delta=float(data.get("belief_delta", 0.0)),
            success=bool(data.get("success", False)),
            bridge_plan_valid=bool(data.get("bridge_plan_valid", False)),
            next_graph_snapshot_json=data.get("next_graph_snapshot_json", ""),
            run_id=data.get("run_id", ""),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
        )

    @property
    def total_reward(self) -> float:
        return 0.6 * self.belief_delta + 0.4 * self.intrinsic_reward.total


# ------------------------------------------------------------------ #
# Experience buffer                                                    #
# ------------------------------------------------------------------ #

class ExperienceBuffer:
    """
    Thread-safe circular buffer of ExperienceRecords.

    Provides sampling for training PAV, Neural BP, and the discovery policy.
    Supports disk persistence for multi-run continuity.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self._capacity = capacity
        self._buffer: List[ExperienceRecord] = []
        self._lock = threading.Lock()

    def add(self, record: ExperienceRecord) -> None:
        with self._lock:
            if len(self._buffer) >= self._capacity:
                self._buffer.pop(0)
            self._buffer.append(record)

    def sample_batch(self, batch_size: int) -> List[ExperienceRecord]:
        with self._lock:
            n = min(batch_size, len(self._buffer))
            return random.sample(self._buffer, n) if n > 0 else []

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = [r.to_dict() for r in self._buffer[-10_000:]]  # save last 10k
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("ExperienceBuffer saved %d records to %s", len(data), path)

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            records = [
                ExperienceRecord.from_dict(item)
                for item in data
                if isinstance(item, dict)
            ]
            with self._lock:
                self._buffer = records[-self._capacity :]
            logger.info("ExperienceBuffer loaded %d records from %s", len(records), path)
        except Exception as exc:
            logger.warning("Failed to load experience buffer: %s", exc)

    def statistics(self) -> Dict[str, Any]:
        with self._lock:
            if not self._buffer:
                return {"size": 0}
            rewards = [r.total_reward for r in self._buffer]
            successes = sum(1 for r in self._buffer if r.success)
            return {
                "size": len(self._buffer),
                "mean_reward": round(sum(rewards) / len(rewards), 4),
                "success_rate": round(successes / len(self._buffer), 4),
                "module_distribution": self._module_distribution(),
            }

    def _module_distribution(self) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for r in self._buffer:
            dist[r.action_module] = dist.get(r.action_module, 0) + 1
        return dist


# ------------------------------------------------------------------ #
# DPO preference pair construction                                     #
# ------------------------------------------------------------------ #

@dataclass
class DPOPair:
    """One (preferred, rejected) action pair for Direct Preference Optimisation."""

    preferred: ExperienceRecord
    rejected: ExperienceRecord
    reward_gap: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preferred": self.preferred.to_dict(),
            "rejected": self.rejected.to_dict(),
            "reward_gap": round(self.reward_gap, 4),
        }


def build_dpo_pairs(
    buffer: ExperienceBuffer,
    min_reward_gap: float = 0.1,
    max_pairs: int = 1000,
) -> List[DPOPair]:
    """
    Build (preferred, rejected) pairs for DPO training.

    Grouping: experiences with the same target_node_id are candidates.
    Pairing: highest-reward action paired with lowest-reward action.

    Args:
        buffer: The experience buffer to sample from.
        min_reward_gap: Minimum reward difference to include a pair.
        max_pairs: Maximum number of pairs to return.

    Returns:
        List of DPOPair objects sorted by reward_gap (descending).
    """
    records = buffer.sample_batch(min(5000, len(buffer)))
    if not records:
        return []

    # Group by target_node_id + coarse graph state hash so preference pairs
    # compare actions taken from comparable discovery states.
    groups: Dict[str, List[ExperienceRecord]] = {}
    for r in records:
        state_hash = hashlib.sha1(r.graph_snapshot_json.encode("utf-8")).hexdigest()[:12]
        groups.setdefault(f"{r.target_node_id}:{state_hash}", []).append(r)

    pairs: List[DPOPair] = []
    for target_id, group in groups.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda r: r.total_reward, reverse=True)
        # Pair best with worst, second-best with second-worst, etc.
        n = len(group)
        for i in range(min(n // 2, 5)):  # at most 5 pairs per target
            preferred = group[i]
            rejected = group[n - 1 - i]
            gap = preferred.total_reward - rejected.total_reward
            if gap >= min_reward_gap:
                pairs.append(DPOPair(
                    preferred=preferred,
                    rejected=rejected,
                    reward_gap=gap,
                ))

    pairs.sort(key=lambda p: p.reward_gap, reverse=True)
    return pairs[:max_pairs]


# ------------------------------------------------------------------ #
# Expert Iteration Loop                                                #
# ------------------------------------------------------------------ #

@dataclass
class ExpertIterationResult:
    """Result of one Expert Iteration cycle."""

    iteration: int
    experiences_collected: int
    pav_loss: float = 0.0
    neural_bp_loss: float = 0.0
    dpo_pairs_trained: int = 0
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "experiences_collected": self.experiences_collected,
            "pav_loss": round(self.pav_loss, 4),
            "neural_bp_loss": round(self.neural_bp_loss, 4),
            "dpo_pairs_trained": self.dpo_pairs_trained,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }


class ExpertIterationLoop:
    """
    Self-improvement loop for Discovery Zero.

    Each iteration:
      1. Run benchmark suite with current policy → collect ExperienceRecords
      2. Train PAV on (graph_state, action) → belief_delta
      3. Train Neural BP on (standard_bp_beliefs, actual_beliefs)
      4. Build DPO pairs and train policy (if LLM fine-tuning is available)
      5. Deploy updated models

    This creates a closed self-improvement loop analogous to AlphaProof's RL
    cycle, but at the discovery level rather than the Lean tactic level.
    """

    def __init__(
        self,
        experience_buffer: ExperienceBuffer,
        pav: Optional[Any] = None,           # ProcessAdvantageVerifier
        neural_bp: Optional[Any] = None,     # NeuralBPCorrector
        suite_runner: Optional[Any] = None,  # Callable to run benchmark
        checkpoint_dir: Optional[Path] = None,
        batch_size: int = 32,
        min_buffer_size_for_training: int = 100,
    ) -> None:
        self._buffer = experience_buffer
        self._pav = pav
        self._neural_bp = neural_bp
        self._suite_runner = suite_runner
        self._checkpoint_dir = checkpoint_dir or Path("./expert_iter_checkpoints")
        self._batch_size = batch_size
        self._min_buffer = min_buffer_size_for_training
        self._iteration = 0

    def run_iteration(self) -> ExpertIterationResult:
        """
        One full collect-train-deploy cycle.

        Returns ExpertIterationResult with metrics from this iteration.
        """
        import time
        t0 = time.monotonic()
        self._iteration += 1

        # Phase 1: Collect experiences by running the benchmark suite
        n_before = len(self._buffer)
        if self._suite_runner is not None:
            try:
                self._collect_from_suite()
            except Exception as exc:
                logger.warning("Expert iteration collect phase failed: %s", exc)
        n_after = len(self._buffer)
        experiences_collected = n_after - n_before

        result = ExpertIterationResult(
            iteration=self._iteration,
            experiences_collected=experiences_collected,
        )

        # Phase 2: Train PAV
        if len(self._buffer) >= self._min_buffer:
            result.pav_loss = self._train_pav()
            result.neural_bp_loss = self._train_neural_bp()
            result.dpo_pairs_trained = self._build_and_log_dpo_pairs()
        self.update_blend_ratio()

        # Save checkpoints
        self._save_checkpoints()

        result.elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "ExpertIteration #%d: collected=%d pav_loss=%.4f neural_bp_loss=%.4f dpo_pairs=%d",
            self._iteration,
            experiences_collected,
            result.pav_loss,
            result.neural_bp_loss,
            result.dpo_pairs_trained,
        )
        return result

    def _collect_from_suite(self) -> None:
        """Run the benchmark suite and collect experiences into the buffer."""
        if self._suite_runner is not None:
            self._suite_runner(experience_buffer=self._buffer)

    def collect_from_mcts_run(self, experiences: List[ExperienceRecord]) -> int:
        """Append MCTS-produced experiences into the shared replay buffer."""
        added = 0
        for item in experiences:
            self._buffer.add(item)
            added += 1
        self.update_blend_ratio()
        return added

    def _train_pav(self) -> float:
        """Train PAV for one epoch on the experience buffer."""
        if self._pav is None:
            return 0.0
        try:
            import torch
            from discovery_zero.planning.value_net import PAVTrainingSample
            optimizer = torch.optim.Adam(
                self._pav._net.parameters(), lr=1e-4
            ) if hasattr(self._pav, "_net") and self._pav._net else None
            if optimizer is None:
                return 0.0

            batch = self._buffer.sample_batch(self._batch_size)
            if not batch:
                return 0.0

            # Convert ExperienceRecord → PAVTrainingSample
            samples = []
            for r in batch:
                try:
                    graph = HyperGraph.model_validate_json(r.graph_snapshot_json)
                    module = Module(r.action_module)
                    samples.append(PAVTrainingSample(
                        graph_snapshot=graph,
                        target_node_id=r.target_node_id,
                        action_node_id=r.action_node_id,
                        action_module=module,
                        actual_belief_gain=r.belief_delta,
                        success=r.success,
                    ))
                except Exception:
                    continue

            return self._pav.train_step(samples, optimizer) if samples else 0.0
        except Exception as exc:
            logger.debug("PAV training failed: %s", exc)
            return 0.0

    def _train_neural_bp(self) -> float:
        """Train Neural BP corrector for one epoch on the experience buffer."""
        if self._neural_bp is None:
            return 0.0
        try:
            import torch
            from discovery_zero.graph.neural_bp import BPTrainingSample
            optimizer = torch.optim.Adam(
                self._neural_bp._net.parameters(), lr=1e-4
            ) if hasattr(self._neural_bp, "_net") and self._neural_bp._net else None
            if optimizer is None:
                return 0.0

            batch = self._buffer.sample_batch(self._batch_size)
            if not batch:
                return 0.0

            # Use graph_snapshot as "standard beliefs" and next_snapshot beliefs as targets
            samples = []
            for r in batch:
                if not r.graph_snapshot_json or not r.next_graph_snapshot_json:
                    continue
                try:
                    from discovery_zero.graph.adapter import run_gaia_bp, write_back_beliefs
                    graph = HyperGraph.model_validate_json(r.graph_snapshot_json)
                    next_graph = HyperGraph.model_validate_json(r.next_graph_snapshot_json)

                    result = run_gaia_bp(graph)
                    write_back_beliefs(graph, result)
                    standard = {nid: node.belief for nid, node in graph.nodes.items()}
                    true_b = {nid: node.belief for nid, node in next_graph.nodes.items()}

                    samples.append(BPTrainingSample(
                        graph=graph,
                        standard_beliefs=standard,
                        true_beliefs=true_b,
                        run_id=r.run_id,
                        node_id=r.target_node_id,
                    ))
                except Exception:
                    continue

            return self._neural_bp.train_step(samples, optimizer) if samples else 0.0
        except Exception as exc:
            logger.debug("Neural BP training failed: %s", exc)
            return 0.0

    def _build_and_log_dpo_pairs(self) -> int:
        """Build DPO pairs for policy training and save to checkpoint dir."""
        pairs = build_dpo_pairs(self._buffer, min_reward_gap=0.1)
        if pairs:
            pairs_path = self._checkpoint_dir / f"dpo_pairs_iter_{self._iteration:04d}.json"
            try:
                pairs_path.parent.mkdir(parents=True, exist_ok=True)
                pairs_path.write_text(
                    json.dumps([p.to_dict() for p in pairs[:500]], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                logger.info(
                    "Saved %d DPO pairs for iteration %d", len(pairs), self._iteration
                )
            except Exception as exc:
                logger.debug("Failed to save DPO pairs: %s", exc)
        return len(pairs)

    def _save_checkpoints(self) -> None:
        """Persist PAV, Neural BP weights, and experience buffer."""
        iter_dir = self._checkpoint_dir / f"iter_{self._iteration:04d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        if self._pav is not None:
            try:
                self._pav.save(iter_dir / "pav.pt")
            except Exception:
                pass

        if self._neural_bp is not None:
            try:
                self._neural_bp.save(iter_dir / "neural_bp.pt")
            except Exception:
                pass

        try:
            self._buffer.save(iter_dir / "experience_buffer.json")
        except Exception:
            pass

    def update_blend_ratio(self, decay_experiences: int = 500) -> float:
        """Reduce dependence on external PRM as replay data accumulates.

        Uses CONFIG.pav_blend_decay_experiences if available and the caller
        did not supply an explicit override.
        """
        effective_decay = decay_experiences
        try:
            from discovery_zero.config import CONFIG
            cfg_decay = int(getattr(CONFIG, "pav_blend_decay_experiences", 0))
            if cfg_decay > 0 and decay_experiences == 500:
                # Only substitute the default; explicit caller values are respected.
                effective_decay = cfg_decay
        except Exception:
            pass
        if self._pav is not None and hasattr(self._pav, "update_blend_ratio"):
            try:
                return float(self._pav.update_blend_ratio(len(self._buffer), effective_decay))
            except Exception:
                return 0.0
        return 0.0

    def run_n_iterations(self, n: int) -> List[ExpertIterationResult]:
        """Run n consecutive expert iteration cycles."""
        results = []
        for _ in range(n):
            results.append(self.run_iteration())
        return results
