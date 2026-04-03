"""
Process Advantage Verifier (PAV) — predicts the belief gain expected from
taking a (node, module) action in the current hypergraph state.

Original innovation: extends Setlur et al. (ICLR 2025)'s token-level process
advantage verification to *hypergraph-level* discovery, predicting continuous
belief delta rather than binary step correctness.  This is the first application
of PAV to knowledge discovery over dynamic factor graphs.

Architecture:
  HypergraphStateEncoder: encodes (graph, target_node_id) into a fixed-dim vector.
    - Uses Gaia's EmbeddingModel for semantic node statement embeddings.
    - Aggregates node embeddings with attention weighted by graph structure.

  ProcessAdvantageVerifier: MLP that takes (state_encoding, action_features)
    and predicts expected belief delta on the target node after BP.

When torch is unavailable the PAV returns 0.0 (neutral, falls back to UCB).
"""

from __future__ import annotations

import logging
import math
import asyncio
import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dz_hypergraph.models import HyperGraph, Module

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None     # type: ignore
    F = None      # type: ignore


# ------------------------------------------------------------------ #
# Action feature encoding                                              #
# ------------------------------------------------------------------ #

MODULE_INDEX = {
    Module.PLAUSIBLE: 0,
    Module.EXPERIMENT: 1,
    Module.LEAN: 2,
    Module.ANALOGY: 3,
    Module.DECOMPOSE: 4,
    Module.SPECIALIZE: 5,
    Module.RETRIEVE: 6,
}
ACTION_DIM = len(MODULE_INDEX) + 8


def encode_action(
    module: Module,
    node_belief: float,
    node_prior: float,
    degree_in: int,
    degree_out: int,
) -> "Optional[torch.Tensor]":
    """Encode a (module, node) action into a fixed-dim vector."""
    if not _TORCH_AVAILABLE:
        return None
    module_oh = [0.0 for _ in range(len(MODULE_INDEX))]
    if module in MODULE_INDEX:
        module_oh[MODULE_INDEX[module]] = 1.0

    node_feats = [
        node_belief,
        node_prior,
        1.0 - node_belief,  # uncertainty
        min(1.0, degree_in / 5.0),
        min(1.0, degree_out / 5.0),
        1.0 if module == Module.LEAN and node_belief > 0.7 else 0.0,
        1.0 if module == Module.EXPERIMENT and 0.3 < node_belief < 0.7 else 0.0,
        1.0 if module == Module.PLAUSIBLE and node_belief < 0.5 else 0.0,
    ]
    features = module_oh + node_feats
    return torch.tensor(features, dtype=torch.float32)


# ------------------------------------------------------------------ #
# Hypergraph state encoder                                            #
# ------------------------------------------------------------------ #

if _TORCH_AVAILABLE:
    class _AttentionAggregator(nn.Module):
        """Aggregate a set of node embeddings with content attention."""

        def __init__(self, embed_dim: int, hidden_dim: int) -> None:
            super().__init__()
            self.attn = nn.Linear(embed_dim, 1)
            self.proj = nn.Linear(embed_dim, hidden_dim)

        def forward(self, embeddings: "torch.Tensor") -> "torch.Tensor":
            """
            embeddings: (N, embed_dim)
            Returns: (hidden_dim,)
            """
            if embeddings.shape[0] == 0:
                return torch.zeros(self.proj.out_features)
            weights = F.softmax(self.attn(embeddings), dim=0)  # (N, 1)
            agg = (weights * embeddings).sum(0)               # (embed_dim,)
            return F.relu(self.proj(agg))                      # (hidden_dim,)
else:
    _AttentionAggregator = None  # type: ignore


class HypergraphStateEncoder:
    """
    Encode the current hypergraph into a fixed-dim state vector.

    Uses Gaia's EmbeddingModel for semantic node embeddings, then
    aggregates them with attention over the graph structure.

    When the EmbeddingModel is not available, falls back to handcrafted
    statistics (mean belief, variance, density, etc.).
    """

    def __init__(
        self,
        embedding_model: Any = None,
        aggregation: str = "attention",
        hidden_dim: int = 128,
    ) -> None:
        self._embedding_model = embedding_model
        self._aggregation = aggregation
        self._hidden_dim = hidden_dim
        self._aggregator: Optional[Any] = None

        if _TORCH_AVAILABLE and aggregation == "attention" and embedding_model is not None:
            try:
                # Probe embedding dim
                test_emb = self._embed_texts(["test"])
                embed_dim = len(test_emb[0]) if test_emb else 64
                self._aggregator = _AttentionAggregator(embed_dim, hidden_dim)
            except Exception:
                pass

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self._embedding_model is None:
            return []
        result = self._embedding_model.embed(texts)
        if inspect.isawaitable(result):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(result)
            raise RuntimeError(
                "EmbeddingModel.embed returned awaitable in active event loop; "
                "provide a sync wrapper for HypergraphStateEncoder."
            ) from loop
        return result

    @property
    def output_dim(self) -> int:
        """Dimensionality of the state encoding."""
        if self._aggregator is not None:
            return self._hidden_dim
        return 16  # handcrafted statistics

    def encode(
        self,
        graph: HyperGraph,
        focus_node_id: str,
    ) -> "Optional[torch.Tensor]":
        """
        Returns a (hidden_dim,) or (16,) tensor representing the graph state
        focused on the focus node (typically the target).
        """
        if not _TORCH_AVAILABLE:
            return None

        # Try neural encoding via EmbeddingModel
        if self._aggregator is not None and self._embedding_model is not None:
            try:
                return self._neural_encode(graph, focus_node_id)
            except Exception as exc:
                logger.debug("Neural encoding failed, using statistics: %s", exc)

        # Handcrafted statistics fallback
        return self._stats_encode(graph, focus_node_id)

    def _neural_encode(
        self, graph: HyperGraph, focus_node_id: str
    ) -> "torch.Tensor":
        """Aggregate semantic embeddings of graph nodes."""
        statements = [n.statement for n in graph.nodes.values()]
        embeddings = self._embed_texts(statements[:50])  # cap at 50
        emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
        return self._aggregator(emb_tensor)

    def _stats_encode(
        self, graph: HyperGraph, focus_node_id: str
    ) -> "torch.Tensor":
        """Fast handcrafted feature vector."""
        beliefs = [n.belief for n in graph.nodes.values()]
        n = len(beliefs)
        focus = graph.nodes.get(focus_node_id)
        proven_frac = sum(1 for nd in graph.nodes.values() if nd.state == "proven") / max(n, 1)
        refuted_frac = sum(1 for nd in graph.nodes.values() if nd.state == "refuted") / max(n, 1)
        mean_b = sum(beliefs) / max(n, 1)
        var_b = sum((b - mean_b) ** 2 for b in beliefs) / max(n, 1)
        density = len(graph.edges) / max(n * n, 1)
        focus_belief = focus.belief if focus else 0.5
        focus_prior = focus.prior if focus else 0.5
        focus_in = len(graph.get_edges_to(focus_node_id)) if focus else 0
        focus_out = len(graph.get_edges_from(focus_node_id)) if focus else 0

        # Belief percentiles
        sorted_b = sorted(beliefs)
        p25 = sorted_b[n // 4] if n > 3 else mean_b
        p75 = sorted_b[3 * n // 4] if n > 3 else mean_b

        features = [
            n / 100.0, len(graph.edges) / 100.0, density,
            mean_b, var_b, p25, p75,
            proven_frac, refuted_frac,
            focus_belief, focus_prior, 1.0 - focus_belief,
            min(1.0, focus_in / 5.0), min(1.0, focus_out / 5.0),
            1.0 if (focus and focus.formal_statement) else 0.0,
            1.0 if focus_belief > 0.8 else 0.0,
        ]
        return torch.tensor(features[:16], dtype=torch.float32)


# ------------------------------------------------------------------ #
# PAV network                                                          #
# ------------------------------------------------------------------ #

if _TORCH_AVAILABLE:
    class _PAVNet(nn.Module):
        def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]) -> None:
            super().__init__()
            layers = []
            in_dim = state_dim + action_dim
            for hd in hidden_dims:
                layers += [nn.Linear(in_dim, hd), nn.ReLU()]
                in_dim = hd
            layers.append(nn.Linear(in_dim, 1))
            layers.append(nn.Tanh())  # output in (-1, 1)
            self.net = nn.Sequential(*layers)

        def forward(self, state: "torch.Tensor", action: "torch.Tensor") -> "torch.Tensor":
            return self.net(torch.cat([state, action]))
else:
    _PAVNet = None  # type: ignore


# ------------------------------------------------------------------ #
# PAV Training Sample                                                  #
# ------------------------------------------------------------------ #

@dataclass
class PAVTrainingSample:
    """One (state, action, actual_belief_gain) training example."""

    graph_snapshot: HyperGraph
    target_node_id: str
    action_node_id: str
    action_module: Module
    actual_belief_gain: float  # belief(target) after BP − before
    success: bool = True


# ------------------------------------------------------------------ #
# ProcessAdvantageVerifier                                             #
# ------------------------------------------------------------------ #

class ProcessAdvantageVerifier:
    """
    Predicts the expected belief gain on the target node after executing
    a (candidate_node, candidate_module) action and running Gaia BP.

    PAV(s, a) ≈ E[belief_after_BP(target) − belief_before(target)]

    This provides a continuous progress signal usable as:
    - Policy prior in HTPS PUCT (replaces uniform 1/N)
    - Value estimate in RMaxTS (replaces rollout)
    - Curriculum signal in Expert Iteration DPO

    Cold-start: returns 0 (neutral) until enough training data accumulated.
    """

    def __init__(
        self,
        state_encoder: Optional[HypergraphStateEncoder] = None,
        hidden_dims: Optional[List[int]] = None,
        model_path: Optional[Path] = None,
        external_prm: Optional[Any] = None,
        blend_ratio: float = 1.0,
    ) -> None:
        self._encoder = state_encoder or HypergraphStateEncoder()
        hidden_dims = hidden_dims or [128, 64]
        self._net: Optional[Any] = None
        self._trained = False
        self._external_prm = external_prm
        self._blend_ratio = max(0.0, min(1.0, blend_ratio))

        if _TORCH_AVAILABLE:
            state_dim = self._encoder.output_dim
            self._net = _PAVNet(state_dim, ACTION_DIM, hidden_dims)
            if model_path is not None and model_path.exists():
                try:
                    self._net.load_state_dict(
                        torch.load(model_path, map_location="cpu", weights_only=True)
                    )
                    self._net.eval()
                    self._trained = True
                    logger.info("Loaded PAV model from %s", model_path)
                except Exception as exc:
                    logger.warning("Failed to load PAV: %s", exc)

    def predict_advantage(
        self,
        graph: HyperGraph,
        target_node_id: str,
        candidate_node_id: str,
        candidate_module: Module,
    ) -> float:
        """
        Predict expected belief gain on target after executing (candidate_node, candidate_module).

        Returns value in [-1, 1].  Positive = beneficial action.
        Returns 0.0 if PAV is untrained or torch unavailable.
        """
        neural_prediction = 0.0
        neural_available = bool(_TORCH_AVAILABLE and self._net is not None and self._trained)

        if neural_available:
            with torch.no_grad():
                state_enc = self._encoder.encode(graph, target_node_id)
                if state_enc is not None:
                    cand_node = graph.nodes.get(candidate_node_id)
                    if cand_node is not None:
                        action_enc = encode_action(
                            candidate_module,
                            cand_node.belief,
                            cand_node.prior,
                            len(graph.get_edges_to(candidate_node_id)),
                            len(graph.get_edges_from(candidate_node_id)),
                        )
                        if action_enc is not None:
                            prediction = self._net(state_enc, action_enc)
                            neural_prediction = float(prediction)

        external_prediction: Optional[float] = None
        if self._external_prm is not None:
            try:
                external_prediction = float(
                    self._external_prm.estimate_value(
                        graph,
                        target_node_id,
                        candidate_node_id,
                        candidate_module,
                    )
                )
            except Exception:
                external_prediction = None

        if external_prediction is None:
            return neural_prediction if neural_available else 0.0
        if not neural_available:
            return external_prediction

        blend = self._blend_ratio
        return (1.0 - blend) * neural_prediction + blend * external_prediction

    def train_step(
        self,
        batch: List[PAVTrainingSample],
        optimizer: Any,
    ) -> float:
        """One gradient step on a batch of training samples."""
        if not _TORCH_AVAILABLE or self._net is None:
            return 0.0

        loss_fn = nn.MSELoss()
        optimizer.zero_grad()
        total_loss = 0.0

        for sample in batch:
            state_enc = self._encoder.encode(sample.graph_snapshot, sample.target_node_id)
            cand_node = sample.graph_snapshot.nodes.get(sample.action_node_id)
            if state_enc is None or cand_node is None:
                continue

            action_enc = encode_action(
                sample.action_module,
                cand_node.belief,
                cand_node.prior,
                len(sample.graph_snapshot.get_edges_to(sample.action_node_id)),
                len(sample.graph_snapshot.get_edges_from(sample.action_node_id)),
            )
            if action_enc is None:
                continue

            pred = self._net(state_enc, action_enc)
            target = torch.tensor([sample.actual_belief_gain], dtype=torch.float32)
            loss = loss_fn(pred, target)
            loss.backward()
            total_loss += loss.item()

        if batch:
            optimizer.step()
            self._trained = True
            return total_loss / len(batch)
        return 0.0

    def save(self, path: Path) -> None:
        if self._net is not None and _TORCH_AVAILABLE:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._net.state_dict(), path)

    def update_blend_ratio(self, experience_count: int, decay_experiences: int = 500) -> float:
        if decay_experiences <= 0:
            self._blend_ratio = 0.0
            return self._blend_ratio
        self._blend_ratio = max(0.0, 1.0 - float(experience_count) / float(decay_experiences))
        return self._blend_ratio


def migrate_pav_checkpoint(
    old_path: Path,
    new_path: Path,
    *,
    old_module_count: int = 3,
    new_module_count: int = 7,
    state_dim: int = 16,
    node_feat_dim: int = 8,
) -> None:
    """Migrate a PAV checkpoint from an old module count to a new one.

    The weight layout for the first linear layer is:
        [state_dim | module_one_hot(old_module_count) | node_feat_dim]
    Columns for new modules are zero-initialised.

    Args:
        old_path: Path to the checkpoint to migrate.
        new_path: Destination path for the migrated checkpoint.
        old_module_count: Number of modules in the old checkpoint (e.g. 3).
        new_module_count: Number of modules in the new system (e.g. 7).
        state_dim: Dimension of the hypergraph state encoding (default 16).
        node_feat_dim: Number of per-node scalar features (default 8).
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for PAV checkpoint migration.")

    old_action_dim = old_module_count + node_feat_dim
    new_action_dim = new_module_count + node_feat_dim
    old_input_dim = state_dim + old_action_dim
    new_input_dim = state_dim + new_action_dim

    old_sd: dict = torch.load(old_path, map_location="cpu", weights_only=True)
    first_key = "net.0.weight"
    if first_key not in old_sd:
        raise ValueError(
            f"Checkpoint {old_path} does not contain expected key '{first_key}'."
        )
    old_w: "torch.Tensor" = old_sd[first_key]
    if old_w.shape[1] != old_input_dim:
        raise ValueError(
            f"Expected old first-layer input dim {old_input_dim}, "
            f"but checkpoint has {old_w.shape[1]}."
        )

    hidden_dim = old_w.shape[0]
    new_w = torch.zeros(hidden_dim, new_input_dim, dtype=old_w.dtype)

    # Copy state columns [0 : state_dim]
    new_w[:, :state_dim] = old_w[:, :state_dim]

    # Copy old module one-hot columns [state_dim : state_dim + old_module_count]
    # into the corresponding positions in new layout
    new_w[:, state_dim : state_dim + old_module_count] = (
        old_w[:, state_dim : state_dim + old_module_count]
    )
    # Columns [state_dim + old_module_count : state_dim + new_module_count]
    # are new modules → stay as zero.

    # Copy node feature columns [state_dim + old_module_count : old_input_dim]
    new_w[:, state_dim + new_module_count :] = (
        old_w[:, state_dim + old_module_count :]
    )

    new_sd = dict(old_sd)
    new_sd[first_key] = new_w
    new_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_sd, new_path)
    logger.info(
        "PAV checkpoint migrated: %s -> %s (input_dim %d -> %d)",
        old_path,
        new_path,
        old_input_dim,
        new_input_dim,
    )
