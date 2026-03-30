"""
Neural Enhanced Belief Propagation — FG-GNN Message Correction Layer.

Original innovation: applies Garcia Satorras & Welling (2021) Neural Enhanced BP
to the *mathematical discovery hypergraph* — a dynamically growing factor graph
with heterogeneous edge types and dual-state (proven/refuted/unverified) node
semantics.  Prior work targets static LDPC/Ising graphs; our setting is unique.

Architecture:
  1. Gaia's loopy BP runs one iteration to compute raw variable-to-factor and
     factor-to-variable messages.
  2. An FG-GNN reads node features, factor features, and current messages.
  3. It outputs per-message correction deltas Δ.
  4. Corrected messages = raw + alpha * Δ (alpha = correction_strength).
  5. Gaia BP continues with corrected messages as the new starting point.

Integration points with Gaia:
  - `FactorGraph` / `BeliefPropagation` from libs/inference/
  - `EmbeddingModel` from libs/embedding.py for semantic node features
  - `run_gaia_bp()` in graph/adapter.py wraps the integration

Training:
  - BPTrainingCollector harvests (graph_snapshot, standard_beliefs, actual_beliefs)
    triples from benchmark run logs.
  - The correction network is trained to minimise MSE between corrected beliefs
    and final confirmed beliefs.

PyTorch is an optional dependency.  When not installed the module falls back
to identity (no correction), so the rest of the system is unaffected.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from discovery_zero.graph.models import HyperGraph

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Optional PyTorch import                                              #
# ------------------------------------------------------------------ #

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
# Feature extraction                                                   #
# ------------------------------------------------------------------ #

@dataclass
class GraphEncoding:
    """Encoded feature matrices for one HyperGraph."""

    node_features: "Optional[torch.Tensor]"
    """Shape: (num_nodes, node_feature_dim)"""

    factor_features: "Optional[torch.Tensor]"
    """Shape: (num_edges, factor_feature_dim)"""

    node_ids: List[str] = field(default_factory=list)
    edge_ids: List[str] = field(default_factory=list)
    node_index: Dict[str, int] = field(default_factory=dict)
    edge_index: Dict[str, int] = field(default_factory=dict)


TABULAR_NODE_FEATURE_DIM = 8  # belief, prior, state_onehot(3), degree_in, degree_out, has_formal
FACTOR_FEATURE_DIM = 6  # confidence, review_confidence, edge_type_onehot(3), num_premises

# When EmbeddingModel is available the node feature dim extends by the embedding dim.
# We keep this mutable so it adjusts dynamically.
_embedding_dim: int = 0


def _get_node_feature_dim() -> int:
    return TABULAR_NODE_FEATURE_DIM + _embedding_dim


# Backward-compat alias
NODE_FEATURE_DIM = TABULAR_NODE_FEATURE_DIM


def _run_embedding_sync(model: Any, texts: list[str]) -> list[list[float]]:
    """Run an async EmbeddingModel.embed call synchronously."""
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, model.embed(texts)).result()
    return asyncio.run(model.embed(texts))


def encode_graph_for_nn(
    graph: HyperGraph,
    *,
    embedding_model: Any = None,
) -> GraphEncoding:
    """
    Encode a HyperGraph into fixed-size feature matrices for the FG-GNN.

    Node features (tabular): [belief, prior, state_proven, state_refuted,
        state_unverified, degree_in, degree_out, has_formal_statement]

    When ``embedding_model`` (a Gaia ``EmbeddingModel``) is provided, each
    node's ``statement`` is embedded and the embedding vector is concatenated
    to the tabular features, yielding richer semantic representations.

    Factor features: [confidence, review_confidence, type_heuristic, type_formal,
                      type_decomposition, num_premises_normalised]
    """
    global _embedding_dim, NODE_FEATURE_DIM

    if not _TORCH_AVAILABLE:
        return GraphEncoding(node_features=None, factor_features=None)

    node_ids = list(graph.nodes.keys())
    edge_ids = list(graph.edges.keys())
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    edge_index = {eid: i for i, eid in enumerate(edge_ids)}

    n = len(node_ids)
    e = len(edge_ids)

    # Optionally compute semantic embeddings via Gaia EmbeddingModel
    embeddings: Optional[list[list[float]]] = None
    if embedding_model is not None and n > 0:
        try:
            texts = [graph.nodes[nid].statement for nid in node_ids]
            embeddings = _run_embedding_sync(embedding_model, texts)
            if embeddings and len(embeddings[0]) != _embedding_dim:
                _embedding_dim = len(embeddings[0])
                NODE_FEATURE_DIM = TABULAR_NODE_FEATURE_DIM + _embedding_dim
        except Exception:
            logger.debug("EmbeddingModel call failed; falling back to tabular features")
            embeddings = None

    effective_dim = TABULAR_NODE_FEATURE_DIM + (_embedding_dim if embeddings else 0)
    node_feats = torch.zeros(n, effective_dim)
    for i, nid in enumerate(node_ids):
        node = graph.nodes[nid]
        node_feats[i, 0] = node.belief
        node_feats[i, 1] = node.prior
        node_feats[i, 2] = 1.0 if node.state == "proven" else 0.0
        node_feats[i, 3] = 1.0 if node.state == "refuted" else 0.0
        node_feats[i, 4] = 1.0 if node.state == "unverified" else 0.0
        node_feats[i, 5] = min(1.0, len(graph.get_edges_to(nid)) / 5.0)
        node_feats[i, 6] = min(1.0, len(graph.get_edges_from(nid)) / 5.0)
        node_feats[i, 7] = 1.0 if node.formal_statement else 0.0
        if embeddings:
            for k, v in enumerate(embeddings[i]):
                node_feats[i, TABULAR_NODE_FEATURE_DIM + k] = v

    factor_feats = torch.zeros(e, FACTOR_FEATURE_DIM)
    for j, eid in enumerate(edge_ids):
        edge = graph.edges[eid]
        factor_feats[j, 0] = edge.confidence
        factor_feats[j, 1] = edge.review_confidence if edge.review_confidence is not None else 0.5
        factor_feats[j, 2] = 1.0 if edge.edge_type == "heuristic" else 0.0
        factor_feats[j, 3] = 1.0 if edge.edge_type == "formal" else 0.0
        factor_feats[j, 4] = 1.0 if edge.edge_type == "decomposition" else 0.0
        factor_feats[j, 5] = min(1.0, len(edge.premise_ids) / 5.0)

    return GraphEncoding(
        node_features=node_feats,
        factor_features=factor_feats,
        node_ids=node_ids,
        edge_ids=edge_ids,
        node_index=node_index,
        edge_index=edge_index,
    )


# ------------------------------------------------------------------ #
# FG-GNN Correction Network                                           #
# ------------------------------------------------------------------ #

if _TORCH_AVAILABLE:
    class _MessageCorrectionLayer(nn.Module):
        """One layer of FG-GNN message passing."""

        def __init__(self, hidden_dim: int, node_feat_dim: int = TABULAR_NODE_FEATURE_DIM) -> None:
            super().__init__()
            self.node_update = nn.Linear(hidden_dim + node_feat_dim, hidden_dim)
            self.factor_update = nn.Linear(hidden_dim + FACTOR_FEATURE_DIM, hidden_dim)
            self.message_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(
            self,
            node_h: "torch.Tensor",       # (N, H)
            factor_h: "torch.Tensor",     # (E, H)
            node_feats: "torch.Tensor",   # (N, node_F)
            factor_feats: "torch.Tensor", # (E, factor_F)
            premise_indices: List[List[int]],  # [edge_idx] -> [premise_node_indices]
            conclusion_indices: List[int],     # [edge_idx] -> conclusion_node_idx
        ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """Returns updated (node_h, factor_h, message_corrections)."""

            E = factor_h.shape[0]
            # Factor → Node aggregation
            agg = torch.zeros_like(node_h)
            for j in range(E):
                # Message from factor j to conclusion node
                cidx = conclusion_indices[j]
                msg_in = torch.cat([factor_h[j], node_h[cidx]], dim=0)
                msg = self.message_mlp(msg_in)
                agg[cidx] = agg[cidx] + msg.squeeze(-1) * factor_feats[j, 0]

                # Messages from factor j to each premise
                for pidx in premise_indices[j]:
                    msg_in_p = torch.cat([factor_h[j], node_h[pidx]], dim=0)
                    msg_p = self.message_mlp(msg_in_p)
                    agg[pidx] = agg[pidx] + msg_p.squeeze(-1) * factor_feats[j, 0]

            # Update node hidden states
            new_node_h = F.relu(self.node_update(torch.cat([node_h, node_feats], dim=-1)))
            # Update factor hidden states
            new_factor_h = F.relu(self.factor_update(torch.cat([factor_h, factor_feats], dim=-1)))

            # Compute per-factor correction delta
            corrections = torch.zeros(E, 1)
            for j in range(E):
                cidx = conclusion_indices[j]
                corr_in = torch.cat([new_factor_h[j], new_node_h[cidx]], dim=0)
                corrections[j] = self.message_mlp(corr_in)

            return new_node_h, new_factor_h, corrections

    class _NeuralBPCorrectorNet(nn.Module):
        """FG-GNN correction network."""

        def __init__(
            self,
            hidden_dim: int = 64,
            num_layers: int = 3,
            node_feat_dim: int = TABULAR_NODE_FEATURE_DIM,
        ) -> None:
            super().__init__()
            self.node_feat_dim = node_feat_dim
            self.node_embed = nn.Linear(node_feat_dim, hidden_dim)
            self.factor_embed = nn.Linear(FACTOR_FEATURE_DIM, hidden_dim)
            self.layers = nn.ModuleList([
                _MessageCorrectionLayer(hidden_dim, node_feat_dim)
                for _ in range(num_layers)
            ])
            self.output_head = nn.Linear(hidden_dim, 1)

        def forward(
            self,
            node_feats: "torch.Tensor",
            factor_feats: "torch.Tensor",
            premise_indices: List[List[int]],
            conclusion_indices: List[int],
        ) -> "torch.Tensor":
            """Returns belief corrections of shape (N,) in [-1, 1]."""
            node_h = F.relu(self.node_embed(node_feats))
            factor_h = F.relu(self.factor_embed(factor_feats))
            for layer in self.layers:
                node_h, factor_h, _ = layer(
                    node_h, factor_h, node_feats, factor_feats,
                    premise_indices, conclusion_indices
                )
            corrections = torch.tanh(self.output_head(node_h))  # (N, 1)
            return corrections.squeeze(-1)  # (N,)

else:
    _NeuralBPCorrectorNet = None  # type: ignore


# ------------------------------------------------------------------ #
# NeuralBPCorrector — public interface                                 #
# ------------------------------------------------------------------ #

class NeuralBPCorrector:
    """
    Learned message correction on top of Gaia's factor graph BP.

    At each BP iteration:
      1. Gaia BP computes standard messages / beliefs.
      2. FG-GNN reads graph features and outputs per-node belief corrections.
      3. Corrected belief = clip(standard_belief + alpha * correction, 0, 1).
      4. Corrected beliefs are passed back as warmstart for the next iteration.

    When alpha=0 (cold start) this is identical to standard Gaia BP.
    The alpha is gradually increased as the correction network proves reliable.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        correction_strength: float = 0.0,  # alpha: 0 = disabled, 1 = full
        model_path: Optional[Path] = None,
        embedding_model: Optional[Any] = None,
    ) -> None:
        self.correction_strength = correction_strength
        self._model_path = model_path
        self._embedding_model = embedding_model
        self._net: Optional[Any] = None

        if _TORCH_AVAILABLE and correction_strength > 0:
            self._net = _NeuralBPCorrectorNet(hidden_dim, num_layers)
            if model_path is not None and model_path.exists():
                try:
                    self._net.load_state_dict(
                        torch.load(model_path, map_location="cpu", weights_only=True)
                    )
                    self._net.eval()
                    logger.info("Loaded Neural BP model from %s", model_path)
                except Exception as exc:
                    logger.warning("Failed to load Neural BP model: %s", exc)
                    self._net = None

    @property
    def is_active(self) -> bool:
        return _TORCH_AVAILABLE and self._net is not None and self.correction_strength > 0

    def correct_beliefs(
        self,
        graph: HyperGraph,
        standard_beliefs: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Apply learned corrections to standard BP beliefs.

        Args:
            graph: The reasoning hypergraph.
            standard_beliefs: {node_id: belief} from standard Gaia BP.

        Returns:
            Corrected {node_id: belief} dict.
        """
        if not self.is_active:
            return standard_beliefs

        enc = encode_graph_for_nn(graph, embedding_model=self._embedding_model)
        if enc.node_features is None:
            return standard_beliefs

        node_feats = enc.node_features
        factor_feats = enc.factor_features

        premise_indices: List[List[int]] = []
        conclusion_indices: List[int] = []
        for eid in enc.edge_ids:
            edge = graph.edges[eid]
            premise_indices.append([
                enc.node_index[pid]
                for pid in edge.premise_ids
                if pid in enc.node_index
            ])
            cidx = enc.node_index.get(edge.conclusion_id, 0)
            conclusion_indices.append(cidx)

        with torch.no_grad():
            corrections = self._net(node_feats, factor_feats, premise_indices, conclusion_indices)

        corrected: Dict[str, float] = {}
        for nid, belief in standard_beliefs.items():
            if nid not in enc.node_index:
                corrected[nid] = belief
                continue
            i = enc.node_index[nid]
            delta = float(corrections[i]) * self.correction_strength
            corrected[nid] = max(0.0, min(1.0, belief + delta))

        return corrected

    def apply_to_graph(self, graph: HyperGraph) -> None:
        """Apply corrections in-place to the graph's beliefs."""
        standard = {nid: node.belief for nid, node in graph.nodes.items()}
        corrected = self.correct_beliefs(graph, standard)
        for nid, belief in corrected.items():
            node = graph.nodes.get(nid)
            if node and not node.is_locked():
                node.belief = belief

    def save(self, path: Path) -> None:
        if self._net is not None and _TORCH_AVAILABLE:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._net.state_dict(), path)

    def train_step(
        self,
        batch: List["BPTrainingSample"],
        optimizer: Any,
        *,
        loss_fn: Any = None,
    ) -> float:
        """One gradient step on a batch of training samples."""
        if not _TORCH_AVAILABLE or self._net is None:
            return 0.0
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        total_loss = 0.0
        optimizer.zero_grad()
        for sample in batch:
            enc = encode_graph_for_nn(sample.graph)
            if enc.node_features is None:
                continue

            premise_indices = []
            conclusion_indices = []
            for eid in enc.edge_ids:
                edge = sample.graph.edges[eid]
                premise_indices.append([
                    enc.node_index[p] for p in edge.premise_ids if p in enc.node_index
                ])
                conclusion_indices.append(enc.node_index.get(edge.conclusion_id, 0))

            predictions = self._net(
                enc.node_features, enc.factor_features,
                premise_indices, conclusion_indices
            )

            # Target: difference between true beliefs and standard BP beliefs
            targets = torch.zeros(len(enc.node_ids))
            for i, nid in enumerate(enc.node_ids):
                std = sample.standard_beliefs.get(nid, 0.5)
                true_b = sample.true_beliefs.get(nid, std)
                targets[i] = (true_b - std) / max(self.correction_strength, 0.01)

            loss = loss_fn(predictions, targets)
            loss.backward()
            total_loss += loss.item()

        if batch:
            optimizer.step()
            return total_loss / len(batch)
        return 0.0


# ------------------------------------------------------------------ #
# Training data collection                                             #
# ------------------------------------------------------------------ #

@dataclass
class BPTrainingSample:
    """One training example for the Neural BP corrector."""

    graph: HyperGraph
    """Graph state at snapshot time."""

    standard_beliefs: Dict[str, float]
    """Beliefs from standard Gaia BP at this snapshot."""

    true_beliefs: Dict[str, float]
    """Final beliefs after full exploration + Lean verification (ground truth)."""

    run_id: str = ""
    node_id: str = ""  # the target node for this run


class BPTrainingCollector:
    """
    Harvest (graph_state, standard_beliefs, true_beliefs) triples from
    historical benchmark run logs for Neural BP supervised training.

    Collection process:
      - For each completed benchmark run directory:
        1. Load snapshots from exploration_log.json
        2. For each snapshot: re-run standard Gaia BP on the snapshot graph
        3. Load the final graph.json to get confirmed beliefs (ground truth)
        4. Yield BPTrainingSample(snapshot_graph, standard_bp_beliefs, final_beliefs)
    """

    def collect_from_run_dirs(
        self,
        run_dirs: List[Path],
        *,
        max_samples: int = 10_000,
    ) -> List[BPTrainingSample]:
        samples: List[BPTrainingSample] = []

        for run_dir in run_dirs:
            if len(samples) >= max_samples:
                break
            try:
                new_samples = self._collect_from_run(run_dir)
                samples.extend(new_samples)
            except Exception as exc:
                logger.debug("BPTrainingCollector: skipping %s: %s", run_dir, exc)

        return samples[:max_samples]

    def _collect_from_run(self, run_dir: Path) -> List[BPTrainingSample]:
        from discovery_zero.graph.persistence import load_graph as _load_graph
        from discovery_zero.graph.adapter import run_gaia_bp, write_back_beliefs

        final_graph_path = run_dir / "graph.json"
        if not final_graph_path.exists():
            return []

        final_graph = _load_graph(final_graph_path)
        true_beliefs = {nid: node.belief for nid, node in final_graph.nodes.items()}

        # Load snapshots from log
        log_path = run_dir / "exploration_log.json"
        if not log_path.exists():
            return []

        samples: List[BPTrainingSample] = []
        log_lines = log_path.read_text(encoding="utf-8").splitlines()

        for line in log_lines:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                snapshot_str = entry.get("graph_snapshot")
                if not snapshot_str:
                    continue
                snap_graph = HyperGraph.model_validate_json(snapshot_str)

                # Run standard Gaia BP on snapshot
                result = run_gaia_bp(snap_graph)
                write_back_beliefs(snap_graph, result)
                standard = {nid: node.belief for nid, node in snap_graph.nodes.items()}

                samples.append(BPTrainingSample(
                    graph=snap_graph,
                    standard_beliefs=standard,
                    true_beliefs={
                        nid: true_beliefs.get(nid, standard.get(nid, 0.5))
                        for nid in snap_graph.nodes
                    },
                    run_id=str(run_dir),
                    node_id=entry.get("target_node_id", ""),
                ))
            except Exception:
                continue

        return samples
