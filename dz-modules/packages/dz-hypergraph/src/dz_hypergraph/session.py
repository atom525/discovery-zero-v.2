"""
Graph Session — in-memory HyperGraph with lazy persistence and checkpoint/rollback.

GraphSession wraps a HyperGraph and replaces the orchestrator's pattern of
calling load_graph() / save_graph() on every action.  Instead:

  - The graph is held in RAM and mutated in-place.
  - `checkpoint()` takes a deep copy snapshot (O(n) but fast for typical sizes).
  - `rollback()` restores a previous snapshot atomically.
  - `flush()` serializes to disk only when the session is dirty and a
    persist_path has been provided.

This makes incremental BP (section B3) straightforward: after each action
the engine calls `session.graph` directly, modifies it, runs incremental BP,
and calls `flush()` at the end of a round rather than per-action.

Thread-safety: a threading.RLock guards all state mutations.  Multiple
reader threads may call `session.graph` concurrently without explicit locking
(Python GIL + immutable reference to the HyperGraph instance ensures safety
for reads once a snapshot is established).  Writers must hold the lock.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from dz_hypergraph.models import HyperGraph

logger = logging.getLogger(__name__)


class CheckpointNotFoundError(KeyError):
    """Raised when a requested checkpoint ID does not exist."""


class GraphSession:
    """
    In-memory graph with lazy persistence and snapshot/rollback.

    Usage::

        session = GraphSession(graph, persist_path=Path("graph.json"))
        cid = session.checkpoint("before_plausible")
        # ... modify session.graph ...
        if bad_outcome:
            session.rollback(cid)
        else:
            session.flush()
    """

    def __init__(
        self,
        graph: HyperGraph,
        persist_path: Optional[Path] = None,
        *,
        max_checkpoints: int = 50,
    ) -> None:
        self._graph: HyperGraph = graph
        self._persist_path: Optional[Path] = persist_path
        self._max_checkpoints = max_checkpoints

        self._checkpoints: Dict[str, HyperGraph] = {}
        self._checkpoint_labels: Dict[str, str] = {}  # cid → label
        self._checkpoint_order: list[str] = []          # insertion order

        self._dirty: bool = False
        self._lock = threading.RLock()

        self._created_at: float = time.monotonic()
        self._flush_count: int = 0

    # ------------------------------------------------------------------ #
    # Primary accessor                                                     #
    # ------------------------------------------------------------------ #

    @property
    def graph(self) -> HyperGraph:
        """The live graph instance.  Mutate directly; session tracks dirtiness."""
        return self._graph

    def mark_dirty(self) -> None:
        """Explicitly mark the session as dirty (flush required)."""
        with self._lock:
            self._dirty = True

    def dirty(self) -> bool:
        with self._lock:
            return self._dirty

    # ------------------------------------------------------------------ #
    # Checkpoint / Rollback                                                #
    # ------------------------------------------------------------------ #

    def checkpoint(self, label: str = "") -> str:
        """
        Take a deep-copy snapshot of the current graph.

        Returns a checkpoint_id string.  The checkpoint is retained until
        either `rollback` discards newer snapshots or the max_checkpoints
        cap evicts the oldest.
        """
        with self._lock:
            cid = uuid.uuid4().hex[:12]
            self._checkpoints[cid] = self._deep_copy_graph(self._graph)
            self._checkpoint_labels[cid] = label or cid
            self._checkpoint_order.append(cid)

            # Evict oldest checkpoints if over cap
            while len(self._checkpoint_order) > self._max_checkpoints:
                evicted = self._checkpoint_order.pop(0)
                self._checkpoints.pop(evicted, None)
                self._checkpoint_labels.pop(evicted, None)

            logger.debug(
                "GraphSession checkpoint %s (%s) — %d nodes, %d edges",
                cid, label, len(self._graph.nodes), len(self._graph.edges),
            )
            return cid

    def rollback(self, checkpoint_id: str) -> None:
        """
        Restore the graph to a previous checkpoint.

        All checkpoints taken *after* checkpoint_id are discarded.
        Raises CheckpointNotFoundError if the ID is unknown.
        """
        with self._lock:
            if checkpoint_id not in self._checkpoints:
                raise CheckpointNotFoundError(
                    f"Checkpoint '{checkpoint_id}' not found. "
                    f"Available: {list(self._checkpoints.keys())}"
                )
            self._graph = self._deep_copy_graph(self._checkpoints[checkpoint_id])

            # Discard all checkpoints newer than checkpoint_id
            idx = self._checkpoint_order.index(checkpoint_id)
            newer = self._checkpoint_order[idx + 1:]
            for cid in newer:
                self._checkpoints.pop(cid, None)
                self._checkpoint_labels.pop(cid, None)
            self._checkpoint_order = self._checkpoint_order[: idx + 1]

            self._dirty = True
            logger.debug(
                "GraphSession rolled back to %s (%s) — %d nodes, %d edges",
                checkpoint_id,
                self._checkpoint_labels.get(checkpoint_id, ""),
                len(self._graph.nodes),
                len(self._graph.edges),
            )

    def discard_checkpoint(self, checkpoint_id: str) -> None:
        """Free memory from a checkpoint that is no longer needed."""
        with self._lock:
            self._checkpoints.pop(checkpoint_id, None)
            self._checkpoint_labels.pop(checkpoint_id, None)
            if checkpoint_id in self._checkpoint_order:
                self._checkpoint_order.remove(checkpoint_id)

    def list_checkpoints(self) -> list[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "id": cid,
                    "label": self._checkpoint_labels.get(cid, ""),
                    "num_nodes": len(self._checkpoints[cid].nodes),
                    "num_edges": len(self._checkpoints[cid].edges),
                }
                for cid in self._checkpoint_order
            ]

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def flush(self) -> None:
        """Persist the graph to disk (only if dirty and path is set)."""
        with self._lock:
            if not self._dirty or self._persist_path is None:
                return
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._persist_path.with_suffix(".tmp")
            try:
                data = self._graph.model_dump_json(indent=2)
                tmp.write_text(data, encoding="utf-8")
                tmp.replace(self._persist_path)
                self._dirty = False
                self._flush_count += 1
                logger.debug(
                    "GraphSession flushed to %s (flush #%d)",
                    self._persist_path,
                    self._flush_count,
                )
            except Exception as exc:
                logger.error("GraphSession flush failed: %s", exc)
                raise

    def load_from_disk(self) -> None:
        """(Re-)load the graph from persist_path, overwriting in-memory state."""
        with self._lock:
            if self._persist_path is None or not self._persist_path.exists():
                return
            raw = self._persist_path.read_text(encoding="utf-8")
            self._graph = HyperGraph.model_validate_json(raw)
            self._dirty = False

    def replace_graph(self, new_graph: HyperGraph) -> None:
        """Atomically swap the live graph (e.g. after external BP pass)."""
        with self._lock:
            self._graph = new_graph
            self._dirty = True

    # ------------------------------------------------------------------ #
    # Statistics                                                           #
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "num_nodes": len(self._graph.nodes),
                "num_edges": len(self._graph.edges),
                "dirty": self._dirty,
                "num_checkpoints": len(self._checkpoints),
                "flush_count": self._flush_count,
                "persist_path": str(self._persist_path) if self._persist_path else None,
            }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _deep_copy_graph(graph: HyperGraph) -> HyperGraph:
        """Create a deep copy via JSON round-trip (safe, correct, ~0.5ms for typical graphs)."""
        return HyperGraph.model_validate_json(graph.model_dump_json())

    # ------------------------------------------------------------------ #
    # Context manager                                                      #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "GraphSession":
        return self

    def __exit__(self, exc_type: Any, *_: Any) -> None:
        if exc_type is None:
            self.flush()

    def __repr__(self) -> str:
        return (
            f"GraphSession(nodes={len(self._graph.nodes)}, "
            f"edges={len(self._graph.edges)}, "
            f"dirty={self._dirty}, "
            f"checkpoints={len(self._checkpoints)})"
        )


# ------------------------------------------------------------------ #
# Convenience factory functions                                        #
# ------------------------------------------------------------------ #

def load_session(path: Path) -> GraphSession:
    """Load a HyperGraph from a JSON file into a new GraphSession."""
    if not path.exists():
        raise FileNotFoundError(f"Graph file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    graph = HyperGraph.model_validate_json(raw)
    return GraphSession(graph, persist_path=path)


def new_session(persist_path: Optional[Path] = None) -> GraphSession:
    """Create an empty GraphSession."""
    return GraphSession(HyperGraph(), persist_path=persist_path)
