"""Core engine facade for Discovery Zero."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from dz_engine.mcts_engine import MCTSConfig, MCTSDiscoveryEngine
from dz_hypergraph.models import HyperGraph


def run_discovery(
    *,
    graph_path: Path,
    target_node_id: str,
    config: Optional[MCTSConfig] = None,
    model: Optional[str] = None,
    backend: str = "bp",
    kwargs: Optional[dict[str, Any]] = None,
):
    """Run MCTS discovery with a minimal high-level API."""
    engine = MCTSDiscoveryEngine(
        graph_path=graph_path,
        target_node_id=target_node_id,
        config=config or MCTSConfig(),
        model=model,
        backend=backend,
        **(kwargs or {}),
    )
    return engine.run()


__all__ = [
    "MCTSConfig",
    "MCTSDiscoveryEngine",
    "run_discovery",
    "HyperGraph",
]
