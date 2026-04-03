"""MCP server exposing Discovery Zero modular capabilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from mcp.server.fastmcp import FastMCP

from dz_engine import MCTSConfig, run_discovery
from dz_hypergraph import analyze_belief_gaps, load_graph, propagate_beliefs, save_graph
from dz_verify import extract_claims, verify_claims

mcp = FastMCP("dz")


def _load_graph_from_input(graph_json_or_path: str):
    raw = (graph_json_or_path or "").strip()
    if not raw:
        raise ValueError("graph_json_or_path must not be empty")
    if raw.startswith("{"):
        from dz_hypergraph.models import HyperGraph

        return HyperGraph.model_validate_json(raw), None
    path = Path(raw)
    graph = load_graph(path)
    return graph, path


@mcp.tool()
def dz_extract_claims(
    prose: str,
    context: str,
    source_memo_id: str,
    model: Optional[str] = None,
) -> dict[str, Any]:
    claims = extract_claims(
        prose=prose,
        context=context,
        source_memo_id=source_memo_id,
        model=model,
    )
    return {"claims": [item.model_dump() for item in claims]}


@mcp.tool()
def dz_verify_claims(
    prose: str,
    context: str,
    graph_json_or_path: str,
    source_memo_id: str,
    model: Optional[str] = None,
) -> dict[str, Any]:
    graph, path = _load_graph_from_input(graph_json_or_path)
    summary = verify_claims(
        prose=prose,
        context=context,
        graph=graph,
        source_memo_id=source_memo_id,
        model=model,
    )
    if path is not None:
        save_graph(graph, path)
    return {
        "claims": [item.model_dump() for item in summary.claims],
        "results": [item.__dict__ for item in summary.results],
    }


@mcp.tool()
def dz_analyze_gaps(
    graph_json_or_path: str,
    target_node_id: str,
    top_k: int = 5,
) -> dict[str, Any]:
    graph, _ = _load_graph_from_input(graph_json_or_path)
    gaps = analyze_belief_gaps(graph, target_node_id=target_node_id, top_k=top_k)
    return {"gaps": [{"node_id": nid, "gain": gain} for nid, gain in gaps]}


@mcp.tool()
def dz_propagate_beliefs(
    graph_json_or_path: str,
    max_iterations: int = 50,
    damping: float = 0.5,
    tol: float = 1e-6,
) -> dict[str, Any]:
    graph, path = _load_graph_from_input(graph_json_or_path)
    iterations = propagate_beliefs(
        graph,
        max_iterations=max_iterations,
        damping=damping,
        tol=tol,
    )
    if path is not None:
        save_graph(graph, path)
    return {
        "iterations": iterations,
        "num_nodes": len(graph.nodes),
        "num_edges": len(graph.edges),
    }


@mcp.tool()
def dz_load_graph(path: str) -> dict[str, Any]:
    graph = load_graph(Path(path))
    return {
        "summary": graph.summary(),
        "num_nodes": len(graph.nodes),
        "num_edges": len(graph.edges),
    }


@mcp.tool()
def dz_run_discovery(
    graph_path: str,
    target_node_id: str,
    config_json: Optional[str] = None,
    model: Optional[str] = None,
) -> dict[str, Any]:
    config_data = json.loads(config_json) if config_json else {}
    config = MCTSConfig(**config_data)
    result = run_discovery(
        graph_path=Path(graph_path),
        target_node_id=target_node_id,
        config=config,
        model=model,
    )
    return {
        "success": result.success,
        "iterations_completed": result.iterations_completed,
        "target_belief_initial": result.target_belief_initial,
        "target_belief_final": result.target_belief_final,
    }


def main() -> None:
    """Run MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
