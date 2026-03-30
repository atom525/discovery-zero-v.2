"""
Semantic + graph-structure retrieval over the reasoning hypergraph.

This module is intentionally lightweight:
  - semantic retrieval uses a pluggable embedding client (remote BGE-M3 API or
    any local object exposing ``embed(list[str])``)
  - structural retrieval boosts semantically similar nodes that are also near
    the active target in the hypergraph
  - when no embedding service is configured, retrieval gracefully returns no
    results instead of blocking the main search loop
"""

from __future__ import annotations

import json
import math
import asyncio
import inspect
import urllib.error
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from discovery_zero.graph.models import HyperGraph

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is expected but kept optional
    np = None  # type: ignore


@dataclass
class RetrievalConfig:
    max_results: int = 6
    min_similarity: float = 0.15
    graph_proximity_weight: float = 0.25
    max_hops: int = 4
    min_node_belief: float = 0.6
    include_proven_only: bool = False
    embedding_api_base: str = ""
    request_timeout_seconds: float = 20.0
    use_gaia_storage: bool = False
    gaia_vector_top_k: int = 8


@dataclass
class RetrievalResult:
    node_id: str
    statement: str
    belief: float
    semantic_similarity: float
    graph_proximity: float
    combined_score: float
    provenance: Optional[str] = None


class RemoteEmbeddingClient:
    """Tiny HTTP client for a simple ``POST /embed`` JSON endpoint."""

    def __init__(self, api_base: str, *, timeout_seconds: float = 20.0) -> None:
        self._api_base = api_base.rstrip("/")
        self._timeout = timeout_seconds

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = json.dumps({"texts": texts}).encode("utf-8")
        req = urllib.request.Request(
            f"{self._api_base}/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        embeddings = body.get("embeddings", [])
        return embeddings if isinstance(embeddings, list) else []


class GaiaStorageRetrievalClient:
    """Optional retrieval against Gaia's vector storage manager."""

    def __init__(self) -> None:
        self._manager = None
        self._embedding_model = None
        try:
            from libs.storage.config import StorageConfig
            from libs.storage.manager import StorageManager
            from libs.embedding import DPEmbeddingModel
            self._manager = StorageManager(StorageConfig())
            self._embedding_model = DPEmbeddingModel()
        except Exception:
            self._manager = None
            self._embedding_model = None

    @property
    def available(self) -> bool:
        return self._manager is not None and self._embedding_model is not None

    def search(self, query: str, top_k: int) -> list[tuple[str, str, float]]:
        if not self.available:
            return []

        async def _search_async() -> list[tuple[str, str, float]]:
            query_embs = await self._embedding_model.embed([query])
            if not query_embs:
                return []
            scored = await self._manager.search_vector(query_embs[0], top_k=top_k)
            results: list[tuple[str, str, float]] = []
            for item in scored:
                knowledge = getattr(item, "knowledge", None)
                if knowledge is None:
                    continue
                statement = str(getattr(knowledge, "content", "") or "")
                knowledge_id = str(getattr(knowledge, "knowledge_id", "") or "")
                score = float(getattr(item, "score", 0.0))
                if statement and knowledge_id:
                    results.append((knowledge_id, statement, score))
            return results

        try:
            return asyncio.run(_search_async())
        except Exception:
            return []


class HypergraphRetrievalIndex:
    """Index proven/high-belief nodes for retrieval-augmented prompting."""

    def __init__(
        self,
        *,
        config: Optional[RetrievalConfig] = None,
        embedding_model: Any = None,
        gaia_client: Any = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        self._embedding_model = embedding_model
        self._gaia_client = gaia_client  # Optional GaiaClient for global graph search
        self._node_ids: list[str] = []
        self._statements: list[str] = []
        self._beliefs: list[float] = []
        self._provenance: list[Optional[str]] = []
        self._matrix: Any = None
        self._gaia = GaiaStorageRetrievalClient() if self.config.use_gaia_storage else None

    @property
    def ready(self) -> bool:
        return bool(self._node_ids) and self._matrix is not None

    def build_from_graph(self, graph: HyperGraph) -> int:
        candidates: list[tuple[str, str, float, Optional[str]]] = []
        for node_id, node in graph.nodes.items():
            if not node.statement.strip():
                continue
            if self.config.include_proven_only:
                if node.state != "proven":
                    continue
            else:
                if node.state != "proven" and float(node.belief) < self.config.min_node_belief:
                    continue
            candidates.append((node_id, node.statement, float(node.belief), node.provenance))

        self._node_ids = [item[0] for item in candidates]
        self._statements = [item[1] for item in candidates]
        self._beliefs = [item[2] for item in candidates]
        self._provenance = [item[3] for item in candidates]
        self._matrix = None

        if not candidates:
            return 0

        try:
            vectors = self._embed_texts(self._statements)
        except Exception:
            self._node_ids = []
            self._statements = []
            self._beliefs = []
            self._provenance = []
            self._matrix = None
            return 0

        if not vectors:
            self._node_ids = []
            self._statements = []
            self._beliefs = []
            self._provenance = []
            self._matrix = None
            return 0

        self._matrix = self._normalize_rows(vectors)
        return len(self._node_ids)

    def retrieve(
        self,
        query: str,
        *,
        graph: HyperGraph,
        target_node_id: Optional[str] = None,
        exclude_node_ids: Optional[set[str]] = None,
        max_results: Optional[int] = None,
    ) -> list[RetrievalResult]:
        if not self.ready or not query.strip():
            return []

        try:
            query_vector_list = self._embed_texts([query])
        except Exception:
            return []
        if not query_vector_list:
            return []
        query_vector = self._normalize_vector(query_vector_list[0])
        exclude = exclude_node_ids or set()
        scores = self._semantic_scores(query_vector)
        limit = max_results or self.config.max_results
        results: list[RetrievalResult] = []
        for idx, semantic_similarity in scores:
            node_id = self._node_ids[idx]
            if node_id in exclude:
                continue
            if semantic_similarity < self.config.min_similarity:
                continue
            proximity = self._graph_proximity(graph, target_node_id, node_id)
            combined = (
                (1.0 - self.config.graph_proximity_weight) * semantic_similarity
                + self.config.graph_proximity_weight * proximity
            )
            results.append(
                RetrievalResult(
                    node_id=node_id,
                    statement=self._statements[idx],
                    belief=self._beliefs[idx],
                    semantic_similarity=round(float(semantic_similarity), 6),
                    graph_proximity=round(float(proximity), 6),
                    combined_score=round(float(combined), 6),
                    provenance=self._provenance[idx],
                )
            )
        if self._gaia is not None and self._gaia.available:
            gaia_hits = self._gaia.search(query=query, top_k=self.config.gaia_vector_top_k)
            for _, statement, score in gaia_hits:
                node_ids = graph.find_node_ids_by_statement(statement)
                if len(node_ids) != 1:
                    continue
                node_id = node_ids[0]
                if node_id in exclude:
                    continue
                node = graph.nodes.get(node_id)
                if node is None:
                    continue
                proximity = self._graph_proximity(graph, target_node_id, node_id)
                semantic = max(self.config.min_similarity, min(1.0, score))
                combined = (
                    (1.0 - self.config.graph_proximity_weight) * semantic
                    + self.config.graph_proximity_weight * proximity
                )
                results.append(
                    RetrievalResult(
                        node_id=node_id,
                        statement=node.statement,
                        belief=float(node.belief),
                        semantic_similarity=round(float(semantic), 6),
                        graph_proximity=round(float(proximity), 6),
                        combined_score=round(float(combined), 6),
                        provenance="gaia_vector_store",
                    )
                )
        # Optional: query Gaia global graph when a GaiaClient is configured.
        # This pulls in verified facts from previous runs or the global knowledge graph.
        if self._gaia_client is not None:
            try:
                global_hits = self._gaia_client.search_global_graph(
                    query=query,
                    top_k=self.config.gaia_vector_top_k,
                )
                for hit in global_hits:
                    if not isinstance(hit, dict):
                        continue
                    statement = str(hit.get("representative_content", "") or "").strip()
                    g_id = str(hit.get("global_canonical_id", "") or "").strip()
                    if not statement or not g_id:
                        continue
                    node_ids_local = graph.find_node_ids_by_statement(statement)
                    node_id = node_ids_local[0] if len(node_ids_local) == 1 else f"gaia:{g_id}"
                    if node_id in exclude:
                        continue
                    proximity = self._graph_proximity(graph, target_node_id, node_id) if node_id in graph.nodes else 0.0
                    score = float(hit.get("score", self.config.min_similarity))
                    combined = (
                        (1.0 - self.config.graph_proximity_weight) * score
                        + self.config.graph_proximity_weight * proximity
                    )
                    belief = float(graph.nodes[node_id].belief) if node_id in graph.nodes else 0.8
                    results.append(
                        RetrievalResult(
                            node_id=node_id,
                            statement=statement,
                            belief=belief,
                            semantic_similarity=round(float(score), 6),
                            graph_proximity=round(float(proximity), 6),
                            combined_score=round(float(combined), 6),
                            provenance="gaia_global_graph",
                        )
                    )
            except Exception:
                pass

        results.sort(key=lambda item: item.combined_score, reverse=True)
        deduped: list[RetrievalResult] = []
        seen: set[str] = set()
        for item in results:
            if item.node_id in seen:
                continue
            deduped.append(item)
            seen.add(item.node_id)
            if len(deduped) >= limit:
                break
        return deduped

    def format_retrieval_context(
        self,
        results: list[RetrievalResult],
        graph: HyperGraph,
    ) -> str:
        if not results:
            return ""
        lines = ["Possible relevant established results:"]
        for i, item in enumerate(results, start=1):
            node = graph.nodes.get(item.node_id)
            state = node.state if node is not None else "unknown"
            provenance = f" via {item.provenance}" if item.provenance else ""
            lines.append(
                f"{i}. [{item.node_id}] belief={item.belief:.2f} state={state} "
                f"semantic={item.semantic_similarity:.2f} graph={item.graph_proximity:.2f}{provenance}"
            )
            lines.append(f"   {item.statement}")
        return "\n".join(lines)

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if self._embedding_model is not None:
            if hasattr(self._embedding_model, "embed"):
                embedded = self._embedding_model.embed(texts)
                if inspect.isawaitable(embedded):
                    embedded = asyncio.run(embedded)
                return [list(map(float, row)) for row in embedded]
        if self.config.embedding_api_base:
            client = RemoteEmbeddingClient(
                self.config.embedding_api_base,
                timeout_seconds=self.config.request_timeout_seconds,
            )
            embedded = client.embed(texts)
            return [list(map(float, row)) for row in embedded]
        raise RuntimeError("No embedding model configured for retrieval.")

    def _semantic_scores(self, query_vector: list[float]) -> list[tuple[int, float]]:
        if self._matrix is None:
            return []
        if np is not None:
            query_arr = np.asarray(query_vector, dtype=float)
            sims = self._matrix @ query_arr
            order = np.argsort(-sims)
            return [(int(idx), float(sims[idx])) for idx in order]
        scored: list[tuple[int, float]] = []
        for idx, row in enumerate(self._matrix):
            score = sum(float(a) * float(b) for a, b in zip(row, query_vector))
            scored.append((idx, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored

    def _graph_proximity(
        self,
        graph: HyperGraph,
        source_node_id: Optional[str],
        target_node_id: str,
    ) -> float:
        if not source_node_id or source_node_id not in graph.nodes or target_node_id not in graph.nodes:
            return 0.0
        if source_node_id == target_node_id:
            return 1.0
        frontier = deque([(source_node_id, 0)])
        seen = {source_node_id}
        while frontier:
            current, dist = frontier.popleft()
            if dist >= self.config.max_hops:
                continue
            neighbors = self._neighbors(graph, current)
            for nxt in neighbors:
                if nxt in seen:
                    continue
                if nxt == target_node_id:
                    return max(0.0, 1.0 - (dist + 1) / max(self.config.max_hops, 1))
                seen.add(nxt)
                frontier.append((nxt, dist + 1))
        return 0.0

    def _neighbors(self, graph: HyperGraph, node_id: str) -> set[str]:
        out: set[str] = set()
        for edge_id in graph.get_edges_to(node_id):
            edge = graph.edges.get(edge_id)
            if edge is None:
                continue
            out.update(edge.premise_ids)
        for edge_id in graph.get_edges_from(node_id):
            edge = graph.edges.get(edge_id)
            if edge is None:
                continue
            out.add(edge.conclusion_id)
            out.update(edge.premise_ids)
        out.discard(node_id)
        return out

    def _normalize_rows(self, vectors: list[list[float]]) -> Any:
        if np is not None:
            arr = np.asarray(vectors, dtype=float)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return arr / norms
        return [self._normalize_vector(row) for row in vectors]

    def _normalize_vector(self, vector: Iterable[float]) -> list[float]:
        vals = [float(v) for v in vector]
        norm = math.sqrt(sum(v * v for v in vals))
        if norm <= 1e-12:
            return vals
        return [v / norm for v in vals]
