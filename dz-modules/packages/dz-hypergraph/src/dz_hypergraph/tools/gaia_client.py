"""HTTP client for Gaia gateway integration."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class GaiaClientConfig:
    api_base: str
    timeout_seconds: float = 30.0


class GaiaClient:
    def __init__(self, config: GaiaClientConfig) -> None:
        self.config = config

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.config.api_base.rstrip('/')}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get_json(self, path: str) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self.config.api_base.rstrip('/')}{path}",
            headers={"Content-Type": "application/json"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def infer(self, graph_payload: dict[str, Any]) -> dict[str, Any]:
        return self._post_json("/infer", graph_payload)

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._post_json("/embed", {"texts": texts})
        embeddings = result.get("embeddings", [])
        return embeddings if isinstance(embeddings, list) else []

    def search_global_graph(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        result = self._get_json(f"/global-graph/search?query={query}&top_k={top_k}")
        items = result.get("items", [])
        return items if isinstance(items, list) else []


def build_gaia_client(api_base: Optional[str]) -> Optional[GaiaClient]:
    if not api_base:
        return None
    return GaiaClient(GaiaClientConfig(api_base=api_base))
