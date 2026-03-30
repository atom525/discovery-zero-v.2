"""Persistent store of execution-verified helper code snippets."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class VerifiedFunction:
    name: str
    signature: str
    docstring: str
    code: str
    success: bool
    summary: str


class VerifiedCodeLibrary:
    def __init__(self, library_path: Path) -> None:
        self.library_path = library_path
        self.entries: dict[str, VerifiedFunction] = {}
        self._load()

    def _load(self) -> None:
        if not self.library_path.exists():
            self.entries = {}
            return
        raw = json.loads(self.library_path.read_text(encoding="utf-8"))
        loaded: dict[str, VerifiedFunction] = {}
        if isinstance(raw, dict):
            for key, item in raw.items():
                if not isinstance(item, dict):
                    continue
                loaded[key] = VerifiedFunction(
                    name=str(item.get("name", key)),
                    signature=str(item.get("signature", "")),
                    docstring=str(item.get("docstring", "")),
                    code=str(item.get("code", "")),
                    success=bool(item.get("success", False)),
                    summary=str(item.get("summary", "")),
                )
        self.entries = loaded

    def _save(self) -> None:
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {key: asdict(value) for key, value in self.entries.items()}
        self.library_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_verified(
        self,
        *,
        func_name: str,
        code: str,
        signature: str,
        docstring: str,
        success: bool,
        summary: str,
    ) -> None:
        if not success:
            return
        self.entries[func_name] = VerifiedFunction(
            name=func_name,
            signature=signature,
            docstring=docstring,
            code=code,
            success=success,
            summary=summary,
        )
        self._save()

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token for token in text.lower().replace("\n", " ").split(" ") if token}

    def retrieve_relevant(self, task_description: str, top_k: int = 5) -> list[VerifiedFunction]:
        query = self._tokenize(task_description)
        scored: list[tuple[float, VerifiedFunction]] = []
        for item in self.entries.values():
            tokens = self._tokenize(f"{item.name} {item.signature} {item.docstring} {item.summary} {item.code}")
            if not tokens:
                continue
            overlap = len(query & tokens)
            denom = math.sqrt(max(1, len(query))) * math.sqrt(max(1, len(tokens)))
            score = overlap / denom
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [item for _, item in scored[:top_k] if _ > 0]

    def get_import_block(self, func_names: list[str]) -> str:
        lines = []
        for name in func_names:
            entry: Optional[VerifiedFunction] = self.entries.get(name)
            if entry is None:
                continue
            lines.append(f"# verified helper: {name}")
            lines.append(entry.code)
        return "\n\n".join(lines)
