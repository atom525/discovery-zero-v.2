"""
Lean boundary policy enforcement for Discovery Zero.

This module enforces explicit, machine-checkable constraints on generated Lean
code so that historical / knowledge-boundary experiments are not controlled by
prompt text alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re


class LeanPolicyError(RuntimeError):
    """Raised when Lean code violates the configured boundary policy."""


@dataclass
class LeanBoundaryPolicy:
    """
    Explicit constraints for generated Lean code.

    - `allowed_import_prefixes`: every `import ...` must start with one of these
      prefixes. If empty, all imports are allowed.
    - `forbidden_identifiers`: forbidden exact or dotted identifiers.
    - `forbidden_regexes`: regex patterns that must not appear in the source.
    """

    allowed_import_prefixes: list[str] = field(default_factory=list)
    forbidden_identifiers: list[str] = field(default_factory=list)
    forbidden_regexes: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict | None) -> "LeanBoundaryPolicy":
        if not data:
            return cls()
        return cls(
            allowed_import_prefixes=list(data.get("allowed_import_prefixes", [])),
            forbidden_identifiers=list(data.get("forbidden_identifiers", [])),
            forbidden_regexes=list(data.get("forbidden_regexes", [])),
        )

    def prompt_constraints_text(self) -> str:
        lines: list[str] = []
        if self.allowed_import_prefixes:
            joined = ", ".join(self.allowed_import_prefixes)
            lines.append(f"Allowed Lean imports must start with: {joined}.")
        if self.forbidden_identifiers:
            joined = ", ".join(self.forbidden_identifiers)
            lines.append(f"Forbidden Lean identifiers/theorems: {joined}.")
        if self.forbidden_regexes:
            joined = ", ".join(self.forbidden_regexes)
            lines.append(f"Forbidden Lean regex patterns: {joined}.")
        return "\n".join(lines)


IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z0-9_.]+)\s*$", re.MULTILINE)
BLOCK_COMMENT_RE = re.compile(r"/-(?:.|\n)*?-/", re.MULTILINE)
LINE_COMMENT_RE = re.compile(r"--.*$", re.MULTILINE)
STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"', re.DOTALL)
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*")


def _strip_noncode_text(source: str) -> str:
    """Remove comments and string literals before identifier checks."""
    stripped = STRING_RE.sub(" ", source)
    stripped = BLOCK_COMMENT_RE.sub(" ", stripped)
    stripped = LINE_COMMENT_RE.sub(" ", stripped)
    return stripped


def validate_lean_code(source: str, policy: LeanBoundaryPolicy) -> None:
    """Validate generated Lean code against the explicit boundary policy."""
    if policy.allowed_import_prefixes:
        for match in IMPORT_RE.finditer(source):
            module_name = match.group(1)
            if not any(module_name.startswith(prefix) for prefix in policy.allowed_import_prefixes):
                raise LeanPolicyError(
                    f"Import '{module_name}' violates allowed import prefixes: "
                    f"{policy.allowed_import_prefixes}"
                )

    code_only = _strip_noncode_text(source)
    identifiers = set(IDENT_RE.findall(code_only))

    for ident in policy.forbidden_identifiers:
        if ident in identifiers:
            raise LeanPolicyError(
                f"Lean code uses forbidden identifier/theorem '{ident}'."
            )

    for pattern in policy.forbidden_regexes:
        if re.search(pattern, code_only):
            raise LeanPolicyError(
                f"Lean code matches forbidden regex pattern '{pattern}'."
            )
