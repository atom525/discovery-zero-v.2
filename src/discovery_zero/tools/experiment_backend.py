"""
Unified experiment execution backend for Discovery Zero.

Provides a common interface for executing LLM-generated scientific Python code
with three backend strategies:

  - **local**: subprocess with AST validation and extended SAFE_IMPORTS
    (numpy, scipy, sympy, mpmath).  Default and lightest option.
  - **docker**: ``docker run`` with network isolation, memory/CPU limits,
    and a pre-built scientific image.  Most secure for untrusted code.
  - **sandbox**: in-process sandbox via ``tools/sandbox.py`` (restricted
    builtins + import allowlist).

All backends share the same ``ExperimentResult`` output type and the same
unified import allowlist so that validation/execution behaviour is consistent.
"""

from __future__ import annotations

import abc
import ast
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Unified import allowlist (shared across all backends)                #
# ------------------------------------------------------------------ #

SAFE_IMPORTS: frozenset[str] = frozenset({
    # Standard library
    "math", "cmath", "random", "json", "statistics",
    "itertools", "functools", "fractions", "decimal", "collections",
    "hashlib", "re", "string", "operator", "bisect", "heapq",
    "array", "struct", "copy", "pprint", "textwrap",
    "numbers", "typing", "dataclasses", "enum", "abc",
    # Scientific computing
    "numpy", "scipy", "sympy", "mpmath",
    # numpy sub-modules
    "numpy.linalg", "numpy.fft", "numpy.random", "numpy.polynomial",
    # scipy sub-modules
    "scipy.linalg", "scipy.optimize", "scipy.stats", "scipy.special",
    "scipy.integrate", "scipy.signal",
    # sympy sub-modules
    "sympy.core", "sympy.solvers", "sympy.geometry", "sympy.ntheory",
    "sympy.combinatorics", "sympy.series", "sympy.matrices",
})

BANNED_MODULE_PREFIXES: frozenset[str] = frozenset({
    "os", "sys", "subprocess", "socket", "pathlib", "shutil",
    "urllib", "http", "ftplib", "smtplib", "telnetlib",
    "poplib", "imaplib", "pickle", "shelve", "ctypes", "cffi",
    "importlib", "_imp", "builtins", "threading", "multiprocessing",
    "concurrent", "signal", "resource", "gc", "tempfile", "glob",
})

BANNED_NAMES: frozenset[str] = frozenset({
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
})


# ------------------------------------------------------------------ #
# Result type                                                          #
# ------------------------------------------------------------------ #

@dataclass
class ExperimentResult:
    """Uniform result from any experiment backend."""

    success: bool
    stdout: str
    stderr: str
    parsed_json: Optional[Any] = None
    execution_time_ms: float = 0.0
    error_message: str = ""
    timed_out: bool = False
    backend: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "stdout": self.stdout[:2000],
            "stderr": self.stderr[:1000],
            "parsed_json": self.parsed_json,
            "execution_time_ms": round(self.execution_time_ms, 1),
            "error_message": self.error_message,
            "timed_out": self.timed_out,
            "backend": self.backend,
        }


# ------------------------------------------------------------------ #
# AST validation (shared)                                              #
# ------------------------------------------------------------------ #

class CodeValidationError(Exception):
    pass


def validate_python_code(code: str) -> None:
    """Reject unsafe Python code before execution."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeValidationError(f"Invalid syntax: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            else:
                modules = [node.module or ""]
            for module in modules:
                root = module.split(".")[0]
                if root in BANNED_MODULE_PREFIXES or (root not in SAFE_IMPORTS and module not in SAFE_IMPORTS):
                    raise CodeValidationError(f"Disallowed import: '{module}'")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_NAMES:
                raise CodeValidationError(f"Disallowed builtin: '{node.func.id}'")
            if isinstance(node.func, ast.Attribute) and node.func.attr in BANNED_NAMES:
                raise CodeValidationError(f"Disallowed attribute call: '{node.func.attr}'")


# ------------------------------------------------------------------ #
# JSON extraction from stdout                                         #
# ------------------------------------------------------------------ #

def _parse_stdout_json(stdout: str) -> Optional[Any]:
    """Try to extract a JSON object/array from stdout."""
    text = stdout.strip()
    if not text:
        return None
    for candidate in [text.splitlines()[-1], text]:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    import re
    match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ------------------------------------------------------------------ #
# Abstract backend                                                     #
# ------------------------------------------------------------------ #

class ExperimentBackend(abc.ABC):
    """Abstract interface for experiment execution."""

    @abc.abstractmethod
    def execute(self, code: str, *, timeout: int = 120) -> ExperimentResult:
        ...


# ------------------------------------------------------------------ #
# Local subprocess backend                                             #
# ------------------------------------------------------------------ #

class LocalSubprocessBackend(ExperimentBackend):
    """Execute code in a subprocess with AST validation."""

    def execute(self, code: str, *, timeout: int = 120) -> ExperimentResult:
        import time

        validate_python_code(code)

        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write(code)
            tmp_path = Path(tmp.name)

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={"PATH": "/usr/bin:/bin"},
            )
            elapsed = (time.monotonic() - t0) * 1000
            parsed = _parse_stdout_json(proc.stdout)
            return ExperimentResult(
                success=proc.returncode == 0,
                stdout=proc.stdout[:50_000],
                stderr=proc.stderr[:10_000],
                parsed_json=parsed,
                execution_time_ms=elapsed,
                error_message="" if proc.returncode == 0 else proc.stderr[:500],
                backend="local",
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.monotonic() - t0) * 1000
            return ExperimentResult(
                success=False, stdout="", stderr="",
                execution_time_ms=elapsed,
                error_message=f"Timed out after {timeout}s",
                timed_out=True, backend="local",
            )
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass


# ------------------------------------------------------------------ #
# Docker backend                                                       #
# ------------------------------------------------------------------ #

_DEFAULT_DOCKER_IMAGE = "python:3.11-slim"


class DockerBackend(ExperimentBackend):
    """Execute code inside a Docker container with strict isolation.

    The container runs with:
      - ``--network none`` (no network access)
      - ``--memory`` / ``--cpus`` resource limits
      - ``--read-only`` filesystem (writable /tmp only)
      - Wall-clock timeout
    """

    def __init__(
        self,
        image: str = _DEFAULT_DOCKER_IMAGE,
        memory_mb: int = 512,
        cpus: float = 1.0,
    ) -> None:
        self._image = image
        self._memory = f"{memory_mb}m"
        self._cpus = str(cpus)

    def execute(self, code: str, *, timeout: int = 120) -> ExperimentResult:
        import time

        validate_python_code(code)

        if not shutil.which("docker"):
            logger.warning("Docker not found; falling back to local subprocess")
            return LocalSubprocessBackend().execute(code, timeout=timeout)

        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8", dir="/tmp"
        ) as tmp:
            tmp.write(code)
            tmp_path = Path(tmp.name)

        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--memory", self._memory,
            "--cpus", self._cpus,
            "--read-only",
            "--tmpfs", "/tmp:rw,size=64m",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "--pids-limit", "64",
            "-v", f"{tmp_path}:/code/script.py:ro",
            self._image,
            "python", "/code/script.py",
        ]

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=timeout + 10,
            )
            elapsed = (time.monotonic() - t0) * 1000
            parsed = _parse_stdout_json(proc.stdout)
            return ExperimentResult(
                success=proc.returncode == 0,
                stdout=proc.stdout[:50_000],
                stderr=proc.stderr[:10_000],
                parsed_json=parsed,
                execution_time_ms=elapsed,
                error_message="" if proc.returncode == 0 else proc.stderr[:500],
                backend="docker",
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.monotonic() - t0) * 1000
            return ExperimentResult(
                success=False, stdout="", stderr="",
                execution_time_ms=elapsed,
                error_message=f"Docker execution timed out after {timeout}s",
                timed_out=True, backend="docker",
            )
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass


# ------------------------------------------------------------------ #
# In-process sandbox backend                                           #
# ------------------------------------------------------------------ #

class SandboxBackend(ExperimentBackend):
    """Execute code using the in-process sandbox (tools/sandbox.py)."""

    def execute(self, code: str, *, timeout: int = 120) -> ExperimentResult:
        from discovery_zero.tools.sandbox import SandboxConfig, execute_sandboxed

        validate_python_code(code)

        config = SandboxConfig(timeout_seconds=timeout)
        result = execute_sandboxed(code, config=config)
        return ExperimentResult(
            success=result.success,
            stdout=result.stdout,
            stderr=result.stderr,
            parsed_json=result.parsed_json,
            execution_time_ms=result.execution_time_ms,
            error_message=result.error_message,
            timed_out=result.timed_out,
            backend="sandbox",
        )


# ------------------------------------------------------------------ #
# Factory                                                              #
# ------------------------------------------------------------------ #

def get_experiment_backend(backend_name: str = "local") -> ExperimentBackend:
    """Create an experiment backend by name."""
    if backend_name == "docker":
        return DockerBackend()
    elif backend_name == "sandbox":
        return SandboxBackend()
    else:
        return LocalSubprocessBackend()
