"""
Lean 4 formal verification module for Discovery Zero.

Provides programmatic access to the Lean workspace: proof writing, lake build
execution, output parsing, and structured result formatting for hypergraph ingestion.

Production-oriented behaviour:
- Robust error handling and logging
- Subprocess isolation with configurable timeouts
- Idempotent workspace initialization
- Full parsing of Lean/lake output for diagnostics

Concurrency: Concurrent writes to the same Proofs.lean are not safe; no file locking
is implemented. The benchmark harness uses ``prepare_benchmark_lean_sandbox`` so each
run writes to its own ``Proofs.lean`` under ``run_dir/lean_workspace/``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

__all__ = [
    "LeanGoal",
    "LeanDecomposeResult",
    "LeanVerifyResult",
    "get_workspace_path",
    "ensure_workspace",
    "prepare_benchmark_lean_sandbox",
    "init_workspace",
    "write_proof_file",
    "run_lake_build",
    "run_lake_env_lean",
    "verify_proof",
    "verify_workspace_proofs",
    "decompose_proof_skeleton",
    "extract_theorem_names",
    "extract_theorem_statement",
]

logger = logging.getLogger(__name__)

# Default workspace name and file paths (relative to discovery-zero skill root)
DEFAULT_WORKSPACE_DIR = "lean_workspace"
# Lake lean_lib "Discovery" + srcDir "Discovery" => module Discovery.X in Discovery/Discovery/X.lean
PROOFS_FILE = "Discovery/Discovery/Proofs.lean"
MAIN_FILE = "Discovery/Discovery/Main.lean"

# Regex patterns for parsing Lean output
THEOREM_PATTERN = re.compile(
    r"^\s*(?:theorem|lemma)\s+(discovery_\w+)\s*(?::\s*[^:=]+)?\s*(?::=\s*by)?",
    re.MULTILINE,
)
THEOREM_FULL_PATTERN = re.compile(
    r"(theorem|lemma)\s+(discovery_\w+)\s*([^:=]*(?::[^:=]+)?)\s*(?::=\s*by\s+[^;]+;?\s*)?$",
    re.MULTILINE | re.DOTALL,
)
LEAN_ERROR_PATTERN = re.compile(
    r"(?:error|Error):\s*(.+?)(?:\n\n|\Z)",
    re.DOTALL | re.IGNORECASE,
)
LEAN_FILE_LINE_PATTERN = re.compile(
    r"([^:]+):(\d+):(\d+):\s*(?:error|Error)?\s*(.*)",
    re.IGNORECASE,
)
LEAN_BLOCK_COMMENT_RE = re.compile(r"/-(?:.|\n)*?-/", re.MULTILINE)
LEAN_LINE_COMMENT_RE = re.compile(r"--.*$", re.MULTILINE)
LEAN_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"', re.DOTALL)
PLACEHOLDER_TOKEN_RE = re.compile(r"\b(sorry|admit)\b")
UNSOLVED_GOAL_BLOCK_PATTERN = re.compile(
    r"([^:]+):(\d+):(\d+):\s+error:\s+don't know how to synthesize placeholder\s*"
    r"(?:\ncontext:\n(?P<context>.*?))?\n⊢\s*(?P<target>.+?)(?=\n[^:\n]+:\d+:\d+:|\Z)",
    re.DOTALL,
)


@dataclass
class LeanGoal:
    """A single unresolved Lean goal extracted from build output."""

    file: str
    line: int
    col: int
    target: str
    context: str = ""


@dataclass
class LeanDecomposeResult:
    """Result of decomposing a partial Lean proof skeleton into subgoals."""

    success: bool
    transformed_source: str
    goals: list[LeanGoal] = field(default_factory=list)
    error_message: Optional[str] = None
    stderr: Optional[str] = None
    stdout: Optional[str] = None
    exit_code: Optional[int] = None


@dataclass
class LeanVerifyResult:
    """Result of a Lean proof verification attempt."""

    success: bool
    theorem_statement: Optional[str] = None
    theorem_names: list[str] = field(default_factory=list)
    formal_statement: Optional[str] = None
    error_message: Optional[str] = None
    stderr: Optional[str] = None
    stdout: Optional[str] = None
    exit_code: Optional[int] = None
    attempt: int = 1

    def to_ingest_dict(
        self,
        premises: list[dict],
        conclusion_statement: str,
        steps: list[str],
        domain: str = "geometry",
    ) -> dict:
        """Convert to the JSON format expected by ingest_skill_output."""
        if self.success:
            return {
                "premises": premises,
                "steps": steps,
                "conclusion": {
                    "statement": conclusion_statement,
                    "formal_statement": self.formal_statement or self.theorem_statement,
                },
                "module": "lean",
                "domain": domain,
                "confidence": 0.99,
            }
        return {
            "status": "failed",
            "last_error": self.error_message or self.stderr or "Unknown error",
            "attempts": self.attempt,
            "suggestion": (
                "Check Lean error output, ensure Mathlib imports are correct, "
                "and that the statement matches the intended conjecture."
            ),
        }


def get_skill_root() -> Path:
    """Resolve the discovery-zero skill root (parent of src)."""
    this_file = Path(__file__).resolve()
    # src/discovery_zero/lean.py -> skill root
    return this_file.parent.parent.parent.parent


def get_workspace_path(
    base_path: Optional[Path] = None,
) -> Path:
    """
    Return the absolute path to the Lean workspace directory.

    Args:
        base_path: Optional base directory. Defaults to discovery-zero skill root.

    Returns:
        Path to lean_workspace/
    """
    root = base_path if base_path is not None else get_skill_root()
    return (root / DEFAULT_WORKSPACE_DIR).resolve()


def _run_cmd(
    cmd: list[str],
    cwd: Path,
    timeout: int = 300,
    capture_output: bool = True,
    stream: bool = False,
) -> subprocess.CompletedProcess:
    """
    Run a command with isolation and timeout.

    Args:
        cmd: Command and arguments.
        cwd: Working directory.
        timeout: Timeout in seconds.
        capture_output: Whether to capture stdout/stderr (ignored if stream=True).
        stream: If True, print output in real time to terminal (no capture).

    Returns:
        CompletedProcess instance.

    Raises:
        subprocess.TimeoutExpired: If the command exceeds timeout.
        FileNotFoundError: If the command binary is not found.
    """
    if stream:
        print(f"Running lake build in {cwd} ...", flush=True)
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=None,
            stderr=None,
            stdin=subprocess.DEVNULL,
            env={**subprocess.os.environ},
            start_new_session=True,
        )
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, 9)
            except OSError:
                proc.kill()
            proc.wait(timeout=10)
            raise
        return subprocess.CompletedProcess(cmd, proc.returncode or 0, None, None)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        stdin=subprocess.DEVNULL,
        text=True,
        env={**subprocess.os.environ},
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, 9)
        except OSError:
            proc.kill()
        stdout, stderr = proc.communicate(timeout=10)
        raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(cmd, proc.returncode or 0, stdout, stderr)


def init_workspace(
    workspace_path: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """
    Initialize or recreate the Lean workspace with Mathlib.

    Creates:
        - lean_workspace/
        - lean_workspace/lakefile.toml
        - lean_workspace/lean-toolchain
        - lean_workspace/Discovery/Discovery/{Main,Proofs}.lean

    Args:
        workspace_path: Override workspace directory. If given, must be the
            workspace root. If None, uses default lean_workspace under skill root.
        force: If True, overwrite existing files.

    Returns:
        Path to the workspace root.
    """
    if workspace_path is not None:
        base = Path(workspace_path).resolve()
    else:
        base = get_workspace_path()
    base.mkdir(parents=True, exist_ok=True)

    # lean-toolchain (sync with Mathlib)
    toolchain_file = base / "lean-toolchain"
    toolchain_content = "leanprover/lean4:v4.29.0-rc3\n"
    if force or not toolchain_file.exists():
        toolchain_file.write_text(toolchain_content, encoding="utf-8")
        logger.info("Wrote lean-toolchain")

    # lakefile.toml
    lakefile = base / "lakefile.toml"
    lakefile_content = '''# Discovery Zero Lean Workspace
# Lean 4 project with Mathlib for formal verification.
# See: https://github.com/leanprover-community/mathlib4/wiki/Using-mathlib4-as-a-dependency

name = "discovery_zero"
version = "0.1.0"
defaultTargets = ["Discovery"]

[[require]]
name = "mathlib"
scope = "leanprover-community"

[[lean_lib]]
name = "Discovery"
srcDir = "Discovery"
'''
    if force or not lakefile.exists():
        lakefile.write_text(lakefile_content, encoding="utf-8")
        logger.info("Wrote lakefile.toml")

    # Discovery/Discovery/{Main,Proofs}.lean (Lake lean_lib srcDir structure)
    discovery_dir = base / "Discovery" / "Discovery"
    discovery_dir.mkdir(parents=True, exist_ok=True)
    main_lean = discovery_dir / "Main.lean"
    main_content = """/-
Discovery Zero - Lean 4 Formal Verification Module

This module holds theorems discovered and verified by the Discovery Zero system.
-/

import Mathlib
"""
    if force or not main_lean.exists():
        main_lean.write_text(main_content, encoding="utf-8")
        logger.info("Wrote Discovery/Main.lean")

    proofs_lean = discovery_dir / "Proofs.lean"
    proofs_content = """/-
Discovery Zero - Dynamic Proof File

Auto-generated by the lean verification pipeline.
Replace this content when adding new proofs.
-/

import Mathlib

theorem discovery_placeholder : True := trivial
"""
    if force or not proofs_lean.exists():
        proofs_lean.write_text(proofs_content, encoding="utf-8")
        logger.info("Wrote Discovery/Proofs.lean")

    return base


def ensure_workspace(
    workspace_path: Optional[Path] = None,
) -> Path:
    """
    Ensure the Lean workspace exists. Create if missing.

    Args:
        workspace_path: Override workspace directory. If given, must be the
            workspace root (directory containing lakefile.toml). If None,
            uses default lean_workspace under skill root.

    Returns:
        Path to the workspace root.
    """
    if workspace_path is not None:
        base = Path(workspace_path).resolve()
    else:
        base = get_workspace_path()
    if not base.exists() or not (base / "lakefile.toml").exists():
        init_workspace(base, force=False)
    return base


def prepare_benchmark_lean_sandbox(template: Path, sandbox: Path) -> Path:
    """
    Create an isolated Lean workspace for a benchmark run.

    Copies ``lakefile.toml``, ``lean-toolchain``, optional ``lake-manifest.json``,
    and ``Discovery/Discovery/{Main,Proofs}.lean`` from ``template``. Symlinks
    ``template/.lake/packages`` into the sandbox when present so Mathlib does not
    need to be re-fetched; ``lake build`` writes Discovery artifacts only under
    ``sandbox/.lake/build``.

    Args:
        template: A workspace that already has ``lakefile.toml`` and Lean sources.
        sandbox: Destination root; must not exist yet (e.g. ``run_dir / "lean_workspace"``).

    Returns:
        Resolved ``sandbox`` path.

    Raises:
        FileExistsError: If ``sandbox`` already exists.
        FileNotFoundError: If ``template`` is missing required files.
    """
    template = template.resolve()
    sandbox = sandbox.resolve()
    if sandbox.exists():
        raise FileExistsError(f"Lean sandbox already exists: {sandbox}")
    lakefile = template / "lakefile.toml"
    toolchain = template / "lean-toolchain"
    main_src = template / MAIN_FILE
    proofs_src = template / PROOFS_FILE
    for path, label in (
        (lakefile, "lakefile.toml"),
        (toolchain, "lean-toolchain"),
        (main_src, MAIN_FILE),
        (proofs_src, PROOFS_FILE),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Lean template missing {label}: {path}")

    sandbox.mkdir(parents=True)
    shutil.copy2(lakefile, sandbox / "lakefile.toml")
    shutil.copy2(toolchain, sandbox / "lean-toolchain")
    manifest = template / "lake-manifest.json"
    if manifest.is_file():
        shutil.copy2(manifest, sandbox / "lake-manifest.json")

    rel_discovery = Path("Discovery") / "Discovery"
    dest_discovery = sandbox / rel_discovery
    dest_discovery.mkdir(parents=True, exist_ok=True)
    shutil.copy2(main_src, dest_discovery / "Main.lean")
    shutil.copy2(proofs_src, dest_discovery / "Proofs.lean")

    lake_dir = sandbox / ".lake"
    lake_dir.mkdir(parents=True, exist_ok=True)
    template_packages = template / ".lake" / "packages"
    dest_packages = lake_dir / "packages"
    if template_packages.is_dir():
        try:
            os.symlink(template_packages.resolve(), dest_packages, target_is_directory=True)
        except OSError as exc:
            raise RuntimeError(
                "Could not symlink template .lake/packages into sandbox; "
                "symlinks are required for shared Mathlib (e.g. run on Linux or "
                "enable Windows Developer Mode). "
                f"template={template}, err={exc}"
            ) from exc
    else:
        logger.warning(
            "Template workspace has no .lake/packages; first lake build in sandbox "
            "may download Mathlib. Consider running `lake build` in %s first.",
            template,
        )

    logger.info("Prepared isolated Lean sandbox at %s (template=%s)", sandbox, template)
    return sandbox


def write_proof_file(
    lean_code: str,
    workspace_path: Optional[Path] = None,
    filename: str = "Proofs.lean",
) -> Path:
    """
    Write Lean proof code to Discovery/Discovery/Proofs.lean (or specified file).

    The content should be a complete Lean file (including imports).
    Replaces the entire file content.

    Args:
        lean_code: Full Lean source code.
        workspace_path: Override workspace directory.
        filename: Name of the file under Discovery/Discovery/ (default: Proofs.lean).

    Returns:
        Path to the written file.

    Raises:
        FileNotFoundError: If workspace does not exist.
    """
    base = ensure_workspace(workspace_path)
    target = base / "Discovery" / "Discovery" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(lean_code, encoding="utf-8")
    logger.info("Wrote proof file: %s", target)
    return target


def run_lake_build(
    workspace_path: Optional[Path] = None,
    timeout: int = 300,
    stream: bool = False,
) -> tuple[int, str, str]:
    """
    Run `lake build` in the workspace.

    Args:
        workspace_path: Override workspace directory.
        timeout: Build timeout in seconds.
        stream: If True, print output in real time (stdout/stderr will be empty in return).

    Returns:
        (exit_code, stdout, stderr).
    """
    base = ensure_workspace(workspace_path)
    lake = shutil.which("lake")
    if not lake:
        raise FileNotFoundError(
            "`lake` not found. Install Lean 4 toolchain (elan + lake). "
            "See: https://lean-lang.org/lean4/doc/setup.html"
        )
    try:
        proc = _run_cmd(
            [lake, "build"],
            cwd=base,
            timeout=timeout,
            stream=stream,
        )
        out = "" if stream else (proc.stdout or "")
        err = "" if stream else (proc.stderr or "")
        return (proc.returncode or 0, out, err)
    except subprocess.TimeoutExpired as e:
        logger.error("lake build timed out after %s seconds", timeout)
        raise


def run_lake_env_lean(
    relative_file: str,
    workspace_path: Optional[Path] = None,
    timeout: int = 300,
    stream: bool = False,
) -> tuple[int, str, str]:
    """
    Run `lake env lean <relative_file>` inside the workspace.

    This is useful for partial proof / subgoal extraction flows, where we want
    to typecheck a temporary file without overwriting the main Proofs.lean build
    target.
    """
    base = ensure_workspace(workspace_path)
    lake = shutil.which("lake")
    if not lake:
        raise FileNotFoundError(
            "`lake` not found. Install Lean 4 toolchain (elan + lake). "
            "See: https://lean-lang.org/lean4/doc/setup.html"
        )
    try:
        proc = _run_cmd(
            [lake, "env", "lean", relative_file],
            cwd=base,
            timeout=timeout,
            stream=stream,
        )
        out = "" if stream else (proc.stdout or "")
        err = "" if stream else (proc.stderr or "")
        return (proc.returncode or 0, out, err)
    except subprocess.TimeoutExpired:
        logger.error("lake env lean timed out after %s seconds", timeout)
        raise


def extract_theorem_statement(lean_source: str) -> Optional[str]:
    """
    Extract the first discovery_* theorem/lemma statement from Lean source.

    Looks for:
        theorem discovery_<name> : <stmt> := by ...
        lemma discovery_<name> : <stmt> := ...

    Returns:
        Full theorem/lemma line (without trailing tactics) or None.
    """
    for match in THEOREM_FULL_PATTERN.finditer(lean_source):
        kind, name, stmt_part = match.group(1), match.group(2), match.group(3)
        stmt_part = stmt_part.strip()
        if stmt_part.startswith(":"):
            stmt_part = stmt_part[1:].strip()
        return f"{kind} {name} {stmt_part}"
    for match in THEOREM_PATTERN.finditer(lean_source):
        return match.group(0).strip()
    return None


def extract_theorem_names(lean_source: str) -> list[str]:
    """Extract all declared discovery_* theorem/lemma names from Lean source."""
    seen: set[str] = set()
    names: list[str] = []
    for match in THEOREM_PATTERN.finditer(lean_source):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _rewrite_sorry_to_placeholders(lean_source: str) -> str:
    """
    Rewrite common `sorry` occurrences into explicit placeholders so Lean emits
    structured unsolved-goal diagnostics.

    This is intentionally conservative and targets the common tactic-style
    skeletons produced by LLMs.
    """
    lines = lean_source.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        indent = line[: len(line) - len(line.lstrip())]
        if stripped == "sorry":
            out.append(f"{indent}exact ?_")
            continue
        if stripped.endswith(":= by sorry"):
            out.append(line.replace(":= by sorry", ":= by\n" + indent + "  exact ?_"))
            continue
        if stripped.endswith("by sorry"):
            out.append(line.replace("by sorry", "by\n" + indent + "  exact ?_"))
            continue
        out.append(line)
    return "\n".join(out) + ("\n" if lean_source.endswith("\n") else "")


def extract_unsolved_goals(stderr: str, stdout: str) -> list[LeanGoal]:
    """Extract unresolved goal blocks from Lean output."""
    combined = (stderr or "") + "\n" + (stdout or "")
    goals: list[LeanGoal] = []
    for match in UNSOLVED_GOAL_BLOCK_PATTERN.finditer(combined):
        file = match.group(1)
        line = int(match.group(2))
        col = int(match.group(3))
        context = (match.group("context") or "").strip()
        target = match.group("target").strip()
        goals.append(
            LeanGoal(
                file=file,
                line=line,
                col=col,
                context=context,
                target=target,
            )
        )
    return goals


def parse_lean_error(stderr: str, stdout: str) -> str:
    """
    Extract the first meaningful error message from Lean/lake output.

    Prefers file:line:col style errors, then generic Error: lines.
    """
    combined = (stderr or "") + "\n" + (stdout or "")
    for match in LEAN_FILE_LINE_PATTERN.finditer(combined):
        return match.group(0).strip()
    for match in LEAN_ERROR_PATTERN.finditer(combined):
        return match.group(1).strip()
    if stderr:
        first_lines = stderr.strip().split("\n")[:5]
        return "\n".join(first_lines)
    if stdout:
        for line in stdout.split("\n"):
            if "error" in line.lower() or "Error" in line:
                return line.strip()
    return combined.strip()[:500] if combined.strip() else "Unknown error"


def _strip_lean_noncode_text(source: str) -> str:
    """Remove comments and string literals before placeholder checks."""
    stripped = LEAN_STRING_RE.sub(" ", source)
    stripped = LEAN_BLOCK_COMMENT_RE.sub(" ", stripped)
    stripped = LEAN_LINE_COMMENT_RE.sub(" ", stripped)
    return stripped


def contains_placeholder_proof(source: str) -> bool:
    """Return True when Lean source still contains sorry/admit placeholders."""
    return PLACEHOLDER_TOKEN_RE.search(_strip_lean_noncode_text(source)) is not None


def verify_proof(
    lean_code: str,
    workspace_path: Optional[Path] = None,
    timeout: int = 300,
    stream: bool = False,
) -> LeanVerifyResult:
    """
    Write the proof to Proofs.lean, run lake build, and return structured result.

    This is the main entry point for the lean_proof skill and dz lean verify.

    Args:
        lean_code: Complete Lean file content (import Mathlib + theorem).
        workspace_path: Override workspace directory.
        timeout: Build timeout in seconds.
        stream: If True, print lake build output in real time.

    Returns:
        LeanVerifyResult with success/failure and extracted data.
    """
    if contains_placeholder_proof(lean_code):
        return LeanVerifyResult(
            success=False,
            error_message="Lean proof contains placeholder proof terms (sorry/admit).",
            exit_code=-1,
        )
    base = ensure_workspace(workspace_path)
    write_proof_file(lean_code, base, "Proofs.lean")

    try:
        exit_code, stdout, stderr = run_lake_build(
            base, timeout=timeout, stream=stream
        )
    except subprocess.TimeoutExpired:
        return LeanVerifyResult(
            success=False,
            error_message="Build timed out",
            stderr="lake build exceeded timeout",
            exit_code=-1,
        )
    except FileNotFoundError as e:
        return LeanVerifyResult(
            success=False,
            error_message=str(e),
            exit_code=-1,
        )

    success = exit_code == 0
    theorem_stmt = extract_theorem_statement(lean_code) if success else None
    theorem_names = extract_theorem_names(lean_code) if success else []
    error_msg = None if success else parse_lean_error(stderr, stdout)

    return LeanVerifyResult(
        success=success,
        theorem_statement=theorem_stmt,
        theorem_names=theorem_names,
        formal_statement=theorem_stmt,
        error_message=error_msg,
        stderr=stderr or None,
        stdout=stdout or None,
        exit_code=exit_code,
    )


def verify_workspace_proofs(
    workspace_path: Optional[Path] = None,
    theorem_names: Optional[list[str]] = None,
    timeout: int = 300,
    stream: bool = False,
) -> LeanVerifyResult:
    """
    Strictly verify the existing Proofs.lean in the workspace.

    Unlike `verify_proof`, this does not overwrite the workspace. It only succeeds
    if `lake build` succeeds; if `theorem_names` is provided, all requested
    `discovery_*` declarations must also be present as actual theorem/lemma
    declarations in `Proofs.lean`.
    """
    base = ensure_workspace(workspace_path)
    proofs_path = base / PROOFS_FILE
    if not proofs_path.exists():
        return LeanVerifyResult(
            success=False,
            error_message=f"Proof file not found: {proofs_path}",
            exit_code=-1,
        )

    lean_code = proofs_path.read_text(encoding="utf-8")
    if contains_placeholder_proof(lean_code):
        return LeanVerifyResult(
            success=False,
            error_message="Proofs.lean contains placeholder proof terms (sorry/admit).",
            exit_code=-1,
        )
    declared_names = extract_theorem_names(lean_code)

    try:
        exit_code, stdout, stderr = run_lake_build(
            base, timeout=timeout, stream=stream
        )
    except subprocess.TimeoutExpired:
        return LeanVerifyResult(
            success=False,
            error_message="Build timed out",
            stderr="lake build exceeded timeout",
            exit_code=-1,
        )
    except FileNotFoundError as e:
        return LeanVerifyResult(
            success=False,
            error_message=str(e),
            exit_code=-1,
        )

    success = exit_code == 0
    error_msg = None if success else parse_lean_error(stderr, stdout)
    if success and theorem_names:
        missing = [name for name in theorem_names if name not in declared_names]
        if missing:
            success = False
            error_msg = (
                "Verified workspace is missing requested theorem declarations: "
                + ", ".join(missing)
            )

    theorem_stmt = extract_theorem_statement(lean_code) if success else None
    verified_names = declared_names if success else []

    return LeanVerifyResult(
        success=success,
        theorem_statement=theorem_stmt,
        theorem_names=verified_names,
        formal_statement=theorem_stmt,
        error_message=error_msg,
        stderr=stderr or None,
        stdout=stdout or None,
        exit_code=exit_code,
    )


def decompose_proof_skeleton(
    lean_code: str,
    workspace_path: Optional[Path] = None,
    timeout: int = 300,
    stream: bool = False,
) -> LeanDecomposeResult:
    """
    Turn a partial Lean skeleton into explicit unresolved goals.

    The function rewrites common `sorry` placeholders into explicit goal holes,
    writes a temporary Lean file inside the workspace, runs `lake env lean` on
    that file, and parses unresolved goals from the real Lean diagnostics.
    """
    base = ensure_workspace(workspace_path)
    transformed = _rewrite_sorry_to_placeholders(lean_code)
    unique_suffix = uuid.uuid4().hex[:8]
    relative_file = f"Discovery/Discovery/PartialGoals_{unique_suffix}.lean"
    target = base / relative_file
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(transformed, encoding="utf-8")

    try:
        exit_code, stdout, stderr = run_lake_env_lean(
            relative_file,
            workspace_path=base,
            timeout=timeout,
            stream=stream,
        )
    except subprocess.TimeoutExpired:
        return LeanDecomposeResult(
            success=False,
            transformed_source=transformed,
            error_message="Lean decomposition timed out",
            stderr="lake env lean exceeded timeout",
            exit_code=-1,
        )
    except FileNotFoundError as e:
        return LeanDecomposeResult(
            success=False,
            transformed_source=transformed,
            error_message=str(e),
            exit_code=-1,
        )
    finally:
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass

    goals = extract_unsolved_goals(stderr, stdout)
    success = exit_code == 0 and not goals
    error_msg = None if success else (parse_lean_error(stderr, stdout) if (stderr or stdout) else "Unknown error")
    return LeanDecomposeResult(
        success=success,
        transformed_source=transformed,
        goals=goals,
        error_message=error_msg,
        stderr=stderr or None,
        stdout=stdout or None,
        exit_code=exit_code,
    )


def get_setup_instructions() -> str:
    """Return human-readable setup instructions for the Lean workspace."""
    return """
To set up the Lean workspace:

  cd {workspace}
  lake update
  lake exe cache get   # Optional: fetch precompiled Mathlib (recommended)
  lake build

If `lake` is not found, install the Lean 4 toolchain:
  https://lean-lang.org/lean4/doc/setup.html
""".format(
        workspace=get_workspace_path()
    ).strip()
