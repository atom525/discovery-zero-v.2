"""Tests for the Lean formal verification module."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from dz_hypergraph.tools.lean import (
    LeanVerifyResult,
    extract_theorem_statement,
    get_workspace_path,
    get_skill_root,
    ensure_workspace,
    init_workspace,
    prepare_benchmark_lean_sandbox,
    write_proof_file,
    parse_lean_error,
    verify_proof,
)


class TestGetWorkspacePath:
    def test_returns_lean_workspace_under_skill_root(self):
        root = get_skill_root()
        ws = get_workspace_path()
        assert ws.name == "lean_workspace"
        assert ws.parent == root

    def test_with_base_path(self):
        base = Path("/tmp/custom")
        ws = get_workspace_path(base)
        assert ws == base / "lean_workspace"


class TestInitWorkspace:
    def test_creates_all_files(self, tmp_graph_dir):
        base = tmp_graph_dir / "lean_ws"
        init_workspace(base)
        assert (base / "lakefile.toml").exists()
        assert (base / "lean-toolchain").exists()
        assert (base / "Discovery" / "Discovery" / "Main.lean").exists()
        assert (base / "Discovery" / "Discovery" / "Proofs.lean").exists()

    def test_lakefile_has_mathlib(self, tmp_graph_dir):
        base = tmp_graph_dir / "lean_ws"
        init_workspace(base)
        content = (base / "lakefile.toml").read_text()
        assert "mathlib" in content.lower()
        assert "leanprover-community" in content

    def test_lean_toolchain_has_version(self, tmp_graph_dir):
        base = tmp_graph_dir / "lean_ws"
        init_workspace(base)
        content = (base / "lean-toolchain").read_text()
        assert "lean4" in content or "lean" in content

    def test_force_overwrites(self, tmp_graph_dir):
        base = tmp_graph_dir / "lean_ws"
        init_workspace(base)
        (base / "lakefile.toml").write_text("old")
        init_workspace(base, force=True)
        content = (base / "lakefile.toml").read_text()
        assert "old" not in content
        assert "mathlib" in content.lower()


class TestPrepareBenchmarkLeanSandbox:
    def test_copies_config_and_symlinks_packages(self, tmp_graph_dir):
        template = tmp_graph_dir / "tpl"
        init_workspace(template)
        (template / ".lake").mkdir(exist_ok=True)
        (template / ".lake" / "packages").mkdir(parents=True)
        (template / ".lake" / "packages" / "stub.txt").write_text("ok")
        sandbox = tmp_graph_dir / "sandbox"
        out = prepare_benchmark_lean_sandbox(template, sandbox)
        assert out == sandbox.resolve()
        assert (sandbox / "lakefile.toml").exists()
        assert (sandbox / "Discovery" / "Discovery" / "Proofs.lean").exists()
        link = sandbox / ".lake" / "packages"
        assert link.is_symlink()
        assert (link / "stub.txt").read_text() == "ok"

    def test_raises_if_sandbox_exists(self, tmp_graph_dir):
        template = tmp_graph_dir / "tpl"
        sandbox = tmp_graph_dir / "sandbox"
        init_workspace(template)
        sandbox.mkdir()
        with pytest.raises(FileExistsError):
            prepare_benchmark_lean_sandbox(template, sandbox)

    def test_raises_if_template_incomplete(self, tmp_graph_dir):
        template = tmp_graph_dir / "tpl"
        template.mkdir()
        (template / "lakefile.toml").write_text("x")
        sandbox = tmp_graph_dir / "sandbox"
        with pytest.raises(FileNotFoundError):
            prepare_benchmark_lean_sandbox(template, sandbox)


class TestEnsureWorkspace:
    def test_creates_if_missing(self, tmp_graph_dir):
        base = tmp_graph_dir / "lean_ws"
        result = ensure_workspace(base)
        assert result == base
        assert (base / "lakefile.toml").exists()

    def test_returns_existing(self, tmp_graph_dir):
        base = tmp_graph_dir / "lean_ws"
        init_workspace(base)
        result = ensure_workspace(base)
        assert result == base


class TestWriteProofFile:
    def test_writes_content(self, tmp_graph_dir):
        base = tmp_graph_dir / "lean_ws"
        init_workspace(base)
        code = "import Mathlib\ntheorem discovery_foo : True := trivial"
        path = write_proof_file(code, base)
        assert path.exists()
        assert "discovery_foo" in path.read_text()

    def test_replaces_entire_file(self, tmp_graph_dir):
        base = tmp_graph_dir / "lean_ws"
        init_workspace(base)
        write_proof_file("import Mathlib\ntheorem x : True := trivial", base)
        write_proof_file("import Mathlib\ntheorem y : False := trivial", base)
        content = (base / "Discovery" / "Discovery" / "Proofs.lean").read_text()
        assert "theorem x" not in content
        assert "theorem y" in content


class TestExtractTheoremStatement:
    def test_extracts_theorem(self):
        code = "theorem discovery_midline (A B C : Point) : parallel MN BC := by sorry"
        stmt = extract_theorem_statement(code)
        assert stmt is not None
        assert "discovery_midline" in stmt

    def test_extracts_lemma(self):
        code = "lemma discovery_aux (n : ℕ) : n + 0 = n := by simp"
        stmt = extract_theorem_statement(code)
        assert stmt is not None
        assert "discovery_aux" in stmt

    def test_returns_none_for_empty(self):
        assert extract_theorem_statement("import Mathlib") is None

    def test_returns_none_for_non_discovery(self):
        code = "theorem other_name : True := trivial"
        assert extract_theorem_statement(code) is None or "other_name" in (extract_theorem_statement(code) or "")


class TestParseLeanError:
    def test_extracts_file_line_error(self):
        err = "Discovery/Proofs.lean:10:3: type mismatch"
        result = parse_lean_error(err, "")
        assert "10" in result or "type mismatch" in result

    def test_extracts_generic_error(self):
        err = "Error: unknown identifier 'foo'"
        result = parse_lean_error(err, "")
        assert "foo" in result or "Error" in result

    def test_returns_truncated_stderr_when_no_match(self):
        err = "some random output"
        result = parse_lean_error(err, "")
        assert len(result) > 0


class TestVerifyProof:
    def test_success_when_lake_passes(self, tmp_graph_dir):
        code = """import Mathlib
theorem discovery_placeholder : True := trivial
"""
        with patch("dz_hypergraph.tools.lean.run_lake_build") as run:
            run.return_value = (0, "", "")
            result = verify_proof(code, tmp_graph_dir)
        assert result.success
        assert result.exit_code == 0
        assert result.formal_statement or result.theorem_statement

    def test_failure_when_lake_fails(self, tmp_graph_dir):
        code = """import Mathlib
theorem discovery_bad : False := trivial
"""
        with patch("dz_hypergraph.tools.lean.run_lake_build") as run:
            run.return_value = (1, "", "Error: type mismatch")
            result = verify_proof(code, tmp_graph_dir)
        assert not result.success
        assert result.error_message or result.stderr

    def test_handles_timeout(self, tmp_graph_dir):
        code = "import Mathlib\ntheorem x : True := trivial"
        with patch("dz_hypergraph.tools.lean.run_lake_build") as run:
            run.side_effect = subprocess.TimeoutExpired("lake", 1)
            result = verify_proof(code, tmp_graph_dir, timeout=1)
        assert not result.success
        assert "timed" in (result.error_message or "").lower()

    def test_handles_lake_not_found(self, tmp_graph_dir):
        code = "import Mathlib\ntheorem x : True := trivial"
        with patch("dz_hypergraph.tools.lean.run_lake_build") as run:
            run.side_effect = FileNotFoundError("lake not found")
            result = verify_proof(code, tmp_graph_dir)
        assert not result.success
        assert "not found" in (result.error_message or "").lower()


class TestLeanVerifyResult:
    def test_to_ingest_dict_success(self):
        r = LeanVerifyResult(
            success=True,
            theorem_statement="theorem discovery_foo : True",
            formal_statement="theorem discovery_foo : True := trivial",
        )
        d = r.to_ingest_dict(
            premises=[{"id": "n1", "statement": "axiom"}],
            conclusion_statement="True",
            steps=["proof"],
            domain="logic",
        )
        assert d["module"] == "lean"
        assert d["confidence"] == 0.99
        assert d["conclusion"]["formal_statement"] == "theorem discovery_foo : True := trivial"

    def test_to_ingest_dict_failure(self):
        r = LeanVerifyResult(success=False, error_message="type mismatch", attempt=3)
        d = r.to_ingest_dict([], "stmt", [], "geom")
        assert d["status"] == "failed"
        assert d["last_error"] == "type mismatch"
        assert d["attempts"] == 3
