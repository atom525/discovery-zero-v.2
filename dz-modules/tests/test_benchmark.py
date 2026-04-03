import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dz_engine.benchmark import (
    BenchmarkCaseConfig,
    load_case_config,
    load_suite_config,
    run_suite,
    summarize_run,
)
from dz_engine.cli import app
from dz_hypergraph.models import HyperGraph
from dz_hypergraph.persistence import save_graph


runner = CliRunner()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _minimal_bridge_plan(target_statement: str) -> dict:
    return {
        "target_statement": target_statement,
        "propositions": [
            {
                "id": "P1",
                "statement": "Seed fact",
                "role": "seed",
                "grade": "A",
                "depends_on": [],
            },
            {
                "id": "P2",
                "statement": "Bridge lemma",
                "role": "bridge",
                "grade": "B",
                "depends_on": ["P1"],
            },
            {
                "id": "P3",
                "statement": target_statement,
                "role": "target",
                "grade": "A",
                "depends_on": ["P2"],
            },
        ],
        "chain": [
            {
                "id": "S1",
                "statement": "Derive bridge lemma",
                "uses": ["P1"],
                "concludes": ["P2"],
                "grade": "B",
            },
            {
                "id": "S2",
                "statement": "Conclude target",
                "uses": ["P2"],
                "concludes": ["P3"],
                "grade": "A",
            },
        ],
        "summary": "Minimal valid bridge plan for benchmark testing.",
    }


_EVAL_DIR = Path(__file__).resolve().parents[1] / "evaluate"
_SUITE_PATH = _EVAL_DIR / "suite.json"
_SUITE_AVAILABLE = _SUITE_PATH.exists()
_skip_no_suite = pytest.mark.skipif(not _SUITE_AVAILABLE, reason="suite.json not found")

_HOMOCHIRALITY_CASE = _EVAL_DIR / "cases" / "homochirality_mechanism" / "case.json"
_skip_no_case = pytest.mark.skipif(not _HOMOCHIRALITY_CASE.exists(), reason="case not found")


@_skip_no_suite
def test_load_core_suite_config():
    suite = load_suite_config(_SUITE_PATH)
    assert suite.suite_id == "frontier_open_problems_v1"
    assert len(suite.case_files) >= 1


@_skip_no_case
def test_load_case_config_resolves_source_path():
    case = load_case_config(_HOMOCHIRALITY_CASE)
    assert case.case_id == "homochirality_mechanism"
    assert case.source_proof_config.exists()
    assert case.lean_policy["mode"] == "selective"
    assert case.planning_constraints


def test_summarize_run_extracts_metrics(tmp_graph_dir):
    run_dir = tmp_graph_dir / "run_01"
    run_dir.mkdir()
    graph_path = run_dir / "graph.json"
    log_path = run_dir / "exploration_log.json"
    bridge_path = run_dir / "bridge_plan.json"

    graph = HyperGraph()
    theorem = graph.add_node("Test theorem", belief=0.5, domain="test")
    save_graph(graph, graph_path)

    _write_json(
        log_path,
        {
            "node_ids": {"theorem": theorem.id},
            "steps": [
                {
                    "phase": "plausible",
                    "normalized": {"steps": ["a", "b", "c"]},
                    "judge": {"confidence": 0.8},
                },
                {
                    "phase": "experiment",
                    "normalized": {
                        "steps": [
                            "from fractions import Fraction\nprint('ok')",
                            "Actual execution summary: "
                            + json.dumps(
                                {
                                    "passed": True,
                                    "trials": 300,
                                    "max_error": 0,
                                    "counterexample": None,
                                    "summary": "all checks passed",
                                },
                                ensure_ascii=False,
                            ),
                            "Used independent crosscheck between two methods.",
                        ]
                    },
                },
                {
                    "phase": "decomposition",
                    "subgoals": [{"statement": "subgoal"}],
                },
                {
                    "phase": "strict_lean",
                    "success": False,
                    "error": "Discovery/Proofs.lean:7:5: error: expected token",
                    "raw": "theorem demo := by\n  sorry",
                },
            ],
        },
    )
    _write_json(bridge_path, _minimal_bridge_plan("Test theorem"))

    summary = summarize_run(
        case=BenchmarkCaseConfig(
            case_id="demo",
            display_name="Demo",
            source_proof_config=Path(__file__).resolve().parents[1] / "evaluate" / "cases" / "homochirality_mechanism" / "proof_config.json",
            benchmark_scope="test scope",
        ),
        run_dir=run_dir,
        graph_path=graph_path,
        log_path=log_path,
        bridge_path=bridge_path,
        repeat_index=1,
    )

    assert summary["success"] is False
    assert summary["first_failure_stage"] == "strict_lean"
    assert summary["metrics"]["experiment_trials"] == 300
    assert summary["metrics"]["experiment_exact"] == 1
    assert summary["metrics"]["experiment_independent_crosscheck"] == 1
    assert summary["metrics"]["lean_decompose_success"] == 1
    assert summary["metrics"]["first_failure_is_localized"] == 1
    assert summary["metrics"]["placeholder_rejected_correctly"] == 1
    assert "claims_extracted_per_iteration" in summary["metrics"]
    assert "verification_driven_belief_progress" in summary["metrics"]
    assert summary["metrics"]["PQI"] > 0


def test_summarize_run_falls_back_to_target_key_when_theorem_missing(tmp_graph_dir):
    run_dir = tmp_graph_dir / "run_01"
    run_dir.mkdir()
    graph_path = run_dir / "graph.json"
    log_path = run_dir / "exploration_log.json"
    bridge_path = run_dir / "bridge_plan.json"

    graph = HyperGraph()
    theorem = graph.add_node("Fallback theorem", belief=0.73, domain="test")
    save_graph(graph, graph_path)
    _write_json(
        run_dir / "resolved_proof_config.json",
        {
            "target": {"key": "lrc_n11", "statement": "Fallback theorem"},
            "seed_nodes": [],
        },
    )
    _write_json(
        log_path,
        {
            "node_ids": {"lrc_n11": theorem.id},
            "steps": [
                {
                    "phase": "plausible",
                    "normalized": {"steps": ["route"]},
                    "judge": {"confidence": 0.7},
                }
            ],
        },
    )
    _write_json(bridge_path, _minimal_bridge_plan("Fallback theorem"))

    summary = summarize_run(
        case=BenchmarkCaseConfig(
            case_id="demo_fallback",
            display_name="Demo fallback",
            source_proof_config=Path(__file__).resolve().parents[1] / "evaluate" / "cases" / "homochirality_mechanism" / "proof_config.json",
            benchmark_scope="test scope",
        ),
        run_dir=run_dir,
        graph_path=graph_path,
        log_path=log_path,
        bridge_path=bridge_path,
        repeat_index=1,
    )

    assert summary["final_target_belief"] == pytest.approx(0.73)


def test_summarize_run_records_policy_skip(tmp_graph_dir):
    run_dir = tmp_graph_dir / "run_01"
    run_dir.mkdir()
    graph_path = run_dir / "graph.json"
    log_path = run_dir / "exploration_log.json"
    bridge_path = run_dir / "bridge_plan.json"

    graph = HyperGraph()
    theorem = graph.add_node("Selective theorem", belief=0.5, domain="test")
    save_graph(graph, graph_path)
    _write_json(
        log_path,
        {
            "node_ids": {"theorem": theorem.id},
            "steps": [
                {
                    "phase": "plausible",
                    "normalized": {"steps": ["route"]},
                    "judge": {"confidence": 0.9},
                },
                {
                    "phase": "experiment",
                    "normalized": {
                        "steps": [
                            "print('ok')",
                            "Actual execution summary: "
                            + json.dumps(
                                {
                                    "passed": True,
                                    "trials": 20,
                                    "max_error": 0,
                                    "counterexample": None,
                                    "summary": "ok",
                                },
                                ensure_ascii=False,
                            ),
                        ]
                    },
                },
                {
                    "phase": "decomposition",
                    "skipped": True,
                    "skipped_by_policy": True,
                    "reason": "Lean subgoal decomposition is disabled (lean_policy.enable_decomposition is false).",
                    "subgoals": [],
                },
                {
                    "phase": "strict_lean",
                    "skipped": True,
                    "skipped_by_policy": True,
                    "reason": "Strict Lean mode `object_setup` is outside the selective allowlist ['direct_proof', 'lemma'].",
                    "success": False,
                    "strict_mode": "object_setup",
                },
            ],
        },
    )
    _write_json(bridge_path, _minimal_bridge_plan("Selective theorem"))

    summary = summarize_run(
        case=BenchmarkCaseConfig(
            case_id="selective",
            display_name="Selective",
            source_proof_config=Path(__file__).resolve().parents[1] / "evaluate" / "cases" / "homochirality_mechanism" / "proof_config.json",
            benchmark_scope="selective scope",
        ),
        run_dir=run_dir,
        graph_path=graph_path,
        log_path=log_path,
        bridge_path=bridge_path,
        repeat_index=1,
    )

    assert summary["strict_lean_skipped_by_policy"] is True
    assert summary["lean_decompose_skipped_by_policy"] is True
    assert summary["metrics"]["strict_lean_attempted"] == 0
    assert summary["metrics"]["strict_lean_skipped_by_policy"] == 1
    assert summary["current_bottleneck"] == "lean_deferred_mode_mismatch"
    assert summary["progress_without_closure"] is True
    assert summary["metrics"]["bridge_consumption_ready"] == 0


def test_summarize_run_marks_bridge_ready_as_consumption_ready(tmp_graph_dir):
    run_dir = tmp_graph_dir / "run_01"
    run_dir.mkdir()
    graph_path = run_dir / "graph.json"
    log_path = run_dir / "exploration_log.json"
    bridge_path = run_dir / "bridge_plan.json"

    graph = HyperGraph()
    theorem = graph.add_node("Ready theorem", belief=0.5, domain="test")
    save_graph(graph, graph_path)
    _write_json(
        log_path,
        {
            "node_ids": {"theorem": theorem.id},
            "steps": [
                {
                    "phase": "plausible",
                    "normalized": {"steps": ["route"]},
                    "judge": {"confidence": 0.8},
                },
                {
                    "phase": "bridge_consumption",
                    "experiment_target_proposition_id": "P2",
                },
                {
                    "phase": "bridge_experiment",
                    "normalized": {
                        "steps": [
                            "print('ok')",
                            "Actual execution summary: "
                            + json.dumps(
                                {
                                    "passed": True,
                                    "trials": 20,
                                    "max_error": 0,
                                    "counterexample": None,
                                    "summary": "ok",
                                },
                                ensure_ascii=False,
                            ),
                        ]
                    },
                },
                {
                    "phase": "bridge_ready",
                    "ready_bridge_proposition_id": "P3",
                    "ready_graph_node_id": theorem.id,
                },
            ],
        },
    )
    _write_json(bridge_path, _minimal_bridge_plan("Ready theorem"))

    summary = summarize_run(
        case=BenchmarkCaseConfig(
            case_id="ready",
            display_name="Ready",
            source_proof_config=Path(__file__).resolve().parents[1] / "evaluate" / "cases" / "homochirality_mechanism" / "proof_config.json",
            benchmark_scope="ready scope",
        ),
        run_dir=run_dir,
        graph_path=graph_path,
        log_path=log_path,
        bridge_path=bridge_path,
        repeat_index=1,
    )

    assert summary["metrics"]["bridge_consumption_ready"] == 1
    assert summary["benchmark_outcome"] == "bridge_consumption_ready"


def test_summarize_run_timeout_not_localized(tmp_graph_dir):
    run_dir = tmp_graph_dir / "run_01"
    run_dir.mkdir()
    graph_path = run_dir / "graph.json"
    log_path = run_dir / "exploration_log.json"
    bridge_path = run_dir / "bridge_plan.json"

    graph = HyperGraph()
    theorem = graph.add_node("Timeout theorem", belief=0.5, domain="test")
    save_graph(graph, graph_path)
    _write_json(
        log_path,
        {
            "node_ids": {"theorem": theorem.id},
            "steps": [
                {
                    "phase": "plausible",
                    "normalized": {"steps": ["route"]},
                    "judge": {"confidence": 0.55},
                },
                {
                    "phase": "experiment",
                    "error": "TimeoutError: The read operation timed out",
                },
            ],
        },
    )
    _write_json(bridge_path, _minimal_bridge_plan("Timeout theorem"))

    summary = summarize_run(
        case=BenchmarkCaseConfig(
            case_id="timeout",
            display_name="Timeout",
            source_proof_config=Path(__file__).resolve().parents[1] / "evaluate" / "cases" / "homochirality_mechanism" / "proof_config.json",
            benchmark_scope="timeout scope",
        ),
        run_dir=run_dir,
        graph_path=graph_path,
        log_path=log_path,
        bridge_path=bridge_path,
        repeat_index=1,
    )

    assert summary["first_failure_stage"] == "experiment"
    assert summary["metrics"]["first_failure_is_localized"] == 0


def test_run_suite_creates_unique_directories(tmp_graph_dir, monkeypatch):
    source_proof_config = tmp_graph_dir / "proof_config.json"
    _write_json(source_proof_config, {"theorem": "demo"})
    case_config_path = tmp_graph_dir / "case.json"
    suite_config_path = tmp_graph_dir / "suite.json"
    _write_json(
        case_config_path,
        {
            "case_id": "demo_case",
            "display_name": "Demo Case",
            "source_proof_config": str(source_proof_config),
            "benchmark_scope": "demo scope",
        },
    )
    _write_json(
        suite_config_path,
        {
            "suite_id": "demo_suite",
            "display_name": "Demo Suite",
            "description": "demo",
            "repeats": 1,
            "run_mode": "serial",
            "cases": [str(case_config_path)],
        },
    )

    def fake_run_case_once(case, *, run_dir, repeat_index, suite_id, **kwargs):
        run_dir.mkdir(parents=True, exist_ok=False)
        summary = {
            "case_id": case.case_id,
            "display_name": case.display_name,
            "benchmark_scope": case.benchmark_scope,
            "repeat_index": repeat_index,
            "run_dir": str(run_dir),
            "log_path": str(run_dir / "exploration_log.json"),
            "graph_path": str(run_dir / "graph.json"),
            "bridge_plan_path": str(run_dir / "bridge_plan.json"),
            "final_target_state": "unverified",
            "final_target_belief": 0.5,
            "success": False,
            "progress_without_closure": True,
            "strict_lean_success": False,
            "strict_lean_skipped_by_policy": False,
            "lean_decompose_skipped_by_policy": False,
            "first_failure_stage": "strict_lean",
            "first_failure_message": "expected token",
            "benchmark_outcome": "partial_progress",
            "current_bottleneck": "strict_lean",
            "current_bottleneck_detail": None,
            "metrics": {
                "bridge_plan_valid": 1,
                "bridge_consumption_ready": 1,
                "bridge_experiment_attempted": 1,
                "bridge_experiment_success": 1,
                "best_path_judge_confidence": 0.7,
                "grade_a_count": 1,
                "grade_b_count": 1,
                "grade_c_count": 0,
                "grade_d_count": 0,
                "experiment_trials": 10,
                "experiment_found_counterexample": 0,
                "strict_lean_attempted": 1,
                "lean_decompose_attempted": 1,
                "lean_decompose_success": 1,
                "lean_subgoal_count": 1,
                "lean_decompose_skipped_by_policy": 0,
                "strict_lean_success": 0,
                "strict_lean_skipped_by_policy": 0,
                "experiment_ran": 1,
                "first_failure_is_localized": 1,
                "replan_triggered": 0,
                "PQI": 70.0,
                "FRI": 60.0,
                "ESI": 55.0,
                "ODB": 63.25,
            },
        }
        _write_json(run_dir / "summary.json", summary)
        return summary

    monkeypatch.setattr("dz_engine.benchmark.run_case_once", fake_run_case_once)

    first = run_suite(suite_config_path, output_root=tmp_graph_dir / "evaluation")
    second = run_suite(suite_config_path, output_root=tmp_graph_dir / "evaluation")

    assert first.suite_run_dir != second.suite_run_dir
    assert first.suite_summary_path.exists()
    assert second.suite_scorecard_path.exists()
    first_summary = json.loads(first.suite_summary_path.read_text(encoding="utf-8"))
    assert first_summary["repeats"] == 1
    assert first_summary["cases"][0]["current_bottleneck_counts"]["strict_lean"] == 1
    scorecard_text = second.suite_scorecard_path.read_text(encoding="utf-8")
    assert "## 分题摘要" in scorecard_text
    assert "Solve Rate" not in scorecard_text
    assert "结果类型" in scorecard_text


def test_cli_benchmark_run_suite(tmp_graph_dir, monkeypatch):
    class DummyResult:
        suite_run_dir = tmp_graph_dir / "runs" / "demo"
        suite_summary_path = tmp_graph_dir / "reports" / "suite_summary.json"
        suite_scorecard_path = tmp_graph_dir / "reports" / "suite_scorecard_zh.md"

    monkeypatch.setattr("dz_engine.benchmark.run_suite", lambda *args, **kwargs: DummyResult())
    suite_path = _SUITE_PATH
    result = runner.invoke(app, ["benchmark", "run-suite", "--suite", str(suite_path)])
    assert result.exit_code == 0
    assert "Suite summary:" in result.stdout
