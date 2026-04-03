from typer.testing import CliRunner
import json
from unittest.mock import patch

from dz_engine.cli import app
from dz_hypergraph.models import Module
from dz_engine.orchestrator import ActionResult

runner = CliRunner()


class TestCLI:
    def test_init(self, tmp_graph_dir):
        path = tmp_graph_dir / "graph.json"
        result = runner.invoke(app, ["init", "--path", str(path)])
        assert result.exit_code == 0
        assert path.exists()

    def test_summary_empty(self, tmp_graph_dir):
        path = tmp_graph_dir / "graph.json"
        runner.invoke(app, ["init", "--path", str(path)])
        result = runner.invoke(app, ["summary", "--path", str(path)])
        assert result.exit_code == 0
        assert "0 nodes" in result.stdout

    def test_add_node(self, tmp_graph_dir):
        path = tmp_graph_dir / "graph.json"
        runner.invoke(app, ["init", "--path", str(path)])
        result = runner.invoke(
            app, ["add-node", "--path", str(path),
                  "--statement", "Two points determine a line",
                  "--belief", "1.0"]
        )
        assert result.exit_code == 0
        assert "Added node" in result.stdout

    def test_show_nodes(self, tmp_graph_dir):
        path = tmp_graph_dir / "graph.json"
        runner.invoke(app, ["init", "--path", str(path)])
        runner.invoke(
            app, ["add-node", "--path", str(path),
                  "--statement", "Axiom 1", "--belief", "1.0"]
        )
        result = runner.invoke(app, ["show", "--path", str(path)])
        assert result.exit_code == 0
        assert "Axiom 1" in result.stdout

    def test_lean_init(self, tmp_graph_dir):
        ws = tmp_graph_dir / "lean_ws"
        result = runner.invoke(app, ["lean", "init", "--path", str(ws)])
        assert result.exit_code == 0
        assert (ws / "lakefile.toml").exists()
        assert (ws / "Discovery" / "Discovery" / "Proofs.lean").exists()

    def test_lean_path(self):
        result = runner.invoke(app, ["lean", "path"])
        assert result.exit_code == 0
        assert "lean_workspace" in result.stdout

    def test_htps_step_persists_state(self, tmp_graph_dir):
        path = tmp_graph_dir / "graph.json"
        state_path = tmp_graph_dir / "graph.htps_state.json"
        runner.invoke(app, ["init", "--path", str(path)])
        runner.invoke(
            app,
            [
                "add-node", "--path", str(path),
                "--statement", "axiom", "--belief", "1.0", "--state", "proven",
            ],
        )
        runner.invoke(
            app,
            [
                "add-node", "--path", str(path),
                "--statement", "goal", "--belief", "0.4",
            ],
        )
        from dz_hypergraph.persistence import load_graph, save_graph
        from dz_hypergraph.models import Module as M
        g = load_graph(path)
        ids = list(g.nodes.keys())
        g.add_hyperedge([ids[0]], ids[1], M.PLAUSIBLE, [], 0.8)
        save_graph(g, path)

        result1 = runner.invoke(
            app,
            ["htps-step", "--path", str(path), "--root", ids[1], "--state-path", str(state_path)],
        )
        assert result1.exit_code == 0
        result2 = runner.invoke(
            app,
            ["htps-step", "--path", str(path), "--root", ids[1], "--state-path", str(state_path)],
        )
        assert result2.exit_code == 0
        data = json.loads(state_path.read_text(encoding="utf-8"))
        visit_counts = list(data["N"].values())
        assert max(visit_counts) >= 2

    def test_htps_run_expands_leaf_with_orchestrator(self, tmp_graph_dir):
        path = tmp_graph_dir / "graph.json"
        state_path = tmp_graph_dir / "graph.htps_state.json"
        runner.invoke(app, ["init", "--path", str(path)])
        runner.invoke(
            app,
            [
                "add-node", "--path", str(path),
                "--statement", "axiom", "--belief", "1.0", "--state", "proven",
            ],
        )
        runner.invoke(
            app,
            [
                "add-node", "--path", str(path),
                "--statement", "helper", "--belief", "0.3",
            ],
        )
        runner.invoke(
            app,
            [
                "add-node", "--path", str(path),
                "--statement", "goal", "--belief", "0.4",
            ],
        )
        from dz_hypergraph.persistence import load_graph, save_graph
        from dz_hypergraph.models import Module as M
        g = load_graph(path)
        ids = list(g.nodes.keys())
        axiom_id, helper_id, goal_id = ids[0], ids[1], ids[2]
        g.add_hyperedge([axiom_id, helper_id], goal_id, M.PLAUSIBLE, [], 0.8)
        save_graph(g, path)

        fake_action = ActionResult(
            action="plausible",
            target_node_id=helper_id,
            selected_module=Module.PLAUSIBLE.value,
            normalized_output={"module": "plausible"},
            success=True,
            message="ok",
        )
        fake_ingested = ActionResult(
            action="plausible",
            target_node_id=helper_id,
            selected_module=Module.PLAUSIBLE.value,
            normalized_output={"module": "plausible"},
            ingest_edge_id="edge123",
            success=True,
            message="Ingested and propagated.",
        )
        with (
            patch("dz_engine.orchestrator.execute_action", return_value=fake_action),
            patch("dz_engine.orchestrator.ingest_action_output", return_value=fake_ingested),
            patch("dz_engine.orchestrator.execute_bridge_followups", return_value=[]),
        ):
            result = runner.invoke(
                app,
                [
                    "htps-run",
                    "--path", str(path),
                    "--root", goal_id,
                    "--rounds", "1",
                    "--state-path", str(state_path),
                ],
            )
        assert result.exit_code == 0
        assert "ingest_edge_id=edge123" in result.stdout

    def test_extract_claims_command(self, monkeypatch):
        monkeypatch.setattr(
            "dz_verify.claim_pipeline.ClaimPipeline.extract_claims",
            lambda self, **kwargs: [],
        )
        result = runner.invoke(
            app,
            [
                "extract-claims",
                "--prose",
                "For all n>=2, n^2-n is even.",
                "--context",
                "number theory",
            ],
        )
        assert result.exit_code == 0
        assert result.stdout.strip().startswith("[")

    def test_verify_loop_command(self, monkeypatch, tmp_graph_dir):
        path = tmp_graph_dir / "graph.json"
        runner.invoke(app, ["init", "--path", str(path)])
        runner.invoke(
            app,
            [
                "add-node",
                "--path",
                str(path),
                "--statement",
                "Target theorem",
                "--belief",
                "0.2",
            ],
        )
        from dz_hypergraph.persistence import load_graph

        graph = load_graph(path)
        target_id = next(iter(graph.nodes.keys()))
        monkeypatch.setattr(
            "dz_verify.verification_loop.VerificationLoop.run",
            lambda self, **kwargs: type(
                "R",
                (),
                {
                    "success": False,
                    "iterations_completed": 1,
                    "latest_feedback": "ok",
                    "traces": [],
                },
            )(),
        )
        result = runner.invoke(
            app,
            [
                "verify-loop",
                "--path",
                str(path),
                "--target",
                target_id,
                "--iterations",
                "1",
            ],
        )
        assert result.exit_code == 0
        assert '"iterations_completed": 1' in result.stdout
