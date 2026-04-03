import json
import pytest

from dz_hypergraph.models import HyperGraph
from dz_engine.orchestrator import (
    _bridge_plan_input,
    _judge_requires_replan,
    _normalize_bridge_payload,
    _resolve_supported_conclusion_statement,
    _run_experiment_skill_with_code_fallback,
    _compile_skeleton_from_bridge_plan,
    ActionResult,
    execute_bridge_followups,
    plan_bridge_consumption,
    run_experiment_action,
    run_loop,
    select_experiment_focus_proposition,
)
from dz_engine.bridge import validate_bridge_plan_payload
from dz_hypergraph.tools.lean import contains_placeholder_proof
from dz_hypergraph.tools.lean_policy import LeanBoundaryPolicy, LeanPolicyError, validate_lean_code
from dz_hypergraph.persistence import save_graph


def test_resolve_supported_conclusion_statement_matches_existing_theorem():
    g = HyperGraph()
    theorem = g.add_node("For all real numbers x and y, x^2 + y^2 ≥ 2*x*y.", belief=0.5)
    resolved, summary = _resolve_supported_conclusion_statement(
        g,
        "Experimental evidence strongly supports: For all real numbers x and y, x^2 + y^2 ≥ 2*x*y.",
        fallback_target_statement=theorem.statement,
    )
    assert resolved == theorem.statement
    assert summary is not None


def test_resolve_supported_conclusion_statement_uses_fallback_target():
    g = HyperGraph()
    theorem = g.add_node("Prime quadratic reciprocity theorem", belief=0.5)
    resolved, summary = _resolve_supported_conclusion_statement(
        g,
        "Experimental evidence strongly supports: a theorem about odd primes.",
        fallback_target_statement=theorem.statement,
    )
    assert resolved == theorem.statement
    assert summary is not None


def test_lean_policy_rejects_forbidden_identifier():
    policy = LeanBoundaryPolicy(
        forbidden_identifiers=["Matrix.aeval_self_charpoly"]
    )
    code = "import Mathlib\n\ntheorem t : True := by\n  exact Matrix.aeval_self_charpoly\n"
    with pytest.raises(LeanPolicyError):
        validate_lean_code(code, policy)


def test_lean_policy_rejects_import_outside_whitelist():
    policy = LeanBoundaryPolicy(
        allowed_import_prefixes=["Mathlib", "Batteries"]
    )
    code = "import Std\n\ntheorem t : True := by trivial\n"
    with pytest.raises(LeanPolicyError):
        validate_lean_code(code, policy)


def test_lean_policy_ignores_forbidden_names_in_comments_and_strings():
    policy = LeanBoundaryPolicy(forbidden_identifiers=["Cayley", "Hamilton"])
    code = """import Mathlib

-- Cayley and Hamilton are historical labels here only.
def label : String := "Cayley-Hamilton theorem"

theorem discovery_ok : True := by
  trivial
"""
    validate_lean_code(code, policy)


def test_judge_requires_replan_for_low_confidence_or_explicit_gaps():
    assert _judge_requires_replan({"confidence": 0.2})
    assert _judge_requires_replan(
        {
            "confidence": 0.61,
            "reasoning": "The route has a major gap at the matrix-evaluation step.",
            "concerns": [],
            "suggestion": None,
        }
    )


def test_normalize_bridge_payload_converts_old_style_fields():
    payload = {
        "propositions": [
            {
                "id": "P1",
                "statement": "Seed fact",
                "role": "seed",
                "grade": "A",
                "dependencies": [],
            },
            {
                "id": "P2",
                "statement": "Target theorem",
                "role": "target",
                "grade": "B",
                "dependencies": ["P1"],
            },
        ],
        "reasoning_steps": [
            {
                "id": "S1",
                "justification": "Use the seed to derive the target.",
                "uses": ["P1"],
                "concludes": "P2",
                "grade": "B",
            }
        ],
    }

    normalized = _normalize_bridge_payload(payload, target_statement="Target theorem")
    plan = validate_bridge_plan_payload(normalized)

    assert plan.target_statement == "Target theorem"
    assert plan.propositions[1].depends_on == ["P1"]
    assert plan.chain[0].statement == "Use the seed to derive the target."
    assert plan.chain[0].concludes == ["P2"]


def test_contains_placeholder_proof_ignores_comments_and_strings():
    assert contains_placeholder_proof("theorem t : True := by\n  admit\n")
    assert contains_placeholder_proof("theorem t : True := by\n  sorry\n")
    assert not contains_placeholder_proof(
        'theorem t : True := by\n  let s := "admit"\n  -- sorry in comment\n  trivial\n'
    )
    assert not _judge_requires_replan(
        {
            "confidence": 0.72,
            "reasoning": "Reasonable route with only minor presentational issues.",
            "concerns": [],
            "suggestion": None,
        }
    )


def test_plan_bridge_consumption_routes_c_and_d_away_from_lean():
    plan = validate_bridge_plan_payload(
        {
            "target_statement": "Target theorem",
            "propositions": [
                {"id": "P1", "statement": "Seed fact", "role": "seed", "grade": "A", "depends_on": []},
                {"id": "P2", "statement": "Risky bridge", "role": "bridge", "grade": "D", "depends_on": ["P1"]},
                {"id": "P3", "statement": "Lean-ready bridge", "role": "bridge", "grade": "B", "depends_on": ["P1"]},
                {"id": "P4", "statement": "Experimental witness", "role": "bridge", "grade": "C", "depends_on": ["P3"]},
                {"id": "P5", "statement": "Target theorem", "role": "target", "grade": "B", "depends_on": ["P2", "P3"]},
            ],
            "chain": [
                {"id": "S1", "statement": "Derive the lean-ready bridge.", "uses": ["P1"], "concludes": ["P3"], "grade": "B"},
                {"id": "S2", "statement": "Experimental side-check.", "uses": ["P3"], "concludes": ["P4"], "grade": "C"},
                {"id": "S3", "statement": "Risky step remains informal.", "uses": ["P1"], "concludes": ["P2"], "grade": "D"},
                {"id": "S4", "statement": "Finish the target.", "uses": ["P2", "P3"], "concludes": ["P5"], "grade": "B"},
            ],
        }
    )

    decision = plan_bridge_consumption(plan)

    assert decision.decomposition_focus_proposition_id == "P2"
    assert decision.decomposition_target_proposition_id == "P2"
    assert decision.strict_focus_proposition_id == "P3"
    assert decision.strict_target_proposition_id == "P3"
    assert decision.strict_mode == "lemma"
    assert decision.experiment_proposition_ids == ["P4"]
    assert decision.natural_language_proposition_ids == ["P2"]


def test_plan_bridge_consumption_prefers_sibling_goal_subplan():
    plan = validate_bridge_plan_payload(
        {
            "target_statement": "Outer target",
            "propositions": [
                {"id": "P1", "statement": "Seed fact", "role": "seed", "grade": "A", "depends_on": []},
                {"id": "P2", "statement": "Sibling one", "role": "bridge", "grade": "A", "depends_on": ["P1"]},
                {"id": "P3", "statement": "Sibling two", "role": "bridge", "grade": "B", "depends_on": ["P1"]},
                {"id": "P4", "statement": "Lean bottleneck", "role": "bridge", "grade": "B", "depends_on": ["P2", "P3"]},
                {"id": "P5", "statement": "Outer target", "role": "target", "grade": "B", "depends_on": ["P4"]},
            ],
            "chain": [
                {"id": "S1", "statement": "Build sibling one.", "uses": ["P1"], "concludes": ["P2"], "grade": "A"},
                {"id": "S2", "statement": "Build sibling two.", "uses": ["P1"], "concludes": ["P3"], "grade": "B"},
                {"id": "S3", "statement": "Combine siblings.", "uses": ["P2", "P3"], "concludes": ["P4"], "grade": "B"},
                {"id": "S4", "statement": "Conclude the outer target.", "uses": ["P4"], "concludes": ["P5"], "grade": "B"},
            ],
        }
    )

    decision = plan_bridge_consumption(plan)

    assert decision.decomposition_focus_proposition_id == "P4"
    assert decision.decomposition_used_sibling_package is True
    assert decision.decomposition_target_proposition_id == "P4__siblings"
    assert decision.decomposition_bridge_plan is not None
    assert decision.strict_focus_proposition_id == "P2"
    assert decision.strict_mode == "direct_proof"


def test_plan_bridge_consumption_can_skip_lean_when_only_c_or_d_remain():
    plan = validate_bridge_plan_payload(
        {
            "target_statement": "Hard target",
            "propositions": [
                {"id": "P1", "statement": "Seed fact", "role": "seed", "grade": "A", "depends_on": []},
                {"id": "P2", "statement": "Experimental bridge", "role": "bridge", "grade": "C", "depends_on": ["P1"]},
                {"id": "P3", "statement": "Risky target", "role": "target", "grade": "D", "depends_on": ["P2"]},
            ],
            "chain": [
                {"id": "S1", "statement": "Get experimental evidence.", "uses": ["P1"], "concludes": ["P2"], "grade": "C"},
                {"id": "S2", "statement": "Tentatively conclude the target.", "uses": ["P2"], "concludes": ["P3"], "grade": "D"},
            ],
        }
    )

    decision = plan_bridge_consumption(plan)

    assert decision.decomposition_bridge_plan is not None
    assert decision.decomposition_focus_proposition_id == "P3"
    assert decision.strict_focus_proposition_id is None
    assert decision.experiment_proposition_ids == ["P2"]
    assert decision.natural_language_proposition_ids == ["P3"]


def test_plan_bridge_consumption_can_route_experiment_friendly_d_to_experiment():
    plan = validate_bridge_plan_payload(
        {
            "target_statement": "Concept target",
            "propositions": [
                {"id": "P1", "statement": "Seed fact", "role": "seed", "grade": "A", "depends_on": []},
                {
                    "id": "P2",
                    "statement": "Risky but testable claim: enumerate the composition table and check closure of all discovered motions.",
                    "role": "bridge",
                    "grade": "D",
                    "depends_on": ["P1"],
                    "experiment_notes": "Finite exact table check should verify the local claim before any formal treatment.",
                },
                {"id": "P3", "statement": "Concept target", "role": "target", "grade": "D", "depends_on": ["P2"]},
            ],
            "chain": [
                {"id": "S1", "statement": "Local testable closure claim.", "uses": ["P1"], "concludes": ["P2"], "grade": "D"},
                {"id": "S2", "statement": "Tentative concept target.", "uses": ["P2"], "concludes": ["P3"], "grade": "D"},
            ],
        }
    )

    decision = plan_bridge_consumption(plan)

    assert decision.experiment_focus_proposition_id == "P2"
    assert "P2" in decision.experiment_proposition_ids
    assert "P2" not in decision.natural_language_proposition_ids


def test_plan_bridge_consumption_marks_object_layer_strict_mode():
    plan = validate_bridge_plan_payload(
        {
            "target_statement": "Main target",
            "propositions": [
                {"id": "P1", "statement": "Seed fact", "role": "seed", "grade": "A", "depends_on": []},
                {
                    "id": "P2",
                    "statement": "Define the coercion from scalar polynomials into matrix-valued expressions.",
                    "role": "bridge",
                    "grade": "B",
                    "depends_on": ["P1"],
                    "formalization_notes": "Need an explicit embedding and notation bridge before later lemmas typecheck.",
                },
                {"id": "P3", "statement": "Main target", "role": "target", "grade": "B", "depends_on": ["P2"]},
            ],
            "chain": [
                {"id": "S1", "statement": "Set up the objects.", "uses": ["P1"], "concludes": ["P2"], "grade": "B"},
                {"id": "S2", "statement": "Conclude the target.", "uses": ["P2"], "concludes": ["P3"], "grade": "B"},
            ],
        }
    )

    decision = plan_bridge_consumption(plan)

    assert decision.strict_focus_proposition_id == "P2"
    assert decision.strict_mode == "object_setup"


def test_plan_bridge_consumption_falls_back_to_local_experiment_target():
    plan = validate_bridge_plan_payload(
        {
            "target_statement": "Main target",
            "propositions": [
                {"id": "P1", "statement": "Seed fact", "role": "seed", "grade": "A", "depends_on": []},
                {
                    "id": "P2",
                    "statement": "Enumerate the six symmetry compositions and check closure exactly.",
                    "role": "bridge",
                    "grade": "B",
                    "depends_on": ["P1"],
                    "experiment_notes": "Finite exact enumeration should confirm the local closure claim.",
                },
                {"id": "P3", "statement": "Main target", "role": "target", "grade": "B", "depends_on": ["P2"]},
            ],
            "chain": [
                {"id": "S1", "statement": "Build the local closure bridge.", "uses": ["P1"], "concludes": ["P2"], "grade": "B"},
                {"id": "S2", "statement": "Conclude the target.", "uses": ["P2"], "concludes": ["P3"], "grade": "B"},
            ],
        }
    )

    decision = plan_bridge_consumption(plan)

    assert decision.experiment_focus_proposition_id == "P2"
    assert decision.experiment_target_proposition_id == "P2"
    assert "P2" in decision.experiment_proposition_ids


def test_select_experiment_focus_proposition_prefers_experiment_friendly_bridge():
    plan = validate_bridge_plan_payload(
        {
            "target_statement": "Concept target",
            "propositions": [
                {"id": "P1", "statement": "Seed fact", "role": "seed", "grade": "A", "depends_on": []},
                {
                    "id": "P2",
                    "statement": "Show the composition table is closed on the six discovered motions.",
                    "role": "bridge",
                    "grade": "B",
                    "depends_on": ["P1"],
                },
                {
                    "id": "P3",
                    "statement": "Derive the abstract reversible structure.",
                    "role": "target",
                    "grade": "B",
                    "depends_on": ["P2"],
                },
            ],
            "chain": [
                {"id": "S1", "statement": "Build the composition table.", "uses": ["P1"], "concludes": ["P2"], "grade": "B"},
                {"id": "S2", "statement": "Abstract the structure.", "uses": ["P2"], "concludes": ["P3"], "grade": "B"},
            ],
        }
    )

    assert select_experiment_focus_proposition(plan) == "P2"


def test_experiment_skill_accepts_raw_python_block(monkeypatch):
    g = HyperGraph()
    theorem = g.add_node(
        "Test conjecture",
        belief=0.5,
        domain="number_theory",
    )

    monkeypatch.setattr(
        "dz_engine.orchestrator.load_skill_prompt",
        lambda _name: "experiment prompt",
    )
    monkeypatch.setattr(
        "dz_engine.orchestrator.chat_completion",
        lambda **_kwargs: {
            "choices": [
                {
                    "message": {
                        "content": "```python\nimport json\nprint(json.dumps({'passed': True}))\n```"
                    }
                }
            ]
        },
    )

    raw, parsed = _run_experiment_skill_with_code_fallback(
        g,
        theorem.id,
        "Conjecture:\nTest conjecture",
        model="gpt-5.2",
    )

    assert "```python" in raw
    assert parsed["module"] == "experiment"
    assert parsed["domain"] == "number_theory"
    assert parsed["steps"][0].startswith("import json")
    assert parsed["conclusion"]["statement"].startswith(
        "Experimental evidence for claim tested in code"
    )


def test_bridge_plan_input_enforces_explicit_target_and_risk_rules():
    g = HyperGraph()
    target = g.add_node("Target proposition", belief=0.3, domain="logic")
    prompt = _bridge_plan_input(
        graph=g,
        target_node_id=target.id,
        reasoning_output={"module": "plausible", "steps": ["route"]},
        judge_output={"confidence": 0.7},
    )
    assert "exactly one proposition with role='target'" in prompt
    assert "must match the target statement exactly" in prompt
    assert "must NOT appear in depends_on of any non-risk proposition" in prompt


def test_run_experiment_action_weakens_on_zero_pass_large_trials(monkeypatch):
    """Experiments should never hard-refute; even the strongest case uses 'weakened'."""
    g = HyperGraph()
    target = g.add_node("Conjecture", belief=0.4, domain="number_theory")

    monkeypatch.setattr(
        "dz_engine.orchestrator._run_experiment_skill_with_code_fallback",
        lambda *args, **kwargs: (
            "raw",
            {
                "module": "experiment",
                "domain": "number_theory",
                "steps": ["print('check')"],
                "conclusion": {"statement": "Conjecture"},
            },
        ),
    )
    monkeypatch.setattr(
        "dz_engine.orchestrator.normalize_skill_output",
        lambda *args, **kwargs: {
            "module": "experiment",
            "domain": "number_theory",
            "steps": ["print('check')"],
            "conclusion": {"statement": "Conjecture"},
        },
    )
    monkeypatch.setattr(
        "dz_engine.orchestrator._execute_with_repair",
        lambda *args, **kwargs: (
            "print('check')",
            "",
            "",
            {
                "passed": False,
                "trials": 1000,
                "pass_rate": 0.0,
                "counterexample": {"example": [1, 2, 3]},
                "summary": "counterexample found",
            },
        ),
    )

    _, normalized, judge = run_experiment_action(g, target.id, model="gpt-5.2")
    assert normalized["outcome"] == "weakened"
    assert normalized["confidence"] == pytest.approx(0.95)
    assert judge["confidence"] == pytest.approx(0.95)


def test_run_experiment_action_softens_on_zero_pass_medium_trials(monkeypatch):
    g = HyperGraph()
    target = g.add_node("Conjecture", belief=0.4, domain="number_theory")

    monkeypatch.setattr(
        "dz_engine.orchestrator._run_experiment_skill_with_code_fallback",
        lambda *args, **kwargs: (
            "raw",
            {
                "module": "experiment",
                "domain": "number_theory",
                "steps": ["print('check')"],
                "conclusion": {"statement": "Conjecture"},
            },
        ),
    )
    monkeypatch.setattr(
        "dz_engine.orchestrator.normalize_skill_output",
        lambda *args, **kwargs: {
            "module": "experiment",
            "domain": "number_theory",
            "steps": ["print('check')"],
            "conclusion": {"statement": "Conjecture"},
        },
    )
    monkeypatch.setattr(
        "dz_engine.orchestrator._execute_with_repair",
        lambda *args, **kwargs: (
            "print('check')",
            "",
            "",
            {
                "passed": False,
                "trials": 120,
                "pass_rate": 0.0,
                "counterexample": {"example": [1, 2, 3]},
                "summary": "counterexample found",
            },
        ),
    )

    _, normalized, judge = run_experiment_action(g, target.id, model="gpt-5.2")
    assert normalized["outcome"] == "weakened"
    assert normalized["confidence"] == pytest.approx(0.90)
    assert judge["confidence"] == pytest.approx(0.90)


def test_compile_skeleton_from_bridge_plan_retries_after_missing_target_marker(monkeypatch):
    g = HyperGraph()
    target = g.add_node(
        "Local bridge target",
        belief=0.5,
        domain="logic",
    )
    plan = validate_bridge_plan_payload(
        {
            "target_statement": "Local bridge target",
            "propositions": [
                {"id": "P1", "statement": "Seed fact", "role": "seed", "grade": "A", "depends_on": []},
                {"id": "P2", "statement": "Local bridge target", "role": "target", "grade": "D", "depends_on": ["P1"]},
            ],
            "chain": [
                {"id": "S1", "statement": "Derive the local bridge target.", "uses": ["P1"], "concludes": ["P2"], "grade": "D"},
            ],
        }
    )

    responses = iter(
        [
            (
                "first raw",
                {
                    "premises": [],
                    "steps": [
                        "import Mathlib\n\ntheorem discovery_first : True := by\n  -- BRIDGE-STEP: S1\n  sorry\n"
                    ],
                    "conclusion": {"statement": "Lean skeleton", "formal_statement": "theorem discovery_first : True"},
                    "module": "lean",
                    "domain": "logic",
                },
            ),
            (
                "second raw",
                {
                    "premises": [],
                    "steps": [
                        "import Mathlib\n\n-- BRIDGE-PROP: P2\n-- BRIDGE-STEP: S1\ntheorem discovery_second : True := by\n  sorry\n"
                    ],
                    "conclusion": {"statement": "Lean skeleton", "formal_statement": "theorem discovery_second : True"},
                    "module": "lean",
                    "domain": "logic",
                },
            ),
        ]
    )

    monkeypatch.setattr(
        "dz_engine.orchestrator.run_skill",
        lambda *args, **kwargs: next(responses),
    )

    raw, normalized, lean_code = _compile_skeleton_from_bridge_plan(
        g,
        target.id,
        plan,
        model="gpt-5.2",
        max_attempts=2,
    )

    assert raw == "second raw"
    assert normalized["module"] == "lean"
    assert "-- BRIDGE-PROP: P2" in lean_code


def test_execute_bridge_followups_runs_local_bridge_experiment(monkeypatch, tmp_graph_dir):
    graph_path = tmp_graph_dir / "graph.json"
    g = HyperGraph()
    seed = g.add_node("Seed fact", belief=1.0, domain="geometry", state="proven")
    theorem = g.add_node("Main theorem", belief=0.5, domain="geometry")
    save_graph(g, graph_path)

    plan = validate_bridge_plan_payload(
        {
            "target_statement": "Main theorem",
            "propositions": [
                {"id": "P1", "statement": "Seed fact", "role": "seed", "grade": "A", "depends_on": []},
                {
                    "id": "P2",
                    "statement": "Enumerate the local composition table and check closure exactly.",
                    "role": "bridge",
                    "grade": "B",
                    "depends_on": ["P1"],
                    "experiment_notes": "Finite exact closure check should validate this local bridge first."
                },
                {
                    "id": "P3",
                    "statement": "Check the resulting composition law against the second local consistency relation.",
                    "role": "bridge",
                    "grade": "B",
                    "depends_on": ["P2"],
                    "experiment_notes": "A second exact local check should run after P2 becomes supported.",
                },
                {"id": "P4", "statement": "Main theorem", "role": "target", "grade": "B", "depends_on": ["P3"]},
            ],
            "chain": [
                {"id": "S1", "statement": "Build the local bridge.", "uses": ["P1"], "concludes": ["P2"], "grade": "B"},
                {"id": "S2", "statement": "Run a second local check.", "uses": ["P2"], "concludes": ["P3"], "grade": "B"},
                {"id": "S3", "statement": "Conclude the target.", "uses": ["P3"], "concludes": ["P4"], "grade": "B"},
            ],
        }
    )

    monkeypatch.setattr(
        "dz_engine.orchestrator.run_bridge_planning_action",
        lambda *args, **kwargs: ("raw bridge", plan),
    )
    monkeypatch.setattr(
        "dz_engine.orchestrator.run_experiment_action",
        lambda graph, target_node_id, model=None: (
            f"raw exp {target_node_id}",
            {
                "premises": [],
                "steps": ["print('ok')"],
                "conclusion": {"statement": graph.nodes[target_node_id].statement},
                "module": "experiment",
                "domain": graph.nodes[target_node_id].domain,
            },
            {"confidence": 0.8},
        ),
    )
    monkeypatch.setattr(
        "dz_engine.orchestrator.propagate_beliefs",
        lambda *args, **kwargs: 0,
    )

    results = execute_bridge_followups(
        graph_path,
        theorem.id,
        reasoning_output={"module": "plausible", "conclusion": {"statement": theorem.statement}},
        judge_output={"confidence": 0.7},
    )

    actions = [item.action for item in results]
    assert actions[0] == "bridge_consumption"
    assert "bridge_experiment" in actions or "bridge_ready" in actions
    for r in results[1:]:
        assert r.success is True


def test_run_loop_appends_bridge_followups_after_plausible(monkeypatch, tmp_graph_dir):
    graph_path = tmp_graph_dir / "graph.json"
    g = HyperGraph()
    theorem = g.add_node("Main theorem", belief=0.3, domain="geometry")
    save_graph(g, graph_path)

    monkeypatch.setattr(
        "dz_engine.orchestrator.execute_action",
        lambda graph, target_node_id, selected_module, model=None: ActionResult(
            action="execute",
            target_node_id=target_node_id,
            selected_module=selected_module.value,
            normalized_output={"module": "plausible", "conclusion": {"statement": graph.nodes[target_node_id].statement}},
            judge_output={"confidence": 0.7},
            success=True,
            message="ok",
        ),
    )
    monkeypatch.setattr(
        "dz_engine.orchestrator.ingest_action_output",
        lambda graph_path, action_result, backend="bp": action_result,
    )
    monkeypatch.setattr(
        "dz_engine.orchestrator.execute_bridge_followups",
        lambda *args, **kwargs: [
            ActionResult(
                action="bridge_consumption",
                target_node_id=theorem.id,
                selected_module="bridge",
                success=True,
                message="planned",
            ),
            ActionResult(
                action="bridge_experiment",
                target_node_id=theorem.id,
                selected_module="experiment",
                success=True,
                message="consumed",
            ),
            ActionResult(
                action="bridge_ready",
                target_node_id=theorem.id,
                selected_module="bridge",
                success=True,
                message="ready",
            ),
        ],
    )

    results = run_loop(graph_path, rounds=1)

    assert [item.action for item in results] == ["execute", "bridge_consumption", "bridge_experiment", "bridge_ready"]
