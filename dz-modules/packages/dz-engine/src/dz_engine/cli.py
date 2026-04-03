"""CLI for managing the Discovery Zero hypergraph."""

import json
from pathlib import Path
from typing import Optional

import typer

from dz_hypergraph.models import HyperGraph, Module
from dz_hypergraph.persistence import save_graph, load_graph, DEFAULT_GRAPH_PATH
from dz_hypergraph.inference import propagate_beliefs
from dz_hypergraph.strategy import rank_nodes, suggest_module
from dz_hypergraph.ingest import ingest_skill_output

app = typer.Typer(help="Discovery Zero: reasoning hypergraph manager")

# Lean sub-app for formal verification
lean_app = typer.Typer(help="Lean 4 formal verification (lake build, proof file management)")
app.add_typer(lean_app, name="lean")

# LLM sub-app for real model-backed skills
llm_app = typer.Typer(help="LLM-backed skills via LiteLLM-compatible proxy")
app.add_typer(llm_app, name="llm")

# Benchmark sub-app for repeated evaluation suites
benchmark_app = typer.Typer(help="Artifact-backed benchmark suite execution")
app.add_typer(benchmark_app, name="benchmark")


@app.command()
def init(path: Path = typer.Option(DEFAULT_GRAPH_PATH, help="Graph file path")):
    """Initialize an empty hypergraph."""
    g = HyperGraph()
    save_graph(g, path)
    typer.echo(f"Initialized empty hypergraph at {path}")


@app.command()
def summary(path: Path = typer.Option(DEFAULT_GRAPH_PATH, help="Graph file path")):
    """Show hypergraph summary."""
    g = load_graph(path)
    s = g.summary()
    typer.echo(
        f"{s['num_nodes']} nodes, {s['num_edges']} edges | "
        f"axioms: {s.get('num_axioms', 0)}, proven: {s.get('num_proven', 0)}, "
        f"refuted: {s.get('num_refuted', 0)}, unverified: {s.get('num_unverified', 0)}"
    )


@app.command()
def add_node(
    statement: str = typer.Option(..., help="Proposition statement"),
    belief: float = typer.Option(0.0, help="Initial belief (1.0 for axioms)"),
    domain: Optional[str] = typer.Option(None, help="Mathematical domain"),
    state: Optional[str] = typer.Option(None, help="State: unverified (default), proven, refuted"),
    path: Path = typer.Option(DEFAULT_GRAPH_PATH, help="Graph file path"),
):
    """Add a proposition node to the hypergraph."""
    g = load_graph(path)
    node = g.add_node(
        statement=statement,
        belief=belief,
        domain=domain,
        state=state or "unverified",
    )
    save_graph(g, path)
    typer.echo(f"Added node {node.id} [{node.state}] {statement[:60]}")


@app.command()
def add_edge(
    premises: str = typer.Option(..., help="Comma-separated premise node IDs"),
    conclusion: str = typer.Option(..., help="Conclusion node ID"),
    module: Module = typer.Option(..., help="Module: plausible/experiment/lean"),
    confidence: float = typer.Option(..., help="Confidence score"),
    steps: str = typer.Option("", help="Semicolon-separated reasoning steps"),
    path: Path = typer.Option(DEFAULT_GRAPH_PATH, help="Graph file path"),
):
    """Add a reasoning hyperedge."""
    g = load_graph(path)
    premise_ids = [p.strip() for p in premises.split(",")]
    step_list = [s.strip() for s in steps.split(";") if s.strip()]
    edge = g.add_hyperedge(premise_ids, conclusion, module, step_list, confidence)
    save_graph(g, path)
    typer.echo(f"Added edge {edge.id}: [{premises}] -> {conclusion} "
               f"({module.value}, conf={confidence})")


@app.command()
def show(
    path: Path = typer.Option(DEFAULT_GRAPH_PATH, help="Graph file path"),
):
    """Show all nodes and edges in the hypergraph."""
    g = load_graph(path)
    if not g.nodes:
        typer.echo("Empty hypergraph.")
        return
    typer.echo("--- Nodes ---")
    for nid, node in g.nodes.items():
        st = getattr(node, "state", "unverified")
        typer.echo(f"  [{nid}] (prior={node.prior:.2f}, belief={node.belief:.2f}, {st}) {node.statement[:55]}")
    if g.edges:
        typer.echo("--- Edges ---")
        for eid, edge in g.edges.items():
            et = getattr(edge, "edge_type", "heuristic")
            premise_stmts = [g.nodes[p].statement[:25] for p in edge.premise_ids]
            concl_stmt = g.nodes[edge.conclusion_id].statement[:25]
            typer.echo(f"  [{eid}] {premise_stmts} -> {concl_stmt} "
                       f"({edge.module.value}, {et}, conf={edge.confidence:.2f})")


@app.command()
def propagate(
    path: Path = typer.Option(DEFAULT_GRAPH_PATH, help="Graph file path"),
    backend: str = typer.Option("gaia", help="Backend: gaia (Gaia BP, default) or energy (Łukasiewicz energy min)"),
):
    """Run belief propagation or energy minimization on the hypergraph."""
    g = load_graph(path)
    if backend == "energy":
        from dz_hypergraph.inference_energy import propagate_beliefs_energy
        iterations = propagate_beliefs_energy(g)
        typer.echo(f"Energy minimization finished in {iterations} iterations")
    else:
        iterations = propagate_beliefs(g)
        typer.echo(f"Belief propagation converged in {iterations} iterations")
    save_graph(g, path)
    for nid, node in g.nodes.items():
        st = getattr(node, "state", "?")
        typer.echo(f"  [{nid}] prior={node.prior:.4f} belief={node.belief:.4f} state={st} {node.statement[:45]}")


@app.command("extract-claims")
def extract_claims_cmd(
    prose: str = typer.Option(..., help="Reasoning prose to extract claims from"),
    context: str = typer.Option("", help="Additional context for extraction"),
    source_memo_id: str = typer.Option("cli_memo", help="Memo identifier"),
    model: Optional[str] = typer.Option(None, help="Override model name"),
):
    """Extract structured claims from free-form memo text."""
    from dz_verify.claim_pipeline import ClaimPipeline

    pipeline = ClaimPipeline()
    claims = pipeline.extract_claims(
        prose=prose,
        context=context,
        source_memo_id=source_memo_id,
        model=model,
    )
    typer.echo(
        json.dumps(
            [claim.model_dump() for claim in claims],
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command("verify-loop")
def verify_loop_cmd(
    path: Path = typer.Option(DEFAULT_GRAPH_PATH, help="Graph file path"),
    target: str = typer.Option(..., help="Target node ID"),
    iterations: int = typer.Option(3, help="Maximum verification iterations"),
    model: Optional[str] = typer.Option(None, help="Override model name"),
):
    """Run the verification-driven loop and persist graph updates."""
    from dz_hypergraph.config import CONFIG
    from dz_verify.verification_loop import VerificationLoop, VerificationLoopConfig

    graph = load_graph(path)
    config = VerificationLoopConfig(
        max_iterations=iterations,
        verification_parallel_workers=CONFIG.verification_parallel_workers,
        max_claims_per_memo=CONFIG.max_claims_per_memo,
        bp_propagation_threshold=CONFIG.bp_propagation_threshold,
        max_decompose_depth=CONFIG.max_decompose_depth,
    )
    loop = VerificationLoop(config=config, model=model or CONFIG.claim_extraction_model or None)
    result = loop.run(graph=graph, target_node_id=target, max_iterations=iterations)
    save_graph(graph, path)
    typer.echo(
        json.dumps(
            {
                "success": result.success,
                "iterations_completed": result.iterations_completed,
                "latest_feedback": result.latest_feedback,
                "traces": [trace.__dict__ for trace in result.traces],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command()
def htps_step(
    path: Path = typer.Option(DEFAULT_GRAPH_PATH, help="Graph file path"),
    root: str = typer.Option(..., help="Root node ID for HTPS (e.g. conjecture to prove)"),
    max_depth: int = typer.Option(20, help="Max selection depth"),
    c_puct: float = typer.Option(1.4, help="PUCT exploration constant"),
    state_path: Optional[Path] = typer.Option(None, help="Persistent HTPS state file path"),
):
    """One HTPS step: select path to leaf, evaluate, backup. Leaf is the suggested node to expand."""
    from dz_engine.htps import (
        htps_step as do_htps_step,
        load_htps_state,
        save_htps_state,
    )

    g = load_graph(path)
    if root not in g.nodes:
        typer.echo(f"Node {root} not found.", err=True)
        raise typer.Exit(1)
    resolved_state_path = state_path or path.with_name(path.stem + ".htps_state.json")
    state = load_htps_state(resolved_state_path)
    leaf_id, path_trace, value = do_htps_step(g, state, root, max_depth, c_puct)
    save_htps_state(state, resolved_state_path)
    typer.echo(f"Leaf: {leaf_id} (value={value:.4f})")
    typer.echo(f"Path: {path_trace}")
    node = g.nodes[leaf_id]
    typer.echo(f"  -> Expand node: {node.statement[:60]}")
    typer.echo(f"State saved to {resolved_state_path}")


@app.command("htps-run")
def htps_run(
    path: Path = typer.Option(DEFAULT_GRAPH_PATH, help="Graph file path"),
    root: str = typer.Option(..., help="Root node ID for HTPS (e.g. conjecture to prove)"),
    rounds: int = typer.Option(1, help="Number of HTPS expansion rounds"),
    max_depth: int = typer.Option(20, help="Max selection depth"),
    c_puct: float = typer.Option(1.4, help="PUCT exploration constant"),
    state_path: Optional[Path] = typer.Option(None, help="Persistent HTPS state file path"),
    model: Optional[str] = typer.Option(None, help="Override LLM model for expansions"),
    backend: str = typer.Option("bp", help="Propagation backend: bp or energy"),
):
    """Run HTPS for multiple rounds and expand selected leaves using the orchestrator."""
    from dz_engine.htps import (
        htps_step as do_htps_step,
        load_htps_state,
        save_htps_state,
    )
    from dz_engine.orchestrator import (
        execute_action,
        execute_bridge_followups,
        ingest_action_output,
        OrchestrationError,
    )
    from dz_hypergraph.strategy import suggest_module

    resolved_state_path = state_path or path.with_name(path.stem + ".htps_state.json")
    state = load_htps_state(resolved_state_path)

    for i in range(rounds):
        g = load_graph(path)
        if root not in g.nodes:
            typer.echo(f"Node {root} not found.", err=True)
            raise typer.Exit(1)
        leaf_id, path_trace, value = do_htps_step(g, state, root, max_depth, c_puct)
        save_htps_state(state, resolved_state_path)
        node = g.nodes[leaf_id]
        if node.is_locked():
            typer.echo(f"[round {i+1}] leaf={leaf_id} locked state={node.state} value={value:.4f}")
            continue
        module = suggest_module(g, leaf_id)
        typer.echo(f"[round {i+1}] leaf={leaf_id} module={module.value} value={value:.4f}")
        result = execute_action(g, leaf_id, module, model=model)
        if not result.success:
            typer.echo(f"  expansion failed: {result.message}", err=True)
            continue
        try:
            result = ingest_action_output(path, result, backend=backend)
        except OrchestrationError as e:
            typer.echo(f"  ingest failed: {e}", err=True)
            continue
        typer.echo(f"  ingest_edge_id={result.ingest_edge_id} message={result.message}")
        if module == Module.PLAUSIBLE and result.normalized_output:
            try:
                followups = execute_bridge_followups(
                    path,
                    leaf_id,
                    result.normalized_output,
                    judge_output=result.judge_output,
                    model=model,
                    backend=backend,
                )
            except OrchestrationError as e:
                typer.echo(f"  bridge follow-up failed: {e}", err=True)
                continue
            for followup in followups:
                typer.echo(
                    f"  bridge_action={followup.action} success={followup.success} "
                    f"target={followup.target_node_id} message={followup.message}"
                )


@app.command(name="next")
def next_action(
    path: Path = typer.Option(DEFAULT_GRAPH_PATH, help="Graph file path"),
    top_n: int = typer.Option(5, help="Number of suggestions to show"),
):
    """Suggest the next exploration action based on value-uncertainty trade-off."""
    g = load_graph(path)
    ranked = rank_nodes(g)
    if not ranked:
        typer.echo("No nodes to explore. Add conjectures or run plausible reasoning.")
        return
    typer.echo("--- Suggested next actions ---")
    for nid, priority in ranked[:top_n]:
        node = g.nodes[nid]
        module = suggest_module(g, nid)
        typer.echo(
            f"  [{nid}] priority={priority:.2f} belief={node.belief:.2f} "
            f"-> use {module.value}\n"
            f"    {node.statement[:70]}"
        )


@app.command()
def ingest(
    json_str: str = typer.Argument(..., help="JSON string of skill output"),
    path: Path = typer.Option(DEFAULT_GRAPH_PATH, help="Graph file path"),
):
    """Ingest a skill output (JSON) into the hypergraph."""
    g = load_graph(path)
    output = json.loads(json_str)
    edge = ingest_skill_output(g, output)
    save_graph(g, path)
    if edge is None:
        typer.echo("Conclusion marked refuted (no edge added).")
    else:
        typer.echo(f"Ingested edge {edge.id} ({edge.module.value}, conf={edge.confidence:.2f})")


# ---------- LLM commands ----------


@llm_app.command("models")
def llm_models():
    """List models from the configured LiteLLM-compatible endpoint."""
    from dz_hypergraph.tools.llm import list_models, LLMError

    try:
        models = list_models()
    except LLMError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    for model in models:
        typer.echo(model)


@llm_app.command("plausible")
def llm_plausible(
    direction: str = typer.Option(..., help="What area or conjecture to explore"),
    context: str = typer.Option("", help="Relevant graph context / known facts"),
    model: Optional[str] = typer.Option(None, help="Override model name"),
    output: Optional[Path] = typer.Option(None, help="Optional file to save JSON output"),
):
    """Run the plausible_reasoning skill with a real LLM."""
    from dz_hypergraph.tools.llm import run_skill, LLMError

    task_input = (
        f"Direction:\n{direction}\n\n"
        f"Context:\n{context or '(none provided)'}\n"
    )
    try:
        _, parsed = run_skill(
            "plausible_reasoning.skill.md",
            task_input,
            model=model,
        )
    except LLMError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    text = json.dumps(parsed, ensure_ascii=False, indent=2)
    if output:
        output.write_text(text, encoding="utf-8")
        typer.echo(f"Saved plausible output to {output}")
    else:
        typer.echo(text)


@llm_app.command("judge")
def llm_judge(
    json_str: Optional[str] = typer.Option(None, help="Reasoning hyperedge JSON to judge"),
    file: Optional[Path] = typer.Option(None, help="Path to reasoning hyperedge JSON file"),
    model: Optional[str] = typer.Option(None, help="Override model name"),
    output: Optional[Path] = typer.Option(None, help="Optional file to save JSON output"),
):
    """Run the judge skill with a real LLM."""
    from dz_hypergraph.tools.llm import run_skill, LLMError

    if not json_str and not file:
        typer.echo("Provide either --json-str or --file.", err=True)
        raise typer.Exit(1)
    if json_str and file:
        typer.echo("Use only one of --json-str or --file.", err=True)
        raise typer.Exit(1)

    payload = json_str
    if file:
        if not file.exists():
            typer.echo(f"File not found: {file}", err=True)
            raise typer.Exit(1)
        payload = file.read_text(encoding="utf-8")

    task_input = f"Hyperedge to evaluate:\n{payload}"
    try:
        _, parsed = run_skill(
            "judge.skill.md",
            task_input,
            model=model,
        )
    except LLMError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    text = json.dumps(parsed, ensure_ascii=False, indent=2)
    if output:
        output.write_text(text, encoding="utf-8")
        typer.echo(f"Saved judge output to {output}")
    else:
        typer.echo(text)


@llm_app.command("bridge")
def llm_bridge(
    path: Path = typer.Option(DEFAULT_GRAPH_PATH, help="Graph file path"),
    node_id: str = typer.Option(..., help="Target conjecture node ID"),
    reasoning_file: Path = typer.Option(..., help="Path to normalized reasoning JSON"),
    judge_file: Optional[Path] = typer.Option(None, help="Optional judge JSON for the same route"),
    model: Optional[str] = typer.Option(None, help="Override model name"),
    output: Optional[Path] = typer.Option(None, help="Optional file to save bridge plan JSON"),
):
    """Compile a reasoning route into a validated bridge-layer plan."""
    from dz_engine.orchestrator import run_bridge_planning_action, OrchestrationError

    g = load_graph(path)
    if node_id not in g.nodes:
        typer.echo(f"Node {node_id} not found.", err=True)
        raise typer.Exit(1)
    if not reasoning_file.exists():
        typer.echo(f"Reasoning file not found: {reasoning_file}", err=True)
        raise typer.Exit(1)
    reasoning = json.loads(reasoning_file.read_text(encoding="utf-8"))
    judge_output = None
    if judge_file:
        if not judge_file.exists():
            typer.echo(f"Judge file not found: {judge_file}", err=True)
            raise typer.Exit(1)
        judge_output = json.loads(judge_file.read_text(encoding="utf-8"))
    try:
        _, plan = run_bridge_planning_action(
            g,
            node_id,
            reasoning,
            judge_output=judge_output,
            model=model,
        )
    except (LLMError, OrchestrationError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    text = plan.model_dump_json(indent=2)
    if output:
        output.write_text(text, encoding="utf-8")
        typer.echo(f"Saved bridge plan to {output}")
    else:
        typer.echo(text)


@llm_app.command("experiment")
def llm_experiment(
    path: Path = typer.Option(DEFAULT_GRAPH_PATH, help="Graph file path"),
    node_id: str = typer.Option(..., help="Target conjecture node ID"),
    model: Optional[str] = typer.Option(None, help="Override model name"),
    output: Optional[Path] = typer.Option(None, help="Optional file to save JSON output"),
):
    """Run a real experiment skill, execute generated code, and return normalized output."""
    from dz_engine.orchestrator import execute_action

    g = load_graph(path)
    if node_id not in g.nodes:
        typer.echo(f"Node {node_id} not found.", err=True)
        raise typer.Exit(1)
    result = execute_action(g, node_id, Module.EXPERIMENT, model=model)
    if not result.success or not result.normalized_output:
        typer.echo(result.message, err=True)
        raise typer.Exit(1)
    text = json.dumps(result.normalized_output, ensure_ascii=False, indent=2)
    if output:
        output.write_text(text, encoding="utf-8")
        typer.echo(f"Saved experiment output to {output}")
    else:
        typer.echo(text)


@llm_app.command("lean")
def llm_lean(
    path: Path = typer.Option(DEFAULT_GRAPH_PATH, help="Graph file path"),
    node_id: str = typer.Option(..., help="Target conjecture node ID"),
    model: Optional[str] = typer.Option(None, help="Override model name"),
    output: Optional[Path] = typer.Option(None, help="Optional file to save JSON output"),
):
    """Run the Lean proof skill and strictly verify the returned code."""
    from dz_engine.orchestrator import execute_action

    g = load_graph(path)
    if node_id not in g.nodes:
        typer.echo(f"Node {node_id} not found.", err=True)
        raise typer.Exit(1)
    result = execute_action(g, node_id, Module.LEAN, model=model)
    if not result.success or not result.normalized_output:
        typer.echo(result.message, err=True)
        raise typer.Exit(1)
    text = json.dumps(result.normalized_output, ensure_ascii=False, indent=2)
    if output:
        output.write_text(text, encoding="utf-8")
        typer.echo(f"Saved lean output to {output}")
    else:
        typer.echo(text)


@llm_app.command("lean-decompose")
def llm_lean_decompose(
    path: Path = typer.Option(DEFAULT_GRAPH_PATH, help="Graph file path"),
    node_id: str = typer.Option(..., help="Target conjecture node ID"),
    model: Optional[str] = typer.Option(None, help="Override model name"),
    backend: str = typer.Option("bp", help="Propagation backend: bp or energy"),
):
    """Generate a Lean skeleton, extract real subgoals, and ingest a decomposition edge."""
    from dz_engine.orchestrator import (
        ingest_decomposition_output,
        run_lean_decompose_action,
        OrchestrationError,
    )

    g = load_graph(path)
    if node_id not in g.nodes:
        typer.echo(f"Node {node_id} not found.", err=True)
        raise typer.Exit(1)
    try:
        raw, normalized, subgoals = run_lean_decompose_action(g, node_id, model=model)
        result = ingest_decomposition_output(
            path,
            node_id,
            normalized,
            subgoals,
            backend=backend,
        )
    except OrchestrationError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    typer.echo(
        json.dumps(
            {
                "ingest_edge_id": result.ingest_edge_id,
                "created_node_ids": result.created_node_ids,
                "num_subgoals": len(subgoals),
                "raw": raw,
            },
            ensure_ascii=False,
        )
    )


@llm_app.command("loop")
def llm_loop(
    path: Path = typer.Option(DEFAULT_GRAPH_PATH, help="Graph file path"),
    rounds: int = typer.Option(1, help="Number of loop iterations"),
    model: Optional[str] = typer.Option(None, help="Override model name"),
    backend: str = typer.Option("bp", help="Propagation backend: bp or energy"),
    log: Optional[Path] = typer.Option(None, help="Optional JSONL log path"),
):
    """Run a real discovery loop: choose action, execute, ingest, propagate, repeat."""
    from dz_engine.orchestrator import run_loop

    results = run_loop(path, rounds=rounds, model=model, backend=backend, log_path=log)
    if not results:
        typer.echo("No actionable nodes found.")
        return
    for item in results:
        typer.echo(
            json.dumps(
                {
                    "target_node_id": item.target_node_id,
                    "selected_module": item.selected_module,
                    "success": item.success,
                    "message": item.message,
                    "ingest_edge_id": item.ingest_edge_id,
                },
                ensure_ascii=False,
            )
        )


# ---------- Lean commands ----------


@lean_app.command("init")
def lean_init(
    path: Optional[Path] = typer.Option(None, help="Workspace path (default: lean_workspace in project)"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing files"),
):
    """Initialize or recreate the Lean workspace with Mathlib."""
    from dz_hypergraph.tools.lean import init_workspace, get_workspace_path

    base = init_workspace(path, force=force)
    typer.echo(f"Lean workspace initialized at {base}")
    typer.echo("Next: cd <workspace> && lake update && lake exe cache get && lake build")


@lean_app.command("verify")
def lean_verify(
    file: Path = typer.Option(..., "--file", "-f", help="Path to .lean file with proof"),
    workspace: Path = typer.Option(None, help="Lean workspace path"),
    timeout: int = typer.Option(1200, help="Build timeout in seconds"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Print lake build output in real time"),
):
    """Verify a Lean proof by copying to workspace and running lake build."""
    from dz_hypergraph.tools.lean import verify_proof, get_workspace_path

    if not file.exists():
        typer.echo(f"File not found: {file}", err=True)
        raise typer.Exit(1)
    code = file.read_text(encoding="utf-8")
    base = workspace if workspace else get_workspace_path()
    result = verify_proof(code, base, timeout=timeout, stream=stream)
    if result.success:
        typer.echo("Build succeeded.")
        if result.formal_statement:
            typer.echo(f"Theorem: {result.formal_statement}")
    else:
        typer.echo("Build failed.", err=True)
        typer.echo(result.error_message or result.stderr or "Unknown error", err=True)
        raise typer.Exit(1)


@lean_app.command("path")
def lean_path():
    """Print the Lean workspace path."""
    from dz_hypergraph.tools.lean import get_workspace_path

    typer.echo(get_workspace_path())


@benchmark_app.command("run-suite")
def benchmark_run_suite(
    suite: Path = typer.Option(..., help="Path to benchmark suite JSON config"),
    repeats: Optional[int] = typer.Option(None, help="Optional override for repeat count"),
    output_root: Optional[Path] = typer.Option(
        None,
        help="Optional evaluation root directory",
    ),
):
    """Run a benchmark suite with isolated per-run artifacts and aggregated reports."""
    from dz_engine.benchmark import BenchmarkError, run_suite

    try:
        result = run_suite(
            suite.resolve(),
            repeats_override=repeats,
            output_root=output_root.resolve() if output_root else None,
        )
    except BenchmarkError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    typer.echo(f"Suite run directory: {result.suite_run_dir}")
    typer.echo(f"Suite summary: {result.suite_summary_path}")
    typer.echo(f"Suite scorecard: {result.suite_scorecard_path}")


@benchmark_app.command("expert-iterate")
def benchmark_expert_iterate(
    suite: Path = typer.Option(..., help="Path to benchmark suite JSON config"),
    iterations: int = typer.Option(1, help="Number of expert-iteration cycles"),
    repeats: Optional[int] = typer.Option(None, help="Optional override for repeat count"),
    output_root: Optional[Path] = typer.Option(None, help="Optional evaluation root directory"),
):
    """Run benchmark collection with replay-buffer capture, then train one or more ExIt cycles."""
    from dz_engine.benchmark import BenchmarkError, run_suite
    from dz_hypergraph.config import CONFIG
    from dz_engine.expert_iteration import ExperienceBuffer, ExpertIterationLoop
    from dz_engine.value_net import ProcessAdvantageVerifier

    buffer = ExperienceBuffer()
    pav = ProcessAdvantageVerifier()
    loop = ExpertIterationLoop(
        experience_buffer=buffer,
        pav=pav,
        checkpoint_dir=Path(CONFIG.expert_iter_checkpoint_dir),
    )
    try:
        run_suite(
            suite.resolve(),
            repeats_override=repeats,
            output_root=output_root.resolve() if output_root else None,
            experience_buffer=buffer,
        )
        results = loop.run_n_iterations(iterations)
    except BenchmarkError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)
    typer.echo(json.dumps([item.to_dict() for item in results], ensure_ascii=False, indent=2))


@benchmark_app.command("train-pav")
def benchmark_train_pav(
    experience_buffer_path: Path = typer.Option(..., help="Path to saved experience_buffer.json"),
):
    """Train PAV once from an existing replay buffer checkpoint."""
    from dz_hypergraph.config import CONFIG
    from dz_engine.expert_iteration import ExperienceBuffer, ExpertIterationLoop
    from dz_engine.value_net import ProcessAdvantageVerifier

    buffer = ExperienceBuffer()
    buffer.load(experience_buffer_path)
    loop = ExpertIterationLoop(
        experience_buffer=buffer,
        pav=ProcessAdvantageVerifier(),
        checkpoint_dir=Path(CONFIG.expert_iter_checkpoint_dir),
    )
    result = loop.run_iteration()
    typer.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


@benchmark_app.command("train-neural-bp")
def benchmark_train_neural_bp(
    experience_buffer_path: Path = typer.Option(..., help="Path to saved experience_buffer.json"),
):
    """Train Neural BP once from an existing replay buffer checkpoint."""
    from dz_hypergraph.config import CONFIG
    from dz_hypergraph.neural_bp import NeuralBPCorrector
    from dz_engine.expert_iteration import ExperienceBuffer, ExpertIterationLoop

    buffer = ExperienceBuffer()
    buffer.load(experience_buffer_path)
    loop = ExpertIterationLoop(
        experience_buffer=buffer,
        neural_bp=NeuralBPCorrector(),
        checkpoint_dir=Path(CONFIG.expert_iter_checkpoint_dir),
    )
    result = loop.run_iteration()
    typer.echo(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
