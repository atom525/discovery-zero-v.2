"""End-to-end smoke test for the Discovery Zero pipeline."""

from dz_hypergraph.models import HyperGraph, Module
from dz_hypergraph.ingest import ingest_skill_output
from dz_hypergraph.inference import propagate_beliefs
from dz_hypergraph.strategy import rank_nodes, suggest_module


def test_full_discovery_pipeline():
    g = HyperGraph()

    # Step 1: Seed axioms
    ax1 = g.add_node("Two points determine a line", belief=1.0, state="proven", domain="geometry")
    ax2 = g.add_node("Midpoint divides segment into two equal parts", belief=1.0, state="proven", domain="geometry")

    # Step 2: Plausible reasoning
    plausible_output = {
        "premises": [
            {"id": ax1.id, "statement": ax1.statement},
            {"id": ax2.id, "statement": ax2.statement},
        ],
        "steps": [
            "Consider triangle ABC with midpoints M, N of sides AB, AC",
            "By similarity of triangles AMN and ABC (ratio 1:2)",
            "MN should be parallel to BC and half its length",
        ],
        "conclusion": {
            "statement": "Triangle midline is parallel to the third side and half its length",
            "formal_statement": None,
        },
        "module": "plausible",
        "confidence": 0.55,
        "domain": "geometry",
    }
    e1 = ingest_skill_output(g, plausible_output)
    propagate_beliefs(g)

    conjecture_id = e1.conclusion_id
    assert g.nodes[conjecture_id].belief > 0.0
    assert g.nodes[conjecture_id].belief < 0.7

    assert suggest_module(g, conjecture_id) in (Module.PLAUSIBLE, Module.EXPERIMENT)

    # Step 3: Experiment
    experiment_output = {
        "premises": [
            {"id": conjecture_id, "statement": g.nodes[conjecture_id].statement},
        ],
        "steps": [
            "import numpy as np\n# 1000 random triangles tested",
            "Results: 1000/1000 passed, max error = 2.1e-15",
        ],
        "conclusion": {
            "statement": "Triangle midline is parallel to the third side and half its length",
        },
        "module": "experiment",
        "confidence": 0.93,
        "domain": "geometry",
    }
    ingest_skill_output(g, experiment_output)
    propagate_beliefs(g)

    assert g.nodes[conjecture_id].belief > 0.12
    assert suggest_module(g, conjecture_id) in (Module.PLAUSIBLE, Module.EXPERIMENT, Module.LEAN)

    # Step 4: Lean proof
    lean_output = {
        "premises": [
            {"id": ax2.id, "statement": ax2.statement},
        ],
        "steps": [
            "theorem midline_parallel (A B C : Point) : ...",
            "(full lean proof)",
        ],
        "conclusion": {
            "statement": "Triangle midline is parallel to the third side and half its length",
            "formal_statement": "theorem midline_parallel ...",
        },
        "module": "lean",
        "confidence": 0.99,
        "domain": "geometry",
    }
    ingest_skill_output(g, lean_output)
    propagate_beliefs(g)

    assert g.nodes[conjecture_id].belief >= 0.99

    s = g.summary()
    assert s["num_nodes"] >= 3
    assert s["num_edges"] >= 3


def test_multi_conjecture_ranking():
    g = HyperGraph()
    ax = g.add_node("axiom", belief=1.0)
    c1 = g.add_node("easy conjecture", belief=0.9)
    c2 = g.add_node("hard conjecture", belief=0.2)
    c3 = g.add_node("medium conjecture", belief=0.5)

    ranked = rank_nodes(g)
    ids = [r[0] for r in ranked]

    assert ids[0] == c2.id
