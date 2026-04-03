import json
import pytest
from dz_hypergraph.models import HyperGraph, Module
from dz_hypergraph.ingest import ingest_skill_output


class TestIngest:
    def test_ingest_plausible_with_existing_premises(self):
        g = HyperGraph()
        n1 = g.add_node("triangle ABC", belief=1.0)
        n2 = g.add_node("M is midpoint of AB", belief=1.0)
        output = {
            "premises": [
                {"id": n1.id, "statement": "triangle ABC"},
                {"id": n2.id, "statement": "M is midpoint of AB"},
            ],
            "steps": ["By similarity..."],
            "conclusion": {
                "statement": "MN is parallel to BC",
                "formal_statement": None,
            },
            "module": "plausible",
            "domain": "geometry",
        }
        edge = ingest_skill_output(g, output)
        assert edge.module == Module.PLAUSIBLE
        assert edge.conclusion_id in g.nodes
        assert g.nodes[edge.conclusion_id].statement == "MN is parallel to BC"

    def test_ingest_creates_new_premise_nodes(self):
        g = HyperGraph()
        output = {
            "premises": [
                {"id": None, "statement": "Let X be a prime number"},
            ],
            "steps": ["By Fermat's little theorem..."],
            "conclusion": {
                "statement": "a^X = a mod X",
            },
            "module": "plausible",
            "domain": "number_theory",
        }
        edge = ingest_skill_output(g, output)
        assert len(g.nodes) == 2  # 1 new premise + 1 conclusion

    def test_ingest_experiment(self):
        g = HyperGraph()
        n1 = g.add_node("conjecture: sum of angles = 180", belief=0.5)
        output = {
            "premises": [
                {"id": n1.id, "statement": "conjecture: sum of angles = 180"},
            ],
            "steps": ["code...", "1000/1000 passed"],
            "conclusion": {
                "statement": "Experimental evidence supports: sum of angles = 180",
            },
            "module": "experiment",
        }
        edge = ingest_skill_output(g, output)
        assert edge.module == Module.EXPERIMENT

    def test_ingest_with_confidence(self):
        g = HyperGraph()
        output = {
            "premises": [],
            "steps": [],
            "conclusion": {"statement": "trivial"},
            "module": "plausible",
            "confidence": 0.42,
        }
        edge = ingest_skill_output(g, output)
        assert edge.confidence == 0.42

    def test_ingest_default_confidence(self):
        g = HyperGraph()
        output = {
            "premises": [],
            "steps": [],
            "conclusion": {"statement": "something"},
            "module": "plausible",
        }
        edge = ingest_skill_output(g, output)
        assert edge.confidence == 0.5  # default for plausible

    def test_ingest_rejects_failed_lean_output(self):
        g = HyperGraph()
        output = {
            "status": "failed",
            "last_error": "type mismatch",
            "attempts": 1,
        }
        with pytest.raises(ValueError, match="Cannot ingest failed"):
            ingest_skill_output(g, output)

    def test_ingest_rejects_output_without_module(self):
        g = HyperGraph()
        output = {
            "premises": [],
            "conclusion": {"statement": "x"},
        }
        with pytest.raises(ValueError, match="'module'"):
            ingest_skill_output(g, output)

    def test_ingest_lean_sets_conclusion_proven_when_premises_proven(self):
        g = HyperGraph()
        a = g.add_node("a,b,c > 0", belief=1.0, state="proven")
        b = g.add_node("Nesbitt holds", belief=1.0, state="proven")
        output = {
            "premises": [
                {"id": a.id, "statement": "a,b,c > 0"},
                {"id": b.id, "statement": "Nesbitt holds"},
            ],
            "conclusion": {"statement": "Squared-Nesbitt ≥ 3/4", "formal_statement": "discovery_squared_nesbitt"},
            "module": "lean",
            "steps": ["lake build"],
        }
        edge = ingest_skill_output(g, output)
        assert edge is not None
        assert edge.edge_type == "formal"
        c = g.nodes[edge.conclusion_id]
        assert c.state == "proven"
        assert c.belief == 1.0

    def test_ingest_refutation_marks_conclusion_refuted(self):
        """Experiment refutation should weaken (not hard-refute) the node.
        Only formal (Lean) verification may hard-refute."""
        g = HyperGraph()
        n = g.add_node("Bad conjecture", belief=0.5)
        assert g.nodes[n.id].state == "unverified"
        out = {
            "module": "experiment",
            "outcome": "refuted",
            "conclusion": {"statement": "Bad conjecture"},
        }
        edge = ingest_skill_output(g, out)
        assert edge is None
        # Experiment weakens belief sharply but does NOT hard-refute
        assert g.nodes[n.id].state == "unverified"
        assert g.nodes[n.id].prior < 0.1
        assert g.nodes[n.id].prior > 0.0

    def test_ingest_lean_refutation_marks_conclusion_refuted(self):
        """Lean (formal) refutation SHOULD hard-refute the node."""
        g = HyperGraph()
        n = g.add_node("Wrong lemma", belief=0.5)
        out = {
            "module": "lean",
            "outcome": "refuted",
            "conclusion": {"statement": "Wrong lemma"},
        }
        edge = ingest_skill_output(g, out)
        assert edge is None
        assert g.nodes[n.id].state == "refuted"
        assert g.nodes[n.id].belief == 0.0

    def test_ingest_filters_self_loop_when_target_is_premise(self):
        """When target_node_id causes conclusion to be the same as a premise,
        the self-loop premise should be silently removed."""
        g = HyperGraph()
        n1 = g.add_node("supporting fact", belief=0.9)
        target = g.add_node("the conjecture", belief=0.5)
        output = {
            "premises": [
                {"id": n1.id, "statement": "supporting fact"},
                {"id": target.id, "statement": "the conjecture"},
            ],
            "steps": ["experiment verified the conjecture"],
            "conclusion": {"statement": "the conjecture"},
            "module": "experiment",
        }
        edge = ingest_skill_output(g, output, target_node_id=target.id)
        assert edge is not None
        assert target.id not in edge.premise_ids
        assert n1.id in edge.premise_ids
        assert edge.conclusion_id == target.id

    def test_ingest_refutation_uses_canonical_statement_matching(self):
        g = HyperGraph()
        n = g.add_node("Bad conjecture.", belief=0.5)
        out = {
            "module": "experiment",
            "outcome": "refuted",
            "conclusion": {"statement": "Bad conjecture"},
        }
        edge = ingest_skill_output(g, out)
        assert edge is None
        assert len(g.nodes) == 1
        # Experiment weakens but does not hard-refute
        assert g.nodes[n.id].state == "unverified"
        assert g.nodes[n.id].prior < 0.1
