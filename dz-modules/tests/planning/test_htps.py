"""Tests for HTPS skeleton."""

import pytest
from pathlib import Path
from dz_hypergraph.models import HyperGraph, Module
from dz_engine.htps import (
    HTPSState,
    htps_select,
    htps_backup,
    htps_step,
    save_htps_state,
    load_htps_state,
)


class TestHTPS:
    def test_select_to_leaf_no_edges(self):
        g = HyperGraph()
        root = g.add_node("conjecture", belief=0.5)
        state = HTPSState()
        leaf_id, path = htps_select(g, state, root.id, max_depth=5)
        assert leaf_id == root.id
        assert path == []

    def test_select_follows_edge_to_premise(self):
        g = HyperGraph()
        a = g.add_node("axiom", belief=1.0, state="proven")
        b = g.add_node("conj", belief=0.5)
        g.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, [], 0.8)
        state = HTPSState()
        leaf_id, path = htps_select(g, state, b.id, max_depth=5)
        assert len(path) >= 1
        assert path[0][0] == b.id
        assert leaf_id == a.id or leaf_id == b.id

    def test_backup_updates_N_and_Q(self):
        state = HTPSState()
        path = [("n1", "e1"), ("n2", "e2")]
        htps_backup(state, path, 0.7)
        assert state.get_N("n1", "e1") == 1
        assert state.get_Q("n1", "e1") == pytest.approx(0.7)
        htps_backup(state, path, 0.9)
        assert state.get_N("n1", "e1") == 2
        assert state.get_Q("n1", "e1") == pytest.approx(0.8)

    def test_htps_step_returns_leaf_and_value(self):
        g = HyperGraph()
        a = g.add_node("ax", belief=1.0, state="proven")
        b = g.add_node("conj", belief=0.4)
        g.add_hyperedge([a.id], b.id, Module.PLAUSIBLE, [], 0.7)
        state = HTPSState()
        leaf_id, path, value = htps_step(g, state, b.id)
        assert leaf_id in g.nodes
        assert 0 <= value <= 1
        assert state.get_N(b.id, path[0][1]) == 1 if path else True

    def test_state_roundtrip(self, tmp_graph_dir):
        path = tmp_graph_dir / "htps_state.json"
        state = HTPSState()
        htps_backup(state, [("n1", "e1"), ("n1", "e2")], 0.7)
        save_htps_state(state, path)
        loaded = load_htps_state(path)
        assert loaded.get_N("n1", "e1") == 1
        assert loaded.get_Q("n1", "e1") == pytest.approx(0.7)
        assert loaded.get_N("n1", "e2") == 1
