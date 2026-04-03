from __future__ import annotations

import pytest

from gaia.bp.exact import exact_inference
from gaia.bp.factor_graph import CROMWELL_EPS, FactorGraph, FactorType
from gaia.bp.potentials import evaluate_potential


def test_conjunction_potential_truth_table():
    fg = FactorGraph()
    fg.add_variable("A", 0.5)
    fg.add_variable("B", 0.5)
    fg.add_variable("M", 0.5)
    fg.add_factor(
        "conj",
        FactorType.CONJUNCTION,
        ["A", "B"],
        "M",
    )
    factor = fg.factors[0]

    assert evaluate_potential(factor, {"A": 1, "B": 1, "M": 1}) == pytest.approx(1.0 - CROMWELL_EPS)
    assert evaluate_potential(factor, {"A": 1, "B": 1, "M": 0}) == pytest.approx(CROMWELL_EPS)
    assert evaluate_potential(factor, {"A": 1, "B": 0, "M": 0}) == pytest.approx(1.0 - CROMWELL_EPS)
    assert evaluate_potential(factor, {"A": 0, "B": 0, "M": 1}) == pytest.approx(CROMWELL_EPS)


def test_soft_entailment_truth_table():
    fg = FactorGraph()
    fg.add_variable("A", 0.5)
    fg.add_variable("B", 0.5)
    fg.add_factor(
        "soft",
        FactorType.SOFT_ENTAILMENT,
        ["A"],
        "B",
        p1=0.85,
        p2=0.7,
    )
    factor = fg.factors[0]

    assert evaluate_potential(factor, {"A": 1, "B": 1}) == pytest.approx(0.85)
    assert evaluate_potential(factor, {"A": 1, "B": 0}) == pytest.approx(0.15)
    assert evaluate_potential(factor, {"A": 0, "B": 0}) == pytest.approx(0.7)
    assert evaluate_potential(factor, {"A": 0, "B": 1}) == pytest.approx(0.3)


def test_soft_entailment_requires_supportive_parameters():
    fg = FactorGraph()
    fg.add_variable("A", 0.5)
    fg.add_variable("B", 0.5)
    with pytest.raises(ValueError, match="p1 \\+ p2 > 1"):
        fg.add_factor(
            "bad_soft",
            FactorType.SOFT_ENTAILMENT,
            ["A"],
            "B",
            p1=0.4,
            p2=0.5,
        )


def test_exact_inference_supports_new_factor_types():
    fg = FactorGraph()
    fg.add_variable("A", 0.8)
    fg.add_variable("B", 0.9)
    fg.add_variable("M", 0.5)
    fg.add_variable("C", 0.5)
    fg.add_factor(
        "conj",
        FactorType.CONJUNCTION,
        ["A", "B"],
        "M",
    )
    fg.add_factor(
        "soft",
        FactorType.SOFT_ENTAILMENT,
        ["M"],
        "C",
        p1=0.92,
        p2=0.55,
    )

    beliefs, z = exact_inference(fg)
    assert z > 0.0
    assert beliefs["C"] > 0.5
