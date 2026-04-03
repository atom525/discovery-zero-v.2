"""Dual-state energy minimization: Łukasiewicz logic + global energy (RENEW).

Uses analytic gradients of the Łukasiewicz t-norm constraint energy,
replacing the prior O(N·E) finite-difference approach with O(E) per iteration.

Łukasiewicz edge energy for edge e with premises P, conclusion c:
  E(e) = max(0, Σ(x_p for p in P) - |P| + 1 - x_c)

Analytic partial derivatives:
  ∂E(e)/∂x_c = -w_e      when E(e) > 0, else 0
  ∂E(e)/∂x_p = +w_e      when E(e) > 0, else 0  (for each p in P)

Global gradient:
  ∂E_total/∂x_n = Σ_{e: n is conclusion} (-w_e · 1_{E(e)>0})
                + Σ_{e: n is premise}    (+w_e · 1_{E(e)>0})

This reduces gradient computation from O(N·E) to O(E) per iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gaia.bp.factor_graph import CROMWELL_EPS

from dz_hypergraph.models import HyperGraph, Module


@dataclass
class EnergyConfig:
    """Configuration for energy minimization."""

    weight_plausible: float = 1.0
    weight_decomposition: float = 1.5
    weight_experiment: float = 5.0
    weight_formal: float = 100.0
    max_iterations: int = 200
    step_size: float = 0.1
    tol: float = 1e-6


def _edge_weight(edge, config: EnergyConfig) -> float:
    if edge.edge_type == "formal":
        return config.weight_formal
    if edge.edge_type == "decomposition":
        return config.weight_decomposition
    if edge.module == Module.EXPERIMENT:
        return config.weight_experiment
    return config.weight_plausible


def _single_edge_energy(*args) -> float:
    """Łukasiewicz-style edge energy: max(0, Σ(premises) - |P| + 1 - x_conclusion)."""
    # Backward-compatible signature support:
    #   _single_edge_energy(edge, x)
    #   _single_edge_energy(graph, edge, x)
    if len(args) == 2:
        edge, x = args
    elif len(args) == 3:
        _, edge, x = args
    else:
        raise TypeError("_single_edge_energy expects (edge, x) or (graph, edge, x)")
    pre_sum = sum(x.get(pid, 0.5) for pid in edge.premise_ids)
    x_c = x.get(edge.conclusion_id, 0.5)
    k = len(edge.premise_ids)
    if k == 0:
        return max(0.0, 1.0 - x_c)
    return max(0.0, pre_sum - k + 1.0 - x_c)


def _global_energy(
    graph: HyperGraph,
    x: dict[str, float],
    config: EnergyConfig,
) -> float:
    total = 0.0
    for edge in graph.edges.values():
        concl = graph.nodes.get(edge.conclusion_id)
        if concl is None or concl.state != "unverified":
            continue
        w = _edge_weight(edge, config)
        total += w * _single_edge_energy(edge, x)
    return total


def _analytic_gradient(
    graph: HyperGraph,
    x: dict[str, float],
    config: EnergyConfig,
    unverified_ids: set[str],
) -> dict[str, float]:
    """
    Compute the analytic gradient of the global energy w.r.t. all unverified nodes.

    O(E) per call — each edge contributes to at most |P| + 1 gradient components.
    """
    grad: dict[str, float] = {nid: 0.0 for nid in unverified_ids}

    for edge in graph.edges.values():
        concl = graph.nodes.get(edge.conclusion_id)
        if concl is None or concl.state != "unverified":
            continue
        e_val = _single_edge_energy(edge, x)
        if e_val <= 0.0:
            continue  # inactive constraint, gradient is zero
        w = _edge_weight(edge, config)

        # ∂E/∂x_c = -w  (active constraint)
        cid = edge.conclusion_id
        if cid in grad:
            grad[cid] -= w

        # ∂E/∂x_p = +w  for each premise (active constraint)
        for pid in edge.premise_ids:
            if pid in grad:
                grad[pid] += w

    return grad


def propagate_beliefs_energy(
    graph: HyperGraph,
    config: Optional[EnergyConfig] = None,
) -> int:
    """
    Update beliefs of unverified nodes by minimizing Łukasiewicz global energy.

    Uses analytic gradients — O(E) per iteration vs O(N·E) finite differences.

    Proven nodes stay at 1.0; refuted at 0.0.  Uses projected gradient descent.
    Returns number of iterations performed.
    """
    if config is None:
        config = EnergyConfig()

    unverified_ids = {
        nid for nid, n in graph.nodes.items() if n.state == "unverified"
    }
    if not unverified_ids:
        return 0

    x = {nid: graph.nodes[nid].belief for nid in unverified_ids}
    eta = config.step_size
    iterations = config.max_iterations

    for it in range(config.max_iterations):
        grad = _analytic_gradient(graph, x, config, unverified_ids)
        max_change = 0.0
        for nid in unverified_ids:
            g = grad.get(nid, 0.0)
            new_x = max(CROMWELL_EPS, min(1.0 - CROMWELL_EPS, x[nid] - eta * g))
            change = abs(new_x - x[nid])
            max_change = max(max_change, change)
            x[nid] = new_x

        if max_change < config.tol:
            iterations = it + 1
            break

    for nid in unverified_ids:
        graph.nodes[nid].belief = x[nid]

    return iterations
