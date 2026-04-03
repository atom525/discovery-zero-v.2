"""Ingest structured skill output into the reasoning hypergraph.

Applies dual-state transitions:
  - Lean success => conclusion Proven (when premises are Proven or axiom)
  - outcome "refuted" => conclusion Refuted (ONLY for formal/Lean refutation)
  - outcome "weakened" => belief penalty on conclusion (experiment-level soft refutation)
  - outcome "inconclusive" => no state change, no edge (experiment failure without evidence)
"""

from __future__ import annotations

from gaia.bp.factor_graph import CROMWELL_EPS

from dz_hypergraph.config import CONFIG
from dz_hypergraph.models import HyperGraph, Hyperedge, Module

DEFAULT_CONFIDENCE = {
    Module.PLAUSIBLE: CONFIG.default_confidence_plausible,
    Module.EXPERIMENT: CONFIG.default_confidence_experiment,
    Module.LEAN: CONFIG.default_confidence_lean,
    Module.ANALOGY: CONFIG.default_confidence_analogy,
    Module.DECOMPOSE: CONFIG.default_confidence_decompose,
    Module.SPECIALIZE: CONFIG.default_confidence_specialize,
    Module.RETRIEVE: CONFIG.default_confidence_retrieve,
}
DEFAULT_UNVERIFIED_CLAIM_PRIOR = CONFIG.unverified_claim_prior
EXPERIMENT_PRIOR_CAP = CONFIG.experiment_prior_cap
VERIFIED_PRIOR_FLOOR = CONFIG.verified_prior_floor
INCONCLUSIVE_PRIOR_CAP = CONFIG.inconclusive_prior_cap
REFUTATION_PRIOR_MULTIPLIER = CONFIG.refutation_prior_multiplier


def _is_axiom_or_proven(graph: HyperGraph, node_id: str) -> bool:
    """True if node is axiom (no incoming edges) with belief 1, or state is proven."""
    node = graph.nodes[node_id]
    if node.state == "proven":
        return True
    incoming = graph.get_edges_to(node_id)
    if not incoming and node.belief >= 1.0:
        return True
    return False


def _all_premises_proven_or_axiom(graph: HyperGraph, premise_ids: list[str]) -> bool:
    return all(_is_axiom_or_proven(graph, pid) for pid in premise_ids)


def _set_node_proven(graph: HyperGraph, node_id: str) -> None:
    n = graph.nodes[node_id]
    n.state = "proven"
    n.belief = 1.0
    n.prior = 1.0


def _set_node_refuted(graph: HyperGraph, node_id: str) -> None:
    n = graph.nodes[node_id]
    n.state = "refuted"
    n.belief = 0.0
    n.prior = 0.0


def ingest_skill_output(
    graph: HyperGraph,
    output: dict,
    *,
    target_node_id: str | None = None,
) -> Hyperedge | None:
    """Parse skill output JSON and add nodes/edges; apply state transitions.

    Args:
        graph: The HyperGraph to mutate.
        output: Normalised skill output dict.
        target_node_id: The MCTS target graph node ID.  When supplied, the
            conclusion statement is matched against this node using a
            canonicalised prefix comparison.  If they describe the same
            proposition (e.g. differ only by minor wording like added variable
            lists), the existing target node is reused as the conclusion instead
            of creating a new duplicate node.  This ensures that plausible
            reasoning edges point to the actual target node tracked by MCTS
            rather than a semantically equivalent but disconnected duplicate.

    Refutation: pass outcome="refuted" (and conclusion) to mark conclusion Refuted
    without adding a supporting edge. For Lean, success implies conclusion can
    become Proven if all premises are Proven or axiom.

    Raises:
        ValueError: If output is a failed Lean result (status == "failed")
            or missing required key "module".
    """
    if output.get("status") == "failed":
        raise ValueError(
            "Cannot ingest failed skill output. "
            "Use the result only when LeanVerifyResult.success is True."
        )
    if "module" not in output:
        raise ValueError(
            "Skill output must contain 'module' (plausible, experiment, or lean)."
        )
    module = Module(output["module"])
    outcome = output.get("outcome", "supported")

    provenance = output.get("provenance", module.value)

    if outcome == "refuted":
        # Only formal (Lean) verification may hard-refute a node.
        # Experiments can produce false refutations (e.g. testing an incorrect
        # auxiliary formula instead of the target), so they are downgraded to
        # "weakened" — a strong belief penalty that still lets BP recover if
        # other evidence supports the claim.
        if module == Module.EXPERIMENT:
            penalty = float(output.get("confidence", 0.9))
            conclusion = output.get("conclusion")
            if not conclusion:
                return None
            conclusion_statement = (
                conclusion if isinstance(conclusion, str) else conclusion.get("statement", "")
            )
            existing = graph.find_node_ids_by_statement(conclusion_statement)
            if existing:
                node = graph.nodes[existing[0]]
                if not node.is_locked():
                    node.prior = max(CROMWELL_EPS, node.prior * (1.0 - penalty))
            else:
                weakened_belief = max(CROMWELL_EPS, 0.5 * (1.0 - penalty))
                graph.add_node(
                    statement=conclusion_statement,
                    belief=weakened_belief,
                    prior=weakened_belief,
                    domain=output.get("domain"),
                    provenance=provenance,
                )
            return None

        conclusion = output.get("conclusion")
        if not conclusion:
            raise ValueError("Refutation output must contain 'conclusion'.")
        conclusion_statement = (
            conclusion if isinstance(conclusion, str) else conclusion.get("statement", "")
        )
        existing = graph.find_node_ids_by_statement(conclusion_statement)
        if existing:
            _set_node_refuted(graph, existing[0])
        else:
            node = graph.add_node(
                statement=conclusion_statement,
                belief=CROMWELL_EPS,
                prior=CROMWELL_EPS,
                domain=output.get("domain"),
                provenance=provenance,
            )
            _set_node_refuted(graph, node.id)
        return None

    if outcome == "weakened":
        # Soft refutation: reduce belief/prior proportionally but do NOT set state=refuted.
        # This allows BP to propagate a moderate negative signal without killing the node.
        conclusion = output.get("conclusion")
        if not conclusion:
            return None
        conclusion_statement = (
            conclusion if isinstance(conclusion, str) else conclusion.get("statement", "")
        )
        penalty = float(output.get("confidence", 0.3))
        existing = graph.find_node_ids_by_statement(conclusion_statement)
        if existing:
            node = graph.nodes[existing[0]]
            if not node.is_locked():
                node.prior = max(CROMWELL_EPS, node.prior * (1.0 - penalty))
        else:
            weakened_belief = max(CROMWELL_EPS, 0.5 * (1.0 - penalty))
            graph.add_node(
                statement=conclusion_statement,
                belief=weakened_belief,
                prior=weakened_belief,
                domain=output.get("domain"),
                provenance=provenance,
            )
        return None

    if outcome == "inconclusive":
        # No evidence either way; do not modify the graph at all.
        return None

    premise_ids = []
    try:
        unverified_prior = float(getattr(CONFIG, "unverified_claim_prior", DEFAULT_UNVERIFIED_CLAIM_PRIOR))
    except Exception:
        unverified_prior = DEFAULT_UNVERIFIED_CLAIM_PRIOR
    for p in output.get("premises", []):
        if p.get("id") and p["id"] in graph.nodes:
            premise_ids.append(p["id"])
        else:
            initial_belief = 1.0 if module == Module.LEAN else unverified_prior
            node = graph.add_node(
                statement=p["statement"],
                belief=initial_belief,
                prior=initial_belief,
                domain=output.get("domain"),
                provenance=provenance,
            )
            premise_ids.append(node.id)

    conclusion = output["conclusion"]
    conclusion_statement = (
        conclusion if isinstance(conclusion, str) else conclusion.get("statement", "")
    )
    formal = None if isinstance(conclusion, str) else conclusion.get("formal_statement")

    # Determine conclusion node, preferring an existing node when possible.
    # Priority order:
    # 1. If target_node_id is supplied, use that node directly as the conclusion.
    #    The caller (MCTS main loop) passes target_node_id only when the action
    #    was explicitly targeting the overall MCTS goal, so the conclusion of that
    #    action IS the goal node — regardless of how the LLM chose to phrase it.
    #    This is much more reliable than text-prefix matching because LLM output
    #    wording varies every run (e.g. "for n=11 (10 runners)" vs "holds for n=11").
    # 2. Exact / canonicalised text match against existing graph nodes.
    # 3. Create a new node.
    conclusion_id: str | None = None

    if target_node_id and target_node_id in graph.nodes:
        conclusion_id = target_node_id

    if conclusion_id is None:
        existing = graph.find_node_ids_by_statement(conclusion_statement)
        if existing:
            conclusion_id = existing[0]

    if conclusion_id is None:
        conclusion_node = graph.add_node(
            statement=conclusion_statement,
            formal_statement=formal,
            domain=output.get("domain"),
            provenance=provenance,
            prior=unverified_prior,
            belief=unverified_prior,
        )
        conclusion_id = conclusion_node.id

    premise_ids = [pid for pid in premise_ids if pid != conclusion_id]

    confidence = output.get("confidence", DEFAULT_CONFIDENCE[module])
    steps = output.get("steps", [])

    edge = graph.add_hyperedge(
        premise_ids=premise_ids,
        conclusion_id=conclusion_id,
        module=module,
        steps=steps,
        confidence=confidence,
    )

    if module == Module.LEAN and _all_premises_proven_or_axiom(graph, premise_ids):
        _set_node_proven(graph, conclusion_id)

    # Successful experiment = observational evidence for the conclusion.
    # Raise conclusion prior so BP can propagate support through this edge.
    # This mirrors ingest_verified_claim's experiment path (prior → 0.85).
    if module == Module.EXPERIMENT:
        conclusion_node = graph.nodes[conclusion_id]
        if not conclusion_node.is_locked():
            exp_prior = min(EXPERIMENT_PRIOR_CAP, max(conclusion_node.prior, confidence))
            conclusion_node.prior = exp_prior

    return edge


def ingest_verified_claim(
    graph: HyperGraph,
    *,
    claim_text: str,
    verification_source: str,
    verdict: str,
    domain: str | None = None,
    source_memo_id: str | None = None,
    claim_id: str | None = None,
    target_node_id: str | None = None,
    parent_edge_id: str | None = None,
) -> str:
    """Ingest one verified/refuted/pending claim into the graph and return node_id.

    Args:
        claim_text: Natural-language text of the claim.
        verification_source: Backend that produced the verdict (e.g. "experiment").
        verdict: "verified", "refuted", or "inconclusive".
        domain: Optional domain label for the node.
        source_memo_id: ID of the originating ResearchMemo.
        claim_id: Internal Claim model ID (for reference, not used for node lookup).
        target_node_id: When supplied and present in the graph, use this node
            directly instead of searching by statement.  This allows callers
            that already hold a precise node ID (e.g. via bridge_node_map) to
            avoid text-matching entirely.
        parent_edge_id: When supplied and a NEW node is created (no existing
            match), create a hyperedge connecting the new node as a premise to
            the conclusion of the referenced edge.  This prevents newly created
            claim nodes from becoming disconnected orphans.
    """
    claim_text = (claim_text or "").strip()
    if not claim_text:
        raise ValueError("claim_text must not be empty")

    is_new_node = False

    # Fastest path: caller already knows the exact node ID.
    if target_node_id and target_node_id in graph.nodes:
        node_id = target_node_id
        node = graph.nodes[node_id]
    else:
        existing = graph.find_node_ids_by_statement(claim_text)
        if existing:
            node_id = existing[0]
            node = graph.nodes[node_id]
        else:
            is_new_node = True
            try:
                unverified_prior = float(getattr(CONFIG, "unverified_claim_prior", DEFAULT_UNVERIFIED_CLAIM_PRIOR))
            except Exception:
                unverified_prior = DEFAULT_UNVERIFIED_CLAIM_PRIOR
            node = graph.add_node(
                statement=claim_text,
                belief=unverified_prior,
                prior=unverified_prior,
                domain=domain,
                provenance=f"verification:{verification_source}",
                verification_source=verification_source,
                memo_ref=source_memo_id,
            )
            node_id = node.id

    node.verification_source = verification_source
    if source_memo_id:
        node.memo_ref = source_memo_id
    if domain and not node.domain:
        node.domain = domain

    verdict_normalized = verdict.strip().lower()
    if verdict_normalized == "verified":
        if verification_source == "lean":
            node.state = "proven"
            node.prior = 1.0
            node.belief = 1.0
        elif verification_source == "experiment":
            node.prior = max(node.prior, EXPERIMENT_PRIOR_CAP)
            if node.state == "refuted":
                node.state = "unverified"
        else:
            node.prior = max(node.prior, VERIFIED_PRIOR_FLOOR)
            if node.state == "refuted":
                node.state = "unverified"
    elif verdict_normalized == "refuted":
        if verification_source == "lean":
            # Only formal Lean verification can hard-refute a node.
            node.state = "refuted"
            node.prior = 0.0
            node.belief = 0.0
        else:
            # All non-formal sources (experiment, llm_judge, heuristic, etc.)
            # apply a sharp belief penalty but let BP determine the posterior.
            if not node.is_locked():
                node.prior = max(CROMWELL_EPS, node.prior * REFUTATION_PRIOR_MULTIPLIER)
    else:
        try:
            unverified_prior = float(getattr(CONFIG, "unverified_claim_prior", DEFAULT_UNVERIFIED_CLAIM_PRIOR))
        except Exception:
            unverified_prior = DEFAULT_UNVERIFIED_CLAIM_PRIOR
        if not node.is_locked():
            node.prior = max(unverified_prior, min(node.prior, INCONCLUSIVE_PRIOR_CAP))

    # When a new node was created and a parent edge is provided, create a
    # hyperedge connecting this new node as a premise to the conclusion of
    # the parent edge.  This ensures the new claim node is not a graph orphan
    # and can participate in belief propagation toward the ultimate target.
    if is_new_node and parent_edge_id and parent_edge_id in graph.edges:
        parent_edge = graph.edges[parent_edge_id]
        module = Module.EXPERIMENT if verification_source == "experiment" else Module.LEAN
        conclusion_id = parent_edge.conclusion_id
        if conclusion_id in graph.nodes:
            # Idempotency: skip if an identical edge already exists.
            already = any(
                set(e.premise_ids) == {node_id} and e.conclusion_id == conclusion_id
                for e in graph.edges.values()
            )
            if not already and node_id != conclusion_id:
                graph.add_hyperedge(
                    premise_ids=[node_id],
                    conclusion_id=conclusion_id,
                    module=module,
                    steps=[f"Claim verification ({verification_source}): {verdict_normalized}"],
                    confidence=node.belief,
                )

    return node_id


def estimate_created_node_ids(before_node_ids: set[str], graph: HyperGraph) -> list[str]:
    """Return node IDs created since the caller captured ``before_node_ids``."""
    return [nid for nid in graph.nodes.keys() if nid not in before_node_ids]
