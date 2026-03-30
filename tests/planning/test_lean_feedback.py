from discovery_zero.graph.memo import Claim, ClaimType
from discovery_zero.graph.models import HyperGraph
from discovery_zero.planning.lean_feedback import LeanFeedbackParser, StructuralClaimRouter


def test_parse_lean_error_unknown_identifier():
    parser = LeanFeedbackParser()
    gap = parser.parse_lean_error("Discovery/Proofs.lean:7:5: error: unknown identifier 'fooBar'")
    assert gap.gap_type == "unknown_identifier"
    assert gap.identifier == "fooBar"


def test_gap_to_feedback_unsolved_goal():
    parser = LeanFeedbackParser()
    claim = Claim(claim_text="if A then B", claim_type=ClaimType.STRUCTURAL, source_memo_id="m")
    gap = parser.parse_lean_error("error: unsolved goals: A -> B")
    feedback = parser.gap_to_feedback(gap, claim)
    assert "Unsolved goal" in feedback


def test_router_assess_complexity_and_route():
    router = StructuralClaimRouter(max_decompose_depth=2, structural_complexity_threshold=1)
    simple_claim = Claim(claim_text="if A then B", claim_type=ClaimType.STRUCTURAL, source_memo_id="m")
    complex_claim = Claim(
        claim_text="For all x and for all y, if A x and B y then there exists z such that C z and D x y z",
        claim_type=ClaimType.STRUCTURAL,
        source_memo_id="m",
    )
    assert router.assess_complexity(simple_claim) == "simple"
    assert router.assess_complexity(complex_claim) == "complex"
    plan = router.route_structural_claim(complex_claim, depth=0)
    assert plan.mode == "decompose"


def test_suggest_subgoals_uses_graph_fallback():
    parser = LeanFeedbackParser()
    graph = HyperGraph()
    graph.add_node("lemma one", belief=0.4, prior=0.4)
    graph.add_node("lemma two", belief=0.3, prior=0.3)
    gap = parser.parse_lean_error("opaque lean failure")
    suggestions = parser.suggest_subgoals(gap, graph)
    assert suggestions

