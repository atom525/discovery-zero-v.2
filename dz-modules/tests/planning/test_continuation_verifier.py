from dz_verify.continuation_verifier import ContinuationVerifier


def test_cluster_sizes_groups_similar_continuations():
    verifier = ContinuationVerifier()
    sizes = verifier._cluster_sizes(
        [
            "The step follows from compactness and averaging.",
            "This follows from averaging and compactness.",
            "A random walk argument suggests a different route.",
        ]
    )
    assert sizes[0] >= 2


def test_calibrated_belief_increases_with_consistency():
    verifier = ContinuationVerifier()
    low = verifier.calibrated_belief(prior=0.5, consistency=0.2)
    high = verifier.calibrated_belief(prior=0.5, consistency=0.8)
    assert high > low
