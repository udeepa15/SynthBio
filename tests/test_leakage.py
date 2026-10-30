"""
The leakage guard.

The critic is trained on the specialists' predictions. If a specialist's
training rows overlap the critic's training or evaluation rows, the critic
learns that the specialist is more accurate than it is, and the agreement score
becomes meaningless in exactly the cases it exists for. Such a system would
report high confidence and high agreement and be wrong, which is strictly worse
than the present model, because the present model claims nothing.

These tests cannot check an overlap that does not exist yet, because no critic
has been trained and the clustering that would reserve a critic pool is blocked.
So they check the two things that stop it happening by accident: that no critic
model can be in use while the pool is unreserved, and that the assertions
themselves become real the moment a cluster assignment is committed.
"""

import pytest

from src.agents.base import EXPRESSION, AgentRequest
from src.agents.critic import combine
from src.agents.expression import ExpressionAgent
from src.agents.variant import VariantAgent
from src.datasets import clustering

SEQ = "ATGAAACGCATTAGCCTGACCCTGCTGAGCGCGGCGATTCTGCTGACCGCGCTGAGCGCGTAA"


def test_no_critic_model_is_in_use_while_the_critic_pool_is_unreserved():
    assignment = clustering.load_assignment()
    verdict = combine([ExpressionAgent(model=None).opine(AgentRequest(sequence=SEQ))])
    if assignment is None or not assignment.critic_pool:
        assert verdict.critic_model is None, (
            "A critic model is in use but no critic pool has been reserved, so every agent's "
            "accuracy as the critic sees it is contaminated by its own training rows."
        )


def test_a_committed_cluster_assignment_keeps_the_pool_disjoint_from_training():
    assignment = clustering.load_assignment()
    if assignment is None:
        pytest.skip(f"No cluster assignment committed at {clustering.CLUSTER_REPORT}.")
    pool = set(assignment.critic_pool)
    training_clusters = {assignment.clusters[row] for row in assignment.training_ids()}
    assert not (training_clusters & pool), (
        "A cluster appears in both a specialist's training set and the critic pool."
    )
    assert set(assignment.clusters.values()) >= pool, "The critic pool names clusters that do not exist."


def test_the_critic_pool_is_reserved_by_cluster_and_never_by_row():
    clusters = {f"row{i}": f"cluster{i // 5}" for i in range(50)}
    pool = clustering.reserve_critic_pool(clusters, fraction=0.2, seed=1)
    assert len(pool) == 2
    # Every row of a reserved cluster is reserved with it. A row level split is
    # the mistake this function exists to make impossible.
    for row, cluster in clusters.items():
        if cluster in pool:
            assert row not in set(
                r for r, c in clusters.items() if c not in pool
            )


def test_the_critic_pool_is_drawn_once_and_reproducibly():
    clusters = {f"row{i}": f"cluster{i // 4}" for i in range(80)}
    first = clustering.reserve_critic_pool(clusters, seed=clustering.CRITIC_POOL_SEED)
    second = clustering.reserve_critic_pool(clusters, seed=clustering.CRITIC_POOL_SEED)
    assert first == second, (
        "A later run drawing a different pool would invalidate every gate measured against the old "
        "one without anything failing."
    )


def test_clustering_reports_why_it_is_blocked_rather_than_failing_quietly():
    """
    A blocked clustering must name its own obstacle.

    The two obstacles are independent and are reported separately, because the
    remedies differ: a missing sequence table is recovered with
    `scripts/fetch_datasets.py --source boel`, while a missing MMseqs2 is an
    install. Only the first one stops the sequences being loaded at all.
    """
    status = clustering.clustering_status()
    if status is None:
        pytest.skip("Identity clustering is available in this working tree.")
    assert "blocked" in status.lower()

    if clustering.boel_sequence_path() is None:
        assert "nothing to cluster" in status
        with pytest.raises(clustering.ClusteringUnavailable):
            clustering.load_sequences()
    else:
        records = clustering.load_sequences()
        assert records, "a sequence table is on disk but no sequence came back from it"
        assert all(protein for _, protein in records)

    if clustering.mmseqs_path() is None:
        assert "MMseqs2 is not installed" in status
        with pytest.raises(clustering.ClusteringUnavailable):
            clustering.cluster_with_mmseqs([("a", "MKAILV"), ("b", "MKAILW")])


def test_organism_grouping_is_available_but_is_not_treated_as_identity_clustering():
    groups = clustering.organism_groups(["AaR17", "AaR21", "BsX4", "EcN100"])
    assert groups[0] == groups[1], "two NESG ids from one organism must group together"
    assert len(set(groups)) == 3
    # Organism grouping must never be recorded as the clustered split, because
    # paralogues inside one organism still leak.
    assignment = clustering.load_assignment()
    if assignment is not None:
        assert "mmseqs" in assignment.method


def test_welch_is_never_a_training_source_for_the_variant_agent():
    from src.datasets.variant_sets import load_variant_sets

    for variant_set in load_variant_sets():
        assert "welch" not in variant_set.name.lower(), (
            "Welch 2009 is the held out external gate for Agent C and must never be trained on."
        )


def test_an_ungated_variant_artifact_does_not_speak():
    agent = VariantAgent(ranker=object(), require_gate=True)
    agent.metadata = {"gate": {"status": "ungated", "summary": "never measured"}}
    opinion = agent.opine(AgentRequest(sequence=SEQ, comparison=(SEQ[:-3] + "TAG",)))
    assert opinion.applies is False
    assert "Gate C" in opinion.abstain_reason
