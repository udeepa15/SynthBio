"""
Agent C, the within gene codon variant ranker.

Two things are being tested. The feature allowlist, because the one way this
agent can silently become the between gene model it replaces is by being shown
a feature that cannot change under a synonymous substitution. And the trainer
and the gate, on a synthetic library, because the real sources are not in the
working tree and the point of the trainer is that it runs the day they arrive.
"""

import random

import numpy as np
import pandas as pd
import pytest

from src.agents.base import AgentRequest
from src.agents.variant import (
    FORBIDDEN_FEATURES,
    VARIANT_FEATURES,
    GC3Baseline,
    VariantAgent,
    constant_within_group,
    variant_feature_frame,
)
from src.datasets.variant_sets import MIN_GROUP_SIZE, canonicalise
from src.features import calculate_gc3, extract_features
from src.redesign import _synonym_map, split_codons, translate_codons

PROTEIN_DNA = (
    "ATGAAACGCATTAGCCTGACCCTGCTGAGCGCGGCGATTCTGCTGACCGCGCTGAGCGCG"
    "GAACTGAGCCTGAAAGCGCTGGATGAAATTCGCAAAGCGCTGGAAGGCGCGAAAGATTAA"
)


def synonymous_variants(dna: str, n: int, seed: int = 0):
    """Synonymous variants of one coding sequence, protein held constant."""
    rng = random.Random(seed)
    synonyms = _synonym_map()
    codons = split_codons(dna)
    protein = translate_codons(codons)
    out = []
    for _ in range(n):
        candidate = list(codons)
        for position in range(1, len(candidate) - 1):
            options = synonyms.get(candidate[position], [])
            if options and rng.random() < 0.5:
                candidate[position] = rng.choice(options)
        variant = "".join(candidate)
        assert translate_codons(split_codons(variant)) == protein
        out.append(variant)
    return out


# --- the allowlist ----------------------------------------------------------


def test_the_allowlist_excludes_everything_a_synonymous_swap_cannot_move():
    for name in ("gravy_score", "pI", "length_codons"):
        assert name not in VARIANT_FEATURES
        assert name in FORBIDDEN_FEATURES


def test_the_allowlist_is_exactly_the_matrix_the_agent_sees():
    frame = variant_feature_frame([extract_features(PROTEIN_DNA)])
    assert list(frame.columns) == list(VARIANT_FEATURES)
    assert not set(frame.columns) & set(FORBIDDEN_FEATURES)


def test_the_excluded_features_really_are_constant_across_synonymous_variants():
    variants = synonymous_variants(PROTEIN_DNA, 6, seed=3)
    features = [extract_features(v) for v in variants]
    for name in ("gravy_score", "pI", "length_codons"):
        values = {round(float(f[name]), 6) for f in features}
        assert len(values) == 1, f"{name} moved under a synonymous change, which should be impossible"


def test_the_allowed_features_do_vary_across_synonymous_variants():
    variants = synonymous_variants(PROTEIN_DNA, 8, seed=4)
    frame = variant_feature_frame([extract_features(v) for v in variants])
    constant = constant_within_group(frame)
    assert "gc3_content" not in constant
    assert len(constant) < len(VARIANT_FEATURES), "no allowed feature varied, so the group is wrong"


def test_gc3_baseline_orders_by_gc3_and_nothing_else():
    variants = synonymous_variants(PROTEIN_DNA, 5, seed=5)
    features = [extract_features(v) for v in variants]
    order = GC3Baseline.order(features)
    gc3 = [calculate_gc3(variants[i]) for i in order]
    assert gc3 == sorted(gc3, reverse=True)


# --- the agent's abstention contract ----------------------------------------


def test_the_agent_abstains_without_an_artifact_even_with_a_group_to_rank():
    variants = synonymous_variants(PROTEIN_DNA, 3, seed=6)
    opinion = VariantAgent(load=False).opine(
        AgentRequest(sequence=variants[0], comparison=tuple(variants[1:]))
    )
    assert opinion.applies is False
    assert "no variant ranker artifact" in opinion.abstain_reason.lower()


def test_rank_refuses_to_run_without_an_artifact_rather_than_returning_an_order():
    with pytest.raises(ValueError):
        VariantAgent(load=False).rank(synonymous_variants(PROTEIN_DNA, 3, seed=7))


# --- the trainer, on a synthetic library ------------------------------------


@pytest.fixture(scope="module")
def synthetic_library():
    """
    Three query groups of synonymous variants, with a readout built from GC3.

    The readout is deliberately a monotonic function of a feature the agent is
    allowed to see, so a working trainer must recover the ordering. That makes
    this a test of the machinery and not evidence about biology, which is what
    Gate C on Welch is for.
    """
    rows = []
    for group_index, seed in enumerate((11, 12, 13)):
        variants = synonymous_variants(PROTEIN_DNA, 12, seed=seed)
        for variant_index, dna in enumerate(variants):
            rows.append({
                "group_id": f"group{group_index}",
                "variant_id": f"g{group_index}_v{variant_index}",
                "dna": dna,
                "readout": calculate_gc3(dna) + group_index * 100.0,
            })
    frame = canonicalise(pd.DataFrame(rows), "synthetic", "synthetic_gc3")
    return frame.reset_index(drop=True)


def test_canonicalise_drops_unusable_rows_and_small_groups(synthetic_library):
    assert len(synthetic_library) > 0
    assert synthetic_library["group_id"].nunique() == 3
    sizes = synthetic_library.groupby("group_id").size()
    assert (sizes >= MIN_GROUP_SIZE).all()
    bad = pd.DataFrame([
        {"group_id": "g", "variant_id": "1", "dna": "ATGNNN", "readout": 1.0},
        {"group_id": "g", "variant_id": "2", "dna": "ATGAA", "readout": 2.0},
    ])
    assert canonicalise(bad, "synthetic", "synthetic").empty


def test_the_trainer_recovers_a_within_group_ordering(synthetic_library):
    from scripts.train_agent_c import fit_ranker

    feature_dicts = [extract_features(dna) for dna in synthetic_library["dna"]]
    ranker, _ = fit_ranker(synthetic_library, feature_dicts)

    agent = VariantAgent(ranker=ranker, require_gate=False)
    from scipy.stats import spearmanr

    for group_id, index in synthetic_library.groupby("group_id").groups.items():
        positions = synthetic_library.index.get_indexer(index)
        ranking = agent.rank(
            [synthetic_library.loc[i, "dna"] for i in index],
            [feature_dicts[p] for p in positions],
        )
        truth = [synthetic_library.loc[i, "readout"] for i in index]
        rho = spearmanr(list(ranking.scores), truth).statistic
        assert rho > 0.5, f"{group_id}: the ranker did not recover its own training ordering"


def test_the_trainer_refuses_a_grouping_that_is_not_grouping_synonymous_variants():
    """
    One row per group means every allowed feature is constant inside every
    group, which is how a mislabelled group column would present itself.
    """
    from scripts.train_agent_c import fit_ranker

    variants = synonymous_variants(PROTEIN_DNA, 4, seed=21)
    frame = pd.DataFrame([
        {"group_id": f"g{i}", "variant_id": str(i), "dna": dna, "readout": float(i)}
        for i, dna in enumerate(variants)
    ])
    feature_dicts = [extract_features(dna) for dna in frame["dna"]]
    with pytest.raises(ValueError, match="constant inside every query group"):
        fit_ranker(frame, feature_dicts)


def test_relevance_is_binned_inside_each_group_and_not_across_groups(synthetic_library):
    from scripts.train_agent_c import within_group_relevance

    relevance = within_group_relevance(synthetic_library)
    assert len(relevance) == len(synthetic_library)
    # Group 2's readouts are 200 higher than group 0's, so a pooled binning
    # would give group 0 nothing but the bottom level.
    for _, index in synthetic_library.groupby("group_id").groups.items():
        positions = synthetic_library.index.get_indexer(index)
        assert len(set(relevance[positions])) > 1


def test_the_gate_requires_both_welch_series_to_be_positive():
    from src.gating import GateCOutcome, WelchScoring, evaluate_gate_c

    def scoring(name, overall, polymerase, scfv):
        return WelchScoring(
            name=name,
            overall=(overall, 0.0, 62),
            per_series={"polymerase": (polymerase, 0.0, 30), "scFv": (scfv, 0.0, 32)},
        )

    gc3 = scoring("GC3 alone", 0.452, 0.643, 0.394)
    gc_percent = scoring("GC per cent alone", 0.719, 0.531, 0.380)

    # The served model's actual figures: right on one series, backwards on the
    # other. This is the case Gate C exists to refuse.
    served = scoring("served", 0.524, -0.408, 0.469)
    assert evaluate_gate_c(served, gc3, gc_percent).passed is False

    # Positive on both but below GC3 on one.
    weak = scoring("weak", 0.5, 0.2, 0.45)
    assert evaluate_gate_c(weak, gc3, gc_percent).passed is False

    strong = scoring("strong", 0.75, 0.70, 0.45)
    outcome = evaluate_gate_c(strong, gc3, gc_percent)
    assert outcome.passed is True
    assert outcome.baselines_beaten

    assert evaluate_gate_c(None, gc3, gc_percent).passed is False


def test_a_failed_gate_keeps_the_agent_quiet_even_with_an_artifact(synthetic_library):
    from scripts.train_agent_c import fit_ranker

    feature_dicts = [extract_features(dna) for dna in synthetic_library["dna"]]
    ranker, _ = fit_ranker(synthetic_library, feature_dicts)
    agent = VariantAgent(ranker=ranker, require_gate=True)
    agent.metadata = {"gate": {"status": "failed", "summary": "Gate C failed on the polymerase series."}}
    variants = synthetic_library["dna"].tolist()[:3]
    opinion = agent.opine(AgentRequest(sequence=variants[0], comparison=tuple(variants[1:])))
    assert opinion.applies is False
    assert opinion.provenance.gate.passed is False
