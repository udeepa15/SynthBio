"""
Tests for the synonymous redesign engine.

The invariant worth most here is that the protein never changes, so it is
tested on the mutation operator directly as well as through a full search: a
redesign that silently alters the product would be worse than no redesign at
all. The conserved-region lock is tested the same way, by checking the exact
codons rather than only the translation, because a synonymous change inside a
catalytic site still changes the DNA the user asked to protect.
"""

import numpy as np
import pytest

from src.redesign import (
    CONSTRAINED,
    CONTRADICTED,
    CORROBORATED,
    PI_AGGREGATION_WINDOW,
    corroborate,
    generate_variants,
    mutable_positions,
    optimize_sequence,
    parse_conserved_regions,
    recommend_fusion_tags,
    split_codons,
    translate_codons,
)

# A construct with a run of synonymous choice in every family, a start codon
# and a terminal stop, so the lock and the operator both have something to act
# on. 40 codons.
SEQUENCE = (
    "ATG" "AAA" "CGC" "ATT" "AGC" "CTG" "ACC" "CTG" "CTG" "AGC"
    "GCG" "GCG" "ATT" "CTG" "CTG" "ACC" "GCG" "CTG" "AGC" "GCG"
    "GAA" "GAT" "AAA" "CAT" "CAG" "AAT" "TTT" "TAT" "TGT" "GGT"
    "GTT" "CCG" "ACG" "TCG" "CGT" "AGA" "GGA" "ATG" "TGG" "TAA"
)


def test_split_and_translate_round_trip():
    codons = split_codons(SEQUENCE)
    assert len(codons) == 40
    assert codons[0] == "ATG"
    assert translate_codons(codons).endswith("*")


def test_trailing_partial_codon_is_dropped():
    assert len(split_codons(SEQUENCE + "AT")) == 40


def test_start_and_stop_are_never_mutable():
    positions = mutable_positions(split_codons(SEQUENCE), set())
    assert 0 not in positions
    assert 39 not in positions


def test_single_codon_amino_acids_are_not_mutable():
    codons = split_codons(SEQUENCE)
    positions = mutable_positions(codons, set())
    # Methionine at index 37 and tryptophan at 38 have no synonymous alternative.
    assert codons[37] == "ATG" and 37 not in positions
    assert codons[38] == "TGG" and 38 not in positions


def test_variants_preserve_the_protein():
    protein = translate_codons(split_codons(SEQUENCE))
    for variant in generate_variants(SEQUENCE, n_variants=50, seed=7):
        assert translate_codons(split_codons(variant)) == protein


def test_variants_actually_differ_from_the_original():
    variants = generate_variants(SEQUENCE, n_variants=20, n_swaps=3, seed=11)
    assert any(v != SEQUENCE for v in variants)


def test_conserved_region_codons_are_untouched():
    original = split_codons(SEQUENCE)
    for variant in generate_variants(SEQUENCE, n_variants=40, conserved_text="10-20", seed=3):
        codons = split_codons(variant)
        for index in range(9, 20):  # amino acids 10 to 20, zero-based
            assert codons[index] == original[index]


def test_conserved_region_parsing_accepts_ranges_and_singles():
    spec = parse_conserved_regions("5-7, 12 ; 3", n_codons=40)
    assert spec.locked == {2, 4, 5, 6, 11}
    assert spec.regions == [(3, 3), (5, 7), (12, 12)]
    assert not spec.problems


def test_conserved_region_parsing_reports_bad_input_without_discarding_the_rest():
    spec = parse_conserved_regions("5-7, banana, 900, 38-900", n_codons=40)
    assert {4, 5, 6}.issubset(spec.locked)
    assert len(spec.problems) == 3
    # The clipped range still applies.
    assert (38, 40) in spec.regions


def test_reversed_range_is_accepted():
    assert parse_conserved_regions("20-10", n_codons=40).regions == [(10, 20)]


def test_empty_conserved_text_locks_nothing():
    spec = parse_conserved_regions("   ", n_codons=40)
    assert spec.locked == set()
    assert spec.described == "none"


def test_fully_locked_sequence_is_refused():
    with pytest.raises(ValueError, match="No codon"):
        optimize_sequence(SEQUENCE, model=object(), conserved_text=f"1-40")


def test_redesign_without_a_model_is_refused():
    with pytest.raises(ValueError, match="heuristic mode"):
        optimize_sequence(SEQUENCE, model=None)


def test_too_short_sequence_is_refused():
    with pytest.raises(ValueError, match="at least two codons"):
        optimize_sequence("ATG", model=object())


class _FakeModel:
    """
    A stand-in scorer that rewards GC at the wobble position.

    The search only needs `predict_proba` and `feature_names_in_`, so a fake
    keeps these tests independent of the saved artifact and of whether
    ViennaRNA is installed, while still giving the hill climb a real gradient
    to follow.
    """

    feature_names_in_ = np.array([
        "GC_percent", "GC3", "CAI", "rare_codon_fraction", "rare_codons_head30",
        "head_folding_48", "genewide_folding_avg", "gravy_score", "pI",
        "length_codons", "folding_energy",
    ])
    classes_ = np.array([0, 1, 2, 3, 4, 5])

    def predict_proba(self, frame):
        out = []
        for gc3 in frame["GC3"]:
            # Mass on the top tier grows with GC3, the rest on the bottom tier.
            top = min(max((float(gc3) - 30.0) / 60.0, 0.0), 1.0)
            row = np.zeros(6)
            row[5] = top
            row[0] = 1.0 - top
            out.append(row)
        return np.array(out)


def test_search_improves_the_objective_and_keeps_the_protein():
    protein = translate_codons(split_codons(SEQUENCE))
    result = optimize_sequence(
        SEQUENCE, model=_FakeModel(), rounds=6, candidates_per_round=12, seed=5,
    )
    assert result.best.expected_tier >= result.original.expected_tier
    assert translate_codons(split_codons(result.best.sequence)) == protein
    assert len(result.best.sequence) == len(SEQUENCE)
    assert result.candidates_scored == 1 + 6 * 12
    assert result.protein == protein


def test_search_respects_the_lock_end_to_end():
    original = split_codons(SEQUENCE)
    result = optimize_sequence(
        SEQUENCE, model=_FakeModel(), conserved_text="10-20",
        rounds=4, candidates_per_round=10, seed=9,
    )
    best = split_codons(result.best.sequence)
    for index in range(9, 20):
        assert best[index] == original[index]
    assert len(result.conserved.locked) == 11


def test_changed_codon_table_lists_only_real_changes():
    result = optimize_sequence(
        SEQUENCE, model=_FakeModel(), rounds=4, candidates_per_round=10, seed=2,
    )
    table = result.changed_codon_table()
    assert len(table) == result.codons_changed
    original = split_codons(result.original.sequence)
    for row in table.to_dict("records"):
        index = row["Amino acid position"] - 1
        assert row["Original codon"] == original[index]
        assert row["Redesigned codon"] != original[index]


def test_feature_comparison_omits_the_protein_level_features():
    result = optimize_sequence(
        SEQUENCE, model=_FakeModel(), rounds=2, candidates_per_round=6, seed=4,
    )
    labels = list(result.feature_comparison()["Feature"])
    assert not any("GRAVY" in label or "soelectric" in label for label in labels)


def test_gravy_and_pi_are_unchanged_by_a_redesign():
    result = optimize_sequence(
        SEQUENCE, model=_FakeModel(), rounds=4, candidates_per_round=10, seed=6,
    )
    for key in ("gravy_score", "pI"):
        before, after = result.original.features[key], result.best.features[key]
        if not (np.isnan(before) and np.isnan(after)):
            assert before == after


def test_hydrophobic_product_gets_a_tag_recommendation():
    tags = recommend_fusion_tags({"gravy_score": 0.9, "pI": 5.0})
    assert len(tags) == 1
    assert "GRAVY" in tags[0].driver
    assert "MBP" in tags[0].recommendation


def test_neutral_pi_gets_a_tag_recommendation():
    low, high = PI_AGGREGATION_WINDOW
    tags = recommend_fusion_tags({"gravy_score": 0.0, "pI": (low + high) / 2})
    assert len(tags) == 1
    assert "soelectric" in tags[0].driver


def test_both_failure_modes_give_both_recommendations():
    assert len(recommend_fusion_tags({"gravy_score": 0.9, "pI": 7.5})) == 2


def test_no_recommendation_when_neither_threshold_is_crossed():
    assert recommend_fusion_tags({"gravy_score": 0.0, "pI": 5.0}) == []


def test_missing_protein_features_give_no_recommendation():
    assert recommend_fusion_tags({"gravy_score": float("nan"), "pI": float("nan")}) == []


class _Stub:
    """Minimal stand-in for a scored result, for the corroboration check."""

    def __init__(self, improved, gc3_before, gc3_after):
        self.improved = improved
        self.original = type("S", (), {"features": {
            "gc3_content": gc3_before, "head_folding_48": -10.0, "codon_adaptation_index": 0.5,
        }})()
        self.best = type("S", (), {"features": {
            "gc3_content": gc3_after, "head_folding_48": -8.0, "codon_adaptation_index": 0.55,
        }})()


def test_rising_gc3_corroborates_a_gain():
    assert corroborate(_Stub(True, 50.0, 55.0)).verdict == CORROBORATED


def test_falling_gc3_contradicts_a_gain():
    result = corroborate(_Stub(True, 55.0, 50.0))
    assert result.verdict == CONTRADICTED
    assert "disagree" in result.summary


def test_no_gain_is_inconclusive():
    assert corroborate(_Stub(False, 50.0, 55.0)).verdict == "inconclusive"


class _AntiGc3Model(_FakeModel):
    """Rewards LOW GC3, the model's documented failure on synonymous variants."""

    def predict_proba(self, frame):
        return super().predict_proba(frame)[:, ::-1]


def _gc3(sequence):
    codons = split_codons(sequence)
    return sum(1 for c in codons if c[2] in "GC") / len(codons)


def test_hold_gc3_never_lowers_gc3_in_any_variant():
    original = _gc3(SEQUENCE)
    for variant in generate_variants(SEQUENCE, n_variants=60, n_swaps=6, seed=3, hold_gc3=True):
        assert _gc3(variant) >= original - 1e-9
        assert translate_codons(split_codons(variant)) == translate_codons(split_codons(SEQUENCE))


def test_hold_gc3_excludes_codons_that_could_only_lose_gc3():
    codons = split_codons(SEQUENCE)
    free = set(mutable_positions(codons, set()))
    held = set(mutable_positions(codons, set(), hold_gc3=True))
    assert held <= free
    # Lysine AAG has only AAA as an alternative, which would lower GC3.
    assert "AAG" not in [codons[i] for i in held]


def test_hold_gc3_resists_a_model_that_rewards_lower_gc3():
    free = optimize_sequence(SEQUENCE, model=_AntiGc3Model(), rounds=6, candidates_per_round=16, seed=1)
    held = optimize_sequence(
        SEQUENCE, model=_AntiGc3Model(), rounds=6, candidates_per_round=16, seed=1, hold_gc3=True,
    )
    assert _gc3(free.best.sequence) < _gc3(SEQUENCE)
    assert _gc3(held.best.sequence) >= _gc3(SEQUENCE) - 1e-9
    assert held.hold_gc3 is True


def test_hold_gc3_is_reported_as_constrained_not_corroborated():
    result = optimize_sequence(
        SEQUENCE, model=_FakeModel(), rounds=4, candidates_per_round=10, seed=2, hold_gc3=True,
    )
    verdict = result.corroboration()
    if result.improved:
        assert verdict.verdict == CONSTRAINED
        assert verdict.constrained is True
        assert "not independent" in verdict.summary


def test_region_limit_leaves_the_rest_of_the_gene_untouched():
    original = split_codons(SEQUENCE)
    result = optimize_sequence(
        SEQUENCE, model=_FakeModel(), rounds=4, candidates_per_round=10, seed=8, region_codons=12,
    )
    best = split_codons(result.best.sequence)
    assert best[12:] == original[12:]
    assert result.region_codons == 12


def test_region_limit_applies_to_generated_variants_too():
    original = split_codons(SEQUENCE)
    for variant in generate_variants(SEQUENCE, n_variants=30, n_swaps=4, seed=4, region_codons=10):
        assert split_codons(variant)[10:] == original[10:]


def test_negative_region_is_refused():
    with pytest.raises(ValueError):
        optimize_sequence(SEQUENCE, model=_FakeModel(), region_codons=-1)


# --- M3: which agent the search ranks on ------------------------------------
#
# The tier model separates one gene from another and ranks the Welch polymerase
# series backwards, so it is the wrong objective for a within gene search.
# Agent C is the right one, and these tests fix the switch between them:
# the agent is used when it holds a gated artifact, and the search falls back to
# the tier model, with the GC3 corroboration check as its guard, when it does
# not.


class _FakeRanker:
    """A stand-in Agent C that prefers low GC3, the opposite of _FakeModel."""

    class _Ranking:
        def __init__(self, scores):
            self.scores = tuple(scores)
            self.order = tuple(sorted(range(len(scores)), key=lambda i: -scores[i]))

    name = "variant"
    require_gate = True
    ranker = object()

    def __init__(self, passed=True):
        self._passed = passed
        self.calls = 0

    def gate(self):
        from src.agents.base import GATE_FAILED, GATE_PASSED, GateResult

        return GateResult(
            status=GATE_PASSED if self._passed else GATE_FAILED,
            summary="Gate C stub.",
        )

    def rank(self, sequences, feature_dicts=None):
        from src.features import calculate_gc3

        self.calls += 1
        return self._Ranking([-calculate_gc3(seq) for seq in sequences])


def test_the_search_ranks_on_agent_c_when_it_holds_a_passed_gate():
    agent = _FakeRanker(passed=True)
    result = optimize_sequence(
        SEQUENCE, model=_FakeModel(), rounds=4, candidates_per_round=10, seed=3,
        variant_agent=agent,
    )
    assert result.objective == "variant"
    assert result.ranked_on_variant_agent is True
    assert agent.calls > 1, "the agent must score every batch, not only the original"
    # The stub agent prefers low GC3 while the stub tier model prefers high, so
    # a search that followed the tier model would move GC3 the other way.
    assert result.best.features["gc3_content"] <= result.original.features["gc3_content"]
    assert result.objective_gain >= 0.0


def test_an_ungated_agent_c_leaves_the_search_on_the_tier_model():
    agent = _FakeRanker(passed=False)
    result = optimize_sequence(
        SEQUENCE, model=_FakeModel(), rounds=4, candidates_per_round=10, seed=3,
        variant_agent=agent,
    )
    assert result.objective == "expression"
    assert agent.calls == 0, "an agent that has not passed its gate must not score anything"
    assert result.best.expected_tier >= result.original.expected_tier


def test_no_agent_at_all_is_todays_behaviour():
    with_none = optimize_sequence(
        SEQUENCE, model=_FakeModel(), rounds=4, candidates_per_round=10, seed=11, variant_agent=None,
    )
    without = optimize_sequence(
        SEQUENCE, model=_FakeModel(), rounds=4, candidates_per_round=10, seed=11,
    )
    assert with_none.best.sequence == without.best.sequence
    assert with_none.objective == "expression"


def test_corroboration_says_gc3_is_not_independent_of_agent_c():
    result = optimize_sequence(
        SEQUENCE, model=_FakeModel(), rounds=3, candidates_per_round=8, seed=2,
        variant_agent=_FakeRanker(passed=True),
    )
    summary = result.corroboration().summary
    if result.improved and result.corroboration().gc3_change:
        assert "its own inputs" in summary
