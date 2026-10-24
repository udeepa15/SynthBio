"""
Tests for the attribution labelling logic.

The model is a six-class tier classifier, so a positive SHAP value is evidence
for the predicted tier rather than evidence of failure. These tests pin the
corrected reading: the verdict attached to an attribution comes from the tier,
and the biological note comes from the feature's own value.
"""

import pytest

from src.xai import (
    tier_is_unfavourable,
    describe_feature_value,
    FEATURE_BIO_MAP,
    LOW_TIER_CEILING,
)


@pytest.mark.parametrize("tier", [0, 1, 2])
def test_low_tiers_are_unfavourable(tier):
    assert tier_is_unfavourable(tier) is True


@pytest.mark.parametrize("tier", [3, 4, 5])
def test_high_tiers_are_not_unfavourable(tier):
    assert tier_is_unfavourable(tier) is False


def test_unknown_tier_gives_no_verdict():
    """Without a tier there is no basis to call an attribution good or bad."""
    assert tier_is_unfavourable(None) is None


def test_ceiling_is_the_boundary():
    assert tier_is_unfavourable(LOW_TIER_CEILING) is True
    assert tier_is_unfavourable(LOW_TIER_CEILING + 1) is False


def test_high_cai_gets_the_high_value_note():
    note = describe_feature_value("CAI", 0.9)
    assert note == FEATURE_BIO_MAP["CAI"]["high_desc"]


def test_low_cai_gets_the_low_value_note():
    note = describe_feature_value("CAI", 0.3)
    assert note == FEATURE_BIO_MAP["CAI"]["low_desc"]


def test_folding_proxy_reads_more_negative_as_tighter():
    """folding_energy runs negative, so below the midpoint is the 'high' case."""
    tight = describe_feature_value("folding_energy", -30.0)
    relaxed = describe_feature_value("folding_energy", -5.0)
    assert tight == FEATURE_BIO_MAP["folding_energy"]["high_desc"]
    assert relaxed == FEATURE_BIO_MAP["folding_energy"]["low_desc"]
    assert tight != relaxed


def test_note_is_independent_of_attribution_sign():
    """
    The regression this guards: a favourable CAI value must keep its favourable
    note even when CAI is the strongest evidence for a low tier.
    """
    assert describe_feature_value("CAI", 0.9) == FEATURE_BIO_MAP["CAI"]["high_desc"]


def test_missing_or_unknown_values_produce_no_note():
    assert describe_feature_value("CAI", None) == ""
    assert describe_feature_value("CAI", float("nan")) == ""
    assert describe_feature_value("not_a_feature", 1.0) == ""


def test_folding_label_does_not_claim_to_be_thermodynamic():
    name = FEATURE_BIO_MAP["folding_energy"]["name"]
    assert "proxy" in name.lower()
    assert "not thermodynamic" in name.lower()
