"""
Known-answer checks against constructs whose behaviour in E. coli is documented.

A model can score respectably on held-out rows and still be unusable, so these
tests hold it to cases where the answer is not in dispute. They are the reason
the GFP regression below is visible at all: it was found by asking the running
application for a protein everyone expresses, not by reading a metric.
"""

import os
import pandas as pd
import pytest

from src.features import extract_features
from src.predict import check_feature_ranges, predict_construct
from src.model import load_model, load_model_metadata

MODEL_PATH = os.path.join("models", "baseline_model.joblib")

# eGFP coding sequence. Expresses strongly and solubly in E. coli; it is the
# standard positive control for an expression construct.
GFP = (
    "ATGAGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCCCAATTCTTGTTGAATTAGATGGTGATGTTAATGGGCACAAATTTTCTGTCAGTGGAGAGGGT"
    "GAAGGTGATGCAACATACGGAAAACTTACCCTTAAATTTATTTGCACTACTGGAAAACTACCTGTTCCATGGCCAACACTTGTCACTACTTTCTCTTAT"
    "GGTGTTCAATGCTTTTCAAGATACCCAGATCATATGAAACAGCATGACTTTTTCAAGAGTGCCATGCCCGAAGGTTATGTACAGGAAAGAACTATATTT"
    "TTCAAAGATGACGGGAACTACAAGACACGTGCTGAAGTCAAGTTTGAAGGTGATACCCTTGTTAATAGAATCGAGTTAAAAGGTATTGATTTTAAAGAA"
    "GATGGAAACATTCTTGGACACAAATTGGAATACAACTATAACTCACACAATGTATACATCATGGCAGACAAACAAAAGAATGGAATCAAAGTTAACTTC"
    "AAAATTAGACACAACATTGAAGATGGAAGCGTTCAACTAGCAGACCATTATCAACAAAATACTCCAATTGGCGATGGCCCTGTCCTTTTACCAGACAAC"
    "CATTACCTGTCCACACAATCTGCCCTTTCGAAAGATCCCAACGAAAAGAGAGACCACATGGTCCTTCTTGAGTTTGTAACAGCTGCTGGGATTACACAT"
    "GGCATGGATGAACTATACAAATAA"
)

requires_model = pytest.mark.skipif(
    not os.path.exists(MODEL_PATH),
    reason="No trained model artifact; nothing to hold to a known answer.",
)


@pytest.fixture(scope="module")
def artifacts():
    model = load_model(MODEL_PATH)
    metadata = load_model_metadata(MODEL_PATH) or {}
    return model, metadata.get("feature_ranges")


def test_gfp_features_are_plausible():
    """GFP is a real coding sequence, so every feature should compute."""
    features = extract_features(GFP)
    assert 30.0 < features["gc_content"] < 70.0
    assert 0.0 < features["codon_adaptation_index"] <= 1.0
    assert features["mrna_folding_stability"] < 0.0
    assert not pd.isna(features["gravy_score"])


def test_out_of_range_features_are_detected():
    """A value beyond the training range must be reported, not silently scored."""
    ranges = {"CAI": {"min": 0.6, "max": 0.9}, "GC_percent": {"min": 25.0, "max": 78.0}}
    inside = pd.DataFrame([{"CAI": 0.75, "GC_percent": 50.0}])
    below = pd.DataFrame([{"CAI": 0.45, "GC_percent": 50.0}])

    assert check_feature_ranges(inside, ranges) == []
    flagged = check_feature_ranges(below, ranges)
    assert len(flagged) == 1
    assert flagged[0]["feature"] == "CAI"
    assert flagged[0]["side"] == "below"


def test_missing_ranges_are_not_an_error():
    """An artifact without a metadata sidecar still predicts, just unguarded."""
    assert check_feature_ranges(pd.DataFrame([{"CAI": 0.5}]), None) == []


@requires_model
def test_artifact_features_are_computable_from_a_sequence(artifacts):
    """
    Every feature the artifact requires must be derivable from a pasted sequence.

    The Boel folding columns are RNAstructure energies over the vector 5' UTR,
    which this pipeline cannot reproduce. A model fitted on them loads happily and
    then never runs, leaving the fixed rules to answer while the interface still
    names a trained model. This test is what makes that visible.
    """
    model, ranges = artifacts
    result = predict_construct(GFP, model=model, feature_ranges=ranges)
    assert not result["unservable_features"], (
        "The saved artifact requires features that cannot be computed from a sequence: "
        f"{result['unservable_features']}. The application silently falls back to its "
        "fixed rules, so the tier it shows carries nothing from the training data. "
        "See doc/accuracy_roadmap.md step 1."
    )
    assert result["prediction_source"] == "trained_model"


@requires_model
def test_gfp_is_not_predicted_as_a_non_expresser(artifacts):
    """
    GFP expresses well, so a usable model must not place it in the bottom tiers.

    The assertion on the prediction source is the point: without it this test
    passes whenever the fallback rules answer instead of the model, which is a
    failure dressed as a success.
    """
    model, ranges = artifacts
    result = predict_construct(GFP, model=model, feature_ranges=ranges)
    assert result["prediction_source"] == "trained_model", (
        "The tier came from the fixed rules, not the model, so this case proves nothing."
    )
    assert result["predicted_score"] is not None
    assert result["predicted_score"] >= 3, (
        f"GFP was predicted as tier {result['predicted_score']} "
        f"({result['predicted_tier']}), which contradicts its documented behaviour."
    )
