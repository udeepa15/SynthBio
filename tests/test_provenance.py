"""Tests for provenance reporting shown under the verdict."""

from src import provenance


class FakeModel:
    """Stands in for a fitted estimator carrying feature names."""

    def __init__(self, features):
        self.feature_names_in_ = features

    def predict(self, X):
        return [0] * len(X)


def test_heuristic_mode_when_no_model():
    prov = provenance.describe(None)
    assert not prov.is_trained
    assert prov.mode_label == "Heuristic fallback"
    assert "Heuristic fallback" in prov.as_line()
    assert "not applicable" in prov.as_line()


def test_heuristic_mode_states_that_figures_are_not_model_derived():
    prov = provenance.describe(None)
    caveats = " ".join(prov.caveats())
    assert "No trained model artifact was found" in caveats
    assert "SHAP" in caveats


def test_trained_mode_reports_model_class():
    prov = provenance.describe(FakeModel(["GC_percent", "CAI", "folding_energy"]))
    assert prov.is_trained
    assert prov.model_class == "FakeModel"
    assert "FakeModel" in prov.as_line()
    assert provenance.TRAINING_DATASET in prov.as_line()


def test_partial_feature_set_is_disclosed():
    """A model fitted on fewer than the computable features must say so."""
    prov = provenance.describe(FakeModel(["GC_percent", "CAI"]))
    caveats = " ".join(prov.caveats())
    assert "2 of the" in caveats
    assert "gravy_score" in caveats
    assert "pI" in caveats


def test_full_feature_set_leaves_only_the_scope_caveat():
    """
    With every computable feature fitted, no feature is reported as missing.

    The scope caveat remains: the model separates genes, not synonymous variants
    of one gene, and that holds however many features it uses.
    """
    prov = provenance.describe(FakeModel(list(provenance.SERVABLE_FEATURE_NAMES)))
    caveats = prov.caveats()
    assert len(caveats) == 1
    assert "synonymous codon variants" in caveats[0]
    assert "computed and shown" not in caveats[0]


def test_line_always_carries_the_analysis_time():
    for model in (None, FakeModel(["GC_percent"])):
        assert "Analysed:" in provenance.describe(model).as_line()
