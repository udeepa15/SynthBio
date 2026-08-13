import pytest
import pandas as pd
import numpy as np
from src.predict import predict_construct, calculate_failure_risk
def test_cross_validation():
    X = pd.DataFrame({
        "GC_percent": np.random.uniform(40, 60, 20),
        "CAI": np.random.uniform(0.6, 0.9, 20),
        "folding_energy": np.random.uniform(-15, -5, 20)
    })
    y = pd.Series(np.random.choice([0, 1, 2], 20))
    model = train_baseline_classifier(X, y)
    cv_metrics = cross_validate_model(model, X, y, n_splits=3)
    
    assert "mean_accuracy" in cv_metrics
    assert "mean_f1_macro" in cv_metrics
    assert 0.0 <= cv_metrics["mean_accuracy"] <= 1.0

def test_shap_explanation():
    X = pd.DataFrame({
        "GC_percent": [50.0, 45.0, 55.0],
        "CAI": [0.8, 0.7, 0.85],
        "folding_energy": [-5.0, -10.0, -3.0]
    })
    y = pd.Series([2, 1, 2])
    model = train_baseline_classifier(X, y)
    explainer = get_shap_explainer(model)
    if explainer is not None:
        shap_dict = explain_prediction_shap(explainer, X.head(1))
        assert isinstance(shap_dict, dict)
