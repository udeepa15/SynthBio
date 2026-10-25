import pytest
import pandas as pd
import numpy as np
import xgboost
from src.xai import render_shap_waterfall
import shap

def test_render_shap_waterfall_calculation():
    # Synthetic DataFrame with 5 features
    X = pd.DataFrame({
        "GC_percent": [50.0, 40.0],
        "CAI": [0.8, 0.6],
        "folding_energy": [-10.0, -18.0],
        "gravy_score": [0.5, -0.2],
        "pI": [7.0, 5.5]
    })
    y = np.array([1, 0])
    
    # Train XGBoost regressor/classifier
    model = xgboost.XGBClassifier(n_estimators=10, random_state=42).fit(X, y)
    
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X.iloc[[0]])
    
    assert explanation is not None
    assert len(explanation) == 1
    assert explanation[0].values is not None

def test_xai_module_import():
    import src.xai
    assert hasattr(src.xai, "render_shap_waterfall")


def test_render_shap_waterfall_dimension_resilience():
    # 6-class model trained on 11 features (6 classes * 11 features = 66)
    feature_cols = [
        "GC_percent", "GC3", "CAI", "rare_codon_fraction", "rare_codons_head30",
        "head_folding_48", "genewide_folding_avg", "gravy_score", "pI",
        "length_codons", "folding_energy",
    ]
    X_train = pd.DataFrame(np.random.randn(24, len(feature_cols)), columns=feature_cols)
    y_train = np.array([0, 1, 2, 3, 4, 5] * 4)
    model = xgboost.XGBClassifier(n_estimators=5, random_state=42).fit(X_train, y_train)

    # Input dataframe has only 5 features (like the previous app bug)
    partial_df = pd.DataFrame([{
        "GC_percent": 50.0,
        "CAI": 0.7,
        "folding_energy": -12.0,
        "gravy_score": 0.1,
        "pI": 7.2,
    }])

    # Should not raise any std::accumulate / shape check mismatch exception
    render_shap_waterfall(model, partial_df, predicted_class=2)

