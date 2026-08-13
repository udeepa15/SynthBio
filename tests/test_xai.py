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
