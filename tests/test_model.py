import pytest
from src.predict import predict_construct, calculate_failure_risk

def test_predict_construct():
    seq = "ATGAAACGCATTAGCCTGACCCTGCTGAGCGCGGCGATTCTGCTGACCGCGCTGAGCGCG"
    res = predict_construct(seq)
    
    assert "risk_score" in res
    assert "predicted_tier" in res
    assert "feature_importances" in res
    assert 0.0 <= res["risk_score"] <= 100.0
    assert res["predicted_tier"] in ["High", "Medium", "Low"]

def test_risk_scoring():
    optimal_features = {"gc_content": 50.0, "codon_adaptation_index": 0.85, "mrna_folding_stability": -5.0}
    risk = calculate_failure_risk(optimal_features)
    assert risk < 30.0
