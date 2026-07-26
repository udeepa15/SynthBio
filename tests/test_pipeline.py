import os
import pytest
from src.pipeline import run_pipeline

def test_pipeline_execution(tmp_path):
    model_artifact = os.path.join(tmp_path, "test_model.joblib")
    model, results_df, eval_report = run_pipeline(
        csv_path="data/processed/feature_table.csv",
        model_output_path=model_artifact,
        use_dummy=True
    )
    
    assert model is not None
    assert not results_df.empty
    assert "gene_id" in results_df.columns
    assert "predicted_score" in results_df.columns
    assert os.path.exists(model_artifact)
