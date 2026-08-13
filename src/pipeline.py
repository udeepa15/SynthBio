"""
End-to-End SynthBio Pipeline Skeleton:
1. Load CSV dataset (data/processed/feature_table.csv)
2. Run Stratified K-Fold Cross-Validation for model validation
3. Fit XGBoost/Dummy baseline model on training set
4. Output predictions and evaluation metrics
5. Save model artifact to models/
"""

import os
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.data_processing import load_and_prep_feature_table, split_data
from src.model import train_dummy_classifier, train_baseline_classifier, cross_validate_model, evaluate_model, save_model

def generate_fake_data(n_samples: int = 100) -> pd.DataFrame:
    """Generate synthetic fallback data if feature_table.csv is missing."""
    np.random.seed(42)
    return pd.DataFrame({
        "gene_id": [f"Synth_{i:03d}" for i in range(n_samples)],
        "GC_percent": np.random.uniform(35.0, 65.0, size=n_samples),
        "CAI": np.random.uniform(0.65, 0.88, size=n_samples),
        "folding_energy": np.random.uniform(-20.0, -5.0, size=n_samples),
        "expression_score": np.random.choice([0, 1, 2], size=n_samples)
    })

def run_pipeline(
    csv_path: str = "data/processed/feature_table.csv",
    model_output_path: str = "models/baseline_model.joblib",
    use_dummy: bool = False,
    n_splits: int = 5
) -> Tuple[Any, pd.DataFrame, Dict[str, Any]]:
    """Execute end-to-end ML pipeline with Stratified K-Fold Cross Validation."""
    print("=" * 60)
    print("[SYNTHBIO END-TO-END PIPELINE]")
    print("=" * 60)

    # 1. Load CSV (or generate fake data if CSV doesn't exist)
    if os.path.exists(csv_path):
        print(f"\n[STAGE 1] Loading dataset from '{csv_path}'...")
        gene_ids, X, y, df = load_and_prep_feature_table(csv_path=csv_path)
    else:
        print(f"\n[STAGE 1] CSV '{csv_path}' not found. Generating synthetic fallback dataset...")
        df = generate_fake_data()
        gene_ids = df["gene_id"]
        X = df[["GC_percent", "CAI", "folding_energy"]]
        y = df["expression_score"]

    # 2. Train / Test Split
    print("\n[STAGE 2] Performing Train/Test Split (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = split_data(
        df,
        feature_cols=["GC_percent", "CAI", "folding_energy"],
        target_col="expression_score",
        test_size=0.2,
        random_state=42
    )
    test_gene_ids = df.loc[X_test.index, "gene_id"]
    print(f"  - Training samples: {X_train.shape[0]}")
    print(f"  - Testing samples:  {X_test.shape[0]}")

    # 3. K-Fold Cross-Validation
    print(f"\n[STAGE 3] Running {n_splits}-Fold Cross-Validation on Training Data...")
    temp_model = train_dummy_classifier(X_train, y_train) if use_dummy else train_baseline_classifier(X_train, y_train)
    cv_metrics = cross_validate_model(temp_model, X_train, y_train, n_splits=min(n_splits, len(y_train.unique())))
    print(f"  - Mean CV Accuracy: {cv_metrics['mean_accuracy'] * 100:.2f}% (± {cv_metrics['std_accuracy'] * 100:.2f}%)")
    print(f"  - Mean CV Macro F1: {cv_metrics['mean_f1_macro']:.3f}")

    # 4. Final Model Training & Test Set Evaluation
    print(f"\n[STAGE 4] Fitting Final {'Dummy Baseline' if use_dummy else 'XGBoost'} Model...")
    if use_dummy:
        model = train_dummy_classifier(X_train, y_train, strategy="most_frequent")
    else:
        model = train_baseline_classifier(X_train, y_train)
    print("  - Model training complete.")

    predictions = model.predict(X_test)
    eval_report = evaluate_model(model, X_test, y_test)
    accuracy = eval_report.get("accuracy", 0.0)
    print(f"  - Final Test Accuracy: {accuracy * 100:.2f}%")

    results_df = pd.DataFrame({
        "gene_id": test_gene_ids.values,
        "GC_percent": X_test["GC_percent"].values,
        "CAI": X_test["CAI"].values,
        "folding_energy": X_test["folding_energy"].values,
        "actual_score": y_test.values,
        "predicted_score": predictions
    })

    print("\n--- Sample Prediction Output (First 10 Test Constructs) ---")
    print(results_df.head(10).to_string(index=False))

    # 5. Save Model Artifact
    print(f"\n[STAGE 5] Exporting trained model artifact to '{model_output_path}'...")
    save_model(model, model_output_path)
    print("  - Model artifact successfully saved.")
    print("=" * 60)
    print("[SUCCESS] PIPELINE EXECUTION SUCCESSFUL")
    print("=" * 60)

    return model, results_df, eval_report


if __name__ == "__main__":
    run_pipeline()

