"""
End-to-End SynthBio Pipeline Execution Module:
1. Load CSV dataset (data/processed/feature_table.csv or data/processed/feature_table_boel_5features.csv)
2. Hold out a test set, then cross-validate on the remainder with whole organisms held out of each fold
3. Fit the tuned XGBoost classifier (or a dummy baseline) on the training set
4. Score once on the held-out test set, against the trivial baselines it must beat
5. Save the model artifact to models/baseline_model.joblib with a metadata sidecar

The scores this prints are the ones to quote. Earlier revisions cross-validated
on a plain random split, which let genes from one organism sit on both sides of
a fold and reported a figure better than the model achieves on unseen organisms.
"""

import hashlib
import os
import logging
import platform
from datetime import datetime
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import sklearn

from src.data_processing import (
    load_and_prep_feature_table,
    split_data,
    organism_groups,
    feature_ranges,
    resolve_dataset_path,
    FEATURE_CANDIDATES,
)
from src.model import (
    train_dummy_classifier,
    train_tuned_classifier,
    grouped_cross_validate,
    baseline_scores,
    evaluate_model,
    save_model,
    save_model_metadata,
    TUNED_XGB_PARAMS,
)
from src.predict import SERVABLE_FEATURES
from src.provenance import TRAINING_DATASET

# Configure logger
logger = logging.getLogger("SynthBio.pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.pipeline] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def generate_fake_data(n_samples: int = 100) -> pd.DataFrame:
    """Generate synthetic fallback data if feature_table.csv is missing."""
    logger.info(f"Generating synthetic fallback dataset with {n_samples} samples...")
    np.random.seed(42)
    return pd.DataFrame({
        "gene_id": [f"Synth_{i:03d}" for i in range(n_samples)],
        "GC_percent": np.random.uniform(35.0, 65.0, size=n_samples),
        "CAI": np.random.uniform(0.65, 0.88, size=n_samples),
        "folding_energy": np.random.uniform(-20.0, -5.0, size=n_samples),
        "expression_score": np.random.choice([0, 1, 2], size=n_samples)
    })


def _file_digest(path: str) -> str:
    """SHA-256 of the training file, so an artifact can be tied to its data."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_pipeline(
    csv_path: str = None,
    model_output_path: str = "models/baseline_model.joblib",
    use_dummy: bool = False,
    n_splits: int = 5,
    feature_cols: list = None,
    servable_only: bool = True
) -> Tuple[Any, pd.DataFrame, Dict[str, Any]]:
    """
    Execute end-to-end ML pipeline with organism-grouped cross-validation.

    By default the model is restricted to features the application can compute
    from a pasted sequence (`servable_only`). The v2 table's strongest feature,
    the Boel head-folding energy, is not among them: it is an RNAstructure
    energy over the vector 5' UTR. A model fitted on it loads in the app and then
    never runs, leaving the fixed rules to answer in its place, so it is excluded
    unless `servable_only` is turned off for a research run.
    """
    logger.info("============================================================")
    logger.info("[SYNTHBIO END-TO-END PIPELINE STARTING]")
    logger.info("============================================================")

    # 1. Load CSV (or fallback)
    csv_path = resolve_dataset_path(csv_path)

    if csv_path and os.path.exists(csv_path):
        logger.info(f"[STAGE 1] Loading dataset from '{csv_path}'...")
        gene_ids, X, y, df = load_and_prep_feature_table(csv_path=csv_path)
        data_digest = _file_digest(csv_path)
    else:
        logger.warning(f"[STAGE 1] CSV '{csv_path}' not found. Generating synthetic fallback dataset...")
        df = generate_fake_data()
        gene_ids = df["gene_id"]
        X = df[["GC_percent", "CAI", "folding_energy"]]
        y = df["expression_score"]
        data_digest = None

    # 2. Train / Test Split. The test set is scored once, in stage 4, and is not
    #    used for cross-validation or for any choice of settings.
    logger.info("[STAGE 2] Performing Train/Test Split (80% Train, 20% Test)...")
    candidates = FEATURE_CANDIDATES if feature_cols is None else feature_cols
    feature_cols = [c for c in candidates if c in df.columns]
    if servable_only:
        excluded = [c for c in feature_cols if c not in SERVABLE_FEATURES]
        feature_cols = [c for c in feature_cols if c in SERVABLE_FEATURES]
        if excluded:
            logger.warning(
                f"  - Excluding {', '.join(excluded)}: the application cannot compute "
                "these from a sequence, so a model fitted on them could not be served. "
                "Pass servable_only=False for a research run that uses them."
            )
    if not feature_cols:
        raise ValueError(
            "No usable feature columns. The dataset's features cannot be computed from a "
            f"sequence; servable features are {SERVABLE_FEATURES}."
        )
    logger.info(f"  - Features: {', '.join(feature_cols)}")
    target_col = "expression_score" if "expression_score" in df.columns else "expression_yield"
    X_train, X_test, y_train, y_test = split_data(
        df,
        feature_cols=feature_cols,
        target_col=target_col,
        test_size=0.2,
        random_state=42
    )
    test_gene_ids = df.loc[X_test.index, "gene_id"] if "gene_id" in df.columns else X_test.index
    train_groups = organism_groups(df.loc[X_train.index, "gene_id"]) if "gene_id" in df.columns else None
    logger.info(f"  - Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
    logger.info(f"  - Testing samples:  {X_test.shape[0]}")

    # 3. Organism-grouped cross-validation on the training portion only.
    logger.info(f"[STAGE 3] Running {n_splits}-fold organism-grouped cross-validation on training data...")
    temp_model = train_dummy_classifier(X_train, y_train) if use_dummy else train_tuned_classifier(X_train, y_train)
    if train_groups is None:
        logger.warning("  - No gene_id column, so folds cannot be grouped by organism. Scores may be optimistic.")
        train_groups = pd.Series(range(len(y_train)), index=y_train.index).astype(str)
    cv_metrics = grouped_cross_validate(temp_model, X_train, y_train, groups=train_groups, n_splits=n_splits)
    if "grouped_mean_f1_macro" in cv_metrics:
        logger.info(
            f"  - Organism-held-out macro F1: {cv_metrics['grouped_mean_f1_macro']:.3f} "
            f"(+/-{cv_metrics['grouped_std_f1_macro']:.3f})"
        )

    # 4. Final Model Training & Test Set Evaluation
    logger.info(f"[STAGE 4] Fitting Final {'Dummy Baseline' if use_dummy else 'Tuned Gradient Boosted'} Model...")
    if use_dummy:
        model = train_dummy_classifier(X_train, y_train, strategy="most_frequent")
    else:
        model = train_tuned_classifier(X_train, y_train)
    logger.info("  - Model fitting complete.")

    predictions = model.predict(X_test)
    eval_report = evaluate_model(model, X_test, y_test)
    baselines = baseline_scores(y_train, y_test)
    eval_report["baselines"] = baselines
    eval_report["cross_validation"] = cv_metrics

    logger.info("  - Held-out test results, against the baselines they must beat:")
    logger.info(
        f"      model              accuracy {eval_report.get('accuracy', 0.0)*100:5.2f}%   "
        f"macro F1 {eval_report['f1_macro']:.3f}   balanced accuracy {eval_report['balanced_accuracy']*100:5.2f}%"
    )
    logger.info(
        f"      always most common accuracy {baselines['majority_accuracy']*100:5.2f}%   "
        f"macro F1 {baselines['majority_f1_macro']:.3f}"
    )
    logger.info(
        f"      random by frequency accuracy {baselines['random_accuracy']*100:5.2f}%   "
        f"macro F1 {baselines['random_f1_macro']:.3f}"
    )
    if eval_report["f1_macro"] <= max(baselines["majority_f1_macro"], baselines["random_f1_macro"]):
        logger.warning(
            "  - The model does not beat a trivial baseline on macro F1. Treat its predictions as "
            "uninformative and see doc/accuracy_roadmap.md."
        )

    results_df = pd.DataFrame({
        "gene_id": test_gene_ids.values if hasattr(test_gene_ids, "values") else test_gene_ids,
        "actual_score": y_test.values,
        "predicted_score": predictions
    })

    logger.info("\n--- Sample Test Predictions Output ---")
    logger.info("\n" + results_df.head(10).to_string(index=False))

    # 5. Save Model Artifact and its provenance
    logger.info(f"[STAGE 5] Exporting trained model artifact to '{model_output_path}'...")
    save_model(model, model_output_path)
    save_model_metadata(model_output_path, {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_class": type(model).__name__,
        "model_params": None if use_dummy else TUNED_XGB_PARAMS,
        "training_data": {
            "path": csv_path if data_digest else "synthetic fallback",
            "sha256": data_digest,
            "dataset": TRAINING_DATASET if data_digest else None,
            "n_rows": int(len(df)),
            "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        },
        "features": list(X_train.columns),
        "servable_only": servable_only,
        "feature_ranges": feature_ranges(X_train),
        "metrics": {
            "test_accuracy": float(eval_report.get("accuracy", 0.0)),
            "test_f1_macro": eval_report["f1_macro"],
            "test_balanced_accuracy": eval_report["balanced_accuracy"],
            "baselines": baselines,
            "cross_validation": cv_metrics,
        },
        "versions": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    })
    logger.info("============================================================")
    logger.info("[SYNTHBIO PIPELINE EXECUTION SUCCESSFUL]")
    logger.info("============================================================")

    return model, results_df, eval_report


if __name__ == "__main__":
    run_pipeline()
