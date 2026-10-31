"""
Model training and evaluation module for gradient boosting baseline models (XGBoost / LightGBM).
"""

from typing import Dict, Any, Tuple, Optional
import json
import os
import logging
import warnings
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score, accuracy_score
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_validate, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Configure logger
logger = logging.getLogger("SynthBio.model")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] [SynthBio.model] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

try:
    from xgboost import XGBClassifier, XGBRegressor
    logger.info("XGBoost library detected and imported successfully.")
except ImportError:
    XGBClassifier = RandomForestClassifier
    XGBRegressor = None
    logger.warning("XGBoost library not found. Falling back to RandomForestClassifier.")

try:
    import shap
    logger.info("SHAP library detected and imported successfully.")
except ImportError:
    shap = None
    logger.warning("SHAP library not found. Explainable AI (XAI) waterfall rendering will be disabled.")

FEATURE_NAMES = ["GC_percent", "CAI", "folding_energy"]

# Settings selected by a randomised search of 40 candidates per algorithm,
# scored by macro F1 under organism-grouped 5-fold cross-validation. XGBoost
# (organism-held-out macro F1 0.199) beat LightGBM (0.193). Retuning these is
# only worth doing once the feature set changes; see doc/accuracy_roadmap.md.
TUNED_XGB_PARAMS = {
    "objective": "multi:softprob",
    "n_estimators": 800,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
    "reg_lambda": 10,
    "random_state": 42,
    "tree_method": "hist",
}


def balanced_sample_weight(y: pd.Series) -> np.ndarray:
    """
    Per-sample weights that equalise the influence of each tier.

    Tier counts in the Boel set are heavily uneven (tier 1 has 131 rows against
    tier 0's 1754), so without weighting the model learns to ignore rare tiers.
    """
    counts = y.value_counts()
    weights = len(y) / (len(counts) * counts)
    return weights.reindex(y).to_numpy()


def train_dummy_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str = "most_frequent"
) -> DummyClassifier:
    """Train a simple dummy model baseline for end-to-end pipeline validation."""
    logger.info(f"Training DummyClassifier with strategy='{strategy}' on {len(X_train)} samples...")
    model = DummyClassifier(strategy=strategy, random_state=42)
    model.fit(X_train, y_train)
    logger.info("DummyClassifier training complete.")
    return model


def train_baseline_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any] = None
) -> Any:
    """Train baseline gradient boosted / random forest classifier on core features."""
    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.05,
            "random_state": 42
        }
        
    model_name = "RandomForestClassifier" if XGBClassifier == RandomForestClassifier else "XGBClassifier"
    logger.info(f"Fitting {model_name} on {X_train.shape[0]} samples with features {list(X_train.columns)}...")
    
    if XGBClassifier == RandomForestClassifier:
        clf_params = {k: v for k, v in params.items() if k != "learning_rate"}
    else:
        clf_params = params
        
    model = XGBClassifier(**clf_params)
    model.fit(X_train, y_train)
    logger.info(f"{model_name} model fit successfully.")
    return model


def train_tuned_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any] = None
) -> Any:
    """
    Fit the tuned gradient boosted classifier that the application loads.

    Falls back to `train_baseline_classifier` when XGBoost is unavailable, since
    the tuned settings are XGBoost-specific.
    """
    if XGBClassifier == RandomForestClassifier:
        logger.warning("XGBoost unavailable; tuned settings do not apply. Using the baseline classifier.")
        return train_baseline_classifier(X_train, y_train)

    params = dict(TUNED_XGB_PARAMS if params is None else params)
    logger.info(f"Fitting tuned XGBClassifier on {X_train.shape[0]} samples with features {list(X_train.columns)}...")
    model = XGBClassifier(**params)
    model.fit(X_train, y_train, sample_weight=balanced_sample_weight(y_train))
    logger.info("Tuned model fit successfully.")
    return model


def baseline_scores(y_train: pd.Series, y_test: pd.Series) -> Dict[str, float]:
    """
    Scores of the trivial predictors any real model has to beat.

    Accuracy alone is misleading on this target: always answering the most
    frequent tier scores higher than a model that genuinely separates tiers but
    spreads its errors, which is why macro F1 is reported alongside it.
    """
    majority = [y_train.mode()[0]] * len(y_test)
    rng = np.random.default_rng(42)
    shares = (y_train.value_counts(normalize=True)).sort_index()
    random_guess = rng.choice(shares.index.to_numpy(), size=len(y_test), p=shares.to_numpy())
    return {
        "majority_accuracy": float(accuracy_score(y_test, majority)),
        "majority_f1_macro": float(f1_score(y_test, majority, average="macro", zero_division=0)),
        "random_accuracy": float(accuracy_score(y_test, random_guess)),
        "random_f1_macro": float(f1_score(y_test, random_guess, average="macro", zero_division=0)),
    }


def grouped_cross_validate(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Cross-validate with whole organisms held out of each fold.

    A plain random split lets genes from one organism sit on both sides of the
    fold, which inflates the score without improving real predictions. This is
    the figure to quote.
    """
    usable = min(n_splits, groups.nunique(), int(y.value_counts().min()))
    if usable < 2:
        logger.warning("Too few groups or rare-class rows for grouped CV; falling back to stratified CV.")
        return cross_validate_model(model, X, y, n_splits=max(2, min(n_splits, int(y.value_counts().min()))))

    logger.info(f"Running {usable}-fold organism-grouped cross-validation over {groups.nunique()} groups...")
    cv = StratifiedGroupKFold(n_splits=usable, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv, groups=groups, scoring="f1_macro")
    metrics = {
        "grouped_mean_f1_macro": float(np.mean(scores)),
        "grouped_std_f1_macro": float(np.std(scores)),
        "n_groups": int(groups.nunique()),
        "n_splits": int(usable),
    }
    logger.info(
        f"Organism-held-out macro F1: {metrics['grouped_mean_f1_macro']:.3f} "
        f"(+/-{metrics['grouped_std_f1_macro']:.3f}) over {usable} folds"
    )
    return metrics


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, float]:
    """Perform Stratified K-Fold Cross-Validation for small dataset reliability."""
    logger.info(f"Running {n_splits}-Fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
    
    metrics = {
        "mean_accuracy": float(np.mean(cv_results['test_accuracy'])),
        "std_accuracy": float(np.std(cv_results['test_accuracy'])),
        "mean_f1_macro": float(np.mean(cv_results['test_f1_macro'])),
        "std_f1_macro": float(np.std(cv_results['test_f1_macro'])),
        "mean_precision": float(np.mean(cv_results['test_precision_macro'])),
        "mean_recall": float(np.mean(cv_results['test_recall_macro'])),
    }
    logger.info(
        f"CV Results -> Accuracy: {metrics['mean_accuracy']*100:.2f}% (±{metrics['std_accuracy']*100:.2f}%), "
        f"Macro F1: {metrics['mean_f1_macro']:.3f}"
    )
    return metrics


def get_shap_explainer(model: Any) -> Any:
    """Initialize SHAP TreeExplainer for feature importance interpretation."""
    if shap is None:
        logger.warning("Cannot initialize SHAP TreeExplainer: `shap` package is not available.")
        return None
    try:
        logger.info("Initializing SHAP TreeExplainer for model...")
        explainer = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer initialized successfully.")
        return explainer
    except Exception as err:
        logger.error(f"Failed to initialize SHAP TreeExplainer: {str(err)}", exc_info=False)
        return None


def explain_prediction_shap(
    explainer: Any,
    X_sample: pd.DataFrame,
    predicted_class: Optional[int] = None,
) -> Dict[str, float]:
    """
    Local SHAP attribution for one construct, keyed by feature name.

    With `predicted_class` given, the signed values for that tier are returned:
    positive means the feature pushed the model towards the tier it predicted,
    negative means it pushed away. That is the same convention the waterfall in
    `src/xai.py` uses, so the two cannot tell different stories about one
    construct.

    Without it, the mean absolute value across tiers is returned, which says
    how much a feature mattered but not which way. Mixing the two was a real
    defect in this project: one panel showed signed values for the predicted
    tier while another showed magnitudes across all six, under labels that
    implied they were the same thing.
    """
    if explainer is None or shap is None:
        return {}
    try:
        logger.info("Extracting SHAP feature attribution for the sample feature vector...")
        values = explainer(X_sample).values
        if isinstance(values, list):
            # Older SHAP returns one array per class.
            values = np.stack(values, axis=-1)
        values = np.asarray(values)

        if values.ndim == 3:  # (samples, features, classes)
            if predicted_class is not None and 0 <= int(predicted_class) < values.shape[2]:
                per_feature = values[0, :, int(predicted_class)]
            else:
                per_feature = np.abs(values).mean(axis=(0, 2))
        elif values.ndim == 2:  # (samples, features)
            per_feature = values[0]
        else:
            per_feature = values.flatten()

        names = X_sample.columns.tolist()
        result = {
            names[i]: float(per_feature[i]) for i in range(min(len(names), len(per_feature)))
        }
        logger.info(f"SHAP attribution computed for {len(result)} features.")
        return result
    except Exception as err:
        logger.error(f"SHAP explanation computation failed: {str(err)}", exc_info=False)
        return {}


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Evaluate model performance on test set."""
    logger.info(f"Evaluating model on {len(X_test)} test samples...")
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    acc = report.get("accuracy", 0.0)
    report["f1_macro"] = float(f1_score(y_test, preds, average="macro", zero_division=0))
    report["balanced_accuracy"] = float(balanced_accuracy_score(y_test, preds))
    logger.info(
        f"Test Set Evaluation Complete -> Accuracy: {acc*100:.2f}%, "
        f"Macro F1: {report['f1_macro']:.3f}, Balanced Accuracy: {report['balanced_accuracy']*100:.2f}%"
    )
    return report


def metadata_path(model_filepath: str) -> str:
    """Path of the sidecar file recording how a model artifact was produced."""
    return os.path.splitext(model_filepath)[0] + "_metadata.json"


def save_model_metadata(model_filepath: str, metadata: Dict[str, Any]) -> str:
    """
    Write the artifact's provenance next to it.

    Records the training data and its hash, the feature list and its observed
    ranges, library versions and the measured scores, so a deployed model can be
    traced back to the run that produced it.
    """
    path = metadata_path(model_filepath)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    logger.info(f"Model metadata written to '{path}'.")
    return path


def load_model_metadata(model_filepath: str) -> Optional[Dict[str, Any]]:
    """Read an artifact's sidecar metadata, or None when it has none."""
    path = metadata_path(model_filepath)
    if not os.path.exists(path):
        logger.warning(f"No metadata sidecar found at '{path}'.")
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as err:
        logger.error(f"Could not read model metadata from '{path}': {err}")
        return None


def save_model(model: Any, filepath: str) -> None:
    """Save trained model artifact to disk."""
    logger.info(f"Saving trained model artifact to '{filepath}'...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    logger.info("Model artifact saved successfully.")


def load_model(filepath: str) -> Any:
    """Load model artifact from disk with version warning handling."""
    logger.info(f"Loading trained model artifact from '{filepath}'...")
    if not os.path.exists(filepath):
        logger.error(f"Model artifact file not found at path: {filepath}")
        raise FileNotFoundError(f"Model file '{filepath}' does not exist.")
    
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = joblib.load(filepath)
            for warning in w:
                logger.warning(f"Unpickling Warning: {warning.message}")
        logger.info(f"Successfully loaded model artifact: {type(model).__name__}")
        return model
    except Exception as err:
        logger.error(f"Failed to load model artifact from '{filepath}': {str(err)}")
        raise err
