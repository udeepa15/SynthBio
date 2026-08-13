"""
Model training and evaluation module for gradient boosting baseline models (XGBoost / LightGBM).
"""

from typing import Dict, Any, Tuple
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = RandomForestClassifier
    XGBRegressor = None

try:
    import shap
except ImportError:
    shap = None

FEATURE_NAMES = ["GC_percent", "CAI", "folding_energy"]


def train_dummy_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str = "most_frequent"
) -> DummyClassifier:
    """Train a simple dummy model baseline for end-to-end pipeline validation."""
    model = DummyClassifier(strategy=strategy, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_baseline_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any] = None
) -> Any:
    """Train baseline gradient boosted / random forest classifier on 3 core features."""
    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.05,
            "random_state": 42
        }
        
    if XGBClassifier == RandomForestClassifier:
        clf_params = {k: v for k, v in params.items() if k != "learning_rate"}
    else:
        clf_params = params
        
    model = XGBClassifier(**clf_params)
    model.fit(X_train, y_train)
    return model


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, float]:
    """Perform Stratified K-Fold Cross-Validation for small dataset reliability."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
    
    return {
        "mean_accuracy": float(np.mean(cv_results['test_accuracy'])),
        "std_accuracy": float(np.std(cv_results['test_accuracy'])),
        "mean_f1_macro": float(np.mean(cv_results['test_f1_macro'])),
        "std_f1_macro": float(np.std(cv_results['test_f1_macro'])),
        "mean_precision": float(np.mean(cv_results['test_precision_macro'])),
        "mean_recall": float(np.mean(cv_results['test_recall_macro'])),
    }

def get_shap_explainer(model: Any) -> Any:
    """Initialize SHAP TreeExplainer for feature importance interpretation."""
    if shap is None:
        return None
    try:
        return shap.TreeExplainer(model)
    except Exception:
        return None

def explain_prediction_shap(explainer: Any, X_sample: pd.DataFrame) -> Dict[str, float]:
    """Extract local SHAP feature importances for a sample dataframe."""
    if explainer is None or shap is None:
        return {}
    try:
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            # For multi-class, average absolute SHAP values across classes
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
        feature_names = X_sample.columns.tolist()
        return {feature_names[i]: float(mean_abs_shap[i]) for i in range(len(feature_names))}
    except Exception:
        return {}

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """Evaluate model performance on test set."""
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    return report

def save_model(model: Any, filepath: str) -> None:
    """Save trained model artifact to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def load_model(filepath: str) -> Any:
    """Load model artifact from disk."""
    return joblib.load(filepath)


