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

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = Any
    XGBRegressor = Any

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
) -> XGBClassifier:
    """Train baseline XGBoost classifier on 3 core features."""
    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.05,
            "random_state": 42
        }
        
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model

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

